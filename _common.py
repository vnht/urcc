"""Shared utilities for the UOC (Unlearning Over-Commitment) pipeline.

Loaded by every step script. Keeps the step files focused on their specific
responsibility instead of repeating boilerplate (model loading, tokenisation,
forward passes, JSONL I/O, the macOS Triton mock).
"""

from __future__ import annotations

# ── macOS Triton mock — must precede any torch/transformers import ────────────
# transformers / torch._inductor sometimes import `triton.*` at module load
# time on systems where triton isn't installed (e.g. macOS). Pre-populate
# sys.modules with mocks before any other import touches torch.
import sys
from unittest.mock import MagicMock

if sys.platform == "darwin":
    import importlib.abc
    import importlib.machinery

    _TRITON_MODS = (
        "triton", "triton.language", "triton.runtime", "triton.runtime.jit",
        "triton.backends", "triton.backends.compiler", "triton.backends.cuda",
        "triton.compiler", "triton.compiler.compiler",
    )
    for _m in _TRITON_MODS:
        sys.modules.setdefault(_m, MagicMock())

    class _TritonMockLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mock = sys.modules.get(spec.name) or MagicMock()
            mock.__name__ = spec.name
            mock.__package__ = spec.name.rpartition(".")[0] or spec.name
            mock.__path__ = []
            mock.__spec__ = spec
            mock.__loader__ = self
            sys.modules[spec.name] = mock
            return mock

        def exec_module(self, module):
            pass

    class _TritonMockFinder(importlib.abc.MetaPathFinder):
        _loader = _TritonMockLoader()

        def find_spec(self, fullname, path, target=None):
            if fullname == "triton" or fullname.startswith("triton."):
                return importlib.machinery.ModuleSpec(
                    fullname, self._loader, is_package=True
                )
            return None

    sys.meta_path.insert(0, _TritonMockFinder())


# ── Standard imports ──────────────────────────────────────────────────────────
import json
import logging
import os
import re
import time
from pathlib import Path

import torch

if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

from dotenv import load_dotenv

from config import (
    GPTOSS_ATTN_IMPLEMENTATION,
    GPTOSS_MAX_NEW_TOKENS_ESCALATED,
    GPTOSS_REASONING_EFFORT,
    KUQ_PROMPT_TEMPLATE,
    LAYER_SLICE,
    MODEL_REGISTRY,
    REPO_ROOT,
    SQUAD_PROMPT_TEMPLATE,
    domain_of,
)

load_dotenv(REPO_ROOT / ".env")


# ── Logging (one consistent format across step scripts) ───────────────────────

def setup_logging(name: str | None = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy in ("httpx", "huggingface_hub.file_download", "numexpr",
                  "transformers.tokenization_utils_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # torchao emits WARNINGs when its optional fused kernels (e.g. _C_mxfp8,
    # _C_cutlass_90a) can't load. Those target Hopper (sm_90) and are not used
    # by our bf16 LoRA path — they won't load on Blackwell/other GPUs anyway.
    # PEFT only needs torchao importable, so the failures are harmless noise.
    logging.getLogger("torchao").setLevel(logging.ERROR)
    return logging.getLogger(name or "uoc")


log = setup_logging()


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """Append a single row to a JSONL file and flush. Crash-safe: each row is
    durable as soon as this returns, so subsequent runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ── Progress + timing ─────────────────────────────────────────────────────────

def format_duration(secs: float) -> str:
    """Human-readable duration. ``inf``/``nan`` render as ``—``."""
    if secs is None or secs != secs or secs == float("inf"):
        return "—"
    secs = max(int(secs), 0)
    h, secs = divmod(secs, 3600)
    m, s = divmod(secs, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Stopwatch:
    """Context manager that logs `<label> took <duration>` on exit.

        with Stopwatch("set A forward passes"):
            ...
    """

    def __init__(self, label: str, logger: logging.Logger | None = None) -> None:
        self.label = label
        self.logger = logger or log
        self.t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.t0 = time.time()
        return self

    def __exit__(self, *_exc) -> bool:
        self.elapsed = time.time() - self.t0
        self.logger.info("  %s took %s", self.label, format_duration(self.elapsed))
        return False


class Progress:
    """Periodic progress logger with rate, elapsed, ETA, and per-instance timing.

    Example
    -------
        prog = Progress(total=len(items), desc="step 0 mine", log_every=25)
        for it in items:
            ... do work ...
            prog.tick(extras={"C": commits, "A": abstains})
        prog.done()

    Always logs the first tick, every ``log_every`` ticks, and the final tick.
    Rate and ETA are rolled over the entire run (no smoothing) to keep the
    numbers transparent.
    """

    def __init__(
        self,
        total: int,
        *,
        desc: str = "",
        log_every: int = 25,
        logger: logging.Logger | None = None,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.log_every = max(int(log_every), 1)
        self.logger = logger or log
        self.t0 = time.time()
        self.last_tick_t = self.t0
        self.last_tick_dur = 0.0
        self.n = 0

    def tick(self, n: int = 1, extras: dict | None = None) -> None:
        now = time.time()
        self.last_tick_dur = now - self.last_tick_t
        self.last_tick_t = now
        self.n += int(n)

        is_last = self.total > 0 and self.n >= self.total
        is_first = self.n <= n
        if not (is_first or is_last or self.n % self.log_every == 0):
            return

        elapsed = now - self.t0
        rate = self.n / max(elapsed, 1e-6)
        remaining = max(self.total - self.n, 0)
        eta = remaining / rate if rate > 0 else float("inf")

        extra_str = (
            "  " + "  ".join(f"{k}={v}" for k, v in extras.items())
            if extras else ""
        )
        self.logger.info(
            "  %s [%d/%d] %.2f/s  elapsed=%s  eta=%s%s",
            self.desc, self.n, self.total, rate,
            format_duration(elapsed), format_duration(eta), extra_str,
        )

    def done(self, extras: dict | None = None) -> float:
        elapsed = time.time() - self.t0
        rate = self.n / max(elapsed, 1e-6)
        extra_str = (
            "  " + "  ".join(f"{k}={v}" for k, v in extras.items())
            if extras else ""
        )
        self.logger.info(
            "  %s done. %d items in %s (%.2f/s)%s",
            self.desc, self.n, format_duration(elapsed), rate, extra_str,
        )
        return elapsed


# ── Prompt building (must match mining templates) ─────────────────────────────

def build_unanswerable_prompt(dataset: str, row: dict) -> str:
    """Reconstruct the prompt used at mining time for forget rows."""
    if "generation_prompt" in row and row["generation_prompt"]:
        return row["generation_prompt"]
    domain = domain_of(dataset)
    if domain == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    if domain == "squad":
        return SQUAD_PROMPT_TEMPLATE.format(
            question=row["question"], context=row.get("context", ""),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def build_answerable_prompt(dataset: str, row: dict) -> str:
    """Build the prompt for retain-answerable QA rows."""
    domain = domain_of(dataset)
    if domain == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    if domain == "squad":
        return SQUAD_PROMPT_TEMPLATE.format(
            question=row["question"], context=row.get("context", ""),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


# ── Model + tokenizer loading ─────────────────────────────────────────────────

def load_model_and_tokenizer(model_key: str, eval_only: bool = True):
    """Load model + tokenizer from HF. eval_only=True calls model.eval()."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key={model_key!r}. "
                       f"Choices: {list(MODEL_REGISTRY)}")
    model_id = MODEL_REGISTRY[model_key]
    hf_token = os.environ.get("HF_TOKEN", "")

    log.info("Loading model %s ...", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Qwen BPE tokenizer doesn't use WordPiece-style space cleanup;
        # transformers already ignores the True default and warns. Set it False
        # explicitly to silence the warning (no behavioural change).
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, clean_up_tokenization_spaces=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
    elif model_id.startswith(("meta-llama/", "microsoft/")):
        # Llama 3.1 (base) and Phi-4 (instruct): standard dense CausalLM,
        # standard AutoTokenizer (Phi-4's chat template ships with the
        # tokenizer; its fused qkv/gate_up projections only matter for LoRA
        # targeting, not loading).
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        # No dedicated pad token on these checkpoints; point it at eos so
        # generate() doesn't warn about an unset pad_token_id.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
        # Llama 3.1 generation_config sets max_length=131072; drop it to avoid
        # "both max_new_tokens and max_length set" warnings on every generate().
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
    elif model_id.startswith("mistralai/"):
        from transformers import (
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)

        # The `-BF16` variant ships native bfloat16 weights (no quantisation).
        # The unsuffixed variant ships FP8 weights and needs explicit dequant.
        is_fp8_variant = not model_id.endswith("-BF16")

        kwargs: dict = {
            "token":               hf_token,
            "tie_word_embeddings": False,
        }
        if is_fp8_variant:
            from transformers import FineGrainedFP8Config
            kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)

        if sys.platform == "darwin":
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu", **kwargs,
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
        else:
            kwargs.setdefault("dtype", torch.bfloat16)
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", **kwargs,
            )
        # Ministral's generation_config ships max_length=262144, which collides
        # with our explicit max_new_tokens on every generate() call and floods
        # the log with a "both max_new_tokens and max_length set" warning. We
        # always size generations via max_new_tokens, so drop the stale default.
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
    elif model_id.startswith("openai/"):
        # gpt-oss: dense CausalLM wrapper over a sparse MoE. Standard
        # AutoTokenizer (the harmony chat template ships with the tokenizer).
        # The released checkpoint stores expert weights in MXFP4; dequantise to
        # bf16 so the model trains/runs without Hopper-only MXFP4 kernels
        # (~40GB). Attention uses sinks, so sdpa is unsupported; the backend is
        # configurable (GPTOSS_ATTN_IMPLEMENTATION) — flex_attention by default,
        # far faster than eager while still handling the sink renormalisation.
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        # Attention backend depends on the workload: generation/forward passes
        # (eval, mining, extraction) decode/attend over long contexts where the
        # fast sink-aware backend (flex_attention) is a big win over eager.
        # Training uses short prompt+K sequences and is MoE/checkpoint-bound, not
        # attention-bound, so it stays on eager to avoid torch.compile recompiles
        # on variable-length batches.
        attn_impl = GPTOSS_ATTN_IMPLEMENTATION if eval_only else "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            quantization_config=Mxfp4Config(dequantize=True),
            attn_implementation=attn_impl,
        )
        log.info("  gpt-oss attention backend: %s", attn_impl)
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    if eval_only:
        model.eval()
    log.info("  Loaded in %.1fs", time.time() - t0)
    return model, tokenizer


def free_model(model) -> None:
    """Free GPU/MPS memory after a long extraction."""
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Tokenisation ──────────────────────────────────────────────────────────────

def encode_text(tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if isinstance(ids, list):
        return ids
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return list(ids)


def tokenise_prompt_plus_answer(
    tokenizer,
    model_key: str,
    prompt: str,
    answer: str,
    k_answer_tokens: int | None = None,
) -> tuple[list[int], int, int]:
    """Returns (full_ids, prompt_len, n_answer_tokens).

    The prompt is wrapped with the model's chat template via the *same* helper
    used at mining and eval time (``_build_generation_input_ids``), so the
    extracted answer-window activations live in the exact surface regime the
    model actually answers in. This matters most for harmony/CoT reasoning
    models (gpt-oss): a raw-text prompt omits the harmony system + channel
    scaffolding and is badly off-distribution, which previously left the whole
    A/B/C/D intervention built on a format the model never sees at inference.
    Base models fall back to plain text inside the helper.

    gpt-oss additionally gets the `final`-channel header
    (`<|channel|>final<|message|>`) appended after the assistant generation
    prompt. The chat template's generation prompt ends at `<|start|>assistant`,
    a position where the model always emits a channel marker (normally
    `analysis`) and NEVER raw answer text — replaying the answer there puts the
    K-token window somewhere the model never visits at inference, so V/μ± and
    the forget loss could only capture prompt-surface features. With the header
    the window sits where committal final answers actually live (the same
    placement the perplexity eval uses via ``open_final_channel``).

    The answer is encoded without special tokens and (if k_answer_tokens is
    given) truncated to its first K tokens — this matches the mining
    convention so that anchors and forget activations align.
    """
    prompt_ids = _build_generation_input_ids(tokenizer, model_key, prompt)
    if "gptoss" in model_key:
        # Prepend a minimal analysis-channel close followed by the final-channel
        # header. Previously only the final-channel header was prepended, so the
        # LoRA learned "output answer given final channel is already open" but
        # never learned to CLOSE the analysis channel and TRANSITION to final.
        # At inference the model generates analysis then transitions itself, and
        # a LoRA trained without that transition signal causes infinite analysis
        # (the analysis channel never closes → parse_harmony_final returns "" →
        # every example scores as ERROR in evaluate.py).
        # Adding the stub trains the LoRA on the correct context: close analysis,
        # open final, then emit the answer — matching the actual inference path.
        prompt_ids = prompt_ids + encode_text(
            tokenizer,
            "<|channel|>analysis<|message|><|end|><|channel|>final<|message|>",
            add_special_tokens=False,
        )
    answer_ids = encode_text(tokenizer, answer, add_special_tokens=False)
    if k_answer_tokens is not None:
        answer_ids = answer_ids[:k_answer_tokens]
    full_ids = list(prompt_ids) + list(answer_ids)
    return full_ids, len(prompt_ids), len(answer_ids)


def tokenise_chat_prompt_response(
    tokenizer,
    model_key: str,
    prompt: str,
    response: str,
    open_final_channel: bool = False,
) -> tuple[list[int], int]:
    """Tokenise (prompt, response) for retain-general training.

    Instruct models use chat templates; base models use plain text.
    Returns (full_ids, response_start).

    open_final_channel (gpt-oss only): when True, the harmony `final`-channel
    header (`<|channel|>final<|message|>`) is appended after the assistant
    generation prompt so the response is scored as legitimate final-channel
    content. Without it the response is appended right after `<|start|>assistant`,
    where the model expects a channel marker — which makes the first response
    token near-impossible and blows up perplexity. Eval (perplexity) opts in;
    training leaves it False so the retain target is unchanged.
    """
    if "base" in model_key:
        prompt_ids   = encode_text(tokenizer, prompt, add_special_tokens=True)
        response_ids = encode_text(tokenizer, response, add_special_tokens=False)
        return list(prompt_ids) + list(response_ids), len(prompt_ids)

    user_msg = [{"role": "user", "content": prompt}]
    if "ministral" in model_key:
        prompt_ids = tokenizer.apply_chat_template(
            user_msg, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
    elif "gptoss" in model_key:
        # Harmony chat prefix at the lowest reasoning effort. The UltraChat
        # response is appended at the generation point; the (prompt, response)
        # construction is identical for every retain-general row and for both
        # the frozen-base and adapter forward passes, so it yields a consistent
        # per-example retain target regardless of channel-position nuances.
        prompt_fmt = tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True,
            reasoning_effort=GPTOSS_REASONING_EFFORT,
        )
        if open_final_channel:
            prompt_fmt += "<|channel|>final<|message|>"
        prompt_ids = tokenizer.encode(prompt_fmt, add_special_tokens=False)
    else:
        prompt_fmt = tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_fmt, add_special_tokens=False)
    response_ids = encode_text(tokenizer, response, add_special_tokens=False)
    return list(prompt_ids) + list(response_ids), len(prompt_ids)


# ── Forward passes ────────────────────────────────────────────────────────────

def forward_hidden_states(
    model,
    input_ids: torch.Tensor,
    layer_indices: list[int] | None = None,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """Run a forward pass and optionally return hidden states for selected layers.

    - If layer_indices is None or empty: returns (logits, []) — used for KL.
    - Else: returns (logits, [h_l for l in layer_indices]) where each h_l has
      shape (1, seq_len, D) on CPU as float32.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    want_hidden = bool(layer_indices)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=want_hidden,
    )
    logits = outputs.logits.float().cpu() if outputs.logits is not None else None

    if not want_hidden:
        return logits, []

    # hidden_states[0] = embedding, hidden_states[l + 1] = layer l output
    all_hidden = outputs.hidden_states
    layers = [all_hidden[l + 1].float().cpu() for l in layer_indices]
    return logits, layers


def mean_answer_activation(
    layer_hiddens: list[torch.Tensor],
    prompt_len: int,
    n_answer_tokens: int,
) -> torch.Tensor:
    """Mean late-layer hidden state over the answer-token window per layer.

    Each entry of layer_hiddens has shape (1, seq_len, D); returns (L, D).
    """
    means = []
    for h in layer_hiddens:
        ans = h[0, prompt_len: prompt_len + n_answer_tokens, :]
        means.append(ans.mean(dim=0))
    return torch.stack(means)


# ── Greedy generation (shared by step 0 mining and step 5 evaluation) ────────

_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)


def parse_harmony_final(decoded: str) -> str:
    """Extract the `final`-channel text from a decoded gpt-oss harmony output.

    gpt-oss always emits an `analysis` (chain-of-thought) channel before the
    user-facing `final` channel; the CoT must never be judged or used as the
    committal answer. We return the content of the *last* final channel.

    If no final channel is present we return "" (an empty completion). This
    happens when the model exhausts the token budget mid-analysis, and an empty
    string is the integrity-preserving choice: callers treat it as a non-answer
    rather than risking the judge / forget-prefix seeing chain-of-thought text.
    It also makes any harmony-token decoding mismatch fail loudly during the
    smoke test instead of silently leaking CoT into the pipeline.
    """
    matches = _HARMONY_FINAL_RE.findall(decoded)
    if matches:
        return matches[-1].strip()
    return ""


def _build_generation_input_ids(tokenizer, model_key: str, prompt: str) -> list[int]:
    """Tokenise a prompt into model input ids for greedy generation, applying
    the model's chat template for instruct models."""
    if "base" in model_key:
        return list(tokenizer.encode(prompt, add_special_tokens=True))

    user_msg = [{"role": "user", "content": prompt}]
    if "ministral" in model_key:
        return list(tokenizer.apply_chat_template(
            user_msg, tokenize=True, add_generation_prompt=True, return_dict=False,
        ))
    if "gptoss" in model_key:
        # gpt-oss harmony template: select the lowest reasoning effort to keep
        # the analysis channel short. Thinking cannot be disabled outright.
        txt = tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True,
            reasoning_effort=GPTOSS_REASONING_EFFORT,
        )
        return list(tokenizer.encode(txt, add_special_tokens=False))
    txt = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(txt, add_special_tokens=False))


def generate_greedy(model, tokenizer, model_key: str, prompt: str,
                    max_new_tokens: int = 64) -> str:
    """One greedy completion. Handles base, instruct and harmony (gpt-oss) chat
    formats. For gpt-oss the raw channel output is parsed down to the committal
    final answer only.

    gpt-oss escalation: the analysis (CoT) channel precedes the final answer
    and occasionally exhausts the whole first-pass budget, leaving no
    final-channel text. Rather than always decoding with a large cap, the
    first pass runs at `max_new_tokens` and, if no final answer was produced,
    is repeated ONCE at GPTOSS_MAX_NEW_TOKENS_ESCALATED. Greedy decoding is
    deterministic, so the escalated pass replays the same analysis prefix and
    simply allows it to finish.
    """
    ids = _build_generation_input_ids(tokenizer, model_key, prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    def _new_tokens(budget: int) -> list[int]:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=budget,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or
                              getattr(tokenizer, "eos_token_id", 0),
            )
        return out[0, input_ids.shape[1]:].tolist()

    new_tokens = _new_tokens(max_new_tokens)

    if "gptoss" in model_key:
        # Decode WITH special tokens so the harmony channel markers survive for
        # parsing, then keep only the final-channel answer.
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        final = parse_harmony_final(decoded)
        if not final and max_new_tokens < GPTOSS_MAX_NEW_TOKENS_ESCALATED:
            decoded = tokenizer.decode(
                _new_tokens(GPTOSS_MAX_NEW_TOKENS_ESCALATED),
                skip_special_tokens=False,
            )
            final = parse_harmony_final(decoded)
        return final
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def first_k_token_prefix(tokenizer, text: str, k: int) -> str:
    """Re-encode `text` and decode the first k tokens.

    Used to build `y_com_prefix_k8` for forget rows so it tokenises identically
    when later replayed during activation extraction (step 1).
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < k:
        return text.strip()
    return tokenizer.decode(ids[:k], skip_special_tokens=True)


# ── Layer set ────────────────────────────────────────────────────────────────

def layer_indices_for(model_key: str) -> list[int]:
    return list(LAYER_SLICE[model_key])
