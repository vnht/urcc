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
import time
from pathlib import Path

import torch

if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

from dotenv import load_dotenv

from config import (
    KUQ_PROMPT_TEMPLATE,
    LAYER_SLICE,
    MODEL_REGISTRY,
    REPO_ROOT,
    SQUAD_PROMPT_TEMPLATE,
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
    if dataset == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    if dataset == "squad":
        return SQUAD_PROMPT_TEMPLATE.format(
            question=row["question"], context=row.get("context", ""),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def build_answerable_prompt(dataset: str, row: dict) -> str:
    """Build the prompt for retain-answerable QA rows."""
    if dataset == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    if dataset == "squad":
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
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
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
    prompt: str,
    answer: str,
    k_answer_tokens: int | None = None,
) -> tuple[list[int], int, int]:
    """Returns (full_ids, prompt_len, n_answer_tokens).

    The answer is encoded without special tokens and (if k_answer_tokens is
    given) truncated to its first K tokens — this matches the mining
    convention so that anchors and forget activations align.
    """
    prompt_ids = encode_text(tokenizer, prompt, add_special_tokens=True)
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
) -> tuple[list[int], int]:
    """Tokenise (prompt, response) for retain-general training.

    Instruct models use chat templates; base models use plain text.
    Returns (full_ids, response_start).
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

def generate_greedy(model, tokenizer, model_key: str, prompt: str,
                    max_new_tokens: int = 64) -> str:
    """One greedy completion. Handles both base and instruct chat templates."""
    if "base" in model_key:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
    else:
        user_msg = [{"role": "user", "content": prompt}]
        if "ministral" in model_key:
            ids = tokenizer.apply_chat_template(
                user_msg, tokenize=True, add_generation_prompt=True, return_dict=False,
            )
            input_ids = torch.tensor([ids], dtype=torch.long)
        else:
            txt = tokenizer.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(txt, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or
                          getattr(tokenizer, "eos_token_id", 0),
        )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
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
