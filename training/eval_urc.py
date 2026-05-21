#!/usr/bin/env python3
"""
Post-training evaluation for a URC training run.

Loads the base model + LoRA adapter from a run directory, then runs the
exact same evaluation pipeline as the pre-training baseline:

    KUQ:       1000 instances  -> answerability metrics
    SQuAD:     1000 instances  -> answerability metrics
    UltraChat: 1000 examples   -> PPL

Saves results to <run-dir>/eval/ and writes a delta comparison against
the saved baseline metrics.

Usage:
    python3 evaluation/eval_run.py --run-dir training/runs/qwen_instruct_retainnorm_urc_last25_r8_beta1
    python3 evaluation/eval_run.py --run-dir training/runs/... --n 20  # smoke test
    python3 evaluation/eval_run.py --run-dir training/runs/... --skip-ppl
"""

# ── Triton mock — must be first ───────────────────────────────────────────────
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
                return importlib.machinery.ModuleSpec(fullname, self._loader, is_package=True)
            return None

    sys.meta_path.insert(0, _TritonMockFinder())

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse
import gc
import json
import logging
import math
import os
import time
import warnings
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*apply_chat_template.*tokenize=False.*")
log = logging.getLogger(__name__)

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
EVAL_DIR  = REPO_ROOT / "training" / "held-out-eval-data"
BASELINE_DIR = REPO_ROOT / "training" / "pre-training-baselines"

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from evaluation.judge import build_judge_prompt, call_judge, make_cerebras_client, JUDGE_MODEL_ID  # type: ignore[import]
from llms.constants import SHORTCUTS  # type: ignore[import]

ID_TO_SHORTCUT = {v: k for k, v in SHORTCUTS.items()}

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

MAX_NEW_TOKENS = 64
MAX_SEQ_LEN    = 256  # for PPL tokenisation, matches baseline


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_with_adapter(model_key: str, model_id: str, run_dir: Path, hf_token: str):
    """Load base model + LoRA adapter from run_dir."""
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required.  pip install peft")

    log.info("  Loading base model %s ...", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )

    elif model_id.startswith("mistralai/"):
        from transformers import (
            MistralCommonBackend, Mistral3ForConditionalGeneration, FineGrainedFP8Config,
        )
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
        if sys.platform == "darwin":
            base_model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu", token=hf_token,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                tie_word_embeddings=False,
            )
            base_model = base_model.to("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            base_model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto",
                token=hf_token, tie_word_embeddings=False,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    log.info("  Base loaded in %.1fs — applying adapter from %s ...", time.time() - t0, run_dir.name)
    model = PeftModel.from_pretrained(base_model, str(run_dir))
    model.eval()
    log.info("  Adapter applied.  Total params: %s", sum(p.numel() for p in model.parameters()))
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(row: dict) -> str:
    if row["dataset"] == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    return SQUAD_PROMPT_TEMPLATE.format(
        context=row["context"], question=row["question"],
    )


def generate_completion(model, tokenizer, prompt: str, model_key: str, device) -> str:
    """Greedy decode with the same prompt format as the baseline pipeline."""
    is_base = "base" in model_key

    if is_base:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        messages = [{"role": "user", "content": prompt}]
        if "qwen" in model_key:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if not isinstance(
                tokenizer.eos_token_id, list
            ) else tokenizer.eos_token_id[0],
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_generation(model, tokenizer, model_key: str, eval_dir: Path, out_path: Path, n=None) -> list[dict]:
    """Generate on KUQ + SQuAD; resume from existing output if present."""
    instances = []
    for dataset in ("kuq", "squad"):
        path = eval_dir / f"{dataset}_1000.jsonl"
        rows_ds = []
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows_ds.append({
                    "dataset": dataset,
                    "id": row["id"],
                    "answerable": row["answerable"],
                    "question": row["question"],
                    "context": row.get("context"),
                })
        if n:
            rows_ds = rows_ds[:n]
        instances.extend(rows_ds)

    done_ids: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        log.info("  Resuming generation — %d / %d done", len(done_ids), len(instances))

    remaining = [inst for inst in instances if inst["id"] not in done_ids]
    device = next(model.parameters()).device

    model_id = MODEL_REGISTRY[model_key]
    with open(out_path, "a") as fout:
        bar = tqdm(remaining, desc="generate", unit="inst", dynamic_ncols=True)
        for inst in bar:
            prompt = build_prompt(inst)
            t0 = time.time()
            try:
                completion = generate_completion(model, tokenizer, prompt, model_key, device)
            except Exception as exc:
                log.warning("  Generation error [%s %s]: %s", inst["dataset"], inst["id"], exc)
                completion = ""

            elapsed = time.time() - t0
            fout.write(json.dumps({
                "id":                inst["id"],
                "dataset":           inst["dataset"],
                "answerable":        inst["answerable"],
                "question":          inst["question"],
                "context":           inst.get("context"),
                "model":             model_id,
                "generation_prompt": prompt,
                "completion":        completion,
                "gen_time_s":        round(elapsed, 3),
            }) + "\n")
            fout.flush()
            bar.set_postfix_str(f"{elapsed:.2f}s | {completion[:50].replace(chr(10), ' ')}")

    return [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]


# ── Judging ───────────────────────────────────────────────────────────────────

def judge_rows(rows: list[dict], gen_path: Path, client) -> list[dict]:
    needs = [r for r in rows if r.get("judge_label") not in ("COMMITTED", "ABSTAINED")]
    if not needs:
        log.info("  All %d rows already judged", len(rows))
        return rows

    log.info("  Judging %d / %d rows", len(needs), len(rows))
    judged_map = {r["id"]: r for r in rows}
    committed = abstained = errors = 0

    bar = tqdm(needs, desc="judge", unit="inst", dynamic_ncols=True)
    for row in bar:
        prompt = build_judge_prompt(
            question=row["question"],
            completion=row["completion"],
            context=row.get("context"),
        )
        t0 = time.time()
        label, raw = call_judge(client, prompt)
        elapsed = time.time() - t0

        judged_map[row["id"]].update({
            "judge_label":      label,
            "judge_model":      JUDGE_MODEL_ID,
            "judge_raw_output": raw,
            "judge_time_s":     round(elapsed, 3),
        })
        if label == "COMMITTED":   committed += 1
        elif label == "ABSTAINED": abstained += 1
        else:                      errors    += 1
        bar.set_postfix(C=committed, A=abstained, E=errors)

    all_rows = list(judged_map.values())
    with open(gen_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    return all_rows


# ── Answerability metrics ─────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], model_id: str, dataset: str) -> dict:
    subset       = [r for r in rows if r["dataset"] == dataset]
    answerable   = [r for r in subset if r["answerable"]]
    unanswerable = [r for r in subset if not r["answerable"]]

    def valid(group):
        return [r for r in group if r.get("judge_label") in ("COMMITTED", "ABSTAINED")]

    ans_valid   = valid(answerable)
    unans_valid = valid(unanswerable)
    num_errors  = len(subset) - len(ans_valid) - len(unans_valid)

    def rate(group, label):
        return (sum(1 for r in group if r.get("judge_label") == label) / len(group)
                if group else float("nan"))

    true_commits  = sum(1 for r in ans_valid   if r.get("judge_label") == "COMMITTED")
    true_abst     = sum(1 for r in unans_valid if r.get("judge_label") == "ABSTAINED")
    dec_acc       = (true_commits + true_abst) / len(subset) if subset else float("nan")

    return {
        "model":                 model_id,
        "dataset":               dataset,
        "num_instances":         len(subset),
        "num_answerable":        len(answerable),
        "num_unanswerable":      len(unanswerable),
        "num_judge_errors":      num_errors,
        "true_commitment_rate":  round(rate(ans_valid,   "COMMITTED"), 4),
        "false_abstention_rate": round(rate(ans_valid,   "ABSTAINED"), 4),
        "true_abstention_rate":  round(rate(unans_valid, "ABSTAINED"), 4),
        "false_commitment_rate": round(rate(unans_valid, "COMMITTED"), 4),
        "decision_accuracy":     round(dec_acc, 4),
    }


# ── PPL ───────────────────────────────────────────────────────────────────────

def _tokenise_ppl(tokenizer, ex: dict, model_key: str) -> tuple[list[int], list[int]]:
    """Returns (full_ids, resp_ids), truncated to MAX_SEQ_LEN.  Matches baseline."""
    is_base = "base" in model_key
    prompt   = ex.get("prompt", "")
    response = ex.get("response", "")

    if is_base:
        full_ids = tokenizer.encode(prompt + " " + response, add_special_tokens=True)
        resp_ids = tokenizer.encode(response, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
        )
        resp_ids   = tokenizer.encode(response, add_special_tokens=False)
        full_ids   = list(prompt_ids) + resp_ids

    if len(full_ids) > MAX_SEQ_LEN:
        keep_resp = min(len(resp_ids), MAX_SEQ_LEN)
        resp_ids  = resp_ids[-keep_resp:]
        full_ids  = full_ids[-MAX_SEQ_LEN:]

    return full_ids, resp_ids


def compute_ppl(model, tokenizer, model_key: str, examples: list[dict]) -> tuple[dict, list[dict]]:
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tok = 0
    skipped   = 0
    rows      = []

    model.eval()
    bar = tqdm(examples, desc="ppl", unit="ex", dynamic_ncols=True)
    for i, ex in enumerate(bar):
        try:
            full_ids, resp_ids = _tokenise_ppl(tokenizer, ex, model_key)
        except Exception as exc:
            log.warning("  PPL tokenisation error [%d]: %s", i, exc)
            skipped += 1
            continue

        if not resp_ids:
            skipped += 1
            continue

        n_prompt  = len(full_ids) - len(resp_ids)
        labels    = [-100] * n_prompt + list(resp_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        label_ids = torch.tensor([labels],  dtype=torch.long, device=device)

        with torch.no_grad():
            nll = model(input_ids=input_ids, labels=label_ids).loss.item()

        ppl_i = math.exp(nll)
        rows.append({
            "id":         ex.get("id", i),
            "num_tokens": len(resp_ids),
            "nll":        round(nll,   4),
            "ppl":        round(ppl_i, 3),
        })
        total_nll += nll * len(resp_ids)
        total_tok += len(resp_ids)

        bar.set_postfix(ppl=f"{math.exp(total_nll/total_tok):.2f}", skip=skipped)

    mean_nll = total_nll / total_tok if total_tok else float("nan")
    ppl      = math.exp(mean_nll)    if total_tok else float("nan")
    log.info("  mean_nll=%.4f  ppl=%.3f  tokens=%d  skipped=%d",
             mean_nll, ppl, total_tok, skipped)

    summary = {
        "num_examples": len(examples) - skipped,
        "num_tokens":   total_tok,
        "mean_nll":     round(mean_nll, 4),
        "ppl":          round(ppl, 3),
    }
    return summary, rows


# ── Baseline loader ───────────────────────────────────────────────────────────

def load_baseline_metrics(model_key: str) -> dict:
    """Load pre-training baseline answerability metrics for this model."""
    model_id  = MODEL_REGISTRY[model_key]
    slug      = ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__"))
    ans_path  = BASELINE_DIR / "eval_baseline_answerability_metrics.jsonl"
    ppl_path  = BASELINE_DIR / "eval_baseline_ultrachat_ppl.jsonl"

    baseline: dict = {"answerability": {}, "ppl": None}

    if ans_path.exists():
        for line in ans_path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                if ID_TO_SHORTCUT.get(m.get("model"), m.get("model")) == slug:
                    baseline["answerability"][m["dataset"]] = m

    if ppl_path.exists():
        for line in ppl_path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                if m.get("model") == slug:
                    baseline["ppl"] = m

    return baseline


# ── Comparison ────────────────────────────────────────────────────────────────

def build_comparison(baseline: dict, post: dict, ppl_baseline, ppl_post) -> dict:
    comparison: dict = {"answerability": {}, "ppl": {}}

    for dataset in ("kuq", "squad"):
        b = baseline["answerability"].get(dataset, {})
        p = post.get(dataset, {})
        delta = {}
        for key in ("true_commitment_rate", "false_abstention_rate",
                    "true_abstention_rate", "false_commitment_rate", "decision_accuracy"):
            bv = b.get(key)
            pv = p.get(key)
            if bv is not None and pv is not None:
                delta[key] = round(pv - bv, 4)
        comparison["answerability"][dataset] = {
            "baseline":      b,
            "post_training": p,
            "delta":         delta,
        }

    comparison["ppl"] = {
        "baseline_ppl":      ppl_baseline.get("ppl") if ppl_baseline else None,
        "post_training_ppl": ppl_post.get("ppl")     if ppl_post     else None,
        "delta_ppl":         (
            round(ppl_post["ppl"] - ppl_baseline["ppl"], 3)
            if ppl_post and ppl_baseline else None
        ),
    }
    return comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for a URC training run."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to the training run directory")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit eval instances (smoke test)")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip UltraChat PPL computation")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        log.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        log.error("training_config.json not found in %s", run_dir)
        sys.exit(1)
    config   = json.loads(config_path.read_text())
    model_key = config["model_key"]
    model_id  = config["model_id"]

    eval_out = run_dir / "eval"
    eval_out.mkdir(exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")
    slug = ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__"))

    log.info("=" * 64)
    log.info("Evaluating run: %s", run_dir.name)
    log.info("  model_key : %s", model_key)
    log.info("  model_id  : %s", model_id)
    log.info("  eval_out  : %s", eval_out)

    # ── 1. Load model + adapter ───────────────────────────────────────────────
    model, tokenizer = load_model_with_adapter(model_key, model_id, run_dir, hf_token)
    device = next(model.parameters()).device

    # ── 2. Generation ─────────────────────────────────────────────────────────
    log.info("")
    log.info("── Generation ──────────────────────────────────────────────────")
    gen_path = eval_out / f"eval_generations_{slug}.jsonl"
    rows = run_generation(model, tokenizer, model_key, EVAL_DIR, gen_path, n=args.n)
    log.info("  %d rows generated/loaded", len(rows))

    # ── 3. Judge ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("── Judging ─────────────────────────────────────────────────────")
    judge_client = make_cerebras_client()
    rows = judge_rows(rows, gen_path, judge_client)

    # ── 4. Answerability metrics ──────────────────────────────────────────────
    all_metrics: list[dict] = []
    post_metrics: dict = {}
    for dataset in ("kuq", "squad"):
        m = compute_metrics(rows, model_id, dataset)
        all_metrics.append(m)
        post_metrics[dataset] = m
        log.info(
            "  %-6s  acc=%.3f  TCR=%.3f  FAR=%.3f  TAR=%.3f  FCR=%.3f",
            dataset, m["decision_accuracy"],
            m["true_commitment_rate"],  m["false_abstention_rate"],
            m["true_abstention_rate"],  m["false_commitment_rate"],
        )

    metrics_path = eval_out / "eval_answerability_metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")
    log.info("  Saved -> %s", metrics_path.name)

    # ── 5. PPL ────────────────────────────────────────────────────────────────
    ppl_summary_post = None
    if not args.skip_ppl:
        log.info("")
        log.info("── UltraChat PPL ───────────────────────────────────────────────")
        ultrachat_path = EVAL_DIR / "ultrachat_1000.jsonl"
        uc_examples = [
            json.loads(l) for l in ultrachat_path.read_text().splitlines() if l.strip()
        ]
        if args.n:
            uc_examples = uc_examples[:args.n]

        ppl_summary_post, ppl_rows = compute_ppl(model, tokenizer, model_key, uc_examples)

        ppl_inst_path = eval_out / f"eval_ultrachat_ppl_{slug}.jsonl"
        with open(ppl_inst_path, "w") as f:
            for row in ppl_rows:
                f.write(json.dumps(row) + "\n")

        ppl_sum_path = eval_out / "eval_ultrachat_ppl.jsonl"
        with open(ppl_sum_path, "w") as f:
            f.write(json.dumps({"model": slug, **ppl_summary_post}) + "\n")
        log.info("  Saved -> %s", ppl_sum_path.name)

    # ── 6. Comparison vs baseline ─────────────────────────────────────────────
    log.info("")
    log.info("── Comparison vs baseline ──────────────────────────────────────")
    baseline = load_baseline_metrics(model_key)
    comparison = build_comparison(baseline, post_metrics, baseline.get("ppl"), ppl_summary_post)
    comparison["run"]       = run_dir.name
    comparison["model_key"] = model_key
    comparison["model_id"]  = model_id

    comp_path = eval_out / "eval_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    log.info("  Saved -> %s", comp_path.name)

    # Print summary
    log.info("")
    log.info("  %-10s  %-6s  %10s  %10s  %10s",
             "dataset", "metric", "baseline", "post", "delta")
    log.info("  " + "-" * 52)
    for dataset in ("kuq", "squad"):
        c = comparison["answerability"].get(dataset, {})
        delta = c.get("delta", {})
        for k in ("false_commitment_rate", "false_abstention_rate", "decision_accuracy"):
            b_val = c.get("baseline", {}).get(k, "—")
            p_val = c.get("post_training", {}).get(k, "—")
            d_val = delta.get(k, "—")
            log.info("  %-10s  %-6s  %10s  %10s  %+10s",
                     dataset, k[:6], b_val, p_val,
                     f"{d_val:+.4f}" if isinstance(d_val, float) else d_val)

    if comparison["ppl"].get("baseline_ppl") is not None:
        p = comparison["ppl"]
        post_ppl  = p.get("post_training_ppl")
        delta_ppl = p.get("delta_ppl")
        if post_ppl is not None and delta_ppl is not None:
            log.info("  %-10s  %-6s  %10.3f  %10.3f  %+10.3f",
                     "ultrachat", "PPL", p["baseline_ppl"], post_ppl, delta_ppl)
        else:
            log.info("  %-10s  %-6s  %10.3f  %10s  %10s",
                     "ultrachat", "PPL", p["baseline_ppl"], "—(skipped)", "—")

    log.info("")
    log.info("Done -> %s", eval_out)


if __name__ == "__main__":
    main()
