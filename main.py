import gc
import os
import re
import sys

import torch
from dotenv import load_dotenv

from llms import ALL_MODELS
from llms.constants import SHORTCUTS

# ── Models to evaluate ────────────────────────────────────────────────────────
TARGET_MODELS: dict[str, str] = {
    "Qwen/Qwen3.5-9B-Base":                  "base",
    "Qwen/Qwen3.5-9B":                        "instruct",
    "mistralai/Ministral-3-8B-Base-2512":     "base",
    "mistralai/Ministral-3-8B-Instruct-2512": "instruct",
    "microsoft/phi-4":                         "instruct",
    "meta-llama/Llama-3.3-70B-Instruct":       "instruct",
    "cerebras/gpt-oss-120b":                   "instruct",
}

BASE_PROMPT = "In a distant galaxy, a young scientist discovered"
INSTRUCT_MESSAGES = [{"role": "user", "content": "Generate a short story about a young scientist discovering a new planet."}]


def _free(pipe) -> None:
    """Release model weights and clear the MPS/CUDA cache."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_model(model_id: str, kind: str, hf_token: str) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Model : {model_id}")
    print(f"  Kind  : {kind}")
    print(sep)

    load_fn = ALL_MODELS[model_id]

    print("  [1/2] Loading …")
    pipe = load_fn(model_id, hf_token)
    print("  [1/2] Loaded.")

    print("  [2/2] Generating …")
    if kind == "base":
        result = pipe(BASE_PROMPT)
        output = result[0]["generated_text"]
        # strip the echo'd prompt so we only show the new tokens
        new_text = output[len(BASE_PROMPT):]
        print(f"\n  Prompt    : {BASE_PROMPT}")
        print(f"  Completion: {new_text.strip()}")
    else:
        result = pipe(INSTRUCT_MESSAGES)
        reply = result[0]["generated_text"][-1]["content"]
        print(f"\n  User  : {INSTRUCT_MESSAGES[0]['content']}")
        print(f"  Model : {reply.strip()}")

    print("  [2/2] Done.")
    _free(pipe)


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN", "")

    # Optional: pass a model name (or unique substring) as a CLI argument.
    # The match must end at a word boundary (end of string or followed by '/' or '-')
    # so "Qwen3.5-9B" matches Qwen/Qwen3.5-9B but not Qwen/Qwen3.5-9B-Base.
    def _matches(model_id: str, query: str) -> bool:
        lo = model_id.lower()
        q = query.lower()
        idx = lo.find(q)
        if idx == -1:
            return False
        end = idx + len(q)
        if end == len(lo) or lo[end] == "/":
            return True
        # Allow a trailing numeric version suffix like "-2512"
        return bool(re.match(r"^-\d+$", lo[end:]))

    if len(sys.argv) > 1:
        query = sys.argv[1]
        # Resolve shortcut first, then fall back to substring match
        if query in SHORTCUTS:
            full_id = SHORTCUTS[query]
            subset = {full_id: TARGET_MODELS[full_id]} if full_id in TARGET_MODELS else {}
        else:
            subset = {m: k for m, k in TARGET_MODELS.items() if _matches(m, query)}
        if not subset:
            print(f"No model matched '{query}'.")
            print("\nShortcuts:")
            for s, m in SHORTCUTS.items():
                print(f"  {s:20s} → {m}")
            print("\nFull model IDs:")
            for m in TARGET_MODELS:
                print(f"  {m}")
            sys.exit(1)
        models_to_run = subset
    else:
        models_to_run = TARGET_MODELS

    missing = [m for m in models_to_run if m not in ALL_MODELS]
    if missing:
        raise KeyError(f"Model(s) not registered in llms.ALL_MODELS: {missing}")

    for model_id, kind in models_to_run.items():
        run_model(model_id, kind, hf_token)

    print("\nDone.")


if __name__ == "__main__":
    main()
