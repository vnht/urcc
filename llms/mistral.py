import sys
import torch
from unittest.mock import MagicMock

# On macOS, torch.compile requires triton (CUDA-only).
# Replacing it with an identity decorator before any FP8 imports prevents the
# entire triton import chain from being triggered at module-load time.
if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kwargs: (fn if fn is not None else lambda f: f)
    # triton.runtime.jit is still referenced by the FP8 dequantisation kernels
    for _mod in ("triton", "triton.language", "triton.runtime", "triton.runtime.jit"):
        sys.modules.setdefault(_mod, MagicMock())

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    FineGrainedFP8Config,
    GenerationConfig,
)

MODELS = [
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
]

_INSTRUCT_MODELS = {m for m in MODELS if "Instruct" in m}
_BASE_MODELS = {m for m in MODELS if "Base" in m}


def _build_gen_config(tokenizer) -> GenerationConfig:
    eos = tokenizer.eos_token_id
    return GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=eos[0] if isinstance(eos, list) else eos,
    )


class MistralPipeline:
    """Instruct pipeline using MistralCommonBackend chat template. Accepts a list of messages."""

    def __init__(self, model, tokenizer, gen_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device

    def __call__(self, messages):
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.gen_config,
            )[0]

        new_tokens = output_ids[len(input_ids[0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": messages + [{"role": "assistant", "content": response}]}]


class MistralBasePipeline:
    """Base model pipeline: accepts a plain-text prompt string, returns a completion."""

    def __init__(self, model, tokenizer, gen_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device

    def __call__(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
            )[0]

        new_tokens = output_ids[len(inputs["input_ids"][0]):]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": prompt + completion}]


def _load_instruct(model_id: str, hf_token: str) -> MistralPipeline:
    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)

    # Load with native FP8 (triton kernel used automatically on CUDA).
    # Do NOT pass FineGrainedFP8Config(dequantize=True) — that BF16 conversion
    # path is broken on older transformers versions. Let transformers handle FP8
    # natively via device_map="auto".
    if sys.platform == "darwin":
        # MPS does not support FP8; dequantize to BF16 on CPU then move to MPS
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cpu",
            token=hf_token,
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
    else:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            token=hf_token,
        )
        device = next(model.parameters()).device

    gen_config = _build_gen_config(tokenizer)
    model.generation_config.max_length = None
    return MistralPipeline(model, tokenizer, gen_config, device)


def _load_base(model_id: str, hf_token: str) -> MistralBasePipeline:
    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
    # Base weights are standard BF16 (no FP8); Mistral3 arch still needs its own class
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        tie_word_embeddings=False,
    )

    device = next(model.parameters()).device
    gen_config = _build_gen_config(tokenizer)
    model.generation_config.max_length = None
    return MistralBasePipeline(model, tokenizer, gen_config, device)


def load(model_id: str, hf_token: str):
    if model_id in _BASE_MODELS:
        return _load_base(model_id, hf_token)
    return _load_instruct(model_id, hf_token)
