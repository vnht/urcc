import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class _SuppressFastPathWarning(logging.Filter):
    """flash-linear-attention and causal-conv1d are CUDA-only; silence the macOS fallback notice."""
    def filter(self, record):
        return "fast path is not available" not in record.getMessage()

MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-35B-A3B",
]


def _build_gen_config(tokenizer) -> GenerationConfig:
    eos = tokenizer.eos_token_id
    return GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=eos[0] if isinstance(eos, list) else eos,
    )


class QwenPipeline:
    """Instruct pipeline: disables Qwen3 thinking mode, accepts a list of messages."""

    def __init__(self, model, tokenizer, gen_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device

    def __call__(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
            )[0]

        new_tokens = output_ids[len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": messages + [{"role": "assistant", "content": response}]}]


class QwenBasePipeline:
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


def load(model_id: str, hf_token: str):
    _filter = _SuppressFastPathWarning()
    _tf_logger = logging.getLogger("transformers")
    for _h in _tf_logger.handlers:
        _h.addFilter(_filter)
    # Also cover the root logger in case transformers logs there
    logging.getLogger().addFilter(_filter)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
    finally:
        for _h in _tf_logger.handlers:
            _h.removeFilter(_filter)
        logging.getLogger().removeFilter(_filter)

    device = next(model.parameters()).device
    gen_config = _build_gen_config(tokenizer)

    if "Base" in model_id:
        return QwenBasePipeline(model, tokenizer, gen_config, device)
    return QwenPipeline(model, tokenizer, gen_config, device)
