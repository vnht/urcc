import os

from cerebras.cloud.sdk import Cerebras

MODELS = [
    "cerebras/gpt-oss-120b",
]

# Map registry key → Cerebras API model ID
_API_IDS: dict[str, str] = {
    "cerebras/gpt-oss-120b": "gpt-oss-120b",
}


class _CerebrasWrapper:
    """Thin wrapper that mimics the HuggingFace pipeline __call__ interface."""

    def __init__(self, api_model_id: str, api_key: str) -> None:
        self._model_id = api_model_id
        self._client = Cerebras(api_key=api_key)

    def __call__(self, input_: str | list[dict]) -> list[dict]:
        if isinstance(input_, str):
            # Base-model style: single prompt string → plain completion
            messages = [{"role": "user", "content": input_}]
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
            )
            content = response.choices[0].message.content
            return [{"generated_text": input_ + content}]
        else:
            # Instruct style: list of message dicts
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=input_,
            )
            content = response.choices[0].message.content
            reply = {"role": "assistant", "content": content}
            return [{"generated_text": list(input_) + [reply]}]


def load(model_id: str, hf_token: str) -> _CerebrasWrapper:  # noqa: ARG001
    api_key = os.environ.get("CEREBRAS_TOKEN", "")
    api_model_id = _API_IDS[model_id]
    return _CerebrasWrapper(api_model_id, api_key)
