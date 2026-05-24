from . import cerebras, gemma, llama, mistral, phi, qwen

ALL_MODELS: dict[str, callable] = {
    model_id: module.load
    for module in (cerebras, gemma, llama, mistral, phi, qwen)
    for model_id in module.MODELS
}
