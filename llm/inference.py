import os
from dotenv import load_dotenv
from llama_cpp import Llama, CreateCompletionResponse
from typing import cast

load_dotenv()

_model = None

def get_model() -> Llama:
    global _model
    if _model is None:
        _model = Llama(
            model_path=os.environ["MODEL_PATH"],
            n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", 41)),
            n_ctx=int(os.environ.get("N_CTX", 4096)),
            verbose=False,
        )
    return _model

def generate(prompt: str, max_tokens: int = 512, stop: list[str] | None = None) -> str:
    llm = get_model()
    result = cast(
        CreateCompletionResponse,
        llm(prompt, max_tokens=max_tokens, stop=stop or ["<|im_end|>", "<|endoftext|>"], stream=False),
    )
    return result["choices"][0]["text"].strip()