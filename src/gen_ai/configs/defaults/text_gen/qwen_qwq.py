from pathlib import Path

CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "transformers_cache"
    / "qwen_qwq"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions about everything. You should think step-by-step and provide detailed answers."

QWEN_QWQ_Q3_K_S_MODEL_ID = "bartowski/QwQ-32B-Preview-GGUF"
QWEN_QWQ_Q3_K_S_MODEL_FILENAME = "QwQ-32B-Preview-Q3_K_S.gguf"
