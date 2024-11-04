from pathlib import Path
from gen_ai.constants.florence_2_task_types import Florence2TaskTypes

FLORENCE2_CAUSAL_LM_MODEL_ID = "microsoft/Florence-2-large-ft"
FLORENCE2_PROCESSOR_MODEL_ID = "microsoft/Florence-2-large-ft"
DEFAULT_TASK = Florence2TaskTypes.OPEN_VOCABULARY_DETECTION
CACHE_DIR = Path(__file__).parent.parent / "models" / "diffusers_cache" / "florence2"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
