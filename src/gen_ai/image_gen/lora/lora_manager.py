import json
from pathlib import Path
from typing import Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from gen_ai.logger import logger
from gen_ai.utils import pathify_strings


class LoraModel(BaseModel):
    path: Path
    trigger_words: Optional[Union[str, List[str]]] = None
    name: str = Field(default="", init=False, repr=True)
    scale: float = Field(default=1.0, init=False, repr=True)
    is_loaded: bool = Field(default=False, init=False, repr=True)

    def model_post_init(self, __context) -> "LoraModel":
        self.name = self.path.stem

        return self

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def set_loaded(self) -> None:
        if not self.is_loaded:
            logger.info(f"LoRA model {self.name} loaded.")
            self.is_loaded = True

    def set_unloaded(self) -> None:
        if self.is_loaded:
            logger.info(f"LoRA model {self.name} unloaded.")
            self.is_loaded = False


@pathify_strings
def _get_trigger_words(lora_path: Path) -> Optional[Union[str, List[str]]]:
    # replace extension of lora path from ".safetensors" to ".json"
    trigger_words_json_path = lora_path.with_suffix(".json")
    if trigger_words_json_path.exists():
        with trigger_words_json_path.open("r") as f:
            trigger_words = json.load(f)["trigger_words"]
        return trigger_words

    # replace extension of lora path from ".safetensors" to ".txt"
    trigger_words_txt_path = lora_path.with_suffix(".txt")
    if trigger_words_txt_path.exists():
        with trigger_words_txt_path.open("r") as f:
            trigger_words = f.readlines()
        return trigger_words

    return None


class LoraManager:
    registered_models: List[LoraModel] = []

    @pathify_strings
    def __init__(
        self, lora_dir: Optional[Path] = None, auto_register: bool = True
    ) -> None:
        self._lora_dir = lora_dir
        self._auto_register = auto_register

        if lora_dir is not None and auto_register:
            self.register_lora_models(lora_dir)

    @property
    def lora_dir(self) -> Path:
        return self._lora_dir

    @property
    def auto_register(self) -> bool:
        return self._auto_register

    @property
    def models(self) -> List[LoraModel]:
        return LoraManager.registered_models

    @property
    def model_paths(self) -> List[Path]:
        return [model.path for model in self.models]

    @property
    def model_names(self) -> List[str]:
        return [model.name for model in self.models]

    @property
    def trigger_words(self) -> List[str]:
        trigger_words = []
        for model in self.models:
            trigger_words.extend(model.trigger_words)
        return trigger_words

    def _get_lora_models(self) -> List[LoraModel]:
        if self.lora_dir is None:
            logger.warning("No LoRA directory provided. Skipping LoRA model loading.")
            return []

        lora_models = []
        for lora_path in self.lora_dir.glob("*.safetensors"):
            trigger_words = _get_trigger_words(lora_path)
            lora_models.append(LoraModel(path=lora_path, trigger_words=trigger_words))

        return lora_models

    @pathify_strings
    def register_lora_model(
        self, lora_path: Path, trigger_words: Optional[Union[str, List[str]]] = None
    ) -> None:
        if trigger_words is None:
            trigger_words = _get_trigger_words(lora_path)

        if lora_path in self.model_paths:
            logger.warning(f"LoRA model {lora_path} is already registered.")
            return

        if any(trigger_word in self.trigger_words for trigger_word in trigger_words):
            logger.warning(
                f"Trigger words for LoRA model {lora_path} are already registered by "
                f"{self.get_model_by_trigger_word(trigger_words)}. "
                "Skipping registration."
            )
            return

        new_model = LoraModel(path=lora_path, trigger_words=trigger_words)
        LoraManager.registered_models.append(new_model)

    @pathify_strings
    def register_lora_models(self, lora_dir: Path) -> None:
        for lora_path in lora_dir.glob("*.safetensors"):
            self.register_lora_model(lora_path)

    def get_model_by_name(self, name: str) -> Optional[LoraModel]:
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_model_by_trigger_word(self, trigger_word: str) -> Optional[LoraModel]:
        for model in self.models:
            if trigger_word in model.trigger_words:
                return model
        return None

    def merge(self, other: Union["LoraManager", List["LoraManager"]]) -> None:
        if isinstance(other, list):
            for manager in other:
                self.merge(manager)
            return

        for model in other.models:
            self.register_lora_model(model.path, model.trigger_words)

    def get_lora_models_from_prompt(self, prompt: str) -> List[LoraModel]:
        lora_models = []

        for model in self.models:
            if model.trigger_words is not None:
                for trigger_word in model.trigger_words:
                    if trigger_word in prompt:
                        lora_models.append(model)

        return lora_models

    def iter_models(self) -> Iterator[LoraModel]:
        for model in self.models:
            yield from model

    def __iter__(self) -> Iterator[LoraModel]:
        return self.iter_models()

    def __len__(self) -> int:
        return len(self.models)
