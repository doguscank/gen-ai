import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from gen_ai.logger import logger
from gen_ai.utils import pathify_strings


class LoraModel(BaseModel):
    path: Path
    trigger_words: Optional[Union[str, List[str]]] = None
    name: str = Field(default="", init=False, repr=True)
    scale: float = Field(default=1.0, init=False, repr=True)

    def model_post_init(self, __context) -> "LoraModel":
        self.name = self.path.stem

        return self

    def set_scale(self, scale: float) -> None:
        self.scale = scale


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

    @property
    def lora_dir(self) -> Path:
        return self._lora_dir

    @property
    def auto_register(self) -> bool:
        return self._auto_register

    @property
    def models(self) -> List[LoraModel]:
        return LoraManager.registered_models
