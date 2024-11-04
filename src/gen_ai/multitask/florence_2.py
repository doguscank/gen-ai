import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from gen_ai.configs import florence_2 as florence_cfg
from gen_ai.multitask.florence_2_model_config import Florence2ModelConfig
from gen_ai.multitask.florence_2_input_config import Florence2InputConfig
from gen_ai.img_utils import load_image
import cv2
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
from gen_ai.utils import pathify_strings
from gen_ai.multitask.florence_2_outputs import (
    BoundingBoxes,
    OpenVocabularyDetection,
    OCR,
    Caption,
    Polygons,
    QuadBoxes,
)
from gen_ai.multitask.florence_2_output_parsers import parse_output


class Florence2:
    def __init__(
        self,
        *,
        config: Optional[Florence2ModelConfig] = None,
    ) -> None:
        """
        Initialize the Florence2 with the given configuration.

        Parameters
        ----------
        config : Florence2ModelConfig, optional
            Florence2ModelConfig object containing model configuration.
        """

        self.model_config = config

        if self.model_config is not None:
            self._load_pipeline()

    def _load_pipeline(
        self,
        causal_lm_hf_model_id: Optional[str] = None,
        processor_hf_model_id: Optional[str] = None,
        causal_lm_model_path: Optional[Path] = None,
        processor_model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Load the pipeline with the given configuration.

        Parameters
        ----------
        causal_lm_hf_model_id : str, optional
            The identifier of the causal language model to use.
        processor_hf_model_id : str, optional
            The identifier of the processor to use.
        causal_lm_model_path : Path, optional
            The path to the causal language model.
        processor_model_path : Path, optional
            The path to the processor.

        Returns
        -------
        None
        """

        if causal_lm_hf_model_id is None and causal_lm_model_path is None:
            raise ValueError("No causal language model provided.")

        if processor_hf_model_id is None and processor_model_path is None:
            raise ValueError("No processor provided.")

        model_descriptor = None
        processor_descriptor = None

        if causal_lm_hf_model_id is not None:
            if self.model_config.causal_lm_hf_model_id == causal_lm_hf_model_id:
                if (
                    processor_hf_model_id is not None
                    and self.model_config.processor_hf_model_id == processor_hf_model_id
                ):
                    return
                if (
                    processor_model_path is not None
                    and self.model_config.processor_model_path == processor_model_path
                ):
                    return
            model_descriptor = causal_lm_hf_model_id

        if causal_lm_model_path is not None:
            if self.model_config.causal_lm_model_path == causal_lm_model_path:
                if (
                    processor_hf_model_id is not None
                    and self.model_config.processor_hf_model_id == processor_hf_model_id
                ):
                    return
                if (
                    processor_model_path is not None
                    and self.model_config.processor_model_path == processor_model_path
                ):
                    return
            model_descriptor = causal_lm_model_path

        if processor_hf_model_id is not None:
            processor_descriptor = processor_hf_model_id

        if processor_model_path is not None:
            processor_descriptor = processor_model_path

        if device is None:
            device = self.model_config.device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_descriptor,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            cache_dir=florence_cfg.CACHE_DIR,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            processor_descriptor,
            trust_remote_code=True,
            cache_dir=florence_cfg.CACHE_DIR,
        )

    def update_pipeline(self, model_config: Florence2ModelConfig) -> None:
        """
        Update the pipeline with the given configuration.

        Parameters
        ----------
        model_config : Florence2ModelConfig
            Florence2ModelConfig object containing model configuration.

        Returns
        -------
        None
        """

        self._load_pipeline(
            causal_lm_hf_model_id=model_config.causal_lm_hf_model_id,
            processor_hf_model_id=model_config.processor_hf_model_id,
            causal_lm_model_path=model_config.causal_lm_model_path,
            processor_model_path=model_config.processor_model_path,
            device=model_config.device,
        )

        self.model_config = model_config

    def predict(
        self,
        config: Florence2InputConfig,
    ) -> Union[
        Caption, BoundingBoxes, Polygons, OpenVocabularyDetection, OCR, QuadBoxes, None
    ]:
        """
        Predict the output for the given input configuration.

        Parameters
        ----------
        config : Florence2InputConfig
            Florence2InputConfig object containing input configuration.

        Returns
        -------
        Union[Caption, BoundingBoxes, Polygons, OpenVocabularyDetection, OCR, 
        QuadBoxes, None]
            The output of the prediction.
        """

        inputs = self.processor(
            text=config.prompt,
            images=config.image,
            return_tensors="pt",
        ).to(self.model_config.device, self.model_config.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_length=config.max_new_tokens,
            num_beams=config.num_beams,
            early_stopping=config.early_stopping,
            do_sample=config.do_sample,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=config.task_prompt,
            image_size=(config.image.width, config.image.height),
        )

        parsed_result = parse_output(data=parsed_answer, task_type=config.task_prompt)

        return parsed_result
