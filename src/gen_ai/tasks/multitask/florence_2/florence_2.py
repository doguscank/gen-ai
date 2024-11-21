from pathlib import Path
from typing import Optional, Union

from transformers import AutoModelForCausalLM, AutoProcessor

from gen_ai.configs.defaults import florence_2_cfg
from gen_ai.logger import logger
from gen_ai.tasks.multitask.florence_2.input import Florence2Input
from gen_ai.tasks.multitask.florence_2.model_config import Florence2ModelConfig
from gen_ai.tasks.multitask.florence_2.output_parsers import parse_output
from gen_ai.tasks.multitask.florence_2.outputs import (
    OCR,
    BoundingBoxes,
    Caption,
    OpenVocabularyDetection,
    Polygons,
    QuadBoxes,
)


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
        self.model = None
        self.processor = None

        if self.model_config is not None:
            logger.info(f"Loading Florence2 model with config: {self.model_config}")

            self._load_pipeline(
                causal_lm_hf_model_id=self.model_config.causal_lm_hf_model_id,
                processor_hf_model_id=self.model_config.processor_hf_model_id,
                causal_lm_model_path=self.model_config.causal_lm_model_path,
                processor_model_path=self.model_config.processor_model_path,
                device=self.model_config.device,
            )
        else:
            logger.info("No Florence2 model configuration provided.")

    def check_model_ready(self) -> bool:
        """
        Check if the model is ready.

        Returns
        -------
        bool
            True if the model is ready, False otherwise.
        """

        return self.model is not None and self.processor is not None

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
                    processor_hf_model_id is not None  # given
                    and (
                        self.model_config.processor_hf_model_id == processor_hf_model_id
                    )  # same with the loaded one
                    and self.model is not None  # already loaded
                    and self.processor is not None  # already loaded
                ):
                    return
                if (
                    processor_model_path is not None  # given
                    and self.model_config.processor_model_path
                    == processor_model_path  # same with the loaded one
                    and self.model is not None  # already loaded
                    and self.processor is not None  # already loaded
                ):
                    return
            model_descriptor = causal_lm_hf_model_id

        if causal_lm_model_path is not None:
            if self.model_config.causal_lm_model_path == causal_lm_model_path:
                if (
                    processor_hf_model_id is not None  # given
                    and self.model_config.processor_hf_model_id
                    == processor_hf_model_id  # same with the loaded one
                    and self.model is not None  # already loaded
                    and self.processor is not None  # already loaded
                ):
                    return
                if (
                    processor_model_path is not None  # given
                    and self.model_config.processor_model_path
                    == processor_model_path  # same with the loaded one
                    and self.model is not None  # already loaded
                    and self.processor is not None  # already loaded
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
            torch_dtype=self.model_config.torch_dtype,
            trust_remote_code=True,
            cache_dir=florence_2_cfg.CACHE_DIR,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(
            processor_descriptor,
            trust_remote_code=True,
            cache_dir=florence_2_cfg.CACHE_DIR,
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

    def __call__(
        self,
        config: Florence2Input,
    ) -> Union[
        Caption, BoundingBoxes, Polygons, OpenVocabularyDetection, OCR, QuadBoxes, None
    ]:
        """
        Predict the output for the given input configuration.

        Parameters
        ----------
        config : Florence2Input
            Florence2Input object containing input configuration.

        Returns
        -------
        Union[Caption, BoundingBoxes, Polygons, OpenVocabularyDetection, OCR,
        QuadBoxes, None]
            The output of the prediction.
        """

        if not self.check_model_ready():
            raise ValueError("Model not loaded.")

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
            task=config.task_prompt.value,
            image_size=(config.image.width, config.image.height),
        )

        logger.info(f"Parsed answer: {parsed_answer}")

        parsed_result = parse_output(data=parsed_answer, task_type=config.task_prompt)

        return parsed_result
