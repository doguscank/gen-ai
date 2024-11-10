from typing import Optional, Tuple

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from gen_ai.configs import sam2_cfg
from gen_ai.logger import logger
from gen_ai.segmentation.segment_anything_2.segment_anything_2_input_config import (
    SegmentAnything2InputConfig,
)
from gen_ai.segmentation.segment_anything_2.segment_anything_2_model_config import (
    SegmentAnything2ModelConfig,
)
from gen_ai.segmentation.segment_anything_2.segment_anything_2_outputs import Mask


class SegmentAnything2:
    def __init__(
        self,
        *,
        config: Optional[SegmentAnything2ModelConfig] = None,
    ) -> None:
        """
        Initialize the SegmentAnything2 with the given configuration.

        Parameters
        ----------
        config : SegmentAnything2ModelConfig, optional
            SegmentAnything2ModelConfig object containing model configuration.
        """

        self.model_config = config
        self.image_predictor = None

        if self.model_config is not None:
            logger.info(
                f"Loading SegmentAnything2 model with config: {self.model_config}"
            )

            self._load_pipeline(
                hf_model_id=self.model_config.hf_model_id,
                model_path=self.model_config.model_path,
                device=self.model_config.device,
            )
        else:
            logger.info("No SegmentAnything2 model configuration provided.")

    def _check_model_ready(self) -> bool:
        """
        Check if the model is ready.

        Returns
        -------
        bool
            Whether the model is ready.
        """

        return self.image_predictor is not None

    def _load_pipeline(
        self,
        hf_model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Load the pipeline with the given configuration.

        Parameters
        ----------
        hf_model_id : str, optional
            The identifier of the model to use.
        model_path : Path, optional
            The path to the model.
        device : str, optional
            The device to run the model on. Defaults to "cuda".
        """

        if hf_model_id is None and model_path is None:
            raise ValueError("No model provided.")

        model_descriptor = None

        if hf_model_id is not None:
            if self.model_config.hf_model_id == hf_model_id:
                if self.image_predictor is not None:
                    return

            model_descriptor = hf_model_id

        if model_path is not None:
            if self.model_config.model_path == model_path:
                if self.image_predictor is not None:
                    return

            model_descriptor = model_path

        if device is None:
            device = self.model_config.device

        self.image_predictor = SAM2ImagePredictor.from_pretrained(
            model_descriptor, cache_dir=sam2_cfg.CACHE_DIR
        )

    def update_pipeline(
        self,
        model_config: SegmentAnything2ModelConfig,
    ) -> None:
        """
        Update the pipeline with the given configuration.

        Parameters
        ----------
        model_config : SegmentAnything2ModelConfig
            The configuration to update the pipeline with.
        """

        self._load_pipeline(
            hf_model_id=self.model_config.hf_model_id,
            model_path=self.model_config.model_path,
            device=self.model_config.device,
        )

        self.model_config = model_config

    def _process_outputs(
        self, masks: np.ndarray, ious: np.ndarray, low_res_masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the outputs.

        Parameters
        ----------
        masks : np.ndarray
            The masks.
        ious : np.ndarray
            The IOUs.
        low_res_masks : np.ndarray
            The low resolution masks.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The processed outputs.
        """

        if len(masks.shape) == 3:
            masks = masks[None, :, :, :]
            ious = ious[None, :]
            low_res_masks = low_res_masks[None, :, :, :]

        return masks, ious, low_res_masks

    def predict(
        self,
        config: SegmentAnything2InputConfig,
    ) -> Mask:
        """
        Predict the segmentation masks.

        Parameters
        ----------
        config : SegmentAnything2InputConfig
            The input configuration.

        Returns
        -------
        Mask
            The result mask.
        """

        if not self._check_model_ready():
            raise ValueError("Model is not ready.")

        with torch.inference_mode(), torch.autocast(
            self.model_config.device, dtype=self.model_config.torch_dtype
        ):
            self.image_predictor.set_image(config.image)

            all_masks, all_scores, all_lowres_masks = self.image_predictor.predict(
                point_coords=config.point_coords,
                point_labels=config.point_labels,
                box=config.bounding_box,
                mask_input=config.mask_input,
                multimask_output=config.multimask_output,
                normalize_coords=config.normalize_coords,
                return_logits=False,
            )

        all_masks, all_scores, all_lowres_masks = self._process_outputs(
            all_masks, all_scores, all_lowres_masks
        )

        binary_mask = np.zeros(
            (config.image.height, config.image.width), dtype=np.uint8
        )

        for masks, scores, lowres_masks in zip(all_masks, all_scores, all_lowres_masks):
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            lowres_masks = lowres_masks[sorted_ind]

            if config.refine_mask:
                mask_input = lowres_masks[np.argmax(scores), :, :]

                new_input_config = config.model_copy(deep=True)
                new_input_config.mask_input = mask_input[None, :, :]
                new_input_config.multimask_output = False
                new_input_config.refine_mask = False

                refined_output = self.predict(new_input_config)

                binary_mask = np.maximum(binary_mask, refined_output.mask)
            else:
                last_mask = masks[np.argmax(scores)]
                binary_mask = np.maximum(binary_mask, last_mask * 255)

        return Mask(
            mask=binary_mask,
        )
