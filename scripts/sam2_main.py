from gen_ai.segmentation.segment_anything_2 import SegmentAnything2
from gen_ai.segmentation.segment_anything_2_input_config import (
    SegmentAnything2InputConfig,
)
from gen_ai.segmentation.segment_anything_2_model_config import (
    SegmentAnything2ModelConfig,
)
from gen_ai.segmentation.segment_anything_2_outputs import SegmentAnything2Output
from gen_ai.configs import segment_anything_2 as sam2_cfg
from gen_ai.img_utils import load_image
from gen_ai.multitask.florence_2 import Florence2
from gen_ai.multitask.florence_2_input_config import Florence2InputConfig
from gen_ai.multitask.florence_2_model_config import Florence2ModelConfig
from gen_ai.multitask.florence_2_outputs import OpenVocabularyDetection
from gen_ai.configs import florence_2 as florence_cfg
from gen_ai.constants.florence_2_task_types import Florence2TaskTypes
from PIL import Image
from gen_ai.logger import logger
import numpy as np
from gen_ai.utils import measure_time
from gen_ai.torch_utils import free_gpu_cache
from gen_ai.image_gen.stable_diffusion import StableDiffusion
from gen_ai.image_gen.stable_diffusion_input_config import StableDiffusionInputConfig
from gen_ai.image_gen.stable_diffusion_model_config import StableDiffusionModelConfig
from gen_ai.configs import stable_diffusion as sd_cfg

if __name__ == "__main__":
    with measure_time("Florence2 Model Configuration"):
        florence2_model_cfg = Florence2ModelConfig(
            causal_lm_hf_model_id=florence_cfg.FLORENCE2_CAUSAL_LM_MODEL_ID,
            processor_hf_model_id=florence_cfg.FLORENCE2_PROCESSOR_MODEL_ID,
            device="cuda",
            task_type=Florence2TaskTypes.OPEN_VOCABULARY_DETECTION,
        )

    with measure_time("Florence2 Model Initialization"):
        florence2_model = Florence2(config=florence2_model_cfg)

    image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input5.jpg"
    image = load_image(image_path)

    with measure_time("Florence2 Input Configuration"):
        florence2_input = Florence2InputConfig(
            image=image,
            text_prompt="clothing",
            task_prompt=Florence2TaskTypes.OPEN_VOCABULARY_DETECTION,
        )

    logger.info(f"Predicting with Florence2 model: {florence2_input}")

    with measure_time("Florence2 Prediction"):
        florence2_output: OpenVocabularyDetection = florence2_model.predict(
            florence2_input
        )

    logger.info(f"Florence2 output: {florence2_output}")

    del florence2_model
    free_gpu_cache()

    with measure_time("SegmentAnything2 Model Configuration"):
        sam2_model_cfg = SegmentAnything2ModelConfig(
            hf_model_id=sam2_cfg.SAM2_MODEL_ID,
            device="cuda",
        )

    with measure_time("SegmentAnything2 Model Initialization"):
        sam2_model = SegmentAnything2(config=sam2_model_cfg)

    with measure_time("SegmentAnything2 Input Configuration"):
        sam2_input = SegmentAnything2InputConfig(
            image=image,
            bounding_box=florence2_output.bounding_boxes.coords_int,
            refine_mask=True,
        )

    logger.info(f"Predicting with SAM2 model: {sam2_input}")

    with measure_time("SegmentAnything2 Prediction"):
        sam2_output: SegmentAnything2Output = sam2_model.predict(sam2_input)

    logger.info(f"SAM2 output: {sam2_output}")

    result_mask = Image.fromarray(sam2_output.mask)
    result_mask.save("E:\\Scripting Workspace\\Python\\GenAI\\result_mask5.png")

    del sam2_model
    free_gpu_cache()

    with measure_time("StableDiffusion Model Configuration"):
        sd_model_cfg = StableDiffusionModelConfig(
            hf_model_id=sd_cfg.INPAINTING_MODEL_ID,
            device="cuda",
        )