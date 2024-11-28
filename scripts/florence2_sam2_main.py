from PIL import Image

from gen_ai.configs.defaults.multitask import florence_2 as florence_cfg
from gen_ai.configs.defaults.segmentation import segment_anything_2 as sam2_cfg
from gen_ai.constants.task_types.florence_2_task_types import Florence2TaskTypes
from gen_ai.logger import logger
from gen_ai.tasks.multitask.florence_2 import (
    Florence2,
    Florence2Input,
    Florence2ModelConfig,
    OpenVocabularyDetection,
)
from gen_ai.tasks.segmentation.segment_anything_2 import (
    Mask,
    SegmentAnything2,
    SegmentAnything2Input,
    SegmentAnything2ModelConfig,
)
from gen_ai.utils import measure_time
from gen_ai.utils.file_ops import load_image
from gen_ai.utils.torch_utils import flush

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

    image_path = "/home/doguscank/python_self_work_ws/gen-ai/inputs/input2.jpg"
    image = load_image(image_path)

    with measure_time("Florence2 Input Configuration"):
        florence2_input = Florence2Input(
            image=image,
            text_prompt="clothing",
            task_prompt=Florence2TaskTypes.OPEN_VOCABULARY_DETECTION,
        )

    logger.info(f"Predicting with Florence2 model: {florence2_input}")

    with measure_time("Florence2 Prediction"):
        florence2_output: OpenVocabularyDetection = florence2_model(florence2_input)

    logger.info(f"Florence2 output: {florence2_output}")

    del florence2_model

    flush()

    with measure_time("SegmentAnything2 Model Configuration"):
        sam2_model_cfg = SegmentAnything2ModelConfig(
            hf_model_id=sam2_cfg.SAM2_MODEL_ID,
            device="cuda",
        )

    with measure_time("SegmentAnything2 Model Initialization"):
        sam2_model = SegmentAnything2(config=sam2_model_cfg)

    with measure_time("SegmentAnything2 Input Configuration"):
        sam2_input = SegmentAnything2Input(
            image=image,
            bounding_box=florence2_output.bounding_boxes.coords_int,
            refine_mask=True,
        )

    logger.info(f"Predicting with SAM2 model: {sam2_input}")

    with measure_time("SegmentAnything2 Prediction"):
        sam2_output: Mask = sam2_model(sam2_input)

    logger.info(f"SAM2 output: {sam2_output}")

    result_mask = Image.fromarray(sam2_output.mask)
    result_mask.save(
        "/home/doguscank/python_self_work_ws/gen-ai/outputs/result_mask2.png"
    )
