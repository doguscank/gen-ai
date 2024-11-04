import torch
from gen_ai.configs import segment_anything_2 as sam2_cfg
from gen_ai.img_utils import load_image
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained(
    model_id=sam2_cfg.SAM2_MODEL_ID, cache_dir=sam2_cfg.CACHE_DIR
)

image = load_image("E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict()

    for i, mask in enumerate(masks):
        mask.save(f"mask_{i}.png")
