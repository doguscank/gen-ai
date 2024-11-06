import time


import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from gen_ai.configs import florence_2 as florence_cfg
from gen_ai.configs import segment_anything_2 as sam2_cfg
from gen_ai.img_utils import load_image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from typing import List, Union
from gen_ai.logger import logger

t0 = time.time()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    florence_cfg.FLORENCE2_CAUSAL_LM_MODEL_ID,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    cache_dir=florence_cfg.CACHE_DIR,
).to(device)
processor = AutoProcessor.from_pretrained(
    florence_cfg.FLORENCE2_PROCESSOR_MODEL_ID,
    trust_remote_code=True,
    cache_dir=florence_cfg.CACHE_DIR,
)


def create_text_prompt(texts: List[str]) -> str:
    """
    Create a text prompt from the given list of texts.

    Parameters
    ----------
    texts : List[str]
        The list of texts to create the prompt from.

    Returns
    -------
    str
        The created text prompt.
    """

    if len(texts) == 1:
        text_prompt = texts[0]
    else:
        text_prompt = " <and> ".join(texts)
    return text_prompt


def postprocess_mask(
    mask: Union[Image.Image, np.ndarray], kernel_size: int = 5
) -> Image.Image:
    """
    Post-process the mask.

    Parameters
    ----------
    mask : Union[Image.Image, np.ndarray]
        The mask to post-process.
    kernel_size : int, optional
        The kernel size to use for post-processing. Defaults to 5.

    Returns
    -------
    Union[Image.Image, np.ndarray]
        The post-processed mask.
    """

    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # apply gaussian blur
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # threshold the mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    return Image.fromarray(mask)


task_prompt = "<OPEN_VOCABULARY_DETECTION>"
text_prompt = create_text_prompt(["clothing"])

prompt = task_prompt + text_prompt

image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(
    device, torch_dtype
)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    early_stopping=False,
    do_sample=False,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text, task=task_prompt, image_size=(image.width, image.height)
)

print(parsed_answer)

# Integrate SAM2
bbox = np.array(parsed_answer[task_prompt]["bboxes"]).astype(int)

predictor = SAM2ImagePredictor.from_pretrained(
    model_id=sam2_cfg.SAM2_MODEL_ID, cache_dir=sam2_cfg.CACHE_DIR
)

image = load_image(image_path)
cv2_img = cv2.imread(image_path)

print("bbox:", bbox)
print("bbox.shape:", bbox.shape)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    all_masks, all_scores, all_logits = predictor.predict(
        box=bbox,
        point_coords=None,
        multimask_output=True,
    )
    
    logger.info(f"all_masks: {all_masks}")

    if len(all_masks.shape) == 3:
        all_masks = all_masks[None, :, :, :]
        all_scores = all_scores[None, :]
        all_logits = all_logits[None, :, :, :]

    # Initialize merged_mask with the same shape and number of channels as cv2_img
    merged_mask = np.zeros(cv2_img.shape, dtype=cv2_img.dtype)
    binary_mask = np.zeros((cv2_img.shape[0], cv2_img.shape[1]), dtype=cv2_img.dtype)

    # zip the masks, scores, and logits together
    for idx in range(len(all_masks)):
        masks = all_masks[idx]
        scores = all_scores[idx]
        logits = all_logits[idx]

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        mask_input = logits[np.argmax(scores), :, :]

        # Save each mask as a separate image
        for i, mask in enumerate(masks):
            mask_3ch = cv2.merge(
                [mask * 0, mask * 255, mask * 0]
            )  # Convert mask to green 3-channel
            mask_3ch = mask_3ch.astype(
                cv2_img.dtype
            )  # Ensure mask type matches image type
            masked_img = cv2.addWeighted(cv2_img, 1, mask_3ch, 0.5, 0)
            cv2.imwrite(
                f"E:\\Scripting Workspace\\Python\\GenAI\\output_mask_{i+idx*len(all_masks)}.jpg",
                masked_img,
            )

        sub_masks, sub_scores, sub_logits = predictor.predict(
            box=bbox,
            point_coords=None,
            multimask_output=False,
            mask_input=mask_input[None, :, :],
        )

        if len(sub_masks.shape) == 3:
            sub_masks = sub_masks[None, :, :, :]
            sub_scores = sub_scores[None, :]
            sub_logits = sub_logits[None, :, :, :]

        best_sub_mask = sub_masks[np.argmax(sub_scores)]
        last_mask = best_sub_mask[0]
        last_mask_3ch = cv2.merge(
            [last_mask * 0, last_mask * 255, last_mask * 0]
        )  # Convert mask to green 3-channel
        last_mask_3ch = last_mask_3ch.astype(
            cv2_img.dtype
        )  # Ensure mask type matches image type

        merged_mask = cv2.addWeighted(merged_mask, 1, last_mask_3ch, 1, 0)
        binary_mask = np.maximum(binary_mask, last_mask * 255)

    masked_img = cv2.addWeighted(cv2_img, 1, merged_mask, 0.5, 0)
    cv2.imwrite(
        f"E:\\Scripting Workspace\\Python\\GenAI\\output_mask_final.jpg", masked_img
    )

binary_mask = postprocess_mask(binary_mask)
binary_mask.save("E:\\Scripting Workspace\\Python\\GenAI\\output_binary_mask.png")

# Draw bbox on cv2_img
for box in bbox:
    cv2.rectangle(cv2_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

# Save the resulting image
cv2.imwrite("E:\\Scripting Workspace\\Python\\GenAI\\output1.jpg", cv2_img)

t1 = time.time()
print("Time taken:", t1 - t0)
