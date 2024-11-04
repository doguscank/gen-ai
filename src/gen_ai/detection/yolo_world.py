from ultralytics import YOLOWorld, YOLO
import numpy as np

keypoint_mapping = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}

facial_indices = [0, 1, 2, 3, 4]

model = YOLOWorld("yolov8m-worldv2.pt")
model.set_classes(["clothing"])

pose_model = YOLO("yolo11x-pose.pt")

results = model.predict("E:\\Scripting Workspace\\Python\\GenAI\\input2.jpg")
pose_results = pose_model.predict("E:\\Scripting Workspace\\Python\\GenAI\\input2.jpg")


bbox = results[0].boxes.xyxy.cpu().numpy().astype(int)
print("bbox:", bbox)

poses = pose_results[0].keypoints.xy.cpu().numpy().astype(int)
print("poses:", poses)

facial_points = poses[:, facial_indices][0]
print("facial points:", facial_points)

point_labels = np.zeros((len(facial_points),), dtype=int)

import torch
from gen_ai.configs import segment_anything_2 as sam2_cfg
from gen_ai.img_utils import load_image
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained(
    model_id=sam2_cfg.SAM2_MODEL_ID, cache_dir=sam2_cfg.CACHE_DIR
)

image = load_image("E:\\Scripting Workspace\\Python\\GenAI\\input2.jpg")
import cv2

cv2_img = cv2.imread("E:\\Scripting Workspace\\Python\\GenAI\\input2.jpg")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        box=bbox,
        point_labels=point_labels,
        point_coords=facial_points,
        multimask_output=True,
    )

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
        mask_3ch = mask_3ch.astype(cv2_img.dtype)  # Ensure mask type matches image type
        masked_img = cv2.addWeighted(cv2_img, 1, mask_3ch, 0.5, 0)
        cv2.imwrite(
            f"E:\\Scripting Workspace\\Python\\GenAI\\output_mask_{i}.jpg", masked_img
        )

    sub_masks, _, _ = predictor.predict(
        box=bbox,
        point_labels=point_labels,
        point_coords=facial_points,
        multimask_output=False,
        mask_input=mask_input[None, :, :],
    )

    last_mask = sub_masks[0]
    last_mask_3ch = cv2.merge(
        [last_mask * 0, last_mask * 255, last_mask * 0]
    )  # Convert mask to green 3-channel
    last_mask_3ch = last_mask_3ch.astype(
        cv2_img.dtype
    )  # Ensure mask type matches image type
    masked_img = cv2.addWeighted(cv2_img, 1, last_mask_3ch, 0.5, 0)
    cv2.imwrite(
        f"E:\\Scripting Workspace\\Python\\GenAI\\output_mask_final.jpg", masked_img
    )

# Draw bbox on cv2_img
for box in bbox:
    cv2.rectangle(cv2_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

for pose in poses:
    for idx, (x, y) in enumerate(pose):
        keypoint_name = keypoint_mapping[idx]

        cv2.circle(cv2_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            cv2_img,
            keypoint_name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# Save the resulting image
cv2.imwrite("E:\\Scripting Workspace\\Python\\GenAI\\output1.jpg", cv2_img)
