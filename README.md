# Gen AI Portfolio Repository

A comprehensive collection of state-of-the-art AI models integrated into a single, cohesive library.

# Integrated Models

The library incorporates the following advanced AI models:

* Florence2
  * A versatile multimodal vision model supporting various computer vision tasks. Mainly used for zero-shot object detection.
* Segment Anything 2
  * An advanced segmentation model that precisely identifies and delineates objects within images. Used with Florence2 for zero-shot object masking.
* Stable Diffusion 1.5
  * A generative AI model specializing in high-quality image synthesis and manipulation.
* YOLOWorld
  * A real-time object detection model with configurable object categories.
* YOLOv11 Pose
  * A state-of-the-art human pose estimation model.

## Florence2

Florence2 represents a comprehensive multimodal vision model that excels in various computer vision tasks including image understanding, object detection, and visual captioning.

## Segment Anything 2

The Segment Anything 2 integration provides robust image segmentation capabilities. The model can precisely identify and isolate multiple objects within images, making it ideal for applications requiring detailed scene understanding and object isolation.

## Stable Diffusion 1.5

Stable Diffusion 1.5 provides advanced image generation and manipulation capabilities through three primary functionalities:

### Text-to-Image Generation

The text-to-image system generates high-quality images from textual descriptions. Users can control various generation parameters to achieve desired artistic styles and content characteristics.

### Image-to-Image Transformation

The image-to-image transformation system enables guided image modification based on textual prompts. This allows for sophisticated style transfers, content alterations, and image enhancements while preserving structural integrity.

### Inpainting

The inpainting system provides precise control over image editing by allowing selective regeneration of masked areas. This enables seamless object removal, addition, or modification within existing images.

## YOLOWorld

YOLOWorld implements real-time object detection with dynamic category support. The system can identify and localize multiple object categories within images, with support for custom category sets.

## YOLOv11 Pose

The YOLOv11 Pose integration enables precise human pose estimation, providing detailed skeletal keypoint detection and pose analysis.

# Examples

## Stable Diffusion 1.5 Inpainting

| Prompt | Input Image | Mask | Result |
|--------|-------------|------|--------|
| RAW photo of a man wearing a red and white fur coat | ![Inpainting Input](assets/inpainting_input.jpg) | ![Inpainting Mask](assets/inpainting_mask.png) | ![Inpainting Output](assets/inpainting_output.png) |

# Installation

1. Install the package using `pip install -e .`
2. Install additional required repositories from the `repositories` folder
3. Refer to [RESOSITORIES.md](repositories/REPOSITORIES.md) for detailed setup instructions

* ~~Implement Stable Diffusion 1.5 pipeline~~
    * ~~Text2Img~~
    * ~~Img2Img~~
    * ~~Inpainting~~
* ~~Add local path loading support~~
    * ~~Make fine-tuned models work~~
* ~~Implement Florence 2 pipeline~~
* ~~Implement Segment Anything 2 pipeline~~
* Add textual inversion
    * ~~Add loading textual inversion~~
    * Add textual inversion training [[Source](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb#scrollTo=E3UREGd7EkLh)]
* Add prompt weighting
    * Make prompt words weighted like in AUTOMATIC1111
* Add LoRA support
    * Add changeable LoRA weights
* Add ControlNet support
    * Add ControlNet weights
    * Add ControlNet start/end iteration controls
* Add FreeU support [[HuggingFace](https://huggingface.co/docs/diffusers/main/en/using-diffusers/image_quality)] [[GitHub](https://github.com/ChenyangSi/FreeU)]
* Add Groq API integration
* Add LoRA usage prediction using LLMs
    * llama 3.2 1B/3B must be tried first since those models can also be run on local.
* Implement UI using Gradio
* Add API support
