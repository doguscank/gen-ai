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

# Features

## Stable Diffusion

### Prompt Weighting

Users can set weights of prompt parts using a syntax similar to AUTOMATIC1111's. To set weight of a prompt part, encapsulate the part within parentheses and use a colon and set weight value before closing the parenthesis. Check examples below.

```python
prompt = "An (oil painting:1.3) image depicting a (lake view at (sunset: 1.1): 1.3)"

from gen_ai.image_gen.clip.prompt_weighting import parse_prompt

parsed_prompt = parse_prompt(prompt)

>>> parsed_prompt
[
  Text Piece: 'An ', Attention Multiplier: 1.0,
  Text Piece: 'oil painting', Attention Multiplier: 1.3,
  Text Piece: ' image depicting a ', Attention Multiplier: 1.0,
  Text Piece: 'lake view at ', Attention Multiplier: 1.3,
  Text Piece: 'sunset', Attention Multiplier: 1.4300000000000002
]
```

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
    * ~~Make prompt words weighted like in AUTOMATIC1111~~
    * Extend prompt length to 75+ tokens
    * Make prompt weighting more optimized
    * Add pre-defined embedding support (saving and loading embedding with unique names)
* Add LoRA support
    * ~~Implement LoRA model manager~~
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
* Update prompt weighting
    * Update automatic LoRA loading from keyword to LoRA name with gt, lt symbols
    * Add prompt weighting support for textual inversion keywords
* Separate textual inversion from the model
* Refactor model, model config and input config
    * ~~Create base abstract classes~~
    * Create mixins
    * Create utilities for shared functions
    * ~~Rename "input config" to "input"~~
    * Add missing input/output dataclasses
        * Stable Diffusion 1.5 output
        * YOLOv11 Pose input and output
* Add auto model unloading
* Add XFormers support (enable_xformers_memory_efficient_attention, disable_xformers_memory_efficient_attention)
* Add TAESD (Tiny AutoEncoder for Stable Diffusion) support [[GitHub](https://github.com/madebyollin/taesd?tab=readme-ov-file)]
* Add Agent Scheduler (or similar) support [[GitHub](https://github.com/ArtVentureX/sd-webui-agent-scheduler?tab=readme-ov-file)]
* Add Instruct pix2pix support [[GitHub](https://github.com/timothybrooks/instruct-pix2pix)]
* Add Spandrel support [[GitHub](https://github.com/chaiNNer-org/spandrel)]
* Add LaMa support (check feature refinement) [[GitHub](https://github.com/advimman/lama)] [[GitHub](https://github.com/advimman/lama/pull/112)] [[Restormer arXiv](https://arxiv.org/pdf/2111.09881)]
* Implement/integrate txt2imghd [[GitHub](https://github.com/jquesnelle/txt2imghd)]
* G-Diffuser-Bot can be a reference for outpainting [[GitHub](https://github.com/parlance-zz/g-diffuser-bot)]
* Pixelization support [[GitHub](https://github.com/WuZongWei6/Pixelization)]
* BrushNet support [[GitHub](https://github.com/TencentARC/BrushNet)] [[ComfyUI implementation](https://github.com/kijai/ComfyUI-BrushNet-Wrapper/tree/main)]
* Add LLaMa-Mesh support [[GitHub](https://github.com/nv-tlabs/LLaMA-Mesh)]
    * Add image-to-text to create similar mesh (don't know how possible is it)
        * If successful, try to implement "room photo-to-text" pipeline and "text-to-room" pipeline
* Add Lotus support [[GitHub](https://github.com/EnVision-Research/Lotus)] [[arXiv](https://arxiv.org/abs/2409.18124)]
    * Lotus can predict depth and normal with a high accuracy but without a scale (we can't get actual value of depth but current output can be fed to ControlNet)
* Add Depth-Pro support [[GitHub](https://github.com/apple/ml-depth-pro)]
    * Fast metric-depth prediction
* Add F5-TTS support [[GitHub](https://github.com/SWivid/F5-TTS)]
* Add AirLLM support to run bigger LLMs with less VRAM [[GitHub](https://github.com/lyogavin/airllm)]
* Add One-DM (One-Shot Diffusion Mimicker for Handwritten Text Generation) support [[GitHub](https://github.com/dailenson/One-DM)]
* Add RobustSAM for degredad image segmentation [[GitHub](https://github.com/robustsam/RobustSAM)]
