# To-Do

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
* Add FreeU support [[🤗 HuggingFace](https://huggingface.co/docs/diffusers/main/en/using-diffusers/image_quality)] [[GitHub](https://github.com/ChenyangSi/FreeU)]
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
	* ~~Integrate model to the package.~~
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
* Add support for Differential Diffusion https://differential-diffusion.github.io
* Add support for Torchao [[GitHub](https://github.com/sayakpaul/diffusers-torchao)]
* Integrate ControlNet++ for SDXL [[🤗 HuggingFace](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)]
* Implement task pipelines infrastructure
* Add DWPose support
