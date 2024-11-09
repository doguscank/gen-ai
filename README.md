# Gen AI Portfolio Repository

This repository will be used as my Gen AI portfolio.

# Integrated Models

* Segment Anything 2
* Florence2
* Stable Diffusion 1.5

# Florence2

# Segment Anything 2

# Stable Diffusion 1.5

## Inpainting

### Examples

| Prompt | Input Image | Mask | Result |
|--------|-------------|------|--------|
| RAW photo of a man wearing a red and white fur coat | ![Inpainting Input](assets/inpainting_input.jpg) | ![Inpainting Mask](assets/inpainting_mask.png) | ![Inpainting Output](assets/inpainting_output.png) |

# Installation

Install package using `pip install -e .`. Then install required repositories in the `repositories` folder. Follow [RESOSITORIES.md](repositories/REPOSITORIES.md) file for details.

# TO-DO

* ~~Implement Stable Diffusion 1.5 pipeline~~
    * ~~Text2Img~~
    * ~~Img2Img~~
    * ~~Inpainting~~
* ~~Add local path loading support~~
    * ~~Make fine-tuned models work~~
* Add textual inversion
    * ~~Add loading textual inversion~~
    * Add textual inversion training (https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb#scrollTo=E3UREGd7EkLh)
* Add prompt weighting
    * Make prompt words weighted like in AUTOMATIC1111
* Add LoRA support
    * Add changeable LoRA weights
* Add ControlNet support
    * Add ControlNet weights
    * Add ControlNet start/end iteration controls
* Add Groq API integration
* Add LoRA usage prediction using LLMs
    * llama 3.2 1B/3B must be tried first since those models can also be run on local.
* Implement UI using Gradio
