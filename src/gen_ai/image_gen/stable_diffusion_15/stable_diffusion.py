from pathlib import Path
from typing import Dict, List, Optional, Union, overload

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from gen_ai.base.model import Model
from gen_ai.configs import stable_diffusion_15 as sd_config
from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes
from gen_ai.image_gen.clip.prompt_weighting import process_input_config
from gen_ai.image_gen.lora.lora_manager import LoraManager, LoraModel
from gen_ai.image_gen.stable_diffusion_15.input import StableDiffusionInput
from gen_ai.image_gen.stable_diffusion_15.model_config import StableDiffusionModelConfig
from gen_ai.image_gen.utils.inpainting_utils import (
    postprocess_outputs,
    preprocess_inputs,
)
from gen_ai.image_gen.utils.scheduler_utils import get_scheduler
from gen_ai.logger import logger
from gen_ai.utils import check_if_hf_cache_exists, pathify_strings
from gen_ai.utils.file_ops import save_images

_PIPELINE_CLS_MAP: Dict[ImageGenTaskTypes, DiffusionPipeline] = {
    ImageGenTaskTypes.TEXT2IMG: StableDiffusionPipeline,
    ImageGenTaskTypes.IMG2IMG: StableDiffusionImg2ImgPipeline,
    ImageGenTaskTypes.INPAINTING: StableDiffusionInpaintPipeline,
}

_PIPELINE_MODEL_MAP: Dict[ImageGenTaskTypes, str] = {
    ImageGenTaskTypes.TEXT2IMG: sd_config.TEXT2IMG_MODEL_ID,
    ImageGenTaskTypes.IMG2IMG: sd_config.IMG2IMG_MODEL_ID,
    ImageGenTaskTypes.INPAINTING: sd_config.INPAINTING_MODEL_ID,
}

_PipelineType = Union[
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
]


class StableDiffusion(Model):
    """
    The Stable Diffusion model for image generation.
    """

    def __init__(
        self,
        *,
        config: Optional[StableDiffusionModelConfig] = None,
    ) -> None:
        """
        Initialize the Stable Diffusion model for image generation.

        Parameters
        ----------
        config : Optional[StableDiffusionModelConfig], optional
            The configuration for the Stable Diffusion model, by default None
        """

        self._pipe: Optional[_PipelineType] = None
        self._model_config = config
        self._lora_manager = LoraManager(
            lora_dir=config.lora_dir,
            auto_register=True,
        )

        if self._model_config is not None:
            if (
                self._model_config.hf_model_id is not None
                or self._model_config.model_path is not None
            ):
                logger.info(
                    f"Loading Stable Diffusion model with config: {self._model_config}"
                )

                self._load_pipeline(
                    hf_model_id=self._model_config.hf_model_id,
                    model_path=self._model_config.model_path,
                    device=self._model_config.device,
                )

    @property
    def tokenizer(self) -> CLIPTokenizer:
        """
        Get the tokenizer for the model.

        Returns
        -------
        CLIPTokenizer
            The tokenizer for the model.
        """

        self._check_model_ready()

        return self._pipe.tokenizer

    @property
    def text_encoder(self) -> CLIPTextModel:
        """
        Get the text encoder for the model.

        Returns
        -------
        CLIPTextModel
            The text encoder for the model.
        """

        self._check_model_ready()

        return self._pipe.text_encoder

    @property
    def device(self) -> str:
        """
        Get the device the model is running on.

        Returns
        -------
        str
            The device the model is running on.
        """

        self._check_model_ready()

        return self._pipe.device

    @property
    def model_config(self) -> StableDiffusionModelConfig:
        """
        Get the model configuration.

        Returns
        -------
        StableDiffusionModelConfig
            The model configuration.
        """

        return self._model_config

    @property
    def pipe(
        self,
    ) -> Optional[_PipelineType]:
        """
        Get the pipeline.

        Returns
        -------
        Optional[_PipelineType]
            The pipeline.
        """

        return self._pipe

    def _check_model_ready(self) -> None:
        """
        Check if the model is ready.

        Raises
        ------
        ValueError
            If the model is not loaded.
        """

        if self._pipe is None:
            raise ValueError("Model not loaded.")

    def _load_pipeline(
        self,
        hf_model_id: Optional[str] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Load the Stable Diffusion pipeline.

        Parameters
        ----------
        hf_model_id : str, optional
            The HuggingFace model ID to use.
        model_path : Path, optional
            The path to the model.
        device : str, optional
            The device to run the model on.

        Returns
        -------
        None
        """

        if hf_model_id is None and model_path is None:
            raise ValueError("Either hf_model_id or model_path must be provided.")

        if hf_model_id is not None and model_path is not None:
            raise ValueError(
                "Only one of hf_model_id or model_path should be provided."
            )

        model_descriptor = None
        is_finetuned = False

        if hf_model_id is not None:
            if self._model_config.hf_model_id == hf_model_id and self._pipe is not None:
                return

            model_descriptor = hf_model_id

        if model_path is not None:
            if self._model_config.model_path == model_path and self._pipe is not None:
                return

            if "diffusers_cache" not in model_path.parts:
                is_finetuned = True
            model_descriptor = str(model_path)

        if device is None:
            device = self._model_config.device

        pipeline_cls = _PIPELINE_CLS_MAP[self._model_config.task_type]

        if is_finetuned:
            self._pipe = pipeline_cls.from_single_file(
                model_descriptor,
                cache_dir=sd_config.CACHE_DIR,
                local_files_only=True,
            ).to(device)
        else:
            self._pipe = pipeline_cls.from_pretrained(
                model_descriptor,
                torch_dtype=torch.float16,
                cache_dir=sd_config.CACHE_DIR,
                local_files_only=check_if_hf_cache_exists(
                    cache_dir=sd_config.CACHE_DIR,
                    model_id=model_descriptor,
                ),
            ).to(device)

        if not self._model_config.check_nsfw:
            self._pipe.safety_checker = None

        # self._pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)

        self._model_config.hf_model_id = hf_model_id
        self._model_config.model_path = model_path
        self._model_config.device = device

    def update_pipeline(self, model_config: StableDiffusionModelConfig) -> None:
        """
        Update the pipeline with a new configuration.

        Parameters
        ----------
        model_config : StableDiffusionModelConfig
            The new configuration.

        Returns
        -------
        None
        """

        self._load_pipeline(
            hf_model_id=model_config.hf_model_id,
            model_path=model_config.model_path,
            device=model_config.device,
        )

        self._model_config = model_config

    def _load_model_hard_set(self) -> None:
        """
        Load the model with the hard set configuration.

        Returns
        -------
        None
        """

        if self._model_config.hf_model_id is not None:
            self._load_pipeline(hf_model_id=self._model_config.hf_model_id)
        elif self._model_config.model_path is not None:
            self._load_pipeline(model_path=self._model_config.model_path)
        else:
            self._load_pipeline(
                hf_model_id=_PIPELINE_MODEL_MAP[self._model_config.task_type],
            )

    @pathify_strings
    def _generate_images_text2img(
        self, config: StableDiffusionInput, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(config.num_batches):
            pipeline_images = self._pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self._model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def _generate_images_img2img(
        self, config: StableDiffusionInput, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(config.num_batches):
            pipeline_images = self._pipe(
                prompt=config.prompt,
                image=config.image,
                strength=config.denoising_strength,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self._model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def _generate_images_inpainting(
        self, config: StableDiffusionInput, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(config.num_batches):
            image, mask_image = preprocess_inputs(
                image=config.image,
                mask=config.mask_image,
                pre_process_type=config.preprocess_type,
                output_width=config.width,
                output_height=config.height,
            )

            pipeline_images = self._pipe(
                prompt=config.prompt,
                image=image,
                mask_image=mask_image,
                masked_image_latents=config.masked_image_latents,
                latents=config.latents,
                height=config.height,
                width=config.width,
                padding_mask_crop=config.padding_mask_crop,
                strength=config.denoising_strength,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self._model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            pipeline_images = [
                postprocess_outputs(
                    image=config.image,
                    mask=config.mask_image,
                    inpainted_image=inpainted_image,
                    pre_process_type=config.preprocess_type,
                    post_process_type=config.postprocess_type,
                    blending_type=config.blending_type,
                )
                for inpainted_image in pipeline_images
            ]

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def __call__(
        self,
        config: StableDiffusionInput,
        output_dir: Optional[Path] = None,
        use_prompt_weighting: bool = True,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.
        output_dir : Optional[Path], optional
            The output directory to save the images, by default None.
        use_prompt_weighting : bool, optional
            Whether to use prompt weighting, by default True.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        self._check_model_ready()

        self._set_lora_adapters_from_prompt(config.prompt)

        # preprocess inputs if needed
        if use_prompt_weighting:
            config = process_input_config(
                input_config=config,
                tokenizer=self.tokenizer,
                model=self.text_encoder,
                update_mean=True,
                device=self.device,
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        if config.scheduler_type is not None:
            self._pipe.scheduler = get_scheduler(scheduler_type=config.scheduler_type)

        if self._model_config.task_type == ImageGenTaskTypes.TEXT2IMG:
            images = self._generate_images_text2img(
                config=config, output_dir=output_dir
            )
        elif self._model_config.task_type == ImageGenTaskTypes.IMG2IMG:
            images = self._generate_images_img2img(config=config, output_dir=output_dir)
        elif self._model_config.task_type == ImageGenTaskTypes.INPAINTING:
            images = self._generate_images_inpainting(
                config=config, output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported task type: {self._model_config.task_type}")

        return images

    @overload
    def load_textual_inversion(
        self, *, file_path: Union[Path, List[Path]], token: Union[str, List[str]]
    ) -> None:
        ...

    @overload
    def load_textual_inversion(self, *, hf_model_id: Union[str, List[str]]) -> None:
        ...

    @pathify_strings
    def load_textual_inversion(
        self,
        *,
        hf_model_id: Optional[Union[str, List[str]]] = None,
        file_path: Optional[Union[Path, List[Path]]] = None,
        token: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Load the textual inversion model.

        Parameters
        ----------
        hf_model_id : Optional[str], optional
            The HuggingFace model ID for the textual inversion model.
        file_path : Optional[Path], optional
            The path to the textual inversion model.
        token : Optional[str], optional
            The token to load.

        Returns
        -------
        None
        """

        self._check_model_ready()

        if file_path is not None:
            if token is None:
                raise ValueError("tokens must be provided when file_paths is provided.")
            if isinstance(file_path, list) and isinstance(token, list):
                if len(file_path) != len(token):
                    raise ValueError(
                        "Number of file paths and tokens must be the same."
                    )
            elif not (isinstance(file_path, Path) and isinstance(token, str)):
                raise ValueError(
                    "file_path and token must be either both single values or both lists."
                )
            self._pipe.load_textual_inversion(file_path, token=token)
        elif hf_model_id is not None:
            self._pipe.load_textual_inversion(
                hf_model_id,
                cache_dir=sd_config.CACHE_DIR,
                only_local_files=check_if_hf_cache_exists(
                    cache_dir=sd_config.CACHE_DIR, model_id=hf_model_id
                ),
            )
        else:
            raise ValueError("Either file_path or hf_model_id must be provided.")

    def unload_textual_inversion(self, *, tokens: Union[str, List[str]]) -> None:
        """
        Unload the textual inversion model by tokens.

        Parameters
        ----------
        tokens : Union[str, List[str]]
            The tokens to unload.

        Returns
        -------
        None
        """

        self._check_model_ready()

        self._pipe.unload_textual_inversion(tokens=tokens)

    def unload_all_textual_inversion(self) -> None:
        """
        Unload all textual inversion models.

        Returns
        -------
        None
        """

        self._check_model_ready()

        self._pipe.unload_textual_inversion()

    def add_lora(
        self, lora_path: Path, trigger_words: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Add a LoRA model.

        Parameters
        ----------
        lora_path : Path
            The path to the LoRA model.
        trigger_words : Optional[Union[str, List[str]]], optional
            The trigger words for the LoRA model, by default None.

        Returns
        -------
        None
        """

        self._lora_manager.register_lora_model(lora_path, trigger_words)

    def merge_lora_manager(
        self, manager: Union["LoraManager", List["LoraManager"]]
    ) -> None:
        """
        Merge a LoRA manager.

        Parameters
        ----------
        manager : Union["LoraManager", List["LoraManager"]]
            The LoRA manager to merge.

        Returns
        -------
        None
        """

        self._lora_manager.merge(manager)

    def load_lora(
        self,
        lora: Union[
            str, List[str], LoraModel, List[LoraModel], List[Union[str, LoraModel]]
        ],
    ) -> None:
        """
        Load a LoRA model.

        Parameters
        ----------
        lora : Union[str, List[str], LoraModel, List[LoraModel]]
            The LoRA model to load.

        Returns
        -------
        None
        """

        if isinstance(lora, list):
            if len(lora) == 0:
                return

            if all(isinstance(model, (str, LoraModel)) for model in lora):
                for model in lora:
                    self.load_lora(model)
                return

        if isinstance(lora, str):
            lora_model = self._lora_manager.get_model_by_name(lora)

            if lora_model is None:
                logger.warning(f"LoRA model with name {lora} not found.")
                return

            self.load_lora(lora_model)
            return

        if isinstance(lora, LoraModel):
            if lora.is_loaded:
                return

            self._pipe.load_lora_weights(
                lora.path,
                weight_name=lora.path.name,
                adapter_name=lora.name,
            )
            self._pipe.fuse_lora()
            lora.set_loaded()
            return

        raise ValueError(f"Unsupported LoRA model type: {type(lora)}")

    def unload_loras(self) -> None:
        """
        Unload all LoRA models.

        Returns
        -------
        None
        """

        self._pipe.unfuse_lora()
        self._pipe.unload_lora_weights()

        for lora_model in self._lora_manager.models:
            lora_model.set_unloaded()

    def enable_lora(self) -> None:
        """
        Enable LoRA.

        Returns
        -------
        None
        """

        self._pipe.enable_lora()

    def disable_lora(self) -> None:
        """
        Disable LoRA.

        Returns
        -------
        None
        """

        self._pipe.disable_lora()

    def _set_lora_adapters_from_prompt(self, prompt: str) -> None:
        """
        Set the adapters from the prompt.

        Parameters
        ----------
        prompt : str
            The prompt to set the adapters from.

        Returns
        -------
        None
        """

        lora_models_to_activate = self._lora_manager.get_lora_models_from_prompt(prompt)

        self.unload_loras()

        if len(lora_models_to_activate) == 0:
            return

        self.load_lora(lora=[model.name for model in lora_models_to_activate])

        self._pipe.set_adapters(
            adapter_names=[model.name for model in lora_models_to_activate],
            adapter_weights=[model.scale for model in lora_models_to_activate],
        )
