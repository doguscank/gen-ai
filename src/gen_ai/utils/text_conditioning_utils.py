from typing import List, Optional, Tuple, Union

import torch


def fix_conditioning_inputs(
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[Union[str, List[str]]],
    Optional[Union[str, List[str]]],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Validate and fix conditioning inputs for a model.

    Parameters
    ----------
    prompt : Optional[Union[str, List[str]]]
        The prompt(s) to condition the model on.
    negative_prompt : Optional[Union[str, List[str]]]
        The negative prompt(s) to condition the model on.
    prompt_embeds : Optional[torch.Tensor]
        The prompt embeddings to condition the model on.
    negative_prompt_embeds : Optional[torch.Tensor]
        The negative prompt embeddings to condition the model on.

    Returns
    -------
    Tuple[Optional[Union[str, List[str]]], Optional[Union[str, List[str]]], Optional[torch.Tensor], Optional[torch.Tensor]]
    """

    if not isinstance(prompt, (str, list)) and prompt is not None:
        raise ValueError("prompt should be a string or a list of strings.")
    if not isinstance(negative_prompt, (str, list)) and negative_prompt is not None:
        raise ValueError("negative_prompt should be a string or a list of strings.")
    if not isinstance(prompt_embeds, torch.Tensor) and prompt_embeds is not None:
        raise ValueError("prompt_embeds should be a torch.Tensor.")
    if (
        not isinstance(negative_prompt_embeds, torch.Tensor)
        and negative_prompt_embeds is not None
    ):
        raise ValueError("negative_prompt_embeds should be a torch.Tensor.")

    if isinstance(prompt, list) and isinstance(negative_prompt, list):
        if len(prompt) == 1 and len(negative_prompt) > 1:
            prompt = prompt * len(negative_prompt)
        elif len(prompt) > 1 and len(negative_prompt) == 1:
            negative_prompt = negative_prompt * len(prompt)
        elif len(prompt) != len(negative_prompt):
            raise ValueError("Prompt and negative_prompt should have the same length.")
    if isinstance(prompt, list) and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * len(prompt)
    if isinstance(prompt, str) and isinstance(negative_prompt, list):
        prompt = [prompt] * len(negative_prompt)

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "prompt_embeds and negative_prompt_embeds should have the same shape."
                f" Got `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}"
            )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError("prompt and prompt_embeds should not be provided together.")
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            "negative_prompt and negative_prompt_embeds should not be provided together."
        )

    return prompt, negative_prompt, prompt_embeds, negative_prompt_embeds


def fix_dual_conditioning_inputs(
    prompt: Optional[Union[str, List[str]]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[Union[str, List[str]]],
    Optional[Union[str, List[str]]],
    Optional[Union[str, List[str]]],
    Optional[Union[str, List[str]]],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Validate and fix dual conditioning inputs for a model.

    Parameters
    ----------
    prompt : Optional[Union[str, List[str]]]
        The prompt(s) to condition the model on.
    prompt_2 : Optional[Union[str, List[str]]]
        The second prompt(s) to condition the model on.
    negative_prompt : Optional[Union[str, List[str]]]
        The negative prompt(s) to condition the model on.
    negative_prompt_2 : Optional[Union[str, List[str]]]
        The second negative prompt(s) to condition the model on.
    prompt_embeds : Optional[torch.Tensor]
        The prompt embeddings to condition the model on.
    negative_prompt_embeds : Optional[torch.Tensor]
        The negative prompt embeddings to condition the model on.

    Returns
    -------
    Tuple[Optional[Union[str, List[str]]], Optional[Union[str, List[str]]], Optional[Union[str, List[str]], Optional[Union[str, List[str]], Optional[torch.Tensor], Optional[torch.Tensor]]
    """

    if not isinstance(prompt, (str, list)) and prompt is not None:
        raise ValueError("prompt should be a string or a list of strings.")
    if not isinstance(prompt_2, (str, list)) and prompt_2 is not None:
        raise ValueError("prompt_2 should be a string or a list of strings.")
    if not isinstance(negative_prompt, (str, list)) and negative_prompt is not None:
        raise ValueError("negative_prompt should be a string or a list of strings.")
    if not isinstance(negative_prompt_2, (str, list)) and negative_prompt_2 is not None:
        raise ValueError("negative_prompt_2 should be a string or a list of strings.")
    if not isinstance(prompt_embeds, torch.Tensor) and prompt_embeds is not None:
        raise ValueError("prompt_embeds should be a torch.Tensor.")
    if (
        not isinstance(negative_prompt_embeds, torch.Tensor)
        and negative_prompt_embeds is not None
    ):
        raise ValueError("negative_prompt_embeds should be a torch.Tensor.")

    if isinstance(prompt, list) and isinstance(negative_prompt, list):
        if len(prompt) == 1 and len(negative_prompt) > 1:
            prompt = prompt * len(negative_prompt)
        elif len(prompt) > 1 and len(negative_prompt) == 1:
            negative_prompt = negative_prompt * len(prompt)
        elif len(prompt) != len(negative_prompt):
            raise ValueError("Prompt and negative_prompt should have the same length.")

    if isinstance(prompt_2, list) and isinstance(negative_prompt_2, list):
        if len(prompt_2) == 1 and len(negative_prompt_2) > 1:
            prompt_2 = prompt_2 * len(negative_prompt_2)
        elif len(prompt_2) > 1 and len(negative_prompt_2) == 1:
            negative_prompt_2 = negative_prompt_2 * len(prompt_2)
        elif len(prompt_2) != len(negative_prompt_2):
            raise ValueError(
                "Prompt_2 and negative_prompt_2 should have the same length."
            )

    if isinstance(prompt, list) and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * len(prompt)
    if isinstance(prompt, str) and isinstance(negative_prompt, list):
        prompt = [prompt] * len(negative_prompt)

    if isinstance(prompt_2, list) and isinstance(negative_prompt_2, str):
        negative_prompt_2 = [negative_prompt_2] * len(prompt_2)
    if isinstance(prompt_2, str) and isinstance(negative_prompt_2, list):
        prompt_2 = [prompt_2] * len(negative_prompt_2)

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "prompt_embeds and negative_prompt_embeds should have the same shape."
                f" Got `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}"
            )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError("prompt and prompt_embeds should not be provided together.")
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            "negative_prompt and negative_prompt_embeds should not be provided together."
        )
    if prompt_2 is not None and prompt_embeds is not None:
        raise ValueError("prompt_2 and prompt_embeds should not be provided together.")
    if negative_prompt_2 is not None and negative_prompt_embeds is not None:
        raise ValueError(
            "negative_prompt_2 and negative_prompt_embeds should not be provided together."
        )

    return (
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
    )
