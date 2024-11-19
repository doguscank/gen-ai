"""
Those functions are highly inspired by the CLIP hijack implementation in
AUTOMATIC1111's implementation.
"""

import re
from typing import List, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from gen_ai.image_gen.stable_diffusion_15.input import StableDiffusionInput

_MAX_TOKEN_COUNT = 75  # excluding the bos and eos tokens
_DEFAULT_ATTENTION = 1.0


class PromptPiece:
    """
    A piece of text in the prompt with an associated attention multiplier.

    Parameters
    ----------
    text_piece : str
        The text piece.
    attention_multiplier : float
        The attention multiplier for the text piece.

    Attributes
    ----------
    text_piece : str
        The text piece.
    attention_multiplier : float
        The attention multiplier for the text piece.

    Methods
    -------
    multiply_attention(multiplier: float)
        Multiplies the attention multiplier by the given multiplier.
    """

    def __init__(self, text_piece, attention_multiplier):
        self._text_piece = text_piece
        self._attention_multiplier = attention_multiplier

    @property
    def text_piece(self):
        return self._text_piece

    @property
    def attention_multiplier(self):
        return self._attention_multiplier

    def multiply_attention(self, multiplier: float):
        self._attention_multiplier *= multiplier

    def __str__(self):
        return f"Text Piece: '{self.text_piece}', Attention Multiplier: {self.attention_multiplier}"

    def __repr__(self):
        return str(self)


class TokenizedPrompt:
    """
    A tokenized version of the prompt with associated weights.

    Parameters
    ----------
    input_ids : List[int]
        The input IDs of the tokenized prompt.
    weights : List[float]
        The weights associated with each token.
    bos_token_id : int
        The ID of the beginning of sentence token.
    eos_token_id : int
        The ID of the end of sentence token.
    pad_token_id : int
        The ID of the padding token.
    max_token_count : int, optional
        The maximum token count for the tokenized prompt. Default is 75.

    Attributes
    ----------
    input_ids : List[int]
        The input IDs of the tokenized prompt.
    weights : List[float]
        The weights associated with each token.
    bos_token_id : int
        The ID of the beginning of sentence token.
    eos_token_id : int
        The ID of the end of sentence token.
    pad_token_id : int
        The ID of the padding token.
    max_token_count : int
        The maximum token count for the tokenized prompt.
    """

    def __init__(
        self,
        input_ids: List[int],
        weights: List[float],
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_token_count: int = _MAX_TOKEN_COUNT,
    ):
        self._input_ids = input_ids
        self._weights = weights
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._max_token_count = max_token_count

        self._add_special_characters()

    def _add_special_characters(self) -> None:
        self._input_ids = [self.bos_token_id] + self._input_ids
        self._weights = [1] + self._weights

        if len(self._input_ids) <= self.max_token_count:
            self._input_ids.extend(
                [self._eos_token_id] * (self.max_token_count - len(self._input_ids) + 1)
            )
            self._weights.extend([1] * (self.max_token_count - len(self._weights) + 1))

    @property
    def input_ids(self):
        return self._input_ids

    @property
    def weights(self):
        return self._weights

    @property
    def bos_token_id(self):
        return self._bos_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def max_token_count(self):
        return self._max_token_count

    def __str__(self):
        return f"Input IDs: {self.input_ids}, Weights: {self.weights}"

    def __repr__(self):
        return str(self)


def multiply_attention(
    prompt_pieces: List[PromptPiece],
    start_pos: int,
    multiplier: float,
    inplace: bool = True,
) -> Union[List[PromptPiece], None]:
    """
    Multiplies the attention of prompt pieces starting from given position.

    Parameters
    ----------
    prompt_pieces : List[PromptPiece]
        The list of prompt pieces to modify
    start_pos : int
        Starting position (must be >= 0)
    multiplier : float
        Attention multiplier (must be > 0)
    inplace : bool, optional
        If True, modifies list in-place. Default True

    Returns
    -------
    Union[List[PromptPiece], None]
        None if inplace is True, otherwise a new list of Prompt Pieces.

    Raises
    ------
    ValueError
        If start_pos or multiplier are invalid
    """

    if not prompt_pieces:
        return [] if not inplace else None

    if start_pos < 0 or start_pos >= len(prompt_pieces):
        raise ValueError(f"start_pos must be between 0 and {len(prompt_pieces)-1}")

    if multiplier == 0:
        raise ValueError("Multiplier must be non-zero")

    working_pieces = prompt_pieces if inplace else prompt_pieces.copy()

    for piece in working_pieces[start_pos:]:
        piece.multiply_attention(multiplier)

    return None if inplace else working_pieces


def parse_prompt(
    prompt: str,
    default_attention: float = _DEFAULT_ATTENTION,
) -> List[PromptPiece]:
    """
    Parses the prompt into individual pieces with attention multipliers.

    Parameters
    ----------
    prompt : str
        The prompt to parse.
    default_attention : float, optional
        The default attention multiplier. Default is 1.0.

    Returns
    -------
    List[PromptPiece]
        The list of prompt pieces.
    """

    attention_regex = re.compile(
        r"""
        \(|  # opening parenthesis
        :\s*([+-]?[.\d]+)\s*\)|  # attention multiplier with optional sign
        \)|  # closing parenthesis
        [^\():]+  # any character except parenthesis
        """,
        re.X,
    )
    prompt_pieces: List[PromptPiece] = []
    parenthesis_stack = []

    for match in attention_regex.finditer(prompt):
        piece = match.group(0)
        weight = match.group(1)

        if piece == "(":  # attention start
            parenthesis_stack.append(len(prompt_pieces))
        elif weight is not None and parenthesis_stack:  # found attention
            multiply_attention(prompt_pieces, parenthesis_stack.pop(), float(weight))
        else:
            prompt_pieces.append(PromptPiece(piece, default_attention))

    return prompt_pieces


def tokenize(
    prompt_pieces: List[PromptPiece],
    tokenizer: CLIPTokenizer,
    max_token_count: int = _MAX_TOKEN_COUNT,
) -> List[TokenizedPrompt]:
    """
    Tokenizes the prompt pieces.

    Parameters
    ----------
    prompt_pieces : List[PromptPiece]
        The list of prompt pieces.
    tokenizer : CLIPTokenizer
        The tokenizer to use.
    max_token_count : int, optional
        The maximum token count for each tokenized prompt. Default is 75.

    Returns
    -------
    List[TokenizedPrompt]
        The tokenized prompts.
    """

    inputs = tokenizer(
        [p.text_piece for p in prompt_pieces],
        truncation=False,
        add_special_tokens=False,
    )

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = eos_token_id

    input_ids = []
    weights = []

    for idx, input_id in enumerate(inputs["input_ids"]):
        input_ids.extend(input_id)
        weights.extend([prompt_pieces[idx].attention_multiplier] * len(input_id))

    tokenized_prompts: List[TokenizedPrompt] = []

    for idx in range(0, len(input_ids), max_token_count):
        tokenized_prompts.append(
            TokenizedPrompt(
                input_ids=input_ids[idx : idx + max_token_count],
                weights=weights[idx : idx + max_token_count],
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_token_count=max_token_count,
            )
        )

    return tokenized_prompts


def process(
    prompt: str,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    update_mean: bool = True,
) -> torch.Tensor:
    """
    Processes the prompt and returns the embeddings.

    Parameters
    ----------
    prompt : str
        The prompt to process.
    tokenizer : CLIPTokenizer
        The tokenizer to use.
    model : CLIPTextModel
        The model to use.
    update_mean : bool, optional
        If True, updates the mean of the embeddings. Default is True.

    Returns
    -------
    torch.Tensor
        The embeddings.
    """

    parsed_prompt = parse_prompt(prompt)
    tokenized_prompts = tokenize(parsed_prompt, tokenizer)

    embeddings = []

    for tokenized_prompt in tokenized_prompts:
        output = model(
            torch.tensor([tokenized_prompt.input_ids]).to(device=model.device),
            output_hidden_states=True,
        )

        z = output["last_hidden_state"]

        weights = torch.asarray(tokenized_prompt.weights).to(device=model.device)
        weights = weights.reshape(weights.shape + (1,))
        weights = weights.expand(output["last_hidden_state"].shape)

        final_output = output["last_hidden_state"] * weights

        if update_mean:
            z_mean = z.mean()
            new_mean = final_output.mean()
            final_output = final_output * (z_mean / new_mean)

        embeddings.append(final_output)

    return torch.hstack(embeddings)


def process_input_config(
    input_config: StableDiffusionInput,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    update_mean: bool = True,
    device: str = "cuda",
) -> StableDiffusionInput:
    """
    Processes the input config and returns the embeddings.

    Parameters
    ----------
    input_config : StableDiffusionInput
        The input config to process.
    tokenizer : CLIPTokenizer
        The tokenizer to use.
    model : CLIPTextModel
        The model to use.
    update_mean : bool, optional
        If True, updates the mean of the embeddings. Default is True.
    device : str, optional
        The device to use. Default is "cuda".

    Returns
    -------
    StableDiffusionInput
        The updated input config.
    """

    prompt = input_config.prompt
    negative_prompt = input_config.negative_prompt

    input_config.prompt_embeds = process(
        prompt=prompt, tokenizer=tokenizer, model=model, update_mean=update_mean
    ).to(device=device)
    input_config.negative_prompt_embeds = process(
        prompt=negative_prompt,
        tokenizer=tokenizer,
        model=model,
        update_mean=update_mean,
    ).to(device=device)

    input_config.prompt = None
    input_config.negative_prompt = None

    return input_config
