# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/generation.py
# TODO - disable CUDA
# TODO - propagate rotary embed len vs cache len change

from dataclasses import dataclass
from typing import Generator, List, Optional

import torch
import torch.nn.functional as F

from termcolor import cprint

from .model import ModelArgs
from .chat_format import ChatFormat, LLMInput
from .tokenizer import Tokenizer
from .model import Transformer


@dataclass
class CompletionPrediction:
    generation: str
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: Optional[List[float]] = None


class Llama:

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)


    @torch.inference_mode()
    def generate(
        self,
        model_input: LLMInput,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = True,
        echo: bool = False,
        print_model_input: bool = True,
        repetition_penalty: float = 1.2,  # Added repetition penalty
    ) -> Generator:
        params = self.model.params

        if print_model_input:
            cprint(
                "Input to model:\n" + self.tokenizer.decode(model_input.tokens) + "\n",
                "red",
            )
        prompt_tokens = [model_input.tokens]

        bsz = 1
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.cache_len:
            cprint(f"Out of token budget {max_prompt_len} vs {params.cache_len}", "red")
            return

        total_len = min(max_gen_len + max_prompt_len, params.cache_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id

        if echo:
            for i, t in enumerate(model_input.tokens):
                yield TokenResult(
                    token=t,
                    text=self.tokenizer.decode([t]),
                    logprobs=(token_logprobs[0, i : i + 1].tolist() if logprobs else None),
                )

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # Apply repetition penalty
            past_tokens = tokens[:, :cur_pos]  # Extract previously generated tokens

            scores = logits[:, -1]

            past_scores = torch.gather(scores, 1, past_tokens)

            past_scores = torch.where(
                past_scores < 0, past_scores * repetition_penalty, past_scores / repetition_penalty
            )

            scores = scores.scatter(1, past_tokens, past_scores)

            # Sample next token
            if temperature > 0:
                probs = torch.softmax(scores / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(scores, dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=target,
                    reduction="none",
                    ignore_index=pad_id,
                )

            eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            yield TokenResult(
                token=next_token[0].item(),
                text=self.tokenizer.decode(next_token.tolist()),
                logprobs=(token_logprobs[:, cur_pos : cur_pos + 1][0].tolist() if logprobs else None),
            )

            prev_pos = cur_pos
            if all(eos_reached):
                break

    def text_completion(
        self,
        content: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = True,
        echo: bool = False,
    ) -> CompletionPrediction:
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.params.cache_len:
            max_gen_len = self.model.params.cache_len - 1

        model_input = self.formatter.encode_content(content)

        tokens = []
        token_logprobs = []
        decoded_tokens = []
        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        ):
            tokens.append(result.token)
            if logprobs:
                decoded_tokens.append(result.text)
                token_logprobs.append(result.logprobs)

        generation = self.tokenizer.decode(tokens)
        if logprobs:
            return CompletionPrediction(
                generation=generation,
                logprobs=token_logprobs,
                decoded_tokens=decoded_tokens,
            )

        return CompletionPrediction(generation=generation)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token