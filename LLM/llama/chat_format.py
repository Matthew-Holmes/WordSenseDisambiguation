# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.


from dataclasses import dataclass
from typing import List

from .tokenizer import Tokenizer


@dataclass
class LLMInput:
    tokens: List[int]


class ChatFormat:

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_content(self, content: str, bos: bool = False) -> LLMInput:
        tokens = []

        added_bos = False

        def _process(c):
            nonlocal added_bos, bos

            tokens.extend(self.tokenizer.encode(c, bos=False if added_bos else bos, eos=False))
            added_bos = True

        if isinstance(content, list):
            for c in content:
                _process(c)
        else:
            _process(content)

        return LLMInput(tokens = tokens)
    
  
    
