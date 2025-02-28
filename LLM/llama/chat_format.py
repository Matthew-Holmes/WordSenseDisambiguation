# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import io
import uuid

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .datatypes import (
    BuiltinTool,
    RawContent,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    Role,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)

from .tokenizer import Tokenizer

from .tool_utils import ToolUtils


@dataclass
class LLMInput:
    tokens: List[int]


def role_str(role: Role) -> str:
    role_strs = {
        Role.user: "user",
        Role.system: "system",
        Role.tool: "ipython",  # special
        Role.assistant: "assistant",
    }
    return role_strs[role]


class ChatFormat:
    possible_headers: Dict[Role, str]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.possible_headers = {role: f"<|start_header_id|>{role_str(role)}<|end_header_id|>\n\n" for role in Role}

    def _encode_header(self, role: str) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_content(self, content: RawContent, bos: bool = False) -> LLMInput:
        tokens = []

        added_bos = False

        def _process(c):
            nonlocal added_bos, bos

            if isinstance(c, str) or isinstance(c, RawTextItem):
                if isinstance(c, RawTextItem):
                    c = c.text
                tokens.extend(self.tokenizer.encode(c, bos=False if added_bos else bos, eos=False))
                added_bos = True

        if isinstance(content, list):
            for c in content:
                _process(c)
        else:
            _process(content)

        return LLMInput(tokens = tokens)
    
    def encode_message(
        self, message: RawMessage, tool_prompt_format: ToolPromptFormat
    ) -> List[int]:
        
        tokens = self._encode_header(message.role)

        def _process_content(c):
            toks = self.encode_content(c).tokens
            tokens.extend(toks)

        if (
            message.role == "assistant"
            and len(message.tool_calls) > 0
            and message.tool_calls[0].tool_name == BuiltinTool.code_interpreter
        ):
            tokens.append(self.tokenizer.special_tokens["<|python_tag|>"])

        _process_content(message.content)

        if message.role == "user" and message.context is not None:
            # This is RAG context; why is it here in the chat format? I don't think
            # this is needed and can be moved upwards
            _process_content("\n\n")
            _process_content(message.context)

        if message.role == "assistant":
            for t in message.tool_calls:
                content = ToolUtils.encode_tool_call(t, tool_prompt_format)
                _process_content(content)

        eom = False
        if message.role == "assistant":
            eom = message.stop_reason == StopReason.end_of_message

        tokens.append(self.tokenizer.special_tokens["<|eom_id|>" if eom else "<|eot_id|>"])
        return tokens
    

    def encode_dialog_prompt(
        self,
        messages: List[RawMessage],
        tool_prompt_format: Optional[ToolPromptFormat] = None,
    ) -> LLMInput:
        tool_prompt_format = tool_prompt_format or ToolPromptFormat.json
        tokens = []
        images = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in messages:
            toks  = self.encode_message(message, tool_prompt_format)
            tokens.extend(toks)

        # Add the start of an assistant message for the model to complete.
        tokens.extend(self._encode_header("assistant"))

        return LLMInput(tokens = tokens)

    # TODO(this should be generic, not only for assistant messages)
    def decode_assistant_message(self, tokens: List[int], stop_reason: StopReason) -> RawMessage:
        content = self.tokenizer.decode(tokens)

        return self.decode_assistant_message_from_content(content, stop_reason)

    def decode_assistant_message_from_content(self, content: str, stop_reason: StopReason) -> RawMessage:
        content = content.strip(" ")
        header_str = self.possible_headers[Role.assistant]
        if content.startswith(header_str):
            content = content[len(header_str) :]

        ipython = content.startswith("<|python_tag|>")
        if ipython:
            content = content[len("<|python_tag|>") :]

        if content.endswith("<|eot_id|>"):
            content = content[: -len("<|eot_id|>")]
            stop_reason = StopReason.end_of_turn
        elif content.endswith("<|eom_id|>"):
            content = content[: -len("<|eom_id|>")]
            stop_reason = StopReason.end_of_message

        tool_name = None
        tool_arguments = {}

        custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
        if custom_tool_info is not None:
            tool_name, tool_arguments = custom_tool_info
            # Sometimes when agent has custom tools alongside builin tools
            # Agent responds for builtin tool calls in the format of the custom tools
            # This code tries to handle that case
            if tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
                tool_arguments = {
                    "query": list(tool_arguments.values())[0],
                }
        else:
            builtin_tool_info = ToolUtils.maybe_extract_builtin_tool_call(content)
            if builtin_tool_info is not None:
                tool_name, query = builtin_tool_info
                tool_arguments = {
                    "query": query,
                }
                if tool_name in BuiltinTool.__members__:
                    tool_name = BuiltinTool[tool_name]
            elif ipython:
                tool_name = BuiltinTool.code_interpreter
                tool_arguments = {
                    "code": content,
                }

        tool_calls = []
        if tool_name is not None and tool_arguments is not None:
            call_id = str(uuid.uuid4())
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=tool_arguments,
                )
            )
            content = ""

        return RawMessage(
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )