# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/chat_format.py

import uuid

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


from .datatypes import (
    BuiltinTool,
    RawContent,
    RawMessage,
    RawTextItem,
    Role,
    StopReason,
    ToolCall,
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
        self.vision_token = self.tokenizer.special_tokens["<|image|>"]

    def _encode_header(self, role: str) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

 
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
