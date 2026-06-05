"""Converts domain conversation models to LLM-specific formats."""

import json
from typing import Any

from beartype import beartype

from mcp_chat.domain.conversation import Conversation, ToolSchema


@beartype
def conversation_to_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Convert domain conversation to generic LLM message format (no tool metadata)."""
    return [{"role": msg.role, "content": msg.content} for msg in conversation.messages]


@beartype
def conversation_to_anthropic_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Convert domain conversation to Anthropic message format.

    Tool-use assistant messages become content blocks; tool results become user messages.
    """
    result: list[dict[str, Any]] = []
    for msg in conversation.messages:
        if msg.role == "assistant" and msg.tool_use_id:
            result.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": msg.tool_use_id,
                            "name": msg.tool_name,
                            "input": msg.tool_use_input or {},
                        }
                    ],
                }
            )
        elif msg.role == "tool" and msg.tool_use_id:
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_use_id,
                            "content": msg.content,
                        }
                    ],
                }
            )
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


@beartype
def conversation_to_openai_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Convert domain conversation to OpenAI message format.

    Tool-use assistant messages become tool_calls; tool results keep role=tool.
    """
    result: list[dict[str, Any]] = []
    for msg in conversation.messages:
        if msg.role == "assistant" and msg.tool_use_id:
            result.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": msg.tool_use_id,
                            "type": "function",
                            "function": {
                                "name": msg.tool_name,
                                "arguments": json.dumps(msg.tool_use_input or {}),
                            },
                        }
                    ],
                }
            )
        elif msg.role == "tool" and msg.tool_use_id:
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_use_id,
                    "content": msg.content,
                }
            )
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


@beartype
def tools_to_definitions(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert domain tools to generic definition format.

    Args:
        tools: Domain tool schemas.

    Returns:
        List of tool definitions (name, description, input_schema).
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]
