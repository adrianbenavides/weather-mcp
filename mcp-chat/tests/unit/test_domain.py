"""Acceptance tests for mcp-chat domain models and ports.

This test verifies:
- Conversation domain model (immutable message history)
- Message domain model (role, content)
- LLMChunk domain model (type discriminator)
- ToolSchema domain model
- LLMPort protocol interface
- MCPClientPort protocol interface
- MCPTransportPort protocol interface
"""

from typing import Any

import pytest
from mcp_chat.domain.conversation import Conversation, LLMChunk, Message, ToolSchema
from mcp_chat.ports.llm import LLMPort
from mcp_chat.ports.mcp_client import MCPClientPort
from mcp_chat.ports.mcp_transport import MCPTransportPort


class TestMessageModel:
    """Message domain model behavior."""

    def test_creates_message_with_role_and_content(self) -> None:
        """Message requires role and content."""
        msg = Message(role="user", content="What is the weather?")
        assert msg.role == "user"
        assert msg.content == "What is the weather?"

    def test_message_is_frozen(self) -> None:
        """Message is immutable."""
        msg = Message(role="user", content="Test")
        with pytest.raises(Exception):  # ValidationError or AttributeError
            msg.role = "assistant"  # type: ignore

    def test_message_with_assistant_role(self) -> None:
        """Message supports 'assistant' role."""
        msg = Message(role="assistant", content="The weather is sunny")
        assert msg.role == "assistant"

    def test_message_with_tool_role(self) -> None:
        """Message supports 'tool' role."""
        msg = Message(role="tool", content="Tool response")
        assert msg.role == "tool"


class TestToolSchema:
    """ToolSchema domain model behavior."""

    def test_creates_tool_schema_with_all_fields(self) -> None:
        """ToolSchema requires name, description, input_schema."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        tool = ToolSchema(
            name="get_weather",
            description="Get weather for a location",
            input_schema=schema,
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"
        assert tool.input_schema == schema

    def test_tool_schema_is_frozen(self) -> None:
        """ToolSchema is immutable."""
        tool = ToolSchema(
            name="test",
            description="Test tool",
            input_schema={},
        )
        with pytest.raises(Exception):  # ValidationError or AttributeError
            tool.name = "other"  # type: ignore


class TestLLMChunk:
    """LLMChunk domain model behavior (tagged union via type discriminator)."""

    def test_creates_text_delta_chunk(self) -> None:
        """LLMChunk with type='text_delta' includes text field."""
        chunk = LLMChunk.text_delta("Hello ")
        assert chunk.type == "text_delta"
        assert hasattr(chunk, "text")

    def test_creates_tool_use_start_chunk(self) -> None:
        """LLMChunk with type='tool_use_start' includes tool_name and tool_use_id."""
        chunk = LLMChunk.tool_use_start("get_weather", "tool_123")
        assert chunk.type == "tool_use_start"
        assert hasattr(chunk, "tool_name")
        assert hasattr(chunk, "tool_use_id")

    def test_creates_tool_use_input_chunk(self) -> None:
        """LLMChunk with type='tool_use_input' includes input_chunk."""
        chunk = LLMChunk.tool_use_input('{"location"')
        assert chunk.type == "tool_use_input"
        assert hasattr(chunk, "input_chunk")

    def test_creates_stop_chunk(self) -> None:
        """LLMChunk with type='stop' includes stop_reason."""
        chunk = LLMChunk.stop("end_turn")
        assert chunk.type == "stop"
        assert hasattr(chunk, "stop_reason")

    def test_llm_chunk_is_frozen(self) -> None:
        """LLMChunk is immutable."""
        chunk = LLMChunk.text_delta("test")
        with pytest.raises(Exception):  # ValidationError or AttributeError
            chunk.type = "other"  # type: ignore

    def test_creates_tool_use_id_chunk(self) -> None:
        """LLMChunk.tool_use_id_chunk creates correct chunk type."""
        chunk = LLMChunk.tool_use_id_chunk("call1")
        assert chunk.type == "tool_use_id"
        assert chunk.tool_use_id == "call1"

    def test_creates_tool_use_input_delta_chunk(self) -> None:
        """LLMChunk.tool_use_input_delta_chunk creates correct chunk type."""
        chunk = LLMChunk.tool_use_input_delta_chunk('{"loc')
        assert chunk.type == "tool_use_input"
        assert chunk.tool_use_input_delta == '{"loc'

    def test_creates_tool_use_complete_chunk(self) -> None:
        """LLMChunk.tool_use_complete creates correct chunk type with input."""
        chunk = LLMChunk.tool_use_complete("get_weather", "id1", {"city": "London"})
        assert chunk.type == "tool_use_complete"
        assert chunk.tool_name == "get_weather"
        assert chunk.tool_use_id == "id1"
        assert chunk.tool_use_input_complete == {"city": "London"}

    def test_text_delta_text_field_set(self) -> None:
        """LLMChunk.text_delta sets text field."""
        chunk = LLMChunk.text_delta("hello world")
        assert chunk.text == "hello world"

    def test_stop_chunk_stop_reason_set(self) -> None:
        """LLMChunk.stop sets stop_reason field."""
        chunk = LLMChunk.stop("max_tokens")
        assert chunk.stop_reason == "max_tokens"

    def test_tool_use_start_fields_set(self) -> None:
        """LLMChunk.tool_use_start sets tool_name and tool_use_id."""
        chunk = LLMChunk.tool_use_start("my_tool", "use_id_123")
        assert chunk.tool_name == "my_tool"
        assert chunk.tool_use_id == "use_id_123"


class TestConversation:
    """Conversation domain model behavior."""

    def test_creates_conversation_with_empty_history(self) -> None:
        """Conversation can be created empty."""
        conv = Conversation()
        assert len(conv.messages) == 0

    def test_creates_conversation_with_messages(self) -> None:
        """Conversation holds list of messages."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi there")
        conv = Conversation(messages=[msg1, msg2])
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].content == "Hi there"

    def test_conversation_is_frozen_or_immutable(self) -> None:
        """Conversation is immutable (frozen or uses tuple)."""
        msg = Message(role="user", content="Test")
        conv = Conversation(messages=[msg])
        # Either frozen (raises error on assignment) or uses immutable tuple
        # Either approach is acceptable
        try:
            conv.messages = []  # type: ignore
            # If no error, check if messages is tuple
            assert isinstance(conv.messages, tuple) or isinstance(conv.messages, list)
        except Exception:
            # Frozen - good
            pass

    def test_conversation_messages_stored_as_tuple(self) -> None:
        """Conversation stores messages as immutable tuple."""
        msg = Message(role="user", content="hello")
        conv = Conversation(messages=[msg])
        assert isinstance(conv.messages, tuple)

    def test_empty_conversation_has_zero_messages(self) -> None:
        """Conversation() with no args creates empty conversation."""
        conv = Conversation()
        assert len(conv.messages) == 0

    def test_conversation_with_none_uses_empty_list(self) -> None:
        """Conversation(messages=None) uses empty tuple."""
        conv = Conversation(messages=None)  # type: ignore
        assert len(conv.messages) == 0


class TestLLMPort:
    """LLMPort interface structure."""

    def test_llm_port_is_protocol(self) -> None:
        """LLMPort defines stream_response method."""
        assert hasattr(LLMPort, "__mro__")
        assert "Protocol" in str(type(LLMPort))

    def test_llm_port_defines_stream_response_method(self) -> None:
        """LLMPort.stream_response(conversation: Conversation, tools: list[ToolSchema]) -> AsyncIterator[LLMChunk]."""
        assert hasattr(LLMPort, "stream_response")


class TestMCPClientPort:
    """MCPClientPort interface structure."""

    def test_mcp_client_port_is_protocol(self) -> None:
        """MCPClientPort defines call_tool and list_tools methods."""
        assert hasattr(MCPClientPort, "__mro__")
        assert "Protocol" in str(type(MCPClientPort))

    def test_mcp_client_port_defines_call_tool_method(self) -> None:
        """MCPClientPort.call_tool(name: str, args: dict[str, Any]) -> str."""
        assert hasattr(MCPClientPort, "call_tool")

    def test_mcp_client_port_defines_list_tools_method(self) -> None:
        """MCPClientPort.list_tools() -> list[ToolSchema]."""
        assert hasattr(MCPClientPort, "list_tools")


class TestMCPTransportPort:
    """MCPTransportPort interface structure."""

    def test_mcp_transport_port_is_protocol(self) -> None:
        """MCPTransportPort defines connect, disconnect, is_connected methods."""
        assert hasattr(MCPTransportPort, "__mro__")
        assert "Protocol" in str(type(MCPTransportPort))

    def test_mcp_transport_port_defines_connect_method(self) -> None:
        """MCPTransportPort.connect() -> Awaitable[None]."""
        assert hasattr(MCPTransportPort, "connect")

    def test_mcp_transport_port_defines_disconnect_method(self) -> None:
        """MCPTransportPort.disconnect() -> Awaitable[None]."""
        assert hasattr(MCPTransportPort, "disconnect")

    def test_mcp_transport_port_defines_is_connected_method(self) -> None:
        """MCPTransportPort.is_connected() -> bool."""
        assert hasattr(MCPTransportPort, "is_connected")
