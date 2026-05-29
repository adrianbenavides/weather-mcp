"""Integration tests for LLM adapters with real API calls.

This test verifies:
- AnthropicAdapter.stream_response() with live Anthropic API yields LLMChunk objects
- OpenAIAdapter.stream_response() with live OpenAI API yields LLMChunk objects
- All LLMChunk variants are produced correctly
"""

import os

import pytest
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import Conversation, LLMChunk, Message, ToolSchema


class TestAnthropicAdapterIntegration:
    """AnthropicAdapter integration tests with live API."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_anthropic_adapter_streams_response_yielding_llm_chunks(self) -> None:
        """AnthropicAdapter.stream_response() yields LLMChunk objects."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter

        config = AppConfig(anthropic_api_key=api_key)
        adapter = AnthropicAdapter(config=config)

        conversation = Conversation(messages=[Message(role="user", content="Say 'Hello' briefly.")])
        tools: list[ToolSchema] = []

        chunks: list[LLMChunk] = []
        async for chunk in adapter.stream_response(conversation, tools):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should yield at least one chunk"

        # Verify at least one chunk is a text_delta
        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) > 0, "Should have at least one text_delta chunk"

        # Verify each chunk is a valid LLMChunk
        for chunk in chunks:
            assert isinstance(chunk, LLMChunk)
            assert chunk.type in [
                "text_delta",
                "tool_use_start",
                "tool_use_id",
                "tool_use_input",
                "tool_use_complete",
                "stop",
            ]


class TestOpenAIAdapterIntegration:
    """OpenAIAdapter integration tests with live API."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_openai_adapter_streams_response_yielding_llm_chunks(self) -> None:
        """OpenAIAdapter.stream_response() yields LLMChunk objects."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        from mcp_chat.adapters.llm.openai_adapter import OpenAIAdapter

        config = AppConfig(openai_api_key=api_key)
        adapter = OpenAIAdapter(config=config)

        conversation = Conversation(messages=[Message(role="user", content="Say 'Hello' briefly.")])
        tools: list[ToolSchema] = []

        chunks: list[LLMChunk] = []
        async for chunk in adapter.stream_response(conversation, tools):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should yield at least one chunk"

        # Verify at least one chunk is a text_delta
        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) > 0, "Should have at least one text_delta chunk"

        # Verify each chunk is a valid LLMChunk
        for chunk in chunks:
            assert isinstance(chunk, LLMChunk)
            assert chunk.type in [
                "text_delta",
                "tool_use_start",
                "tool_use_id",
                "tool_use_input",
                "tool_use_complete",
                "stop",
            ]
