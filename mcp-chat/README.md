# mcp-chat

CLI agent that connects to an MCP server via stdio, sends your query to an LLM, and streams the response. The LLM uses tool-calling to fetch live weather data.

See the [root README](../README.md) for setup and usage.

## Usage

```bash
# Single query
uv run --directory mcp-chat python -m mcp_chat "What is the weather in London?"

# Pipe from stdin
echo "What is the weather in Paris?" | uv run --directory mcp-chat python -m mcp_chat -
```

`mcp-server` is spawned automatically as a subprocess — no separate server process needed.

## Architecture

Hexagonal (Ports & Adapters):

```
CLI arg
  └── AgentService          # tool-call loop
        ├── LLMPort          # AnthropicAdapter / OpenAIAdapter
        └── MCPClientPort    # MCPClientAdapter → StdioMCPTransport → mcp-server subprocess
```

Streaming chunks flow as `LLMChunk` domain objects through `render_to_cli`.
