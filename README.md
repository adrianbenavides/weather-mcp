# weather-mcp

AI weather agent using the [Model Context Protocol](https://modelcontextprotocol.io). Ask a question in plain text — the LLM calls `get_current_weather` via MCP, fetches live data from [Open-Meteo](https://open-meteo.com), and answers.

Two packages in a uv workspace:

- **mcp-server** — exposes `get_current_weather` as an MCP tool over stdio
- **mcp-chat** — CLI client that connects to the server, sends your query to an LLM, and streams the response

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Running locally

`mcp-chat` spawns `mcp-server` automatically as a subprocess — you only need one command.

### Anthropic (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run --directory mcp-chat python -m mcp_chat "What is the weather in London?"
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
uv run --directory mcp-chat python -m mcp_chat "What is the weather in Tokyo?"
```

### Pipe a query from stdin

```bash
echo "What is the weather in Paris?" | uv run --directory mcp-chat python -m mcp_chat -
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (Anthropic) | — | Anthropic API key |
| `OPENAI_API_KEY` | Yes (OpenAI) | — | OpenAI API key |
| `LLM_PROVIDER` | No | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | No | `claude-haiku-4-5-20251001` | Override the model |
| `LOG_FORMAT` | No | `json` | `json` or `console` |

## Running tests

```bash
# All tests
uv run --directory mcp-chat pytest tests/unit -v
uv run --directory mcp-server pytest tests/unit -v

# Integration tests (hit live APIs — requires API keys)
uv run --directory mcp-chat pytest tests/integration -v -m integration
uv run --directory mcp-server pytest tests/integration -v -m integration
```
