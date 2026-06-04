# weather-mcp

AI weather agent using the [Model Context Protocol](https://modelcontextprotocol.io). Ask a question in plain text — the LLM calls `get_current_weather` via MCP, fetches live data from [Open-Meteo](https://open-meteo.com), and answers.

Demo [here](https://asciinema.org/a/D3F578dA8Kh5Mj3B).

## How to use

### With docker

Requires Docker and an `.env` file with your API keys (see [Environment variables](#environment-variables) below).

```bash
cp .env.example .env   # edit with your keys
```

```bash
docker compose run --rm chat
```

### Run tests

```bash
docker compose run --rm test
```

### Locally

```bash
uv sync
```

### Anthropic (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run --directory mcp-chat python -m mcp_chat
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
uv run --directory mcp-chat python -m mcp_chat
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (Anthropic) | — | Anthropic API key |
| `OPENAI_API_KEY` | Yes (OpenAI) | — | OpenAI API key |
| `LLM_PROVIDER` | No | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | No | `claude-haiku-4-5-20251001` | Override the model |
| `LOG_FORMAT` | No | `json` | `json` or `console` |

## Quality checks and tests

```bash
uv run poe lint
uv run poe test
```
