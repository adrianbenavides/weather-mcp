## Overview

The implementation involves two Python packages:
- `mcp-server`: exposes the MCP tools (only `get_current_weather` has been implemented) over stdio and handles the MCP tools logic. No LLM awareness, stateless per call.
- `mcp-chat`: the user-facing application exposing a "chat". Drives the LLM usage and routes tool calls to the `mcp-server`. This implementation exposes only a CLI running a REPL. A new interface (e.g. web UI) can be easily added under `transport`.

See [README.md](README.md) to learn how to use it.

## Design/Architecture

Both packages follow hexagonal architecture (which is very well supported out of the box by `nwave`) with an immutability-first domain model (I use a custom agent+skill that I created to use some ideas from `rust`).

The system touches three external boundaries (LLM provider, MCP transport, weather API) each with different protocols and failure modes. Hexagonal architecture isolates the core domain from all infrastructure concerns, allowing adapters to be swapped (e.g., Anthropic -> OpenAI) without touching business logic, and making domain logic unit-testable without network calls. A bit more boilerplate is needed, but it pays off in extension, maintainability, and testability.

### Rusty python

Here's an overview of the main features I use in python to exploit its typing features:
- Type checking with `mypy`, `pydantic` (for models, immutable by default), `pandera` (for dataframes), `beartype` (for functions)
- `NewType` for domain primitives (e.g., `OrderId = NewType('OrderId', str)`)
- Define behavior with `Protocols` instead of using class inheritance (easier dependency-injection and testability)

### LLM abstraction

Provider SDKs have incompatible message formats, tool schemas, and streaming APIs. Wrapping them behind a single protocol means:
- `AgentService` depends on `LLMPort`, not on `anthropic` or `openai` — mypy enforces this via import-linter contracts
- Provider swap requires only an env var change; zero application code changes
- Tests mock `LLMPort` rather than SDK classes (faster, no real API key needed)

I only tested `anthropic` because that's what I have, but I still wanted to implement the abstraction.

### tool-call loop

The tool-call loop is split across two components with clear responsibilities:

- **`AgentService.run_turn()`** — single LLM call; yields all chunks (text, tool_use, stop) as an `AsyncIterator[LLMChunk]`; holds no loop logic internally
- **CLI REPL** — consumes chunks, renders text immediately, pauses on `tool_use`, calls `MCPClientPort.call_tool()`, appends the result to the conversation, and recurses for the next LLM turn.

This separation keeps `AgentService` testable in isolation and preserves true first-token streaming (text tokens appear before the tool round-trip completes). Session context persists in `AgentService._messages` across turns within a process.

### Conversation context

One of the requirements was keeping the conversation during a chat session. The `AgentService` keeps the list of messages and exposes a `compact` function to summarize the messages (unnecessary given the scale of the responses, but still something I wanted to explore).

## Future work

For the toy-project scope:
- More MCP tools (we'd need to extend `mcp-server/src/mcp_server/transport/mcp_handler.py` and new services under `mcp-server/src/mcp_server/application` if needed).
- More detailed tracing (currently enough to see that the `chat` sends a request and the `server` receives it).
- Web UI + HTTP transport (should be pretty straightforward with the current architecture).

For a more production-ready scope:
- HTTP API layer. The CLI REPL will have to be refactored to use FastAPI routes (e.g. `POST /api/chat/{id}/turns`).
- Session persistence. Conversations live in the running process. Any restart loses all state. We'd need PostgreSQL + ConversationRepository port.
- The server's stdio MCP is not horizontally scalable (StdioMCPTransport spawns subprocesses, can't scale across containers) -> switch to HTTP MCP transport to handle concurrent connections and enable the deployment of multiple MCP server replicas behind a load balancer
- Client auth. There is no JWT, no user identity, no rate limiting. Any user can run unlimited LLM calls.