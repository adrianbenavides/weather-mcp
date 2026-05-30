# Stage 1 - builder: Install dependencies and projects
FROM python:3.14-slim AS builder

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy workspace definition and package metadata
COPY pyproject.toml uv.lock ./
COPY mcp-server/ mcp-server/
COPY mcp-chat/ mcp-chat/

# Install all dependencies and projects into .venv
RUN uv sync --frozen


# Stage 2 - runtime: Minimal image with only what's needed to run mcp-chat
FROM python:3.14-slim AS runtime

# Copy uv (required at runtime by StdioMCPTransport)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy workspace and virtual environment from builder
COPY --from=builder /app /app

WORKDIR /app

# Configure environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Default entrypoint runs mcp-chat
ENTRYPOINT ["python", "-m", "mcp_chat"]


# Stage 3 - test: Extends builder with pytest for running tests
FROM builder AS test

WORKDIR /app

# Configure environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Run tests by default
CMD ["pytest", "mcp-chat/tests/", "-v"]
