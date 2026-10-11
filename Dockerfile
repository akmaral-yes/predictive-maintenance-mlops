# Serving image: code and locked dependencies only. No dataset, model, mlflow.db or mlruns;
# the exported champion is mounted at runtime and selected with AI4I_MODEL_URI.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first so this layer is reused when only the source changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1

EXPOSE 8000

# 503 (model unavailable) raises in urlopen, so the container is reported unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "ai4i_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
