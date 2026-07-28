FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "multisusie"]
