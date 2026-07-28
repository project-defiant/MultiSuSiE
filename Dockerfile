FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "multisusie"]
