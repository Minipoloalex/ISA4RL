FROM nvidia/cuda:12.9.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

WORKDIR /app
COPY ./highway-env highway-env
COPY ./pic pic

WORKDIR /app/src/common
COPY ./src/common/common common
COPY ./src/common/uv.lock uv.lock
COPY ./src/common/pyproject.toml pyproject.toml

WORKDIR /app/src/methods
COPY ./src/methods/methods methods
COPY ./src/methods/uv.lock uv.lock
COPY ./src/methods/pyproject.toml pyproject.toml

WORKDIR /app/src/highway_agents
COPY ./src/highway_agents/.python-version .python-version
COPY ./src/highway_agents/uv.lock uv.lock
COPY ./src/highway_agents/pyproject.toml pyproject.toml
COPY ./src/highway_agents/main.py main.py

RUN uv sync --no-cache --no-editable --frozen

COPY ./src/highway_agents/.env .env

WORKDIR /app
COPY ./config config
