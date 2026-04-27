FROM nvidia/cuda:12.9.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

WORKDIR /app
COPY ./metadrive metadrive
COPY ./pic pic

WORKDIR /app/src/common
COPY ./src/common/common common
COPY ./src/common/uv.lock uv.lock
COPY ./src/common/pyproject.toml pyproject.toml

WORKDIR /app/src/methods
COPY ./src/methods/methods methods
COPY ./src/methods/uv.lock uv.lock
COPY ./src/methods/pyproject.toml pyproject.toml

WORKDIR /app/src/metadrive_agents
COPY ./src/metadrive_agents/.python-version .python-version
COPY ./src/metadrive_agents/uv.lock uv.lock
COPY ./src/metadrive_agents/pyproject.toml pyproject.toml
COPY ./src/metadrive_agents/main.py main.py

RUN uv sync --no-cache --no-editable --frozen

COPY ./src/metadrive_agents/.env .env

WORKDIR /app
COPY ./config config
