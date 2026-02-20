FROM nvidia/cuda:12.9.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

WORKDIR /app
COPY ./highway-env highway-env

WORKDIR /app/src/main

COPY ./src/main/uv.lock uv.lock
COPY ./src/main/pyproject.toml pyproject.toml

RUN uv sync --no-cache --no-editable --frozen

WORKDIR /app

COPY ./src/main src/main
COPY ./src/common src/common
COPY ./config config
