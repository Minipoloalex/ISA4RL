#!/bin/bash
for env in highway roundabout merge exit racetrack; do uv run --env-file .env main.py --env $env --task check; done
