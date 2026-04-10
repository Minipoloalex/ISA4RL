#!/bin/bash
#SBATCH --job-name=diagnose_highway_env_racetrack
#SBATCH --output=logs/output_%j.log
#SBATCH --partition=dev-arm
#SBATCH --account=F202500007HPCVLABUPORTOa
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

source ~/.bashrc
UV_BIN="$HOME/.local/bin/arm64/uv"
export UV_PROJECT_ENVIRONMENT=".venv_arm"

cd /projects/$(cat ~/project.txt)/fmartins.up/ISA4RL-article/src/main

srun $UV_BIN run --frozen --no-sync --env-file .env main.py \
    --task check --env racetrack
