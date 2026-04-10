#!/bin/bash
#SBATCH --job-name=highway_env_merge
#SBATCH --output=logs/output_%j.log
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=srv02

source .bashrc

START=0
STEP=432
END=$(( START + STEP ))
MAX_END=432

# Ensure the last job doesn't exceed 432
if [ $END -gt $MAX_END ]; then
    END=$MAX_END
fi

echo "Processing range: $START to $END"

srun apptainer exec --nv --containall \
    --bind ./results:/app/results \
    "highway-env-container.sif" \
    bash -c "cd /app/src/main && uv run --env-file .env --no-sync main.py --task evaluate --env merge --start $START --end $END"
