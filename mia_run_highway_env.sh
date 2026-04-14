#!/bin/bash
#SBATCH --job-name=highway_env_parking
#SBATCH --output=logs/output_%j.log
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --nodelist=srv02

source .bashrc

START=0
END=396
MAX_END=396

# Ensure the last job doesn't exceed MAX_END
if [ $END -gt $MAX_END ]; then
    END=$MAX_END
fi

echo "Processing range: $START to $END"

srun apptainer exec --nv --containall \
    --bind ./results:/app/results \
    "highway-env-container.sif" \
    bash -c "cd /app/src/main && uv run --env-file .env --no-sync main.py --task evaluate --env parking --start $START --end $END"
