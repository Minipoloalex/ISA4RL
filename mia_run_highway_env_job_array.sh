#!/bin/bash
#SBATCH --job-name=highway_env_parking
#SBATCH --output=logs/output_%A_%a.log   # Use %A (array job ID) and %a (task ID) to prevent overwriting logs
#SBATCH --array=1-10                     # number of lines in ranges.txt
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --nodelist=srv02

source .bashrc

# Extract the parameters for this specific task ID from ranges.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ranges.txt)

# Split the line into the START and END variables
read START END <<< $PARAMS

MAX_END=396

# Ensure the last job doesn't exceed MAX_END
if [ $END -gt $MAX_END ]; then
    END=$MAX_END
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID | Processing range: $START to $END"

srun apptainer exec --nv --containall \
    --bind ./results:/app/results \
    "highway-env-container.sif" \
    bash -c "cd /app/src/main && uv run --env-file .env --no-sync main.py --task train --env parking --start $START --end $END"
