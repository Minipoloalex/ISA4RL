#!/bin/bash
#SBATCH --job-name=highway_env_racetrack
#SBATCH --time=48:00:00
#SBATCH --account=F202500007HPCVLABUPORTOa
#SBATCH --partition=large-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --array=0-7	# 8 nodes total (running 2 tasks each = 16 tasks)
#SBATCH --output=logs/main_job_racetrack_%A_%a.out   # Catch high-level script errors

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felix.marcial@hotmail.com

# Architecture: probably ARM
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    echo "Architecture: ARM64. Using ARM binary."
    UV_BIN="$HOME/.local/bin/arm64/uv"
    export UV_PROJECT_ENVIRONMENT=".venv_arm"
else
    echo "Architecture: x86_64. Using x86 binary."
    UV_BIN="$HOME/.local/bin/uv"
    export UV_PROJECT_ENVIRONMENT=".venv"
fi

PROJ_PATH="/projects/$(cat $HOME/project.txt)/fmartins.up"
LOG_PATH="$PROJ_PATH/logs"

mkdir -p $LOG_PATH

cd "$PROJ_PATH/ISA4RL-article/src/main"
source "$HOME/.bashrc"
module purge

BASE_START=100
STEP_SIZE=10

# Calculate the global task indices for the two tasks on this specific node
# Array 0 handles tasks 0 & 1. Array 1 handles tasks 2 & 3, etc.
TASK_A_IDX=$(($SLURM_ARRAY_TASK_ID * 2))
TASK_B_IDX=$(($SLURM_ARRAY_TASK_ID * 2 + 1))

# Start/End for Task A
START_A=$(($BASE_START + $TASK_A_IDX * STEP_SIZE))
END_A=$(($BASE_START + ($TASK_A_IDX + 1) * STEP_SIZE))

# Start/End for Task B
START_B=$(($BASE_START + $TASK_B_IDX * STEP_SIZE))
END_B=$(($BASE_START + ($TASK_B_IDX + 1) * STEP_SIZE))

echo "Node Allocation: $SLURM_ARRAY_TASK_ID"
echo "Launching Task A (Index $TASK_A_IDX): Range $START_A to $END_A"
echo "Launching Task B (Index $TASK_B_IDX): Range $START_B to $END_B"

$UV_BIN run --frozen --no-sync --env-file .env main.py \
    --task train --env racetrack \
    --start $START_A --end $END_A > "$LOG_PATH/racetrack_task_${START_A}_to_${END_A}.log" 2>&1 &

# Launch Task B in the background
$UV_BIN run --frozen --no-sync --env-file .env main.py \
    --task train --env racetrack \
    --start $START_B --end $END_B > "$LOG_PATH/racetrack_task_${START_B}_to_${END_B}.log" 2>&1 &

# Wait for both tasks to finish before releasing the node back to the cluster
wait
echo "Both tasks on this node have finished."
