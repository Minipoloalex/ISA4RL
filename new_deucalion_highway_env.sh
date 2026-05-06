#!/bin/bash
#SBATCH --job-name=highway_env_racetrack
#SBATCH --time=72:00:00
#SBATCH --account=F202500007HPCVLABUPORTOa
#SBATCH --partition=large-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --array=1-11 # 1-based index to easily read lines from ranges.txt
#SBATCH --output=logs/main_job_racetrack_%A_%a.out

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
yPROJ_PATH="/projects/$(cat $HOME/project.txt)/fmartins.up"
LOG_PATH="$PROJ_PATH/logs"

mkdir -p $LOG_PATH

cd "$PROJ_PATH/ISA4RL-article/src/main"
source "$HOME/.bashrc"
module purge

# ==========================================
# READ ARBITRARY RANGES FROM FILE
# ==========================================
# Get the specific line from ranges.txt corresponding to this array task ID
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PROJ_PATH/ranges.txt")

# Split the line into the 4 variables
read START_A END_A START_B END_B <<< $PARAMS
# ==========================================

echo "Node Allocation: $SLURM_ARRAY_TASK_ID"
echo "Launching Task A: Range $START_A to $END_A"
echo "Launching Task B: Range $START_B to $END_B"

$UV_BIN run --frozen --no-sync --env-file .env main.py \
    --task train --env racetrack \
    --start $START_A --end $END_A > "$LOG_PATH/racetrack_${START_A}_to_${END_A}.log" 2>&1 &

$UV_BIN run --frozen --no-sync --env-file .env main.py \
    --task train --env racetrack \
    --start $START_B --end $END_B > "$LOG_PATH/racetrack_${START_B}_to_${END_B}.log" 2>&1 &

wait
echo "Both tasks on this node have finished."
