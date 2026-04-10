#!/bin/bash

# --- CONFIGURATION ---
# Define the specific path to your .env file here
BASE_PATH=$HOME
cd $BASE_PATH/ISA4RL-article/src/main
# ---------------------

# Grab the command (start, stop, status, logs)
COMMAND=$1
# Grab the training parameters
ENV=$2
START=$3
END=$4

# Function to show usage instructions
show_usage() {
    echo "Usage: $0 {start|stop|status|logs} <env> <start> <end>"
    echo ""
    echo "Examples:"
    echo "  $0 start HalfCheetah-v4 900 1000"
    echo "  $0 logs HalfCheetah-v4 900 1000"
    echo "  $0 stop HalfCheetah-v4 900 1000"
    echo ""
    echo "Helper Commands:"
    echo "  $0 list   - Shows all currently saved PID files"
}

# Check for helper command first
if [ "$COMMAND" == "list" ]; then
    echo "Currently tracked PID files:"
    ls $BASE_PATH/pids/*.pid 2>/dev/null || echo "  No active PIDs found."
    exit 0
fi

# For the main commands, ensure the user provided the required parameters
if [[ ! "$COMMAND" =~ ^(start|stop|status|logs)$ ]] || [ -z "$ENV" ] || [ -z "$START" ] || [ -z "$END" ]; then
    show_usage
    exit 1
fi

# Define dynamic POSIX paths based on your parameters
LOG_OUT=$BASE_PATH/logs/train_${ENV}_${START}_${END}.out
LOG_ERR=$BASE_PATH/logs/train_${ENV}_${START}_${END}.err
PID_FILE=$BASE_PATH/pids/train_${ENV}_${START}_${END}.pid

start() {
    echo "Starting training: ENV=$ENV | START=$START | END=$END"
    echo "Using .env file at: $ENV_FILE_PATH"

    mkdir -p $BASE_PATH/logs
    mkdir -p $BASE_PATH/pids

    # Launch using `uv run` in the background with nohup.
    # Redirect stdout to LOG_OUT and stderr to LOG_ERR.
    nohup uv run --env-file .env main.py \
        --env "$ENV" --task "train" --start "$START" --end "$END" \
        > "$LOG_OUT" 2> "$LOG_ERR" < /dev/null &

    # Capture the Process ID of the last background command
    PID=$!
    echo $PID > "$PID_FILE"

    echo "Process started successfully in the background!"
    echo "PID ($PID) saved to: $PID_FILE"
    echo "Logs: $LOG_OUT"
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Checking process status for $ENV ($START to $END)..."
        # Check if the process is actually running
        if ps -p "$PID" > /dev/null; then
            echo "Status: RUNNING"
            # Print a clean format of the process details
            ps -p "$PID" -o pid,%cpu,%mem,lstart,cmd
        else
            echo "Status: STOPPED (PID file exists, but process $PID is no longer running)."
        fi
    else
        echo "Status: PID file not found for these parameters. The process is likely not running."
    fi
}

logs() {
    echo "Tailing logs for $ENV ($START to $END)... (Press Ctrl+C to stop)"
    touch "$LOG_OUT" "$LOG_ERR"
    # Tail both standard out and standard error
    tail -f "$LOG_OUT" "$LOG_ERR"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping process for $ENV ($START to $END) with PID $PID..."
        
        # Check if process is running before trying to kill it
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Process $PID stopped."
        else
            echo "Process $PID was not running."
        fi

        rm "$PID_FILE"
        echo "PID file removed."
    else
        echo "PID file not found. Cannot stop the process (it might have already finished)."
    fi
}

# Route the command
case "$COMMAND" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
esac
