#!/bin/bash

# --- CONFIGURATION ---
# Define the specific path to your .env file here
ENV_FILE_PATH="./.env"
# ---------------------

# Grab the command (start, stop, status, logs, check)
COMMAND=$1
# Grab the parameters
ENV=$2
START=$3
END=$4
# Grab the optional task parameter (defaults to "train" if omitted)
TASK=${5:-train}

# Function to show usage instructions
show_usage() {
    echo "Usage: $0 {start|stop|status|logs|check} <env> <start> <end> [task]"
    echo ""
    echo "Note: [task] is optional. If not provided, it defaults to 'train'."
    echo ""
    echo "Examples:"
    echo "  $0 start HalfCheetah-v4 900 1000          # Runs task 'train'"
    echo "  $0 start HalfCheetah-v4 900 1000 evaluate # Runs task 'evaluate'"
    echo "  $0 logs HalfCheetah-v4 900 1000 evaluate  # Views logs for 'evaluate'"
    echo "  $0 stop HalfCheetah-v4 900 1000           # Stops 'train'"
    echo "  $0 check HalfCheetah-v4 900 1000          # Runs check sync"
    echo ""
    echo "Helper Commands:"
    echo "  $0 list   - Shows all currently saved PID files"
}

# Check for helper command first
if [ "$COMMAND" == "list" ]; then
    echo "Currently tracked PID files:"
    ls ~/MSc_Felix/pids/*.pid 2>/dev/null || echo "  No active PIDs found."
    exit 0
fi

# For the main commands, ensure the user provided the required parameters
if [[ ! "$COMMAND" =~ ^(start|stop|status|logs|check)$ ]] || [ -z "$ENV" ] || [ -z "$START" ] || [ -z "$END" ]; then
    show_usage
    exit 1
fi

# Define dynamic POSIX paths based on your parameters (Updated to include TASK)
LOG_OUT=~/MSc_Felix/logs/${TASK}_${ENV}_${START}_${END}.out
LOG_ERR=~/MSc_Felix/logs/${TASK}_${ENV}_${START}_${END}.err
PID_FILE=~/MSc_Felix/pids/${TASK}_${ENV}_${START}_${END}.pid

start() {
    echo "Starting process: TASK=$TASK | ENV=$ENV | START=$START | END=$END"
    echo "Using .env file at: $ENV_FILE_PATH"

    mkdir -p ~/MSc_Felix/logs
    mkdir -p ~/MSc_Felix/pids

    # Compute Windows paths for PowerShell
    PYTHON_WIN=$(cygpath -w ./ISA4RL-article/src/main/.venv/Scripts/python.exe)
    SCRIPT_WIN=$(cygpath -w ./ISA4RL-article/src/main/main.py)
    ENV_FILE_WIN=$(cygpath -w "$ENV_FILE_PATH")
    LOG_WIN=$(cygpath -w "$LOG_OUT")
    ERR_WIN=$(cygpath -w "$LOG_ERR")
    PID_WIN=$(cygpath -w "$PID_FILE")
    WD_WIN=$(cygpath -w "$PWD")

    # Launch using `uv run`.
    powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'uv' \
     -ArgumentList @('run', '--env-file', \"$ENV_FILE_WIN\", '--python', \"$PYTHON_WIN\", \"$SCRIPT_WIN\", '--env', \"$ENV\", '--task', \"$TASK\", '--start', \"$START\", '--end', \"$END\") \
     -WorkingDirectory \"$WD_WIN\" \
     -RedirectStandardOutput \"$LOG_WIN\" -RedirectStandardError \"$ERR_WIN\" \
     -WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id | Set-Content -Path \"$PID_WIN\""

    echo "Process started successfully in the background!"
    echo "Logs: $LOG_OUT"
    echo "PID saved to: $PID_FILE"
}

check() {
    echo "Running check for: ENV=$ENV | START=$START | END=$END"
    echo "Using .env file at: $ENV_FILE_PATH"

    # Compute Windows paths for PowerShell
    PYTHON_WIN=$(cygpath -w ./ISA4RL-article/src/main/.venv/Scripts/python.exe)
    SCRIPT_WIN=$(cygpath -w ./ISA4RL-article/src/main/main.py)
    ENV_FILE_WIN=$(cygpath -w "$ENV_FILE_PATH")
    WD_WIN=$(cygpath -w "$PWD")

    # Run the check task synchronously in the foreground so you can immediately see the output
    powershell.exe -NoProfile -Command \
    "Set-Location -Path \"$WD_WIN\"; uv run --env-file \"$ENV_FILE_WIN\" --python \"$PYTHON_WIN\" \"$SCRIPT_WIN\" --env \"$ENV\" --task 'check' --start \"$START\" --end \"$END\""
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID_WIN=$(cygpath -w "$PID_FILE")
        echo "Checking process status for $TASK | $ENV ($START to $END)..."
        powershell.exe -NoProfile -Command "Get-Process -Id (Get-Content \"$PID_WIN\")"
    else
        echo "Status: PID file not found for these parameters. The process is likely not running."
    fi
}

logs() {
    echo "Tailing logs for $TASK | $ENV ($START to $END)... (Press Ctrl+C to stop)"
    touch "$LOG_OUT" "$LOG_ERR"
    tail -f "$LOG_OUT" "$LOG_ERR"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID_WIN=$(cygpath -w "$PID_FILE")
        echo "Stopping process for $TASK | $ENV ($START to $END)..."
        powershell.exe -NoProfile -Command "Stop-Process -Id (Get-Content \"$PID_WIN\") -Force"

        rm "$PID_FILE"
        echo "Process stopped and PID file removed."
    else
        echo "PID file not found. Cannot stop the process (it might have already finished)."
    fi
}

# Route the command
case "$COMMAND" in
    start) start ;;
    check) check ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
esac
