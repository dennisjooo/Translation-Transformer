#!/bin/bash

# Create Python virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating Python virtual environment..."
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
else
    echo "Python environment exists, activating..."
    source env/bin/activate
fi

# Download the dataset
python download_data.py

# Train the tokenizers
python train_tokenizer.py

# Create a unique session name using timestamp
SESSION_NAME="training_$(date +%Y%m%d_%H%M%S)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Store the log file path
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Start a new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Send commands to the tmux session
tmux send-keys -t "$SESSION_NAME" "source env/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Starting training at $(date)'" C-m

# Run training with output logged
tmux send-keys -t "$SESSION_NAME" "python train.py > '$LOG_FILE' 2>&1" C-m

# Attach to the tmux session
echo "Starting training in tmux session: $SESSION_NAME"
echo "To detach from the session: press Ctrl+B, then D"
echo "To reattach later: tmux attach-session -t $SESSION_NAME"
tmux attach-session -t "$SESSION_NAME"

# Note: The session will continue running even if you detach or close your terminal
# To list all sessions: tmux ls
# To kill the session: tmux kill-session -t $SESSION_NAME
