#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source /home/grange/Program/naviwhere/detic/devel/setup.sh" Enter
tmux send-keys "roslaunch vint_locobot.launch" Enter

tmux select-pane -t 1
echo $1
tmux send-keys "bash ./_two_stage_record.sh $1" Enter

tmux attach-session -t $session_name
# Attach to the tmux session