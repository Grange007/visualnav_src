
#!/bin/bash

# Create a new tmux session
session_name="second_navigation$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "roslaunch pure_odom.launch" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
# tmux send-keys "conda activate vint_deployment" Enter
sleep 8
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 second_navigation.py" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 2
sleep 12
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 pid.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
