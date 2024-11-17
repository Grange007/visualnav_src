#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 30 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 70 # split it into two halves
tmux selectp -t 3
tmux splitw -v -p 50
tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 0    # go back to the first pane


# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ../../../detic/devel/setup.sh" Enter
tmux send-keys "roslaunch vint_locobot.launch" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 navigate.py $1 $2 $3 $4" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 2
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 pd_controller.py" Enter

tmux select-pane -t 4
tmux send-keys "cd ../topomaps/records" Enter
tmux send-keys "rosbag record /usb_cam/image_raw -o $2_$4" # change topic if necessary

tmux select-pane -t 5
tmux send-keys "python3 teleop_speed_limit.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
