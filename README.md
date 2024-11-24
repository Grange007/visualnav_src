# Visualnav-src

This is the repository to integrate the fine-navigation with [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer).

## Install

First, follow the **software installation** in **Deployment** of [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer).

You will have a conda env called `vint_deployment`.

```shell
conda activate vint_deployment
```

In `vint_deployment`, install open3D:

```shell
pip install open3d
```

Install [rtabmap_ros](https://github.com/introlab/rtabmap_ros) for visual localization:

```shell
sudo apt install ros-$ROS_DISTRO-rtabmap-ros
```

## Deployment

### Hardware setup

First, launch the B1 Base and Realsense camera.

Remember to change the RGB and Depth image topic names in `topic_names.py` according to the Realsense camera:

```python
DEPTH_CAMERA_RGB_TOPIC = "/camera/color/image_raw" # change according to the Realsense camera
DEPTH_CAMERA_DEPTH_TOPIC = "/camera/depth/image_raw" # change according to the Realsense camera
```

### Collecting a Topological Map

_Make sure to run these scripts inside the `vint_release/deployment/src/` directory._

#### Record the rosbag:

```bash
./_record_bag.sh <bag_name>
```

Run this command to teleoperate the robot with the joystick and camera. This command opens up two windows:

1. `roslaunch vint_locobot.launch`: This launch file opens the `usb_cam` node for the camera, the joy node for the joystick, and nodes for the robot’s mobile base.
   Before launching `vint_locobot.launch`, you will first need to source where your `yocs_cmd_vel_mux` lies. You should modify `tmux send-keys "source /home/grange/Program/naviwhere/detic/devel/setup.sh" Enter` manually according to your folder structure.
2. `rosbag record /usb_cam/image_raw -o <bag_name>`:  Unlike the original `record_bag.sh`, this command **runs immediately**. It will be run in the vint_release/deployment/topomaps/bags directory, where we recommend you store your rosbags.

After finishing the recording, use `Ctrl+C` in the panel of `rosbag record /usb_cam/image_raw -o <bag_name>`.  The panel will then run `python ./image_storer.py`. It will record the rgb and depth image of its current view in `deployment/topomaps/rgbd/<bag_name>/`.

#### Make the topological map:

```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

This command opens up 3 windows:

1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/vint_release/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 1.5 <bag_filename>`: This command plays the rosbag at x5 speed, so the python script is actually recording nodes 1.5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension. You can change this value in the `make_topomap.sh` file. The command does not run until you hit Enter, which you should only do once the python script gives its waiting message. Once you play the bag, move to the screen where the python script is running so you can kill it when the rosbag stops playing.

When the bag stops playing, kill the tmux session.

### Navigation

_Make sure to run this script inside the `vint_release/deployment/src/` directory._

```bash
./_navigate.sh “--model <model_name> --dir <topomap_dir>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).

Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_locobot.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base).
   Before launching `vint_locobot.launch`, you will first need to source where your `yocs_cmd_vel_mux` lies. You should modify `tmux send-keys "source /home/grange/Program/naviwhere/detic/devel/setup.sh" Enter` manually according to your folder structure.
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic.
3. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.

When the robot has finished its first-stage navigating,  `pd_controller.py` will automatically stop. Then it wil open three terminal windows:

1. `roslaunch pure_odom.launch`: This launch file starts rtab-map to provide visual-odometry for localization of the car.
   Remember to change the `rgb/image`, `depth/image`, `rgb/camera_info` here to the topics according to Realsense camera.
   ```xml
       <node pkg="nodelet" type="nodelet" name="rgbd_sync" args="standalone rtabmap_sync/rgbd_sync" output="screen">
         <remap from="rgb/image"        to="/camera/color/image_raw"/>
         <remap from="depth/image"      to="/camera/depth/image_raw"/>
         <remap from="rgb/camera_info"  to="/camera/color/camera_info"/>
         <remap from="rgbd_image"       to="rgbd_image"/> <!-- output -->

         <!-- Should be true for not synchronized camera topics 
              (e.g., false for kinectv2, zed, realsense, true for xtion, kinect360)-->
         <param name="approx_sync"       value="true"/> 
       </node>
   ```
2. `source ~/.zshrc; conda activate vint_deployment; python3 second_navigation.py`: This script stores the rgb and depth image of its current view in `deployment/topomaps/rgbd/<bag_name>/` and publishes the transform from current pose to target pose.
3. `source ~/.zshrc; conda activate vint_deployment; python pid.py`: This script controls the base to move to target pose with PID controller.

For some reasons, I have to use `gnome-terminal -- zsh -c "source ~/.zshrc; conda activate vint_deployment"` to open a new terminal window. If you use `bash` instead of `zsh`, change `zsh` here to `bash` and `.zshrc` to `.bashrc`.

### Visualize trajectory

The trajectory visualization hasn't been integrated with the fine-tuning navigation. It only involves the first stage of navigation.

To visualize the trajectory, you can record & create topomap as before:

#### Record the rosbag:

```bash
./record_bag.sh <bag_name>
```

#### Make the topological map:

```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

#### Navigation with visualization

Then, you should use `navigation_visualize.sh` instead of `navigation.sh`:

```
./navigate_visualize.sh “--model <model_name> --dir <topomap_dir>”
```

Or you can choose to visualize all the proposed trajectories by add  param '`-v'`:

```
./navigate_visualize.sh “--model <model_name> --dir <topomap_dir> -v”
```

You can see the visualized trajectories in Rviz by adding topic `/waypoint_img`
