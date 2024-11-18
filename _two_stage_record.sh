first_process() {
    echo "Running first process..."
    cd ../topomaps/bags
    echo $1
    # rosbag record /usb_cam/image_raw -o $1 # change topic if necessary
    rosbag record /usb_cam/image_raw -o $1 # change topic if necessary

}

second_process() {
    echo "First process terminated, starting second process..."
    cd ../../visualnav_src
    python ./image_storer.py $1
}

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
trap 'second_process $1' SIGINT

# 启动第一个进程
first_process $1
