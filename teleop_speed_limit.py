import yaml
import numpy as np
import rospy
from geometry_msgs.msg import Twist

vel_msg = Twist()
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
 
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_teleop_topic"]

teleop_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)

def callback_teleop(vel):
    global vel_msg
    vel_msg.linear.x = min(vel.linear.x, MAX_V)
    vel_msg.linear.x = max(vel_msg.linear.x, -MAX_V)
    vel_msg.angular.z = min(vel.angular.z, MAX_W)
    vel_msg.angular.z = max(vel_msg.angular.z, -MAX_W)
    teleop_pub.publish(vel_msg)

def main():
    rospy.init_node("Teleop_Speed_Limit", anonymous=False)
    teleop_sub = rospy.Subscriber("cmd_vel", Twist, callback_teleop)
    rospy.spin()

if __name__ == '__main__':
    main()
        