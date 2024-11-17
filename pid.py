import rospy
import tf
import math
from geometry_msgs.msg import Pose, Twist, PoseStamped

class PIDController:
    def __init__(self):
        rospy.init_node('pid_controller', anonymous=True)

        # 订阅目标位置
        # rospy.Subscriber('/goal_pose', Pose, self.goal_callback)
        # rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_stamped_callback)
        rospy.Subscriber('/goal_pose', PoseStamped, self.goal_stamped_callback)

        # 发布速度命令
        self.velocity_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

        # PID参数
        self.kp_linear = 0.35
        self.kp_angular = 1.0
        
        self.ki_linear = 0.007  # 位置的积分增益
        self.ki_angular = 0.05  # 角度的积分增益

        # 积分项累加器
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_yaw = 0.0


        # 初始化变量
        self.goal = None
        self.tf_listener = tf.TransformListener()

    def goal_callback(self, msg):
        self.goal = msg
    
    def goal_stamped_callback(self, msg):
        self.goal = msg.pose

    def get_current_pose(self):
        try:
            print("Getting current pose")
            # 获取base_link相对于map的变换
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            current_pose = Pose()
            current_pose.position.x = trans[0]
            current_pose.position.y = trans[1]
            current_pose.orientation.x = rot[0]
            current_pose.orientation.y = rot[1]
            current_pose.orientation.z = rot[2]
            current_pose.orientation.w = rot[3]
            print(current_pose)
            return current_pose
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get transform from map to base_link")
            self.goal = None
            self.velocity_pub.publish(Twist())
            return None

    def get_yaw(self, orientation):
        # 从四元数获取偏航角
        euler = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        return euler[2]

    def move_to_goal(self):
        rate = rospy.Rate(10)  # 10 Hz
        print("Starting to move to goal")
        self.velocity_pub.publish(Twist())
        while not rospy.is_shutdown():
            if self.goal:
                # 获取当前位姿
                current_pose = self.get_current_pose()
                if current_pose is None:
                    continue

                # 阶段1：移动到目标位置
                error_x = self.goal.position.x - current_pose.position.x
                error_y = self.goal.position.y - current_pose.position.y
                distance = math.sqrt(error_x**2 + error_y**2)

                # 更新积分误差
                self.integral_x += error_x
                self.integral_y += error_y

                if distance > 0.03:
                    # 使用积分项控制x和y方向的速度
                    twist = Twist()
                    twist.linear.x = self.kp_linear * error_x + self.ki_linear * self.integral_x
                    if (twist.linear.x > 0.2): twist.linear.x = 0.2
                    elif (twist.linear.x < -0.2): twist.linear.x = -0.2
                    elif (0 < twist.linear.x < 0.08): twist.linear.x = 0.08
                    elif (-0.08 < twist.linear.x < 0): twist.linear.x = -0.08
                    twist.linear.y = self.kp_linear * error_y + self.ki_linear * self.integral_y
                    if (twist.linear.y > 0.2): twist.linear.y = 0.2
                    elif (twist.linear.y < -0.2): twist.linear.y = -0.2
                    elif (0 < twist.linear.y < 0.08): twist.linear.y = 0.08
                    elif (-0.08 < twist.linear.y < 0): twist.linear.y = -0.08
                    # elif (abs(twist.linear.y) < 0.1): twist.linear.y = 0 
                    self.velocity_pub.publish(twist)
                else:
                    # 阶段2：在目标位置进行转向
                    goal_yaw = self.get_yaw(self.goal.orientation)
                    current_yaw = self.get_yaw(current_pose.orientation)
                    yaw_error = goal_yaw - current_yaw

                    # 更新积分误差
                    self.integral_yaw += yaw_error

                    if abs(yaw_error) > 0.05:
                        twist = Twist()
                        twist.angular.z = self.kp_angular * yaw_error + self.ki_angular * self.integral_yaw
                        if (twist.angular.z > 0.37): twist.angular.z = 0.37
                        elif (twist.angular.z < -0.37): twist.angular.z = -0.37
                        elif (0 < twist.angular.z < 0.3): twist.angular.z = 0.3
                        elif (-0.3 < twist.angular.z < 0): twist.angular.z = -0.3
                        self.velocity_pub.publish(twist)
                    else:
                        # 已达到目标位置和朝向，停止机器人
                        self.velocity_pub.publish(Twist())
                        return
            else: self.velocity_pub.publish(Twist()) 
            rate.sleep()

if __name__ == '__main__':
    controller = PIDController()
    controller.move_to_goal()
