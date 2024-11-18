import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ctypes
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import os
import sys

from topic_names import DEPTH_CAMERA_RGB_TOPIC, DEPTH_CAMERA_DEPTH_TOPIC


class ImageConverter:
    def __init__(self):
        self.folder_path = os.path.join("../topomaps/rgbd/", sys.argv[1])
        print(self.folder_path)
        os.makedirs(self.folder_path, exist_ok=True)
        self.bridge = CvBridge()
        self.image_count = 0
        self.color_sub = rospy.Subscriber(DEPTH_CAMERA_RGB_TOPIC, Image, self.color_callback)
        self.depth_sub = rospy.Subscriber(DEPTH_CAMERA_DEPTH_TOPIC, Image, self.depth_callback)
        # self.pointcloud_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pointcloud_callback)
        self.rgb_image = None
        self.depth_image = None
        self.save_images_flag = False 
        self.pointcloud_count = 0
        self.save_pointcloud_flag = False
        self.saved = False

    def save_image(self, image, filename):
        cv2.imwrite(filename, image)
        print(f"Image saved as {filename}")

    def color_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data)

        # cv2.imshow("Color Image", self.rgb_image)

        if self.saved == False and self.rgb_image is not None and self.depth_image is not None:
            self.save_image(self.rgb_image, os.path.join(self.folder_path, "rgb_target.png"))
            self.save_image(self.depth_image, os.path.join(self.folder_path, "depth_target.png"))
            self.saved = True

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except:
            self.depth_image = self.bridge.imgmsg_to_cv2(data)
        
            
    def save_pointcloud(self, pointcloud, filename):
        o3d.io.write_point_cloud(filename, pointcloud)
        print(f"Point cloud saved as {filename}")
        # 可视化点云
        o3d.visualization.draw_geometries([pointcloud], window_name="PointCloud", width=800, height=600)

    def pointcloud_callback(self, data):
        # 将ROS点云消息转换为Open3D点云对象
        pointcloud = self.convert_ros_to_open3d(data)

        if self.save_pointcloud_flag:
            pcd_filename = os.path.join(self.folder_path, "pointcloud.pcd")
            self.save_pointcloud(pointcloud, pcd_filename)
            self.pointcloud_count += 1
            self.save_pointcloud_flag = False

    def convert_ros_to_open3d(self, ros_pointcloud):
        # 从ROS点云消息中提取点云数据
        points_list = []
        for point in pc2.read_points(ros_pointcloud, skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        # 创建Open3D点云对象
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points_list)
        return pointcloud

def main():
  ic = ImageConverter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main()