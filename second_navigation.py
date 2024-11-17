import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
import tf
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import TransformStamped
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped, Pose
import tf.transformations
import argparse
import os
import sys
from topic_names import DEPTH_CAMERA_RGB_TOPIC, DEPTH_CAMERA_DEPTH_TOPIC


TOPOMAP_IMAGES_DIR = "../topomaps/rgbd"



class Fine_navigation:
    def __init__(self, args):
        self.pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=1)
        self.color_sub = rospy.Subscriber(DEPTH_CAMERA_RGB_TOPIC, Image, self.color_callback)
        self.depth_sub = rospy.Subscriber(DEPTH_CAMERA_DEPTH_TOPIC, Image, self.depth_callback)
        self.folder_path = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        self.bridge = CvBridge()
        self.saved = False
        self.visualize = args.visualize

    def get_inverse_transform(self):
        listener = tf.TransformListener()

        # 等待变换可用
        listener.waitForTransform('base_link', 'camera_link', rospy.Time(0), rospy.Duration(4.0))

        # 获取从 base_link 到 camera_link 的变换
        (trans_base_to_camera, rot_base_to_camera) = listener.lookupTransform('camera_link', 'base_link', rospy.Time(0))

        # 计算 camera_link 到 base_link 的变换
        # 注意：tf 中的转换是从源到目标，所以我们交换顺序
        # 即，从 'base_link' 到 'camera_link' 的变换需要反转
        trans_camera_to_base = [-t for t in trans_base_to_camera]
        rot_camera_to_base = tf.transformations.quaternion_inverse(rot_base_to_camera)

        print("Transformation from camera_link to base_link:")
        print("Translation:", trans_camera_to_base)
        print("Rotation:", rot_camera_to_base)
        inversed_transform_matrix = tf.transformations.quaternion_matrix(rot_camera_to_base)
        inversed_transform_matrix[:3, 3] = trans_base_to_camera
        print(inversed_transform_matrix)
        return inversed_transform_matrix

    def get_inverse_transform_odom(self):
        listener = tf.TransformListener()

        # 等待变换可用
        listener.waitForTransform('odom', 'camera_link', rospy.Time(0), rospy.Duration(4.0))

        (trans_odom_to_camera, rot_odom_to_camera) = listener.lookupTransform('camera_link', 'odom', rospy.Time(0))

        # 计算 camera_link 到 base_link 的变换
        # 注意：tf 中的转换是从源到目标，所以我们交换顺序
        # 即，从 'base_link' 到 'camera_link' 的变换需要反转
        trans_camera_to_base = [-t for t in trans_odom_to_camera]
        rot_camera_to_base = tf.transformations.quaternion_inverse(rot_odom_to_camera)

        # print("Transformation from camera_link to odom:")
        # print("Translation:", trans_camera_to_base)
        # print("Rotation:", rot_camera_to_base)
        inversed_transform_matrix = tf.transformations.quaternion_matrix(rot_camera_to_base)
        inversed_transform_matrix[:3, 3] = trans_odom_to_camera
        # print(inversed_transform_matrix)
        return inversed_transform_matrix

    def compute_pose_odom(self, quaternions, translation):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        rot = quaternions
        pose.pose.orientation.x = rot[0]
        pose.pose.orientation.y = rot[1]
        pose.pose.orientation.z = rot[2]
        pose.pose.orientation.w = rot[3]
        pose.pose.position.x = translation[0]
        pose.pose.position.y = translation[1]
        pose.pose.position.z = 0
        return pose    

    def compute_transformation_matrix(self, use_pcd=False):
        print("Computing transformation matrix...")
        while (self.saved == False):
            try:
                print("Waiting for the rgbd images to be saved...")
            except KeyboardInterrupt:
                print("Shutting down")
                sys.exit(0)
        print("folder_path:", self.folder_path)
        source_color_path = os.path.join(self.folder_path, "rgb_source.png")
        source_depth_path = os.path.join(self.folder_path, "depth_source.png")
        target_color_path = os.path.join(self.folder_path, "rgb_target.png")
        target_depth_path = os.path.join(self.folder_path, "depth_target.png")


        # 读取输入图像
        image1 = cv2.imread(source_color_path)  # 参考图像
        image2 = cv2.imread(target_color_path)  # 目标图像

        source_color = o3d.io.read_image(source_color_path)
        source_depth = o3d.io.read_image(source_depth_path)
        target_color = o3d.io.read_image(target_color_path)
        target_depth = o3d.io.read_image(target_depth_path)
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth)
        intrinsic_matrix = [[748.5086059570312, 0.0, 630.3563842773438],
                            [0.0, 748.41845703125, 357.4473571777344],
                            [0.0, 0.0, 1.0]]
        width = 1280
        height = 720
        intrinsic_matrix_ = [748.5086059570312, 0.0, 630.3563842773438,
                            0.0, 748.41845703125, 357.4473571777344,
                            0.0, 0.0, 1.0]

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.set_intrinsics(width, height, intrinsic_matrix_[0], intrinsic_matrix_[4], intrinsic_matrix_[2], intrinsic_matrix_[5])

        if (use_pcd):
            target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                target_rgbd_image, pinhole_camera_intrinsic)
            source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                source_rgbd_image, pinhole_camera_intrinsic)

        # 初始化 ORB 特征提取器
        orb = cv2.ORB_create()
        orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        # 提取关键点和描述符
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
        fx = intrinsic_matrix[0][0]
        fy = intrinsic_matrix[1][1]
        cx = intrinsic_matrix[0][2]
        cy = intrinsic_matrix[1][2]

        # 使用BFMatcher进行特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # 仅保留好的匹配
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:80]

        # 提取匹配的关键点
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        print(src_pts)
        print((src_pts).astype(int))
        depth_map = cv2.imread(source_depth_path, cv2.IMREAD_UNCHANGED)

        # image_points = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 2)
        print(source_depth)

        # 将2D图像点转换为3D点
        valid_points = []
        object_points = []
        invalid_indices = []
        # for u in range(depth_map.shape[1]):
        #     for v in range(depth_map.shape[0]):
        for pt in src_pts:
                u, v = int(pt[0]), int(pt[1])
                valid_points.append([u, v])
                depth = depth_map[v, u]  # 获取对应的深度值

                # # 深度值有效性检查，跳过无效深度值
                if depth == 0:
                    invalid_indices.append(len(valid_points) - 1)
                    continue

                # 将深度值转换为米（假设深度图的单位是毫米）
                depth = depth / 1000.0

                # 使用相机内参将2D点转换为3D点
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                object_points.append([x, y, z])

        object_points = np.array(object_points, dtype=np.float32)

        # 创建Open3D点云对象Q
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        valid_dst_pts = np.delete(dst_pts, invalid_indices, axis=0)

        # 可视化
        # o3d.visualization.draw_geometries([source_pcd, pcd],
        #                                     zoom=0.8,
        #                                     front=[0.5, 0.5, 0.5],
        #                                     lookat=[0, 0, 0],
        #                                     up=[0, 1, 0])

        # 假设相机内参矩阵
        camera_matrix = np.array(intrinsic_matrix, dtype=np.float32)

        # 假设相机的畸变系数（如果已知）
        dist_coeffs = [0.0761912539601326, -0.10373377054929733, 2.3656342818867415e-05, -0.00010011430276790634, 0.042324796319007874, 0.0, 0.0, 0.0]
        dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

        print(object_points.shape)
        print(valid_dst_pts.shape)
        # 使用solvePnP计算位姿
        # ret, rvec, tvec = cv2.solvePnP(object_points, dst_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, valid_dst_pts, camera_matrix, dist_coeffs)
        rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers], valid_dst_pts[inliers], camera_matrix, dist_coeffs, rvec, tvec)


        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 输出位姿
        # print(inliners)
        print("Rotation Matrix:\n", rotation_matrix)
        print("Translation Vector:\n", tvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.T
        print("Transformation Matrix:\n", transformation_matrix)
        # 可视化匹配
        if (self.visualize):
            matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)
            cv2.imshow('ORB Feature Matching', matched_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        target_pcd_store = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_store = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_orb_term = source_pcd_store.transform(transformation_matrix)
        target_pcd_orb_term = target_pcd_store.transform(transformation_matrix)
        if self.visualize:
            o3d.visualization.draw_geometries([source_pcd_store, target_pcd_store],
                                                zoom=0.48,
                                                front=[0.0999, -0.1787, -0.9788],
                                                lookat=[0.0345, -0.0937, 1.8033],
                                                up=[-0.0067, -0.9838, 0.1790])
            # o3d.visualization.draw_geometries([source_pcd_orb_term],
            #                                     zoom=0.48,
            #                                     front=[0.0999, -0.1787, -0.9788],
            #                                     lookat=[0.0345, -0.0937, 1.8033],
            #                                     up=[-0.0067, -0.9838, 0.1790])
            # o3d.visualization.draw_geometries([target_pcd_orb_term, source_pcd_store],
            #                                     zoom=0.48,
            #                                     front=[0.0999, -0.1787, -0.9788],
            #                                     lookat=[0.0345, -0.0937, 1.8033],
            #                                     up=[-0.0067, -0.9838, 0.1790])
                                                
            o3d.visualization.draw_geometries([source_pcd_orb_term, target_pcd_store],
                                                zoom=0.48,
                                                front=[0.0999, -0.1787, -0.9788],
                                                lookat=[0.0345, -0.0937, 1.8033],
                                                up=[-0.0067, -0.9838, 0.1790])
        # transformation_matrix[:3, 1] = [0, 1, 0]
        # transformation_matrix[:3, 0] = [1, 0, 0]
        # transformation_matrix[:3, 2] = [0, 0, 1]
        # transformation_matrix[2, 3] = 0.0                                    
        
        return transformation_matrix

    def opencv_to_ros_transform(self, transformation_matrix):
        # Define the transformation matrix from OpenCV to ROS coordinate system
        opencv_to_ros = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32)
        # Apply the transformation
        ros_transformation_matrix = np.dot(opencv_to_ros, transformation_matrix)
        return ros_transformation_matrix

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
            self.save_image(self.rgb_image, os.path.join(self.folder_path, "rgb_source.png"))
            self.save_image(self.depth_image, os.path.join(self.folder_path, "depth_source.png"))
            self.saved = True

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except:
            self.depth_image = self.bridge.imgmsg_to_cv2(data)

    def compute_target_pose(self):
        transformation_matrix = self.compute_transformation_matrix()  # Replace with your actual transformation matrix
        print(111)
        ros_transformation_matrix = self.opencv_to_ros_transform(transformation_matrix)
        # transformation_matrix[:,1] = [0, 1, 0, 0]  # Set the y-axis to point upwards
        # transformation_matrix[2, 3] = 0.0
        print(ros_transformation_matrix)
        euler_angles = tf.transformations.euler_from_matrix(ros_transformation_matrix)
        print("Euler Angles (radians):", euler_angles)
        
        quaternions = tf.transformations.quaternion_from_euler(0.0, 0.0, euler_angles[1])
        translation = [-ros_transformation_matrix[2, 3], ros_transformation_matrix[0, 3], 0.0]
        # Create a TF broadcaster
        inverse_transform = self.get_inverse_transform() 
        br1 = tf.TransformBroadcaster()
        br2 = tf.TransformBroadcaster()
        br1.sendTransform(translation,
                            quaternions,
                            rospy.Time.now(),
                            "target_camera_link_pose",  # Target frame name
                            "camera_link")  # Source frame name
        br2.sendTransform((inverse_transform[0, 3], inverse_transform[1, 3], inverse_transform[2, 3]),
                            tf.transformations.quaternion_from_matrix(inverse_transform),
                            rospy.Time.now(),
                            "target_pose",  # Target frame name
                            "target_camera_link_pose")
        target_pose = self.compute_pose_odom(quaternions, translation)
        

        
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Publish the transformation
            # br1.sendTransform((ros_transformation_matrix[0, 3], ros_transformation_matrix[1, 3], ros_transformation_matrix[2, 3]),
            #                  normalize_quaternion(tf.transformations.quaternion_from_matrix(ros_transformation_matrix)),
            #                  rospy.Time.now(),
            #                  "target_camera_link_pose",  # Target frame name
            #                  "camera_link")  # Source frame name

            br1.sendTransform(translation,
                            quaternions,
                            rospy.Time.now(),
                            "target_camera_link_pose",  # Target frame name
                            "camera_link")  # Source frame name
            br2.sendTransform((inverse_transform[0, 3], inverse_transform[1, 3], inverse_transform[2, 3]),
                            tf.transformations.quaternion_from_matrix(inverse_transform),
                            rospy.Time.now(),
                            "target_pose",  # Target frame name
                            "target_camera_link_pose")
            self.pose_pub.publish(target_pose)
            rate.sleep()
        

def main(args):
    rospy.init_node('orb_transform_node', anonymous=True)
    fn = Fine_navigation(args)
    try:
        fn.compute_target_pose()
    except KeyboardInterrupt:
        print("Shutting down")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="publish the transform to target pose")
    parser.add_argument(
        "--visualize",
        "-v",
        default=False,
        type=bool,
        help="whether to visualize the ORB feature matching"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="rgbd",
        type=str,
        help="the directory of the rgbd images"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    args = parser.parse_args()
    main(args)
