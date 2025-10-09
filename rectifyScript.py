import cv2 as cv
import numpy as np
import yaml
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

def load_calibration_data(file_path):
    with open(file_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    return calib_data

class ImageRectifier(Node):
    def __init__(self):
        super().__init__('image_rectifier')
        
        self.bridge = CvBridge()
        
        # Load calibration data
        calib_data_left = load_calibration_data(f'calibrationdata/left.yaml')
        calib_data_right = load_calibration_data(f'calibrationdata/right.yaml')
        
        self.K1 = np.array(calib_data_left['camera_matrix']['data']).reshape(3, 3)
        self.D1 = np.array(calib_data_left['distortion_coefficients']['data'])
        self.K2 = np.array(calib_data_right['camera_matrix']['data']).reshape(3, 3)
        self.D2 = np.array(calib_data_right['distortion_coefficients']['data'])
        auxMatrix = np.array(calib_data_right['projection_matrix']['data']).reshape(3, 4)
        self.R = auxMatrix[:, :3]
        self.T = auxMatrix[:, 3]

        self.image_size = (calib_data_left["image_width"], calib_data_left["image_height"])

        # Compute rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, 
            self.image_size, self.R, self.T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Precompute the undistortion and rectification maps
        self.map1x, self.map1y = cv.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, 
            self.image_size, cv.CV_32FC1
        )
        self.map2x, self.map2y = cv.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, 
            self.image_size, cv.CV_32FC1
        )
        
        # Subscribers
        self.sub_left = self.create_subscription(
            Image,
            '/stereo/left/image_raw',
            self.left_image_callback,
            10
        )
        
        self.sub_right = self.create_subscription(
            Image,
            '/stereo/right/image_raw',
            self.right_image_callback,
            10
        )
        
        # Publishers
        self.pub_left_rect = self.create_publisher(Image, '/stereo/left/image_rect', 10)
        self.pub_right_rect = self.create_publisher(Image, '/stereo/right/image_rect', 10)

    def left_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rectified_image = cv.remap(cv_image, self.map1x, self.map1y, interpolation=cv.INTER_LINEAR)
        rect_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rect_msg.header = msg.header
        self.pub_left_rect.publish(rect_msg)

    def right_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rectified_image = cv.remap(cv_image, self.map2x, self.map2y, interpolation=cv.INTER_LINEAR)
        rect_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rect_msg.header = msg.header
        self.pub_right_rect.publish(rect_msg)

def main(args=None):
    rclpy.init(args=args)
    
    rectifier = ImageRectifier()
    
    rclpy.spin(rectifier)
    
    rectifier.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()