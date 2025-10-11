import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.executors import MultiThreadedExecutor
from datetime import datetime as time
def load_calibration_data(file_path):
    with open(file_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    return calib_data

class BagPlayer(Node):
    def __init__(self):
        super().__init__('bag_player')
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)  # Publicar cada 0.1 segundos
        self.reader = SequentialReader()
        try:
            self.reader.open(StorageOptions(uri='V1_01_easy'), ConverterOptions( input_serialization_format='cdr', output_serialization_format='cdr'))
        except Exception as e:
            self.get_logger().error(f'Failed to open bag file: {e}')
            # rclpy.shutdown()
            return
        self.remap_topics = {
            '/cam0/image_raw': self.create_publisher(Image, '/stereo/left/image_raw', 10),
            '/cam1/image_raw': self.create_publisher(Image, '/stereo/right/image_raw', 10),
        }
        self.timer = self.create_timer(0.1, self.timer_callback)
  
    def timer_callback(self):
        while self.reader.has_next():
            (topic, raw_data, _) = self.reader.read_next()
            if topic in self.remap_topics.keys():
                self.remap_topics[topic].publish(raw_data)
                break  # Publicar solo un mensaje por llamada al timer
            else:
                continue

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
        
        # Publishers
        self.pub_left_rect = self.create_publisher(Image, '/stereo/left/image_rect', 10)
        self.pub_right_rect = self.create_publisher(Image, '/stereo/right/image_rect', 10)
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


class KeypointDetector(Node):
    def __init__(self):
        super().__init__('keypoint_detector')
        self.bridge = CvBridge()
        self.orb = cv.ORB_create()
        self.subscription_left = self.create_subscription(
            Image,
            '/stereo/left/image_rect',
            self.left_callback, 10)
        self.subscription_right = self.create_subscription(
            Image,
            '/stereo/right/image_rect',
            self.right_callback, 10)
        self.remap_topics = {
            'left': self.create_publisher(Image, '/stereo/left/image_keypoints', 10),
            'right': self.create_publisher(Image, '/stereo/right/image_keypoints', 10),
        }
        self.printed = {"left": False, "right": False}

    def left_callback(self, msg):
        self.listener_callback(msg, "left")

    def right_callback(self, msg):
        self.listener_callback(msg, "right")

    def listener_callback(self, msg, name):
        if not self.printed[name]:
            print("Processing:", name)
            self.printed[name] = True
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Obtener los keypoints de la imagen
        keypoints, descriptors = self.orb.detectAndCompute(img,None)

        # Dibujar los puntos en la imagen
        output_image = cv.drawKeypoints(img, keypoints,None,color=(0,255,0))
        # output_image = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        key_msg.header = msg.header
        self.remap_topics[name].publish(key_msg)
            
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    print("Starting bag player")
    bag_player = BagPlayer()
    executor = MultiThreadedExecutor()
    executor.add_node(bag_player)
    print("Starting image rectifier")
    rectifier = ImageRectifier()
    executor.add_node(rectifier)
    print("Starting keypoint detector")
    keyDetector = KeypointDetector()
    executor.add_node(keyDetector)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bag_player.destroy_node()
        rectifier.destroy_node()
        keyDetector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()