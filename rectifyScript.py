import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.executors import MultiThreadedExecutor
from datetime import datetime as time
import struct
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
            (topic, raw_data, stamp) = self.reader.read_next()
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


class Matcher(Node):
    def __init__(self, triangulation):
        super().__init__('matcher')
        self.bridge = CvBridge()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.publisher = self.create_publisher(Image, '/stereo/matched_keypoints', 10)
        self.info = {}
        self.triangulation = triangulation
        
        
    def set_info(self, info, side):
        stamp = info['header'].stamp
        stamp_obj = (stamp.sec, stamp.nanosec)
        if stamp_obj not in self.info:
            self.info[stamp_obj] = {}
        self.info[stamp_obj][side] = info
        self.match_keypoints(stamp_obj)

    def match_keypoints(self, stamp):
        # print("side: ", "left" if self.left_info is not None else "right")
        if stamp in self.info and "left" in self.info[stamp] and "right" in self.info[stamp]:
            print("Matching keypoints between left and right images")
            left = self.info[stamp]["left"]
            right = self.info[stamp]["right"]
            matches = self.bf.match(left['descriptors'], right['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            self.triangulation.compute(matches, left, right)
            matched_image = cv.drawMatches(
                left['image'], left['keypoints'],
                right['image'], right['keypoints'],
                matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            match_msg = self.bridge.cv2_to_imgmsg(matched_image, encoding='bgr8')
            match_msg.header = left['header']
            match_msg.header.frame_id = "matched_keypoints"
            self.publisher.publish(match_msg)
            # Reset images after processing
            # cv.imshow("Matched Keypoints", matched_image)
            # cv.waitKey(1)
            

class KeypointDetector(Node):
    def __init__(self, side, matcher):
        super().__init__('keypoint_detector')
        self.bridge = CvBridge()
        self.orb = cv.ORB_create()
        self.subscription_left = self.create_subscription(
            Image,
            f'/stereo/{side}/image_rect',
            self.listener_callback, 10)
        self.publisher = self.create_publisher(Image, f'/stereo/{side}/image_keypoints', 10)
        self.side = side
        self.matcher = matcher

    def listener_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Obtener los keypoints de la imagen
        keypoints, descriptors = self.orb.detectAndCompute(img,None)

        # Enviar la información al matcher
        info = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': img,
            'header': msg.header
            # 'stamp': msg.header.stamp
        }
        self.matcher.set_info(info, self.side)

        # Dibujar los puntos en la imagen
        output_image = cv.drawKeypoints(img, keypoints,None,color=(0,255,0))
        # output_image = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        key_msg.header = msg.header
        self.publisher.publish(key_msg)
            
        # cv.waitKey(1)

    
    
class TriangulatePoints(Node):
    def __init__(self):
        super().__init__('triangulate_points')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PointCloud2, '/stereo/triangulated_points', 10)
        calib_data_left = load_calibration_data(f'calibrationdata/left.yaml')
        calib_data_right = load_calibration_data(f'calibrationdata/right.yaml')
        projection_left = np.array(calib_data_left['projection_matrix']['data']).reshape(3, 4)
        projection_right = np.array(calib_data_right['projection_matrix']['data']).reshape(3, 4)
        self.projections = np.array([projection_left, projection_right])
    
    def publish(self, points, header):
        field = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        buffer = b''.join([struct.pack('fff',*p) for p in points])

        msg = PointCloud2()
        msg.width = points.shape[0]
        msg.height = 1
        msg.point_step = 12
        msg.row_step = 12 * points.shape[0]
        msg.header = header
        msg.header.frame_id = "map"
        msg.data = buffer
        msg.fields = field
        msg.is_bigendian = False
        msg.is_dense = False
        self.publisher.publish(msg)
    
    def compute(self, matches, left_info, right_info):
        pts_left = np.float32([left_info['keypoints'][m.queryIdx].pt for m in matches])
        pts_right = np.float32([right_info['keypoints'][m.trainIdx].pt for m in matches])
        triangulatedPoints = cv.triangulatePoints(self.projections[0], self.projections[1], pts_left.T, pts_right.T)
        triangulatedPoints = (triangulatedPoints[:3] / triangulatedPoints[3]).T
        
        self.publish(triangulatedPoints, left_info['header'])
        
        # Publicar los puntos 3D como PointCloud2

class TriangulatedPointsRansac(TriangulatePoints):
    def __init__(self):
        super().__init__()
        self.publisherImg = self.create_publisher(Image, '/stereo/transformed_image', 10)
    
    def compute(self, matches, left_info, right_info):
        pts_left = np.float32([left_info['keypoints'][m.queryIdx].pt for m in matches])
        pts_right = np.float32([right_info['keypoints'][m.trainIdx].pt for m in matches])
        # Aplicar RANSAC para eliminar outliers
        H, mask = cv.findHomography(pts_left, pts_right, cv.RANSAC, 5.0)
        pts_left = pts_left[mask.ravel().astype(bool)]
        pts_right = pts_right[mask.ravel().astype(bool)]
        triangulatedPoints = cv.triangulatePoints(self.projections[0], self.projections[1], pts_left.T, pts_right.T)
        triangulatedPoints = (triangulatedPoints[:3] / triangulatedPoints[3]).T

        pts_left_transformed = cv.perspectiveTransform(pts_left.reshape(-1,1,2), H)

        img_vis = right_info['image'].copy()
        for p in pts_left_transformed:
            cv.circle(img_vis, tuple(np.int32(p[0])), 3, (0,255,0), -1)
        msg = self.bridge.cv2_to_imgmsg(img_vis, encoding='bgr8')
        msg.header = left_info['header']
        self.publisherImg.publish(msg)

        self.publish(triangulatedPoints, left_info['header'])
        
class DisparityMap(Node):
    def __init__(self):
        super().__init__('disparity_map')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/stereo/disparity', 10)
        self.info = {}
        self.stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 5,
            blockSize=11,
            P1=8 * 1 * 11**2,   # 1 instead of 3 for grayscale
            P2=32 * 1 * 11**2,  # 1 instead of 3 for grayscale
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.subscription_left = self.create_subscription(
            Image,
            f'/stereo/left/image_rect',
            self.left_callback, 10)
        self.subscription_right = self.create_subscription(
            Image,
            f'/stereo/right/image_rect',
            self.right_callback, 10)
    def right_callback(self,msg):
        self.set_info(msg, "right")
    def left_callback(self,msg):
        self.set_info(msg, "left")
        
    
    
    def set_info(self, msg, side):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        stamp = msg.header.stamp
        stamp_obj = (stamp.sec, stamp.nanosec)
        if stamp_obj not in self.info:
            self.info[stamp_obj] = {}
        self.info[stamp_obj][side] = img
        self.compute(stamp_obj, msg.header)

    def compute(self, stamp, header):
        if stamp in self.info and "left" in self.info[stamp] and "right" in self.info[stamp]:
            left_img = self.info[stamp]["left"]
            right_img = self.info[stamp]["right"]
            disparity_map = self.stereo.compute(left_img, right_img)
            # disp_normalized = cv.normalize(disparity_map,None,255,0,cv.NORM_MINMAX, cv.CV_8U)
            disp_msg = self.bridge.cv2_to_imgmsg(disparity_map)
            disp_msg.header = header
            self.publisher.publish(disp_msg)
  
def main(args=None):
    rclpy.init(args=args)
    print("Starting bag player")
    bag_player = BagPlayer()
    executor = MultiThreadedExecutor()
    executor.add_node(bag_player)
    print("Starting image rectifier")
    rectifier = ImageRectifier()
    executor.add_node(rectifier)
    print("Starting image disparity")
    dispMap = DisparityMap()
    executor.add_node(dispMap)
    print("starting Triangulation")
    # triangulation = TriangulatePoints()
    triangulation = TriangulatedPointsRansac()
    executor.add_node(triangulation)
    print("Starting matcher")
    matcher = Matcher(triangulation)
    executor.add_node(matcher)
    print("Starting keypoint detector")
    keyDetectorL = KeypointDetector("left", matcher)    
    keyDetectorR = KeypointDetector("right", matcher)
    executor.add_node(keyDetectorL)
    executor.add_node(keyDetectorR)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bag_player.destroy_node()
        rectifier.destroy_node()
        dispMap.destroy_node()
        keyDetectorL.destroy_node()
        keyDetectorR.destroy_node()
        matcher.destroy_node()
        triangulation.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()