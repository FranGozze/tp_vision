import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import sys
import rclpy
import transforms3d
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.executors import MultiThreadedExecutor
from datetime import datetime as time
import struct
from std_msgs.msg import Header
import time

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    return calib_data

ground_truth_camera = []

def get_distance(lastPoint, newPoint):
    distance = np.sqrt((lastPoint[0] - newPoint[0])**2 + (lastPoint[1] - newPoint[1])**2 + (lastPoint[2] - newPoint[2])**2)
    return  distance

def transformPath(x,y,z,qw,qx,qy,qz):
    rotation = transforms3d.quaternions.quat2mat([qw, qx, qy, qz])
    translation = np.array([x, y, z]).reshape((3, 1))
    # Creamos la matriz de rototraslacion
    transform = np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))
    

    return transform

def load_GT_data(file_path='ground_truth.csv'):
    df = pd.read_csv(file_path, header=0)
    xi = load_yaml(f'kalibr_imucam_chain.yaml')
    ImuToCam =  np.array(xi['cam0']['T_imu_cam']).reshape(4,4)
    for index, row in df.iterrows():
        newPoint = np.array([row['px'], row['py'], row['pz']])
        if index == 0:
            lastPoint = newPoint
        else:
            lastPoint = ground_truth_camera[-1][0]
        distance = get_distance(lastPoint, newPoint)
        transform = transformPath(row['px'], row['py'], row['pz'], row['qw'], row['qx'], row['qy'], row['qz'])
        transform = ImuToCam @ transform  
        ground_truth_camera.append((newPoint, distance,transform))

#   Clase para reproducir el bag file y remapear los topics
class BagPlayer(Node):
    def __init__(self):
        super().__init__('bag_player')
        self.bridge = CvBridge()
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
        if not self.reader.has_next():
            raise KeyboardInterrupt
            return
        while self.reader.has_next():
            (topic, raw_data, stamp) = self.reader.read_next()
            if topic in self.remap_topics.keys():    
                self.remap_topics[topic].publish(raw_data)
                break  # Publicar solo un mensaje por llamada al timer
            else:
                continue
        

#  Esta clase es para el ejercio 2-a: Rectificacion de imagenes
class ImageRectifier(Node):
    def __init__(self, map3d):
        super().__init__('image_rectifier')
        self.bridge = CvBridge()
        # Load calibration data
        calib_data_left = load_yaml(f'calibrationdata/left.yaml')
        calib_data_right = load_yaml(f'calibrationdata/right.yaml')
        
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
        if map3d is not None:
            map3d.setQ(self.Q)
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

#  Esta clase es para el ejercio 2-c: Match de keypoints 
class Matcher(Node):
    def __init__(self, triangulation):
        super().__init__('matcher')
        self.bridge = CvBridge()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.publisher = self.create_publisher(Image, '/stereo/matched_keypoints', 10)
        self.publisherFiltered = self.create_publisher(Image, '/stereo/filtered_matched_keypoints', 10)
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
        if stamp in self.info and "left" in self.info[stamp] and "right" in self.info[stamp]:
            left = self.info[stamp]["left"]
            right = self.info[stamp]["right"]
            matches = self.bf.match(left['descriptors'], right['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            self.triangulation.compute(matches, left, right)
            matched_image = cv.drawMatches(
                left['image'], left['keypoints'],
                right['image'], right['keypoints'],
                matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            match_msg = self.bridge.cv2_to_imgmsg(matched_image, encoding='bgr8')
            match_msg.header = left['header']
            match_msg.header.frame_id = "matched_keypoints"
            self.publisher.publish(match_msg)
            filtered_matches = filter(lambda m: m.distance < 30, matches)
            filtered_image = cv.drawMatches(
                left['image'], left['keypoints'],
                right['image'], right['keypoints'],
                list(filtered_matches), None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_image, encoding='bgr8')
            filtered_msg.header = left['header']
            filtered_msg.header.frame_id = "filtered_matched_keypoints"
            self.publisherFiltered.publish(filtered_msg)
            # cv.imshow("Matched Keypoints", matched_image)
            # cv.waitKey(1)
            
#  Esta clase es para el ejercio 2-b: Extraccion de keypoints
class KeypointDetector(Node):
    def __init__(self, side, dependencies, node_name='keypoint_detector'):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.orb = cv.ORB_create()
        self.subscription_left = self.create_subscription(
            Image,
            f'/stereo/{side}/image_rect',
            self.listener_callback, 10)
        self.publisher = self.create_publisher(Image, f'/stereo/{side}/image_keypoints', 10)
        self.side = side
        self.dependencies = dependencies

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
        for dependency in self.dependencies:
            dependency.set_info(info, self.side)
        # Dibujar los puntos en la imagen
        output_image = cv.drawKeypoints(img, keypoints,None,color=(0,255,0))
        # output_image = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        key_msg.header = msg.header
        self.publisher.publish(key_msg)
            
        # cv.waitKey(1)


def create_pointcloud2(points, header):
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
    return msg

#  Esta clase es para el ejercio 2-d: Triangulacion de puntos
class TriangulatePoints(Node):
    def __init__(self, node_name='triangulate_points', map=None):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PointCloud2, f'/stereo/{node_name}', 10)
        calib_data_left = load_yaml(f'calibrationdata/left.yaml')
        calib_data_right = load_yaml(f'calibrationdata/right.yaml')
        projection_left = np.array(calib_data_left['projection_matrix']['data']).reshape(3, 4)
        projection_right = np.array(calib_data_right['projection_matrix']['data']).reshape(3, 4)
        self.projections = np.array([projection_left, projection_right])
        self.map = map
    
    def publish(self, points, header):
        msg = create_pointcloud2(points, header)
        
        self.publisher.publish(msg)
    
    def compute(self, matches, left_info, right_info):
        pts_left = np.float32([left_info['keypoints'][m.queryIdx].pt for m in matches])
        pts_right = np.float32([right_info['keypoints'][m.trainIdx].pt for m in matches])
        triangulatedPoints = cv.triangulatePoints(self.projections[0], self.projections[1], pts_left.T, pts_right.T)
        triangulatedPoints = (triangulatedPoints[:3] / triangulatedPoints[3]).T
        if self.map is not None:
            self.map.set_info(triangulatedPoints, left_info['header'])
        self.publish(triangulatedPoints, left_info['header'])
        
        # Publicar los puntos 3D como PointCloud2

#  Esta clase es para el ejercio 2-e: Triangulacion de puntos con RANSAC
class TriangulatedPointsRansac(TriangulatePoints):
    def __init__(self, node_name='triangulated_points_ransac', map=None):
        super().__init__(node_name, map)
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

#  Esta clase es para el ejercio 2-f: Mapeo de Features triangulados al sistema de coordenadas mundial
class  MappingPoints(Node):
    def __init__(self, node_name='mapping_points'):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PointCloud2, '/stereo/mapped_triangulated_points', 10)
        self.publisher_persistent = self.create_publisher(PointCloud2, '/stereo/mapped_triangulated_points_persistent', 10)
        self.frame = 0
        self.accumulated_points = []

    def set_info(self, points, header):
        self.compute(points,header)

    def compute(self, points, header):        
        
        gt_transform = ground_truth_camera[self.frame][2]
        points_world = []
        R = gt_transform[:3, :3]
        t = gt_transform[:3, 3].reshape(1, 3)
        for point in points:
            newPoint = (R @ point.T) + t.ravel()             
            points_world.append(newPoint)
        self.publisher.publish(create_pointcloud2(np.array(points_world), header))

        self.accumulated_points.extend(points_world)
        persistent_msg = create_pointcloud2(np.array(self.accumulated_points), header)
        self.publisher_persistent.publish(persistent_msg)
        self.frame += 1
#  Esta clase es para el ejercio 2-g: Calculo del mapa de disparidad
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

    def publish(self, disp, header):
        disp_msg = self.bridge.cv2_to_imgmsg(disp)
        disp_msg.header = header
        self.publisher.publish(disp_msg)

    def compute(self, stamp, header):
        if stamp in self.info and "left" in self.info[stamp] and "right" in self.info[stamp]:
            left_img = self.info[stamp]["left"]
            right_img = self.info[stamp]["right"]
            disparity_map = self.stereo.compute(left_img, right_img)
            # disp_normalized = cv.normalize(disparity_map,None,255,0,cv.NORM_MINMAX, cv.CV_8U)
            self.publish(disparity_map, header)

#  Esta clase es para el ejercio 2-h: Reconstruccion densa a partir del mapa de disparidad
class Map3D(Node):
    def __init__(self, node_name='map_3d'):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PointCloud2, f'/stereo/{node_name}', 10)
        # definimos la matriz de rotacion para girar la proyeccion 3d 90 grados en x
        self.RMatrix = np.array([[0, 0, -1],
                                 [0, -1, 0],
                                 [1, 0, 0]])
        self.Q = None 
        self.subscription = self.create_subscription(
            Image,
            f'/stereo/disparity',
            self.callback, 10)
    def callback(self,msg):
        disparity = self.bridge.imgmsg_to_cv2(msg)
        self.compute(disparity, msg.header)

    def setQ(self, Q):
        self.Q = Q
        
    def compute(self, disparity, header):
        if self.Q is not None:
            # Reescalar la disparidad
            disparity = disparity.astype(np.float32) / 16.0
            # Reproyectar a 3D
            points_3d = cv.reprojectImageTo3D(disparity, self.Q)
            mask = disparity > disparity.min()
            points_3d = points_3d[mask]
            points_3d = points_3d[ ~np.isnan(points_3d[:,2]) & ~np.isinf(points_3d[:,2]) & (points_3d[:,2] != 0)]            
            points_3d = points_3d.reshape(-1, 3)

            msg = create_pointcloud2(points_3d[:] / 1000, header)
            self.publisher.publish(msg)

#  Esta clase es para el ejercio 2-i: Reconstruccion densa con ground truth
# Debido a que por cada frame se generan muchos puntos 3D, solo se publican 1 de cada 50 puntos
class Map3dDense(Map3D):
    def __init__(self, node_name='map_3d_dense'):
        super().__init__(node_name)
        self.publisher_persistent = self.create_publisher(PointCloud2, f'/stereo/{node_name}_persistent', 10)
        self.frame = 0
        self.accumulated_points = []

    def compute(self, disparity, header):
        if self.Q is not None:
            if self.frame % 5 == 0:
            # Reescalar la disparidad
                disparity = disparity.astype(np.float32) / 16.0
                # Reproyectar a 3D
                points_3d = cv.reprojectImageTo3D(disparity, self.Q)
                mask = disparity > disparity.min()
                points_3d = points_3d[mask]
                points_3d = points_3d[ ~np.isnan(points_3d[:,2]) & ~np.isinf(points_3d[:,2]) & (points_3d[:,2] != 0)]
                points_3d = points_3d.reshape(-1, 3)
                gt_transform = ground_truth_camera[self.frame][2]
                R = gt_transform[:3, :3]
                t = gt_transform[:3, 3].reshape(1, 3)
                world_points = []
                for index in range(0,len(points_3d),50):                
                    point = points_3d[index]
                    newPoint = (R @ point.T) + t.ravel()
                    world_points.append(newPoint / 1000)
                msg = create_pointcloud2(np.array(world_points), header)
                self.publisher.publish(msg)
                self.accumulated_points.extend(world_points)
                persistent_msg = create_pointcloud2(np.array(self.accumulated_points), header)
                self.publisher_persistent.publish(persistent_msg)      
        self.frame += 1 

# Clase base para la estimacion de pose
class EstimatePose(Node):
    def __init__(self, node_name='estimate_pose'):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.calculated = False
        self.info = {}
        calib_data_left = load_yaml(f'calibrationdata/left.yaml')
        calib_data_right = load_yaml(f'calibrationdata/right.yaml')
        
        self.K1 = np.array(calib_data_left['camera_matrix']['data']).reshape(3, 3)
        self.K2 = np.array(calib_data_right['camera_matrix']['data']).reshape(3, 3)
        self.t = None
        self.baseline = abs(calib_data_right['projection_matrix']['data'][3]) / 1000  # Convertir a metros
        

    def set_info(self, info, side):
        pass

    def compute(self):
        pass

    def show(self, R, t):
        pass

#  Esta clase es para el ejercio 2-j-1: Estimacion de pose entre dos camaras
class ShowEstimatePoseLR(EstimatePose):
    def __init__(self):
        super().__init__(node_name='show_estimate_pose_lr')
        print("Baseline (m): ", self.baseline)
    def set_info(self, info, side):
        stamp = info['header'].stamp
        stamp_obj = (stamp.sec, stamp.nanosec)
        if stamp_obj not in self.info:
            self.info[stamp_obj] = {}
        self.info[stamp_obj][side] = info
        self.compute(stamp_obj)
        
    def compute(self, stamp):
        if not self.calculated:
            if stamp in self.info and "left" in self.info[stamp] and "right" in self.info[stamp]:
                left = self.info[stamp]["left"]
                right = self.info[stamp]["right"]
                matches = self.bf.match(left['descriptors'], right['descriptors'])
                matches = sorted(matches, key=lambda x: x.distance)
                # print([kp.pt for kp in left['keypoints']])
                point1 = np.float32([left['keypoints'][m.queryIdx].pt for m in matches])
                point2 =  np.float32([right['keypoints'][m.trainIdx].pt for m in matches])

                E, mask = cv.findEssentialMat(  point2, 
                                                point1,
                                                self.K1,
                                                method=cv.RANSAC, 
                                                prob=0.999, 
                                                threshold=1.0)

                inliers1 = point1[mask]
                inliers2 = point2[mask]

                _, R, t, mask = cv.recoverPose(E, inliers2, inliers1, mask=mask, cameraMatrix=self.K1)

                t = t * self.baseline                
                self.t = t
    def show(self, R = None, t = None):  
        # Plot origin and translation vector t in 3D
        try:
            origin = np.zeros(3)
            tvec = np.asarray(self.t).ravel()

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
        # Draw axis arrows from origin
            scale = np.linalg.norm(tvec)
            if scale == 0:
                scale = 1.0
            axis_len = scale * 0.6
            ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, axis_len, 0, color='b', arrow_length_ratio=0.1, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, 0, -axis_len, color='g', arrow_length_ratio=0.1, linewidth=1.5)
            ax.text(axis_len, 0, 0, 'X', color='r')
            ax.text(0, axis_len, 0, 'Z', color='b')
            ax.text(0, 0, -axis_len, 'Y', color='g')
            ax.scatter(*origin, color='red', s=50, label='Left Camera (Origin)')          
            ax.scatter(tvec[0], tvec[1], tvec[2], color='blue', s=50, label='Right Camera')
            ax.plot([0, tvec[0]], [0, tvec[1]], [0, tvec[2]], color='gray', linestyle='--')

            # Set equal scaling
            pts = np.vstack((origin, tvec))
            max_range = np.max(np.ptp(pts, axis=0))
            if max_range == 0:
                max_range = 1.0
            mid = np.mean(pts, axis=0)
            ax.set_xlim((mid[0] - max_range/2)*2, (mid[0] + max_range/2)*2)
            ax.set_ylim((mid[1] - max_range/2)*2, (mid[1] + max_range/2)*2)
            ax.set_zlim((mid[2] - max_range/2)*2, (mid[2] + max_range/2)*2)
            
            ax.legend()
            plt.savefig('pose_estimation.png')
            plt.close()
        except Exception as e:
          print("Could not plot 3D points:", e)
        self.calculated = True

#  Esta clase es para el ejercio 2-j-2: Estimacion de camino realizado por la camara izquierda
class EstimatePath(EstimatePose):
    def __init__(self):
        super().__init__(node_name='estimate_path')
        # info is the list of keypoints
        self.info = []        
        
        # path is the list of positions
        self.path = []
        # self.trajectory = [np.eye(4)]
        
        self.trajectory = [ground_truth_camera[0][2]] # this should be np.eye(4)
        
    def set_info(self, info, side):
        if side == "left":
            self.info.append(info)
    
    def compute(self):
        # Sort collected frames by their ROS2 header timestamp (sec, nanosec)        
        self.info.sort(key=lambda x: (x['header'].stamp.sec, x['header'].stamp.nanosec))
        for i in range(1,len(self.info)):
            last_point_info = self.info[i-1]
            point = self.info[i]
            matches = self.bf.match(last_point_info['descriptors'], point['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            # print([kp.pt for kp in left['keypoints']])
            last_point = np.float32([last_point_info['keypoints'][m.queryIdx].pt for m in matches])
            new_point =  np.float32([point['keypoints'][m.trainIdx].pt for m in matches])
            self.info.append(point)
            E, mask = cv.findEssentialMat(  
                                            new_point, 
                                            last_point,
                                            self.K1,
                                            method=cv.RANSAC, 
                                            prob=0.999, 
                                            threshold=1.0)

            inliers1 = last_point[mask]
            inliers2 = new_point[mask]

            _, R, t, mask = cv.recoverPose(E, inliers2, inliers1, mask=mask, cameraMatrix=self.K1)
            # print(f"Pre escalado: {t}")
            baseline = ground_truth_camera[i][1]
            t = t * baseline
            T = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

            self.trajectory.append(ground_truth_camera[i-1][2] @ T)
            # print(f"Post escalado: {t}, escala: {ground_truth[len(self.info)][1]}")
            # xi_camera_i_to_next = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
            # last_pose = self.path[-1][1].reshape(4,4)
            
            # self.path.append((np.dot(last_pose, np.vstack((t,[1]))).ravel(), xi_camera_i_to_next))
        self.show()
    def show(self, R=None, t=None):
        # print(self.path)
        fig = plt.figure(figsize=(10, 6))        
        ax = fig.add_subplot(111, projection='3d')
        aux_traj = self.trajectory
        poses = np.array([traj[:3, 3] for traj in aux_traj])
        x = poses[:, 0]
        y = poses[:, 1]
        z = poses[:, 2]
        # x = [e[0][0] for e in self.path]
        # y = [e[0][1] for e in self.path]
        # z = [e[0][2] for e in self.path]
        # gt = [e[2] for e in ground_truth] @ self.ImuToCam
        # keep only the first half of ground truth poses (after transforming by ImuToCam)
        # gt_mats = [e[2] for e in ground_truth]
        
        gt = ground_truth_camera[:len(self.trajectory)]
        gxs = [e[2][:3,3][0] for e in gt]
        gys = [e[2][:3,3][1] for e in gt]
        gzs = [e[2][:3,3][2] for e in gt]
        ax.plot(x, y, z, c='red', label='Estimated Path', alpha=0.7, linewidth=0.75)
        sc2 = ax.plot(gxs, gys, gzs, c='blue', label='Ground Truth Path', alpha=0.7, linewidth=0.75)
        print("Initial Position GT: ", gxs[0], gys[0], gzs[0])
        print("Initial Position Est: ", x[0], y[0], z[0])
        ax.scatter(gxs[0],gys[0],gzs[0], color='green', s=50, label='origin gt')
        ax.scatter(x[0],y[0],z[0], color='yellow', s=50, label='origin est')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Estimated Left Camera Path')
        ax.legend()

        plt.savefig('LeftCameraPath.svg')
        plt.close()

def main(args=None):
    print("Loading ground truth data")
    load_GT_data('ground_truth.csv')
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    print("Starting bag player")
    bag_player = BagPlayer()
    executor.add_node(bag_player)
    print("Starting 3D mapper")
    # map3d = Map3D()
    map3d = Map3dDense()
    executor.add_node(map3d)
    print("Starting image rectifier")
    rectifier = ImageRectifier(map3d)
    executor.add_node(rectifier)
    print("Starting image disparity")
    dispMap = DisparityMap()
    executor.add_node(dispMap)
    print("Starting mapping points from triangulation")
    mappingPoints = MappingPoints()
    executor.add_node(mappingPoints)
    print("starting Triangulation")    
    triangulation = TriangulatePoints(map=mappingPoints)
    # triangulation = TriangulatedPointsRansac()    
    executor.add_node(triangulation)
    print("Starting matcher")
    matcher = Matcher(triangulation)
    executor.add_node(matcher)
    print("Starting pose estimator")
    estimator = ShowEstimatePoseLR()
    pathEstimator = EstimatePath()    
    print("Starting keypoint detector")
    keyDetectorL = KeypointDetector("left", [matcher, estimator, pathEstimator], node_name='keypoint_detector_left')
    keyDetectorR = KeypointDetector("right", [matcher, estimator], node_name='keypoint_detector_right')
    executor.add_node(keyDetectorL)
    executor.add_node(keyDetectorR)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        estimator.show()
        pathEstimator.compute()        
        bag_player.destroy_node()
        rectifier.destroy_node()
        map3d.destroy_node()
        dispMap.destroy_node()
        keyDetectorL.destroy_node()
        keyDetectorR.destroy_node()
        matcher.destroy_node()
        triangulation.destroy_node()
        estimator.destroy_node()
        pathEstimator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()