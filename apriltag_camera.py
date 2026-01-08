import collections
import threading
import time
import queue
from typing import Optional, Dict, List, Tuple, Any
import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag
from scipy.spatial.transform import Rotation
from common.circular_buffer import CircularBuffer


class AprilTagDetector:
    """
    RealSence Apriltag detector
    
    use case:
        detector = AprilTagDetector(tag_size=0.05212, families="tag25h9")
        detector.start()
        
        
        detections = detector.get_latest_detections()
        if detections:
            for tag_id, position, quaternion in detections:
                print(f"Tag {tag_id}: pos={position}, quat={quaternion}")
        
        detector.stop()
    """
    
    def __init__(self, 
                 tag_size: float = 50 / 1000,
                 #tag_size: float = 71.12 / 1000,
                 families: str = "tag16h5",
                 tag_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 coordinate_transform: bool = True,
                 history_length: int = 5,
                 pose_type: str = 'quat',
                 record_quat = True
                 ):
        """
        Args:
            tag_size: AprilTag tag size
            families: AprilTag family
            resolution: resolution
            fps: 
            enable_visualization: 
            coordinate_transform: if use transform
        """
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.resolution = resolution
        self.fps = fps
        self.coordinate_transform = coordinate_transform
        self.record_quat = record_quat
        
        # RealSense pipeline
        self.pipeline = None
        self.config = None
        self.camera_params = None
        
        # AprilTag detector
        options = apriltag.DetectorOptions(families=families)
        self.detector = apriltag.Detector(options)
        
        # Threading
        self.running = False
        self.thread = None
        self.detection_queue = queue.Queue(maxsize=10)
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        

        # buffer
        self.history_length = history_length
        self.pose_type = pose_type
        data_shape = 7
        if self.pose_type == '6d':
            data_shape = 3 + 6
        elif self.pose_type == 'euler_xyz':
            data_shape = 3 + 3
        else:
            data_shape = 7

        self.object_pose_buff = CircularBuffer(max_len=history_length, data_shape=(data_shape,))
        if self.record_quat:
            self.object_quat_buff = CircularBuffer(max_len=history_length, data_shape=(4,))
        else:
            self.object_quat_buff = None
        
        self.prev_pos = np.zeros(data_shape)
        self.prev_quat = np.array([1,0,0,0])

        # transform
        self.transform = np.eye(4,4)
        self.transform[:3, :3] = np.array([
                [0,  0,  1],   
                [-1,  0,  0],   
                [0, -1,  0]    
            ])

        self.flip = np.eye(4,4)
        self.flip[:3, :3] = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
    
    def _initialize_camera(self):
        """Initialize RealSense"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        width, height = self.resolution
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        
        self.pipeline.start(self.config)
        
        # get intrinsic
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        
        self.camera_params = [
            intrinsics.fx,  # focal length x
            intrinsics.fy,  # focal length y
            intrinsics.ppx, # principal point x
            intrinsics.ppy  # principal point y
        ]
        
        print(f"Camera initialized: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
              f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    
    def _process_detection(self, detection) -> np.ndarray:
        """
        Process a single detection and return 7D pose vector
        
        Returns:
            7D array: [x, y, z, qw, qx, qy, qz]
        """
        # Get pose from detection
        pose, _, _ = self.detector.detection_pose(
            detection=detection,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        
        
        
        # Apply coordinate transformation if enabled
        if self.coordinate_transform:
            pose = self.transform @ pose @ self.flip

            
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        # Convert rotation matrix to quaternion (w, x, y, z)
        r = Rotation.from_matrix(rotation_matrix)
        orientation = None
        quat_xyzw = r.as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
        if self.pose_type == 'quat':
            orientation = quat_wxyz
        elif self.pose_type == '6d':
            col1 = rotation_matrix[:, 0]  # First column (3,)
            col2 = rotation_matrix[:, 1]  # Second column (3,)
            orientation = np.concatenate([col1, col2], axis=-1)  # Shape: (6,)
        else:
            orientation = r.as_euler(str='XYZ')
            

        # Combine position and quaternion into 7D vector
        pose_7d = np.concatenate([position, orientation])
        
        return pose_7d, quat_wxyz
    
    
    def _detection_loop(self):
        """Main detection loop running in background thread"""
        while self.running:
            try:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to images
                color_image = np.asanyarray(color_frame.get_data())
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                
                # Detect AprilTags
                detections = self.detector.detect(gray_image)

                # Look for target tag
                target_detected = False
                for detection in detections:
                    if detection.tag_id == self.tag_id:
                        try:
                            pose_7d, object_quat = self._process_detection(detection)
                            
                            # Add to buffer (thread-safe)
                            with self.detection_lock:
                                self.prev_pos = pose_7d.copy()
                                self.prev_quat = object_quat.copy()
                                self.object_pose_buff.append(pose_7d)
                                if self.record_quat:
                                    self.object_quat_buff.append(object_quat)
                            
                            target_detected = True
                            break  # Found target, no need to check other detections
                            
                        except Exception as e:
                            print(f"Error processing tag {detection.tag_id}: {e}")
                            continue
                
                # if not detected use last detection
                if not target_detected:
                    with self.detection_lock:
                        self.object_pose_buff.append(self.prev_pos)
                        if self.record_quat:
                            self.object_quat_buff.append(self.prev_quat)
                       
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.01)  # Avoid busy waiting 
    
    def start(self):
        """Start the detector"""
        if self.running:
            print("Detector is already running")
            return
        
        self._initialize_camera()
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("AprilTag detector started")
    
    def stop(self):
        """Stop the detector"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.pipeline:
            self.pipeline.stop()
        
        print("AprilTag detector stopped")
    
    def get_object_obs(self) -> np.ndarray:
        """
        Get latest object observations as flattened array
        
        Returns:
            Flattened array of shape (history_length * 7,) containing pose history
            Each 7D element is [x, y, z, qw, qx, qy, qz]
        """
        with self.detection_lock:
            obs = self.object_pose_buff.buffer
        
        return obs
    
    def get_object_quat(self) -> np.ndarray:
        """
        Get latest object observations as flattened array
        
        Returns:
            Flattened array of shape (history_length * 7,) containing pose history
            Each 7D element is [x, y, z, qw, qx, qy, qz]
        """
        if not self.record_quat or self.object_quat_buff is None:
            raise RuntimeError("Quat recording is disabled. Set record_quat=True")
        with self.detection_lock:
            obs = self.object_quat_buff.buffer
        
        return obs
    
    def get_last_obj_pose(self)-> np.ndarray:
        with self.detection_lock:
            obs = self.object_pose_buff[0]
        return obs
    
    
    def get_last_quat(self) -> np.ndarray:
        if not self.record_quat or self.object_quat_buff is None:
            raise RuntimeError("Quat recording is disabled. Set record_quat=True")
        with self.detection_lock:
            obs = self.object_quat_buff[0]
        return obs
    
    def is_running(self) -> bool:
        
        return self.running
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()

    def reset(self):
        """Reset buffers"""
        with self.detection_lock:
            self.object_pose_buff.reset()
            if self.record_quat and self.object_quat_buff is not None:
                self.object_quat_buff.reset()
        



if __name__ == "__main__":
    def test_detector():
        """Test the detector"""
        detector = AprilTagDetector(
            tag_size=52.12 / 1000,
            families="tag25h9",
            tag_id=0,
            history_length=5
        )
        
        try:
            detector.start()
            
            # Let it run for a bit
            time.sleep(2)
            
            # Get observations
            while True:
                obs = detector.get_object_obs()
                print(f"  Obs shape: {obs.shape}")
                print(f"  Obs: {obs}")
                print("---")
                
                time.sleep(0.02)
                
        finally:
            detector.stop()
    
    test_detector()