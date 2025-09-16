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
                 tag_size: float = 52.12 / 1000,
                 families: str = "tag25h9",
                 tag_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 coordinate_transform: bool = True,
                 history_length: int = 5
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
        
        
        self.transform_matrix = np.array([
            [0,  0,  1],   # camera Z -> robot X 
            [-1, 0,  0],   # camera -X -> robot Y  
            [0, -1,  0]    # camera -Y -> robot Z 
        ])

        # buffer
        self.history_length = history_length
        self.object_pose_buff = CircularBuffer(max_len=history_length, data_shape=(7,))
        self.default_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
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
        
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        
        # Apply coordinate transformation if enabled
        if self.coordinate_transform:
            position = self.transform_matrix @ position
            rotation_matrix = self.transform_matrix @ rotation_matrix @ self.transform_matrix.T
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        r = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
        
        # Combine position and quaternion into 7D vector
        pose_7d = np.concatenate([position, quat_wxyz])
        
        return pose_7d
    
    
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
                            pose_7d = self._process_detection(detection)
                            
                            # Add to buffer (thread-safe)
                            with self.detection_lock:
                                self.object_pose_buff.append(pose_7d)
                            
                            target_detected = True
                            break  # Found target, no need to check other detections
                            
                        except Exception as e:
                            print(f"Error processing tag {detection.tag_id}: {e}")
                            continue
                
                # If target tag not detected, add default pose (zeros + identity quaternion)
                if not target_detected:
                    with self.detection_lock:
                        self.object_pose_buff.append(self.default_pose)
                       
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
            obs = self.object_pose_buff.get_history(self.history_length)
        
        return obs.flatten()
    

    def get_last_obs(self) -> np.ndarray:

        with self.detection_lock:
            obs = self.object_pose_buff.get_history(1)
        return obs.flatten()
    
    def is_running(self) -> bool:
        
        return self.running
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()



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