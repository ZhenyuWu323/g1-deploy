import collections
import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag
from scipy.spatial.transform import Rotation
import math
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams (e.g., 640x480, 30fps)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get camera intrinsics (only once, outside the loop)
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()

camera_params = [
    intrinsics.fx,  # focal length x
    intrinsics.fy,  # focal length y
    intrinsics.ppx, # principal point x
    intrinsics.ppy  # principal point y
]

print(f"Camera intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

# apriltag detector
options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)
tag_size = 71.12 / 1000



def draw_pose(overlay, camera_params, tag_size, pose, z_sign=1):
    opoints = np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
        -1, -1, -2*z_sign,
         1, -1, -2*z_sign,
         1,  1, -2*z_sign,
        -1,  1, -2*z_sign,
    ]).reshape(-1, 1, 3) * 0.5*tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)
        
    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)

    ipoints = np.round(ipoints).astype(int)
    
    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)


def draw_tag_axes(img, pose, camera_params):
    
    fx, fy, cx, cy = camera_params
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=float)


    R_ct = pose[:3, :3]
    t_ct = pose[:3, 3]
    rvec, _ = cv2.Rodrigues(R_ct)
    tvec = t_ct.reshape(3,)

    dcoeffs = np.array(intrinsics.coeffs[:5], dtype=float)

    cv2.drawFrameAxes(img, K, dcoeffs, rvec, tvec, length=tag_size*0.75, thickness=2)


def world_to_ros(rotm_world: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from world frame to ROS frame convention.
    
    World convention: +X forward, +Y left, +Z up
    ROS convention: +Z forward, -Y up, +X right
    
    Args:
        quat_world: Quaternion in world frame [qw, qx, qy, qz]
    
    Returns:
        Rotation matrix (3x3) in ROS frame convention
    """
    
    # World to OpenGL: apply inverse transformation
    # T_world = R(pi/2, X) @ R(-pi/2, Y) @ T_opengl
    # So: T_opengl = R(-pi/2, Y)^T @ R(pi/2, X)^T @ T_world
    euler_angles = np.array([np.pi/2, -np.pi/2, 0])
    transform_matrix = Rotation.from_euler('XYZ', euler_angles).as_matrix()
    
    # OpenGL frame rotation: rotm_world = rotm_opengl @ transform_matrix.T
    # So: rotm_opengl = rotm_world @ transform_matrix
    rotm_opengl = rotm_world @ transform_matrix
    
    # OpenGL to ROS: flip Y and Z columns
    rotm_ros = rotm_opengl.copy()
    rotm_ros[:, 1] = -rotm_opengl[:, 1]
    rotm_ros[:, 2] = -rotm_opengl[:, 2]
    
    return rotm_ros


def ros_to_world(rotm_ros: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from ROS frame to world frame convention.
    
    ROS convention: +Z forward, -Y up, +X right
    World convention: +X forward, +Y left, +Z up
    
    Args:
        quat_ros: Quaternion in ROS frame [qw, qx, qy, qz]
    
    Returns:
        Rotation matrix (3x3) in world frame convention
    """
    
    # ROS to OpenGL: flip Y and Z columns
    rotm_opengl = rotm_ros.copy()
    rotm_opengl[:, 1] = -rotm_ros[:, 1]
    rotm_opengl[:, 2] = -rotm_ros[:, 2]
    
    # OpenGL to World: apply transformation
    # T_world = R(pi/2, X) @ R(-pi/2, Y) @ T_opengl
    euler_angles = np.array([np.pi/2, -np.pi/2, 0])
    transform_matrix = Rotation.from_euler('XYZ', euler_angles).as_matrix()
    
    rotm_world = rotm_opengl @ transform_matrix.T
    
    return rotm_world

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        detections, images = detector.detect(gray_image, return_image=True)
        
    
        for detection in detections:
            # Get tag corners
            corners = detection.corners.astype(int)

            pose, e0, e1 = detector.detection_pose(
                detection=detection,
                camera_params=camera_params,
                tag_size=tag_size,
            )

            draw_pose(
                overlay=color_image,
                camera_params=camera_params,
                tag_size=tag_size,
                pose=pose
            )

            # NOTE: RAW pose
            draw_tag_axes(color_image, pose, camera_params)

            # print(detection.tostring(
            #         collections.OrderedDict([('Pose',pose),
            #                                  ('InitError', e0),
            #                                  ('FinalError', e1)]),
            #         indent=2))
            
            transform = np.eye(4,4)
            transform[:3, :3] = np.array([
                [0,  0,  1],   
                [-1,  0,  0],   
                [0, -1,  0]    
            ])
            flip = np.eye(4,4)
            # flip[:3, :3] = np.array([
            #     [1, 0, 0],
            #     [0, -1, 0],
            #     [0, 0, -1]
            # ])
            flip[:3, :3] = np.array([
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, -1]
            ])
            pose = transform @ pose @ flip

            # # rotate apriltag first
            # object_transform = Rotation.from_euler('xy', [90,-90], degrees=True)
            # rot_obj = object_transform.as_matrix()
            # position = position @ rot_obj
            # rotation_matrix = rotation_matrix @ rot_obj

            # # rotate camera
            # rot_camera = np.array([
            #     [0,  0,  1],   
            #     [-1, 0,  0],    
            #     [0, -1,  0]   
            # ])
            # position = rot_camera @ position
            # rotation_matrix = rot_camera @ rotation_matrix

            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            #rotation_matrix = ros_to_world(rotation_matrix)

            
            
            r = Rotation.from_matrix(rotation_matrix)
            quat = Rotation.as_quat(r)
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            euler = Rotation.as_euler(r, seq="XYZ", degrees=True)
            print(euler)
            

            # print(detection.tostring(
            #         collections.OrderedDict([('Pose',position),
            #                                  ('Quat', quat),
            #                                  #("Euler", euler)
            #                                  ]),
            #         indent=2))
            


        # Apply colormap on depth image (optional)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display images
        cv2.imshow('RealSense Color', color_image)
        #cv2.imshow('RealSense Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()