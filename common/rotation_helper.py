import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

def transform_imu_data_pelvis_to_torso(waist_yaw, waist_yaw_omega, pelvis_quat, pelvis_omega):
    """
    Transform IMU data from pelvis frame to torso frame
    
    Args:
        waist_yaw: waist yaw angle (rotation from torso to pelvis)
        waist_yaw_omega: waist yaw angular velocity
        pelvis_quat: quaternion in pelvis frame [w, x, y, z]
        pelvis_omega: angular velocity in pelvis frame
    
    Returns:
        torso_quat: quaternion in torso frame [w, x, y, z]
        torso_omega: angular velocity in torso frame
    """
    # Create rotation matrix from torso to pelvis (around z-axis)
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    
    # Get pelvis orientation matrix
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    
    # Transform to torso frame: R_torso = R_pelvis * RzWaist
    R_torso = np.dot(R_pelvis, RzWaist)
    
    # Transform angular velocity to torso frame
    # torso_omega = RzWaist^T * pelvis_omega + [0, 0, waist_yaw_omega]
    torso_omega = np.dot(RzWaist.T, pelvis_omega) + np.array([0, 0, waist_yaw_omega])
    
    # Convert rotation matrix back to quaternion [w, x, y, z]
    torso_quat = R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]]
    
    return torso_quat, torso_omega



def is_torso_bad_orientation(waist_euler_xyz, pelvis_quat, limit_angle_deg):
    R_Waist = R.from_euler("xyz", waist_euler_xyz).as_matrix()
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    torso_rot_world = np.dot(R_pelvis, R_Waist)

    torso_quat_world = R.from_matrix(torso_rot_world).as_quat()
    torso_quat_world = np.array([torso_quat_world[3], torso_quat_world[0], torso_quat_world[1], torso_quat_world[2]])
    torso_proj = get_gravity_orientation(torso_quat_world)
    tilt_angle = np.arccos(-torso_proj[2])
    
    limit_angle_rad = np.deg2rad(limit_angle_deg)
    is_bad = np.abs(tilt_angle) > limit_angle_rad
    
    return is_bad


def is_object_bad_orientation(waist_euler_xyz, pelvis_quat, object_camera_quat, limit_angle_deg):
    
    R_Waist = R.from_euler("xyz", waist_euler_xyz).as_matrix()
    R_pelvis_w = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_torso_w = np.dot(R_pelvis_w, R_Waist)
    R_camera_torso = R.from_euler("xyz", np.array([0, 0.8307767239493009, 0])).as_matrix()
    R_object_camera = R.from_quat(np.array([object_camera_quat[1], object_camera_quat[2], object_camera_quat[3],object_camera_quat[0]])).as_matrix()

    R_camera_w = np.dot(R_torso_w, R_camera_torso)
    R_object_w = np.dot(R_camera_w, R_object_camera)
    object_quat_world = R.from_matrix(R_object_w).as_quat()
    object_quat_world = np.array([object_quat_world[3], object_quat_world[0], object_quat_world[1] ,object_quat_world[2]])
    object_proj = get_gravity_orientation(object_quat_world)
    tilt_angle = np.arccos(-object_proj[2])
    
    limit_angle_rad = np.deg2rad(limit_angle_deg)
    is_bad = np.abs(tilt_angle) > limit_angle_rad
    
    return is_bad