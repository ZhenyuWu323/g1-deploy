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