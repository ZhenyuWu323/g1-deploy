import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import argparse
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import yaml
from collections import deque
from common.rotation_helper import transform_imu_data_pelvis_to_torso, is_object_bad_orientation, is_torso_bad_orientation, get_object_gravity_orientation
from policy.policy_runner import ResidualPolicyRunner
from config import CONFIG_PATH
from common.circular_buffer import CircularBuffer
from utils import quat_apply
from scipy.spatial.transform import Rotation
import threading
import select
import sys

USE_RESIDUAL = False
LOAD_RESIDUAL = True
ENCODER_HISTORY_STEP = 32
POSE_TYPE = '6d'
ACTUATOR_CONFIG = 'mimic'
INCLUDE_OBJECT_GRAVITY_ORIENTATION = False

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def compute_soft_limit(lower_limit, upper_limit, soft_factor=0.9):
    center = (upper_limit + lower_limit) / 2.0
    half_range = (upper_limit - lower_limit) / 2.0
    soft_lower = center - half_range * soft_factor
    soft_upper = center + half_range * soft_factor
    return (soft_lower, soft_upper) 

def check_joint_limit(action, soft_limit):
    lower_violations = action < soft_limit[0]
    upper_violations = action > soft_limit[1]
    
    within_limit = np.all((action >= soft_limit[0]) & (action <= soft_limit[1]))
    if not within_limit:
        violated_indices = np.where(lower_violations | upper_violations)[0]
        
        print(f"Joint limit violations detected:")
        for idx in violated_indices:
            if lower_violations[idx]:
                print(f"  Joint {idx}: {action[idx]:.4f} < lower_limit {soft_limit[0][idx]:.4f} "
                      f"(violation: {action[idx] - soft_limit[0][idx]:.4f})")
            if upper_violations[idx]:
                print(f"  Joint {idx}: {action[idx]:.4f} > upper_limit {soft_limit[1][idx]:.4f} "
                      f"(violation: {action[idx] - soft_limit[1][idx]:.4f})")
    return within_limit 

def apply_action(
    num_actions,
    d,
    target_lower_pos,
    target_upper_pos,
    lower_body_joint2motor_idx,
    upper_body_joint2motor_idx,
    lower_body_kps,
    upper_body_kps,
    lower_body_kds,
    upper_body_kds,
):
    """Apply action to the robot"""
    q_t = d.qpos[7: 7 + num_actions]
    dq_t = d.qvel[6: 6 + num_actions]
    tau = np.zeros(num_actions)
    tau_lower = pd_control(
        target_lower_pos, # in sim
        q_t[lower_body_joint2motor_idx],
        lower_body_kps,
        np.zeros_like(lower_body_kds),
        dq_t[lower_body_joint2motor_idx], 
        lower_body_kds)
    for i in range(len(tau_lower)):
        motor_idx = lower_body_joint2motor_idx[i]
        tau[motor_idx] = tau_lower[i]

    tau_upper = pd_control(
        target_upper_pos, # in sim
        q_t[upper_body_joint2motor_idx],
        upper_body_kps,
        np.zeros_like(upper_body_kds),
        dq_t[upper_body_joint2motor_idx], 
        upper_body_kds)
    for i in range(len(tau_upper)):
        motor_idx = upper_body_joint2motor_idx[i]
        tau[motor_idx] = tau_upper[i]
    return tau


def get_camera_frame_pos(data, model, orientation_type='quat'):
    cam_site_id = model.site("d435_camera_frame").id
    cam_pos_world = data.site_xpos[cam_site_id]
    cam_rot_world = data.site_xmat[cam_site_id].reshape(3, 3)


    object_body_id = model.body("object").id
    object_pos_world = data.xpos[object_body_id]
    object_rot_world = data.xmat[object_body_id].reshape(3, 3)

    object_world_transform = np.eye(4)
    object_world_transform[:3, :3] = object_rot_world
    object_world_transform[:3, 3] = object_pos_world

    camera_world_transform = np.eye(4)
    camera_world_transform[:3, :3] = cam_rot_world
    camera_world_transform[:3, 3] = cam_pos_world

    object_camera_transform = np.linalg.inv(camera_world_transform) @ object_world_transform
    object_camera_pos = object_camera_transform[:3, 3]
    object_camera_rotation = Rotation.from_matrix(object_camera_transform[:3, :3])
    object_camera_orientation = None
    object_camera_quat = object_camera_rotation.as_quat()
    object_camera_quat = np.array([object_camera_quat[3], object_camera_quat[0], object_camera_quat[1], object_camera_quat[2]])

    if orientation_type == 'quat':
        object_camera_orientation = object_camera_quat
    elif orientation_type == 'euler':
        object_camera_orientation = object_camera_rotation.as_euler(seq='XYZ', degrees=True)
    elif orientation_type == '6d':
        rotation_matrix = object_camera_transform[:3, :3]
        col1 = rotation_matrix[:, 0]  # First column (3,)
        col2 = rotation_matrix[:, 1]  # Second column (3,)
        object_camera_orientation = np.concatenate([col1, col2], axis=-1)  # Shape: (6,)
    else:
        raise ValueError(f"Invalid orientation type: {orientation_type}")

    return object_camera_pos, object_camera_orientation, object_camera_quat


    

def keyboard_listener():
    global USE_RESIDUAL
    while True:
        
        if select.select([sys.stdin], [], [], 0.1)[0]:
            ch = sys.stdin.read(1)
            if ch.lower() == 'r':
                USE_RESIDUAL = not USE_RESIDUAL
                status = "ENABLED" if USE_RESIDUAL else "DISABLED"
                print(f"\n{'='*50}")
                print(f"Residual Policy: {status}")
                print(f"{'='*50}\n")
        time.sleep(0.05) 


if __name__ == "__main__":
    
    with open(CONFIG_PATH / "g1_mujoco.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        

        #xml_path = 'mujoco/g1_xml/scene_29dof.xml'
        xml_path = 'mujoco/g1_description/g1_tray_holder.xml'

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        policy_joints = config["policy_joints"]
        policy_lower_body_joints = config["policy_lower_body_joints"]
        policy_upper_body_joints = config["policy_upper_body_joints"]
        # idx: sim order, value: real motor id
        upper_body_joint2motor_idx = config["upper_body_joint2motor_idx"]
        upper_body_kps = np.array(config["actuator_config"][ACTUATOR_CONFIG]["upper_body_kps"], dtype=np.float32)
        upper_body_kds = np.array(config["actuator_config"][ACTUATOR_CONFIG]["upper_body_kds"], dtype=np.float32)
        upper_body_default_pos = np.array(config["upper_body_default_pos"], dtype=np.float32)

        # idx: sim order, value: real motor id
        lower_body_joint2motor_idx = config["lower_body_joint2motor_idx"]
        lower_body_kps = np.array(config["actuator_config"][ACTUATOR_CONFIG]["lower_body_kps"], dtype=np.float32)
        lower_body_kds = np.array(config["actuator_config"][ACTUATOR_CONFIG]["lower_body_kds"], dtype=np.float32)
        lower_body_default_pos = np.array(config["lower_body_default_pos"], dtype=np.float32)

        # idx: sim order, value: real motor id
        whole_body_joint2motor_idx = config["whole_body_joint2motor_idx"]
        whole_body_default_pos = config["whole_body_default_pos"]
        num_upper_actions=14
        num_lower_actions=15

        # joint limit 
        upper_body_limit_lower = np.asarray(config["upper_body_limit_lower"])
        upper_body_limit_upper = np.asarray(config["upper_body_limit_upper"])
        lower_body_limit_lower = np.asarray(config["lower_body_limit_lower"])
        lower_body_limit_upper = np.asarray(config["lower_body_limit_upper"])

        upper_body_soft_limit = compute_soft_limit(
            upper_body_limit_lower, upper_body_limit_upper, soft_factor=1.6
        )
        lower_body_soft_limit = compute_soft_limit(
            lower_body_limit_lower, lower_body_limit_upper, soft_factor=1.6
        )

        default_angles = np.array(whole_body_default_pos, dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        upper_body_action_scale = config["actuator_config"][ACTUATOR_CONFIG]["upper_body_action_scale"]
        lower_body_action_scale = config["actuator_config"][ACTUATOR_CONFIG]["lower_body_action_scale"]
        if isinstance(upper_body_action_scale, list):
            upper_body_action_scale = np.array(upper_body_action_scale, dtype=np.float32)
        if isinstance(lower_body_action_scale, list):
            lower_body_action_scale = np.array(lower_body_action_scale, dtype=np.float32)
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    residual_action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    residual_action_buff = CircularBuffer(max_len=5, data_shape=(num_actions,))
    joint_pos_buff = CircularBuffer(max_len=5, data_shape=(num_actions,))
    joint_vel_buff = CircularBuffer(max_len=5, data_shape=(num_actions,))
    vel_command_buff = CircularBuffer(max_len=5, data_shape=(3,))
    gravity_orientation_buff = CircularBuffer(max_len=5, data_shape=(3,))
    angular_velocity_buff = CircularBuffer(max_len=5, data_shape=(3,))
    action_buff = CircularBuffer(max_len=5, data_shape=(num_actions,))
    object_pos_dim = 7 if POSE_TYPE == 'quat' else 9
    object_pos_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(object_pos_dim,))
    object_gravity_orientation_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(3,))

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy_to_xml = []
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        if jname in policy_joints:
            idx = policy_joints.index(jname)
            policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)


    default_angles = default_angles[policy_to_xml]
    target_dof_pos = default_angles.copy()

    target_upper_pos = upper_body_default_pos.copy()
    target_lower_pos = lower_body_default_pos.copy()

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        residual_action_buff.append(residual_action.copy())
        action_buff.append(action.copy())
        joint_pos_buff.append(np.zeros(num_actions, dtype=np.float32))
        joint_vel_buff.append(np.zeros(num_actions, dtype=np.float32))
        vel_command_buff.append(np.zeros(3, dtype=np.float32))
        gravity_orientation_buff.append(np.zeros(3, dtype=np.float32))
        angular_velocity_buff.append(np.zeros(3, dtype=np.float32))
        object_pos_buff.append(np.zeros(object_pos_dim, dtype=np.float32))

        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d) 

    
    # ============================================
    # Move to default pos
    # ============================================
    # reset robot
    for i in range(20):
        tau = apply_action(
            num_actions,
            d,
            lower_body_default_pos,
            upper_body_default_pos,
            lower_body_joint2motor_idx,
            upper_body_joint2motor_idx,
            lower_body_kps,
            upper_body_kps,
            lower_body_kds,
            upper_body_kds,
        )
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


    policy_runner = ResidualPolicyRunner(use_residual=LOAD_RESIDUAL)

    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    print("\n" + "="*50)
    print("Controls:")
    print("  R - Toggle Residual Policy")
    print("="*50)
    print(f"Residual Policy: {'ENABLED' if USE_RESIDUAL else 'DISABLED'}\n")
    

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0  
        viewer.cam.distance = 1.5   
        viewer.cam.elevation = -15
          
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        try:
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()
                tau = apply_action(
                    num_actions,
                    d,
                    target_lower_pos,
                    target_upper_pos,
                    lower_body_joint2motor_idx,
                    upper_body_joint2motor_idx,
                    lower_body_kps,
                    upper_body_kps,
                    lower_body_kds,
                    upper_body_kds,
                )

                d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                counter += 1
                if counter % control_decimation == 0:
                    # Apply control signal here.

                    # create observation
                    qj = d.qpos[7: 7 + num_actions]
                    dqj = d.qvel[6: 6 + num_actions]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    """"
                    Object Observation
                    """
                    # object camera pos
                    camera_frame_pos, camera_frame_orientation, camera_frame_quat = get_camera_frame_pos(d, m, POSE_TYPE)
                    camera_frame_obs = np.concatenate([camera_frame_pos, camera_frame_orientation], axis=0)
                    object_pos_buff.append(camera_frame_obs.copy())
                    # object gravity orientation
                    waist_yaw = d.qpos[7 + 12]
                    waist_roll = d.qpos[7 + 13]
                    waist_pitch = d.qpos[7 + 14]
                    waist_euler_xyz = np.array([waist_roll, waist_pitch, waist_yaw])
                    pelvis_quat = d.qpos[3:7]
                    object_gravity_orientation = get_object_gravity_orientation(camera_frame_quat, waist_euler_xyz, pelvis_quat)
                    object_gravity_orientation_buff.append(object_gravity_orientation.copy())

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale

                    waist_yaw = d.qpos[7: 7 + num_actions][12]
                    waist_yaw_omega = d.qvel[6: 6 + num_actions][12]

                    torso_quat, torso_ang_vel = transform_imu_data_pelvis_to_torso(
                        waist_yaw=waist_yaw, 
                        waist_yaw_omega=waist_yaw_omega, 
                        pelvis_quat=quat, 
                        pelvis_omega=omega
                    )

                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale
                    count = counter * simulation_dt

                    action_buff.append(action.copy())
                    residual_action_buff.append(residual_action.copy())
                    joint_pos_buff.append(qj[xml_to_policy].copy())
                    joint_vel_buff.append(dqj[xml_to_policy].copy())
                    vel_command_buff.append(cmd.copy() * cmd_scale)
                    gravity_orientation_buff.append(gravity_orientation.copy())
                    angular_velocity_buff.append(omega.copy())

                    obs_omega = angular_velocity_buff.buffer.flatten()
                    obs_gravity_orientation = gravity_orientation_buff.buffer.flatten()
                    obs_cmd = vel_command_buff.buffer.flatten()
                    obs_pos = joint_pos_buff.buffer.flatten()
                    obs_vel = joint_vel_buff.buffer.flatten()
                    obs_action = action_buff.buffer.flatten()
                    
                    big_group_major = np.concatenate([
                        obs_omega,
                        obs_gravity_orientation,
                        obs_cmd,
                        obs_pos,
                        obs_vel,
                        obs_action,
                    ], axis=0)
                    big_group_major = np.clip(big_group_major, -100, 100)
                    obs_tensor = torch.from_numpy(big_group_major).float().unsqueeze(0)

                    """
                    Residual Gate
                    """
                    use_residual_this_step = USE_RESIDUAL
                    if use_residual_this_step:
                        waist_yaw = d.qpos[7 + 12]
                        waist_roll = d.qpos[7 + 13]
                        waist_pitch = d.qpos[7 + 14]
                        waist_euler_xyz = np.array([waist_roll, waist_pitch, waist_yaw])
                        pelvis_quat = d.qpos[3:7]
                        torso_bad = is_torso_bad_orientation(waist_euler_xyz, pelvis_quat, 10)
                        if torso_bad:
                            #use_residual_this_step = False
                            print(f"[WARNING] TORSO BAD - Residual DISABLED")
                        
                        object_bad = is_object_bad_orientation(waist_euler_xyz, pelvis_quat, camera_frame_quat, 30)
                        if object_bad:
                            #use_residual_this_step = False
                            print(f"[WARNING] OBJECT BAD - Residual DISABLED")


                    if use_residual_this_step:
                        obs_residual_action = residual_action_buff.buffer.flatten()
                        residual_actor_obs = np.concatenate([
                            obs_omega,
                            obs_gravity_orientation,
                            obs_cmd,
                            obs_pos,
                            obs_vel,
                            obs_residual_action
                        ])
                        residual_actor_obs = np.clip(residual_actor_obs, -100, 100)
                        residual_obs_tensor = torch.from_numpy(residual_actor_obs).float().unsqueeze(0)
                        
                        object_pos = object_pos_buff.buffer
                        object_pos = np.clip(object_pos, -100, 100)
                        object_gravity_orientation = object_gravity_orientation_buff.buffer
                        object_gravity_orientation = np.clip(object_gravity_orientation, -100, 100)
                        object_pos_tensor = torch.from_numpy(object_pos).float().unsqueeze(0)
                        if INCLUDE_OBJECT_GRAVITY_ORIENTATION:
                            object_gravity_orientation_tensor = torch.from_numpy(object_gravity_orientation).float().unsqueeze(0)
                            object_tensor = torch.cat([object_pos_tensor, object_gravity_orientation_tensor], axis=-1)
                        else:
                            object_tensor = object_pos_tensor
                        # residual action
                        residual_action = policy_runner.act_residual(residual_obs_tensor, object_tensor).detach().numpy().squeeze()
                        residual_action = np.clip(residual_action, -100, 100)
                    else:
                        residual_action = np.zeros(num_actions)

                    # base action
                    action = policy_runner.act_base(obs_tensor).detach().numpy().squeeze()
                    action = np.clip(action, -100, 100)
                    final_action = action + residual_action
                    upper_action = final_action[:num_upper_actions]
                    lower_action = final_action[num_upper_actions:]

                    target_lower_pos = lower_action * lower_body_action_scale + lower_body_default_pos
                    target_upper_pos = upper_action * upper_body_action_scale + upper_body_default_pos

                    # check joint limit
                    in_upper_limit = check_joint_limit(target_upper_pos, upper_body_soft_limit)
                    in_lower_limit = check_joint_limit(target_lower_pos[:11], (lower_body_soft_limit[0][:11],lower_body_soft_limit[1][:11]))

                    if not in_upper_limit or not in_lower_limit:
                        target_upper_pos = upper_body_default_pos.copy()
                        target_lower_pos = lower_body_default_pos.copy()

                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nError during simulation: {e}")
        finally:
            time.sleep(0.1)
print("Exiting gracefully...")
time.sleep(0.2)
print("Done")