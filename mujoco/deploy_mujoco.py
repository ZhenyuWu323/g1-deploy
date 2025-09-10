import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from config import Config
from policy.policy_runner import ResidualPolicyRunner
from common.circular_buffer import CircularBuffer, IsaacCircularBuffer

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



if __name__ == "__main__":
    # get config file name from command line
    policy_runner = ResidualPolicyRunner()
    config = Config()
    simulation_duration = 60.0
    simulation_dt = 0.002
    control_decimation = 10
    cmd = np.array([0.0, 0, 0])
    ang_vel_scale = config.ang_vel_scale
    dof_pos_scale = config.dof_pos_scale
    dof_vel_scale = config.dof_vel_scale
    action_scale = config.action_scale
    cmd_scale = np.array(config.cmd_scale, dtype=np.float32)
    clip_action = config.clip_action
    clip_observation = config.clip_obervation
    history_length = config.history_length

    # body
    upper_body_joint2motor_idx = config.upper_body_joint2motor_idx
    upper_body_kps = config.upper_body_kps
    upper_body_kds = config.upper_body_kds
    upper_body_default_pos = config.upper_body_default_pos
    num_upper_actions = config.num_upper_actions

    lower_body_joint2motor_idx = config.lower_body_joint2motor_idx
    lower_body_kps = config.lower_body_kps
    lower_body_kds = config.lower_body_kds
    lower_body_default_pos = config.lower_body_default_pos
    num_lower_actions = config.num_lower_actions

    whole_body_joint2motor_idx = config.whole_body_joint2motor_idx
    whole_body_default_pos = config.whole_body_default_pos
    num_actions = config.num_actions
    num_obs = config.num_obs


    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    upper_body_target = upper_body_default_pos.copy()
    lower_body_target = lower_body_default_pos.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0


    # Initialize circular buffer
    joint_pos_buff = IsaacCircularBuffer(max_len=config.history_length,batch_size=1, device="cpu")
    joint_vel_buff = IsaacCircularBuffer(max_len=config.history_length,batch_size=1, device="cpu")
    action_buff = IsaacCircularBuffer(max_len=config.history_length,batch_size=1, device="cpu")
    ang_vel_buff = IsaacCircularBuffer(max_len=config.history_length,batch_size=1, device="cpu")
    projected_gravity_buff = IsaacCircularBuffer(max_len=config.history_length,batch_size=1, device="cpu")


    # Load robot model
    xml_path = "./g1_description/g1_29dof.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    print("=== MuJoCo Joint Information ===")
    print(f"Total joints in model: {m.njnt}")
    print(f"Total actuators: {m.nu}")

    print("\nJoint names and indices:")
    for i in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name != "floating_base_joint":  # 跳过自由关节
            print(f"Joint {i}: {joint_name}")

    print("\nActuator names and indices:")
    for i in range(m.nu):
        actuator_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"Actuator {i}: {actuator_name}")




    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            all_joint_pos = d.qpos[7:]  # shape: (29,)
            all_joint_vel = d.qvel[6:]  # shape: (29,)
            
            
            upper_body_action = pd_control(upper_body_target, all_joint_pos[upper_body_joint2motor_idx], upper_body_kps, np.zeros_like(upper_body_kds), all_joint_vel[upper_body_joint2motor_idx], upper_body_kds)
            lower_body_action = pd_control(lower_body_target, all_joint_pos[lower_body_joint2motor_idx], lower_body_kps, np.zeros_like(lower_body_kds), all_joint_vel[lower_body_joint2motor_idx], lower_body_kds)
            #tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            #d.ctrl[:] = tau
            d.ctrl[upper_body_joint2motor_idx] = upper_body_action
            d.ctrl[lower_body_joint2motor_idx] = lower_body_action


            # fill buffer with default
            for i in range(history_length):
                # create observation
                mujoco_joint_pos = d.qpos[7:]
                sim_ordered_pos = np.zeros(len(whole_body_joint2motor_idx), dtype=np.float32)
                for i in range(len(sim_ordered_pos)):
                    motor_idx = whole_body_joint2motor_idx[i]
                    sim_ordered_pos[i] = mujoco_joint_pos[motor_idx]

                mujoco_joint_vel = d.qvel[6:]
                sim_ordered_vel = np.zeros(len(whole_body_joint2motor_idx), dtype=np.float32)
                for i in range(len(sim_ordered_vel)):
                    motor_idx = whole_body_joint2motor_idx[i]
                    sim_ordered_vel[i] = mujoco_joint_vel[motor_idx]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                q_t = (sim_ordered_pos - whole_body_default_pos) * dof_pos_scale
                dq_t = sim_ordered_vel * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # add to history
                joint_pos_buff.append(torch.from_numpy(sim_ordered_pos).float().unsqueeze(0))
                joint_vel_buff.append(torch.from_numpy(sim_ordered_vel).float().unsqueeze(0))
                action_buff.append(torch.from_numpy(action).float().unsqueeze(0))
                ang_vel_buff.append(torch.from_numpy(omega).float().unsqueeze(0))
                projected_gravity_buff.append(torch.from_numpy(gravity_orientation).float().unsqueeze(0))


            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                mujoco_joint_pos = d.qpos[7:]
                sim_ordered_pos = np.zeros(len(whole_body_joint2motor_idx), dtype=np.float32)
                for i in range(len(sim_ordered_pos)):
                    motor_idx = whole_body_joint2motor_idx[i]
                    sim_ordered_pos[i] = mujoco_joint_pos[motor_idx]

                mujoco_joint_vel = d.qvel[6:]
                sim_ordered_vel = np.zeros(len(whole_body_joint2motor_idx), dtype=np.float32)
                for i in range(len(sim_ordered_vel)):
                    motor_idx = whole_body_joint2motor_idx[i]
                    sim_ordered_vel[i] = mujoco_joint_vel[motor_idx]

                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                q_t = (sim_ordered_pos - whole_body_default_pos) * dof_pos_scale
                dq_t = sim_ordered_vel * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # add to history
                joint_pos_buff.append(torch.from_numpy(sim_ordered_pos).float().unsqueeze(0))
                joint_vel_buff.append(torch.from_numpy(sim_ordered_vel).float().unsqueeze(0))
                action_buff.append(torch.from_numpy(action).float().unsqueeze(0))
                ang_vel_buff.append(torch.from_numpy(omega).float().unsqueeze(0))
                projected_gravity_buff.append(torch.from_numpy(gravity_orientation).float().unsqueeze(0))
                

                joint_pos_buff_flat =joint_pos_buff.buffer.reshape(1, -1)
                joint_vel_buff_flat =joint_vel_buff.buffer.reshape(1, -1)
                action_buff_flat =action_buff.buffer.reshape(1, -1)
                ang_vel_buff_flat =ang_vel_buff.buffer.reshape(1, -1)
                projected_gravity_buff_flat =projected_gravity_buff.buffer.reshape(1, -1)

                obs_tensor = torch.cat([
                    ang_vel_buff_flat,
                    projected_gravity_buff_flat,
                    torch.from_numpy(cmd).float().unsqueeze(0).to(device="cpu"),
                    torch.from_numpy(upper_body_default_pos).float().unsqueeze(0).to(device="cpu"),
                    joint_pos_buff_flat,
                    joint_vel_buff_flat,
                    action_buff_flat,
                ], dim=-1).to(device="cpu")

                obs_tensor = torch.clip(obs_tensor, -clip_observation, clip_observation)
                print(obs_tensor)
                # policy inference
                action = policy_runner.act_base(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -clip_action, clip_action)
                # transform action to target_dof_pos
                upper_body_actions = action[:num_upper_actions]
                upper_body_target = upper_body_default_pos + upper_body_actions * action_scale
                lower_body_actions = action[num_upper_actions:]
                lower_body_target = lower_body_default_pos + lower_body_actions * action_scale

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)