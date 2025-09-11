import time
import argparse
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from collections import deque
from policy.policy_runner import ResidualPolicyRunner
from config import CONFIG_PATH


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
    
    with open(CONFIG_PATH / "g1_mujoco.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        

        xml_path = 'mujoco/g1_xml/scene_29dof.xml'

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        policy_joints = config["policy_joints"]
        policy_lower_body_joints = config["policy_lower_body_joints"]
        policy_upper_body_joints = config["policy_upper_body_joints"]
        # idx: sim order, value: real motor id
        upper_body_joint2motor_idx = [15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        upper_body_kps=[40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
        upper_body_kds=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        upper_body_default_pos=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #TODO: print check

        # idx: sim order, value: real motor id
        lower_body_joint2motor_idx=[12, 13, 14, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        lower_body_kps=[200, 200, 200, 100, 100, 100, 100, 100, 100, 150, 150, 40, 40, 40, 40]
        lower_body_kds=[5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2]
        lower_body_default_pos=[0, 0, 0, -0.1, -0.1, 0, 0, 0, 0, 0.3, 0.3, -0.2, -0.2, 0, 0] #TODO: print check

        # idx: sim order, value: real motor id
        whole_body_joint2motor_idx=[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        whole_body_default_pos=[-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0,
        0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0] # order: sim order
        num_upper_actions=14
        num_lower_actions=15

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        #num_obs = config["num_obs"]
        num_obs = 93
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy_to_xml = []
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        idx = policy_joints.index(jname)
        policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)

    # lower_policy_to_xml = []
    # for i in range(1, m.njnt):
    #     jname = mujoco.mj_id2name(m, 3, i)
    #     if jname in policy_lower_body_joints:
    #         idx = policy_lower_body_joints.index(jname)
    #         lower_policy_to_xml.append(idx)
    
    # lower_xml_to_policy = []
    # for i in range(len(lower_policy_to_xml)):
    #     idx = lower_policy_to_xml.index(i)
    #     lower_xml_to_policy.append(idx)

    # print("lower_policy_to_xml:", lower_policy_to_xml)
    # print("lower_xml_to_policy:", lower_xml_to_policy)


    # upper_policy_to_xml = []
    # for i in range(1, m.njnt):
    #     jname = mujoco.mj_id2name(m, 3, i)
    #     if jname in policy_upper_body_joints:
    #         print(jname)
    #         idx = policy_upper_body_joints.index(jname)
    #         upper_policy_to_xml.append(idx)
    
    # upper_xml_to_policy = []
    # for i in range(len(upper_policy_to_xml)):
    #     idx = upper_policy_to_xml.index(i)
    #     upper_xml_to_policy.append(idx)

    # print("upper_policy_to_xml:", upper_policy_to_xml)
    # print("upper_xml_to_policy:", upper_xml_to_policy)

    default_angles = default_angles[policy_to_xml]
    target_dof_pos = default_angles.copy()

    target_upper_pos = upper_body_default_pos.copy()
    target_lower_pos = lower_body_default_pos.copy()

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d) 


    # load policy
    #policy = torch.jit.load(policy_path)

    policy_runner = ResidualPolicyRunner()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            q_t = d.qpos[7:]
            dq_t = d.qvel[6:]
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


            #tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                count = counter * simulation_dt

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                #obs[6:9] = cmd * cmd_scale
                obs[6 : 6 + num_actions] = qj[xml_to_policy]
                obs[6 + num_actions : 6 + 2 * num_actions] = dqj[xml_to_policy]
                obs[6 + 2 * num_actions : 6 + 3 * num_actions] = action

                frame_stack.append(obs.copy())
                stacked_obs = np.concatenate(frame_stack, axis=0)
                
                
                obs_omega = np.asarray(stacked_obs).reshape(5, 93)[:, 0:3].reshape(-1)
                obs_gravity_orientation = np.asarray(stacked_obs).reshape(5, 93)[:, 3:6].reshape(-1)
                #obs_cmd = np.asarray(stacked_obs).reshape(5, 93)[:, 6:9].reshape(-1)
                obs_pos = np.asarray(stacked_obs).reshape(5, 93)[:, 6:6 + num_actions].reshape(-1)
                obs_vel = np.asarray(stacked_obs).reshape(5, 93)[:, 6 + num_actions : 6 + 2 * num_actions].reshape(-1)
                obs_action = np.asarray(stacked_obs).reshape(5, 93)[:, 6 + 2 * num_actions : 6 + 3 * num_actions].reshape(-1)
                big_group_major = np.concatenate([
                    obs_omega,
                    obs_gravity_orientation,
                    cmd * cmd_scale,
                    upper_body_default_pos,
                    obs_pos,
                    obs_vel,
                    obs_action,
                ], axis=0)
                big_group_major = np.clip(big_group_major, -100, 100)
                obs_tensor = torch.from_numpy(big_group_major).float().unsqueeze(0)
                

                # policy inference
                action = policy_runner.act_base(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -100, 100)
                upper_action = action[:num_upper_actions]
                lower_action = action[num_upper_actions:]

                target_lower_pos = lower_action * action_scale + lower_body_default_pos
                target_upper_pos = upper_action * action_scale + upper_body_default_pos
                #action = policy(obs_tensor).detach().numpy().squeeze()
                #action = action[policy_to_xml]
                # transform action to target_dof_pos
                #target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
