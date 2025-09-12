from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from policy.policy_runner import ResidualPolicyRunner
from common.circular_buffer import CircularBuffer


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy_runner = ResidualPolicyRunner()
        # Initializing process variables
        #self.qj = np.zeros(config.num_actions, dtype=np.float32)
        #self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.upper_body_target = config.upper_body_default_pos.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # Initialize circular buffer
        self.joint_pos_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.joint_vel_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.action_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.ang_vel_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))
        self.projected_gravity_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))
        self.vel_command_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.upper_body_joint2motor_idx + self.config.lower_body_joint2motor_idx
        kps = self.config.upper_body_kps + self.config.lower_body_kps
        kds = self.config.upper_body_kds + self.config.lower_body_kds
        default_pos = np.concatenate((self.config.upper_body_default_pos, self.config.lower_body_default_pos), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    def get_obs(self):
        # joint pos and joint vel
        q_t =  np.zeros(self.config.num_actions, dtype=np.float32)
        dq_t = np.zeros(self.config.num_actions, dtype=np.float32)

        for i in range(len(self.config.whole_body_joint2motor_idx)):
            q_t[i] = self.low_state.motor_state[self.config.whole_body_joint2motor_idx[i]].q
            dq_t[i] = self.low_state.motor_state[self.config.whole_body_joint2motor_idx[i]].dq

        q_t = (q_t - self.config.whole_body_default_pos) * self.config.dof_pos_scale
        dq_t = dq_t * self.config.dof_vel_scale
        
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
        ang_vel = ang_vel * self.config.ang_vel_scale

        # projected gravity
        gravity_orientation = get_gravity_orientation(quat)

        # command
        cmd = self.cmd * self.config.cmd_scale * self.config.max_cmd

        # add to history
        self.joint_pos_buff.append(q_t)
        self.joint_vel_buff.append(dq_t)
        self.ang_vel_buff.append(ang_vel)
        self.projected_gravity_buff.append(gravity_orientation)
        self.vel_command_buff.append(cmd)
        self.action_buff.append(self.action.copy())

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # update obs
            self.get_obs()
            # set upper body
            for i in range(len(self.config.upper_body_joint2motor_idx)):
                motor_idx = self.config.upper_body_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.upper_body_default_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.upper_body_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.upper_body_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # set lower body
            for i in range(len(self.config.lower_body_joint2motor_idx)):
                motor_idx = self.config.lower_body_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.lower_body_default_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.lower_body_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.lower_body_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # get obs
        self.get_obs()
        joint_pos = self.joint_pos_buff.get_history(self.config.history_length).flatten()
        joint_vel = self.joint_vel_buff.get_history(self.config.history_length).flatten()
        ang_vel = self.ang_vel_buff.get_history(self.config.history_length).flatten()
        projected_gravity = self.projected_gravity_buff.get_history(self.config.history_length).flatten()
        actions = self.action_buff.get_history(self.config.history_length).flatten()
        vel_command = self.vel_command_buff.get_history(self.config.history_length).flatten()

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        self.obs = np.concatenate([
            ang_vel,                        # 15 (5 * 3)
            projected_gravity,              # 15 (5 * 3)  
            #self.cmd * self.config.cmd_scale * self.config.max_cmd,                       # 3
            #self.upper_body_target,  # 14
            vel_command,
            joint_pos,                      # 145 (5 * 29)
            joint_vel,                      # 145 (5 * 29)
            actions                          # 145 (5 * 29)
        ])
        self.obs = np.clip(self.obs, -self.config.clip_obervation, self.config.clip_obervation)
        print(f"Observation: {self.obs}")
        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).float().unsqueeze(0)
        self.action = self.policy_runner.act_base(obs_tensor).detach().numpy().squeeze()
        self.action = np.clip(self.action, -self.config.clip_action, self.config.clip_action)
        print(f"Action: {self.action}")
        
        # transform action to target_dof_pos
        upper_body_actions = self.action[:self.config.num_upper_actions]
        upper_body_target = self.config.upper_body_default_pos + upper_body_actions * self.config.action_scale

        lower_body_actions = self.action[self.config.num_upper_actions:]
        lower_body_target = self.config.lower_body_default_pos + lower_body_actions * self.config.action_scale
        
        # Build low cmd

        # Upper body
        for i in range(len(self.config.upper_body_joint2motor_idx)):
            motor_idx = self.config.upper_body_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = upper_body_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.upper_body_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.upper_body_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        # Lower Body
        for i in range(len(self.config.lower_body_joint2motor_idx)):
            motor_idx = self.config.lower_body_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = lower_body_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.lower_body_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.lower_body_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    #parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    #config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config()

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
