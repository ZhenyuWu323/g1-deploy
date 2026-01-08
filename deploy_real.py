import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from typing import Union
import numpy as np
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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
from common.rotation_helper import get_gravity_orientation, transform_imu_data, is_object_bad_orientation, is_torso_bad_orientation
from common.rotation_helper import get_object_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config
from policy.policy_runner import ResidualPolicyRunner
from common.circular_buffer import CircularBuffer
from apriltag_camera import AprilTagDetector
LOAD_RESIDUAL=True
USE_RESIDUAL=False
ENCODER_HISTORY_STEP = 32
SOFT_LIMIT_FACTOR = 1.8
POSE_TYPE = '6d'
ACTUATOR_CONFIG = 'mimic'

class Controller:
    def __init__(self, config: Config, load_residual=False, use_residual=False) -> None:
        if use_residual and not load_residual:
            raise ValueError("Cannot use_residual without load_residual!")
        
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy_runner = ResidualPolicyRunner(use_residual=load_residual)
        self.use_residual = use_residual
        self.load_residual = load_residual

        # Initialize camera
        self.apriltag_detector = None
        if load_residual:
            self.apriltag_detector = AprilTagDetector(
                history_length=ENCODER_HISTORY_STEP,
                pose_type=POSE_TYPE
            )
            self.apriltag_detector.start()
            time.sleep(2)
        # Initializing process variables
        #self.qj = np.zeros(config.num_actions, dtype=np.float32)
        #self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.residual_action = np.zeros(config.num_actions, dtype=np.float32)
        self.upper_body_target = config.upper_body_default_pos.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.residual_obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.button_states = {}

        # Initialize circular buffer
        self.joint_pos_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.joint_vel_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.action_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.ang_vel_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))
        self.projected_gravity_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))
        self.vel_command_buff = CircularBuffer(max_len=config.history_length, data_shape=(3,))
        self.residual_action_buff = CircularBuffer(max_len=config.history_length, data_shape=(config.num_actions,))
        self.object_projected_gravity_buff = None
        self.encoder_residual_action_buff = None
        self.encoder_joint_pos_buff = None
        self.encoder_joint_vel_buff = None
        self.encoder_gravity_orientation_buff = None
        self.encoder_angular_velocity_buff = None
        if load_residual:
            #self.object_projected_gravity_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(3,))
            self.encoder_angular_velocity_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(3,))
            self.encoder_gravity_orientation_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(3,))
            self.encoder_joint_pos_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(config.num_actions,))
            self.encoder_joint_vel_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(config.num_actions,))
            self.encoder_residual_action_buff = CircularBuffer(max_len=ENCODER_HISTORY_STEP, data_shape=(config.num_actions,))
        

        # joint limits
        self.upper_body_limit_lower = np.asarray(config.upper_body_limit_lower)
        self.upper_body_limit_upper = np.asarray(config.upper_body_limit_upper)
        self.lower_body_limit_lower = np.asarray(config.lower_body_limit_lower)
        self.lower_body_limit_upper = np.asarray(config.lower_body_limit_upper)

        # soft limits
        self.upper_body_soft_limit = self.compute_soft_limit(
            self.upper_body_limit_lower, self.upper_body_limit_upper, soft_factor=SOFT_LIMIT_FACTOR
        )
        
        self.lower_body_soft_limit = self.compute_soft_limit(
            self.lower_body_limit_lower, self.lower_body_limit_upper, soft_factor=SOFT_LIMIT_FACTOR
        )

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

    def is_button_pressed(self, key):
        current = self.remote_controller.button[key]
        last = self.button_states.get(key, 0)
        self.button_states[key] = current
        return current == 1 and last == 0


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
        cmd = self.cmd 

        # object projected gravity
        if self.load_residual:
            # waist_yaw = self.low_state.motor_state[12].q
            # waist_roll = self.low_state.motor_state[13].q
            # waist_pitch = self.low_state.motor_state[14].q
            # waist_euler_xyz = np.array([waist_roll, waist_pitch, waist_yaw])
            # pelvis_quat = self.low_state.imu_state.quaternion
            # object_camera_quat = self.apriltag_detector.get_last_quat()
            # object_proj_gravity = get_object_gravity_orientation(object_camera_quat, waist_euler_xyz, pelvis_quat)
            # self.object_projected_gravity_buff.append(object_proj_gravity)

            self.encoder_angular_velocity_buff.append(ang_vel)
            self.encoder_gravity_orientation_buff.append(gravity_orientation)
            self.encoder_joint_pos_buff.append(q_t)
            self.encoder_joint_vel_buff.append(dq_t)
            self.encoder_residual_action_buff.append(self.residual_action.copy())


        # add to history
        self.joint_pos_buff.append(q_t)
        self.joint_vel_buff.append(dq_t)
        self.ang_vel_buff.append(ang_vel)
        self.projected_gravity_buff.append(gravity_orientation)
        self.vel_command_buff.append(cmd)
        self.action_buff.append(self.action.copy())
        self.residual_action_buff.append(self.residual_action.copy())

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # update obs
            self.get_obs()
            #print(self.apriltag_detector.get_last_obs())
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

    def to_cmd_fixed_value(self, analog_value, threshold, max_value, min_value):
        
        if abs(analog_value) <= threshold:
            return 0.0
        elif analog_value > 0:
            return max_value     
        else:
            return min_value

    def compute_soft_limit(self, lower_limit, upper_limit, soft_factor=0.9):
        center = (upper_limit + lower_limit) / 2.0
        half_range = (upper_limit - lower_limit) / 2.0
        soft_lower = center - half_range * soft_factor
        soft_upper = center + half_range * soft_factor
        return (soft_lower, soft_upper)   

    def check_joint_limit(self, action, soft_limit):
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
    
    def run(self):
        start_time = time.time()
        self.counter += 1
        # get obs
        self.get_obs()
        joint_pos = self.joint_pos_buff.buffer.flatten()
        joint_vel = self.joint_vel_buff.buffer.flatten()
        ang_vel = self.ang_vel_buff.buffer.flatten()
        projected_gravity = self.projected_gravity_buff.buffer.flatten()
        actions = self.action_buff.buffer.flatten()
        vel_command = self.vel_command_buff.buffer.flatten()
        residual_actions = self.residual_action_buff.buffer.flatten()
        
        self.cmd[0] = self.to_cmd_fixed_value(self.remote_controller.ly, 0.1, self.config.vel_x_cmd[1], self.config.vel_x_cmd[0])
        self.cmd[1] = self.to_cmd_fixed_value(self.remote_controller.lx * -1, 0.1, self.config.vel_y_cmd[1], self.config.vel_y_cmd[0])
        self.cmd[2] = self.to_cmd_fixed_value(self.remote_controller.rx * -1, 0.1, self.config.yaw_cmd[1], self.config.yaw_cmd[0])

        # base proprio obs
        base_obs = np.concatenate([
            ang_vel,                        # 15 (5 * 3)
            projected_gravity,              # 15 (5 * 3)
            vel_command,                    # 3
            joint_pos,                      # 145 (5 * 29)
            joint_vel,                      # 145 (5 * 29)
        ])

        # joint policy obs
        self.obs = np.concatenate([base_obs, actions])
        self.obs = np.clip(self.obs, -self.config.clip_obervation, self.config.clip_obervation)

        # joint policy action
        obs_tensor = torch.from_numpy(self.obs).float().unsqueeze(0)
        self.action = self.policy_runner.act_base(obs_tensor).detach().numpy().squeeze()
        self.action = np.clip(self.action, -self.config.clip_action, self.config.clip_action)
        
        """
        Check for residual
        """
        use_residual_this_step = self.use_residual
        # if use_residual_this_step:
        #     # check torso
        #     waist_yaw = self.low_state.motor_state[12].q
        #     waist_roll = self.low_state.motor_state[13].q
        #     waist_pitch = self.low_state.motor_state[14].q
        #     waist_euler_xyz = np.array([waist_roll, waist_pitch, waist_yaw])
        #     pelvis_quat = self.low_state.imu_state.quaternion
        #     torso_bad = is_torso_bad_orientation(waist_euler_xyz, pelvis_quat, 10)
        #     if torso_bad:
        #         use_residual_this_step = False
        #         print(f"[WARNING] TORSO BAD - Residual DISABLED")

        #     # check object
        #     if use_residual_this_step and self.apriltag_detector is not None:
        #         object_quat = self.apriltag_detector.get_last_quat()
        #         object_bad = is_object_bad_orientation(waist_euler_xyz, pelvis_quat, object_quat, 30)
        #         if object_bad:
        #             use_residual_this_step = False
        #             print(f"[WARNING] OBJECT BAD - Residual DISABLED")


        if use_residual_this_step:
            # get camera obs
            object_pos_obs = self.apriltag_detector.get_object_obs()
            object_pos_obs = np.clip(object_pos_obs, -100, 100)
            object_pos_obs_tensor = torch.from_numpy(object_pos_obs).float().unsqueeze(0)

            # get object projected gravity
            # object_projected_gravity = self.object_projected_gravity_buff.buffer
            # object_projected_gravity = np.clip(object_projected_gravity, -100, 100)
            # object_projected_gravity_tensor = torch.from_numpy(object_projected_gravity).float().unsqueeze(0)

            # get robot proprioceptive encoder obs
            encoder_proprio_obs = np.concatenate([
                self.encoder_angular_velocity_buff.buffer,
                self.encoder_gravity_orientation_buff.buffer,
                self.encoder_joint_pos_buff.buffer,
                self.encoder_joint_vel_buff.buffer,
                self.encoder_residual_action_buff.buffer,
            ], axis=-1)
            encoder_proprio_obs = np.clip(encoder_proprio_obs, -100, 100)
            encoder_obs_tensor = torch.from_numpy(encoder_proprio_obs).float().unsqueeze(0)
            # concat
            #object_obs_tensor = torch.cat([object_pos_obs_tensor, object_projected_gravity_tensor], axis=-1)
            object_obs_tensor = torch.cat([encoder_obs_tensor, object_pos_obs_tensor], axis=-1)

            # get residual actor obs
            self.residual_obs = np.concatenate([base_obs, residual_actions])
            self.residual_obs = np.clip(self.residual_obs, -self.config.clip_obervation, self.config.clip_obervation)
            residual_obs_tensor = torch.from_numpy(self.residual_obs).float().unsqueeze(0)
            self.residual_action = self.policy_runner.act_residual(residual_obs_tensor, object_obs_tensor).detach().numpy().squeeze()
            self.residual_action = np.clip(self.residual_action, -self.config.clip_action, self.config.clip_action)
        else:
            self.residual_action = np.zeros(self.config.num_actions, dtype=np.float32)
        
        # transform action to target_dof_pos
        final_action = self.action + self.residual_action if use_residual_this_step else self.action
        upper_body_actions = final_action[:self.config.num_upper_actions]
        upper_body_target = self.config.upper_body_default_pos + upper_body_actions * self.config.upper_body_action_scale

        lower_body_actions = final_action[self.config.num_upper_actions:]
        lower_body_target = self.config.lower_body_default_pos + lower_body_actions * self.config.lower_body_action_scale
        

        # check joint limit
        in_upper_limit = self.check_joint_limit(upper_body_target, self.upper_body_soft_limit)
        in_lower_limit = self.check_joint_limit(lower_body_target[3:11], (self.lower_body_soft_limit[0][3:11],self.lower_body_soft_limit[1][3:11]))

        if not in_upper_limit or not in_lower_limit:
            print(f"[WARNING] JOINT LIMIT VIOLATION - SKIPPING COMMAND")
            return 0 # Skip sending command if joint limits are violated


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
        control_duration = time.time() - start_time
        time_til_next_step = self.config.control_dt - control_duration
        if time_til_next_step < 0:
            print('[WARNING] control over time')
        else:
            time.sleep(time_til_next_step)

        return 1



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    #parser.add_argument("use_residual", type=bool, default=False, help='use residual network')
    #parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    #config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config()

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config, LOAD_RESIDUAL, USE_RESIDUAL)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()


    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            if controller.is_button_pressed(KeyMap.B):
                controller.use_residual = not controller.use_residual
                status = "ENABLED" if controller.use_residual else "DISABLED"
                print(f"\n{'='*50}")
                print(f"Residual Policy: {status}")
                print(f"{'='*50}\n")
            run = controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1 or run == 0:
                print("Killed By SELECTED")
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    if LOAD_RESIDUAL:
        controller.apriltag_detector.stop()
    print("Exit")
