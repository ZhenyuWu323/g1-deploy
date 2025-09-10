import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = PROJECT_ROOT / 'configs'
CHECKPOINT_PATH = PROJECT_ROOT / 'policy' / 'checkpoints'

class Config:
    def __init__(self) -> None:
        file_path = CONFIG_PATH / 'g1.yaml'
        assert file_path.exists(), f"Config {file_path} doesn't exists"
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            # Upper Body
            self.upper_body_joint2motor_idx = config["upper_body_joint2motor_idx"]
            self.upper_body_kps = config["upper_body_kps"]
            self.upper_body_kds = config["upper_body_kds"]
            self.upper_body_default_pos = np.array(config["upper_body_default_pos"], dtype=np.float32)
            self.num_upper_actions = config["num_upper_actions"]

            # Lower Body
            self.lower_body_joint2motor_idx = config["lower_body_joint2motor_idx"]
            self.lower_body_kps = config["lower_body_kps"]
            self.lower_body_kds = config["lower_body_kds"]
            self.lower_body_default_pos = np.array(config["lower_body_default_pos"], dtype=np.float32)
            self.num_lower_actions = config["num_lower_actions"]

            # Whole Body
            self.whole_body_joint2motor_idx = config["whole_body_joint2motor_idx"]
            self.whole_body_default_pos = np.array(config["whole_body_default_pos"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.clip_action = config["clip_action"]
            self.clip_obervation = config["clip_observation"]
            self.history_length = config["history_length"]
