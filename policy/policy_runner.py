from .joint_policy import ActorCritic
from .residual_policy import ResidualAdaptiveModule
import torch
from config import CHECKPOINT_PATH, CONFIG_PATH
import yaml




class ResidualPolicyRunner():

    def __init__(self):
        # load configs and checkpoints
        self.checkpoint = CHECKPOINT_PATH / 'model_4700.pt'
        self.policy_cfg = CONFIG_PATH / 'policy.yaml'
        assert self.checkpoint.exists(), f"Checkpoint not found: {self.checkpoint}"
        assert self.policy_cfg.exists(), f"Config file not found: {self.policy_cfg}"

        self.body_keys = ['upper_body', 'lower_body', 'residual_whole_body']
        self.obs_keys = ['actor_obs', 'residual_actor_obs', 'encoder_obs']

        # step up policies
        self.__step_up_policy()
        


    def __step_up_policy(self):
        # load state dict
        loaded_dict = torch.load(self.checkpoint, weights_only=False)

        with open(self.policy_cfg, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            upper_body_cfg = config['upper_body_policy']
            lower_body_cfg = config['lower_body_policy']
            residual_wbc_cfg = config['residual_whole_body_policy']

            self.upper_body_policy = ActorCritic(
                num_actions = upper_body_cfg['num_actions'],
                num_actor_obs = upper_body_cfg['num_actor_obs'],
                actor_hidden_dims = upper_body_cfg['actor_hidden_dims'],
                activation = upper_body_cfg['activation']
            )
            load_upper = self.upper_body_policy.load_state_dict(loaded_dict[f"model_state_dict_{upper_body_cfg['body_key']}"])
            if load_upper:
                print('[INFO]: Load Upper Body Policy')

            self.lower_body_policy = ActorCritic(
                num_actions = lower_body_cfg['num_actions'],
                num_actor_obs = lower_body_cfg['num_actor_obs'],
                actor_hidden_dims = lower_body_cfg['actor_hidden_dims'],
                activation = lower_body_cfg['activation']
            )
            load_lower = self.lower_body_policy.load_state_dict(loaded_dict[f"model_state_dict_{lower_body_cfg['body_key']}"])
            if load_lower:
                print('[INFO]: Load Lower Body Policy')

            # self.residual_wbc_policy = ResidualAdaptiveModule(
            #     num_actions = residual_wbc_cfg['num_actions'],
            #     num_actor_obs = residual_wbc_cfg['num_actor_obs'],
            #     num_encoder_obs = residual_wbc_cfg['num_encoder_obs'],
            #     num_time_steps = residual_wbc_cfg['num_time_steps'],
            #     num_encoder_output = residual_wbc_cfg['num_encoder_output'],
            #     actor_hidden_dims = residual_wbc_cfg['actor_hidden_dims'],
            #     encoder_d_model = residual_wbc_cfg['encoder_d_model'],
            #     encoder_nhead = residual_wbc_cfg['encoder_nhead'],
            #     encoder_num_layers = residual_wbc_cfg['encoder_num_layers'],
            #     activation = residual_wbc_cfg['activation']
            # )
            # load_residual = self.residual_wbc_policy.load_state_dict(loaded_dict[f"model_state_dict_{residual_wbc_cfg['body_key']}"])
            # if load_residual:
            #     print('[INFO]: Load Residual WBC Policy')

        self.lower_body_policy.eval()
        self.upper_body_policy.eval()
        #self.residual_wbc_policy.eval()

        #assert load_upper and load_lower and load_residual, 'Failed to load Checkpoint'


    def act_base(self, obs):
        """Act Upper and Lower policies only"""
        base_action_upper = self.upper_body_policy.act_inference(obs)
        base_action_lower = self.lower_body_policy.act_inference(obs)
        base_action = torch.cat([base_action_upper, base_action_lower], dim=-1)
        return base_action
    

    def act_residual(self, obs, encoder_obs):
        """Act Residual WBC"""
        residual_action = self.residual_wbc_policy.act_inference(obs, encoder_obs)
        return residual_action

        



    