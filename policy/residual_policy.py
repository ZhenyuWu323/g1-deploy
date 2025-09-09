# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import resolve_nn_activation
from .encoder import TransformerEncoder

class ResidualAdaptiveModule(nn.Module):

    def __init__(
        self,
        num_actor_obs=482,
        num_actions=29,
        num_encoder_obs=7,
        num_time_steps=5,
        num_encoder_output=32,
        actor_hidden_dims=[512, 256, 128],
        encoder_d_model=32,
        encoder_nhead=2,
        encoder_num_layers=2,
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ResidualAdaptiveModule.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)


        self.num_encoder_obs = num_encoder_obs
        self.num_time_steps = num_time_steps
        self.encoder_total_dim = num_time_steps * num_encoder_obs

        mlp_input_dim_a = int(num_actor_obs + num_encoder_output)
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        self.actor.eval()


        # Encoder
        self.encoder = TransformerEncoder(num_encoder_obs, encoder_d_model, encoder_nhead, encoder_num_layers, num_encoder_output, num_time_steps)
        self.encoder.eval()
        print(f"Actor MLP: {self.actor}")
        print(f"Encoder: {self.encoder}")


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    

    def act_inference(self, observations, encoder_obs):
        with torch.no_grad():
            encoder_obs = encoder_obs.view(encoder_obs.shape[0], self.num_time_steps, self.num_encoder_obs)
            encoded_obs = self.encoder(encoder_obs)
            actor_input = torch.cat([observations, encoded_obs], dim=-1)
            actions_mean = self.actor(actor_input)
        return actions_mean


    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-encoder model."""
        encoder_state_dict = {}
        actor_state_dict = {}
        for key, value in state_dict.items():
            if "student_encoder." in key:
                encoder_state_dict[key.replace("student_encoder.", "")] = value
            elif "actor." in key:
                actor_state_dict[key.replace("actor.", "")] = value

        self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        self.actor.load_state_dict(actor_state_dict, strict=strict)
        return True

