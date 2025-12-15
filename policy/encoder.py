import torch
import torch.nn as nn
import math
## NOTE: OLD ENCODER
# class TransformerEncoder(nn.Module):

#     def __init__(
#         self, 
#         num_encoder_obs=7, 
#         d_model=32, 
#         nhead=2, 
#         num_layers=2, 
#         output_dim=32, 
#         num_time_steps=5
#     ):
#         super().__init__()
#         self.d_model = d_model
        
#         # input projection
#         self.input_projection = nn.Linear(num_encoder_obs, d_model)
        
#         # time step position encoding
#         self.positional_encoding = nn.Parameter(torch.randn(num_time_steps, d_model))
        
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.output_projection = nn.Linear(d_model, output_dim)
        
#     def forward(self, x):
#         # x: (batch_size, 5, 21)
#         x = self.input_projection(x)  # (batch_size, 5, d_model)
#         x = x + self.positional_encoding.unsqueeze(0) 
#         x = self.transformer(x)  # (batch_size, 5, d_model)
#         x = x.mean(dim=1)  # Global average pooling: (batch_size, d_model)
#         x = self.output_projection(x)  # (batch_size, output_dim)
#         return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,num_encoder_obs=21, 
            d_model=64, 
            nhead=4, 
            num_layers=2, 
            output_dim=32, 
            num_time_steps=5,
            use_sinusoidal_pe=False
        ):
        super().__init__()
        self.d_model = d_model
        
        # input projection
        self.input_projection = nn.Linear(num_encoder_obs, d_model)

        # layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # time step position encoding
        if use_sinusoidal_pe:
            pe = self._create_sinusoidal_pe(num_time_steps, d_model)
            self.register_buffer('positional_encoding', pe)
        else:
            self.positional_encoding = nn.Parameter(torch.randn(num_time_steps, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def _create_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe 
        
    def forward(self, x):
        # x: (batch_size, num_time_steps, num_encoder_obs)
        x = self.input_projection(x)  # (batch_size, num_time_steps, d_model)
        x = self.layer_norm(x)
        x = x + self.positional_encoding.unsqueeze(0) 
        x = self.transformer(x)  # (batch_size, num_time_steps, d_model)
        x = x[:, -1, :]
        x = self.output_projection(x)  # (batch_size, output_dim)
        return x