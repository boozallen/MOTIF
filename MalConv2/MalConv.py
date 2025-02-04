from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

@dataclass
class MalConvConfig:
    out_size: int = 2
    channels: int = 128
    window_size: int = 512
    stride: Optional[int] = None
    embd_size: int = 8
    log_stride: Optional[int] = None
    dropout_rate: float = 0.1

def get_optuna_params() -> dict:
    """Define hyperparameter search space for Optuna optimization."""
    return {
        'channels': {
            'type': 'suggest_int',
            'params': {'name': 'channels', 'low': 32, 'high': 1024}
        },
        'log_stride': {
            'type': 'suggest_int',
            'params': {'name': 'log2_stride', 'low': 2, 'high': 9}
        },
        'window_size': {
            'type': 'suggest_int',
            'params': {'name': 'window_size', 'low': 32, 'high': 512}
        },
        'embd_size': {
            'type': 'suggest_int',
            'params': {'name': 'embd_size', 'low': 4, 'high': 64}
        }
    }

class MalConv(nn.Module):
    def __init__(self, config: MalConvConfig):
        super().__init__()
        
        self.stride = 2 ** config.log_stride if config.log_stride is not None else config.stride
        if self.stride is None:
            raise ValueError("Either stride or log_stride must be provided")

        self.embd = nn.Embedding(257, config.embd_size, padding_idx=0)
        
        conv_params = {
            'in_channels': config.embd_size,
            'out_channels': config.channels,
            'kernel_size': config.window_size,
            'stride': self.stride,
            'bias': True
        }
        
        self.conv_1 = nn.Conv1d(**conv_params)
        self.conv_2 = nn.Conv1d(**conv_params)
        
        self.fc = nn.Sequential(
            nn.Linear(config.channels, config.channels),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.channels, config.out_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def process_range(self, x: Tensor) -> Tensor:
        """Process input through embedding and convolutional layers."""
        x = self.embd(x).transpose(-1, -2)
        return self.conv_1(x) * torch.sigmoid(self.conv_2(x))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning (output, penultimate, post_conv)."""

        post_conv = checkpoint(self.process_range, x) if self.training else self.process_range(x)
        
        x = torch.max(post_conv, dim=-1)[0]
        penult = x = self.fc[:-1](x) 
        x = self.fc[-1](x)  
        
        return x, penult, post_conv

def init_model(**kwargs) -> MalConv:
    """Initialize MalConv model with given parameters."""
    config = MalConvConfig(**{k: v for k, v in kwargs.items() if k in MalConvConfig.__annotations__})
    return MalConv(config)
