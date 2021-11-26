from collections import deque
from collections import OrderedDict 

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from .LowMemConv import LowMemConvBase


def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':512}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':64}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConv(**new_args)


class MalConv(LowMemConvBase):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, embd_size=8, log_stride=None):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2**log_stride
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)

        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    def processRange(self, x):
        x = self.embd(x)
        x = torch.transpose(x,-1,-2)
         
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x, penult, post_conv
