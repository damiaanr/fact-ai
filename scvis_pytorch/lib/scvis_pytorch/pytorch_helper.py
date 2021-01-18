import torch
import torch.nn as nn

def weight_xavier_relu(fan_in_out):
    initial_w = torch.empty(fan_in_out[0], dtype=torch.float32)
    initial_w = nn.init.xavier_normal(initial_w)
    return initial_w

def bias_variable(fan_in_out, mean=0.1):
    tensor = torch.ones(fan_in_out, dtype=torch.float32) * mean
    return tensor

def shape(tensor: torch.Tensor):
    return list(tensor.size())
