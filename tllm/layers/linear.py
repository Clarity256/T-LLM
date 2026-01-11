import torch
import torch.nn as nn

class QKVParallelLinear(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super().__init__()
        self.Linear = nn.Linear(in_feature, out_feature, bias = bias)
    
    def forward(self, x):
        return self.Linear(x)
    
    @property
    def weight(self):
        return self.Linear.weight
    
    @property
    def bias(self):
        return self.Linear.bias
    
class RowParallelLinear(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super().__init__()
        self.Linear = nn.Linear(in_feature, out_feature, bias = bias)
    
    def forward(self, x):
        return self.Linear(x)
    
    @property
    def weight(self):
        return self.Linear.weight
    
    @property
    def bias(self):
        return self.Linear.bias