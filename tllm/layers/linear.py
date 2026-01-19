import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class LinearBase(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
            tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        if dist.is_available() and dist.is_initialized():
            tp_rank = dist.get_rank()
            tp_size = dist.get_world_size()
        else:
            tp_rank, tp_size = 0, 1
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class QKVParallelLinear(LinearBase):
    def __init__(
            self,
            hidden_size: int,
            head_dim: int,
            num_heads: int,
            num_kv_heads: int,
            bias: bool = False,
    ):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size, bias = bias)
    
    def forward(self, x):
        return self.Linear(x)
    
    @property
    def weight(self):
        return self.Linear.weight
    
    @property
    def bias(self):
        return self.Linear.bias
    
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias = False):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size, bias = bias)
    
    def forward(self, x):
        return self.Linear(x)
    
    @property
    def weight(self):
        return self.Linear.weight
    
    @property
    def bias(self):
        return self.Linear.bias