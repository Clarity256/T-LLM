import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist

# def _tp_info():
#     if dist.is_available() and dist.is_initialized():
#             return dist.get_rank(), dist.get_world_size()
#     return 0, 1

class LinearBase(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
            tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        # self.tp_rank, self.tp_size = _tp_info()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
        """
        tp will be coming soon...
        """
        param.copy_(load_weights.to(param.device, dtype=param.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    """
    split output_size (weight dim0). Optional gather_output to keep caller unchanged.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias=bias, tp_dim=1)


class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
        super().__init__(input_size, output_size, bias=bias, tp_dim=1)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
            self,
            hidden_size: int,
            head_dim: int,
            num_heads: int,
            num_kv_heads: int,
            bias: bool = False,
    ):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_size = self.hidden_size * self.num_heads
        self.kv_size = self.hidden_size * self.num_kv_heads
        out_features = self.q_size + 2 * self.kv_size
        
        super().__init__(
            input_size = self.hidden_size,
            output_size = out_features,
            bias = bias,
        )
    

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist


# def _tp_info():
#     if dist.is_available() and dist.is_initialized():
#         return dist.get_rank(), dist.get_world_size()
#     return 0, 1


# def _shard_range(full: int, rank: int, world: int):
#     assert full % world == 0, f"size {full} must be divisible by world_size {world}"
#     part = full // world
#     start = rank * part
#     end = start + part
#     return start, end, part


# class LinearBase(nn.Module):
#     """
#     Base: holds full-shape params (or local-shape params decided by subclass),
#     provides common init & an overridable weight_loader.
#     """
#     def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
#         super().__init__()
#         self.input_size = int(input_size)
#         self.output_size = int(output_size)
#         self.tp_dim = tp_dim

#         self.tp_rank, self.tp_size = _tp_info()

#         self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
#         self.weight.weight_loader = self.weight_loader  # for external loader

#         if bias:
#             self.bias = nn.Parameter(torch.empty(self.output_size))
#             self.bias.weight_loader = self.weight_loader
#         else:
#             self.register_parameter("bias", None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             bound = 1 / math.sqrt(self.input_size) if self.input_size > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
#         raise NotImplementedError

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError


# class ColumnParallelLinear(LinearBase):
#     """
#     Split output_size (weight dim0). Optionally gather_output to keep caller unchanged.
#     """
#     def __init__(self, input_size: int, output_size: int, bias: bool = False, gather_output: bool = False):
#         self.full_output_size = int(output_size)
#         self.tp_rank, self.tp_size = _tp_info()
#         start, end, local_out = _shard_range(self.full_output_size, self.tp_rank, self.tp_size)

#         self.local_out = local_out
#         self.out_start, self.out_end = start, end
#         self.gather_output = gather_output

#         super().__init__(input_size=input_size, output_size=local_out, bias=bias, tp_dim=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = F.linear(x, self.weight, self.bias)  # [..., local_out]
#         if not self.gather_output or self.tp_size == 1:
#             return y

#         # gather outputs along last dim
#         chunks = [torch.empty_like(y) for _ in range(self.tp_size)]
#         dist.all_gather(chunks, y)
#         return torch.cat(chunks, dim=-1)

#     @torch.no_grad()
#     def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
#         """
#         load_weights is FULL weight/bias; copy local shard.
#         weight: [full_out, in]
#         bias:   [full_out]
#         """
#         if load_weights.dim() == 2:  # weight
#             shard = load_weights[self.out_start:self.out_end, :]
#             param.copy_(shard)
#         elif load_weights.dim() == 1:  # bias
#             shard = load_weights[self.out_start:self.out_end]
#             param.copy_(shard)
#         else:
#             raise ValueError(f"Unexpected load_weights shape: {load_weights.shape}")


# class RowParallelLinear(LinearBase):
#     """
#     Split input_size (weight dim1). Forward does x slice + all_reduce.
#     """
#     def __init__(self, input_size: int, output_size: int, bias: bool = False):
#         self.full_input_size = int(input_size)
#         self.tp_rank, self.tp_size = _tp_info()
#         start, end, local_in = _shard_range(self.full_input_size, self.tp_rank, self.tp_size)

#         self.local_in = local_in
#         self.in_start, self.in_end = start, end

#         super().__init__(input_size=local_in, output_size=output_size, bias=bias, tp_dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [..., full_in] or already-sharded [..., local_in]
#         if x.size(-1) == self.full_input_size:
#             x = x[..., self.in_start:self.in_end]
#         y = F.linear(x, self.weight, self.bias)  # [..., out]
#         if self.tp_size > 1:
#             dist.all_reduce(y)
#         return y

#     @torch.no_grad()
#     def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
#         """
#         weight: [out, full_in] -> slice columns
#         bias:   [out] -> same on all ranks
#         """
#         if load_weights.dim() == 2:
#             shard = load_weights[:, self.in_start:self.in_end]
#             param.copy_(shard)
#         elif load_weights.dim() == 1:
#             # bias not sharded in row-parallel
#             param.copy_(load_weights)
#         else:
#             raise ValueError(f"Unexpected load_weights shape: {load_weights.shape}")


# class QKVParallelLinear(ColumnParallelLinear):
#     """
#     Fused QKV projection. To keep your current qwen3.py unchanged,
#     default gather_output=True so caller can split full q,k,v.
#     """
#     def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_kv_heads: int, bias: bool = False):
#         qkv_out = (num_heads + 2 * num_kv_heads) * head_dim
#         super().__init__(input_size=hidden_size, output_size=qkv_out, bias=bias, gather_output=True)


# class MergedColumnParallelLinear(ColumnParallelLinear):
#     """
#     Often used for gate+up merged projection (2 * intermediate).
#     weight_loader supports either:
#       - a single concatenated weight [2*out, in]
#       - a tuple/list of two weights ([out,in], [out,in]) to be concatenated on dim0
#     """
#     def __init__(self, input_size: int, output_size: int, bias: bool = False, merge_parts: int = 2):
#         self.merge_parts = merge_parts
#         super().__init__(input_size=input_size, output_size=output_size, bias=bias, gather_output=True)

#     @torch.no_grad()
#     def weight_loader(self, param: nn.Parameter, load_weights):
#         if isinstance(load_weights, (tuple, list)):
#             assert len(load_weights) == self.merge_parts
#             load_weights = torch.cat(load_weights, dim=0)
#         return super().weight_loader(param, load_weights)
