import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_kv(
        kv: torch.Tensor,
        n_rep: int,
) -> torch.Tensor:
    raise NotImplementedError

def _write_kv_cache():
    raise NotImplementedError

def attention_torch_ref():
    raise NotImplementedError

class Attention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            scaling: float,
            num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.scaling = scaling
        assert self.num_heads % self.num_kv_heads == 0, "GQA requires"
        self.n_req = self.num_heads // self.num_kv_heads
        
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            positions: torch.Tensor | None = None,
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.backend == "torch_ref":
            return attention_torch_ref(q, k, v, positions=positions, kv_cache=kv_cache, scaling=self.scaling)
        raise ValueError(f"Unknown attention backend: {self.backend}")