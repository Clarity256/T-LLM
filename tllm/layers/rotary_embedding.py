import torch
import torch.nn as nn

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.reshape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : ]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            head_dim: int,
            rotary_dim: int | None=None,
            max_position_embeddings: int = 2048,
            base: float = 10000.0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        assert self.rotary_dim % 2 ==0, "rotary_dim must be even"
        assert self.rotary_dim <= self.head_dim, "rotary_dim must be <= head_dim"

        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))

        