import torch
import torch.nn as nn


def apply_rotary_pos_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float = 10000.0
    ):
        super().__init__()
        self.rotary_dim = rotary_dim
        assert self.rotary_dim % 2 ==0, "rotary_dim must be even"
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # inverse frequency
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        positions = torch.arange(self.max_position_embeddings).float()
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
    
    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions].to(device=query.device, dtype=query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        return query, key