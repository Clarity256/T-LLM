import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_kv_heads(
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_heads == num_kv_heads:
        return k, v
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
    n_rep = num_heads // num_kv_heads
    # [B, T, nKv, D] -> [B, T, nKv, n_rep, D] -> [B, T, nH, D]
    k = k.unsqueeze(3).expand(*k.shape[:3], n_rep, k.shape[-1]).reshape(
        k.shape[0], k.shape[1], num_heads, k.shape[-1]
    )
    v = v.unsqueeze(3).expand(*v.shape[:3], n_rep, v.shape[-1]).reshape(
        v.shape[0], v.shape[1], num_heads, v.shape[-1]
    )
    return k, v


class Attention(nn.Module):
    """
    Pure PyTorch attention (no KV cache, no flash/paged).
    q: [B, T, nH, D], k/v: [B, T, nKv, D]
    return: [B, T, nH*D]
    """
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

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("q/k/v must be [B, T, H, D]")
        k, v = _expand_kv_heads(k, v, self.num_heads, self.num_kv_heads)

        B, T, H, D = q.shape
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, torch.finfo(attn_scores.dtype).min)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return out