import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).to(x.dtype)
    
    # @torch.compile
    def residual_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused residual add + RMSNorm; return (normed, new_residual)
        y = x + residual
        variance = y.float().pow(2).mean(dim=-1, keepdim=True)
        y_norm = y * torch.rsqrt(variance + self.eps)
        y_norm = (y_norm * self.weight).to(y.dtype)
        return y_norm, y
        
    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)
