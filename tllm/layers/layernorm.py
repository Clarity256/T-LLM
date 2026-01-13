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

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        return output
    
    @torch.compile
    def residual_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        
        return output
        
    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None:
            return residual_rms_forward(x,residual)
        else:
            return rms_forward(x)
