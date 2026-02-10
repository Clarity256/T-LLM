import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.weight.weight_loader = self.weight_loader
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
        param.copy_(load_weights.to(param.device, dtype=param.dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class ParallelLMHead(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(vocab_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, load_weights: torch.Tensor):
        param.copy_(load_weights.to(param.device, dtype=param.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)
