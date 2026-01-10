import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from tllm.layers.layernorm import RMSNorm
from tllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# class Qwen3Attention(nn.Module):
#     def __init__(self, config):

#     def forward(self, x, freqs_cis):

# class Qwen3DecoderLayer(nn.Module):
#     def __init__(self, config):

#     def forward(self, x, freqs_cis):

# class Qwen3MLP(nn.Module):
#     def __init__(self, config):

#     def forward(self, x, freqs_cis):

class Qwen3Model(nn.Module):
    def __init__(
            self,
            config: Qwen3Config
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self,
                 config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_Size)
        # Qwen3模型中tie_word_embeddings=False
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
            self,
            inpud_ids: torch.Tensor,
            positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(inpud_ids, positions)
    
    def comput_logits(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)