import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from tllm.layers.linear import QKVParallelLinear, RowParallelLinear
from tllm.layers.layernorm import RMSNorm
from tllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Qwen3Attention(nn.Module):
    def __init__(
            self,
            config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim ** -0.5
        self.qkv_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            self.qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            self.qkv_bias,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim = self.head_dim,
            max_position,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.])

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


# class Qwen3MLP(nn.Module):
#     def __init__(self, config):

#     def forward(self, x, freqs_cis):

class Qwen3DecoderLayer(nn.Module):
    def __init__(
            self,
            config: Qwen3Config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config)
        # self.self_attn = Qwen3Attention(
        #     hidden_size = config.hidden_size,
        #     num_heads = config.num_attention_heads,
        #     num_kv_heads = config.num_key_value_heads,
        #     rms_norm_eps = config.rms_norm_eps,
        #     max_position = config.max_position_embeddings,
        #     qkv_bias = getattr(config, "attention_bias", False),
        #     head_dim = getattr(config, "head_dim", None),
        #     rope_theta = config.rope_theta,
        #     rope_scaling = config.rope_scaling,
        # )
        # SwiGLU
        self.mlp = Qwen3MLP(config)
        # Layer Norm (Pre-Norm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            positions: [batch_size, seq_len] 用于 RoPE
            hidden_states: [batch_size, seq_len, hidden_size]
            residual: 用于兼容 vLLM 风格的残差传递 (可选)
        """
        # if the first layer
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # fuesd add and norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


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