import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from tllm.layers.attention_torch import Attention
from tllm.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from tllm.layers.layernorm import RMSNorm
from tllm.layers.activation import SiluAndMul
from tllm.layers.rotary_embedding import RotaryEmbedding
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
        self.max_position = config.max_position_embeddings
        self.qkv_bias = getattr(config, "attention_bias", False)
        self.rms_norm_eps = config.rms_norm_eps

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size = self.hidden_size,
            head_dim = self.head_dim,
            num_heads = self.num_heads,
            num_kv_heads = self.num_kv_heads,
            bias = self.qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            input_size = self.head_dim * self.num_heads,
            output_size = self.hidden_size,
            bias = False,
        )
        self.attn = Attention(
            num_heads = self.num_heads,
            head_dim = self.head_dim,
            scaling = self.scaling,
            num_kv_heads = self.num_kv_heads,
        )
        self.rotary_emb = RotaryEmbedding(
            rotary_dim = self.head_dim,
            max_position = self.max_position,
        )
        self.rms_norm = RMSNorm(
            head_dim = self.head_dim,
            eps = self.rms_norm_eps,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            # kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        """
        Args:
            forward(positions: [B,T], hidden_states: [B,T,H], kv_cache: tuple[k_cache, v_cache] | None) -> [B,T,H]
            k_cache: [B, num_kv_heads, max_seq, head_dim],
            v_cache: [B, num_kv_heads, max_seq, head_dim],
        """
        # === QKV Projection ===
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        B, T, _ = hidden_states.shape
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.rms_norm(q)
            k = self.rms_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    def __init__(
            self,
            config: Qwen3Config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up = MergedColumnParallelLinear(
            input_size = self.hidden_size,
            output_size = self.intermediate_size * 2,
            bias=False,
        )
        self.gate_down = RowParallelLinear(
            input_size = self.intermediate_size,
            output_size = self.hidden_size,
            bias = False,
        )
        self.activation = SiluAndMul()

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            forward(x: [B,T,H]) -> [B,T,H]
        """
        return self.gate_down(self.activation(self.gate_up(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(
            self,
            config: Qwen3Config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config)
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
            positions: [batch_size, seq_len] for RoPE
            hidden_states: [batch_size, seq_len, hidden_size]
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
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # Qwen3模型中tie_word_embeddings=False
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, positions)
    
    def compute_logits(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)