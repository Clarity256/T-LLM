import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from tllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
        key_ptr,
        vlaue_ptr,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
):
    raise NotImplementedError

def store_kvcache(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
):
    num_tokens, num_kv_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.shape == v_cache.shape
    assert slot_mapping.numel() == num_tokens
    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads = num_kv_heads,
        head_dim = head_dim,
        block_size = block_size,
    )

def flash_attention_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        scaling: float,
        num_heads: int, 
        num_kv_heads: int,
        head_dim: int,
) -> torch.Tensor:
    raise NotImplementedError

def paged_attention_decode(
        q: torch.Tensor,
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: int,
        scaling: float, 
        num_heads: int, 
        num_kv_heads: int, 
        head_dim: int, 
        block_size: int,    
) -> torch.Tensor:
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
        # self.n_req = self.num_heads // self.num_kv_heads
        self.block_size = 16
        self.k_cache = self.v_cache = torch.Tensor([])
        
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        # Triton + KV-cache path (kept for future implementation).
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                k_to_store.contiguous()
                v_to_store.contiguous()
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, self.scaling,
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            o = paged_attention_decode(
                q, k_cache, v_cache,
                context.block_tables,
                context.context_lens,
                self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.block_size
            )
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)

