import torch
from tllm.layers.attention import Attention

def test_prefill_cache_equivalence():
    torch.manual_seed(0)
    B, T, nH, nKv, D = 2, 8, 8, 2, 16
    attn = Attention(num_heads=nH, num_kv_heads=nKv, head_dim=D)

    q = torch.randn(B, T, nH, D)
    k = torch.randn(B, T, nKv, D)
    v = torch.randn(B, T, nKv, D)
    pos = torch.arange(T).view(1, T).expand(B, T)

    y0 = attn(q, k, v, positions=pos, kv_cache=None)

    max_seq = 64
    k_cache = torch.zeros(B, nKv, max_seq, D)
    v_cache = torch.zeros(B, nKv, max_seq, D)
    y1 = attn(q, k, v, positions=pos, kv_cache=(k_cache, v_cache))

    assert torch.allclose(y0, y1, atol=1e-5), (y0 - y1).abs().max().item()
    print("prefill cache equivalence OK")

def test_decode_step_shape():
    torch.manual_seed(0)
    B, T0, nH, nKv, D = 2, 5, 8, 2, 16
    attn = Attention(num_heads=nH, num_kv_heads=nKv, head_dim=D)

    max_seq = 64
    k_cache = torch.zeros(B, nKv, max_seq, D)
    v_cache = torch.zeros(B, nKv, max_seq, D)

    # prefill
    q = torch.randn(B, T0, nH, D)
    k = torch.randn(B, T0, nKv, D)
    v = torch.randn(B, T0, nKv, D)
    pos = torch.arange(T0).view(1, T0).expand(B, T0)
    _ = attn(q, k, v, positions=pos, kv_cache=(k_cache, v_cache))

    # decode at position T0
    q1 = torch.randn(B, 1, nH, D)
    k1 = torch.randn(B, 1, nKv, D)
    v1 = torch.randn(B, 1, nKv, D)
    pos1 = torch.full((B, 1), T0, dtype=torch.long)
    y = attn(q1, k1, v1, positions=pos1, kv_cache=(k_cache, v_cache))

    assert y.shape == (B, 1, nH * D)
    print("decode step shape OK")


test_prefill_cache_equivalence()
test_decode_step_shape()

