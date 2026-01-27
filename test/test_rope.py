import torch
from tllm.layers.rotary_embedding import RotaryEmbedding

def test_shapes():
    B, T, H, D = 2, 5, 4, 8
    rope = RotaryEmbedding(rotary_dim=D, max_position_embeddings=128)
    positions = torch.arange(T).unsqueeze(0).repeat(B, 1)  # [B,T]
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    q2, k2 = rope(positions, q, k)
    assert q2.shape == q.shape and k2.shape == k.shape
    print("test_shapes OK")

def test_pos0_identity():
    # 位置 0 时 cos=1 sin=0，旋转应不改变向量（至少对 rotary 部分）
    B, T, H, D = 1, 1, 2, 8
    rope = RotaryEmbedding(rotary_dim=D, max_position_embeddings=128)
    positions = torch.zeros(B, T, dtype=torch.long)
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    q2, k2 = rope(positions, q, k)
    assert torch.allclose(q2, q, atol=1e-6), (q2 - q).abs().max().item()
    assert torch.allclose(k2, k, atol=1e-6), (k2 - k).abs().max().item()
    print("test_pos0_identity OK")

def test_norm_preserved():
    # 旋转是正交变换：rotary 部分的 L2 范数应保持
    B, T, H, D = 2, 7, 3, 8
    rope = RotaryEmbedding(rotary_dim=D, max_position_embeddings=128)
    positions = torch.randint(0, 50, (B, T))
    q = torch.randn(B, T, H, D)
    q2, _ = rope(positions, q, q)

    n1 = torch.linalg.norm(q, dim=-1)
    n2 = torch.linalg.norm(q2, dim=-1)
    assert torch.allclose(n1, n2, atol=1e-5), (n1 - n2).abs().max().item()
    print("test_norm_preserved OK")

def test_gpu_half_optional():
    if not torch.cuda.is_available():
        print("CUDA not available, skip test_gpu_half_optional")
        return
    B, T, H, D = 2, 5, 4, 8
    rope = RotaryEmbedding(rotary_dim=D, max_position_embeddings=128).cuda()
    positions = torch.arange(T, device="cuda").unsqueeze(0).repeat(B, 1)
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)
    q2, k2 = rope(positions, q, k)
    assert q2.dtype == q.dtype and q2.device == q.device
    print("test_gpu_half_optional OK")

if __name__ == "__main__":
    test_shapes()
    test_pos0_identity()
    test_norm_preserved()
    test_gpu_half_optional()
    print("ALL OK")
