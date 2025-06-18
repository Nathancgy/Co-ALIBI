import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from co_alibi_attn import co_alibi_attention

def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")

def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f'{name} all zero')
        return 1.0
    sim = 2 * (x * y).sum() / denominator
    return sim.item()

def assert_similar(x, y, eps=1e-8, name="tensor"):
    sim = calc_sim(x, y, name)
    diff = 1. - sim
    if not (0 <= diff <= eps):
        print_red_warning(f'{name} Error: {diff}')
    else:
        print(f'passed: {name} diff={diff:.2e}')

def ref_forward(q, k, v, scale_factor):
    B, H, M, _ = q.shape
    N = k.shape[2]

    p_raw = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale_factor
    causal_mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), 1)
    dtype_min_neg = -torch.finfo(p_raw.dtype).max
    p_raw_mask_sig = p_raw.masked_fill(causal_mask, dtype_min_neg)
    sig = torch.sigmoid(p_raw_mask_sig).masked_fill(causal_mask, 0.0)

    z_penalty = torch.cumsum(sig.flip(-1), -1).flip(-1)
    p_adj = p_raw.masked_fill(causal_mask, dtype_min_neg) - z_penalty
    p_adj = p_adj.masked_fill(causal_mask, dtype_min_neg)

    s = torch.softmax(p_adj, dim=-1).masked_fill(causal_mask, 0.0)
    out = torch.einsum("bhmn,bhnd->bhmd", s, v)
    return out, (p_raw.detach(), sig.detach(), z_penalty.detach(), s.detach())

def main():
    torch.manual_seed(0)

    B, H, S, D = 1, 16, 4096, 128
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

    scale = 1.0 / math.sqrt(D)

    out_ref, ref_intermediates = ref_forward(q, k, v, scale)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.detach(), k.grad.detach(), v.grad.detach()

    q.grad = k.grad = v.grad = None

    out_tri = co_alibi_attention(q, k, v, causal=True, sm_scale=scale)
    out_tri.backward(dout)
    dq_tri, dk_tri, dv_tri = q.grad.detach(), k.grad.detach(), v.grad.detach()

    if os.getenv("COALIBI_DEBUG", "0") == "1":
        from co_alibi_attn import debug_dp_raw, debug_s

        p_raw_ref, sig_ref, z_penalty_ref, s_ref = ref_intermediates
        sigma_prime_ref = sig_ref * (1.0 - sig_ref)

        ds_ref = torch.einsum("bhid,bhjd->bhij", dout, v)
        sum_ds_s_ref = (ds_ref * s_ref).sum(dim=-1, keepdim=True)
        dp_adj_ref = s_ref * (ds_ref - sum_ds_s_ref)
        c_prefix_ref = torch.cumsum(dp_adj_ref, dim=-1)
        dp_raw_ref = dp_adj_ref - sigma_prime_ref * c_prefix_ref

        print("\n--- Debug backward intermediates (max abs diff) ---")
        if debug_dp_raw is not None:
            print("dp_raw diff:", (debug_dp_raw - dp_raw_ref).abs().max().item())
        if debug_s is not None:
            print("softmax s diff:", (debug_s - s_ref).abs().max().item())

    print("Forward output diff       :", (out_tri - out_ref).abs().max().item())
    print("dq max diff (abs)         :", (dq_tri - dq_ref).abs().max().item())
    print("dk max diff (abs)         :", (dk_tri - dk_ref).abs().max().item())
    print("dv max diff (abs)         :", (dv_tri - dv_ref).abs().max().item())

    # Print the first ten flattened elements of both reference and Triton gradients for q and k to inspect if they are zero
    first_ten_dq_ref = dq_ref.flatten()[:10].tolist()
    first_ten_dq_tri = dq_tri.flatten()[:10].tolist()
    first_ten_dk_ref = dk_ref.flatten()[:10].tolist()
    first_ten_dk_tri = dk_tri.flatten()[:10].tolist()

    print("First 10 elements dq_ref:", first_ten_dq_ref)
    print("First 10 elements dq_tri:", first_ten_dq_tri)
    print("First 10 elements dk_ref:", first_ten_dk_ref)
    print("First 10 elements dk_tri:", first_ten_dk_tri)

    assert_similar(dq_tri, dq_ref, eps=1e-4, name="dq") 
    assert_similar(dk_tri, dk_ref, eps=1e-4, name="dk")
    assert_similar(dv_tri, dv_ref, eps=1e-4, name="dv")

if __name__ == "__main__":
    main() 