import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from co_alibi_attn import co_alibi_attention

# ---------------- Reference implementation ------------------------------------

def ref_forward(q, k, v, scale_factor):
    """Pure-PyTorch forward pass (no Triton) following the paper exactly."""
    B, H, M, _ = q.shape
    N = k.shape[2]

    p_raw = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale_factor
    causal_mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), 1)
    mask_value = -torch.finfo(p_raw.dtype).max

    p_raw_mask_sig = p_raw.masked_fill(causal_mask, -1e9)
    sig = torch.sigmoid(p_raw_mask_sig).masked_fill(causal_mask, 0.0)

    z_penalty = torch.cumsum(sig.flip(-1), -1).flip(-1)
    p_adj = p_raw.masked_fill(causal_mask, mask_value) - z_penalty
    p_adj = p_adj.masked_fill(causal_mask, mask_value)

    s = torch.softmax(p_adj, dim=-1).masked_fill(causal_mask, 0.0)
    out = torch.einsum("bhmn,bhnd->bhmd", s, v)
    return out, (p_raw.detach(), sig.detach(), z_penalty.detach(), s.detach())

# ------------------------------------------------------------------------------

def first_mismatch(a: torch.Tensor, b: torch.Tensor, atol=1e-3, rtol=5e-3):
    diff = torch.abs(a - b)
    mask = diff > (atol + rtol * torch.abs(b))
    if not mask.any():
        return None
    idx = mask.nonzero(as_tuple=False)[0]
    return tuple(idx.tolist()), a[tuple(idx)].item(), b[tuple(idx)].item(), diff[tuple(idx)].item()


def main():
    torch.manual_seed(0)

    # Config ------------------------------------------------------------------
    B, H, S, D = 4, 8, 256, 64  # keep manageable for grad-check
    dtype = torch.float32  # use fp32 for strict checks
    device = "cuda"

    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

    scale = 1.0 / math.sqrt(D)

    # ---------------- Reference fwd+bwd -------------------------------------
    out_ref, ref_intermediates = ref_forward(q, k, v, scale)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.detach(), k.grad.detach(), v.grad.detach()

    # Reset grads -------------------------------------------------------------
    q.grad = k.grad = v.grad = None

    # ---------------- Triton fwd+bwd ----------------------------------------
    out_tri = co_alibi_attention(q, k, v, causal=True, sm_scale=scale)
    out_tri.backward(dout)
    dq_tri, dk_tri, dv_tri = q.grad.detach(), k.grad.detach(), v.grad.detach()

    # ---------------- Optional debug comparisons ---------------------------
    if os.getenv("COALIBI_DEBUG", "0") == "1":
        from co_alibi_attn import debug_dp_raw, debug_s

        # Recompute reference intermediates needed for gradients -------------
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

    # ---------------- Comparison --------------------------------------------
    print("Forward output diff       :", (out_tri - out_ref).abs().max().item())
    print("dq max diff (abs)         :", (dq_tri - dq_ref).abs().max().item())
    print("dk max diff (abs)         :", (dk_tri - dk_ref).abs().max().item())
    print("dv max diff (abs)         :", (dv_tri - dv_ref).abs().max().item())

    # report first mismatches if any
    for name, a, b in [
        ("dq", dq_tri, dq_ref),
        ("dk", dk_tri, dk_ref),
        ("dv", dv_tri, dv_ref),
    ]:
        mm = first_mismatch(a, b, atol=1e-2, rtol=1e-2)
        if mm is None:
            print(f"{name:>4} : OK (allclose)")
        else:
            idx, va, vb, vd = mm
            print(f"{name:>4} : mismatch at {idx}  tri={va:.6g}  ref={vb:.6g} |diff|={vd:.3g}")

    # Intermediate sanity print (optional) -----------------------------------
    print("\n--- Reference intermediate stats (mean, max) ---")
    tags = ["p_raw", "sigmoid", "z_penalty", "softmax"]
    for tag, tensor in zip(tags, ref_intermediates):
        print(f"{tag:<10}: mean={tensor.mean().item():.4e}  max={tensor.abs().max().item():.4e}")


if __name__ == "__main__":
    main() 