import torch, math, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_alibi_attn import co_alibi_attention

def ref_forward(q, k, v, scale_factor, return_intermediates=False, verbose=False):
    B, H, N_q, _ = q.shape
    N_k = k.shape[2]

    p_raw = torch.einsum('bhid,bhjd->bhij', q, k) * scale_factor

    mask_bool = torch.triu(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool), diagonal=1)
    expanded_mask = mask_bool.unsqueeze(0).unsqueeze(0)

    p_raw_mask_sig = p_raw.masked_fill(expanded_mask, -1e9)
    sig_p_raw = torch.sigmoid(p_raw_mask_sig).masked_fill(expanded_mask, 0.0)

    z_penalty = torch.cumsum(sig_p_raw.flip(dims=[-1]), dim=-1).flip(dims=[-1]).masked_fill(expanded_mask, 0.0)

    mask_val = -torch.finfo(p_raw.dtype).max
    p_adjusted = p_raw.masked_fill(expanded_mask, mask_val) - z_penalty
    p_adjusted = p_adjusted.masked_fill(expanded_mask, mask_val)

    m = torch.max(p_adjusted, dim=-1, keepdim=True).values
    exp_scores = torch.exp(p_adjusted - m).masked_fill(expanded_mask, 0.0)
    lse_denom = exp_scores.sum(dim=-1, keepdim=True) + 1e-9
    s = exp_scores / lse_denom

    o = torch.einsum('bhij,bhjd->bhid', s, v)

    lse_val = m.squeeze(-1) + torch.log(lse_denom.squeeze(-1))

    if verbose:
        print("\n--- PyTorch Reference Intermediates ---")
        for name, tensor in [
            ("p_raw", p_raw),
            ("sigmoid(p_raw)", sig_p_raw),
            ("z_penalty", z_penalty),
            ("p_adjusted", p_adjusted),
            ("softmax s", s),
            ("output o", o),
        ]:
            print(f"{name:>15} {list(tensor.shape)}\n{tensor}\n")

    if return_intermediates:
        return o, p_raw, sig_p_raw, z_penalty, lse_val
    return o

def first_mismatch(a: torch.Tensor, b: torch.Tensor, atol=1e-3, rtol=5e-3):
    diff = torch.abs(a - b)
    mask = diff > (atol + rtol * torch.abs(b))
    if not mask.any():
        return None
    idx = mask.nonzero(as_tuple=False)[0]
    return tuple(idx.tolist()), a[tuple(idx)].item(), b[tuple(idx)].item(), diff[tuple(idx)].item()

def main():
    B, H, N_q, N_k, D = 4, 8, 1024, 1024, 64 
    dtype = torch.float32

    torch.manual_seed(0)

    q = torch.randn(B, H, N_q, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N_k, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N_k, D, device='cuda', dtype=dtype)

    scale = 1.0 / math.sqrt(D)

    o_ref = ref_forward(q, k, v, scale_factor=scale, return_intermediates=False, verbose=False)

    print("\nRunning Triton forward...")
    o_tri = co_alibi_attention(q, k, v, causal=True, sm_scale=scale)

    print("\n--- Numerical comparison ---")
    mm = first_mismatch(o_tri, o_ref)
    if mm is None:
        print("output : OK (allclose)")
    else:
        idx, a, b, d = mm
        print(f"output : mismatch at {idx}  tri={a:.6g}  ref={b:.6g} |diff|={d:.3g}")

    import time

    WARMUP = 10
    ITERS  = 100

    def time_fn(fn, *args):
        # warm-up
        for _ in range(WARMUP):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000 / ITERS

    with torch.no_grad():
        ms_ref  = time_fn(lambda a,b,c: ref_forward(a,b,c, scale_factor=scale), q, k, v)
        ms_tri  = time_fn(lambda a,b,c: co_alibi_attention(a,b,c, causal=True, sm_scale=scale)[0], q, k, v)

    print("\n--- Performance ---")
    print(f"PyTorch reference : {ms_ref:.3f} ms / fwd")
    print(f"Triton kernel     : {ms_tri:.3f} ms / fwd  (speed-up ×{ms_ref / ms_tri:.2f})")

if __name__ == "__main__":
    main() 