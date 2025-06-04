import torch, math, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def main():
    B, H, N_q, N_k, D = 1, 16, 4096, 4096, 128
    dtype = torch.bfloat16

    torch.manual_seed(0)

    q = torch.randn(B, H, N_q, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N_k, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N_k, D, device='cuda', dtype=dtype)

    scale = 1.0 / math.sqrt(D)

    o_ref = ref_forward(q, k, v, scale_factor=scale, return_intermediates=False, verbose=False)

    print("\nRunning Triton forward...")
    o_tri = co_alibi_attention(q, k, v, causal=True, sm_scale=scale)

    print("\n--- Numerical comparison ---")
    assert_similar(o_tri, o_ref, eps=1e-4, name="output")

if __name__ == "__main__":
    main() 