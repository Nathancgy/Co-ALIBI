import torch
import torch.nn.functional as F
import math

class CoALIBI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale_factor: float):
        # q: (B, H, N_q, D_h) - Query
        # k: (B, H, N_k, D_h) - Key
        # v: (B, H, N_v, D_h) - Value (N_v typically equals N_k)
        # scale_factor: float, e.g., 1.0 / math.sqrt(D_h)

        B, H, N_q, D_h = q.shape
        N_k = k.shape[2] 
        p_raw = torch.einsum('bhid,bhjd->bhij', q, k) * scale_factor

        mask_value = -torch.finfo(p_raw.dtype).max
        mask_bool = torch.triu(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool), diagonal=1)
        expanded_mask_bool = mask_bool.unsqueeze(0).unsqueeze(0) # (1, 1, N_q, N_k)

        p_raw_masked = p_raw.masked_fill(expanded_mask_bool, mask_value)

        sig_p_raw = torch.sigmoid(p_raw_masked).masked_fill(expanded_mask_bool, 0.0)
        z_penalty = torch.cumsum(sig_p_raw.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        p_adjusted = p_raw_masked - z_penalty
        p_adjusted = p_adjusted.masked_fill(expanded_mask_bool, mask_value)

        m_softmax = torch.max(p_adjusted, dim=-1, keepdim=True).values
        p_adj_minus_max = p_adjusted - m_softmax
        
        s_unnormalized = torch.exp(p_adj_minus_max)
        s_unnormalized = s_unnormalized.masked_fill(expanded_mask_bool, 0.0)

        lse_softmax = torch.sum(s_unnormalized, dim=-1, keepdim=True)
        lse_softmax = lse_softmax + 1e-9 
        
        s = s_unnormalized / lse_softmax
        s = s.masked_fill(expanded_mask_bool, 0.0)
        o = torch.einsum('bhij,bhjd->bhid', s, v)

        ctx.save_for_backward(q, k, v, s, sig_p_raw)
        ctx.scale_factor = scale_factor
        ctx.causal_mask_bool = expanded_mask_bool

        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, s, sig_p_raw = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        causal_mask_bool = ctx.causal_mask_bool

        B, H, N_q, D_h = q.shape
        N_k = k.shape[2]

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        dv = torch.einsum('bhij,bhid->bhjd', s, do)
        ds = torch.einsum('bhid,bhjd->bhij', do, v)

        d_softmax_sum = torch.sum(ds * s, dim=-1, keepdim=True) # (B, H, N_q, 1)
        dp_adjusted = s * (ds - d_softmax_sum)
        dp_adjusted = dp_adjusted.masked_fill(causal_mask_bool, 0.0)

        sigma_prime_p_raw = sig_p_raw * (1.0 - sig_p_raw)
        c_prefix_sum_dp_prime = torch.cumsum(dp_adjusted, dim=-1)

        dp_raw_from_penalty_path = -sigma_prime_p_raw * c_prefix_sum_dp_prime
        dp_raw = dp_adjusted + dp_raw_from_penalty_path
        dp_raw = dp_raw.masked_fill(causal_mask_bool, 0.0)

        dq = torch.einsum('bhij,bhjd->bhid', dp_raw, k) * scale_factor
        dk = torch.einsum('bhij,bhid->bhjd', dp_raw, q) * scale_factor
        
        return dq, dk, dv, None

def print_red_warning(message):
    print(f"\\033[31mWARNING: {message}\\033[0m")

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

if __name__ == '__main__':
    B, H, N, D = 2, 3, 5, 4
    q_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    k_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    v_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    scale = 1.0 / math.sqrt(D)

    q_ref = q_test.detach().clone().requires_grad_(True)
    k_ref = k_test.detach().clone().requires_grad_(True)
    v_ref = v_test.detach().clone().requires_grad_(True)

    def co_alibi_simple_forward(q, k, v, scale_factor):
        p_raw = torch.einsum('bhid,bhjd->bhij', q, k) * scale_factor
        mask_bool = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        expanded_mask_bool = mask_bool.unsqueeze(0).unsqueeze(0)
        
        p_raw_masked_for_sigma = p_raw.masked_fill(expanded_mask_bool, -1e9)
        sig_p_raw = torch.sigmoid(p_raw_masked_for_sigma).masked_fill(expanded_mask_bool, 0.0)
        
        z_penalty = torch.cumsum(sig_p_raw.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        
        mask_value_softmax = -torch.finfo(p_raw.dtype).max
        p_raw_masked_for_softmax = p_raw.masked_fill(expanded_mask_bool, mask_value_softmax)
        p_adjusted = p_raw_masked_for_softmax - z_penalty
        p_adjusted = p_adjusted.masked_fill(expanded_mask_bool, mask_value_softmax)
        
        s_ref = F.softmax(p_adjusted, dim=-1)
        s_ref = s_ref.masked_fill(expanded_mask_bool, 0.0)
        o_ref = torch.einsum('bhij,bhjd->bhid', s_ref, v)
        return o_ref

    print("Running reference CoALIBI (simple_forward)...")
    output_ref = co_alibi_simple_forward(q_ref, k_ref, v_ref, scale)
    do_test_ref = torch.randn_like(output_ref, dtype=torch.float64)
    
    output_ref.backward(do_test_ref)
    dq_ref, dk_ref, dv_ref = q_ref.grad.clone(), k_ref.grad.clone(), v_ref.grad.clone()

    if q_test.grad is not None:
        q_test.grad.zero_()
    if k_test.grad is not None:
        k_test.grad.zero_()
    if v_test.grad is not None:
        v_test.grad.zero_()

    print("\nRunning custom CoALIBI...")
    output_custom = CoALIBI.apply(q_test, k_test, v_test, scale)
    
    output_custom.backward(do_test_ref) 
    dq_custom, dk_custom, dv_custom = q_test.grad.clone(), k_test.grad.clone(), v_test.grad.clone()
    
    print("\n--- Forward Pass Comparison ---")
    assert_similar(output_custom, output_ref, eps=1e-5, name="Forward Output (o)")

    print("\n--- Backward Pass Comparison (Gradients) ---")
    assert_similar(dq_custom, dq_ref, eps=1e-4, name="Gradient dq")
    assert_similar(dk_custom, dk_ref, eps=1e-4, name="Gradient dk")
    assert_similar(dv_custom, dv_ref, eps=1e-4, name="Gradient dv")

    print("\nFinished checks.")