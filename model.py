import torch
import torch.nn.functional as F
import math

class CoALIBI(torch.autograd.Function):
    """
    Raw PyTorch implementation of Co-ALIBI attention with explicit backward pass.
    Assumes Q, K, V are (batch_size, num_heads, seq_len, head_dim).
    Implements causal attention.
    Not optimized for speed or memory (no tiling).
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale_factor: float):
        # Input shapes:
        # q: (B, H, N_q, D_h) - Query
        # k: (B, H, N_k, D_h) - Key
        # v: (B, H, N_v, D_h) - Value (N_v typically equals N_k)
        # scale_factor: float, e.g., 1.0 / math.sqrt(D_h)

        B, H, N_q, D_h = q.shape
        N_k = k.shape[2] # Key sequence length
        # N_v = v.shape[2] # Value sequence length, should be N_k

        # 1. Raw Scores (P_raw)
        # P_raw_tτ = q_t^T k_τ * scale_factor
        # Result shape: (B, H, N_q, N_k)
        p_raw = torch.einsum('bhid,bhjd->bhij', q, k) * scale_factor

        # Create causal mask
        # mask_value is a large negative number for numerical stability in softmax
        mask_value = -torch.finfo(p_raw.dtype).max
        # mask_bool is True for positions to be masked (upper triangle, query i cannot attend to key j if j > i)
        mask_bool = torch.triu(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool), diagonal=1)
        # Expand mask for batch and head dimensions
        expanded_mask_bool = mask_bool.unsqueeze(0).unsqueeze(0) # (1, 1, N_q, N_k)

        # Apply causal mask to raw scores
        p_raw_masked = p_raw.masked_fill(expanded_mask_bool, mask_value)

        # 2. Sigmas of Raw Scores (Sigma_p_raw)
        # σ(p_tτ)
        # For masked positions (large negative p_raw_masked), sigmoid will be ~0.
        # We explicitly mask to 0.0 to ensure no contribution to Z_penalty from masked positions.
        sig_p_raw = torch.sigmoid(p_raw_masked).masked_fill(expanded_mask_bool, 0.0)

        # 3. Penalty Term (Z_penalty)
        # Z_tτ_penalty = sum_{j=τ to t} σ(p_tj)
        # This is a suffix cumulative sum for each row (query t) over keys j.
        # The causal mask on sig_p_raw (making sigmas for j>t zero) ensures the sum is effectively up to t.
        z_penalty = torch.cumsum(sig_p_raw.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        # 4. Adjusted Scores (P_adjusted)
        # p'_tτ = p_tτ - Z_tτ_penalty
        # Use p_raw_masked to ensure masked positions remain large negative.
        p_adjusted = p_raw_masked - z_penalty
        # Re-apply mask to p_adjusted to be absolutely sure.
        p_adjusted = p_adjusted.masked_fill(expanded_mask_bool, mask_value)

        # 5. Attention Weights (S) - Softmax
        # Numerically stable softmax:
        # m_t = max_τ(p'_tτ)
        m_softmax = torch.max(p_adjusted, dim=-1, keepdim=True).values
        # p'_adj_minus_max = p'_tτ - m_t
        p_adj_minus_max = p_adjusted - m_softmax
        
        s_unnormalized = torch.exp(p_adj_minus_max)
        # Mask again after exp, as exp(mask_value) -> 0
        s_unnormalized = s_unnormalized.masked_fill(expanded_mask_bool, 0.0)

        # lse_t = sum_τ exp(p'_tτ - m_t)
        lse_softmax = torch.sum(s_unnormalized, dim=-1, keepdim=True)
        # Add a small epsilon to lse to prevent division by zero if all scores are masked
        lse_softmax = lse_softmax + 1e-9 
        
        s = s_unnormalized / lse_softmax # Final attention weights S_tτ
        s = s.masked_fill(expanded_mask_bool, 0.0) # Final check on masking for s

        # 6. Output (O)
        # O_t = sum_τ S_tτ V_τ
        # (B, H, N_q, N_k) @ (B, H, N_k, D_h) -> (B, H, N_q, D_h)
        o = torch.einsum('bhij,bhjd->bhid', s, v)

        # Save tensors for backward pass
        # p_raw is needed for σ'(p_raw)
        # sig_p_raw is σ(p_raw), useful for σ' = σ(1-σ)
        # s contains the final attention probabilities
        # m_softmax, lse_softmax are for stable softmax backward
        ctx.save_for_backward(q, k, v, s, p_raw, sig_p_raw, m_softmax, lse_softmax)
        ctx.scale_factor = scale_factor
        ctx.causal_mask_bool = expanded_mask_bool # Save the expanded boolean mask

        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        # do: (B, H, N_q, D_h), gradient of loss w.r.t. output o
        q, k, v, s, p_raw, sig_p_raw, m_softmax, lse_softmax = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        causal_mask_bool = ctx.causal_mask_bool

        B, H, N_q, D_h = q.shape
        N_k = k.shape[2]

        # Initialize gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Step 1: Gradient w.r.t. Value (dV)
        # dV_τ = sum_t s_tτ * dO_t
        # s is (B, H, N_q, N_k). Transpose for einsum: (B, H, N_k, N_q)
        # do is (B, H, N_q, D_h)
        # dv should be (B, H, N_k, D_h)
        # Original formula: dV_τ = sum_t s_tτ * dO_t  (sum over query_t, output for key_τ)
        # s[b,h,query,key], do[b,h,query,dim] -> dv[b,h,key,dim]
        # einsum: 'bhij,bhid->bhjd' where i=query, j=key, d=dim
        dv = torch.einsum('bhij,bhid->bhjd', s, do)

        # Step 2: Gradient w.r.t. attention scores S (ds)
        # ds_tτ = dO_t^T V_τ
        # do is (B, H, N_q, D_h)
        # v is (B, H, N_k, D_h).
        # ds should be (B, H, N_q, N_k)
        ds = torch.einsum('bhid,bhjd->bhij', do, v)

        # Step 3: Gradient w.r.t. Adjusted Scores P' (dp_adjusted) - Softmax backward
        # dp'_tτ = s_tτ * (ds_tτ - D_t)
        # where D_t = sum_α ds_tα * s_tα for query t.
        # D_t_sum_ds_s = (ds * s).sum(dim=-1, keepdim=True) # (B, H, N_q, 1)
        # dp_adjusted = s * (ds - D_t_sum_ds_s)
        
        # Alternative for softmax backward using saved m and lse (often used in FlashAttention for stability)
        # s = exp(p_adjusted - m_softmax) / lse_softmax
        # dL/dp_adjusted_ij = s_ij * (dL/ds_ij - sum_k(dL/ds_ik * s_ik))
        # This is equivalent to:
        d_softmax_sum = torch.sum(ds * s, dim=-1, keepdim=True) # (B, H, N_q, 1)
        dp_adjusted = s * (ds - d_softmax_sum)
        
        # Mask out gradients for positions that were causally masked
        dp_adjusted = dp_adjusted.masked_fill(causal_mask_bool, 0.0)

        # Step 4 & 5: Gradient w.r.t. Raw Scores P_raw (dp_raw) via Penalty
        # dp_tμ = dp'_tμ - σ'(p_tμ) * C_tμ
        # where C_tμ = sum_{α=1 to μ} dp'_tα (prefix sum of dp_adjusted for query t)
        # And σ'(p) = σ(p) * (1 - σ(p))

        # sig_p_raw was saved as σ(p_raw)
        # Note: p_raw used for sigma_prime was already masked in forward.
        # So sig_p_raw for masked positions is 0. sigma_prime will also be 0.
        sigma_prime_p_raw = sig_p_raw * (1.0 - sig_p_raw) # σ'(p_raw_tμ)

        # Prefix sum for dp_adjusted: C_tμ
        # For each query row (dim N_q), cumsum over key columns (dim N_k)
        c_prefix_sum_dp_prime = torch.cumsum(dp_adjusted, dim=-1)

        # Gradient contribution from the penalty path
        # dL/dp_raw (via Z) = sum_{alpha<=mu} [ dL/dp'_alpha * (dp'_alpha/dZ_alpha) * (dZ_alpha/dp_raw_mu) ]
        # dL/dp'_alpha * (-1) * sigma'(p_raw_mu)
        dp_raw_from_penalty_path = -sigma_prime_p_raw * c_prefix_sum_dp_prime
        
        # Total dp_raw: gradient from p_adjusted directly + gradient from penalty path
        # dL/dp_raw_mu (direct) = dL/dp'_mu * (dp'_mu / dp_raw_mu) = dL/dp'_mu * 1
        dp_raw = dp_adjusted + dp_raw_from_penalty_path
        
        # Mask out gradients again for causally masked positions
        dp_raw = dp_raw.masked_fill(causal_mask_bool, 0.0)

        # Step 6: Gradients w.r.t. Q and K
        # dp_raw is dL/dp_tτ (shape: B, H, N_q, N_k)
        # dQ_t = scale_factor * sum_τ dp_tτ * K_τ
        # dq should be (B, H, N_q, D_h)
        # k is (B, H, N_k, D_h)
        dq = torch.einsum('bhij,bhjd->bhid', dp_raw, k) * scale_factor

        # dK_τ = scale_factor * sum_t dp_tτ * Q_t
        # dk should be (B, H, N_k, D_h)
        # q is (B, H, N_q, D_h)
        # Formula: dK_τ = scale_factor * sum_t dp_tτ * Q_t
        # dp_raw[b,h,t,τ] (t=Nq, τ=Nk), q[b,h,t,d] (t=Nq, d=Dh) -> dk[b,h,τ,d]
        # einsum: 'bhij,bhid->bhjd' (i=Nq for dp_raw&q, j=Nk for dp_raw, d=Dh for q)
        dk = torch.einsum('bhij,bhid->bhjd', dp_raw, q) * scale_factor
        
        return dq, dk, dv, None # For scale_factor if it doesn't require grad

# Example Usage:
if __name__ == '__main__':
    B, H, N, D = 2, 3, 5, 4 # Batch, Heads, SeqLen, HeadDim
    q_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    k_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    v_test = torch.randn(B, H, N, D, requires_grad=True, dtype=torch.float64)
    scale = 1.0 / math.sqrt(D)

    # Test with PyTorch's autograd
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
        
        s = F.softmax(p_adjusted, dim=-1)
        s = s.masked_fill(expanded_mask_bool, 0.0)
        o = torch.einsum('bhij,bhjd->bhid', s, v)
        return o

    print("Running custom CoALIBI...")
    output_custom = CoALIBI.apply(q_test, k_test, v_test, scale)
    do_test_custom = torch.randn_like(output_custom, dtype=torch.float64)
    output_custom.backward(do_test_custom)
    dq_custom, dk_custom, dv_custom = q_test.grad.clone(), k_test.grad.clone(), v_test.grad.clone()
    
    print("dq_custom norm:", torch.linalg.norm(dq_custom).item())
    print("dk_custom norm:", torch.linalg.norm(dk_custom).item())
    print("dv_custom norm:", torch.linalg.norm(dv_custom).item())
    
    q_test_gc = q_test.detach().clone().requires_grad_(True)
    k_test_gc = k_test.detach().clone().requires_grad_(True)
    v_test_gc = v_test.detach().clone().requires_grad_(True)
    
    print("\nRunning gradcheck...")
    inputs_for_gradcheck = (q_test_gc, k_test_gc, v_test_gc, scale)
    test_passed = torch.autograd.gradcheck(CoALIBI.apply, inputs_for_gradcheck, eps=1e-6, atol=1e-4)
    print(f"Gradcheck passed: {test_passed}")