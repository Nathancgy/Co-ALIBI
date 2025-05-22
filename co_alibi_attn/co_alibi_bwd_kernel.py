import triton
import triton.language as tl

@triton.jit
def _sigmoid(x):
    # Helper for sigma_prime calculation if needed, or use P_raw_in and Sig_P_raw_in directly
    return 1 / (1 + tl.exp(-x))

@triton.jit
def _co_alibi_bwd_kernel(
    # Inputs from forward pass & autograd context
    Q, K, V, sm_scale, causal_mask_value, # Original inputs
    P_raw_in, Sig_P_raw_in, Z_penalty_in, LSE_in, # Saved intermediates from fwd
    DO, # Gradient of loss w.r.t. output O
    # Outputs (gradients to compute)
    DQ, DK, DV,
    # Strides for all tensors
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    p_raw_in_stride_b, p_raw_in_stride_h, p_raw_in_stride_m, p_raw_in_stride_n,
    sig_p_raw_in_stride_b, sig_p_raw_in_stride_h, sig_p_raw_in_stride_m, sig_p_raw_in_stride_n,
    z_penalty_in_stride_b, z_penalty_in_stride_h, z_penalty_in_stride_m, z_penalty_in_stride_n,
    lse_in_stride_b, lse_in_stride_h, lse_in_stride_m,
    do_stride_b, do_stride_h, do_stride_m, do_stride_d,
    dq_stride_b, dq_stride_h, dq_stride_m, dq_stride_k,
    dk_stride_b, dk_stride_h, dk_stride_n, dk_stride_k,
    dv_stride_b, dv_stride_h, dv_stride_n, dv_stride_d,
    # Meta-parameters
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N_KV: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    N_CTX_KV: tl.constexpr, # Max context length for SRAM buffers (power of 2 >= seq_len_kv)
    NUM_WARPS: tl.constexpr
):
    pid_m = tl.program_id(0)  # Operates on a block of BLOCK_M queries
    pid_bh = tl.program_id(1) # Batch and head index

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offsets for the current query block
    offs_m_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # Query indices [0, BLOCK_M-1] relative to pid_m start
    offs_d = tl.arange(0, BLOCK_DMODEL) # Head dimension indices

    # --- Load Q_block and DO_block (for the current BLOCK_M queries) --- 
    q_ptrs = Q + pid_b * q_stride_b + pid_h * q_stride_h + \
             offs_m_q[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_block = tl.load(q_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    do_ptrs = DO + pid_b * do_stride_b + pid_h * do_stride_h + \
              offs_m_q[:, None] * do_stride_m + offs_d[None, :] * do_stride_d
    do_block = tl.load(do_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    # --- Load LSE for the current query block --- (B, H, M)
    lse_ptrs = LSE_in + pid_b * lse_in_stride_b + pid_h * lse_in_stride_h + offs_m_q * lse_in_stride_m
    lse_row = tl.load(lse_ptrs, mask=offs_m_q < seq_len_q, other=0.0).to(tl.float32)[:, None] # Reshape to (BLOCK_M, 1) for broadcasting

    # --- Initialize dQ accumulator in SRAM --- (BLOCK_M, BLOCK_DMODEL)
    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    
    # --- SRAM buffer for dp_adjusted_row prefix sum across key blocks ---
    # C_tμ = sum_{α=0 to μ} dp'_tα. This needs to accumulate across iterations of key blocks.
    # For each query row in BLOCK_M, we need a running sum of dp_adjusted elements.
    c_prefix_sum_acc_row = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # --- Loop over key/value blocks (dim N_KV) ---
    for start_n_kv in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv = start_n_kv + tl.arange(0, BLOCK_N_KV) # Key indices for this block

        # --- Load K_block, V_block --- 
        k_ptrs = K + pid_b * k_stride_b + pid_h * k_stride_h + \
                 offs_n_kv[None, :] * k_stride_n + offs_d[:, None] * k_stride_k # K is (D, N)
        k_block = tl.load(k_ptrs, mask=(offs_n_kv[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0) # (DMODEL, BLOCK_N_KV)
        
        v_ptrs = V + pid_b * v_stride_b + pid_h * v_stride_h + \
                 offs_n_kv[:, None] * v_stride_n + offs_d[None, :] * v_stride_d # V is (N, D)
        v_block = tl.load(v_ptrs, mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim), other=0.0) # (BLOCK_N_KV, DMODEL)

        # --- Load P_raw, Sig_P_raw, Z_penalty for the current Q_block and K_block --- 
        # Shapes: (BLOCK_M, BLOCK_N_KV)
        p_raw_ptrs = P_raw_in + pid_b * p_raw_in_stride_b + pid_h * p_raw_in_stride_h + \
                     offs_m_q[:, None] * p_raw_in_stride_m + offs_n_kv[None, :] * p_raw_in_stride_n
        p_raw_block = tl.load(p_raw_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        sig_p_raw_ptrs = Sig_P_raw_in + pid_b * sig_p_raw_in_stride_b + pid_h * sig_p_raw_in_stride_h + \
                         offs_m_q[:, None] * sig_p_raw_in_stride_m + offs_n_kv[None, :] * sig_p_raw_in_stride_n
        sig_p_raw_block = tl.load(sig_p_raw_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        z_penalty_ptrs = Z_penalty_in + pid_b * z_penalty_in_stride_b + pid_h * z_penalty_in_stride_h + \
                         offs_m_q[:, None] * z_penalty_in_stride_m + offs_n_kv[None, :] * z_penalty_in_stride_n
        z_penalty_block = tl.load(z_penalty_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        # --- Step 1 (from math): Recompute S_block (Attention Probabilities) ---
        # p_adjusted = p_raw - z_penalty
        p_adjusted_block = p_raw_block - z_penalty_block
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m_q[:, None] < offs_n_kv[None, :]
            p_adjusted_block = tl.where(causal_cond, causal_mask_value, p_adjusted_block)
        
        # s = exp(p_adjusted - lse)
        s_block = tl.exp(p_adjusted_block - lse_row) # lse_row is (BLOCK_M, 1)
        if HAS_CAUSAL_MASK:
            s_block = tl.where(causal_cond, 0.0, s_block)
        # Mask S for queries out of bounds (e.g. padding queries)
        s_block = tl.where(offs_m_q[:, None] < seq_len_q, s_block, 0.0)

        # --- Step 2 (from math): Calculate dS_block = DO @ V.T --- (BLOCK_M, BLOCK_N_KV)
        ds_block = tl.dot(do_block, tl.trans(v_block))

        # --- Step 3 (from math): Calculate dP_adjusted_block (Softmax Backward) ---
        # dP'_tτ = S_tτ * (dS_tτ - sum_α dS_tα * S_tα)
        # sum_ds_s needs to be row-wise sum over N_KV dim: (BLOCK_M, 1)
        sum_ds_s = tl.sum(ds_block * s_block, axis=1)[:, None] # Keep dim for broadcasting
        dp_adjusted_block = s_block * (ds_block - sum_ds_s)
        if HAS_CAUSAL_MASK:
            dp_adjusted_block = tl.where(causal_cond, 0.0, dp_adjusted_block)
        dp_adjusted_block = tl.where(offs_m_q[:, None] < seq_len_q, dp_adjusted_block, 0.0)

        # --- Step 4 & 5 (from math): Calculate dP_raw_block via Penalty --- 
        # dP_raw_tμ = dP'_tμ - σ'(P_raw_tμ) * C_tμ
        # σ'(P_raw) = σ(P_raw) * (1 - σ(P_raw))
        sigma_prime_block = sig_p_raw_block * (1.0 - sig_p_raw_block)

        # Calculate C_tμ = prefix_sum(dP'_tα) for the current block
        # This is a local prefix sum within the current dp_adjusted_block for each row.
        # And it needs to be added to the sum from previous blocks (c_prefix_sum_acc_row).
        c_local_prefix_sum_dp_adj = tl.cumsum(dp_adjusted_block, axis=1)
        c_block = c_prefix_sum_acc_row + c_local_prefix_sum_dp_adj
        
        dp_raw_block = dp_adjusted_block - sigma_prime_block * c_block
        if HAS_CAUSAL_MASK:
            dp_raw_block = tl.where(causal_cond, 0.0, dp_raw_block)
        dp_raw_block = tl.where(offs_m_q[:, None] < seq_len_q, dp_raw_block, 0.0)
        
        # Update the running prefix sum for the next iteration of key blocks
        # Add the sum of the current full dp_adjusted_block row to the accumulator for that row
        c_prefix_sum_acc_row += tl.sum(dp_adjusted_block, axis=1)[:, None]

        # --- Step 1 (from math, re-stated): Calculate dV contribution --- (BLOCK_N_KV, DMODEL)
        # dV_contrib = S_block.T @ DO_block
        # S_block is (BLOCK_M, BLOCK_N_KV), DO_block is (BLOCK_M, DMODEL)
        dv_contrib = tl.dot(tl.trans(s_block.to(DV.dtype.element_ty)), do_block.to(DV.dtype.element_ty))
        dv_ptrs = DV + pid_b * dv_stride_b + pid_h * dv_stride_h + \
                  offs_n_kv[:, None] * dv_stride_n + offs_d[None, :] * dv_stride_d
        tl.atomic_add(dv_ptrs, dv_contrib, mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim))

        # --- Step 6 (from math): Calculate dK contribution ---
        # dK_contrib = (dP_raw_block * sm_scale).T @ Q_block
        # dP_raw_block is (BLOCK_M, BLOCK_N_KV), Q_block is (BLOCK_M, DMODEL)
        dk_contrib = tl.dot(tl.trans((dp_raw_block * sm_scale).to(DK.dtype.element_ty)), q_block.to(DK.dtype.element_ty))
        dk_ptrs = DK + pid_b * dk_stride_b + pid_h * dk_stride_h + \
                  offs_n_kv[:, None] * dk_stride_n + offs_d[None, :] * dk_stride_d
        tl.atomic_add(dk_ptrs, dk_contrib, mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim))
        
        # --- Step 6 (from math): Accumulate dQ contribution ---
        # dQ_acc += (dP_raw_block * sm_scale) @ K_block.T
        # dP_raw_block is (BLOCK_M, BLOCK_N_KV), K_block is (DMODEL, BLOCK_N_KV)
        dq_acc += tl.dot((dp_raw_block * sm_scale).to(Q.dtype.element_ty), tl.trans(k_block.to(Q.dtype.element_ty)))

    # --- Store final dQ --- (BLOCK_M, DMODEL)
    dq_ptrs = DQ + pid_b * dq_stride_b + pid_h * dq_stride_h + \
              offs_m_q[:, None] * dq_stride_m + offs_d[None, :] * dq_stride_k
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), \
             mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None,:] < head_dim)) 