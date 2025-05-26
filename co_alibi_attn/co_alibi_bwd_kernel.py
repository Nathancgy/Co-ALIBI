import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]

@triton.jit
def _sigmoid(x):
    return 1 / (1 + tl.exp(-x))

@triton.jit
def _co_alibi_bwd_kernel(
    Q, K, V, sm_scale, causal_mask_value, # Original inputs
    P_raw_in, Sig_P_raw_in, Z_penalty_in, LSE_in, # Saved intermediates from fwd
    DO, 
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
    N_CTX_KV: tl.constexpr,
    NUM_WARPS: tl.constexpr
):
    pid_m = tl.program_id(0) 
    pid_bh = tl.program_id(1) 

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    offs_m_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # --- Load Q_block and DO_block (for the current BLOCK_M queries) --- 
    q_ptrs = Q + pid_b * q_stride_b + pid_h * q_stride_h + \
             offs_m_q[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_block = tl.load(q_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    do_ptrs = DO + pid_b * do_stride_b + pid_h * do_stride_h + \
              offs_m_q[:, None] * do_stride_m + offs_d[None, :] * do_stride_d
    do_block = tl.load(do_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    lse_ptrs = LSE_in + pid_b * lse_in_stride_b + pid_h * lse_in_stride_h + offs_m_q * lse_in_stride_m
    lse_row = tl.load(lse_ptrs, mask=offs_m_q < seq_len_q, other=0.0).to(tl.float32)[:, None] # Reshape to (BLOCK_M, 1) for broadcasting

    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    c_prefix_sum_acc_row = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # --- Loop over key/value blocks (dim N_KV) ---
    for start_n_kv in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv = start_n_kv + tl.arange(0, BLOCK_N_KV)

        k_ptrs = K + pid_b * k_stride_b + pid_h * k_stride_h + \
                 offs_n_kv[None, :] * k_stride_n + offs_d[:, None] * k_stride_k # K is (D, N)
        k_block = tl.load(k_ptrs, mask=(offs_n_kv[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0) # (DMODEL, BLOCK_N_KV)
        
        v_ptrs = V + pid_b * v_stride_b + pid_h * v_stride_h + \
                 offs_n_kv[:, None] * v_stride_n + offs_d[None, :] * v_stride_d # V is (N, D)
        v_block = tl.load(v_ptrs, mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim), other=0.0) # (BLOCK_N_KV, DMODEL)

        p_raw_ptrs = P_raw_in + pid_b * p_raw_in_stride_b + pid_h * p_raw_in_stride_h + \
                     offs_m_q[:, None] * p_raw_in_stride_m + offs_n_kv[None, :] * p_raw_in_stride_n
        p_raw_block = tl.load(p_raw_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        sig_p_raw_ptrs = Sig_P_raw_in + pid_b * sig_p_raw_in_stride_b + pid_h * sig_p_raw_in_stride_h + \
                         offs_m_q[:, None] * sig_p_raw_in_stride_m + offs_n_kv[None, :] * sig_p_raw_in_stride_n
        sig_p_raw_block = tl.load(sig_p_raw_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        z_penalty_ptrs = Z_penalty_in + pid_b * z_penalty_in_stride_b + pid_h * z_penalty_in_stride_h + \
                         offs_m_q[:, None] * z_penalty_in_stride_m + offs_n_kv[None, :] * z_penalty_in_stride_n
        z_penalty_block = tl.load(z_penalty_ptrs, mask=(offs_m_q[:, None] < seq_len_q) & (offs_n_kv[None, :] < seq_len_kv), other=0.0)

        p_adjusted_block = p_raw_block - z_penalty_block

        causal_cond_mask = offs_m_q[:, None] < offs_n_kv[None, :]

        if HAS_CAUSAL_MASK:
            p_adjusted_block = tl.where(causal_cond_mask, causal_mask_value, p_adjusted_block)
        
        _s_block_fp32 = tl.exp(p_adjusted_block - lse_row)
        s_block = _s_block_fp32.to(Q.dtype.element_ty)

        if HAS_CAUSAL_MASK:
            s_block = tl.where(causal_cond_mask, 0.0, s_block)
        s_block = tl.where(offs_m_q[:, None] < seq_len_q, s_block, 0.0)

        ds_block = tl.dot(do_block.to(tl.float32), tl.trans(v_block.to(tl.float32)))

        sum_ds_s = tl.sum(ds_block * s_block.to(tl.float32), axis=1)[:, None]
        dp_adjusted_block = s_block.to(tl.float32) * (ds_block - sum_ds_s)
        
        if HAS_CAUSAL_MASK:
            # Use the already defined causal_cond_mask
            dp_adjusted_block = tl.where(causal_cond_mask, 0.0, dp_adjusted_block)
        dp_adjusted_block = tl.where(offs_m_q[:, None] < seq_len_q, dp_adjusted_block, 0.0)

        sig_p_raw_block_fp32 = sig_p_raw_block.to(tl.float32)
        sigma_prime_block = sig_p_raw_block_fp32 * (1.0 - sig_p_raw_block_fp32)

        c_local_prefix_sum_dp_adj = tl.cumsum(dp_adjusted_block, axis=1)

        c_block = c_prefix_sum_acc_row + c_local_prefix_sum_dp_adj
        
        dp_raw_block = dp_adjusted_block - sigma_prime_block * c_block
        if HAS_CAUSAL_MASK:
            dp_raw_block = tl.where(causal_cond_mask, 0.0, dp_raw_block)
        dp_raw_block = tl.where(offs_m_q[:, None] < seq_len_q, dp_raw_block, 0.0)
        
        c_prefix_sum_acc_row += tl.sum(dp_adjusted_block, axis=1)[:, None]

        dv_contrib = tl.dot(tl.trans(s_block.to(tl.float32)), do_block.to(tl.float32))
        dv_ptrs = DV + pid_b * dv_stride_b + pid_h * dv_stride_h + \
                  offs_n_kv[:, None] * dv_stride_n + offs_d[None, :] * dv_stride_d
        tl.atomic_add(dv_ptrs, dv_contrib.to(DV.dtype.element_ty), mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim))

        dk_contrib = tl.dot(tl.trans(dp_raw_block * sm_scale), q_block.to(tl.float32))
        dk_ptrs = DK + pid_b * dk_stride_b + pid_h * dk_stride_h + \
                  offs_n_kv[:, None] * dk_stride_n + offs_d[None, :] * dk_stride_k
        tl.atomic_add(dk_ptrs, dk_contrib.to(DK.dtype.element_ty), mask=(offs_n_kv[:, None] < seq_len_kv) & (offs_d[None,:] < head_dim))
        
        dq_acc += tl.dot(dp_raw_block * sm_scale, tl.trans(k_block.to(tl.float32)))

    dq_ptrs = DQ + pid_b * dq_stride_b + pid_h * dq_stride_h + \
              offs_m_q[:, None] * dq_stride_m + offs_d[None, :] * dq_stride_k
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), \
             mask=(offs_m_q[:, None] < seq_len_q) & (offs_d[None,:] < head_dim)) 