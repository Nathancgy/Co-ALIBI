import triton
import triton.language as tl

@triton.jit
def _sigmoid(x):
    return 1 / (1 + tl.exp(-x))

@triton.jit
def _co_alibi_fwd_kernel(
    Q, K, V, sm_scale, causal_mask_value, # Inputs
    P_raw_out, Sig_P_raw_out, Z_penalty_out, LSE_out, # Buffers for saving intermediate values
    Out, # Output
    # Strides
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    out_stride_b, out_stride_h, out_stride_m, out_stride_d,
    p_raw_out_stride_b, p_raw_out_stride_h, p_raw_out_stride_m, p_raw_out_stride_n,
    sig_p_raw_out_stride_b, sig_p_raw_out_stride_h, sig_p_raw_out_stride_m, sig_p_raw_out_stride_n,
    z_penalty_out_stride_b, z_penalty_out_stride_h, z_penalty_out_stride_m, z_penalty_out_stride_n,
    lse_out_stride_b, lse_out_stride_h, lse_out_stride_m,
    # Meta-parameters
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N_KV: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    N_CTX_KV: tl.constexpr # Must be power of 2 and >= seq_len_kv for easy buffer handling
):
    pid_m = tl.program_id(0) # Selects the query block
    pid_bh = tl.program_id(1) # Selects batch and head

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Pointers to Q for the current block of queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + pid_b * q_stride_b + pid_h * q_stride_h + \
             offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k

    p_raw_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)
    sig_p_raw_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)

    # --- Pass 1: Compute P_raw and Sig(P_raw) ---
    for start_n_kv_block in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        
        k_ptrs = K + pid_b * k_stride_b + pid_h * k_stride_h + \
                 offs_n_kv_block[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        
        k_block = tl.load(k_ptrs, mask=(offs_n_kv_block[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        q_block = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)
        p_raw_block = tl.dot(q_block, k_block) * sm_scale

        p_raw_out_ptr = P_raw_out + pid_b * p_raw_out_stride_b + pid_h * p_raw_out_stride_h + \
                        offs_m[:, None] * p_raw_out_stride_m + offs_n_kv_block[None, :] * p_raw_out_stride_n
        tl.store(p_raw_out_ptr, p_raw_block, mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv))
        
        tl.store(p_raw_sram + offs_m[:, None] * N_CTX_KV + offs_n_kv_block[None, :], p_raw_block,
                 mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv) & (offs_n_kv_block[None, :] < N_CTX_KV))

        p_raw_for_sig = p_raw_block
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m[:, None] < offs_n_kv_block[None, :]
            p_raw_for_sig = tl.where(causal_cond, -float('inf'), p_raw_for_sig)
            
        sig_p_raw_block = _sigmoid(p_raw_for_sig)
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m[:, None] < offs_n_kv_block[None, :]
            sig_p_raw_block = tl.where(causal_cond, 0.0, sig_p_raw_block)
        
        sig_p_raw_out_ptr = Sig_P_raw_out + pid_b * sig_p_raw_out_stride_b + pid_h * sig_p_raw_out_stride_h + \
                            offs_m[:, None] * sig_p_raw_out_stride_m + offs_n_kv_block[None, :] * sig_p_raw_out_stride_n
        tl.store(sig_p_raw_out_ptr, sig_p_raw_block, mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv))

        tl.store(sig_p_raw_sram + offs_m[:, None] * N_CTX_KV + offs_n_kv_block[None, :], sig_p_raw_block,
                 mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv) & (offs_n_kv_block[None, :] < N_CTX_KV))

    # --- Pass 2: Compute Z_penalty from sig_p_raw_sram ---
    z_penalty_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)
    current_sum = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for start_n_kv_suffix in range(N_CTX_KV - 1, -1, -1):
        col_idx = start_n_kv_suffix + tl.arange(0, 1)
        sig_val_col = tl.load(sig_p_raw_sram + offs_m[:, None] * N_CTX_KV + col_idx[None, :],
                              mask=(offs_m[:, None] < seq_len_q) & (col_idx[None, :] < seq_len_kv) & (col_idx[None,:] < N_CTX_KV), other=0.0)
        current_sum += sig_val_col
        tl.store(z_penalty_sram + offs_m[:, None] * N_CTX_KV + col_idx[None, :], current_sum,
                 mask=(offs_m[:, None] < seq_len_q) & (col_idx[None, :] < seq_len_kv) & (col_idx[None,:] < N_CTX_KV))

    for start_n_kv_block in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        z_penalty_block_to_store = tl.load(z_penalty_sram + offs_m[:, None] * N_CTX_KV + offs_n_kv_block[None, :],
                                           mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv) & (offs_n_kv_block[None,:] < N_CTX_KV), other=0.0)
        z_penalty_out_ptr = Z_penalty_out + pid_b * z_penalty_out_stride_b + pid_h * z_penalty_out_stride_h + \
                            offs_m[:, None] * z_penalty_out_stride_m + offs_n_kv_block[None, :] * z_penalty_out_stride_n
        tl.store(z_penalty_out_ptr, z_penalty_block_to_store, mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv))

    # --- Pass 3: Compute P_adjusted, Softmax, and Output O ---
    m_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n_kv_block in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        p_raw_block = tl.load(p_raw_sram + offs_m[:, None] * N_CTX_KV + offs_n_kv_block[None, :],
                              mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv) & (offs_n_kv_block[None,:] < N_CTX_KV), other=0.0)
        z_penalty_block = tl.load(z_penalty_sram + offs_m[:, None] * N_CTX_KV + offs_n_kv_block[None, :],
                                   mask=(offs_m[:, None] < seq_len_q) & (offs_n_kv_block[None, :] < seq_len_kv) & (offs_n_kv_block[None,:] < N_CTX_KV), other=0.0)

        p_adjusted_block = p_raw_block - z_penalty_block
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m[:, None] < offs_n_kv_block[None, :]
            p_adjusted_block = tl.where(causal_cond, causal_mask_value, p_adjusted_block)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(p_adjusted_block, axis=1))
        p_scores_block = tl.exp(p_adjusted_block - m_i)
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(p_scores_block, axis=1)
        
        v_ptrs = V + pid_b * v_stride_b + pid_h * v_stride_h + \
                 offs_n_kv_block[:, None] * v_stride_n + offs_d[None, :] * v_stride_d 
        v_block = tl.load(v_ptrs, mask=(offs_n_kv_block[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)
        
        acc_o = acc_o * tl.exp(m_i_prev - m_i)
        acc_o += tl.dot(p_scores_block.to(Q.dtype.element_ty), v_block)

    l_i_safe = l_i + 1e-9 
    o_block = acc_o / l_i_safe
    
    out_ptrs = Out + pid_b * out_stride_b + pid_h * out_stride_h + \
               offs_m[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, o_block.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None,:] < head_dim))

    lse_val = m_i + tl.log(l_i_safe)
    lse_out_ptrs = LSE_out + pid_b * lse_out_stride_b + pid_h * lse_out_stride_h + offs_m * lse_out_stride_m
    tl.store(lse_out_ptrs, lse_val, mask=offs_m < seq_len_q) 