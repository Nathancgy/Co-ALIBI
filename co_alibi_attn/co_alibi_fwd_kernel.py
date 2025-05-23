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

    offs_m_global = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_sram_idx = tl.arange(0, BLOCK_M) 
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + pid_b * q_stride_b + pid_h * q_stride_h + \
             offs_m_global[:, None] * q_stride_m + offs_d[None, :] * q_stride_k

    p_raw_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)
    sig_p_raw_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)

    # --- Pass 1: Compute P_raw and Sig(P_raw) ---
    for start_n_kv_block in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_global_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        
        k_ptrs = K + pid_b * k_stride_b + pid_h * k_stride_h + \
                 offs_n_kv_global_block[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        
        k_block = tl.load(k_ptrs, mask=(offs_n_kv_global_block[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        q_block = tl.load(q_ptrs, mask=(offs_m_global[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)
        p_raw_block = tl.dot(q_block, k_block) * sm_scale # Shape (BLOCK_M, BLOCK_N_KV)

        p_raw_out_ptr = P_raw_out + pid_b * p_raw_out_stride_b + pid_h * p_raw_out_stride_h + \
                        offs_m_global[:, None] * p_raw_out_stride_m + offs_n_kv_global_block[None, :] * p_raw_out_stride_n
        tl.store(p_raw_out_ptr, p_raw_block, mask=(offs_m_global[:, None] < seq_len_q) & (offs_n_kv_global_block[None, :] < seq_len_kv))
        
        offs_n_sram_target_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        target_sram_ptrs_p_raw = p_raw_sram + \
                                 offs_m_sram_idx[:, None] * N_CTX_KV + \
                                 offs_n_sram_target_block[None, :]
        mask_sram_store_p_raw = (offs_m_global[:, None] < seq_len_q) & \
                                (offs_n_sram_target_block[None, :] < seq_len_kv) & \
                                (offs_n_sram_target_block[None, :] < N_CTX_KV)
        tl.store(target_sram_ptrs_p_raw, p_raw_block, mask=mask_sram_store_p_raw)

        p_raw_for_sig = p_raw_block
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m_global[:, None] < offs_n_kv_global_block[None, :]
            p_raw_for_sig = tl.where(causal_cond, -float('inf'), p_raw_for_sig)
            
        sig_p_raw_block = _sigmoid(p_raw_for_sig) # Shape (BLOCK_M, BLOCK_N_KV)
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m_global[:, None] < offs_n_kv_global_block[None, :]
            sig_p_raw_block = tl.where(causal_cond, 0.0, sig_p_raw_block)
        
        sig_p_raw_out_ptr = Sig_P_raw_out + pid_b * sig_p_raw_out_stride_b + pid_h * sig_p_raw_out_stride_h + \
                            offs_m_global[:, None] * sig_p_raw_out_stride_m + offs_n_kv_global_block[None, :] * sig_p_raw_out_stride_n
        tl.store(sig_p_raw_out_ptr, sig_p_raw_block, mask=(offs_m_global[:, None] < seq_len_q) & (offs_n_kv_global_block[None, :] < seq_len_kv))

        target_sram_ptrs_sig_p_raw = sig_p_raw_sram + \
                                     offs_m_sram_idx[:, None] * N_CTX_KV + \
                                     offs_n_sram_target_block[None, :]
        mask_sram_store_sig_p_raw = (offs_m_global[:, None] < seq_len_q) & \
                                    (offs_n_sram_target_block[None, :] < seq_len_kv) & \
                                    (offs_n_sram_target_block[None, :] < N_CTX_KV)
        tl.store(target_sram_ptrs_sig_p_raw, sig_p_raw_block, mask=mask_sram_store_sig_p_raw)

    # --- Pass 2: Compute Z_penalty from sig_p_raw_sram ---
    z_penalty_sram = tl.zeros((BLOCK_M, N_CTX_KV), dtype=tl.float32)
    current_sum_for_z = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for sram_col_idx in range(N_CTX_KV - 1, -1, -1):
        current_sram_col_offs = sram_col_idx + tl.arange(0, 1)
        
        sig_val_at_sram_col_ptrs = sig_p_raw_sram + \
                                   offs_m_sram_idx[:, None] * N_CTX_KV + \
                                   current_sram_col_offs[None, :]
        mask_load_sig_sram_col = (offs_m_global[:, None] < seq_len_q) & \
                                 (current_sram_col_offs[None, :] < seq_len_kv) 
        sig_val_loaded_col = tl.load(sig_val_at_sram_col_ptrs, mask=mask_load_sig_sram_col, other=0.0)
        
        current_sum_for_z += sig_val_loaded_col
        
        z_penalty_sram_store_ptrs = z_penalty_sram + \
                                    offs_m_sram_idx[:, None] * N_CTX_KV + \
                                    current_sram_col_offs[None, :]
        mask_store_z_sram_col = (offs_m_global[:, None] < seq_len_q) & \
                                (current_sram_col_offs[None, :] < seq_len_kv)
        tl.store(z_penalty_sram_store_ptrs, current_sum_for_z, mask=mask_store_z_sram_col)

    for start_n_kv_block_for_z_out in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_global_block_z = start_n_kv_block_for_z_out + tl.arange(0, BLOCK_N_KV)
        
        source_z_sram_ptrs = z_penalty_sram + \
                             offs_m_sram_idx[:, None] * N_CTX_KV + \
                             offs_n_kv_global_block_z[None, :]
        mask_load_z_sram_block = (offs_m_global[:, None] < seq_len_q) & \
                                 (offs_n_kv_global_block_z[None, :] < seq_len_kv) & \
                                 (offs_n_kv_global_block_z[None, :] < N_CTX_KV)
        z_penalty_block_loaded = tl.load(source_z_sram_ptrs, mask=mask_load_z_sram_block, other=0.0)
        
        z_penalty_out_ptr = Z_penalty_out + pid_b * z_penalty_out_stride_b + pid_h * z_penalty_out_stride_h + \
                            offs_m_global[:, None] * z_penalty_out_stride_m + offs_n_kv_global_block_z[None, :] * z_penalty_out_stride_n
        tl.store(z_penalty_out_ptr, z_penalty_block_loaded, mask=(offs_m_global[:, None] < seq_len_q) & (offs_n_kv_global_block_z[None, :] < seq_len_kv))

    # --- Pass 3: Compute P_adjusted, Softmax, and Output O ---
    m_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n_kv_block in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n_kv_global_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)

        offs_n_sram_source_block = start_n_kv_block + tl.arange(0, BLOCK_N_KV)
        source_sram_ptrs_p_raw = p_raw_sram + \
                                 offs_m_sram_idx[:, None] * N_CTX_KV + \
                                 offs_n_sram_source_block[None, :]
        mask_sram_load_p_raw = (offs_m_global[:, None] < seq_len_q) & \
                               (offs_n_sram_source_block[None, :] < seq_len_kv) & \
                               (offs_n_sram_source_block[None, :] < N_CTX_KV)
        p_raw_block_loaded = tl.load(source_sram_ptrs_p_raw, mask=mask_sram_load_p_raw, other=0.0)
        
        source_sram_ptrs_z = z_penalty_sram + \
                             offs_m_sram_idx[:, None] * N_CTX_KV + \
                             offs_n_sram_source_block[None, :]
        mask_sram_load_z = (offs_m_global[:, None] < seq_len_q) & \
                           (offs_n_sram_source_block[None, :] < seq_len_kv) & \
                           (offs_n_sram_source_block[None, :] < N_CTX_KV)
        z_penalty_block_loaded = tl.load(source_sram_ptrs_z, mask=mask_sram_load_z, other=0.0)

        p_adjusted_block = p_raw_block_loaded - z_penalty_block_loaded
        if HAS_CAUSAL_MASK:
            causal_cond = offs_m_global[:, None] < offs_n_kv_global_block[None, :]
            p_adjusted_block = tl.where(causal_cond, causal_mask_value, p_adjusted_block)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(p_adjusted_block, axis=1))
        p_adj_minus_max = p_adjusted_block - m_i
        p_scores_block = tl.exp(p_adj_minus_max)
        
        if HAS_CAUSAL_MASK:
            p_scores_block = tl.where(causal_cond, 0.0, p_scores_block)
        p_scores_block = tl.where(offs_n_kv_global_block[None, :] < seq_len_kv, p_scores_block, 0.0)

        l_i_update_term = tl.sum(p_scores_block, axis=1)
        l_i = l_i * tl.exp(m_i_prev - m_i) + l_i_update_term[:,None]
        
        v_ptrs = V + pid_b * v_stride_b + pid_h * v_stride_h + \
                 offs_n_kv_global_block[:, None] * v_stride_n + offs_d[None, :] * v_stride_d 
        v_block = tl.load(v_ptrs, mask=(offs_n_kv_global_block[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)
        
        typed_p_scores_block = p_scores_block.to(Q.dtype.element_ty)
        
        acc_o_update_factor = tl.exp(m_i_prev - m_i)
        acc_o = acc_o * acc_o_update_factor
        
        acc_o += tl.dot(typed_p_scores_block, v_block.to(Q.dtype.element_ty))

    l_i_safe = l_i + 1e-9
    o_block = acc_o / l_i_safe
    
    out_ptrs = Out + pid_b * out_stride_b + pid_h * out_stride_h + \
               offs_m_global[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, o_block.to(Out.dtype.element_ty), mask=(offs_m_global[:, None] < seq_len_q) & (offs_d[None,:] < head_dim))

    lse_val = m_i + tl.log(l_i_safe)
    lse_out_ptrs = LSE_out + pid_b * lse_out_stride_b + pid_h * lse_out_stride_h + offs_m_global * lse_out_stride_m
    tl.store(lse_out_ptrs, lse_val[:,0], mask=offs_m_global < seq_len_q) 