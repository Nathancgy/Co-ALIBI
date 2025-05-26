import triton
import triton.language as tl

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def _co_alibi_fwd_kernel(
    Q, K, V, sm_scale, causal_mask_value,
    P_raw_out, Sig_P_raw_out, Z_penalty_out, LSE_out,
    Out,
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    out_stride_b, out_stride_h, out_stride_m, out_stride_d,
    p_raw_out_stride_b, p_raw_out_stride_h, p_raw_out_stride_m, p_raw_out_stride_n,
    sig_p_raw_out_stride_b, sig_p_raw_out_stride_h, sig_p_raw_out_stride_m, sig_p_raw_out_stride_n,
    z_penalty_out_stride_b, z_penalty_out_stride_h, z_penalty_out_stride_m, z_penalty_out_stride_n,
    lse_out_stride_b, lse_out_stride_h, lse_out_stride_m,
    batch_size: tl.constexpr, num_heads: tl.constexpr,
    seq_len_q: tl.constexpr, seq_len_kv: tl.constexpr, head_dim: tl.constexpr,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, KV_BLOCKS: tl.constexpr
):

    pid_m   = tl.program_id(axis=0)
    pid_bh  = tl.program_id(axis=1)

    # Disable device-side printing for benchmarking/profiling
    SHOULD_PRINT = False  # set to True manually if low-level debug is needed

    bid     = pid_bh // num_heads
    hid     = pid_bh %  num_heads

    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)

    q_ptrs = Q + bid * q_stride_b + hid * q_stride_h + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_block = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0).to(tl.float32)
    if SHOULD_PRINT: tl.device_print("Triton q_block", pid_m, pid_bh, q_block)

    m_i = tl.full((BLOCK_M, 1), -float('inf'), dtype=tl.float32) 
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    z_running = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    if SHOULD_PRINT: tl.device_print("Triton z_running_initial", pid_m, pid_bh, z_running)

    for blk_idx_from_right in tl.static_range(KV_BLOCKS):
        start_k = (KV_BLOCKS - 1 - blk_idx_from_right) * BLOCK_N
        offs_n = start_k + tl.arange(0, BLOCK_N)

        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_block = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        if SHOULD_PRINT: tl.device_print(f"Triton k_block_idx{blk_idx_from_right}", pid_m, pid_bh, k_block)

        v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
        v_block = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)

        p_raw_blk = tl.dot(q_block, k_block.to(tl.float32)) * sm_scale  # float32
        if SHOULD_PRINT: tl.device_print(f"Triton p_raw_blk_idx{blk_idx_from_right}", pid_m, pid_bh, p_raw_blk)

        # ------------------------------------------------------------------
        # Construct masks ---------------------------------------------------
        # ------------------------------------------------------------------
        key_valid_mask = offs_n[None, :] < seq_len_kv  # (BLOCK_M, BLOCK_N) broadcast across query rows
        p_raw_masked_for_sigma_blk = p_raw_blk 
        if HAS_CAUSAL_MASK:
            causal_mask = offs_n[None, :] > offs_m[:, None]  # True where j > i
            p_raw_masked_for_sigma_blk = tl.where(causal_mask, -float('inf'), p_raw_masked_for_sigma_blk)
        # For positions beyond sequence length, also mask to -inf so sigmoid becomes 0
        p_raw_masked_for_sigma_blk = tl.where(key_valid_mask, p_raw_masked_for_sigma_blk, -float('inf'))
        if SHOULD_PRINT: tl.device_print(f"Triton p_raw_masked_idx{blk_idx_from_right}", pid_m, pid_bh, p_raw_masked_for_sigma_blk)
        
        sig_blk = _sigmoid(p_raw_masked_for_sigma_blk)
        # zero-out any invalid/causal positions so they do not contribute downstream
        if HAS_CAUSAL_MASK:
            sig_blk = tl.where(causal_mask, 0.0, sig_blk)
        sig_blk = tl.where(key_valid_mask, sig_blk, 0.0)
        if SHOULD_PRINT: tl.device_print(f"Triton sig_blk_idx{blk_idx_from_right}", pid_m, pid_bh, sig_blk)

        if SHOULD_PRINT: tl.device_print(f"Triton z_running_before_idx{blk_idx_from_right}", pid_m, pid_bh, z_running)
        
        current_block_z_sum = tl.zeros((BLOCK_M, 1), dtype=tl.float32) 
        temp_z_blk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) 
        
        for ii in tl.static_range(BLOCK_N):
            col_idx_in_block = BLOCK_N - 1 - ii 
            col_mask = (tl.arange(0, BLOCK_N) == col_idx_in_block)
            s_increment_col = tl.sum(sig_blk * col_mask[None, :], axis=1)[:, None] 
            current_block_z_sum += s_increment_col
            val_for_z_col = current_block_z_sum + z_running 
            temp_z_blk = tl.where(col_mask[None, :], val_for_z_col, temp_z_blk)

        z_blk = temp_z_blk 
        z_running += current_block_z_sum 
        
        if SHOULD_PRINT: tl.device_print(f"Triton z_blk_idx{blk_idx_from_right}", pid_m, pid_bh, z_blk)
        if SHOULD_PRINT: tl.device_print(f"Triton z_running_after_idx{blk_idx_from_right}", pid_m, pid_bh, z_running)

        p_adj_blk = p_raw_blk - z_blk
        if HAS_CAUSAL_MASK:
            p_adj_blk = tl.where(causal_mask, causal_mask_value, p_adj_blk)
        # Ensure out-of-range keys cannot influence maxima or softmax sums
        p_adj_blk = tl.where(key_valid_mask, p_adj_blk, causal_mask_value)
        if SHOULD_PRINT: tl.device_print(f"Triton p_adj_blk_idx{blk_idx_from_right}", pid_m, pid_bh, p_adj_blk)

        if SHOULD_PRINT: tl.device_print(f"Triton m_i_before_idx{blk_idx_from_right}", pid_m, pid_bh, m_i)
        if SHOULD_PRINT: tl.device_print(f"Triton l_i_before_idx{blk_idx_from_right}", pid_m, pid_bh, l_i)

        m_i_new = tl.maximum(m_i, tl.max(p_adj_blk, axis=1)[:, None])
        if SHOULD_PRINT: tl.device_print(f"Triton m_i_new_idx{blk_idx_from_right}", pid_m, pid_bh, m_i_new)
        
        exp_diff = tl.exp(m_i - m_i_new) 
        
        exp_p_adj_minus_m_new = tl.exp(p_adj_blk - m_i_new)
        if HAS_CAUSAL_MASK: 
            exp_p_adj_minus_m_new = tl.where(causal_mask, 0.0, exp_p_adj_minus_m_new)
        
        key_valid_mask_for_scores = key_valid_mask
        exp_p_adj_minus_m_new = tl.where(key_valid_mask_for_scores, exp_p_adj_minus_m_new, 0.0)

        if SHOULD_PRINT: tl.device_print(f"Triton exp_diff_idx{blk_idx_from_right}", pid_m, pid_bh, exp_p_adj_minus_m_new)

        l_i = l_i * exp_diff + tl.sum(exp_p_adj_minus_m_new, axis=1)[:, None]
        m_i = m_i_new 

        if SHOULD_PRINT: tl.device_print(f"Triton m_i_after_idx{blk_idx_from_right}", pid_m, pid_bh, m_i)
        if SHOULD_PRINT: tl.device_print(f"Triton l_i_after_idx{blk_idx_from_right}", pid_m, pid_bh, l_i)
        
        p_scores_blk = tl.exp(p_adj_blk - m_i) 
        if HAS_CAUSAL_MASK: 
            p_scores_blk = tl.where(causal_mask, 0.0, p_scores_blk)
        
        p_scores_blk = tl.where(key_valid_mask_for_scores, p_scores_blk, 0.0)
        if SHOULD_PRINT: tl.device_print(f"Triton p_scores_blk_idx{blk_idx_from_right}", pid_m, pid_bh, p_scores_blk)

        o_acc = o_acc * exp_diff + tl.dot(p_scores_blk.to(v_block.dtype), v_block) 

        key_valid_mask_store = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
        # --- Debug stores (commented out for performance benchmarking) ---
        # p_raw_out_ptr = P_raw_out + bid * p_raw_out_stride_b + hid * p_raw_out_stride_h + offs_m[:, None] * p_raw_out_stride_m + offs_n[None, :] * p_raw_out_stride_n
        # sig_ptr       = Sig_P_raw_out + bid * sig_p_raw_out_stride_b + hid * sig_p_raw_out_stride_h + offs_m[:, None] * sig_p_raw_out_stride_m + offs_n[None, :] * sig_p_raw_out_stride_n
        # z_ptr         = Z_penalty_out + bid * z_penalty_out_stride_b + hid * z_penalty_out_stride_h + offs_m[:, None] * z_penalty_out_stride_m + offs_n[None, :] * z_penalty_out_stride_n
        #
        # tl.store(p_raw_out_ptr, p_raw_blk.to(P_raw_out.dtype.element_ty), mask=key_valid_mask_store)
        # tl.store(sig_ptr,      sig_blk.to(Sig_P_raw_out.dtype.element_ty), mask=key_valid_mask_store)
        # tl.store(z_ptr,        z_blk.to(Z_penalty_out.dtype.element_ty),    mask=key_valid_mask_store)

    if SHOULD_PRINT: tl.device_print("Triton m_i_final", pid_m, pid_bh, m_i)
    if SHOULD_PRINT: tl.device_print("Triton l_i_final", pid_m, pid_bh, l_i)
    if SHOULD_PRINT: tl.device_print("Triton o_acc_final", pid_m, pid_bh, o_acc)
    
    o_blk = o_acc / (l_i + 1e-9) 
    if SHOULD_PRINT: tl.device_print("Triton o_blk_final", pid_m, pid_bh, o_blk)

    out_ptrs = Out + bid * out_stride_b + hid * out_stride_h + offs_m[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, o_blk.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))

    lse_val = m_i + tl.log(l_i + 1e-9) 
    if SHOULD_PRINT: tl.device_print("Triton lse_val_final", pid_m, pid_bh, lse_val)
    
    lse_flat = tl.reshape(lse_val, (BLOCK_M,)) 
    lse_ptrs = LSE_out + bid * lse_out_stride_b + hid * lse_out_stride_h + offs_m * lse_out_stride_m
    tl.store(lse_ptrs, lse_flat.to(LSE_out.dtype.element_ty), mask=offs_m < seq_len_q) 