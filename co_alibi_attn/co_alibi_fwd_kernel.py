import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]

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

    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    bid = pid_bh // num_heads
    hid = pid_bh %  num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + bid * q_stride_b + hid * q_stride_h + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_block = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    m_i = tl.full((BLOCK_M, 1), -float('inf'), dtype=tl.float32) 
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    z_running = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for blk_idx_from_right in tl.static_range(KV_BLOCKS):
        start_k = (KV_BLOCKS - 1 - blk_idx_from_right) * BLOCK_N
        offs_n = start_k + tl.arange(0, BLOCK_N)

        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_block = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
        v_block = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)

        p_raw_blk = tl.dot(q_block, k_block) * sm_scale
        key_valid_mask = offs_n[None, :] < seq_len_kv
        p_raw_masked_for_sigma_blk = p_raw_blk 
        if HAS_CAUSAL_MASK:
            causal_mask = offs_n[None, :] > offs_m[:, None]
            p_raw_masked_for_sigma_blk = tl.where(causal_mask, -float('inf'), p_raw_masked_for_sigma_blk)
        p_raw_masked_for_sigma_blk = tl.where(key_valid_mask, p_raw_masked_for_sigma_blk, -float('inf'))
        
        sig_blk = _sigmoid(p_raw_masked_for_sigma_blk)
        if HAS_CAUSAL_MASK:
            sig_blk = tl.where(causal_mask, 0.0, sig_blk)
        sig_blk = tl.where(key_valid_mask, sig_blk, 0.0)

        prefix_sum = tl.cumsum(sig_blk, 1)
        row_total = tl.sum(sig_blk, 1)[:, None]
        z_blk = row_total - prefix_sum + sig_blk + z_running
        z_running += row_total

        p_adj_blk = p_raw_blk - z_blk
        if HAS_CAUSAL_MASK:
            p_adj_blk = tl.where(causal_mask, causal_mask_value, p_adj_blk)
        p_adj_blk = tl.where(key_valid_mask, p_adj_blk, causal_mask_value)

        m_i_new = tl.maximum(m_i, tl.max(p_adj_blk, axis=1)[:, None])
        exp_diff = tl.exp(m_i - m_i_new) 
        exp_p_adj_minus_m_new = tl.exp(p_adj_blk - m_i_new)
        if HAS_CAUSAL_MASK: 
            exp_p_adj_minus_m_new = tl.where(causal_mask, 0.0, exp_p_adj_minus_m_new)
        key_valid_mask_for_scores = key_valid_mask
        exp_p_adj_minus_m_new = tl.where(key_valid_mask_for_scores, exp_p_adj_minus_m_new, 0.0)
        l_i = l_i * exp_diff + tl.sum(exp_p_adj_minus_m_new, axis=1)[:, None]
        m_i = m_i_new 
        
        p_scores_blk = exp_p_adj_minus_m_new
        o_acc = o_acc * exp_diff + tl.dot(p_scores_blk.to(v_block.dtype), v_block) 

        # --- Debug ---
        # key_valid_mask_store = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
        # p_raw_out_ptr = P_raw_out + bid * p_raw_out_stride_b + hid * p_raw_out_stride_h + offs_m[:, None] * p_raw_out_stride_m + offs_n[None, :] * p_raw_out_stride_n
        # sig_ptr       = Sig_P_raw_out + bid * sig_p_raw_out_stride_b + hid * sig_p_raw_out_stride_h + offs_m[:, None] * sig_p_raw_out_stride_m + offs_n[None, :] * sig_p_raw_out_stride_n
        # z_ptr         = Z_penalty_out + bid * z_penalty_out_stride_b + hid * z_penalty_out_stride_h + offs_m[:, None] * z_penalty_out_stride_m + offs_n[None, :] * z_penalty_out_stride_n
        #
        # tl.store(p_raw_out_ptr, p_raw_blk.to(P_raw_out.dtype.element_ty), mask=key_valid_mask_store)
        # tl.store(sig_ptr,      sig_blk.to(Sig_P_raw_out.dtype.element_ty), mask=key_valid_mask_store)
        # tl.store(z_ptr,        z_blk.to(Z_penalty_out.dtype.element_ty),    mask=key_valid_mask_store)
    
    o_blk = o_acc / (l_i + 1e-9) 

    out_ptrs = Out + bid * out_stride_b + hid * out_stride_h + offs_m[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, o_blk.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))

    lse_val = m_i + tl.log(l_i + 1e-9) 
    lse_flat = tl.reshape(lse_val, (BLOCK_M,)) 
    lse_ptrs = LSE_out + bid * lse_out_stride_b + hid * lse_out_stride_h + offs_m * lse_out_stride_m
    tl.store(lse_ptrs, lse_flat.to(LSE_out.dtype.element_ty), mask=offs_m < seq_len_q) 