import triton  # type: ignore[import-unresolved]
import triton.language as tl  # type: ignore[import-unresolved]

LOG2_E = tl.constexpr(1.4426950408889634)

_configs = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=4),

    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=4),  # current best
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=6),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=6),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=8, num_stages=4),
]

def _prune_configs(configs, named_args, **kwargs):
    return configs

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * LOG2_E))

@triton.autotune(configs=_configs, key=["seq_len_q", "head_dim"], prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _co_alibi_fwd_kernel(
    Q, K, V, Slopes, sm_scale, causal_mask_value,
    LSE_out,
    RowSigTotal_out,
    Out,
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    out_stride_b, out_stride_h, out_stride_m, out_stride_d,
    lse_out_stride_b, lse_out_stride_h, lse_out_stride_m,
    row_sig_out_stride_b, row_sig_out_stride_h, row_sig_out_stride_m,
    batch_size: tl.constexpr, num_heads: tl.constexpr,
    seq_len_q: tl.constexpr, seq_len_kv: tl.constexpr, head_dim: tl.constexpr,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):

    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    bid = pid_bh // num_heads
    hid = pid_bh %  num_heads

    slopes_ptr = Slopes + hid
    slope = tl.load(slopes_ptr)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + bid * q_stride_b + hid * q_stride_h + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_block = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    m_i = tl.full((BLOCK_M, 1), -float('inf'), dtype=tl.float32) 
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    z_running = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    KV_BLOCKS = (seq_len_kv + BLOCK_N - 1) // BLOCK_N

    for blk_idx_from_right in tl.range(KV_BLOCKS):
        start_k = (KV_BLOCKS - 1 - blk_idx_from_right) * BLOCK_N
        offs_n = start_k + tl.arange(0, BLOCK_N)
        row_end = pid_m * BLOCK_M + (BLOCK_M - 1)
        proceed = (not HAS_CAUSAL_MASK) or (start_k <= row_end)
        if proceed:
            k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_k
            k_block_nd = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)
            k_block = tl.trans(k_block_nd)

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

            sig_blk_scaled = sig_blk

            prefix_sum = tl.cumsum(sig_blk_scaled, 1)
            row_total = tl.sum(sig_blk_scaled, 1)[:, None]
            z_blk = row_total - prefix_sum + sig_blk_scaled + z_running
            z_running += row_total

            p_adj_blk = p_raw_blk - slope * z_blk
            if HAS_CAUSAL_MASK:
                p_adj_blk = tl.where(causal_mask, causal_mask_value, p_adj_blk)
            p_adj_blk = tl.where(key_valid_mask, p_adj_blk, causal_mask_value)

            m_i_new = tl.maximum(m_i, tl.max(p_adj_blk, axis=1)[:, None])
            exp_diff = tl.exp2((m_i - m_i_new) * LOG2_E)
            exp_scores = tl.exp2((p_adj_blk - m_i_new) * LOG2_E)
            if HAS_CAUSAL_MASK:
                exp_scores = tl.where(causal_mask, 0.0, exp_scores)
            exp_scores = tl.where(key_valid_mask, exp_scores, 0.0)
            l_i = l_i * exp_diff + tl.sum(exp_scores, axis=1)[:, None]
            m_i = m_i_new

            o_acc = o_acc * exp_diff + tl.dot(exp_scores.to(v_block.dtype), v_block)

    o_blk = o_acc / (l_i + 1e-9) 

    out_ptrs = Out + bid * out_stride_b + hid * out_stride_h + offs_m[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, o_blk.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))

    lse_val = m_i + tl.log(l_i + 1e-9) 
    lse_flat = tl.reshape(lse_val, (BLOCK_M,)) 
    lse_ptrs = LSE_out + bid * lse_out_stride_b + hid * lse_out_stride_h + offs_m * lse_out_stride_m
    tl.store(lse_ptrs, lse_flat.to(LSE_out.dtype.element_ty), mask=offs_m < seq_len_q) 

    row_sig_total_flat = tl.reshape(z_running, (BLOCK_M,))
    row_sig_ptrs = RowSigTotal_out + bid * row_sig_out_stride_b + hid * row_sig_out_stride_h + offs_m * row_sig_out_stride_m
    tl.store(row_sig_ptrs, row_sig_total_flat.to(RowSigTotal_out.dtype.element_ty), mask=offs_m < seq_len_q) 