import triton  # type: ignore[import-unresolved]
import triton.language as tl  # type: ignore[import-unresolved]
import math  # needed only for documentation clarity (no runtime use)

LOG2_E = tl.constexpr(1.4426950408889634)

# ---------------- Autotune configurations ----------------
# To improve performance on large sequence lengths (e.g., 4k) while avoiding
# register spilling, we keep the tile sizes relatively small. Six hand-picked
# configurations are provided for Triton to autotune over. These configs vary
# the sequence tile sizes (BLOCK_M, BLOCK_N), number of warps and pipeline
# stages — a good starting point for balancing arithmetic intensity and memory
# pressure for Co-ALIBI's heavier register usage.
_configs = [
    # --- Small baseline tiles ---
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},   num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},   num_warps=4, num_stages=6),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128},  num_warps=4, num_stages=6),

    # --- Medium tiles with taller query chunk ---
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},   num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},   num_warps=4, num_stages=6),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},   num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},   num_warps=4, num_stages=6),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128},  num_warps=8, num_stages=4),

    # --- Larger tiles: keep within register limits ---
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},   num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},   num_warps=4, num_stages=6),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},   num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},   num_warps=8, num_stages=6),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128},  num_warps=8, num_stages=6),
]

def _prune_configs(configs, named_args, **kwargs):
    """Prune configs that would overshoot the actual sequence length."""
    seq_len_q = kwargs.get("seq_len_q", 0)
    return [cfg for cfg in configs if cfg.kwargs["BLOCK_M"] <= seq_len_q]

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * LOG2_E))

@triton.autotune(configs=_configs, key=["seq_len_q", "head_dim"], prune_configs_by={"early_config_prune": _prune_configs})
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
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

    # Compute the number of key/value tiles this kernel will iterate over.
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

            prefix_sum = tl.cumsum(sig_blk, 1)
            row_total = tl.sum(sig_blk, 1)[:, None]
            z_blk = row_total - prefix_sum + sig_blk + z_running
            z_running += row_total

            p_adj_blk = p_raw_blk - z_blk
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