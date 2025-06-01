import triton  # type: ignore[import-unresolved]
import triton.language as tl  # type: ignore[import-unresolved]
import math  # needed only for documentation clarity (no runtime use)

# Pre-computed constant so that 2 ** (x * LOG2_E) == exp(x).
# Must be declared as a Triton constexpr so that it is visible inside JITed
# kernels.
LOG2_E = tl.constexpr(1.4426950408889634)  # 1 / ln(2)

@triton.jit
def _sigmoid(x):
    """Numerically-stable sigmoid using base-2 exponentials."""
    return 1.0 / (1.0 + tl.exp2(-x * LOG2_E))

_configs = []
# (BLOCK_M, BLOCK_N, num_warps)
_tile_specs = [
    # Baseline tiles for short sequences
    (64, 64, 4), (64, 128, 8), (128, 64, 8), (128, 128, 8),
    # Extra tall / wide tiles for long sequences
    (256, 64, 8), (256, 128, 8), (128, 256, 8), (64, 256, 8),
    # Square large tile – good arithmetic intensity for very long sequences
    (256, 256, 8),
]

for _BM, _BN, _WARPS in _tile_specs:
    for _STAGES in (4, 6):
        _configs.append(
            triton.Config({"BLOCK_M": _BM, "BLOCK_N": _BN}, num_stages=_STAGES, num_warps=_WARPS)
        )

def _prune_configs(configs, named_args, **kwargs):
    seq_len_q = kwargs["seq_len_q"] if "seq_len_q" in kwargs else kwargs.get("seq_len_q", 0)
    return [c for c in configs if c.kwargs["BLOCK_M"] <= seq_len_q]


@triton.autotune(configs=_configs, key=["seq_len_q", "head_dim"], prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _co_alibi_fwd_kernel_simplified(
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
    # z_running = tl.zeros((BLOCK_M, 1), dtype=tl.float32) # Commented out

    KV_BLOCKS = tl.cdiv(seq_len_kv, BLOCK_N)

    for blk_idx_from_right in tl.range(KV_BLOCKS):
        start_k = (KV_BLOCKS - 1 - blk_idx_from_right) * BLOCK_N
        offs_n = start_k + tl.arange(0, BLOCK_N)
        row_end = pid_m * BLOCK_M + (BLOCK_M - 1)
        if (not HAS_CAUSAL_MASK) or (start_k <= row_end):
            k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_k  # (N, D)
            k_block_nd = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            k_block = tl.trans(k_block_nd)

            v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
            v_block = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim),
                other=0.0,
            )

            p_raw_blk = tl.dot(q_block, k_block) * sm_scale
            key_valid_mask = offs_n[None, :] < seq_len_kv
            causal_mask = offs_n[None, :] > offs_m[:, None]
            if HAS_CAUSAL_MASK:
                p_raw_blk = tl.where(causal_mask, causal_mask_value, p_raw_blk)
            p_raw_blk = tl.where(key_valid_mask, p_raw_blk, causal_mask_value)

            m_i_new = tl.maximum(m_i, tl.max(p_raw_blk, axis=1)[:, None])
            exp_diff = tl.exp2((m_i - m_i_new) * LOG2_E)
            exp_scores = tl.exp2((p_raw_blk - m_i_new) * LOG2_E)
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