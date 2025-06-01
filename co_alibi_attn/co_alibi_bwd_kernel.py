import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def _co_alibi_bwd_kernel(
    Q, K, V, sm_scale, causal_mask_value,
    LSE_in,  # (B, H, M)  float32
    DO,       # upstream grad (B, H, M, D)
    O,        # forward output (B, H, M, D)
    DQ, DK, DV,
    # Strides ------------------------------------------------------------------
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    lse_in_stride_b, lse_in_stride_h, lse_in_stride_m,
    do_stride_b, do_stride_h, do_stride_m, do_stride_d,
    dq_stride_b, dq_stride_h, dq_stride_m, dq_stride_k,
    dk_stride_b, dk_stride_h, dk_stride_n, dk_stride_k,
    dv_stride_b, dv_stride_h, dv_stride_n, dv_stride_d,
    o_stride_b, o_stride_h, o_stride_m, o_stride_d,
    # Debug output (optional) -------------------------------------------------
    DP_RAW_OUT, S_OUT,
    dp_raw_out_stride_b, dp_raw_out_stride_h, dp_raw_out_stride_m, dp_raw_out_stride_n,
    s_out_stride_b, s_out_stride_h, s_out_stride_m, s_out_stride_n,
    # Meta ---------------------------------------------------------------------
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N_KV: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    """Backward pass with recomputation, memory-free.
    * BLOCK_M      – rows (queries) per program-id (threadblock)
    * BLOCK_N_KV   – columns (keys/values) processed per iteration
    * BLOCK_DMODEL – head dim tile loaded at once (should equal head_dim)
    The algorithm follows Sec.-B of the Co-ALIBI note.
    """

    # -------------------- program ids ----------------------------------------
    pid_m  = tl.program_id(axis=0)                       # along queries
    pid_bh = tl.program_id(axis=1)                       # batch*head
    bid    = pid_bh // num_heads
    hid    = pid_bh % num_heads

    # -------------------- offsets -------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # (M,)
    offs_d = tl.arange(0, BLOCK_DMODEL)                  # (D,)

    # -------------------- load Q, dO ----------------------------------------
    q_ptrs = Q + bid * q_stride_b + hid * q_stride_h + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_blk  = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    do_ptrs = DO + bid * do_stride_b + hid * do_stride_h + offs_m[:, None] * do_stride_m + offs_d[None, :] * do_stride_d
    do_blk  = tl.load(do_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    # Load corresponding forward output block
    o_ptrs = O + bid * o_stride_b + hid * o_stride_h + offs_m[:, None] * o_stride_m + offs_d[None, :] * o_stride_d
    o_blk  = tl.load(o_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    # LSE row (stored as float32) ------------------------------------------------
    lse_ptrs = LSE_in + bid * lse_in_stride_b + hid * lse_in_stride_h + offs_m * lse_in_stride_m
    lse_row  = tl.load(lse_ptrs, mask=offs_m < seq_len_q, other=0.0)[:, None]

    # ---------------- accumulators -------------------------------------------
    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # For DK & DV we use atomics (they are accumulated across TBs)

    # First pass: compute per-row total sigmoid sum ---------------------------
    row_sig_total = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n = start_n + tl.arange(0, BLOCK_N_KV)

        # Load K tile (D, N)
        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_blk  = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)

        # Recompute raw scores & sigmoid -------------------------------------
        p_raw = tl.dot(q_blk, k_blk) * sm_scale  # (M,N)

        causal_mask = offs_m[:, None] < offs_n[None, :]
        if HAS_CAUSAL_MASK:
            p_raw = tl.where(causal_mask, -float('inf'), p_raw)

        sig = _sigmoid(p_raw)
        if HAS_CAUSAL_MASK:
            sig = tl.where(causal_mask, 0.0, sig)

        valid_mask = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
        sig = tl.where(valid_mask, sig, 0.0)

        # Accumulate in fp32 for numerical accuracy
        row_sig_total += tl.sum(sig.to(tl.float32), axis=1)[:, None]

    # -------------------- NEW PASS: compute per-row sum(ds * s) -------------
    # -------------------- row_ds_s_total via   Σ_d (do * o) ---------------
    row_ds_s_total = tl.sum(do_blk.to(tl.float32) * o_blk.to(tl.float32), axis=1)[:, None]

    # -------------------- THIRD PASS: compute gradients --------------------
    prefix_sig_sum_row = tl.zeros((BLOCK_M, 1), dtype=tl.float32)  # Σ sig up to prev key
    prefix_dp_adj_row  = tl.zeros((BLOCK_M, 1), dtype=tl.float32)  # Σ dp_adjusted up to prev key

    for start_n in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n = start_n + tl.arange(0, BLOCK_N_KV)

        # K (D,N) and V (N,D) tiles -----------------------------------------
        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_blk  = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)

        v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
        v_blk  = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)

        # Recompute scores, sigmoid, z_penalty, softmax ----------------------
        p_raw = tl.dot(q_blk, k_blk) * sm_scale  # (M,N)
        causal_mask = offs_m[:, None] < offs_n[None, :]
        if HAS_CAUSAL_MASK:
            p_raw = tl.where(causal_mask, -float('inf'), p_raw)

        sig = _sigmoid(p_raw)
        if HAS_CAUSAL_MASK:
            sig = tl.where(causal_mask, 0.0, sig)

        valid_mask = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
        sig = tl.where(valid_mask, sig, 0.0)

        sig_f32 = sig.to(tl.float32)

        # z_penalty: suffix sum of sig ---------------------------------------
        prefix_sig_local = tl.cumsum(sig_f32, axis=1)                # running prefix inside tile
        z_penalty = row_sig_total - (prefix_sig_sum_row + prefix_sig_local) + sig_f32
        prefix_sig_sum_row += tl.sum(sig_f32, axis=1)[:, None]

        p_adj = p_raw - z_penalty
        if HAS_CAUSAL_MASK:
            p_adj = tl.where(causal_mask, causal_mask_value, p_adj)
        p_adj = tl.where(valid_mask, p_adj, causal_mask_value)

        # softmax probability s ---------------------------------------------
        s = tl.exp(p_adj - lse_row)
        if HAS_CAUSAL_MASK:
            s = tl.where(causal_mask, 0.0, s)
        s = tl.where(valid_mask, s, 0.0)

        # --- dv contribution ------------------------------------------------
        dv_contrib = tl.dot(tl.trans(s.to(tl.float32)), do_blk.to(tl.float32))  # (N,D)
        dv_ptrs = DV + bid * dv_stride_b + hid * dv_stride_h + offs_n[:, None] * dv_stride_n + offs_d[None, :] * dv_stride_d
        tl.atomic_add(dv_ptrs, dv_contrib.to(DV.dtype.element_ty), mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim))

        # --- ds -------------------------------------------------------------
        ds = tl.dot(do_blk.to(tl.float32), tl.trans(v_blk.to(tl.float32)))  # (M,N)

        dp_adj = s.to(tl.float32) * (ds - row_ds_s_total)  # (M,N)

        if HAS_CAUSAL_MASK:
            dp_adj = tl.where(causal_mask, 0.0, dp_adj)
        dp_adj = tl.where(valid_mask, dp_adj, 0.0)

        # --- sigma'(p_raw) --------------------------------------------------
        sigma_prime = sig.to(tl.float32) * (1.0 - sig.to(tl.float32))

        # prefix of dp_adj ----------------------------------------------------
        prefix_dp_local = tl.cumsum(dp_adj, axis=1)
        c_block = prefix_dp_adj_row + prefix_dp_local
        prefix_dp_adj_row += tl.sum(dp_adj, axis=1)[:, None]

        dp_raw = dp_adj - sigma_prime * c_block
        if HAS_CAUSAL_MASK:
            dp_raw = tl.where(causal_mask, 0.0, dp_raw)
        dp_raw = tl.where(valid_mask, dp_raw, 0.0)

        # ----------------- debug store -------------------------------------
        if DEBUG:
            dp_raw_out_ptr = DP_RAW_OUT + bid * dp_raw_out_stride_b + hid * dp_raw_out_stride_h + offs_m[:, None] * dp_raw_out_stride_m + offs_n[None, :] * dp_raw_out_stride_n
            s_out_ptr      = S_OUT      + bid * s_out_stride_b      + hid * s_out_stride_h      + offs_m[:, None] * s_out_stride_m      + offs_n[None, :] * s_out_stride_n
            tl.store(dp_raw_out_ptr, dp_raw, mask=valid_mask)
            tl.store(s_out_ptr,      s,      mask=valid_mask)

        # --- DK accumulation (atomic) --------------------------------------
        dk_contrib = tl.dot(tl.trans(dp_raw * sm_scale), q_blk.to(tl.float32))  # (N,D)
        dk_ptrs = DK + bid * dk_stride_b + hid * dk_stride_h + offs_n[:, None] * dk_stride_n + offs_d[None, :] * dk_stride_k
        tl.atomic_add(dk_ptrs, dk_contrib.to(DK.dtype.element_ty), mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim))

        # --- DQ accumulator -------------------------------------------------
        dq_acc += tl.dot(dp_raw * sm_scale, tl.trans(k_blk.to(tl.float32)))

    # Store DQ ----------------------------------------------------------------
    dq_ptrs = DQ + bid * dq_stride_b + hid * dq_stride_h + offs_m[:, None] * dq_stride_m + offs_d[None, :] * dq_stride_k
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)) 