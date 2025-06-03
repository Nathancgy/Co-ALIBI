import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]

LOG2_E = tl.constexpr(1.4426950408889634)

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * LOG2_E))

@triton.jit
def _co_alibi_bwd_kernel(
    Q, K, V, sm_scale, causal_mask_value,
    LSE_in,
    DO,
    O,
    DQ, DK, DV,
    q_stride_b, q_stride_h, q_stride_m, q_stride_k,
    k_stride_b, k_stride_h, k_stride_n, k_stride_k,
    v_stride_b, v_stride_h, v_stride_n, v_stride_d,
    lse_in_stride_b, lse_in_stride_h, lse_in_stride_m,
    do_stride_b, do_stride_h, do_stride_m, do_stride_d,
    dq_stride_b, dq_stride_h, dq_stride_m, dq_stride_k,
    dk_stride_b, dk_stride_h, dk_stride_n, dk_stride_k,
    dv_stride_b, dv_stride_h, dv_stride_n, dv_stride_d,
    o_stride_b, o_stride_h, o_stride_m, o_stride_d,
    DP_RAW_OUT, S_OUT, DP_ADJ_OUT, SIG_PRIME_OUT, C_BLOCK_OUT, SIG_F32_OUT,
    S2_OUT, Z2_OUT, Z_OUT,
    SIG2_OUT,
    P_RAW_PASS2_OUT, P_RAW_PASS3_OUT,
    s2_out_stride_b, s2_out_stride_h, s2_out_stride_m, s2_out_stride_n,
    sig2_out_stride_b, sig2_out_stride_h, sig2_out_stride_m, sig2_out_stride_n,
    z2_out_stride_b, z2_out_stride_h, z2_out_stride_m, z2_out_stride_n,
    z_out_stride_b,  z_out_stride_h,  z_out_stride_m,  z_out_stride_n,
    dp_raw_out_stride_b, dp_raw_out_stride_h, dp_raw_out_stride_m, dp_raw_out_stride_n,
    s_out_stride_b, s_out_stride_h, s_out_stride_m, s_out_stride_n,
    dp_adj_out_stride_b, dp_adj_out_stride_h, dp_adj_out_stride_m, dp_adj_out_stride_n,
    sig_prime_out_stride_b, sig_prime_out_stride_h, sig_prime_out_stride_m, sig_prime_out_stride_n,
    c_block_out_stride_b, c_block_out_stride_h, c_block_out_stride_m, c_block_out_stride_n,
    sig_f32_out_stride_b, sig_f32_out_stride_h, sig_f32_out_stride_m, sig_f32_out_stride_n,
    p_raw_pass2_out_stride_b, p_raw_pass2_out_stride_h, p_raw_pass2_out_stride_m, p_raw_pass2_out_stride_n,
    p_raw_pass3_out_stride_b, p_raw_pass3_out_stride_h, p_raw_pass3_out_stride_m, p_raw_pass3_out_stride_n,
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
    HAS_CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N_KV: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    pid_m  = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    bid    = pid_bh // num_heads
    hid    = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # (M,)
    offs_d = tl.arange(0, BLOCK_DMODEL)                  # (D,)

    q_ptrs = Q + bid * q_stride_b + hid * q_stride_h + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_k
    q_blk  = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    do_ptrs = DO + bid * do_stride_b + hid * do_stride_h + offs_m[:, None] * do_stride_m + offs_d[None, :] * do_stride_d
    do_blk  = tl.load(do_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    o_ptrs = O + bid * o_stride_b + hid * o_stride_h + offs_m[:, None] * o_stride_m + offs_d[None, :] * o_stride_d
    o_blk  = tl.load(o_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    lse_ptrs = LSE_in + bid * lse_in_stride_b + hid * lse_in_stride_h + offs_m * lse_in_stride_m
    lse_row  = tl.load(lse_ptrs, mask=offs_m < seq_len_q, other=0.0)[:, None]
    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    row_sig_total = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # ------------------------------------------------------------
    # FIRST PASS
    # ------------------------------------------------------------

    for start_n in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n = start_n + tl.arange(0, BLOCK_N_KV)
        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_blk  = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        p_raw = tl.dot(q_blk, k_blk) * sm_scale

        causal_mask = offs_m[:, None] < offs_n[None, :]
        valid_mask = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)
        
        p_raw_for_sig = p_raw
        if HAS_CAUSAL_MASK:
            p_raw_for_sig = tl.where(causal_mask, -float('inf'), p_raw_for_sig)
        
        sig = _sigmoid(p_raw_for_sig)
        if HAS_CAUSAL_MASK:
            sig = tl.where(causal_mask, 0.0, sig)
        sig = tl.where(valid_mask, sig, 0.0)
        
        row_sig_total += tl.sum(sig.to(tl.float32), axis=1)[:, None]

    # ------------------------------------------------------------
    # SECOND PASS
    # ------------------------------------------------------------
    row_ds_s_total = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    prefix_sig_sum_row_pass2 = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n = start_n + tl.arange(0, BLOCK_N_KV)
        
        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_blk  = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
        v_blk  = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)

        # Recompute p_raw, masks and sigmoid
        p_raw = tl.dot(q_blk, k_blk) * sm_scale  # (M, N)
        causal_mask = offs_m[:, None] < offs_n[None, :]
        valid_mask  = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)

        if DEBUG: # Store p_raw for pass 2
            p_raw_pass2_out_ptr = P_RAW_PASS2_OUT + bid * p_raw_pass2_out_stride_b + hid * p_raw_pass2_out_stride_h + offs_m[:, None] * p_raw_pass2_out_stride_m + offs_n[None, :] * p_raw_pass2_out_stride_n
            tl.store(p_raw_pass2_out_ptr, p_raw, mask=valid_mask)

        p_raw_for_sig = p_raw
        if HAS_CAUSAL_MASK:
            p_raw_for_sig = tl.where(causal_mask, -float('inf'), p_raw_for_sig)

        sig = _sigmoid(p_raw_for_sig)
        if HAS_CAUSAL_MASK:
            sig = tl.where(causal_mask, 0.0, sig)
        sig = tl.where(valid_mask, sig, 0.0)
        sig_f32 = sig.to(tl.float32)

        prefix_sig_local = tl.cumsum(sig_f32, axis=1)
        z_penalty = row_sig_total - prefix_sig_sum_row_pass2 - prefix_sig_local + sig_f32
        prefix_sig_sum_row_pass2 += tl.sum(sig_f32, axis=1)[:, None]

        if HAS_CAUSAL_MASK:
            p_raw = tl.where(causal_mask, causal_mask_value, p_raw)
        p_raw = tl.where(valid_mask, p_raw, causal_mask_value)

        p_adj = p_raw - z_penalty
        log_s = p_adj - lse_row
        s = tl.exp(log_s)
        if HAS_CAUSAL_MASK:
            s = tl.where(causal_mask, 0.0, s)
        s = tl.where(valid_mask, s, 0.0)

        if DEBUG:
            s2_out_ptr = S2_OUT + bid * s2_out_stride_b + hid * s2_out_stride_h + offs_m[:, None] * s2_out_stride_m + offs_n[None, :] * s2_out_stride_n
            sig2_out_ptr = SIG2_OUT + bid * sig2_out_stride_b + hid * sig2_out_stride_h + offs_m[:, None] * sig2_out_stride_m + offs_n[None, :] * sig2_out_stride_n
            z2_out_ptr = Z2_OUT + bid * z2_out_stride_b + hid * z2_out_stride_h + offs_m[:, None] * z2_out_stride_m + offs_n[None, :] * z2_out_stride_n
            tl.store(s2_out_ptr, s, mask=valid_mask)
            tl.store(sig2_out_ptr, sig_f32, mask=valid_mask)
            tl.store(z2_out_ptr, z_penalty, mask=valid_mask)

        dv_contrib = tl.dot(tl.trans(s.to(tl.float32)), do_blk.to(tl.float32))
        dv_ptrs = DV + bid * dv_stride_b + hid * dv_stride_h + offs_n[:, None] * dv_stride_n + offs_d[None, :] * dv_stride_d
        tl.atomic_add(dv_ptrs, dv_contrib.to(DV.dtype.element_ty), mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim))

        ds = tl.dot(do_blk.to(tl.float32), tl.trans(v_blk.to(tl.float32)))  # (M, N)
        row_ds_s_total += tl.sum(s.to(tl.float32) * ds, axis=1)[:, None]

    # ------------------------------------------------------------
    # THIRD PASS
    # ------------------------------------------------------------

    prefix_sig_sum_row = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    prefix_dp_adj_row  = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, seq_len_kv, BLOCK_N_KV):
        offs_n = start_n + tl.arange(0, BLOCK_N_KV)

        k_ptrs = K + bid * k_stride_b + hid * k_stride_h + offs_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_k
        k_blk  = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len_kv) & (offs_d[:, None] < head_dim), other=0.0)
        v_ptrs = V + bid * v_stride_b + hid * v_stride_h + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d
        v_blk  = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim), other=0.0)

        p_raw = tl.dot(q_blk, k_blk) * sm_scale
        causal_mask = offs_m[:, None] < offs_n[None, :]
        valid_mask  = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_kv)

        if DEBUG:
            p_raw_pass3_out_ptr = P_RAW_PASS3_OUT + bid * p_raw_pass3_out_stride_b + hid * p_raw_pass3_out_stride_h + offs_m[:, None] * p_raw_pass3_out_stride_m + offs_n[None, :] * p_raw_pass3_out_stride_n
            tl.store(p_raw_pass3_out_ptr, p_raw, mask=valid_mask)

        p_raw_for_sig = p_raw
        if HAS_CAUSAL_MASK:
            p_raw_for_sig = tl.where(causal_mask, -float('inf'), p_raw_for_sig)

        sig = _sigmoid(p_raw_for_sig)
        if HAS_CAUSAL_MASK:
            sig = tl.where(causal_mask, 0.0, sig)
        sig = tl.where(valid_mask, sig, 0.0)
        sig_f32 = sig.to(tl.float32)

        prefix_sig_local = tl.cumsum(sig_f32, axis=1)
        z_penalty = row_sig_total - prefix_sig_sum_row - prefix_sig_local + sig_f32
        prefix_sig_sum_row += tl.sum(sig_f32, axis=1)[:, None]

        if HAS_CAUSAL_MASK:
            p_raw = tl.where(causal_mask, causal_mask_value, p_raw)
        p_raw = tl.where(valid_mask, p_raw, causal_mask_value)

        p_adj = p_raw - z_penalty
        log_s = p_adj - lse_row
        s = tl.exp(log_s)
        if HAS_CAUSAL_MASK:
            s = tl.where(causal_mask, 0.0, s)
        s = tl.where(valid_mask, s, 0.0)

        ds = tl.dot(do_blk.to(tl.float32), tl.trans(v_blk.to(tl.float32)))

        dp_adj = s.to(tl.float32) * (ds - row_ds_s_total)
        if HAS_CAUSAL_MASK:
            dp_adj = tl.where(causal_mask, 0.0, dp_adj)
        dp_adj = tl.where(valid_mask, dp_adj, 0.0)

        sigma_prime = sig_f32 * (1.0 - sig_f32)
        sigma_prime = tl.minimum(sigma_prime, 0.25)

        prefix_dp_local = tl.cumsum(dp_adj, axis=1)
        c_block = prefix_dp_adj_row + prefix_dp_local
        prefix_dp_adj_row += tl.sum(dp_adj, axis=1)[:, None]

        dp_raw = dp_adj - sigma_prime * c_block
        if HAS_CAUSAL_MASK:
            dp_raw = tl.where(causal_mask, 0.0, dp_raw)
        dp_raw = tl.where(valid_mask, dp_raw, 0.0)

        if DEBUG:
            dp_raw_out_ptr = DP_RAW_OUT + bid * dp_raw_out_stride_b + hid * dp_raw_out_stride_h + offs_m[:, None] * dp_raw_out_stride_m + offs_n[None, :] * dp_raw_out_stride_n
            s_out_ptr      = S_OUT      + bid * s_out_stride_b      + hid * s_out_stride_h      + offs_m[:, None] * s_out_stride_m      + offs_n[None, :] * s_out_stride_n
            z_out_ptr      = Z_OUT      + bid * z_out_stride_b      + hid * z_out_stride_h      + offs_m[:, None] * z_out_stride_m      + offs_n[None, :] * z_out_stride_n
            tl.store(dp_raw_out_ptr, dp_raw, mask=valid_mask)
            tl.store(s_out_ptr,      s,      mask=valid_mask)
            tl.store(z_out_ptr,      z_penalty, mask=valid_mask)

            dp_adj_out_ptr = DP_ADJ_OUT + bid * dp_adj_out_stride_b + hid * dp_adj_out_stride_h + offs_m[:, None] * dp_adj_out_stride_m + offs_n[None, :] * dp_adj_out_stride_n
            sig_prime_out_ptr = SIG_PRIME_OUT + bid * sig_prime_out_stride_b + hid * sig_prime_out_stride_h + offs_m[:, None] * sig_prime_out_stride_m + offs_n[None, :] * sig_prime_out_stride_n
            c_block_out_ptr = C_BLOCK_OUT + bid * c_block_out_stride_b + hid * c_block_out_stride_h + offs_m[:, None] * c_block_out_stride_m + offs_n[None, :] * c_block_out_stride_n
            sig_f32_out_ptr = SIG_F32_OUT + bid * sig_f32_out_stride_b + hid * sig_f32_out_stride_h + offs_m[:, None] * sig_f32_out_stride_m + offs_n[None, :] * sig_f32_out_stride_n
            tl.store(dp_adj_out_ptr, dp_adj, mask=valid_mask)
            tl.store(sig_prime_out_ptr, sigma_prime, mask=valid_mask)
            tl.store(c_block_out_ptr, c_block, mask=valid_mask)
            tl.store(sig_f32_out_ptr, sig_f32, mask=valid_mask)

        dk_contrib = tl.dot(tl.trans(dp_raw * sm_scale), q_blk.to(tl.float32))
        dk_ptrs = DK + bid * dk_stride_b + hid * dk_stride_h + offs_n[:, None] * dk_stride_n + offs_d[None, :] * dk_stride_k
        tl.atomic_add(dk_ptrs, dk_contrib.to(DK.dtype.element_ty), mask=(offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim))

        dq_acc += tl.dot(dp_raw * sm_scale, tl.trans(k_blk.to(tl.float32)))

    dq_ptrs = DQ + bid * dq_stride_b + hid * dq_stride_h + offs_m[:, None] * dq_stride_m + offs_d[None, :] * dq_stride_k
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))