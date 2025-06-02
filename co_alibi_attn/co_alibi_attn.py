import torch
import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]
import os
import numpy as _np

from co_alibi_fwd_kernel import _co_alibi_fwd_kernel
from co_alibi_fwd_kernel_simplified import _co_alibi_fwd_kernel_simplified
from co_alibi_bwd_kernel import _co_alibi_bwd_kernel

class CoALIBIAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, use_simplified_kernel=False):
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_kv, _ = k.shape

        assert k.shape == (batch_size, num_heads, seq_len_kv, head_dim)
        assert v.shape == (batch_size, num_heads, seq_len_kv, head_dim)
        assert q.dtype == k.dtype == v.dtype, "All inputs must have the same dtype"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        o = torch.empty_like(q)
        
        if use_simplified_kernel:
            p_raw_for_bwd = torch.empty(1, device=q.device, dtype=q.dtype)
            sig_p_raw    = torch.empty(1, device=q.device, dtype=q.dtype)
            z_penalty    = torch.empty(1, device=q.device, dtype=q.dtype)
        else:
            p_raw_for_bwd = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
            sig_p_raw    = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
            z_penalty    = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
        lse = torch.empty((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)

        causal_mask_value_fwd = -torch.finfo(torch.float32).max 

        if use_simplified_kernel:
            def grid(meta):
                return (triton.cdiv(seq_len_q, meta['BLOCK_M']), batch_size * num_heads)

            _co_alibi_fwd_kernel_simplified[grid](
                Q=q, K=k, V=v, sm_scale=sm_scale, causal_mask_value=causal_mask_value_fwd,
                P_raw_out=p_raw_for_bwd, Sig_P_raw_out=sig_p_raw, Z_penalty_out=z_penalty, LSE_out=lse,
                Out=o,
                q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_m=q.stride(2), q_stride_k=q.stride(3),
                k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_n=k.stride(2), k_stride_k=k.stride(3),
                v_stride_b=v.stride(0), v_stride_h=v.stride(1), v_stride_n=v.stride(2), v_stride_d=v.stride(3),
                out_stride_b=o.stride(0), out_stride_h=o.stride(1), out_stride_m=o.stride(2), out_stride_d=o.stride(3),
                p_raw_out_stride_b=p_raw_for_bwd.stride(0) if p_raw_for_bwd.dim()>1 else 0,
                p_raw_out_stride_h=p_raw_for_bwd.stride(1) if p_raw_for_bwd.dim()>1 else 0,
                p_raw_out_stride_m=p_raw_for_bwd.stride(2) if p_raw_for_bwd.dim()>1 else 0,
                p_raw_out_stride_n=p_raw_for_bwd.stride(3) if p_raw_for_bwd.dim()>1 else 0,
                sig_p_raw_out_stride_b=sig_p_raw.stride(0) if sig_p_raw.dim()>1 else 0,
                sig_p_raw_out_stride_h=sig_p_raw.stride(1) if sig_p_raw.dim()>1 else 0,
                sig_p_raw_out_stride_m=sig_p_raw.stride(2) if sig_p_raw.dim()>1 else 0,
                sig_p_raw_out_stride_n=sig_p_raw.stride(3) if sig_p_raw.dim()>1 else 0,
                z_penalty_out_stride_b=z_penalty.stride(0) if z_penalty.dim()>1 else 0,
                z_penalty_out_stride_h=z_penalty.stride(1) if z_penalty.dim()>1 else 0,
                z_penalty_out_stride_m=z_penalty.stride(2) if z_penalty.dim()>1 else 0,
                z_penalty_out_stride_n=z_penalty.stride(3) if z_penalty.dim()>1 else 0,
                lse_out_stride_b=lse.stride(0), lse_out_stride_h=lse.stride(1), lse_out_stride_m=lse.stride(2),
                batch_size=batch_size, num_heads=num_heads, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, head_dim=head_dim,
                HAS_CAUSAL_MASK=causal,
                BLOCK_D=head_dim
            )
            if os.getenv('COALIBI_VERBOSE', '0') == '1':
                try:
                    best_cfg = _co_alibi_fwd_kernel_simplified.get_best_config()
                    print(f"[Co-ALIBI] Simplified kernel best config: {best_cfg.kwargs}, num_warps={best_cfg.num_warps}, num_stages={best_cfg.num_stages}")
                except Exception as _e:
                    pass
        else:
            def grid(meta):
                return (triton.cdiv(seq_len_q, meta["BLOCK_M"]), batch_size * num_heads)

            _co_alibi_fwd_kernel[grid](
                Q=q, K=k, V=v, sm_scale=sm_scale, causal_mask_value=causal_mask_value_fwd,
                P_raw_out=p_raw_for_bwd, Sig_P_raw_out=sig_p_raw, Z_penalty_out=z_penalty, LSE_out=lse,
                Out=o,
                q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_m=q.stride(2), q_stride_k=q.stride(3),
                k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_n=k.stride(2), k_stride_k=k.stride(3),
                v_stride_b=v.stride(0), v_stride_h=v.stride(1), v_stride_n=v.stride(2), v_stride_d=v.stride(3),
                out_stride_b=o.stride(0), out_stride_h=o.stride(1), out_stride_m=o.stride(2), out_stride_d=o.stride(3),
                p_raw_out_stride_b=p_raw_for_bwd.stride(0), p_raw_out_stride_h=p_raw_for_bwd.stride(1), p_raw_out_stride_m=p_raw_for_bwd.stride(2), p_raw_out_stride_n=p_raw_for_bwd.stride(3),
                sig_p_raw_out_stride_b=sig_p_raw.stride(0), sig_p_raw_out_stride_h=sig_p_raw.stride(1), sig_p_raw_out_stride_m=sig_p_raw.stride(2), sig_p_raw_out_stride_n=sig_p_raw.stride(3),
                z_penalty_out_stride_b=z_penalty.stride(0), z_penalty_out_stride_h=z_penalty.stride(1), z_penalty_out_stride_m=z_penalty.stride(2), z_penalty_out_stride_n=z_penalty.stride(3),
                lse_out_stride_b=lse.stride(0), lse_out_stride_h=lse.stride(1), lse_out_stride_m=lse.stride(2),
                batch_size=batch_size, num_heads=num_heads, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, head_dim=head_dim,
                HAS_CAUSAL_MASK=causal,
                BLOCK_D=head_dim
            )
            if os.getenv('COALIBI_VERBOSE', '0') == '1':
                try:
                    best_cfg = _co_alibi_fwd_kernel.get_best_config()
                    print(f"[Co-ALIBI] Original kernel best config: {best_cfg.kwargs}, num_warps={best_cfg.num_warps}, num_stages={best_cfg.num_stages}")
                except Exception as _e:
                    pass
        
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.head_dim = head_dim
        ctx.causal_mask_value = causal_mask_value_fwd
        ctx.use_simplified_kernel = use_simplified_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        head_dim = ctx.head_dim
        causal_mask_value = ctx.causal_mask_value
        use_simplified_kernel = ctx.use_simplified_kernel

        batch_size, num_heads, seq_len_q, _ = q.shape
        _, _, seq_len_kv, _ = k.shape

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        debug_mode = 1

        if seq_len_kv > 2048:
            BLOCK_M_triton = 16
            BLOCK_N_triton = 32
        else:
            BLOCK_M_triton = 16
            BLOCK_N_triton = 32
        
        if head_dim <= 16: BLOCK_D_triton = 16
        elif head_dim <= 32: BLOCK_D_triton = 32
        elif head_dim <= 64: BLOCK_D_triton = 64
        elif head_dim <= 128: BLOCK_D_triton = 128
        else:
            BLOCK_D_triton = 128
        if head_dim in [16, 32, 64, 128]:
            BLOCK_D_triton = head_dim

        BLOCK_N_KV_triton = BLOCK_N_triton
        BLOCK_DMODEL_triton = BLOCK_D_triton

        grid = (triton.cdiv(seq_len_q, BLOCK_M_triton), batch_size * num_heads)
        num_warps = 4
        if BLOCK_N_KV_triton >= 64 and BLOCK_DMODEL_triton >= 64:
            num_warps = 8

        if debug_mode:
            dp_raw_dbg = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            s_dbg      = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            s2_dbg     = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            z_dbg      = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            z2_dbg     = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            sig2_dbg   = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            p_raw_pass2_dbg = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            p_raw_pass3_dbg = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
        else:
            dp_raw_dbg = torch.empty(1, device=q.device, dtype=torch.float32)
            s_dbg      = torch.empty(1, device=q.device, dtype=torch.float32)
            s2_dbg     = torch.empty(1, device=q.device, dtype=torch.float32)
            z_dbg      = torch.empty(1, device=q.device, dtype=torch.float32)
            z2_dbg     = torch.empty(1, device=q.device, dtype=torch.float32)
            sig2_dbg   = torch.empty(1, device=q.device, dtype=torch.float32)
            p_raw_pass2_dbg = torch.empty(1, device=q.device, dtype=torch.float32)
            p_raw_pass3_dbg = torch.empty(1, device=q.device, dtype=torch.float32)

        # Additional debug tensors
        if debug_mode:
            dp_adj_dbg       = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            sig_prime_dbg    = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            c_block_dbg      = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
            sig_f32_dbg      = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=torch.float32)
        else:
            dp_adj_dbg    = torch.empty(1, device=q.device, dtype=torch.float32)
            sig_prime_dbg = torch.empty(1, device=q.device, dtype=torch.float32)
            c_block_dbg   = torch.empty(1, device=q.device, dtype=torch.float32)
            sig_f32_dbg   = torch.empty(1, device=q.device, dtype=torch.float32)

        _co_alibi_bwd_kernel[grid](
            Q=q, K=k, V=v, sm_scale=sm_scale, causal_mask_value=causal_mask_value, 
            LSE_in=lse,
            DO=do,
            O=o,
            DQ=dq, DK=dk, DV=dv,
            q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_m=q.stride(2), q_stride_k=q.stride(3),
            k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_n=k.stride(2), k_stride_k=k.stride(3),
            v_stride_b=v.stride(0), v_stride_h=v.stride(1), v_stride_n=v.stride(2), v_stride_d=v.stride(3),
            lse_in_stride_b=lse.stride(0), lse_in_stride_h=lse.stride(1), lse_in_stride_m=lse.stride(2),
            do_stride_b=do.stride(0), do_stride_h=do.stride(1), do_stride_m=do.stride(2), do_stride_d=do.stride(3),
            dq_stride_b=dq.stride(0), dq_stride_h=dq.stride(1), dq_stride_m=dq.stride(2), dq_stride_k=dq.stride(3),
            dk_stride_b=dk.stride(0), dk_stride_h=dk.stride(1), dk_stride_n=dk.stride(2), dk_stride_k=dk.stride(3),
            dv_stride_b=dv.stride(0), dv_stride_h=dv.stride(1), dv_stride_n=dv.stride(2), dv_stride_d=dv.stride(3),
            o_stride_b=o.stride(0), o_stride_h=o.stride(1), o_stride_m=o.stride(2), o_stride_d=o.stride(3),
            batch_size=batch_size, num_heads=num_heads, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, head_dim=head_dim,
            HAS_CAUSAL_MASK=causal, 
            BLOCK_M=BLOCK_M_triton, BLOCK_N_KV=BLOCK_N_KV_triton, BLOCK_DMODEL=BLOCK_DMODEL_triton,
            NUM_WARPS=num_warps,
            DEBUG=debug_mode,
            DP_RAW_OUT=dp_raw_dbg, S_OUT=s_dbg,
            S2_OUT=s2_dbg, Z2_OUT=z2_dbg, Z_OUT=z_dbg,
            DP_ADJ_OUT=dp_adj_dbg, SIG_PRIME_OUT=sig_prime_dbg, C_BLOCK_OUT=c_block_dbg, SIG_F32_OUT=sig_f32_dbg,
            SIG2_OUT=sig2_dbg,
            P_RAW_PASS2_OUT=p_raw_pass2_dbg, P_RAW_PASS3_OUT=p_raw_pass3_dbg,
            dp_raw_out_stride_b=dp_raw_dbg.stride(0) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_h=dp_raw_dbg.stride(1) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_m=dp_raw_dbg.stride(2) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_n=dp_raw_dbg.stride(3) if dp_raw_dbg.dim()==4 else 0,
            s_out_stride_b=s_dbg.stride(0) if s_dbg.dim()==4 else 0,
            s_out_stride_h=s_dbg.stride(1) if s_dbg.dim()==4 else 0,
            s_out_stride_m=s_dbg.stride(2) if s_dbg.dim()==4 else 0,
            s_out_stride_n=s_dbg.stride(3) if s_dbg.dim()==4 else 0,
            s2_out_stride_b=s2_dbg.stride(0) if s2_dbg.dim()==4 else 0,
            s2_out_stride_h=s2_dbg.stride(1) if s2_dbg.dim()==4 else 0,
            s2_out_stride_m=s2_dbg.stride(2) if s2_dbg.dim()==4 else 0,
            s2_out_stride_n=s2_dbg.stride(3) if s2_dbg.dim()==4 else 0,
            z2_out_stride_b=z2_dbg.stride(0) if z2_dbg.dim()==4 else 0,
            z2_out_stride_h=z2_dbg.stride(1) if z2_dbg.dim()==4 else 0,
            z2_out_stride_m=z2_dbg.stride(2) if z2_dbg.dim()==4 else 0,
            z2_out_stride_n=z2_dbg.stride(3) if z2_dbg.dim()==4 else 0,
            z_out_stride_b=z_dbg.stride(0) if z_dbg.dim()==4 else 0,
            z_out_stride_h=z_dbg.stride(1) if z_dbg.dim()==4 else 0,
            z_out_stride_m=z_dbg.stride(2) if z_dbg.dim()==4 else 0,
            z_out_stride_n=z_dbg.stride(3) if z_dbg.dim()==4 else 0,
            dp_adj_out_stride_b=dp_adj_dbg.stride(0) if dp_adj_dbg.dim()==4 else 0,
            dp_adj_out_stride_h=dp_adj_dbg.stride(1) if dp_adj_dbg.dim()==4 else 0,
            dp_adj_out_stride_m=dp_adj_dbg.stride(2) if dp_adj_dbg.dim()==4 else 0,
            dp_adj_out_stride_n=dp_adj_dbg.stride(3) if dp_adj_dbg.dim()==4 else 0,
            sig_prime_out_stride_b=sig_prime_dbg.stride(0) if sig_prime_dbg.dim()==4 else 0,
            sig_prime_out_stride_h=sig_prime_dbg.stride(1) if sig_prime_dbg.dim()==4 else 0,
            sig_prime_out_stride_m=sig_prime_dbg.stride(2) if sig_prime_dbg.dim()==4 else 0,
            sig_prime_out_stride_n=sig_prime_dbg.stride(3) if sig_prime_dbg.dim()==4 else 0,
            c_block_out_stride_b=c_block_dbg.stride(0) if c_block_dbg.dim()==4 else 0,
            c_block_out_stride_h=c_block_dbg.stride(1) if c_block_dbg.dim()==4 else 0,
            c_block_out_stride_m=c_block_dbg.stride(2) if c_block_dbg.dim()==4 else 0,
            c_block_out_stride_n=c_block_dbg.stride(3) if c_block_dbg.dim()==4 else 0,
            sig_f32_out_stride_b=sig_f32_dbg.stride(0) if sig_f32_dbg.dim()==4 else 0,
            sig_f32_out_stride_h=sig_f32_dbg.stride(1) if sig_f32_dbg.dim()==4 else 0,
            sig_f32_out_stride_m=sig_f32_dbg.stride(2) if sig_f32_dbg.dim()==4 else 0,
            sig_f32_out_stride_n=sig_f32_dbg.stride(3) if sig_f32_dbg.dim()==4 else 0,
            sig2_out_stride_b=sig2_dbg.stride(0) if sig2_dbg.dim()==4 else 0,
            sig2_out_stride_h=sig2_dbg.stride(1) if sig2_dbg.dim()==4 else 0,
            sig2_out_stride_m=sig2_dbg.stride(2) if sig2_dbg.dim()==4 else 0,
            sig2_out_stride_n=sig2_dbg.stride(3) if sig2_dbg.dim()==4 else 0,
            p_raw_pass2_out_stride_b=p_raw_pass2_dbg.stride(0) if p_raw_pass2_dbg.dim()==4 else 0,
            p_raw_pass2_out_stride_h=p_raw_pass2_dbg.stride(1) if p_raw_pass2_dbg.dim()==4 else 0,
            p_raw_pass2_out_stride_m=p_raw_pass2_dbg.stride(2) if p_raw_pass2_dbg.dim()==4 else 0,
            p_raw_pass2_out_stride_n=p_raw_pass2_dbg.stride(3) if p_raw_pass2_dbg.dim()==4 else 0,
            p_raw_pass3_out_stride_b=p_raw_pass3_dbg.stride(0) if p_raw_pass3_dbg.dim()==4 else 0,
            p_raw_pass3_out_stride_h=p_raw_pass3_dbg.stride(1) if p_raw_pass3_dbg.dim()==4 else 0,
            p_raw_pass3_out_stride_m=p_raw_pass3_dbg.stride(2) if p_raw_pass3_dbg.dim()==4 else 0,
            p_raw_pass3_out_stride_n=p_raw_pass3_dbg.stride(3) if p_raw_pass3_dbg.dim()==4 else 0,
        )

        global debug_dp_raw, debug_s, debug_s2, debug_z, debug_z2, debug_sig2, debug_p_raw_pass2, debug_p_raw_pass3
        if debug_mode:
            debug_dp_raw = dp_raw_dbg
            debug_s = s_dbg
            debug_s2 = s2_dbg
            debug_z2 = z2_dbg
            debug_z = z_dbg
            debug_dp_adj = dp_adj_dbg
            debug_sigma_prime = sig_prime_dbg
            debug_c_block = c_block_dbg
            debug_sig_f32 = sig_f32_dbg
            debug_sig2 = sig2_dbg
            debug_p_raw_pass2 = p_raw_pass2_dbg
            debug_p_raw_pass3 = p_raw_pass3_dbg

            # --- Detailed debug analysis for tile position (0,0,1,0) ---
            try:
                b_idx, h_idx, q_idx, k_idx = 0, 0, 1, 1
                tri_dp_raw_val   = debug_dp_raw[b_idx, h_idx, q_idx, k_idx].item()
                tri_dp_adj_val   = debug_dp_adj[b_idx, h_idx, q_idx, k_idx].item()
                tri_sig_prime_val = debug_sigma_prime[b_idx, h_idx, q_idx, k_idx].item()
                tri_c_block_val  = debug_c_block[b_idx, h_idx, q_idx, k_idx].item()
                tri_sig_f32_val  = debug_sig_f32[b_idx, h_idx, q_idx, k_idx].item()
                tri_p_raw_pass2_val = debug_p_raw_pass2[b_idx, h_idx, q_idx, k_idx].item()
                tri_p_raw_pass3_val = debug_p_raw_pass3[b_idx, h_idx, q_idx, k_idx].item()

                tri_s_val = debug_s[b_idx, h_idx, q_idx, k_idx].item()
                tri_s2_val = debug_s2[b_idx, h_idx, q_idx, k_idx].item()
                tri_z2_val = debug_z2[b_idx, h_idx, q_idx, k_idx].item()
                tri_z_val  = debug_z[b_idx, h_idx, q_idx, k_idx].item()
                tri_sig2_val = debug_sig2[b_idx, h_idx, q_idx, k_idx].item()

                # Compute reference values with PyTorch implementation
                q0 = q[b_idx, h_idx]  # (N_q, D)
                k0 = k[b_idx, h_idx]  # (N_k, D)
                v0 = v[b_idx, h_idx]
                do0 = do[b_idx, h_idx]

                p_raw_full = torch.matmul(q0, k0.T) * sm_scale
                if causal:
                    causal_mask_bool = torch.triu(torch.ones_like(p_raw_full, dtype=torch.bool), diagonal=1)
                    p_raw_for_sig_full = p_raw_full.masked_fill(causal_mask_bool, -float('inf'))
                else:
                    p_raw_for_sig_full = p_raw_full

                sig_full = torch.sigmoid(p_raw_for_sig_full)
                if causal:
                    sig_full = sig_full.masked_fill(causal_mask_bool, 0.0)

                row_sig_total_full = sig_full.sum(dim=-1, keepdim=True)

                prefix_sig_full = torch.cumsum(sig_full, dim=-1)
                z_penalty_full = row_sig_total_full - prefix_sig_full + sig_full

                p_raw_masked = p_raw_full.clone()
                if causal:
                    p_raw_masked = p_raw_masked.masked_fill(causal_mask_bool, -torch.finfo(p_raw_full.dtype).max)

                p_adj_full = p_raw_masked - z_penalty_full

                lse_row_full = lse[b_idx, h_idx].unsqueeze(-1)  # (N_q, 1)

                s_full = torch.exp(p_adj_full - lse_row_full)
                if causal:
                    s_full = s_full.masked_fill(causal_mask_bool, 0.0)

                # ds_full and D_t row sum
                ds_full = torch.matmul(do0, v0.T)
                row_ds_s_full = (s_full * ds_full).sum(dim=-1, keepdim=True)

                dp_adj_full = s_full * (ds_full - row_ds_s_full)

                sigma_prime_full = sig_full * (1.0 - sig_full)
                sigma_prime_full = torch.clamp_max(sigma_prime_full, 0.25)

                prefix_dp_full = torch.cumsum(dp_adj_full, dim=-1)
                dp_raw_full = dp_adj_full - sigma_prime_full * prefix_dp_full

                ref_dp_raw_val   = dp_raw_full[q_idx, k_idx].item()
                ref_dp_adj_val   = dp_adj_full[q_idx, k_idx].item()
                ref_sig_prime_val = sigma_prime_full[q_idx, k_idx].item()
                ref_c_block_val  = prefix_dp_full[q_idx, k_idx].item()
                ref_sig_f32_val  = sig_full[q_idx, k_idx].item()
                ref_s_val        = s_full[q_idx, k_idx].item()
                ref_z_val        = z_penalty_full[q_idx, k_idx].item()
                ref_p_raw_val_for_debug = p_raw_full[q_idx, k_idx].item() # Assuming p_raw_full is the correct reference

                print("[Co-ALIBI DEBUG]  (b,h,q,k)=(0,0,1,0):")
                print(f"    Triton  s (pass2): {tri_s2_val:+.6e}")
                print(f"    Triton  s (pass3): {tri_s_val:+.6e}")
                print(f"    Reference s     : {ref_s_val:+.6e}")
                print(f"    |diff| (pass2)  : {abs(tri_s2_val-ref_s_val):.3e}")
                print(f"    |diff|          : {abs(tri_s_val-ref_s_val):.3e}\n")

                print(f"    Triton  dp_raw  : {tri_dp_raw_val:+.6e}")
                print(f"    Reference dp_raw: {ref_dp_raw_val:+.6e}")
                print(f"    |diff|          : {abs(tri_dp_raw_val-ref_dp_raw_val):.3e}\n")

                print(f"    Triton  p_raw (pass2): {tri_p_raw_pass2_val:+.6e}")
                print(f"    Triton  p_raw (pass3): {tri_p_raw_pass3_val:+.6e}")
                print(f"    Reference p_raw      : {ref_p_raw_val_for_debug:+.6e}")
                print(f"    |diff| p_raw (pass2): {abs(tri_p_raw_pass2_val-ref_p_raw_val_for_debug):.3e}")
                print(f"    |diff| p_raw (pass3): {abs(tri_p_raw_pass3_val-ref_p_raw_val_for_debug):.3e}\n")

                print(f"    Triton  dp_adj  : {tri_dp_adj_val:+.6e}")
                print(f"    Reference dp_adj: {ref_dp_adj_val:+.6e}")
                print(f"    |diff|          : {abs(tri_dp_adj_val-ref_dp_adj_val):.3e}\n")

                print(f"    Triton  sigma'  : {tri_sig_prime_val:+.6e}")
                print(f"    Reference sigma': {ref_sig_prime_val:+.6e}")
                print(f"    |diff|          : {abs(tri_sig_prime_val-ref_sig_prime_val):.3e}\n")

                print(f"    Triton  c_block : {tri_c_block_val:+.6e}")
                print(f"    Reference c_block: {ref_c_block_val:+.6e}")
                print(f"    |diff|          : {abs(tri_c_block_val-ref_c_block_val):.3e}\n")

                print(f"    Triton  sig     : {tri_sig_f32_val:+.6e}")
                print(f"    Triton  sig2     : {tri_sig2_val:+.6e}")
                print(f"    Reference sig   : {ref_sig_f32_val:+.6e}")
                print(f"    |diff| sig      : {abs(tri_sig_f32_val-ref_sig_f32_val):.3e}")
                print(f"    |diff| sig2     : {abs(tri_sig2_val-ref_sig_f32_val):.3e}")

                print(f"    Triton  z (pass2): {tri_z2_val:+.6e}")
                print(f"    Triton  z (pass3): {tri_z_val:+.6e}")
                print(f"    Reference z     : {ref_z_val:+.6e}")
                print(f"    |diff| z(pass2) : {abs(tri_z2_val-ref_z_val):.3e}")
                print(f"    |diff| z        : {abs(tri_z_val-ref_z_val):.3e}\n")

                # --- sm_scale and K vector debug ---
                tri_sm_scale = float(sm_scale)
                ref_sm_scale = float(sm_scale)  # should be identical
                print("    sm_scale:")
                print(f"        Triton     : {tri_sm_scale:+.6e}")
                print(f"        Reference  : {ref_sm_scale:+.6e}\n")

                k_vec_tri = k[b_idx, h_idx, k_idx][:8].cpu().numpy()
                k_vec_ref = k[b_idx, h_idx, k_idx][:8].cpu().numpy()
                print("    k[0][:8] (first 8 dims):")
                print("        Triton     : [" + ", ".join(f"{v:+.4e}" for v in k_vec_tri) + "]")
                print("        Reference  : [" + ", ".join(f"{v:+.4e}" for v in k_vec_ref) + "]")

                # --- dq comparison ---
                dq_tri_vec = dq[b_idx, h_idx, q_idx, :8].detach().cpu().to(torch.float64).numpy()
                # reference dq = sm_scale * dp_raw_full @ k0
                dq_ref_full = torch.matmul(dp_raw_full, k0) * sm_scale
                dq_ref_vec = dq_ref_full[q_idx, :8].detach().cpu().to(torch.float64).numpy()
                diff_max = float(_np.max(_np.abs(dq_tri_vec - dq_ref_vec)))

                print("\n    dq (first 8 dims):")
                print("        Triton     : [" + ", ".join(f"{v:+.4e}" for v in dq_tri_vec) + "]")
                print("        Reference  : [" + ", ".join(f"{v:+.4e}" for v in dq_ref_vec) + "]")
                print(f"        max |diff| : {diff_max:.3e}")
            except Exception as dbg_exc:
                print("[Co-ALIBI DEBUG] Detailed debug failed:", dbg_exc)
        else:
            debug_dp_raw = debug_dp_adj = debug_sigma_prime = debug_c_block = debug_sig_f32 = debug_s = debug_s2 = debug_z = debug_z2 = debug_sig2 = None
            debug_p_raw_pass2 = debug_p_raw_pass3 = None

        if use_simplified_kernel:
            pass

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None

def co_alibi_attention(q, k, v, causal=True, sm_scale=None, use_simplified_kernel=False):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1]**0.5)
    q,k,v = q.contiguous(), k.contiguous(), v.contiguous()
    return CoALIBIAttention.apply(q, k, v, causal, sm_scale, use_simplified_kernel) 