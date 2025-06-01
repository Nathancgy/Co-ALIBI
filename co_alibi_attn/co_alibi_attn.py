import torch
import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]
import os

from co_alibi_fwd_kernel import _co_alibi_fwd_kernel
from co_alibi_bwd_kernel import _co_alibi_bwd_kernel

class CoALIBIAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_kv, _ = k.shape

        assert k.shape == (batch_size, num_heads, seq_len_kv, head_dim)
        assert v.shape == (batch_size, num_heads, seq_len_kv, head_dim)
        assert q.dtype == k.dtype == v.dtype, "All inputs must have the same dtype"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        o = torch.empty_like(q)
        
        p_raw_for_bwd = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
        sig_p_raw = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
        z_penalty = torch.empty((batch_size, num_heads, seq_len_q, seq_len_kv), device=q.device, dtype=q.dtype)
        lse = torch.empty((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)

        causal_mask_value_fwd = -torch.finfo(torch.float32).max 

        BLOCK_M_triton = 32
        BLOCK_N_triton = 64
        
        if head_dim <= 16: BLOCK_D_triton = 16
        elif head_dim <= 32: BLOCK_D_triton = 32
        elif head_dim <= 64: BLOCK_D_triton = 64
        elif head_dim <= 128: BLOCK_D_triton = 128
        else:
            BLOCK_D_triton = 128
        if head_dim in [16, 32, 64, 128]:
            BLOCK_D_triton = head_dim

        kv_blocks = (seq_len_kv + BLOCK_N_triton - 1) // BLOCK_N_triton
        grid = (triton.cdiv(seq_len_q, BLOCK_M_triton), batch_size * num_heads)
        
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
            BLOCK_M=BLOCK_M_triton, BLOCK_N=BLOCK_N_triton, BLOCK_D=BLOCK_D_triton, KV_BLOCKS=kv_blocks
        )
        
        ctx.save_for_backward(q, k, v, o, p_raw_for_bwd, sig_p_raw, z_penalty, lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.head_dim = head_dim
        ctx.causal_mask_value = causal_mask_value_fwd 
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, p_raw, sig_p_raw, z_penalty, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        head_dim = ctx.head_dim
        causal_mask_value = ctx.causal_mask_value

        batch_size, num_heads, seq_len_q, _ = q.shape
        _, _, seq_len_kv, _ = k.shape

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        debug_mode = os.getenv("COALIBI_DEBUG", "0") == "1"

        BLOCK_M_triton = 32
        BLOCK_N_triton = 64
        
        if head_dim <= 16: BLOCK_D_triton = 16
        elif head_dim <= 32: BLOCK_D_triton = 32
        elif head_dim <= 64: BLOCK_D_triton = 64
        elif head_dim <= 128: BLOCK_D_triton = 128
        else:
            BLOCK_D_triton = 128
        if head_dim in [16, 32, 64, 128]:
            BLOCK_D_triton = head_dim

        # Aliases for backward kernel expected names
        BLOCK_N_KV_triton = BLOCK_N_triton
        BLOCK_DMODEL_triton = BLOCK_D_triton
        N_CTX_KV_triton = triton.next_power_of_2(seq_len_kv)

        grid = (triton.cdiv(seq_len_q, BLOCK_M_triton), batch_size * num_heads)
        num_warps = 4
        if BLOCK_N_KV_triton >= 64 and BLOCK_DMODEL_triton >= 64:
            num_warps = 8

        # Debug tensors ------------------------------------------------------
        if debug_mode:
            dp_raw_dbg = torch.empty_like(p_raw, dtype=torch.float32)
            s_dbg      = torch.empty_like(p_raw, dtype=torch.float32)
        else:
            # Dummy 1-element tensors to satisfy kernel signature
            dp_raw_dbg = torch.empty(1, device=q.device, dtype=torch.float32)
            s_dbg      = torch.empty(1, device=q.device, dtype=torch.float32)

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
            dp_raw_out_stride_b=dp_raw_dbg.stride(0) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_h=dp_raw_dbg.stride(1) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_m=dp_raw_dbg.stride(2) if dp_raw_dbg.dim()==4 else 0,
            dp_raw_out_stride_n=dp_raw_dbg.stride(3) if dp_raw_dbg.dim()==4 else 0,
            s_out_stride_b=s_dbg.stride(0) if s_dbg.dim()==4 else 0,
            s_out_stride_h=s_dbg.stride(1) if s_dbg.dim()==4 else 0,
            s_out_stride_m=s_dbg.stride(2) if s_dbg.dim()==4 else 0,
            s_out_stride_n=s_dbg.stride(3) if s_dbg.dim()==4 else 0,
        )

        # Expose debug tensors ---------------------------------------------
        global debug_dp_raw, debug_s
        if debug_mode:
            debug_dp_raw = dp_raw_dbg
            debug_s = s_dbg
        else:
            debug_dp_raw = None
            debug_s = None

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None

def co_alibi_attention(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1]**0.5)
    q,k,v = q.contiguous(), k.contiguous(), v.contiguous()
    return CoALIBIAttention.apply(q, k, v, causal, sm_scale) 