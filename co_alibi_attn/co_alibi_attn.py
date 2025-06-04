import torch
import triton # type: ignore[import-unresolved]
import triton.language as tl # type: ignore[import-unresolved]
import os
import numpy as _np

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
        
        lse = torch.empty((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)
        row_sig_total = torch.empty((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)

        causal_mask_value_fwd = -torch.finfo(torch.float32).max 

        def grid(meta):
            return (triton.cdiv(seq_len_q, meta["BLOCK_M"]), batch_size * num_heads)

        _co_alibi_fwd_kernel[grid](
            Q=q, K=k, V=v, sm_scale=sm_scale, causal_mask_value=causal_mask_value_fwd,
            LSE_out=lse, RowSigTotal_out=row_sig_total,
            Out=o,
            q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_m=q.stride(2), q_stride_k=q.stride(3),
            k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_n=k.stride(2), k_stride_k=k.stride(3),
            v_stride_b=v.stride(0), v_stride_h=v.stride(1), v_stride_n=v.stride(2), v_stride_d=v.stride(3),
            out_stride_b=o.stride(0), out_stride_h=o.stride(1), out_stride_m=o.stride(2), out_stride_d=o.stride(3),
            lse_out_stride_b=lse.stride(0), lse_out_stride_h=lse.stride(1), lse_out_stride_m=lse.stride(2),
            row_sig_out_stride_b=row_sig_total.stride(0), row_sig_out_stride_h=row_sig_total.stride(1), row_sig_out_stride_m=row_sig_total.stride(2),
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
        
        ctx.save_for_backward(q, k, v, o, lse, row_sig_total)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.head_dim = head_dim
        ctx.causal_mask_value = causal_mask_value_fwd
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, row_sig_total = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        head_dim = ctx.head_dim
        causal_mask_value = ctx.causal_mask_value

        batch_size, num_heads, seq_len_q, _ = q.shape
        _, _, seq_len_kv, _ = k.shape

        if seq_len_kv > 2048:
            BLOCK_M_triton = 16
            BLOCK_N_triton = 32
        else:
            BLOCK_M_triton = 16
            BLOCK_N_triton = 32
        
        BLOCK_N_KV_triton = BLOCK_N_triton
        BLOCK_DMODEL_triton = head_dim

        dq = torch.zeros_like(q)
        grid_m = triton.cdiv(seq_len_q, BLOCK_M_triton)
        dk_partial = torch.empty(
            (batch_size, num_heads, grid_m, seq_len_kv, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        dv_partial = torch.empty_like(dk_partial)

        dk_partial.zero_()
        dv_partial.zero_()

        def grid(meta):
            return (triton.cdiv(seq_len_q, meta["BLOCK_M"]), batch_size * num_heads)

        _co_alibi_bwd_kernel[grid](
            Q=q, K=k, V=v, sm_scale=sm_scale, causal_mask_value=causal_mask_value, 
            LSE_in=lse, RowSigTotal_in=row_sig_total,
            DO=do,
            O=o,
            DQ=dq, DK=dk_partial, DV=dv_partial,
            q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_m=q.stride(2), q_stride_k=q.stride(3),
            k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_n=k.stride(2), k_stride_k=k.stride(3),
            v_stride_b=v.stride(0), v_stride_h=v.stride(1), v_stride_n=v.stride(2), v_stride_d=v.stride(3),
            lse_in_stride_b=lse.stride(0), lse_in_stride_h=lse.stride(1), lse_in_stride_m=lse.stride(2),
            row_sig_in_stride_b=row_sig_total.stride(0), row_sig_in_stride_h=row_sig_total.stride(1), row_sig_in_stride_m=row_sig_total.stride(2),
            do_stride_b=do.stride(0), do_stride_h=do.stride(1), do_stride_m=do.stride(2), do_stride_d=do.stride(3),
            dq_stride_b=dq.stride(0), dq_stride_h=dq.stride(1), dq_stride_m=dq.stride(2), dq_stride_k=dq.stride(3),
            dk_stride_b=dk_partial.stride(0), dk_stride_h=dk_partial.stride(1), dk_stride_s=dk_partial.stride(2), dk_stride_n=dk_partial.stride(3), dk_stride_k=dk_partial.stride(4),
            dv_stride_b=dv_partial.stride(0), dv_stride_h=dv_partial.stride(1), dv_stride_s=dv_partial.stride(2), dv_stride_n=dv_partial.stride(3), dv_stride_d=dv_partial.stride(4),
            o_stride_b=o.stride(0), o_stride_h=o.stride(1), o_stride_m=o.stride(2), o_stride_d=o.stride(3),
            batch_size=batch_size, num_heads=num_heads, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, head_dim=head_dim,
            HAS_CAUSAL_MASK=causal, 
            BLOCK_DMODEL=BLOCK_DMODEL_triton,
        )

        if os.getenv('COALIBI_VERBOSE', '0') == '1':
            try:
                best_cfg_bwd = _co_alibi_bwd_kernel.get_best_config()
                print(f"[Co-ALIBI] BWD kernel best config: {best_cfg_bwd.kwargs}, num_warps={best_cfg_bwd.num_warps}, num_stages={best_cfg_bwd.num_stages}")
            except Exception:
                pass

        dk = dk_partial.sum(dim=2)
        dv = dv_partial.sum(dim=2)

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None

def co_alibi_attention(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1]**0.5)
    q,k,v = q.contiguous(), k.contiguous(), v.contiguous()
    return CoALIBIAttention.apply(q, k, v, causal, sm_scale)