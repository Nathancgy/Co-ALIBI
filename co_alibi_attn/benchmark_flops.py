import math
import os
import sys
import torch
import triton # type: ignore[import-unresolved]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_alibi_attn import co_alibi_attention

try:
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
except ImportError as e:
    raise RuntimeError(
        "flash-attn must be installed (pip install flash-attn --no-build-isolation) "
        "to run this benchmark.") from e

def benchmark_fwd_bwd_flops(
    B: int = 1,
    H: int = 16,
    seq_lens={4096},
    D: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    device = "cuda"
    torch.manual_seed(0)

    NUM_WARMUP = 50
    NUM_REPS = 100

    print(f"Benchmarking with: B={B}, H={H}, D={D}, dtype={dtype}, Warmup={NUM_WARMUP}, Reps={NUM_REPS}")

    for S in seq_lens:
        print(f"\n----- Sequence Length (S) = {S} -----")

        q_val = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k_val = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v_val = torch.randn(B, H, S, D, device=device, dtype=dtype)
        dout_val = torch.randn(B, H, S, D, device=device, dtype=dtype)
        sm_scale = 1.0 / math.sqrt(D)

        q_flash_val = q_val.transpose(1, 2).contiguous()
        k_flash_val = k_val.transpose(1, 2).contiguous()
        v_flash_val = v_val.transpose(1, 2).contiguous()
        dout_flash_val = dout_val.transpose(1, 2).contiguous()

        q_co_f, k_co_f, v_co_f = q_val.clone(), k_val.clone(), v_val.clone()
        for _ in range(NUM_WARMUP): _ = co_alibi_attention(q_co_f, k_co_f, v_co_f, causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(NUM_REPS): _ = co_alibi_attention(q_co_f, k_co_f, v_co_f, causal=True, sm_scale=sm_scale)
        end_event.record()
        torch.cuda.synchronize()
        ms_co_alibi_fwd = start_event.elapsed_time(end_event) / NUM_REPS

        q_co_b, k_co_b, v_co_b = q_val.clone().requires_grad_(True), k_val.clone().requires_grad_(True), v_val.clone().requires_grad_(True)
        for _ in range(NUM_WARMUP):
            if q_co_b.grad is not None: q_co_b.grad.zero_()
            if k_co_b.grad is not None: k_co_b.grad.zero_()
            if v_co_b.grad is not None: v_co_b.grad.zero_()
            o_co = co_alibi_attention(q_co_b, k_co_b, v_co_b, causal=True, sm_scale=sm_scale)
            o_co.backward(dout_val, retain_graph=True)
        if q_co_b.grad is not None: q_co_b.grad.zero_()
        if k_co_b.grad is not None: k_co_b.grad.zero_()
        if v_co_b.grad is not None: v_co_b.grad.zero_()
        
        o_co_for_bwd = co_alibi_attention(q_co_b, k_co_b, v_co_b, causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(NUM_REPS):
            if q_co_b.grad is not None: q_co_b.grad.zero_()
            if k_co_b.grad is not None: k_co_b.grad.zero_()
            if v_co_b.grad is not None: v_co_b.grad.zero_()
            o_co_for_bwd.backward(dout_val, retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        ms_co_alibi_bwd = start_event.elapsed_time(end_event) / NUM_REPS

        q_fl_f, k_fl_f, v_fl_f = q_flash_val.clone(), k_flash_val.clone(), v_flash_val.clone()
        for _ in range(NUM_WARMUP): _ = flash_attn_func(q_fl_f, k_fl_f, v_fl_f, dropout_p=0.0, softmax_scale=sm_scale, causal=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(NUM_REPS): _ = flash_attn_func(q_fl_f, k_fl_f, v_fl_f, dropout_p=0.0, softmax_scale=sm_scale, causal=True)
        end_event.record()
        torch.cuda.synchronize()
        ms_flash_fwd = start_event.elapsed_time(end_event) / NUM_REPS

        q_fl_b, k_fl_b, v_fl_b = q_flash_val.clone().requires_grad_(True), k_flash_val.clone().requires_grad_(True), v_flash_val.clone().requires_grad_(True)
        for _ in range(NUM_WARMUP):
            if q_fl_b.grad is not None: q_fl_b.grad.zero_()
            if k_fl_b.grad is not None: k_fl_b.grad.zero_()
            if v_fl_b.grad is not None: v_fl_b.grad.zero_()
            o_fl = flash_attn_func(q_fl_b, k_fl_b, v_fl_b, dropout_p=0.0, softmax_scale=sm_scale, causal=True)
            o_fl.backward(dout_flash_val, retain_graph=True)
        if q_fl_b.grad is not None: q_fl_b.grad.zero_()
        if k_fl_b.grad is not None: k_fl_b.grad.zero_()
        if v_fl_b.grad is not None: v_fl_b.grad.zero_()

        o_fl_for_bwd = flash_attn_func(q_fl_b, k_fl_b, v_fl_b, dropout_p=0.0, softmax_scale=sm_scale, causal=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(NUM_REPS):
            if q_fl_b.grad is not None: q_fl_b.grad.zero_()
            if k_fl_b.grad is not None: k_fl_b.grad.zero_()
            if v_fl_b.grad is not None: v_fl_b.grad.zero_()
            o_fl_for_bwd.backward(dout_flash_val, retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        ms_flash_bwd = start_event.elapsed_time(end_event) / NUM_REPS
        
        SKV = S 
        flops_fwd = B * H * S * SKV * (4 * D)
        flops_bwd = 2 * flops_fwd

        tflops_co_alibi_fwd = flops_fwd / (ms_co_alibi_fwd * 1e-3) / 1e12
        tflops_flash_fwd    = flops_fwd / (ms_flash_fwd * 1e-3) / 1e12
        tflops_co_alibi_bwd = flops_bwd / (ms_co_alibi_bwd * 1e-3) / 1e12
        tflops_flash_bwd    = flops_bwd / (ms_flash_bwd * 1e-3) / 1e12

        line = "=" * 80
        print(line)
        print(f"Config: B={B}, H={H}, S={S}, D={D}, dtype={dtype.__str__().split('.')[-1]}")
        print(f"Forward FLOPs: {flops_fwd/1e12:.2f} TF, Backward FLOPs: {flops_bwd/1e12:.2f} TF")
        print(line)
        print(f"{'Operation':<10} {'Kernel':<25}  Latency (ms)   TFLOP/s")
        print("-" * 80)
        
        _report = lambda op, name, ms, tf: print(f"{op:<10} {name:<25}  {ms:12.3f}   {tf:7.2f}")
        
        _report("Forward", "Co-ALIBI (Triton)", ms_co_alibi_fwd, tflops_co_alibi_fwd)
        _report("Forward", "FlashAttention-2", ms_flash_fwd, tflops_flash_fwd)
        _report("Backward", "Co-ALIBI (Triton)", ms_co_alibi_bwd, tflops_co_alibi_bwd)
        _report("Backward", "FlashAttention-2", ms_flash_bwd, tflops_flash_bwd)
        print(line)

        if tflops_flash_fwd > 0 and tflops_co_alibi_fwd > 0:
            speedup_fwd = tflops_co_alibi_fwd / tflops_flash_fwd
            print(f"Forward Co-ALIBI vs FlashAttn-2 Speedup (TFLOP/s): x{speedup_fwd:.2f}")
        if tflops_flash_bwd > 0 and tflops_co_alibi_bwd > 0:
            speedup_bwd = tflops_co_alibi_bwd / tflops_flash_bwd
            print(f"Backward Co-ALIBI vs FlashAttn-2 Speedup (TFLOP/s): x{speedup_bwd:.2f}")
        print(line)

if __name__ == "__main__":
    benchmark_fwd_bwd_flops() 