import math
import os
import sys
import torch
import triton # type: ignore[import-unresolved]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_alibi_attn import co_alibi_attention

def benchmark_fwd_bwd_flops(
    B: int = 1,
    H: int = 16,
    seq_lens: set[int] | list[int] = (4096,),
    D: int = 128,
    q_dtype: torch.dtype = torch.float32,
    k_dtype: torch.dtype = torch.float32,
    v_dtype: torch.dtype = torch.float16,
):
    device = "cuda"
    torch.manual_seed(0)

    NUM_WARMUP = 50
    NUM_REPS = 100

    dtype_str = (
        f"q={str(q_dtype).split('.')[-1]}, "
        f"k={str(k_dtype).split('.')[-1]}, "
        f"v={str(v_dtype).split('.')[-1]}"
    )

    print(
        f"Benchmarking Co-ALIBI with: B={B}, H={H}, D={D}, {dtype_str}, "
        f"Warmup={NUM_WARMUP}, Reps={NUM_REPS}"
    )

    for S in seq_lens:
        print("\n" + "-" * 5 + f" Sequence Length (S) = {S} " + "-" * 5)

        q_val = torch.randn(B, H, S, D, device=device, dtype=q_dtype)
        k_val = torch.randn_like(q_val, dtype=k_dtype)
        v_val = torch.randn_like(q_val, dtype=v_dtype)
        dout_val = torch.randn_like(q_val)

        sm_scale = 1.0 / math.sqrt(D)

        q_f, k_f, v_f = q_val.clone(), k_val.clone(), v_val.clone()
        for _ in range(NUM_WARMUP):
            _ = co_alibi_attention(q_f, k_f, v_f, causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()

        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(NUM_REPS):
            _ = co_alibi_attention(q_f, k_f, v_f, causal=True, sm_scale=sm_scale)
        end_event.record()
        torch.cuda.synchronize()
        ms_fwd = start_event.elapsed_time(end_event) / NUM_REPS

        q_b = q_val.clone().requires_grad_(True)
        k_b = k_val.clone().requires_grad_(True)
        v_b = v_val.clone().requires_grad_(True)

        for _ in range(NUM_WARMUP):
            if q_b.grad is not None: q_b.grad.zero_()
            if k_b.grad is not None: k_b.grad.zero_()
            if v_b.grad is not None: v_b.grad.zero_()
            out = co_alibi_attention(q_b, k_b, v_b, causal=True, sm_scale=sm_scale)
            out.backward(dout_val, retain_graph=True)

        if q_b.grad is not None: q_b.grad.zero_()
        if k_b.grad is not None: k_b.grad.zero_()
        if v_b.grad is not None: v_b.grad.zero_()

        out_bwd = co_alibi_attention(q_b, k_b, v_b, causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(NUM_REPS):
            if q_b.grad is not None: q_b.grad.zero_()
            if k_b.grad is not None: k_b.grad.zero_()
            if v_b.grad is not None: v_b.grad.zero_()
            out_bwd.backward(dout_val, retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        ms_bwd = start_event.elapsed_time(end_event) / NUM_REPS

        SKV = S
        flops_fwd = B * H * S * SKV * (4 * D)
        flops_bwd = 2 * flops_fwd

        tflops_fwd = flops_fwd / (ms_fwd * 1e-3) / 1e12
        tflops_bwd = flops_bwd / (ms_bwd * 1e-3) / 1e12

        line = "=" * 80
        print(line)
        print(
            f"Config: B={B}, H={H}, S={S}, D={D}, "
            f"q_dtype={q_dtype}, k_dtype={k_dtype}, v_dtype={v_dtype}"
        )
        print(f"Forward FLOPs: {flops_fwd/1e12:.2f} TF, Backward FLOPs: {flops_bwd/1e12:.2f} TF")
        print(line)
        print(f"{'Operation':<10} {'Latency (ms)':>15}   {'TFLOP/s':>9}")
        print("-" * 80)
        print(f"{'Forward':<10} {ms_fwd:15.3f}   {tflops_fwd:9.2f}")
        print(f"{'Backward':<10} {ms_bwd:15.3f}   {tflops_bwd:9.2f}")
        print(line)

if __name__ == "__main__":
    # Example run: fp32 Q/K, fp16 V
    benchmark_fwd_bwd_flops() 