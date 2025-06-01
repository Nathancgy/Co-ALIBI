import math
import os
import sys
import torch
import triton # type: ignore[import-unresolved]

# Add project root so that we can import co_alibi_attention
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from co_alibi_attn import co_alibi_attention

try:
    # FlashAttention-2 public interface (installed via pip install flash-attn)
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "flash-attn must be installed (pip install flash-attn --no-build-isolation) "
        "to run this benchmark.") from e


def benchmark_fwd_flops(
    B: int = 1,
    H: int = 16,
    seq_lens={4096},
    D: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """Benchmark forward-pass TFLOPs for Co-ALIBI vs FlashAttention-2.

    Args:
        B: batch size.
        H: number of heads.
        seq_lens: iterable of sequence lengths to test.
        D: head dimension.
        dtype: input dtype (fp16 or bf16 recommended).
    """
    device = "cuda"
    torch.manual_seed(0)

    for S in seq_lens:
        # Q, K, V follow (B, H, S, D) layout for our Triton kernel
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        sm_scale = 1.0 / math.sqrt(D)

        # -------------------------------------------------------------
        # Build minimal callables for do_bench
        # -------------------------------------------------------------

        def co_alibi_fwd_original():
            co_alibi_attention(q, k, v, causal=True, sm_scale=sm_scale, use_simplified_kernel=False)

        def co_alibi_fwd_simplified():
            co_alibi_attention(q, k, v, causal=True, sm_scale=sm_scale, use_simplified_kernel=True)

        # FlashAttention-2 expects (B, S, H, D) layout.
        q_flash = q.transpose(1, 2).contiguous()  # (B, S, H, D)
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()

        def flash_attn_fwd():
            flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=True,
            )

        # Warm-up & benchmark
        ms_co_alibi_original = triton.testing.do_bench(co_alibi_fwd_original, rep=100, warmup=250)
        ms_co_alibi_simplified = triton.testing.do_bench(co_alibi_fwd_simplified, rep=100, warmup=250)
        ms_flash    = triton.testing.do_bench(flash_attn_fwd, rep=100, warmup=250)

        # FLOPs under 4·D convention
        SKV = S
        flops_per_qk = 4 * D
        total_flops = B * H * S * SKV * flops_per_qk

        tflops_co_alibi_original = total_flops / (ms_co_alibi_original * 1e-3) / 1e12
        tflops_co_alibi_simplified = total_flops / (ms_co_alibi_simplified * 1e-3) / 1e12
        tflops_flash    = total_flops / (ms_flash * 1e-3) / 1e12

        line = "=" * 70  # Adjusted line width for better formatting
        print(line)
        print(f"Config: B={B}, H={H}, S={S}, D={D}, dtype={dtype.__str__().split('.')[-1]}  (total_flops={total_flops/1e12:.2f} TF)")
        print(line)

        def _report(name: str, latency_ms: float, tflops: float):
            print(f"{name:<30}  latency = {latency_ms:7.3f} ms   |  throughput = {tflops:6.2f} TFLOP/s")

        _report("Co-ALIBI (Triton Original)", ms_co_alibi_original, tflops_co_alibi_original)
        _report("Co-ALIBI (Triton Simplified)", ms_co_alibi_simplified, tflops_co_alibi_simplified)
        _report("FlashAttention-2", ms_flash, tflops_flash)
        print(line)
        if tflops_flash > 0:
            speedup_original = tflops_co_alibi_original / tflops_flash
            speedup_simplified = tflops_co_alibi_simplified / tflops_flash
            print(f"Co-ALIBI Original vs FlashAttention-2 speed-up: ×{speedup_original:.2f} (TFLOP/s based)")
            print(f"Co-ALIBI Simplified vs FlashAttention-2 speed-up: ×{speedup_simplified:.2f} (TFLOP/s based)")
        print(line)


if __name__ == "__main__":
    benchmark_fwd_flops() 