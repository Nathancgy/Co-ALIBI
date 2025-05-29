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
    B: int = 4,
    H: int = 8,
    seq_lens=(1024, 2048),
    D: int = 64,
    dtype: torch.dtype = torch.float16,
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

        def co_alibi_fwd():
            co_alibi_attention(q, k, v, causal=True, sm_scale=sm_scale)

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
        ms_co_alibi = triton.testing.do_bench(co_alibi_fwd, rep=100, warmup=250)
        ms_flash    = triton.testing.do_bench(flash_attn_fwd, rep=100, warmup=250)

        # FLOPs under 4·D convention
        SKV = S
        flops_per_qk = 4 * D
        total_flops = B * H * S * SKV * flops_per_qk

        tflops_co_alibi = total_flops / (ms_co_alibi * 1e-3) / 1e12
        tflops_flash    = total_flops / (ms_flash * 1e-3) / 1e12

        line = "=" * 60
        print(line)
        print(f"Config: B={B}, H={H}, S={S}, D={D}, dtype={dtype.__str__().split('.')[-1]}  (total_flops={total_flops/1e12:.2f} TF)")
        print(line)

        def _report(name: str, latency_ms: float, tflops: float):
            print(f"{name:<22}  latency = {latency_ms:7.3f} ms   |  throughput = {tflops:6.2f} TFLOP/s")

        _report("Co-ALIBI (Triton)", ms_co_alibi, tflops_co_alibi)
        _report("FlashAttention-2", ms_flash, tflops_flash)
        print(line)
        if tflops_flash > 0:
            speedup = tflops_co_alibi / tflops_flash
            print(f"Co-ALIBI vs FlashAttention-2 speed-up: ×{speedup:.2f} (TFLOP/s based)")
        print(line)


if __name__ == "__main__":
    benchmark_fwd_flops() 