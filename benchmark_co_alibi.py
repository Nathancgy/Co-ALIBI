import torch
import time
import math

from co_alibi_attn import co_alibi_attention # Triton implementation
from co_alibi_reference import co_alibi_attention_pytorch_reference # PyTorch reference

def benchmark(
    bsz, num_heads, q_seqlen, kv_seqlen, head_dim, 
    causal, dtype, device, 
    label, triton_impl, ref_impl,
    num_warmup=10, num_repeats=100
):
    print(f"--- Benchmarking: {label} ---")
    print(f"Bsz: {bsz}, Heads: {num_heads}, Q_len: {q_seqlen}, KV_len: {kv_seqlen}, D_head: {head_dim}, Causal: {causal}, Dtype: {dtype}")

    q = torch.randn(bsz, num_heads, q_seqlen, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(bsz, num_heads, kv_seqlen, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(bsz, num_heads, kv_seqlen, head_dim, dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # --- Correctness Check --- 
    print("Running correctness check...")
    try:
        # Triton implementation
        q_triton, k_triton, v_triton = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
        output_triton = triton_impl(q_triton, k_triton, v_triton, causal=causal, sm_scale=sm_scale)
        do_triton = torch.randn_like(output_triton)
        output_triton.backward(do_triton)
        dq_triton, dk_triton, dv_triton = q_triton.grad, k_triton.grad, v_triton.grad

        # Reference implementation
        q_ref, k_ref, v_ref = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
        output_ref = ref_impl(q_ref, k_ref, v_ref, causal=causal, sm_scale=sm_scale)
        # Use the same do for fair comparison of gradients
        do_ref = do_triton.clone().detach().to(output_ref.device, output_ref.dtype) 
        output_ref.backward(do_ref)
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

        # Compare outputs
        out_allclose = torch.allclose(output_triton, output_ref, atol=1e-5, rtol=1e-4)
        print(f"  Output allclose: {out_allclose}")
        if not out_allclose:
            print(f"    Max diff output: {torch.max(torch.abs(output_triton - output_ref))}")

        # Compare gradients
        dq_allclose = torch.allclose(dq_triton, dq_ref, atol=1e-5, rtol=1e-4)
        print(f"  dQ allclose: {dq_allclose}")
        if not dq_allclose:
            print(f"    Max diff dQ: {torch.max(torch.abs(dq_triton - dq_ref))}")
        
        dk_allclose = torch.allclose(dk_triton, dk_ref, atol=1e-5, rtol=1e-4)
        print(f"  dK allclose: {dk_allclose}")
        if not dk_allclose:
            print(f"    Max diff dK: {torch.max(torch.abs(dk_triton - dk_ref))}")

        dv_allclose = torch.allclose(dv_triton, dv_ref, atol=1e-5, rtol=1e-4)
        print(f"  dV allclose: {dv_allclose}")
        if not dv_allclose:
            print(f"    Max diff dV: {torch.max(torch.abs(dv_triton - dv_ref))}")
        
        if not (out_allclose and dq_allclose and dk_allclose and dv_allclose):
            print("  !!! CORRECTNESS CHECK FAILED !!!")
        else:
            print("  Correctness check passed.")

    except Exception as e:
        print(f"  Correctness check FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        # Continue to performance benchmark if desired, or raise

    # --- Performance Benchmark --- 
    print("Running performance benchmark...")
    # Detach inputs for performance runs as grads are not needed for timing loop itself
    q_perf, k_perf, v_perf = q.clone().detach(), k.clone().detach(), v.clone().detach()
    do_perf = torch.randn_like(output_triton if 'output_triton' in locals() else q_perf) # Dummy grad for bwd timing

    # Forward Pass Only
    def run_forward(impl_fn):
        return impl_fn(q_perf, k_perf, v_perf, causal=causal, sm_scale=sm_scale)

    # Warm-up
    for _ in range(num_warmup):
        _ = run_forward(triton_impl)
        _ = run_forward(ref_impl)
    torch.cuda.synchronize()

    # Triton Forward
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_repeats):
        _ = run_forward(triton_impl)
    end_event.record()
    torch.cuda.synchronize()
    triton_fwd_time = start_event.elapsed_time(end_event) / num_repeats
    print(f"  Triton Forward: {triton_fwd_time:.4f} ms")

    # Reference Forward
    start_event.record()
    for _ in range(num_repeats):
        _ = run_forward(ref_impl)
    end_event.record()
    torch.cuda.synchronize()
    ref_fwd_time = start_event.elapsed_time(end_event) / num_repeats
    print(f"  PyTorch Ref Forward: {ref_fwd_time:.4f} ms")

    # Forward + Backward Pass
    def run_fwd_bwd(impl_fn, q_gb, k_gb, v_gb, do_gb):
        q_gb.grad, k_gb.grad, v_gb.grad = None, None, None # Clear grads
        output = impl_fn(q_gb, k_gb, v_gb, causal=causal, sm_scale=sm_scale)
        output.backward(do_gb, retain_graph=False)
        return q_gb.grad, k_gb.grad, v_gb.grad

    # Prepare inputs with requires_grad for fwd+bwd timing
    q_gb_triton, k_gb_triton, v_gb_triton = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gb_ref, k_gb_ref, v_gb_ref = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    
    # Warm-up for Fwd+Bwd
    for _ in range(num_warmup):
        _ = run_fwd_bwd(triton_impl, q_gb_triton, k_gb_triton, v_gb_triton, do_perf)
        _ = run_fwd_bwd(ref_impl, q_gb_ref, k_gb_ref, v_gb_ref, do_perf) # Use same do_perf shape
    torch.cuda.synchronize()

    # Triton Fwd + Bwd
    start_event.record()
    for _ in range(num_repeats):
        _ = run_fwd_bwd(triton_impl, q_gb_triton, k_gb_triton, v_gb_triton, do_perf)
    end_event.record()
    torch.cuda.synchronize()
    triton_fwd_bwd_time = start_event.elapsed_time(end_event) / num_repeats
    print(f"  Triton Fwd+Bwd: {triton_fwd_bwd_time:.4f} ms")

    # Reference Fwd + Bwd
    start_event.record()
    for _ in range(num_repeats):
        _ = run_fwd_bwd(ref_impl, q_gb_ref, k_gb_ref, v_gb_ref, do_perf)
    end_event.record()
    torch.cuda.synchronize()
    ref_fwd_bwd_time = start_event.elapsed_time(end_event) / num_repeats
    print(f"  PyTorch Ref Fwd+Bwd: {ref_fwd_bwd_time:.4f} ms")
    print("---")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, Triton benchmarks require GPU. Skipping.")
        exit()

    # Common test configurations
    configs = [
        {'bsz': 4, 'num_heads': 8, 'q_seqlen': 512, 'kv_seqlen': 512, 'head_dim': 64, 'causal': True, 'dtype': torch.float16},
        {'bsz': 4, 'num_heads': 8, 'q_seqlen': 1024, 'kv_seqlen': 1024, 'head_dim': 64, 'causal': True, 'dtype': torch.float16},
        {'bsz': 2, 'num_heads': 12, 'q_seqlen': 2048, 'kv_seqlen': 2048, 'head_dim': 64, 'causal': True, 'dtype': torch.float16},
        {'bsz': 1, 'num_heads': 16, 'q_seqlen': 4096, 'kv_seqlen': 4096, 'head_dim': 64, 'causal': True, 'dtype': torch.float16},
        # Add more configurations: different dtypes (float32), non-causal, different head_dims if triton kernels support them well
        {'bsz': 4, 'num_heads': 8, 'q_seqlen': 512, 'kv_seqlen': 512, 'head_dim': 64, 'causal': True, 'dtype': torch.float32},
        {'bsz': 4, 'num_heads': 8, 'q_seqlen': 512, 'kv_seqlen': 512, 'head_dim': 64, 'causal': False, 'dtype': torch.float16},
    ]

    for config in configs:
        benchmark(
            bsz=config['bsz'], num_heads=config['num_heads'], 
            q_seqlen=config['q_seqlen'], kv_seqlen=config['kv_seqlen'], 
            head_dim=config['head_dim'], causal=config['causal'], 
            dtype=config['dtype'], device=device,
            label=f"CoALIBI (Q={config['q_seqlen']}, KV={config['kv_seqlen']}, H={config['head_dim']}, causal={config['causal']}, {config['dtype']})".replace("torch.",""), 
            triton_impl=co_alibi_attention, 
            ref_impl=co_alibi_attention_pytorch_reference
        ) 