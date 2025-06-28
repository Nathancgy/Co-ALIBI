# Co-ALIBI: Contextual Attention with ALiBi

A high-performance GPU implementation of Co-ALIBI (Contextual Attention with Linear Biases), a novel attention mechanism that extends ALiBi with contextual positional encoding through sigmoid-based penalty terms.

## Key Features

- **Contextual Position Encoding**: Uses sigmoid-based cumulative penalties for position-aware attention
- **Optimized Triton Kernels**: Custom CUDA kernels via Triton for both forward and backward passes
- **High Performance**: Achieves ~180 TFLOPS/s (forward) and ~80 TFLOPS/s (backward) on H100
- **FlashAttention-Compatible**: Similar memory efficiency and computational complexity
- **Accuracy**: Passes validation with eps=1e-4 against reference implementation

## Installation

```bash
pip install torch triton
```

## Usage

```python
from co_alibi_attn import co_alibi_attention

# Input tensors (B=batch, H=heads, S=sequence, D=head_dim)
q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# Apply Co-ALIBI attention
output = co_alibi_attention(q, k, v, causal=True)
```

## Algorithm Overview

Co-ALIBI modifies standard attention by introducing contextual position penalties:

1. Compute raw attention scores: `p_raw = Q @ K^T * scale`
2. Calculate sigmoid penalties: `σ(p_raw)` for all valid positions
3. Apply cumulative penalty: `z = Σ_{j>i} σ(q_i · k_j)`
4. Adjust scores: `p_adjusted = p_raw - slope * z`
5. Apply softmax and compute output: `O = softmax(p_adjusted) @ V`

The key innovation is the sigmoid-based cumulative penalty that provides context-aware positional biases.

## Performance Benchmarks

On NVIDIA H100 (sequence length 4096, 16 heads, head_dim 128):

| Operation | TFLOPS/s | Latency (ms) |
|-----------|----------|--------------|
| Forward   | ~160     | ~0.88        |
| Backward  | ~80      | ~3.4         |

## Repository Structure

```
co_alibi_attn/
├── co_alibi_attn.py         # Main attention implementation
├── co_alibi_fwd_kernel.py   # Triton forward kernel
├── co_alibi_bwd_kernel.py   # Triton backward kernel
├── benchmark_flops.py       # Performance benchmarking
├── benchmark_fwd_pass.py    # Forward pass validation
└── benchmark_bwd_pass.py    # Backward pass validation
model.py                     # Reference implementation for testing
```

## Technical Details

- **Causal Masking**: Built-in support for autoregressive models
- **Numerical Stability**: Uses log-sum-exp trick for stable softmax computation
- **Multi-Query Attention**: Supports different numbers of Q and KV heads
- **Configurable Slopes**: ALiBi slopes computed based on number of heads with bias_max parameter

## Benchmarking

Compare performance with FlashAttention 2:

```bash
python co_alibi_attn/benchmark_flops.py
```

Validate accuracy:

```bash
python co_alibi_attn/benchmark_fwd_pass.py
python co_alibi_attn/benchmark_bwd_pass.py
``` 