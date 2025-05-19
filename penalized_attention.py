# penalized_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PenalizedAttention(nn.Module):
    """
    Penalized Attention mechanism.
    Forward pass based on user's "Algorithm 1 Forward Pass Co-ALiBi Qc".
    This version uses PyTorch operations for automatic differentiation.
    It does not use explicit positional encodings.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        assert C == self.n_embd

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # Each is (B, T, C)

        # Reshape q, k, v from (B, T, C) to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- Penalized Attention Calculation ---
        scale_factor = 1.0 / math.sqrt(self.head_dim)

        # 1. P = QK^T (Raw scores)
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        P = (q @ k.transpose(-2, -1)) * scale_factor

        # 2. S = sigmoid(P)
        S = torch.sigmoid(P) # (B, nh, T, T)

        # 3. Z_tτ = sum_{j=τ to t} S_tj (Penalty term)
        # For each query q_t (represented by a row in S), Z_tτ is the sum of S_tj
        # where j ranges from τ up to t.
        # We can achieve this by first creating a version of S where S_tj = 0 if j > t.
        # This is S_lower_tri_inclusive_of_diag = torch.tril(S)
        S_lower_tri = torch.tril(S) # S_lower_tri[b,h,t,j] = S[b,h,t,j] if j <= t, else 0

        # Now, for each S_lower_tri[b,h,t,:], we want Z[b,h,t,τ] = sum S_lower_tri[b,h,t,j] for j from τ to T-1
        # This is a right-to-left cumulative sum.
        Z = S_lower_tri.flip(dims=[3]).cumsum(dim=3).flip(dims=[3])
        # Z[b,h,t,τ] now correctly holds sum_{j=τ to t} S[b,h,t,j]
        # because S_lower_tri is 0 for j > t, so cumsum effectively stops at t.

        # 4. P' = P - Z (Adjusted scores)
        P_prime = P - Z

        # 5. Softmax with Causal Masking
        # Create causal mask: True where attention is allowed (τ <= t)
        # Mask P_prime before softmax: where mask is False, fill with -inf
        # T is sequence length
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        P_prime_masked = P_prime.masked_fill(~causal_mask, float('-inf'))

        att_weights = F.softmax(P_prime_masked, dim=-1) # (B, nh, T, T)
        att_weights = self.attn_dropout(att_weights)    # Apply dropout

        # 6. Output y = att_weights @ v
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att_weights @ v

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_embd)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y