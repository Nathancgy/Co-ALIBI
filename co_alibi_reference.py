import torch
import torch.nn.functional as F
import math

class CoALIBIPyTorch(torch.autograd.Function):
    """
    Direct PyTorch implementation of Co-ALIBI attention with explicit backward pass.
    This serves as a reference for correctness.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale_factor: float, causal: bool):
        B, H, N_q, D_h = q.shape
        N_k = k.shape[2]

        p_raw = torch.einsum('bhid,bhjd->bhij', q, k) * scale_factor

        mask_value = -torch.finfo(p_raw.dtype).max
        expanded_mask_bool = None
        if causal:
            mask_bool = torch.triu(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool), diagonal=1)
            expanded_mask_bool = mask_bool.unsqueeze(0).unsqueeze(0)
            p_raw_masked_for_softmax = p_raw.masked_fill(expanded_mask_bool, mask_value)
            # For sigmoid, ensure masked parts become ~0. If p_raw is used directly, large neg value is fine.
            # If a different masking for sigmoid is needed (e.g. if not causal, but still want some masking for sig)
            # that logic would go here. For CoALIBI as described, causal mask is primary for this stage.
            min_neg = -1e4  # safe for fp16 range
            p_raw_masked_for_sigmoid = p_raw.masked_fill(expanded_mask_bool, min_neg)
        else:
            p_raw_masked_for_softmax = p_raw
            p_raw_masked_for_sigmoid = p_raw # No masking if not causal for sigmoid path
            
        sig_p_raw = torch.sigmoid(p_raw_masked_for_sigmoid)
        if causal and expanded_mask_bool is not None: # Ensure masked sigmas are zero
            sig_p_raw = sig_p_raw.masked_fill(expanded_mask_bool, 0.0)

        z_penalty = torch.cumsum(sig_p_raw.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        if causal and expanded_mask_bool is not None: # Ensure Z_penalty from masked positions is also zero if sig_p_raw was zeroed
             # This might be redundant if sig_p_raw is correctly zeroed before cumsum, but can be a safeguard.
             # However, the cumsum should correctly handle zeros from masked sig_p_raw.
             pass # Original logic was okay here if sig_p_raw is properly masked.

        p_adjusted = p_raw_masked_for_softmax - z_penalty
        if causal and expanded_mask_bool is not None:
            p_adjusted = p_adjusted.masked_fill(expanded_mask_bool, mask_value)

        # Softmax (already numerically stable due to mask_value being large negative)
        s = F.softmax(p_adjusted, dim=-1)
        if causal and expanded_mask_bool is not None:
             s = s.masked_fill(expanded_mask_bool, 0.0) # Ensure strict zeros after softmax for masked positions

        o = torch.einsum('bhij,bhjd->bhid', s, v)

        # Save for backward: q, k, v, s, p_raw (original unmasked QK*scale), sig_p_raw (after sigmoid mask)
        # expanded_mask_bool is important for applying masks in backward correctly.
        ctx.save_for_backward(q, k, v, s, p_raw, sig_p_raw, expanded_mask_bool if causal else torch.tensor([])) # Pass empty if not causal
        ctx.scale_factor = scale_factor
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, s, p_raw, sig_p_raw, expanded_mask_bool = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        causal = ctx.causal
        
        if expanded_mask_bool.numel() == 0: # Check if empty tensor was passed
            expanded_mask_bool = None
            
        dv = torch.einsum('bhji,bhid->bhjd', s, do)
        ds = torch.einsum('bhid,bhjd->bhij', do, v)

        d_softmax_sum = torch.sum(ds * s, dim=-1, keepdim=True)
        dp_adjusted = s * (ds - d_softmax_sum)
        
        if causal and expanded_mask_bool is not None:
            dp_adjusted = dp_adjusted.masked_fill(expanded_mask_bool, 0.0)

        sigma_prime_p_raw = sig_p_raw * (1.0 - sig_p_raw) # sig_p_raw was based on p_raw_masked_for_sigmoid
                                                        # and then explicitly zeroed for causal mask.
                                                        # So sigma_prime will be zero for these.

        # Prefix sum for dp_adjusted: C_tμ
        c_prefix_sum_dp_prime = torch.cumsum(dp_adjusted, dim=-1)
        dp_raw_from_penalty_path = -sigma_prime_p_raw * c_prefix_sum_dp_prime
        
        # dP_raw = dP_adjusted (from softmax bwd) + dP_raw_from_penalty_path
        # This dP_raw is w.r.t. the p_raw that was input to sigmoid and p_adjusted (i.e. QK*scale)
        dp_raw_grad = dp_adjusted + dp_raw_from_penalty_path # Gradient w.r.t. p_raw = QK*scale_factor
        
        if causal and expanded_mask_bool is not None:
            dp_raw_grad = dp_raw_grad.masked_fill(expanded_mask_bool, 0.0)

        dq = torch.einsum('bhij,bhjd->bhid', dp_raw_grad, k) * scale_factor
        dk = torch.einsum('bhij,bhid->bhjd', dp_raw_grad, q) * scale_factor # Corrected: was 'bhji,bhid->bhjd' with dp_raw_grad, q
                                                                       # For dK_τ = sum_t dp_tτ Q_t, if dp is (B,H,Nq,Nk) & Q is (B,H,Nq,Dh)
                                                                       # -> dK should be (B,H,Nk,Dh). einsum 'bhij,bhid->bhjd' means j=Nk, i=Nq, d=Dh.
                                                                       # This means dp_raw_grad[b,h,i,j] and q[b,h,i,d] -> dk[b,h,j,d]. This is correct.

        return dq, dk, dv, None, None # For scale_factor, causal


def co_alibi_attention_pytorch_reference(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return CoALIBIPyTorch.apply(q, k, v, sm_scale, causal) 