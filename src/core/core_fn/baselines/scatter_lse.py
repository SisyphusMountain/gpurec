"""
Custom autograd functions for phylogenetic reconciliation.

This module implements custom PyTorch autograd functions to handle
numerical stability issues in scatter operations, particularly for
log-space computations.
"""

import torch
from typing import Tuple, Optional


class ScatterLogSumExp(torch.autograd.Function):
    """
    Custom autograd function for log-sum-exp scatter operations.
    
    This function implements a numerically stable forward pass using the
    log-sum-exp trick, and a custom backward pass that computes gradients
    using softmax-style weights to avoid NaN gradients.
    
    The forward pass computes:
        out[parent] = log(sum(exp(log_values[i]) for i where parent_idx[i] == parent))
    
    The backward pass computes gradients using:
        grad_input[i] = grad_output[parent[i]] * softmax_weight[i]
    where softmax_weight[i] = exp(log_values[i] - max) / sum(exp(...))
    
    Note: This function supports functorch transforms (jacrev, jacfwd, vmap)
    through proper setup_context, jvp, and vmap implementations.
    """
    
    # We don't use generate_vmap_rule=True because we have custom logic
    generate_vmap_rule = False
    
    @staticmethod
    def forward(log_combined_splits, split_parents, C, ccp_leaves_mask):
        """
        Forward pass: Compute log-sum-exp with scatter operations.
        
        Args:
            log_combined_splits: Log values to aggregate [N_splits, S]
            split_parents: Parent indices for each split [N_splits]
            C: Total number of parent clades
            ccp_leaves_mask: Boolean mask indicating leaf clades [C]
            
        Returns:
            Aggregated log values [C, S]
        """
        device, dtype = log_combined_splits.device, log_combined_splits.dtype
        N_splits, S = log_combined_splits.shape
        
        # Expand parent indices to match shape [N_splits, S]
        split_parents_exp = split_parents.unsqueeze(1).expand(-1, S)
        
        # Step 1: Find maximum values for numerical stability
        max_vals = torch.scatter_reduce(
            torch.full((C, S), float('-inf'), device=device, dtype=dtype),
            0, split_parents_exp, log_combined_splits,
            reduce='amax'
        )
        
        # Step 2: Compute exp(log_val - max) for stability
        gathered_max = torch.gather(max_vals, 0, split_parents_exp)
        exp_terms = torch.exp(log_combined_splits - gathered_max)
        
        # Step 3: Sum the exp terms
        sum_contribs = torch.scatter_add(
            torch.zeros_like(max_vals),
            0, split_parents_exp, exp_terms
        )
        
        # Step 4: Take log and add back the max
        out = torch.log(sum_contribs) + max_vals
        
        # Step 5: Mask out leaf clades (they don't have splits)
        out = torch.where(
            ccp_leaves_mask.unsqueeze(1),  # Broadcast to [C, S]
            torch.full_like(out, float('-inf')),
            out
        )
        
        return out
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for backward pass and functorch compatibility.
        """
        log_combined_splits, split_parents, C, ccp_leaves_mask = inputs
        ctx.input_args = inputs
        device, dtype = log_combined_splits.device, log_combined_splits.dtype
        S = log_combined_splits.shape[1]
        
        split_parents_exp = split_parents.unsqueeze(1).expand(-1, S)
        
        max_vals = torch.scatter_reduce(
            torch.full((C, S), float('-inf'), device=device, dtype=dtype),
            0, split_parents_exp, log_combined_splits,
            reduce='amax'
        )
        
        gathered_max = torch.gather(max_vals, 0, split_parents_exp)
        exp_terms = torch.exp(log_combined_splits - gathered_max)
        
        sum_contribs = torch.scatter_add(
            torch.zeros_like(max_vals),
            0, split_parents_exp, exp_terms
        )
        ctx.save_for_backward(split_parents, exp_terms, sum_contribs, ccp_leaves_mask)
        # C is stored as attribute since it's not a tensor
        ctx.C = C
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: Compute gradients using softmax-style weights.
        
        Args:
            grad_out: Upstream gradient [C, S]
            
        Returns:
            Tuple of gradients for each input (grad_log_combined_splits, None, None, None)
        """
        split_parents, exp_terms, sum_contribs, ccp_leaves_mask = ctx.saved_tensors
        split_parents_exp = split_parents.unsqueeze(1).expand_as(exp_terms)
        
        # Compute softmax weights: exp(x - max) / sum(exp(x - max))
        # Handle zeros in sum_contribs to avoid NaN
        safe_sum = sum_contribs.clone()
        safe_sum[safe_sum == 0] = 1.0
        
        # Weights represent the contribution of each split to its parent's sum
        weights = exp_terms / torch.gather(safe_sum, 0, split_parents_exp)
        
        # Gather upstream gradients for each split
        gathered_grad = torch.gather(grad_out, 0, split_parents_exp)
        
        # Apply chain rule: gradient = upstream_grad * weight
        grad_input = gathered_grad * weights
        
        # Zero out gradients for masked leaf clades
        leaf_mask_exp = torch.gather(
            ccp_leaves_mask.unsqueeze(1),
            0, split_parents.unsqueeze(1)
        ).expand_as(grad_input)
        
        grad_input = torch.where(leaf_mask_exp, torch.zeros_like(grad_input), grad_input)
        
        # Return gradients (None for non-differentiable inputs)
        return grad_input, None, None, None

    @classmethod
    def jvp(cls, ctx, *grad_inputs):
        """
        Forward-mode automatic differentiation (JVP) for ScatterLogSumExp.
        
        This method computes the Jacobian-Vector Product for forward-mode AD.
        It's called just after forward() and before apply() returns.
        
        Args:
            ctx: Context object with saved data from forward()
            grad_inputs: Tuple of gradient tensors for each input:
                - d_log_combined_splits: Tangent for log_combined_splits [N_splits, S] 
                - d_split_parents: None (not differentiable)
                - d_C: None (not differentiable)
                - d_ccp_leaves_mask: None (not differentiable)
        
        Returns:
            Tuple with gradient for the output:
                - d_output: Output tangent [C, S]
        """
        # Extract input gradients
        d_log_combined_splits, d_split_parents, d_C, d_ccp_leaves_mask = grad_inputs
        
        # Get the original input arguments that were passed to forward()
        log_combined_splits, split_parents, C, ccp_leaves_mask = ctx.input_args
        
        # If no gradient for the differentiable input, return zero gradient
        if d_log_combined_splits is None:
            S = log_combined_splits.shape[1] if log_combined_splits is not None else 1
            return (torch.zeros(C, S, device=split_parents.device),)
        
        # Get dimensions
        N_splits, S = d_log_combined_splits.shape
        
        # Recompute the forward computation to get the softmax weights needed for JVP
        device, dtype = d_log_combined_splits.device, d_log_combined_splits.dtype
        
        # Expand parent indices to match shape [N_splits, S]
        split_parents_exp = split_parents.unsqueeze(1).expand(N_splits, S)
        
        # Step 1: Find maximum values for numerical stability (same as forward)
        max_vals = torch.scatter_reduce(
            torch.full((C, S), float('-inf'), device=device, dtype=dtype),
            0, split_parents_exp, log_combined_splits,
            reduce='amax'
        )
        
        # Step 2: Compute exp(log_val - max) for stability
        gathered_max = torch.gather(max_vals, 0, split_parents_exp)
        exp_terms = torch.exp(log_combined_splits - gathered_max)
        
        # Step 3: Sum the exp terms
        sum_contribs = torch.scatter_add(
            torch.zeros_like(max_vals),
            0, split_parents_exp, exp_terms
        )
        
        # Compute softmax weights: exp_terms / sum_contribs (same logic as backward)
        safe_sum = sum_contribs.clone()
        safe_sum[safe_sum == 0] = 1.0
        weights = exp_terms / torch.gather(safe_sum, 0, split_parents_exp)
        
        # Apply the JVP: weighted sum of input gradients, scattered to parent clades
        weighted_grads = weights * d_log_combined_splits
        d_output = torch.scatter_add(
            torch.zeros(C, S, device=device, dtype=dtype),
            0, split_parents_exp, weighted_grads
        )
        
        # Apply leaf mask: leaf clades have -inf output, so zero gradient
        d_output = torch.where(
            ccp_leaves_mask.unsqueeze(1).expand(C, S),
            torch.zeros_like(d_output), 
            d_output
        )
        
        return (d_output,)

    @staticmethod
    def vmap(info, in_dims, log_combined_splits, split_parents, C, ccp_leaves_mask):
        """
        Vectorized map (vmap) implementation for batched ScatterLogSumExp operations.
        
        This provides proper vectorized batching without loops for efficiency.
        Typically only log_combined_splits is batched, with other arguments shared.
        
        Args:
            info: BatchedTensorInfo containing batch_size and other metadata
            in_dims: Tuple indicating which dimensions are batched for each input
            log_combined_splits, split_parents, C, ccp_leaves_mask: Function arguments
            
        Returns:
            Tuple of (batched_output, output_batch_dims)
        """
        # Extract batch dimensions for each input
        log_splits_bdim, split_parents_bdim, C_bdim, mask_bdim = in_dims
        
        # Move batch dimensions to front if needed
        if log_splits_bdim is not None:
            log_combined_splits = log_combined_splits.movedim(log_splits_bdim, 0)
            batch_size = log_combined_splits.shape[0]
        elif split_parents_bdim is not None:
            split_parents = split_parents.movedim(split_parents_bdim, 0)
            batch_size = split_parents.shape[0]
        elif mask_bdim is not None:
            ccp_leaves_mask = ccp_leaves_mask.movedim(mask_bdim, 0)
            batch_size = ccp_leaves_mask.shape[0]
        else:
            # No batched inputs
            result = ScatterLogSumExp.apply(log_combined_splits, split_parents, C, ccp_leaves_mask)
            return result, None
        
        # Expand non-batched inputs to match batch size for vectorized operations
        if log_splits_bdim is None:
            # log_combined_splits not batched, expand it
            log_combined_splits = log_combined_splits.unsqueeze(0).expand(batch_size, -1, -1)
        
        if mask_bdim is None and isinstance(ccp_leaves_mask, torch.Tensor):
            # ccp_leaves_mask not batched, expand it
            ccp_leaves_mask = ccp_leaves_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Now perform vectorized computation
        # log_combined_splits: [batch_size, N_splits, S] 
        # split_parents: [N_splits] (same for all batches)
        # ccp_leaves_mask: [batch_size, C] or [C]
        
        device, dtype = log_combined_splits.device, log_combined_splits.dtype
        B, N_splits, S = log_combined_splits.shape
        
        # Expand parent indices for all batches: [batch_size, N_splits, S]
        split_parents_exp = split_parents.unsqueeze(0).unsqueeze(-1).expand(B, N_splits, S)
        
        # Step 1: Find maximum values for numerical stability
        # Initialize with -inf: [batch_size, C, S]
        max_vals = torch.full((B, C, S), float('-inf'), device=device, dtype=dtype)
        
        # Scatter reduce over batch dimension efficiently
        # Flatten for scatter: [batch_size * N_splits, S] and indices: [batch_size * N_splits, S]
        log_splits_flat = log_combined_splits.reshape(B * N_splits, S)
        parents_flat = split_parents_exp.reshape(B * N_splits, S)
        
        # Add batch offset to parent indices: batch_i maps to indices C*i + parent
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_splits, S)
        batch_offset_parents = parents_flat + C * batch_indices.reshape(B * N_splits, S)
        
        # Scatter reduce to find max values across all batches simultaneously
        max_vals_flat = torch.scatter_reduce(
            torch.full((B * C, S), float('-inf'), device=device, dtype=dtype),
            0, batch_offset_parents, log_splits_flat, reduce='amax'
        )
        max_vals = max_vals_flat.view(B, C, S)
        
        # Step 2: Gather max values and compute exp terms
        gathered_max = torch.gather(max_vals, 1, split_parents_exp)  # [B, N_splits, S]
        exp_terms = torch.exp(log_combined_splits - gathered_max)  # [B, N_splits, S]
        
        # Step 3: Sum the exp terms using vectorized scatter
        sum_contribs_flat = torch.scatter_add(
            torch.zeros(B * C, S, device=device, dtype=dtype),
            0, batch_offset_parents, exp_terms.reshape(B * N_splits, S)
        )
        sum_contribs = sum_contribs_flat.view(B, C, S)
        
        # Step 4: Take log and add back the max
        out = torch.log(sum_contribs) + max_vals
        
        # Step 5: Apply leaf mask
        if mask_bdim is None:
            # Same mask for all batches
            leaf_mask = ccp_leaves_mask[0].unsqueeze(0).unsqueeze(-1).expand(B, C, S)
        else:
            # Different mask per batch
            leaf_mask = ccp_leaves_mask.unsqueeze(-1).expand(B, C, S)
        
        out = torch.where(leaf_mask, torch.full_like(out, float('-inf')), out)
        
        # Output batch dimension is 0
        return out, 0








