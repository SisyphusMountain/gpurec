import torch


def dup_both_survive(Pi_left, Pi_right, log_split_probs, log_pD):
    """
    input shapes: 
        Pi_left: [N_splits, S]
        Pi_right: [N_splits, S]
        log_split_probs: [N_splits]
        log_pD: [1]
    Formula:
        p_D * p(gamma', gamma''|gamma) * Pi_{gamma',e} * Pi_{gamma'',e}
    """
    return (log_split_probs.unsqueeze(1) + 
            Pi_left + Pi_right + log_pD)  # [N_splits, S]

def gather_E_children(E, child_index, internal_mask):
    """Gather extinction probabilities for internal nodes"""
    E_child = torch.full_like(E, float('-inf'))  # [S]
    values = torch.index_select(E, 0, child_index)  # [N_internal_nodes]
    # Will take values in values and put them sequentially where internal_mask is True.
    E_child.masked_scatter_(internal_mask, values)
    return E_child

def gather_Pi_children(Pi, child_index, internal_mask):
    """Differentiable"""
    Pi_child = torch.full_like(Pi, float('-inf'))  # [C, S]. Default log-prob is -inf.
    mask2d = internal_mask.unsqueeze(0).expand(Pi.shape)  # [C, S]
    values = torch.index_select(Pi, 1, child_index).reshape(-1)  # [C * N_internal_nodes]
    # Will take values in values and put them sequentially where mask2d is True.
    Pi_child.masked_scatter_(mask2d, values)
    return Pi_child

def E_step(E, sp_child1_idx, sp_child2_idx, internal_mask, Recipients_mat, theta):
    # theta is a tensor with the elements log_delta, log_tau, log_lambda
    # Ensure theta is on the same device and dtype as E
    device = E.device
    dtype = E.dtype
    theta = theta.to(device=device, dtype=dtype)

    exp_theta = torch.exp(theta)
    delta = exp_theta[0]
    tau = exp_theta[1]
    lambda_param = exp_theta[2]
    rates_sum = 1.0 + delta + tau + lambda_param
    log_pS = torch.log(1.0 / rates_sum)
    log_pD = torch.log(delta / rates_sum)
    log_pT = torch.log(tau / rates_sum)
    log_pL = torch.log(lambda_param / rates_sum)

    E_s1 = gather_E_children(E, sp_child1_idx, internal_mask)
    E_s2 = gather_E_children(E, sp_child2_idx, internal_mask)

    speciation = log_pS + E_s1 + E_s2
    duplication = log_pD + 2 * E

    max_E = torch.max(E)
    Ebar = torch.log(torch.mv(Recipients_mat, torch.exp(E - max_E))) + max_E
    transfer = log_pT + E + Ebar

    # Optimized: incremental logsumexp avoids tensor stacking
    new_E = speciation  # Start with speciation
    new_E = torch.logaddexp(new_E, duplication)
    new_E = torch.logaddexp(new_E, transfer)
    new_E = torch.logaddexp(new_E, log_pL.expand_as(speciation))

    return new_E, E_s1, E_s2, Ebar



def Pi_step(Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, theta):
    """
    Log-space version of Pi_update_ccp_parallel to handle numerical instability.
    
    Args:
        Pi: Log probabilities matrix [C, S] in log space
        ccp_helpers: Dictionary with split information
        species_helpers: Species tree information  
        clade_species_map: Mapping matrix [C, S]
        E, Ebar: Extinction probabilities
        log_pS, log_pD, log_pT: Event probabilities in log space
        debug: If True, log tensor statistics for debugging
        
    Returns:
        new_Pi: Updated log probabilities [C, S]
    """
    # Ensure theta is on the same device and dtype as Pi
    device = Pi.device
    dtype = Pi.dtype
    theta = theta.to(device=device, dtype=dtype)

    exp_theta = torch.exp(theta)
    delta = exp_theta[0]
    tau = exp_theta[1]
    lambda_param = exp_theta[2]
    rates_sum = 1.0 + delta + tau + lambda_param
    log_pS = torch.log(1.0 / rates_sum)
    log_pD = torch.log(delta / rates_sum)
    log_pT = torch.log(tau / rates_sum)
    log_pL = torch.log(lambda_param / rates_sum)
    # region helpers
    # Extract helpers
    split_parents = ccp_helpers['split_parents']
    split_lefts = ccp_helpers['split_lefts']
    split_rights = ccp_helpers['split_rights']
    split_probs = ccp_helpers['split_probs']
    log_split_probs = torch.log(split_probs)
    ccp_leaves_mask = ccp_helpers['ccp_leaves_mask']
    sp_c1_idx = species_helpers['s_C1_indexes'] # index of first child for each internal node
    sp_c2_idx = species_helpers['s_C2_indexes'] # index of second child for each internal node
    Recipients_mat = species_helpers['Recipients_mat']
    internal_mask = species_helpers["sp_internal_mask"]
    C, S = Pi.shape
    device = Pi.device
    dtype = Pi.dtype
    # endregion helpers

    # Get log Pi values for left and right children
    Pi_left = torch.index_select(Pi, 0, split_lefts)    # [N_splits, S]
    Pi_right = torch.index_select(Pi, 0, split_rights)  # [N_splits, S]

    # region duplication
    # both copies survive
    log_D_splits = dup_both_survive(Pi_left, Pi_right, log_split_probs, log_pD)
    # one copy goes extinct
    log_2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))
    log_D_loss = log_2 + log_pD + Pi + E.unsqueeze(0)  # [C, S]
    # endregion duplication

    # region speciation
    Pi_s1 = gather_Pi_children(Pi, sp_c1_idx, internal_mask)  # [C, S]
    Pi_s2 = gather_Pi_children(Pi, sp_c2_idx, internal_mask)  # [C, S]

    # Extract for splits
    Pi_s1_left = torch.index_select(Pi_s1, 0, split_lefts)   # [N_splits, S]
    Pi_s1_right = torch.index_select(Pi_s1, 0, split_rights) # [N_splits, S]
    Pi_s2_left = torch.index_select(Pi_s2, 0, split_lefts)   # [N_splits, S]
    Pi_s2_right = torch.index_select(Pi_s2, 0, split_rights) # [N_splits, S]

    # both copies survive
    log_spec1 = log_split_probs.unsqueeze(1) + log_pS + Pi_s1_left + Pi_s2_right  # [N_splits, S]
    log_spec2 = log_split_probs.unsqueeze(1) + log_pS + Pi_s1_right + Pi_s2_left  # [N_splits, S]


    # one copy goes extinct

    log_S_term1 = log_pS + Pi_s1 + E_s2.unsqueeze(0)  # [C, S]
    log_S_term2 = log_pS + Pi_s2 + E_s1.unsqueeze(0)  # [C, S]
    # For leaf speciation events, one copy doesn't really go extinct, but S event on leaves still gives only one copy, not two.
    # Therefore, the tensor clade_species_map has the same shape as Pi. [C, S]
    log_leaf_contrib = log_pS + clade_species_map

    # endregion speciation
    # region transfer
    # both copies survive
    Pi_max = torch.max(Pi, dim=1, keepdim=True).values
    Pi_linear = torch.exp(Pi - Pi_max)  # [C, S]
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
    Pibar = torch.log(Pibar_linear) + Pi_max  # [C, S]

    # Extract transfer terms for splits
    Pibar_left = torch.index_select(Pibar, 0, split_lefts)   # [N_splits, S]
    Pibar_right = torch.index_select(Pibar, 0, split_rights) # [N_splits, S]
    
    # Transfer: log(p_T * split_probs * (Pi_left * Pibar_right + Pi_right * Pibar_left))
    log_trans1 = log_split_probs.unsqueeze(1) + log_pT + Pi_left + Pibar_right  # [N_splits, S]
    log_trans2 = log_split_probs.unsqueeze(1) + log_pT + Pi_right + Pibar_left  # [N_splits, S]

    # only one copy survives
    log_T_term1 = log_pT + Pi + Ebar.unsqueeze(0)  # [C, S]
    log_T_term2 = log_pT + Pibar + E.unsqueeze(0)  # [C, S]
    # endregion transfer



        
    # === COMBINE ALL CONTRIBUTIONS WITHOUT LOSSES ===
    # Optimized: incremental logsumexp avoids large tensor stack allocation
    log_combined_splits = log_D_splits  # Start with first term
    log_combined_splits = torch.logaddexp(log_combined_splits, log_spec1)
    log_combined_splits = torch.logaddexp(log_combined_splits, log_spec2) 
    log_combined_splits = torch.logaddexp(log_combined_splits, log_trans1)
    log_combined_splits = torch.logaddexp(log_combined_splits, log_trans2)


    # Use custom autograd function for log-sum-exp scatter operations
    # Apply custom ScatterLogSumExp function with manual backward pass
    contribs_1 = ScatterLogSumExp.apply(
        log_combined_splits, split_parents, C, ccp_leaves_mask
    )

    # === COMBINE ALL CONTRIBUTIONS INCLUDING LOSSES ===
    # Optimized: incremental logsumexp avoids large tensor stack allocation
    new_Pi = contribs_1  # Start with scatter contributions
    new_Pi = torch.logaddexp(new_Pi, log_D_loss)
    new_Pi = torch.logaddexp(new_Pi, log_S_term1)
    new_Pi = torch.logaddexp(new_Pi, log_S_term2) 
    new_Pi = torch.logaddexp(new_Pi, log_leaf_contrib)
    new_Pi = torch.logaddexp(new_Pi, log_T_term1)
    new_Pi = torch.logaddexp(new_Pi, log_T_term2)


    return new_Pi


def scatter_logsumexp(log_combined_splits, split_parents, C, ccp_leaves_mask):
    device, dtype = log_combined_splits.device, log_combined_splits.dtype
    N_splits, S = log_combined_splits.shape
    
    # Expand parent indices to match shape [N_splits, S]
    split_parents_exp = split_parents.unsqueeze(1).expand(-1, S)
    
    # Pre-allocate output tensor with leaf mask applied (avoid torch.where later)
    out = torch.full((C, S), float('-inf'), device=device, dtype=dtype)
    
    # Step 1: Find maximum values for numerical stability (reuse out tensor)
    max_vals = torch.scatter_reduce(
        out.clone(),  # Reuse pre-allocated tensor
        0, split_parents_exp, log_combined_splits,
        reduce='amax'
    )
    
    # Step 2: Compute exp(log_val - max) for stability
    gathered_max = torch.gather(max_vals, 0, split_parents_exp)
    exp_terms = torch.exp(log_combined_splits - gathered_max)
    
    # Step 3: Sum the exp terms (reuse max_vals as temporary)
    sum_contribs = torch.scatter_add(
        torch.zeros_like(max_vals),
        0, split_parents_exp, exp_terms
    )
    
    # Step 4: Take log and add back the max (write directly to out)
    torch.add(torch.log(sum_contribs), max_vals, out=out)
    
    # Step 5: Apply leaf mask (leaf clades already have -inf from initialization)
    leaf_mask_expanded = ccp_leaves_mask.unsqueeze(1).expand(-1, S)
    out.masked_fill_(leaf_mask_expanded, float('-inf'))
    
    return out

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
        Optimized forward pass: Compute log-sum-exp with scatter operations.
        
        Optimizations:
        1. Pre-allocate output tensor and reuse it
        2. Use in-place operations where possible
        3. Combine leaf masking with initialization
        4. Reduce temporary tensor allocations
        
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
        
        # Pre-allocate output tensor with leaf mask applied (avoid torch.where later)
        out = torch.full((C, S), float('-inf'), device=device, dtype=dtype)
        
        # Step 1: Find maximum values for numerical stability (reuse out tensor)
        max_vals = torch.scatter_reduce(
            out.clone(),  # Reuse pre-allocated tensor
            0, split_parents_exp, log_combined_splits,
            reduce='amax'
        )
        
        # Step 2: Compute exp(log_val - max) for stability
        gathered_max = torch.gather(max_vals, 0, split_parents_exp)
        exp_terms = torch.exp(log_combined_splits - gathered_max)
        
        # Step 3: Sum the exp terms (reuse max_vals as temporary)
        sum_contribs = torch.scatter_add(
            torch.zeros_like(max_vals),
            0, split_parents_exp, exp_terms
        )
        
        # Step 4: Take log and add back the max (write directly to out)
        torch.add(torch.log(sum_contribs), max_vals, out=out)
        
        # Step 5: Apply leaf mask (leaf clades already have -inf from initialization)
        leaf_mask_expanded = ccp_leaves_mask.unsqueeze(1).expand(-1, S)
        out.masked_fill_(leaf_mask_expanded, float('-inf'))
        
        return out
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Optimized setup context - recompute minimal state for backward pass.
        
        Instead of recomputing expensive scatter operations, we derive the backward
        state from the forward pass output when possible.
        """
        log_combined_splits, split_parents, C, ccp_leaves_mask = inputs
        
        # Store inputs for jvp method (required for functorch compatibility)
        ctx.input_args = inputs
        ctx.C = C
        
        # For backward pass, we need exp_terms and sum_contribs
        # These can be recomputed efficiently using the output
        device, dtype = log_combined_splits.device, log_combined_splits.dtype
        S = log_combined_splits.shape[1]
        
        # Expand parent indices to match shape [N_splits, S]
        split_parents_exp = split_parents.unsqueeze(1).expand(-1, S)
        
        # Derive max_vals from output (avoid recomputing scatter_reduce)
        # output = log(sum_contribs) + max_vals, so we can work backwards
        # But we still need to recompute for numerical stability
        # This is a fundamental limitation - we need these for gradients
        
        # Optimized: Use the same pattern as forward pass
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
        
        # Save minimal state for backward pass
        ctx.save_for_backward(split_parents, exp_terms, sum_contribs, ccp_leaves_mask)
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
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
