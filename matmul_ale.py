# Première version : réconcilie un arbre de gène enraciné et un arbre d'espèces, avec des taux fixés.
# - Les multiplications matricielles ne sont pas sparse pour l'instant.
# - Il faut implémenter le gradient des paramètres d,t,l
# - Il faut faire les calculs sur les log-vraisemblances pour éviter l'underflow.
# - Si on voulait le vrai ALE, il faudrait aussi modifier les matmul pour faire l'amalgamation des arbres de gènes.
# - On n'est pas obligés de refaire les mêmes opérations autant de fois qu'il y a de racines possibles. A la place, on peut le faire pour les 3 orientations possibles de chaque noeud.
from ete3 import Tree
import torch
from tabulate import tabulate
import numpy as np
from time import time
import random
import math

def load_tree(path):
    return Tree(path, format=1)

def make_leaf_map(g_tree):
    return {leaf.name:leaf.name.split('_')[0] for leaf in g_tree.get_leaves()}

def build_helpers(sp_tree_path,
                  g_tree_path,
                  device,
                  dtype,
                  traversalorder):
    """
    Returns a dict containing:
      - C1_br_dense (S×S): 1 if that column is the left‐child of the row‐node
      - C2_br_dense (S×S): 1 if that column is the right‐child of the row‐node
      - ancestors_dense (S×S): 1 if column is in Anc(row), including row itself
      - Recipients_mat  (S×S): each row e has 1/|nonanc(e)| on non‐ancestor columns
      - leaves_mask     (S,) Bool
      - species_internal_index (list of length S): None if leaf, else internal‐index
      - idx_sp          (dict node→e)
      - species_nodes   (list of length S)
    """
    sp_tree = load_tree(sp_tree_path)
    g_tree  = load_tree(g_tree_path)
    species_nodes = list(sp_tree.traverse(traversalorder))
    S = len(species_nodes)
    idx_sp = {node: i for i, node in enumerate(species_nodes)}
    s_C1 = torch.zeros((S, S), dtype=dtype, device=device)
    s_C2 = torch.zeros((S, S), dtype=dtype, device=device)
    sp_leaves_mask  = torch.zeros((S,), dtype=torch.bool, device=device)
    s_children_idx = {}
    # 1) Fill children
    sp_names_by_idx = {}
    for node in species_nodes:
        e = idx_sp[node]
        sp_names_by_idx[e] = node.name
        children = node.get_children()
        if len(children) == 0:
            sp_leaves_mask[e] = True
        else:
            lc, rc = children
            i_l = idx_sp[lc]
            i_r = idx_sp[rc]
            s_C1[e, i_l] = 1.0
            s_C2[e, i_r] = 1.0
            s_children_idx[e] = (i_l, i_r)
    sp_internal_mask = ~sp_leaves_mask
    # 2) Build ancestors_dense
    ancestors_dense = torch.zeros((S, S), dtype=dtype, device=device)
    for node in species_nodes:
        e = idx_sp[node]
        cur = node
        while cur is not None:
            a = idx_sp[cur]
            ancestors_dense[e, a] = 1.0
            cur = cur.up
    
    # 3) Build Recipients_mat
    Recipients_mat = (1-ancestors_dense) / torch.clamp((1-ancestors_dense).sum(dim=1, keepdim=True), min=1)

    gene_nodes = list(g_tree.traverse(traversalorder))
    G = len(gene_nodes)

    species_by_name = {}
    for node in sp_tree.traverse(traversalorder):
        assert node.name not in species_by_name, f"Duplicate species name: {node.name}"
        species_by_name[node.name] = node
    leaves_map   = torch.zeros((G, S), dtype=dtype, device=device)
    leaf_rows  = torch.zeros((G,), dtype=torch.bool, device=device)
    idx_g = {}
    g_names_by_idx = {}
    for (i, node) in enumerate(g_tree.traverse(traversalorder)):
        idx_g[node] = i
        g_names_by_idx[i] = node.name
        if node.is_leaf():
            leaf_rows[i] = True
            gene_name = node.name
            spec_name = gene_name.split('_')[0]
            if spec_name not in species_by_name:
                raise KeyError(f"Gene‐leaf {gene_name} → species {spec_name} not found.")
            e = idx_sp[species_by_name[spec_name]]
            leaves_map[i, e] = 1.0
    g_internal_mask = ~leaf_rows
    g_C1 = torch.zeros((G, G), dtype=dtype, device=device)
    g_C2 = torch.zeros((G, G), dtype=dtype, device=device)
    g_children_idx = {}
    # We need a first pass to fill the g_C1 and g_C2 matrices
    for node, i in idx_g.items():
        if not node.is_leaf():
            lc, rc = node.get_children()
            g_C1[i, idx_g[lc]] = 1.0
            g_C2[i, idx_g[rc]] = 1.0
            g_children_idx[i] = (idx_g[lc], idx_g[rc])
    g_root_idx = idx_g[g_tree]

    return {
        "s_C1":                   s_C1,
        "s_C2":                   s_C2,
        "g_C1":                   g_C1,
        "g_C2":                   g_C2,
        "ancestors_dense":        ancestors_dense,
        "Recipients_mat":         Recipients_mat,
        "sp_leaves_mask":         sp_leaves_mask,
        "g_leaves_mask":          leaf_rows,
        "idx_sp":                 idx_sp,
        "species_by_name":        species_by_name,
        "species_nodes":          species_nodes,
        "leaves_map":             leaves_map,
        "gene_nodes":             gene_nodes,
        "S":                      S,
        "G":                      G,
        "g_root_idx":             g_root_idx,
        "sp_names_by_idx":        sp_names_by_idx,
        "g_names_by_idx":         g_names_by_idx,
        "g_children_idx":         g_children_idx,
        "s_children_idx":         s_children_idx,
        "sp_internal_mask":       sp_internal_mask,
        "g_internal_mask":        g_internal_mask,
    }


######## COMPUTING PROBABILITIES OF EXTINCTION FOR A RANDOM GENE COPY ON A BRANCH OF THE SPECIES TREE ########

def E_step(E, s_C1, s_C2, Recipients_mat, p_S, p_D, p_T, p_L):
    """
    NOTE: we ignore the probability of observing leaf genes p_obs and assume
    all extant leaf genes are observed.
    - E: (n,) vector representing probability of extinction for each 
    **branch** of the species tree
    - C1, C2: (n, n) matrices giving left and right children of each node
    - Recipients_mat: (n, n) matrix giving the possible recipients for a transfer
        from each node, divided by the number of possible recipients
    - p_S, p_D, p_T, p_L: probabilities of speciation, duplication, transfer, and loss
        We can give them either as scalars or tensors of shape (n,)
    """
    E_s1 = torch.mv(s_C1, E)
    E_s2 = torch.mv(s_C2, E)
    speciation = p_S * E_s1 * E_s2
    
    duplication = p_D * E * E
    
    # We don't use sparse matrix multiplication here, so w
    Ebar = torch.mv(Recipients_mat, E)  # Ebar_e = \sum_{j} Recipients_mat_{e, j} * E_j
    transfer = p_T * E * Ebar
     
    # add p_L to represent the probability of a loss on any branch
    return speciation + duplication + transfer + p_L, E_s1, E_s2, Ebar

######## COMPUTING THE LIKELIHOOD OF SOME GENE COPY ON SOME BRANCH OF THE SPECIES TREE GIVING A GIVEN SUBTREE OF THE GENE TREE ########

def Pi_update_helper(Pi, g_C1, g_C2, s_C1, s_C2, Recipients_mat):
    # VERIFY ALL OF THESE ARE CORRECT
    # I use matmul instead of einsum to ensure compatibility with sparse matrices
    # Create unit tests for this function
    # For now we only use dense matrices, but we'll see later for sparse matrices (sparse csr format most likely)
    Pi_g1 = g_C1.mm(Pi) # = (Pi_{c_1(i), j})_{i,j}
    Pi_g2 = g_C2.mm(Pi) # = (Pi_{c_2(i), j})_{i,j}
    Pi_s1 = Pi.mm(s_C1.T)
    Pi_s2 = Pi.mm(s_C2.T)
    
    Pi_g1_s1 = g_C1.mm(Pi_s1)
    Pi_g1_s2 = g_C1.mm(Pi_s2)
    Pi_g2_s1 = g_C2.mm(Pi_s1)
    Pi_g2_s2 = g_C2.mm(Pi_s2)

    Pibar = Pi.mm(Recipients_mat.T) # Pibar_{u,e} = \sum_{j} Pi_{u, j} * Recipients_mat_{e, j}
    Pibar_g1 = g_C1.mm(Pibar)
    Pibar_g2 = g_C2.mm(Pibar)


    return Pi_g1, Pi_g2, Pi_s1, Pi_s2, \
           Pi_g1_s1, Pi_g1_s2, Pi_g2_s1, Pi_g2_s2, \
           Pibar, Pibar_g1, Pibar_g2

def Pi_update(Pi, new_Pi,
              g_C1, g_C2,
              s_C1, s_C2,
              Recipients_mat,
              leaf_map,
              E, Ebar,
              E_s1, E_s2,
              p_S, p_D, p_T):

    Pi_g1, Pi_g2, Pi_s1, Pi_s2, \
    Pi_g1_c1, Pi_g1_c2, Pi_g2_c1, Pi_g2_c2, \
    Pibar, Pibar_g1, Pibar_g2 = Pi_update_helper(Pi, g_C1, g_C2, s_C1, s_C2, Recipients_mat)

    # Duplication term: P_{g_c1, e} * P_{g_c2, e}
    D = p_D * Pi_g1 * Pi_g2
    # Speciation term: P_{g_c1, s_c1} * P_{g_c2, s_c2} + P_{g_c1, s_c2} * P_{g_c2, s_c1}

    S = p_S * (Pi_g1_c1 * Pi_g2_c2 + Pi_g1_c2 * Pi_g2_c1)
    # Transfer term: P_{g_c1, e} * Pbar_{g_c2, e} + Pbar_{g_c1, e} * P_{g_c2, e}
    T = p_T * (Pi_g1 * Pibar_g2 + Pi_g2 * Pibar_g1)
    # Duplication loss term: 2 * P_{u,e} * E_e
    D_loss = 2 * p_D * torch.einsum("ij, j -> ij", Pi, E)
    # Speciation loss term: P_{u, s_c1} * E_{s_c2} + P_{u, s_c2} * E_{s_c1}
    S_loss = p_S * (Pi_s1 * E_s2 + Pi_s2 * E_s1)
    # Transfer loss term: P_{u, e} * Ebar_e + Pbar_{u, e} * E_e 
    T_loss = p_T * (torch.einsum("ij, j -> ij", Pi, Ebar) + torch.einsum("ij, j -> ij", Pibar, E))
    # "Speciation" for leaf branches corresponds to the probability of observing a gene on a leaf branch
    # It's just given by p_S * p_obs which for us is p_S
    S_leaf_branches = p_S * leaf_map
    # Combine all terms
    new_Pi = D + T + S + D_loss + T_loss + S_loss + S_leaf_branches
    return new_Pi

######## SAMPLING A RECONCILIATION FROM THE LIKELIHOOD MATRIX ########

def sample_root_event(g_root_idx, Pi, sp_names_by_idx, g_names_by_idx):
    root_row = Pi[g_root_idx]

    # If all zeros, that is an error
    if torch.sum(root_row).item() == 0.0:
        raise RuntimeError(f"Cannot sample root mapping: all Pi[{g_root_idx}, :] are zero.")

    # We take a uniform prior on the species branches, so we directly use the likelihoods
    e0 = torch.multinomial(root_row, 1).item()

    gene_name     = g_names_by_idx[g_root_idx]
    species_name  = sp_names_by_idx[e0]
    event_record  = (gene_name, "root", species_name, "-", "-")

    return e0, event_record

def sample_next_event(u,
                      e,
                      Pi,
                      E,
                      Recipients,
                      s_children_idx,
                      g_children_idx,
                      g_leaves_mask, 
                      sp_leaves_mask,
                      p_S, p_D, p_T,
                      leaves_mapping,
                      dtype):
    
    u_is_leaf = g_leaves_mask[u]
    e_is_leaf = sp_leaves_mask[e]
    if not u_is_leaf:
        v, w = g_children_idx[u]
        # D
        D = p_D * torch.tensor([Pi[v, e] * Pi[w, e]], dtype=dtype, device=Pi.device)
        # T is a vector with terms p_T * \Pi_{v,e}*Pi_{w,j}*R[e,j]
        T = p_T * torch.cat((Pi[v, e]*Pi[w]*Recipients[e], Pi[w, e]*Pi[v]*Recipients[e]), dim=0)
    else:
        # If u is a gene leaf, the following events are not possible
        D = torch.zeros(1, dtype=dtype, device=Pi.device)
        T = torch.zeros(Recipients.shape[0], dtype=dtype, device=Pi.device)
    if not e_is_leaf:
        f, g = s_children_idx[e]
        # S
        if not u_is_leaf:
            S = p_S * torch.tensor([Pi[v, f]*Pi[w, g], Pi[v, g]*Pi[w, f]], dtype=dtype, device=Pi.device)
        else:
            S = torch.tensor([0.0, 0.0], dtype=dtype, device=Pi.device) # No S on terminal gene leaves
        SL = p_S * torch.tensor([Pi[u, f]*E[g], Pi[u, g]*E[f]], dtype=dtype, device=Pi.device)
    else:
        S = p_S * torch.tensor([leaves_mapping[u,e], 0.0], dtype=dtype, device=Pi.device)
        SL = p_S * torch.tensor([0.0, 0.0], dtype=dtype, device=Pi.device) # No SL on terminal species branches

    
    DL = 2 * p_D * torch.tensor([Pi[u, e] * E[e]], dtype=dtype, device=Pi.device)
    TL = p_T * torch.cat((Pi[u, e] * Recipients[e] * E, E[e] * Recipients[e] * Pi[u]), dim=0)

    choice_tensor = torch.cat((S, D, T, SL, DL, TL), dim=0)
    cumsum_choice_tensor = torch.cumsum(choice_tensor, dim=0)
    if cumsum_choice_tensor[-1] == 0:
        raise ValueError("No events available to sample from. Check the input parameters and the Pi matrix.")
    # print(f"Sampling next event for gene {u} on species {e}: value of S={S} leaves_mapping={leaves_mapping[u,e]}")
    S_start = 0
    S_end = S_start + S.shape[0]
    D_start = S_end
    D_end = D_start + D.shape[0]
    T_start = D_end
    T_end = T_start + T.shape[0]
    SL_start = T_end
    SL_end = SL_start + SL.shape[0]
    DL_start = SL_end
    DL_end = DL_start + DL.shape[0]
    TL_start = DL_end
    TL_end = TL_start + TL.shape[0]
    draw = torch.rand(1, dtype=dtype, device=Pi.device) * cumsum_choice_tensor[-1]
    idx = torch.searchsorted(cumsum_choice_tensor, draw, out_int32=True).item()
    if idx < S_end:
        if sp_leaves_mask[e]:
            return "leaf", u, e, ((u, e),), []
        else:
            v, w = g_children_idx[u]
            f, g = s_children_idx[e]
            if idx == 0: # S_terms[0] was Pi[v,f]*Pi[w,g]
                details = ((v, f), (w, g))
            else: # S_terms[1] was Pi[v,g]*Pi[w,f]
                details = ((v, g), (w, f))
            return "S", u, e, details, list(details)

    elif idx < D_end:
        v, w = g_children_idx[u]
        details = ((v, e), (w, e))
        return "D", u, e, details, list(details)

    elif idx < T_end:
        rel = idx - D_end
        v, w = g_children_idx[u]
        num_recipients = Recipients.shape[1] # This is S
        if rel < num_recipients: # Corresponds to T_terms[0]
            j = rel
            details = (e, (v, e), (w, j)) # donor, survivor1, survivor2
            recursions = [(v, e), (w, j)]
        else: # Corresponds to T_terms[1]
            j = rel - num_recipients
            details = (e, (w, e), (v, j))
            recursions = [(w, e), (v, j)]
        return "T", u, e, details, recursions

    elif idx < SL_end:
        rel = idx - T_end
        f, g = s_children_idx[e]
        if rel == 0: # Corresponds to SL_terms[0]
            details = ((u, f), ('lost_on', g))
            recursions = [(u, f)]
        else: # Corresponds to SL_terms[1]
            details = ((u, g), ('lost_on', f))
            recursions = [(u, g)]
        return "S-L", u, e, details, recursions
        
    elif idx < DL_end:
        details = ((u, e), ('lost_on', e))
        recursions = [(u, e)]
        return "D-L", u, e, details, recursions
        
    elif idx < TL_end: # TL
        rel = idx - DL_end
        num_recipients = Recipients.shape[1] # This is S
        if rel < num_recipients: # Corresponds to first half of TL
            j = rel
            details = (e, (u, e), ('lost_to', j)) # donor, survivor, lost_recipient
            recursions = [(u, e)]
        else: # Corresponds to second half of TL
            j = rel - num_recipients
            details = (e, (u, j), ('lost_on', e)) # donor, survivor, lost_branch
            recursions = [(u, j)]
        return "T-L", u, e, details, recursions
    else:
        raise ValueError(f"Index {idx} out of bounds for choice tensor of size {cumsum_choice_tensor.shape}. "
                         f"Choice tensor: {choice_tensor}, cumsum: {cumsum_choice_tensor}.")

def sample_reconciliation_dense(
    g_root_idx,
    g_names_by_idx,
    sp_names_by_idx,
    s_children_idx,
    g_children_idx,
    g_leaves_mask,
    sp_leaves_mask,
    leaves_mapping,
    Recipients,
    Pi,
    E,
    p_S, p_D, p_T,
):
    dtype = Pi.dtype
    events = []
    memo = {}

    def format_mapping(mapping_tuple):
        """Helper to convert integer tuples to named strings."""
        u_idx, e_idx = mapping_tuple
        return f"{g_names_by_idx[u_idx]}→{sp_names_by_idx[e_idx]}"

    def recurse(u_idx, e_idx):
        if (u_idx, e_idx) in memo:
            return
        memo[(u_idx, e_idx)] = True

        event_type, u, e, details, recursions = sample_next_event(
            u=u_idx, e=e_idx, Pi=Pi, E=E, Recipients=Recipients,
            s_children_idx=s_children_idx, g_children_idx=g_children_idx,
            g_leaves_mask=g_leaves_mask, sp_leaves_mask=sp_leaves_mask,
            p_S=p_S, p_D=p_D, p_T=p_T, leaves_mapping=leaves_mapping, dtype=dtype
        )

        u_name = g_names_by_idx[u]
        e_name = sp_names_by_idx[e]
        
        mapping_or_donor = "-"
        child1_map = "-"
        child2_map = "-"

        if event_type == "leaf":
            mapping_or_donor = format_mapping(details[0])
        elif event_type in ("S", "D"):
            mapping_or_donor = e_name
            child1_map = format_mapping(details[0])
            child2_map = format_mapping(details[1])
        elif event_type == "T":
            donor_e, map1, map2 = details
            mapping_or_donor = f"donor={sp_names_by_idx[donor_e]}"
            child1_map = format_mapping(map1)
            child2_map = format_mapping(map2)
        elif event_type in ("S-L", "D-L"):
            map1, loss_info = details
            mapping_or_donor = e_name
            child1_map = format_mapping(map1)
            loss_type, loss_e_idx = loss_info
            if loss_type == "lost_on":
                child2_map = f"copy lost on {sp_names_by_idx[loss_e_idx]}"
        elif event_type == "T-L":
            donor_e, map1, loss_info = details
            mapping_or_donor = f"donor={sp_names_by_idx[donor_e]}"
            child1_map = format_mapping(map1)
            loss_type, loss_e_idx = loss_info
            if loss_type == 'lost_to':
                child2_map = f"transfer to {sp_names_by_idx[loss_e_idx]} lost"
            else: # 'lost_on'
                child2_map = f"copy lost on {sp_names_by_idx[loss_e_idx]}"

        events.append((u_name, event_type, mapping_or_donor, child1_map, child2_map))

        for (child_u, child_e) in recursions:
            recurse(child_u, child_e)

    # --- Begin backtracking at the root ---
    e0, root_event = sample_root_event(g_root_idx, Pi, sp_names_by_idx, g_names_by_idx)
    events.append(root_event)
    recurse(g_root_idx, e0)
    
    return events

def main_fn(sp_tree_path, g_tree_path, d_rate=-10, t_rate=-1, l_rate=-10, iters=100, traversalorder="postorder", dtype=torch.float64, device=None, n_reconciliations=1, delta=None, tau=None, lambda_param=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    helpers = build_helpers(sp_tree_path, g_tree_path, device, dtype, traversalorder=traversalorder)
    G, S = helpers["G"], helpers["S"]
    
    # Use AleRax parameters if provided, otherwise use default
    if delta is not None and tau is not None and lambda_param is not None:
        # Compute event probabilities from intensities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
    else:
        # Choose global rates so that p_S + p_D + p_T + p_L = 1
        # rates_raw = torch.tensor([d_rate, t_rate, l_rate], dtype=dtype, device=device)
        # rates_pos = torch.nn.functional.softplus(rates_raw)
        rates_pos = torch.tensor([1e-10, 0.0608769, 1e-10], dtype=dtype, device=device)  # Use small positive values for d and l
        denom = 1.0 + rates_pos.sum()
        p_D = rates_pos[0] / denom
        p_T = rates_pos[1] / denom
        p_L = rates_pos[2] / denom
        p_S = 1.0 / denom

    E = torch.full((helpers["S"],), 0.0, dtype=dtype, device=device)
    for iter_e in range(iters):
        E_next, E_s1, E_s2, Ebar = E_step(E, helpers["s_C1"], helpers["s_C2"], helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next

    # Fixed-point iteration for the matrix of likelihoods
    Pi = torch.zeros((G, S), dtype=dtype, device=device)

    new_Pi = torch.empty_like(Pi)

    for iter_pi in range(iters):
        new_Pi = Pi_update(Pi=Pi,
                            new_Pi =          new_Pi,
                            g_C1   =          helpers["g_C1"],
                            g_C2   =          helpers["g_C2"],
                            s_C1   =          helpers["s_C1"],
                            s_C2   =          helpers["s_C2"],
                            Recipients_mat =  helpers["Recipients_mat"],
                            leaf_map        = helpers["leaves_map"],
                            E=E,
                            Ebar=Ebar,
                            E_s1=E_s1,
                            E_s2=E_s2,
                            p_S=p_S,
                            p_D=p_D,
                            p_T=p_T)

        Pi = new_Pi

    print("Done.  Final Π shape:", Pi.shape)
    # Print log-likelihood of the reconciliation, as the mean over species for the root of the gene tree, divided by the mean of probabilities of survival
    survival_probs = 1 - E
    log_likelihood = torch.log((Pi[helpers["g_root_idx"]]).mean()) - torch.log(survival_probs.mean())
    print(f"Log-likelihood of the reconciliation: {log_likelihood:.4f}")
    print(f"computed means: {torch.log(Pi.mean(dim=1))- torch.log(survival_probs.mean())}")


    for i in range(n_reconciliations):
        reconc_events = sample_reconciliation_dense(
                                                    g_root_idx=helpers["g_root_idx"],
                                                    g_names_by_idx=helpers["g_names_by_idx"],
                                                    sp_names_by_idx=helpers["sp_names_by_idx"],
                                                    s_children_idx=helpers["s_children_idx"],
                                                    g_children_idx=helpers["g_children_idx"],
                                                    g_leaves_mask= helpers["g_leaves_mask"],
                                                    sp_leaves_mask=helpers["sp_leaves_mask"],
                                                    leaves_mapping=helpers["leaves_map"],
                                                    Recipients=helpers["Recipients_mat"],
                                                    Pi=Pi,
                                                    E=E,
                                                    p_S=p_S, p_D=p_D, p_T=p_T,
                                                )

        from tabulate import tabulate
        headers = ["Gene node", "Event", "Mapping/Donor", "Child‐1 mapping", "Child‐2 mapping"]
        print("\n=== One sampled reconciliation ===")
        print(tabulate(reconc_events, headers=headers, tablefmt="github"))
        print("===================================\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--gene",    required=True)
    parser.add_argument("--iters",   type=int, default=50)
    parser.add_argument("--traversalorder", choices=["preorder", "postorder"], default="postorder")
    args = parser.parse_args()
    main_fn(args.species, args.gene, iters=args.iters, traversalorder=args.traversalorder)

if __name__ == "__main__":
    main()