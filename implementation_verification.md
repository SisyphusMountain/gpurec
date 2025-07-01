# Verification of Pi_update Implementations Against AleRaxSupp.tex

## Theoretical Formulation (from AleRaxSupp.tex)

### For Internal Branches (equations 137-142):
```
Π_{e,γ} = p^S_e Σ_{γ',γ''|γ} p(γ',γ''|γ) (Π_{f,γ'} Π_{g,γ''} + Π_{f,γ''} Π_{g,γ'})
        + p^S_e (Π_{f,γ} E_g + Π_{g,γ} E_f)
        + p^D_e Σ_{γ',γ''|γ} p(γ',γ''|γ) Π_{e,γ'} Π_{e,γ''}
        + 2 p^D_e Π_{e,γ} E_e
        + p^T_e Σ_{γ',γ''|γ} p(γ',γ''|γ) (Π_{e,γ'} Π̄_{e,γ''} + Π_{e,γ''} Π̄_{e,γ'})
        + p^T_e (Π_{e,γ} Ē_e + Π̄_{e,γ} E_e)
```

### For Terminal Branches (equations 147-150):
```
Π_{e,γ} = p^S_e Π_{l,γ}
        + p^D_e Σ_{γ',γ''|γ} p(γ',γ''|γ) Π_{e,γ'} Π_{e,γ''}
        + 2 p^D_e Π_{e,γ} E_e
        + p^T_e Σ_{γ',γ''|γ} p(γ',γ''|γ) (Π_{e,γ'} Π̄_{e,γ''} + Π_{e,γ''} Π̄_{e,γ'})
        + p^T_e (Π_{e,γ} Ē_e + Π̄_{e,γ} E_e)
```

### For Leaves (equation 159):
```
Π_{l,γ} = σ_{γ,l} + (1 - σ_{γ,l})(1 - p^{obs}_l)
```

Where:
- `σ_{γ,l} = 1` if `|γ| = 1` and `γ` maps to `l`, 0 otherwise
- Default `p^{obs}_l = 1`, so leaf term simplifies to `σ_{γ,l}`

## Analysis of Pi_update_ccp (Loop Version)

### ❌ CRITICAL BUG: Terminal Branch Implementation
In lines 465-498, the terminal branch handling has a serious error:
```python
# Line 470: WRONG!
speciation_term = p_S * new_Pi[gamma_id, leaf_idx]
```
This adds the leaf speciation term TWICE because:
1. Line 440 already adds: `new_Pi += p_S * clade_species_map`
2. Line 470 uses `new_Pi` instead of the original `Pi`

According to equation 147, terminal branches should simply have `p^S_e Π_{l,γ}` where `Π_{l,γ}` is the leaf probability (already computed).

### ✅ Internal Branch Implementation
The internal branch implementation (lines 500-550) correctly implements equations 137-142.

### ⚠️ Minor Issue: E_s1 and E_s2 Parameters
The function receives `E_s1` and `E_s2` as parameters but never uses them. Instead, it computes `E[f]` and `E[g]` directly.

## Analysis of Pi_update_ccp_parallel (GPU Version - ACTUALLY USED)

### ✅ Leaf Initialization (line 578)
```python
new_Pi += p_S * clade_species_map
```
Correct implementation of equation 159 with `p^{obs}_l = 1`.

### ✅ Speciation Terms (lines 618-620)
```python
S_splits = p_S * split_probs_expanded * (
    Pi_s1_left * Pi_s2_right + Pi_s1_right * Pi_s2_left
) * internal_mask.unsqueeze(0)
```
Correctly implements the speciation splitting term for internal branches only.

### ✅ Speciation Loss Terms (line 637)
```python
S_loss = p_S * (Pi_s1 * E_s2 + Pi_s2 * E_s1) * (~sp_leaves_mask).unsqueeze(0).expand(C, -1)
```
Correctly implements `p^S_e (Π_{f,γ} E_g + Π_{g,γ} E_f)` for internal branches.

### ✅ Duplication Terms (lines 613, 636)
```python
D_splits = p_D * split_probs_expanded * Pi_left * Pi_right
D_loss = 2 * p_D * torch.einsum("ij, j -> ij", Pi, E)
```
Correctly implements both duplication terms.

### ✅ Transfer Terms (lines 623-625, 638)
```python
T_splits = p_T * split_probs_expanded * (Pi_left * Pibar_right + Pi_right * Pibar_left)
T_loss = p_T * (torch.einsum("ij, j -> ij", Pi, Ebar) + torch.einsum("ij, j -> ij", Pibar, E))
```
Correctly implements both transfer terms.

### ✅ Terminal Branch Handling
Terminal branches are handled correctly through:
1. Leaf initialization adds the speciation term
2. D, T terms apply to all branches
3. S_loss is masked out for terminal branches using `(~sp_leaves_mask)`

## Analysis of Pi_update_ccp_log (Log-Space Version)

### ✅ Leaf Initialization (lines 182-186)
```python
log_leaf_contrib = torch.where(
    clade_species_map > 0,
    log_p_S + torch.log(clade_species_map),
    float('-inf')
)
```
Correctly implements log(p_S * clade_species_map).

### ✅ Speciation Terms (lines 134-142)
Correctly implements speciation in log space using logsumexp.

### ✅ Duplication Terms (lines 94-95)
Correctly implements duplication in log space.

### ✅ Transfer Terms (lines 166-170)
Correctly implements transfer in log space.

### ✅ Loss Terms (After my fix, lines 250-280)
Now correctly implements all loss terms:
- Duplication loss: `log(2) + log(p_D) + log(Pi) + log(E)`
- Speciation loss: Properly masked for internal branches only
- Transfer loss: Both terms correctly implemented

### ✅ Final Combination (lines 285-294)
Uses logsumexp to correctly combine all contributions in log space.

## Analysis of E_step (Extinction Probability)

### Theoretical Formulation (from AleRaxSupp.tex)

For Internal branches (equation 105):
```
E_e = p^L_e + p^S_e E_f E_g + p^D_e E_e^2 + p^T_e E_e Ē_e
```

For Terminal branches (equation 110):
```
E_e = p^L_e + p^S_e E_l + p^D_e E_e^2 + p^T_e E_e Ē_e
```

For Leaves (equation 115):
```
E_l = 1 - p^{obs}_l
```

### Current Implementation

```python
def E_step(E, s_C1, s_C2, Recipients_mat, p_S, p_D, p_T, p_L):
    E_s1 = torch.mv(s_C1, E)
    E_s2 = torch.mv(s_C2, E)
    speciation = p_S * E_s1 * E_s2
    duplication = p_D * E * E
    Ebar = torch.mv(Recipients_mat, E)
    transfer = p_T * E * Ebar
    return speciation + duplication + transfer + p_L, E_s1, E_s2, Ebar
```

### ⚠️ Minor Issue: Terminal Branch Handling

The implementation treats all branches as internal branches. For terminal branches:
- Should use `p^S_e E_l` instead of `p^S_e E_f E_g`
- With default `p^{obs}_l = 1`, we have `E_l = 0`, so the speciation term should be 0

However, this may not matter in practice if:
1. The species tree structure ensures terminal branches have E_s1 = E_s2 = 0
2. Or if the implementation assumes all observed leaves have p^{obs}_l = 1

## CONCLUSION

1. **Pi_update_ccp (loop version)**: Has a CRITICAL BUG in terminal branch handling that double-counts leaf speciation.

2. **Pi_update_ccp_parallel (GPU version)**: CORRECTLY implements the theoretical formulation from AleRaxSupp.tex.

3. **Pi_update_ccp_log (log-space version)**: After the fix to add loss terms, CORRECTLY implements the theoretical formulation in log space.

The parallel and log-space versions are mathematically equivalent and both correctly implement the reconciliation model described in AleRaxSupp.tex.