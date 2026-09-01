# gpurec glossary

This is the stable vocabulary used by gpurec's code and documentation.

- **Reconciliation:** an explanation of a gene-family history inside a species
  tree using speciation, duplication, transfer, and loss events.
- **Clade:** a set of gene copies that can form a subtree. gpurec integrates
  over conditional clade probabilities rather than requiring one fixed gene
  tree.
- **Wave:** one dependency level of the species tree, scheduled as a batch for
  a kernel launch.
- **Extinction step (`E`, `Ebar`):** the fixed-point computation of the
  probability that a lineage leaves no observed descendant; `Ebar` is its
  transfer-weighted counterpart.
- **Reconciliation state (`Pi`, `Pibar`):** per-clade, per-species likelihoods
  propagated through the species-tree waves; `Pibar` is the transfer-weighted
  counterpart.
- **DTS reduction:** the gene-split reduction combining duplication, transfer,
  and speciation contributions for two child clades.
- **`theta`:** base-2 logarithms of the fitted rates. Tensor column order is
  duplication, loss, transfer (`D`, `L`, `T`).
- **Bits and nats:** the library's negative log-likelihood is in bits (base 2).
  User-facing AleRax comparisons commonly use nats (natural logarithms), with
  `nats = bits * ln(2)`.
- **VJP:** vector-Jacobian product, used by reverse-mode differentiation.
- **JVP:** Jacobian-vector product, used by forward-mode differentiation.
- **HVP:** Hessian-vector product, used by curvature and Newton methods.

## Derivative suffixes

The kernel families have four related passes:

| Form | Suffix | Purpose |
|---|---|---|
| Value | none | likelihood/fixed-point value |
| Reverse derivative | `_backward` | gradient/VJP |
| Forward derivative | `_tangent` | JVP |
| Second order | `_so` | curvature/HVP contribution |

This pattern applies to the extinction, wave-propagation, and DTS-reduction
parts of the likelihood.
