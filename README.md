# gpurec

GPU-accelerated gene tree / species tree reconciliation using the DTL (Duplication, Transfer, Loss) model. A reimplementation of the [AleRax](https://github.com/BenoitMorel/AleRax) likelihood computation on GPU via PyTorch and Triton.

## Overview

gpurec computes the likelihood of observed gene trees given a species tree and DTL event rates. It supports:

- **Fixed-point likelihood computation**: Iterative solvers for the extinction probability vector (E) and the clade-species probability matrix (Pi)
- **Batched computation**: Multiple gene families processed simultaneously on GPU
- **Parameter optimization**: Implicit differentiation of the fixed-point equations via VJP (vector-Jacobian products) with Neumann series / CG / GMRES solvers
- **Stochastic backtracking**: Sampling reconciliation scenarios from the posterior (C++ extension)
- **Custom Triton kernels**: Segmented logsumexp and log-matmul for GPU acceleration

## Installation

```bash
pip install -e .

# With Triton GPU kernel support
pip install -e ".[triton]"

# With development tools
pip install -e ".[dev]"
```

Requires a C++ compiler for JIT compilation of preprocessing extensions.

## Quick start

### CLI

```bash
# Compute log-likelihood for a gene tree given a species tree
gpurec --species species.nwk --gene gene.nwk --delta 0.01 --tau 0.01 --lambda 0.01

# Multiple gene families
gpurec --species species.nwk --gene family1.nwk family2.nwk family3.nwk

# Use GPU with float64 precision
gpurec --species species.nwk --gene gene.nwk --device cuda --dtype float64
```

### Python API

```python
import torch
from src.core.model import GeneDataset

# Load data
dataset = GeneDataset(
    species_tree_path="species.nwk",
    gene_tree_paths=["gene1.nwk", "gene2.nwk"],
    genewise=False,       # shared DTL rates across families
    specieswise=False,    # uniform rates across species branches
    pairwise=False,       # no pairwise transfer coefficients
    dtype=torch.float64,
)

# Set DTL rates
dataset.set_params(0, D=0.01, T=0.01, L=0.01)

# Compute likelihood for a single family
result = dataset.compute_likelihood(0)
print(f"Log-likelihood: {result['log_likelihood']}")

# Batch computation across all families
log_liks = dataset.compute_likelihood_batch()
```

### Parameter optimization

```python
from src.optimization.simple_optim import optimize_theta

result = optimize_theta(
    species_tree_path="species.nwk",
    gene_tree_path="gene.nwk",
    theta_init=torch.tensor([-5.0, -5.0, -5.0]),  # log(D), log(L), log(T)
    steps=200,
    lr=0.2,
    dtype=torch.float64,
)
print(f"Optimized rates: {result['rates']}")
print(f"Log-likelihood: {result['log_likelihood']}")
```

## Algorithm

The algorithm follows the ALE (Amalgamated Likelihood Estimation) framework:

1. **Preprocessing** (C++ JIT-compiled): Parse Newick trees, enumerate clades via CCP (Conditional Clade Probabilities), build species tree index structures

2. **E fixed-point**: Solve for extinction probabilities E(s) at each species branch s. E(s) is the probability that a gene lineage at branch s leaves no observable descendant. This is a fixed-point equation involving speciation, duplication, transfer, and loss events.

3. **Pi fixed-point**: Solve for Pi(gamma, s) -- the probability that clade gamma reconciles at species branch s. Uses DTS terms (Duplication/Transfer/Speciation contributions) and a segmented logsumexp reduction over clade splits.

4. **Log-likelihood**: L = log( sum_s Pi(root, s) / |S| ) - log( 1 - mean(E) )

5. **Gradient computation**: Uses implicit differentiation of the fixed-point equations. Since (Pi*, E*) = H(Pi*, E*, theta), the gradient dL/dtheta is obtained via:
   - Solve (I - J_x H)^T lambda = dL/dx using Neumann series
   - Compute dL/dtheta = (J_theta H)^T lambda

## Parameter modes

The DTL rates can be parameterized at different granularities:

| Mode | `genewise` | `specieswise` | Description |
|------|------------|---------------|-------------|
| Global | False | False | Single (D, T, L) for all families and branches |
| Per-species | False | True | (D, T, L) varies per species branch |
| Per-gene | True | False | (D, T, L) varies per gene family |
| Full | True | True | (D, T, L) varies per family x branch |

The `pairwise` flag additionally allows species-pair-specific transfer coefficients.

## Project structure

```
gpurec/
├── pyproject.toml          # Package configuration
├── README.md
├── pytest.ini
├── src/
│   ├── core/
│   │   ├── model.py            # GeneDataset: data loading and likelihood API
│   │   ├── likelihood.py       # E/Pi fixed-point solvers
│   │   ├── terms.py            # DTS term computation (speciation/duplication/transfer)
│   │   ├── extract_parameters.py  # theta -> (pS, pD, pL, transfer_mat) conversion
│   │   ├── batching.py         # Multi-family collation for batched likelihood
│   │   ├── kernels/            # Triton GPU kernels
│   │   │   ├── scatter_lse.py      # Segmented logsumexp
│   │   │   ├── seg_log_matmul.py   # Segmented log-space matrix multiply
│   │   │   └── lse.py              # Fixed-arity logsumexp
│   │   └── cpp/                # C++ preprocessing extensions (JIT-compiled)
│   │       ├── preprocess.cpp
│   │       ├── tree_utils.cpp/.hpp
│   │       └── clade_utils.cpp/.hpp
│   ├── optimization/
│   │   ├── optim.py            # VJP-based implicit gradient computation
│   │   ├── simple_optim.py     # Full optimization loop (SGD/Adam/L-BFGS)
│   │   └── theta_optimizer.py  # CG/GMRES-based implicit gradient optimizer
│   ├── reconciliation/
│   │   ├── sampler.py          # Stochastic backtracking (calls C++ extension)
│   │   ├── summaries.py        # Output writers (transfer frequencies, events)
│   │   └── ext_loader.py       # C++ backtracking extension loader
│   ├── cli/
│   │   └── reconcile.py        # Command-line interface
│   └── utils/
│       └── debug.py
├── scripts/                    # Standalone experiment scripts
├── tests/                      # Test suite
├── docs/                       # Design notes and references
└── extra/                      # Reference material (gitignored)
```

## Tests

```bash
pytest tests/
```

## References

- Szollosi et al. (2013). "Efficient Exploration of the Space of Reconciled Gene Trees." *Systematic Biology*.
- Morel et al. (2020). "GeneRax: A Tool for Species-Tree-Aware Maximum Likelihood-Based Gene Family Tree Inference under Gene Duplication, Transfer, and Loss." *Molecular Biology and Evolution*.
- AleRax: https://github.com/BenoitMorel/AleRax
