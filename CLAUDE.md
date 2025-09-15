# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated reconciliation tool for phylogenetic trees, implementing the ALE (Amalgamated Likelihood Estimation) algorithm using PyTorch. The project reconciles gene trees with species trees using probabilistic models that account for gene duplication (D), transfer (T), loss (L), and speciation (S) events.

## Key Commands

### Running the main reconciliation algorithm

**Rooted gene tree (original):**
```bash
python matmul_ale.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --iters 50
```

**Unrooted gene tree implementations:**
```bash
# ALE-style approach (RECOMMENDED - matches ALE's pun() function)
python matmul_ale_ale_style.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1

# Simple averaging approach
python matmul_ale_simple_unrooted.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1

# Physical tree rerooting approach  
python matmul_ale_unrooted_v2.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1

# Clade-based approach (computationally intensive)
python matmul_ale_clade.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1
```

### Running original ALE for comparison
Follow the steps in `run_ale.md`:
1. `ALEobserve sp.nwk`
2. `ALEobserve g.nwk`
3. `ALEml_undated sp.nwk sp.nwk.ale output_species_tree=y sample=0 delta=0 tau=0 lambda=0 seed=42`
4. `mv sp.nwk_sp.nwk.ale.spTree sp`
5. `ALEml_undated sp g.nwk.ale tau=1e-1 delta=1e-1 lambda=1e-1`

## Architecture Overview

### Core Algorithm Implementation
The main implementation in `matmul_ale.py` consists of:

1. **Tree Helper Construction** (`build_helpers`): Builds matrix representations of species and gene trees including:
   - Children matrices (C1, C2) for left/right children
   - Ancestors matrix for species tree relationships
   - Recipients matrix for transfer event targets
   - Leaf mappings between gene and species trees

2. **Extinction Probability Computation** (`E_step`): Fixed-point iteration to compute extinction probabilities for each species branch, considering all event types (S, D, T, L).

3. **Likelihood Matrix Computation** (`Pi_update`): Computes the likelihood matrix Pi[g,s] representing the probability of observing gene subtree g on species branch s. Uses matrix multiplications for efficiency.

4. **Reconciliation Sampling** (`sample_reconciliation_dense`): Samples a specific reconciliation from the computed likelihood matrix using the posterior probabilities.

### Key Data Structures
- **Pi matrix** (G×S): Likelihood of each gene node on each species branch
- **E vector** (S,): Extinction probabilities for each species branch
- **Event probabilities**: p_S (speciation), p_D (duplication), p_T (transfer), p_L (loss)

## Dependencies
- `ete3`: For phylogenetic tree parsing and manipulation
- `torch`: For GPU-accelerated matrix operations
- `tabulate`: For formatted output display
- `numpy`: For numerical operations

## Test Data
- `test_trees_1/`: Simple test case with matching gene and species trees
- `test_trees_2/`: More complex test case with additional transfer events

## Theory Reference
The implementation is based on the UndatedDTL model described in `AleRaxSupp.tex`. Key theoretical components:

### UndatedDTL Model
The model describes gene evolution with four event types:
- **Duplication (D)**: Gene duplicates on the same branch with probability p_D
- **Transfer (T)**: Gene transfers to a non-ancestral branch with probability p_T
- **Loss (L)**: Gene is lost with probability p_L
- **Speciation (S)**: Gene speciates with probability p_S = 1 - (p_D + p_T + p_L)

Event probabilities are derived from intensity parameters (δ, τ, λ):
- p_D = δ/(1 + δ + τ + λ)
- p_T = τ/(1 + δ + τ + λ)
- p_L = λ/(1 + δ + τ + λ)
- p_S = 1/(1 + δ + τ + λ)

### Key Equations
1. **Extinction probability (E_e)**: Fixed-point iteration computing probability of extinction on each branch
2. **Likelihood matrix (Π_e,γ)**: Probability that lineage leading to clade γ exists on branch e
3. **Joint likelihood**: P(A|S) ≈ Σ_e p_e^O Π_e,Γ / Σ_e p_e^O (1 - E_e)

The current implementation includes both single gene tree reconciliation and CCP-based reconciliation for handling gene tree uncertainty.

## Conditional Clade Probabilities (CCPs)

### Overview
The CCP implementation in `matmul_ale_ccp.py` extends the basic reconciliation algorithm to handle gene tree uncertainty by working with distributions of clades rather than single fixed gene trees. This follows the theoretical framework described in Section 4 of AleRaxSupp.tex.

### Key Concepts
- **Clades**: Represented as sets of leaf names, each clade γ corresponds to a subtree in potential gene trees
- **Clade Splits**: Each internal clade can split into left/right child clades with associated frequencies
- **Conditional Probabilities**: CCPs represent the probability of observing specific clade splits given the gene sequence data

### Matrix-Based CCP Implementation
The goal is to extend the efficient matrix operations from `matmul_ale.py` to work with clade distributions:

#### CCP Data Structures
- **Pi_ccp matrix** (C×S): Likelihood of each clade c on species branch s
- **Clade splits**: Sparse representation of how clades can decompose into child clades
- **Split frequencies**: Probability weights for different possible splits of each clade

#### Efficient Matrix Operations
The CCP reconciliation uses sparse matrix multiplication to handle:
1. **Clade relationships**: Parent-child relationships between clades via split matrices
2. **Event computation**: Duplication, transfer, speciation, and loss events across all possible clade splits
3. **Likelihood updates**: Parallel computation across all clade-species branch combinations

#### Matrix Representation
- **M_left, M_right**: Sparse matrices encoding left/right child relationships for clade splits
- **M_parent**: Sparse matrix for parent-child clade relationships  
- **Split probabilities**: Weights for different decompositions of each clade

### Running CCP Reconciliation
```bash
# Extract CCPs from single gene tree and run reconciliation
python matmul_ale_ccp.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1

# Log-space CCP reconciliation (recommended for numerical stability)
python matmul_ale_ccp_log.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --delta 0.1 --tau 0.1 --lambda 0.1
```

### Implementation Strategy
The CCP approach maintains the computational efficiency of the single-tree version by:
1. Using sparse matrices to represent clade split relationships
2. Vectorizing likelihood computations across all clades simultaneously  
3. Leveraging PyTorch's GPU acceleration for large-scale matrix operations
4. Following the same fixed-point iteration structure as the original algorithm

This enables scaling to large gene tree distributions while preserving the mathematical rigor of the ALE reconciliation model.

### Log-Space CCP Implementation

The log-space implementation in `matmul_ale_ccp_log.py` addresses numerical underflow issues that can occur with the standard CCP reconciliation, especially for large trees or extreme parameter values.

#### Key Features
- **Log-space computations**: All probabilities are maintained in log space to avoid underflow
- **Numerical stability**: Uses `logsumexp` for addition operations in log space
- **Efficient sparse operations**: Maintains efficient sparse matrix operations while working in log space
- **Gradient-friendly**: The log-space implementation is more suitable for automatic differentiation

#### Technical Details
The `Pi_update_ccp_log` function implements the core likelihood update in log space:
1. **Event probabilities**: Converted to log space with proper handling of zeros (log(0) = -inf)
2. **Duplication events**: log(Pi_left * Pi_right * p_D) = log(Pi_left) + log(Pi_right) + log(p_D)
3. **Speciation events**: Uses logsumexp for combining alternative speciation scenarios
4. **Transfer events**: Carefully handles matrix multiplications involving log probabilities
5. **Scatter operations**: Uses the log-sum-exp trick for numerically stable aggregation

#### Numerical Stability Considerations
- Initializes leaf probabilities in log space based on clade-species mappings
- Uses `torch.logsumexp` for stable addition of log probabilities
- Handles -inf values properly throughout the computation
- Includes NaN detection and debugging output for troubleshooting

## Parameter Optimization

### Overview
The project includes parameter optimization capabilities to find optimal values for the duplication (δ), transfer (τ), and loss (λ) rates by maximizing the reconciliation likelihood.

### Implementations

#### Stable Optimization (`matmul_ale_ccp_optimize_stable.py`)
The stable implementation uses numerical tricks to avoid gradient issues:

```bash
python matmul_ale_ccp_optimize_stable.py --species test_trees_1/sp.nwk --gene test_trees_1/g.nwk --lr 0.01 --epochs 100
```

**Key Features:**
- **Log parameterization**: Uses log(rate) parameterization with exp transformation to ensure positivity
- **Hybrid computation**: Works in linear space with clamping to avoid underflow
- **Gradient detachment**: Computes E and most Pi iterations without gradients for efficiency
- **Stable operations**: Custom `stable_logsumexp` function that properly handles -inf values
- **Gradient clipping**: Prevents gradient explosion with max norm clipping

**Optimization Algorithm:**
1. Initialize parameters with log transformation
2. Compute extinction probabilities E (detached from gradient graph)
3. Run Pi fixed-point iterations (only last 2 iterations tracked for gradients)
4. Compute log-likelihood using stable logsumexp
5. Backpropagate gradients through the computational graph
6. Update parameters using Adam optimizer with gradient clipping

#### Original Log-Space Optimization (`matmul_ale_ccp_optimize.py`)
The original implementation attempts full log-space differentiation but can suffer from numerical instability:
- Uses softplus parameterization for rates
- Attempts to differentiate through all fixed-point iterations
- Can experience NaN gradients due to log(0) and inf-inf operations

### Usage Example
```python
# Run optimization with custom parameters
python matmul_ale_ccp_optimize_stable.py \
  --species species_tree.nwk \
  --gene gene_tree.nwk \
  --init-delta 0.1 \
  --init-tau 0.05 \
  --init-lambda 0.1 \
  --lr 0.01 \
  --epochs 200 \
  --e-iters 30 \
  --pi-iters 30
```

### Output
The optimization produces:
- Best parameter values that maximize the log-likelihood
- History of parameter values and log-likelihood across epochs
- JSON output file with complete optimization results

## Implementation Notes and Design Decisions

### Log-Space CCP Implementation (`matmul_ale_ccp_log.py`)

#### Custom Gradient Implementation
- **ScatterLogSumExp**: Custom autograd function for scatter operations to avoid NaN gradients that appeared with standard PyTorch operations
- Implements safe backward pass with softmax-style weights
- Verified correct using PyTorch's `check_grad` function
- Handles edge cases where values are -inf or zero

#### Mixed Log/Linear Space Operations
- **Transfer computation**: Uses linear space matrix multiplication with numerical stability tricks (lines 304-309)
  - Removes max element before exponentiation to prevent overflow
  - Converts back to log space after matrix multiplication
  - Trade-off: Simpler implementation vs. full log-space computation
  - Future work: Could implement full log-space matrix multiplication using log-sum-exp over pairwise products

#### Fixed-Point Iterations
- **Current approach**: Fixed 100 iterations, empirically sufficient for convergence
- **E_step**: No convergence check (runs all iterations)
- **Pi_update**: Includes convergence check with tolerance 1e-10
- **Future improvements**: 
  - Implement implicit differentiation for fixed-point equations (planned)
  - Consider Anderson acceleration for faster convergence
  - Add adaptive iteration counts based on convergence metrics

#### Numerical Stability
- **Parameter boundaries**: Keep parameters above threshold (e.g., 1e-10) in linear space
- **Matrix properties**: After convergence, no zero values expected in Pi matrix
- **Fixed-point uniqueness**: Unique fixed point exists on open unit cube (theoretical guarantee)
- **Edge cases handled**:
  - Zero split probabilities → -inf in log space
  - Empty clade-species mappings
  - Extremely small parameter values

#### Performance Considerations
- **Sparse vs Dense**: Currently uses mixed approach
  - Dense matrices for Pi and E computations
  - Sparse representations for clade split relationships
  - Benchmarking needed to determine optimal crossover point
- **Memory scaling**: Not yet profiled for large trees (thousands of clades)
- **GPU utilization**: Leverages PyTorch's automatic GPU acceleration

### Development Workflow Priorities

1. **Code refactoring**: Improve clarity and maintainability of `matmul_ale_ccp_log.py`
2. **Testing infrastructure**: Add unit tests for gradient correctness and numerical stability
3. **Performance profiling**: Benchmark sparse vs dense operations, memory usage
4. **Implicit differentiation**: Implement for more efficient gradient computation through fixed points
5. **Parameter constraints**: Add automatic parameter bounding and regularization