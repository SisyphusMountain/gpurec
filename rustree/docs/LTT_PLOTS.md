# Lineages Through Time (LTT) Plots

Comprehensive guide to creating and analyzing LTT plots with rustree Python bindings.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Basic Usage](#basic-usage)
6. [Advanced Visualization](#advanced-visualization)
7. [Analysis and Interpretation](#analysis-and-interpretation)
8. [Performance Considerations](#performance-considerations)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)
11. [See Also](#see-also)

---

## Overview

Lineages Through Time (LTT) plots are a fundamental visualization in phylogenetics that shows how the number of lineages changes throughout evolutionary history. rustree provides built-in LTT support with:

- **Automatic data extraction** from simulated or parsed trees
- **Built-in plotting** with matplotlib integration
- **Manual data access** for custom visualization
- **Efficient computation** for trees of any size

---

## Prerequisites

### Required
- **Python** 3.8 or higher
- **rustree** Python bindings installed

### For Plotting
- **matplotlib** for visualization:
  ```bash
  pip install matplotlib
  ```

### Optional
- **numpy** for advanced analysis (usually installed with matplotlib)
- **pandas** for data manipulation (installed with rustree)
- **scipy** for smoothing and statistical analysis

---

## Quick Start

```python
import rustree
import matplotlib.pyplot as plt

# Simulate a species tree
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)

# Option 1: Use built-in plotting
tree.plot_ltt()  # Display interactively
tree.plot_ltt(filepath='/tmp/ltt.png')  # Save to file

# Option 2: Get data and plot manually
ltt = tree.get_ltt_data()
plt.plot(ltt['times'], ltt['lineages'])
plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.show()
```

---

## API Reference

### `get_ltt_data()`

Returns a dictionary with LTT data suitable for plotting.

**Returns:**
- `dict` with keys:
  - `'times'`: List of time points (from present at 0 going backward)
  - `'lineages'`: Number of lineages alive at each time point

**Example:**
```python
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.3, seed=42)
ltt = tree.get_ltt_data()

print(f"Time points: {len(ltt['times'])}")
print(f"Time range: {min(ltt['times']):.2f} to {max(ltt['times']):.2f}")
print(f"Max lineages: {max(ltt['lineages'])}")
```

### `plot_ltt(filepath=None, title=None, xlabel=None, ylabel=None)`

Creates and displays/saves an LTT plot using matplotlib.

**Parameters:**
- `filepath` (str, optional): Path to save the plot. If `None`, displays interactively.
- `title` (str, optional): Plot title. Default: `"Lineages Through Time"`
- `xlabel` (str, optional): X-axis label. Default: `"Time (past to present)"`
- `ylabel` (str, optional): Y-axis label. Default: `"Number of lineages"`

**Returns:** `None`

**Example:**
```python
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.7, seed=123)

# Display interactively
tree.plot_ltt()

# Save with custom labels
tree.plot_ltt(
    filepath='/tmp/my_ltt.png',
    title='Birth-Death Tree (λ=1.0, μ=0.7)',
    xlabel='Time before present',
    ylabel='Lineage count'
)
```

---

## Basic Usage

### Simple LTT Plot

```python
import rustree

# Simulate tree
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.3, seed=42)

# Get LTT data
ltt = tree.get_ltt_data()

# Manual plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ltt['times'], ltt['lineages'], linewidth=2)
plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.title(f'LTT Plot: {len(ltt["times"])} events')
plt.grid(True, alpha=0.3)
plt.savefig('/tmp/ltt_basic.png', dpi=300)
```

### Built-In Plotting

```python
import rustree

# Simulate tree
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)

# Quick plot (interactive)
tree.plot_ltt()

# Save to file
tree.plot_ltt(filepath='/tmp/ltt.png')

# Custom styling
tree.plot_ltt(
    filepath='/tmp/ltt_custom.png',
    title='Species Diversification',
    xlabel='Millions of years ago',
    ylabel='Species count'
)
```

---

## Advanced Visualization

### Log-Scale LTT Plots

For trees with many lineages, log-scale plots reveal patterns in early diversification:

```python
import rustree
import matplotlib.pyplot as plt

# Large tree
tree = rustree.simulate_species_tree(n=500, lambda_=1.0, mu=0.5, seed=42)
ltt = tree.get_ltt_data()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot(ltt['times'], ltt['lineages'], linewidth=2)
ax1.set_xlabel('Time (past to present)')
ax1.set_ylabel('Lineages')
ax1.set_title('Linear Scale')
ax1.grid(True, alpha=0.3)

# Log scale
ax2.semilogy(ltt['times'], ltt['lineages'], linewidth=2)
ax2.set_xlabel('Time (past to present)')
ax2.set_ylabel('Lineages (log scale)')
ax2.set_title('Log Scale')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/ltt_comparison.png', dpi=300)
```

**Performance note:** Log-scale plots are particularly useful for trees >200 species.

### Comparing Multiple Trees

```python
import rustree
import matplotlib.pyplot as plt

scenarios = [
    {"mu": 0.0, "label": "No extinction", "color": "blue"},
    {"mu": 0.5, "label": "Moderate extinction", "color": "green"},
    {"mu": 0.9, "label": "High extinction", "color": "red"},
]

plt.figure(figsize=(12, 7))

for scenario in scenarios:
    tree = rustree.simulate_species_tree(
        n=50,
        lambda_=1.0,
        mu=scenario['mu'],
        seed=42
    )
    ltt = tree.get_ltt_data()
    plt.plot(
        ltt['times'],
        ltt['lineages'],
        label=scenario['label'],
        color=scenario['color'],
        linewidth=2,
        alpha=0.8
    )

plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.title('Effect of Extinction Rate on LTT')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('/tmp/ltt_extinction_comparison.png', dpi=300)
```

### Smoothed LTT Plots

For noisy trees, apply smoothing to reveal underlying trends:

```python
import rustree
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Simulate large tree
tree = rustree.simulate_species_tree(n=200, lambda_=1.0, mu=0.6, seed=42)
ltt = tree.get_ltt_data()

# Apply Gaussian smoothing
smoothed_lineages = gaussian_filter1d(ltt['lineages'], sigma=3)

plt.figure(figsize=(12, 7))
plt.plot(ltt['times'], ltt['lineages'], alpha=0.3, label='Raw', linewidth=1)
plt.plot(ltt['times'], smoothed_lineages, linewidth=2, label='Smoothed (σ=3)', color='darkblue')
plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.title('LTT Plot with Gaussian Smoothing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/tmp/ltt_smoothed.png', dpi=300)
```

**Performance note:** Smoothing is recommended for trees >100 species with high extinction rates.

---

## Analysis and Interpretation

### Understanding LTT Patterns

**Key features:**
1. **Present (time=0):** Starts with *n* extant species
2. **Past:** Number of lineages changes at each birth/death event
3. **MRCA (root):** Ends at 1 lineage (most recent common ancestor)

**Interpretation:**
- **Steep increases:** Rapid speciation bursts
- **Plateaus:** Balanced speciation/extinction (equilibrium)
- **Decreases:** Occur when going backward through extinction events
- **Pure birth trees:** Always decrease monotonically going backward
- **Trees with extinction:** Show complex, non-monotonic patterns

### Statistical Analysis

```python
import rustree

tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
ltt = tree.get_ltt_data()

# Summary statistics
max_lineages = max(ltt['lineages'])
time_to_mrca = max(ltt['times'])
avg_lineages = sum(ltt['lineages']) / len(ltt['lineages'])

print(f"Maximum lineages: {max_lineages}")
print(f"Time to MRCA: {time_to_mrca:.3f}")
print(f"Average lineages: {avg_lineages:.1f}")

# Find time when lineages peaked
max_idx = ltt['lineages'].index(max_lineages)
peak_time = ltt['times'][max_idx]
print(f"Lineages peaked at time {peak_time:.3f}")

# Calculate diversification phases
import numpy as np
times = np.array(ltt['times'])
lineages = np.array(ltt['lineages'])

# Rate of change (derivative)
dlineages_dt = np.diff(lineages) / np.diff(times)
avg_diversification_rate = np.mean(dlineages_dt[dlineages_dt > 0])

print(f"Average diversification rate: {avg_diversification_rate:.3f} lineages/time")
```

### Example Patterns

```python
import rustree
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Pure birth - smooth exponential
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.0, seed=1)
ltt = tree.get_ltt_data()
axes[0, 0].plot(ltt['times'], ltt['lineages'], linewidth=2)
axes[0, 0].set_title('Pure Birth: Smooth exponential decay')
axes[0, 0].grid(True, alpha=0.3)

# 2. Low extinction - gentle fluctuations
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.2, seed=2)
ltt = tree.get_ltt_data()
axes[0, 1].plot(ltt['times'], ltt['lineages'], linewidth=2, color='green')
axes[0, 1].set_title('Low Extinction: Gentle fluctuations')
axes[0, 1].grid(True, alpha=0.3)

# 3. High extinction - large fluctuations
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.8, seed=3)
ltt = tree.get_ltt_data()
axes[1, 0].plot(ltt['times'], ltt['lineages'], linewidth=2, color='red')
axes[1, 0].set_title('High Extinction: Large fluctuations')
axes[1, 0].grid(True, alpha=0.3)

# 4. Log scale view
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=4)
ltt = tree.get_ltt_data()
axes[1, 1].semilogy(ltt['times'], ltt['lineages'], linewidth=2, color='purple')
axes[1, 1].set_title('Log Scale: Reveals early diversification')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('Time (past to present)')
    ax.set_ylabel('Lineages')

plt.tight_layout()
plt.savefig('/tmp/ltt_patterns.png', dpi=300)
```

---

## Performance Considerations

### Large Trees

rustree handles LTT computation efficiently for trees of any size:

```python
import rustree
import time

# Test with progressively larger trees
sizes = [100, 500, 1000, 5000]

print("LTT computation performance:")
print(f"{'Size':<10} {'Time (ms)':<15} {'Events':<10}")
print("-" * 35)

for n in sizes:
    tree = rustree.simulate_species_tree(n=n, lambda_=1.0, mu=0.5, seed=42)

    start = time.time()
    ltt = tree.get_ltt_data()
    elapsed = (time.time() - start) * 1000

    print(f"{n:<10} {elapsed:<15.2f} {len(ltt['times']):<10}")
```

**Expected performance:**
- 100 species: <1 ms
- 1,000 species: <5 ms
- 10,000 species: <50 ms
- 100,000 species: <500 ms

**Memory usage:** O(n) where n is the number of events (nodes in the tree).

### Memory-Efficient Plotting

For very large trees (>10,000 species), consider downsampling for visualization:

```python
import rustree
import matplotlib.pyplot as plt
import numpy as np

# Very large tree
tree = rustree.simulate_species_tree(n=10000, lambda_=1.0, mu=0.5, seed=42)
ltt = tree.get_ltt_data()

print(f"Original data points: {len(ltt['times'])}")

# Downsample for plotting (every 10th point)
times_sampled = ltt['times'][::10]
lineages_sampled = ltt['lineages'][::10]

print(f"Sampled data points: {len(times_sampled)}")

plt.figure(figsize=(12, 7))
plt.plot(times_sampled, lineages_sampled, linewidth=2)
plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.title(f'LTT Plot (downsampled from {len(ltt["times"])} points)')
plt.grid(True, alpha=0.3)
plt.savefig('/tmp/ltt_large.png', dpi=300)
```

**Note:** Downsampling may obscure fine-scale patterns but reduces file size and rendering time.

---

## Customization

### Styling Options

```python
import rustree
import matplotlib.pyplot as plt

tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
ltt = tree.get_ltt_data()

plt.figure(figsize=(12, 7))
plt.plot(
    ltt['times'],
    ltt['lineages'],
    linewidth=2.5,
    color='darkblue',
    linestyle='-',
    marker='o',
    markersize=3,
    markevery=10,  # Show marker every 10th point
    alpha=0.9,
    label='Lineage count'
)

plt.xlabel('Time (Mya)', fontsize=14, fontweight='bold')
plt.ylabel('Number of lineages', fontsize=14, fontweight='bold')
plt.title('Species Diversification Through Time', fontsize=16, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/ltt_styled.png', dpi=300, bbox_inches='tight')
```

### Multiple Subplot Layouts

```python
import rustree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top: Full LTT plot
ax1 = fig.add_subplot(gs[0, :])
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
ltt = tree.get_ltt_data()
ax1.plot(ltt['times'], ltt['lineages'], linewidth=2)
ax1.set_title('Full LTT Plot', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Lineages')
ax1.grid(True, alpha=0.3)

# Middle left: Recent history (last 25%)
ax2 = fig.add_subplot(gs[1, 0])
cutoff = max(ltt['times']) * 0.25
recent_mask = [t <= cutoff for t in ltt['times']]
ax2.plot(
    [t for t, m in zip(ltt['times'], recent_mask) if m],
    [l for l, m in zip(ltt['lineages'], recent_mask) if m],
    linewidth=2, color='green'
)
ax2.set_title('Recent History (last 25%)', fontsize=12)
ax2.set_xlabel('Time')
ax2.set_ylabel('Lineages')
ax2.grid(True, alpha=0.3)

# Middle right: Ancient history (first 25%)
ax3 = fig.add_subplot(gs[1, 1])
cutoff = max(ltt['times']) * 0.75
ancient_mask = [t >= cutoff for t in ltt['times']]
ax3.plot(
    [t for t, m in zip(ltt['times'], ancient_mask) if m],
    [l for l, m in zip(ltt['lineages'], ancient_mask) if m],
    linewidth=2, color='red'
)
ax3.set_title('Ancient History (first 25%)', fontsize=12)
ax3.set_xlabel('Time')
ax3.set_ylabel('Lineages')
ax3.grid(True, alpha=0.3)

# Bottom: Log scale
ax4 = fig.add_subplot(gs[2, :])
ax4.semilogy(ltt['times'], ltt['lineages'], linewidth=2, color='purple')
ax4.set_title('Log Scale View', fontsize=14, fontweight='bold')
ax4.set_xlabel('Time')
ax4.set_ylabel('Lineages (log)')
ax4.grid(True, alpha=0.3)

plt.savefig('/tmp/ltt_subplots.png', dpi=300, bbox_inches='tight')
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution:** Install matplotlib:
```bash
pip install matplotlib
```

### Issue: LTT plot is empty or shows unexpected pattern

**Cause:** Tree simulation or parsing issue

**Solution:** Verify tree structure:
```python
tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.3, seed=42)
print(f"Leaves: {tree.num_leaves()}")
print(f"Total nodes: {tree.num_nodes()}")

ltt = tree.get_ltt_data()
print(f"LTT data points: {len(ltt['times'])}")
print(f"Lineage range: {min(ltt['lineages'])} to {max(ltt['lineages'])}")
```

### Issue: Plot file is very large

**Cause:** Too many data points or high DPI

**Solutions:**
1. Reduce DPI:
   ```python
   plt.savefig('/tmp/ltt.png', dpi=150)  # Instead of 300
   ```

2. Downsample data (see [Memory-Efficient Plotting](#memory-efficient-plotting))

3. Save as vector format:
   ```python
   plt.savefig('/tmp/ltt.pdf')  # PDF scales infinitely
   ```

### Issue: `plot_ltt()` doesn't display interactively

**Cause:** Backend configuration or missing display

**Solutions:**
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'MacOSX'
import matplotlib.pyplot as plt

tree.plot_ltt()
```

Or use explicit `show()`:
```python
tree.plot_ltt()
plt.show()
```

---

## See Also

- **[Python Tutorial](PYTHON_TUTORIAL.md)** - Complete Python API documentation
- **[R Tutorial](R_TUTORIAL.md)** - R bindings with LTT analysis examples
- **[Performance Profiling](PERFORMANCE_PROFILING.md)** - Optimization and benchmarking
- **matplotlib documentation** - [https://matplotlib.org/](https://matplotlib.org/)
- **rustree GitHub repository** - Source code and examples

---

**Document Version:** 2.0
**Last Updated:** 2026-02-14
**rustree Version:** 0.1.0
**Project Root:** repository root
