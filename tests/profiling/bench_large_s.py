"""Benchmark wave on large species trees (S ~ 10K-20K)."""

import torch
import time
import glob

from src.core.model import GeneDataset


def bench(n_families, sp_path, gene_files, chunk_size=20):
    files = gene_files[:n_families]
    print(f"\n=== {n_families} families ===")

    t0 = time.time()
    ds = GeneDataset(
        sp_path, files,
        genewise=False, specieswise=False, pairwise=False,
        dtype=torch.float32, device=torch.device("cuda"),
    )
    t_init = time.time() - t0
    print(f"  GeneDataset init: {t_init:.1f}s, S={ds.S}, families={len(ds.families)}")

    # Wave
    torch.cuda.synchronize()
    t1 = time.time()
    logLs = ds.compute_likelihood_batch(
        chunk_size=chunk_size,
        max_iters_Pi=200, tol_Pi=1e-3,
    )
    torch.cuda.synchronize()
    t2 = time.time()
    elapsed = t2 - t1
    print(f"  Wave: {elapsed:.1f}s total, {elapsed / n_families * 1000:.1f} ms/family")
    print(f"  logL range: [{min(logLs):.2f}, {max(logLs):.2f}]")
    return logLs


if __name__ == "__main__":
    sp_path = "tests/data/test_trees_10000/sp.nwk"
    gene_files = sorted(glob.glob("tests/data/test_trees_10000/gene_*.nwk"))
    print(f"Available gene trees: {len(gene_files)}")

    # Start small — S=19999 is very memory-heavy
    # Transfer matrix alone = S^2 * 4 bytes = 1.6 GB
    for n in [2, 5, 10]:
        if len(gene_files) >= n:
            bench(n, sp_path, gene_files, chunk_size=2)
