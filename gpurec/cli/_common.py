"""Shared CLI helpers. Heavy imports (torch, GeneReconModel) are done lazily inside
functions so ``gpurec --help`` and argument parsing need no GPU work."""
from __future__ import annotations

import math


def bits_to_nats(x: float) -> float:
    """Convert a log2-space value to natural-log space."""
    return x * math.log(2.0)


def add_common_args(parser) -> None:
    parser.add_argument("--species", required=True, help="species tree (Newick)")
    parser.add_argument("--gene", required=True, nargs="+",
                        help="gene tree file(s), a glob, a directory, or an AleRax [FAMILIES] listfile")
    parser.add_argument("--mode", choices=["global", "specieswise", "genewise"], default="global")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--pi-iters", type=int, default=64)
    parser.add_argument("--neumann-terms", type=int, default=64)
    parser.add_argument("--e-max-iter", type=int, default=128)


def resolve_gene_trees(values) -> list:
    """Resolve --gene values (list / glob / dir / listfile) to gene-tree paths."""
    from gpurec.fit.genewise_fit import _resolve_gene_trees
    return _resolve_gene_trees(values)


def make_dtype(name: str):
    import torch
    return torch.float64 if name == "float64" else torch.float32


def make_solver_options(args):
    from gpurec.api.solver_options import SolverOptions
    return SolverOptions(e_max_iter=args.e_max_iter, pi_iters=args.pi_iters,
                         neumann_terms=args.neumann_terms)


def build_model(args):
    """Build a GeneReconModel from parsed args (lazy heavy import). Returns (model, genes)."""
    from gpurec.api.model import GeneReconModel
    genes = resolve_gene_trees(args.gene)
    model = GeneReconModel(args.species, genes, mode=args.mode, device=args.device,
                           dtype=make_dtype(args.dtype),
                           solver_options=make_solver_options(args))
    return model, genes
