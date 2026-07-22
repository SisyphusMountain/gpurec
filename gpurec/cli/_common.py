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
    parser.add_argument("--dtype", choices=["float32", "float64"], default=None,
                        help="model dtype; overrides [precision].model_dtype from --config")
    parser.add_argument("--config", default=None,
                        help="path to a GpurecConfig TOML file; [solver] and [precision] provide "
                             "defaults overridden by explicit CLI flags")
    # These default to None (not their SolverOptions values) so
    # ``make_solver_options`` can tell "not passed" apart from "passed the same value the
    # config/default already has" -- an explicitly-passed flag must win over --config.
    parser.add_argument("--pi-iters", type=int, default=None)
    parser.add_argument("--neumann-terms", type=int, default=None)
    parser.add_argument("--e-max-iter", type=int, default=None)


def resolve_gene_trees(values) -> list:
    """Resolve --gene values (list / glob / dir / listfile) to gene-tree paths."""
    from gpurec.fit.genewise_fit import _resolve_gene_trees
    return _resolve_gene_trees(values)


def make_dtype(name: str | None):
    from gpurec.config import PrecisionOptions, resolve_torch_dtype

    effective_name = PrecisionOptions().model_dtype if name is None else name
    return resolve_torch_dtype(effective_name)


def make_solver_options(args):
    """Resolve the effective ``SolverOptions`` for a parsed CLI namespace.

    Precedence (highest first): an explicitly-passed solver flag > the matching
    field of ``--config``'s ``[solver]`` table > the
    hardcoded ``SolverOptions`` default (``pi_iters=64``, ``neumann_terms=64``, ``e_max_iter=128``).
    With neither ``--config`` nor an explicit solver flag, this returns ``SolverOptions()``
    unchanged -- identical to today.
    """
    from dataclasses import replace

    from gpurec.config import load_config

    base = load_config(getattr(args, "config", None)).solver
    overrides = {}
    if (pi_iters := getattr(args, "pi_iters", None)) is not None:
        overrides["pi_iters"] = pi_iters
    if (neumann_terms := getattr(args, "neumann_terms", None)) is not None:
        overrides["neumann_terms"] = neumann_terms
    if (e_max_iter := getattr(args, "e_max_iter", None)) is not None:
        overrides["e_max_iter"] = e_max_iter
    return replace(base, **overrides) if overrides else base


def build_model(args):
    """Build a GeneReconModel from parsed args (lazy heavy import). Returns (model, genes)."""
    from gpurec.api.model import GeneReconModel
    from gpurec.config import load_config

    genes = resolve_gene_trees(args.gene)
    config = load_config(getattr(args, "config", None))
    model = GeneReconModel(args.species, genes, mode=args.mode, device=args.device,
                           dtype=(None if args.dtype is None else make_dtype(args.dtype)),
                           config=config,
                           solver_options=make_solver_options(args))
    return model, genes
