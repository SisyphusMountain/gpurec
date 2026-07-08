"""Tests for Task 11: wiring ``GpurecConfig`` into ``GeneReconModel`` and the CLI.

Covers:
  (a) ``GeneReconModel(..., config=...)`` uses ``config.solver`` as ``solver_options`` when no
      explicit ``solver_options`` is passed, and that an explicit ``solver_options`` still wins.
  (b) A TOML file round-trips through ``--config`` + ``make_solver_options`` to a ``SolverOptions``,
      with explicit ``--pi-iters``/``--neumann-terms``/``--e-max-iter`` flags overriding it.
  (c) With neither ``--config`` nor any explicit solver flag, ``make_solver_options`` returns
      exactly today's default ``SolverOptions()`` (pi_iters=64, neumann_terms=64, e_max_iter=128).
"""
from pathlib import Path
import tempfile

from gpurec.api.solver_options import SolverOptions
from gpurec.cli import _common
from gpurec.cli.main import build_parser
from gpurec.config import GpurecConfig


def _build_cpu_model(**kwargs):
    from gpurec.api.model import GeneReconModel

    with tempfile.TemporaryDirectory() as tmp:
        species_tree = Path(tmp) / "species.nwk"
        gene_tree = Path(tmp) / "family_0.nwk"
        species_tree.write_text("(A:1,B:1)Root:1;\n", encoding="utf-8")
        gene_tree.write_text("(A_1:1,B_1:1)GeneRoot:1;\n", encoding="utf-8")
        return GeneReconModel(
            species_tree, [gene_tree], device="cpu",
            family_chunk_size=1, clade_budget=None,
            batch_packing="sequential", max_wave_size=4,
            **kwargs,
        )


# --- (a) model constructor -------------------------------------------------

def test_model_uses_config_solver_when_no_explicit_solver_options():
    cfg = GpurecConfig(solver=SolverOptions(pi_iters=32))
    model = _build_cpu_model(config=cfg)
    assert model.solver_options.pi_iters == 32


def test_model_explicit_solver_options_wins_over_config():
    cfg = GpurecConfig(solver=SolverOptions(pi_iters=32))
    explicit = SolverOptions(pi_iters=16)
    model = _build_cpu_model(config=cfg, solver_options=explicit)
    assert model.solver_options.pi_iters == 16


def test_model_config_none_reproduces_default_solver_options():
    model = _build_cpu_model()
    assert model.solver_options == SolverOptions()


# --- (b)/(c) CLI -----------------------------------------------------------

def test_default_solver_options_unchanged_without_config_or_flags():
    """No --config, no explicit flags: identical to today's hardcoded defaults."""
    args = build_parser().parse_args(
        ["reconcile", "--species", "sp.nwk", "--gene", "g.nwk"])
    opts = _common.make_solver_options(args)
    assert opts == SolverOptions()
    assert (opts.pi_iters, opts.neumann_terms, opts.e_max_iter) == (64, 64, 128)


def test_config_toml_round_trips_through_cli(tmp_path):
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text("[solver]\npi_iters = 32\nneumann_terms = 96\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["reconcile", "--species", "sp.nwk", "--gene", "g.nwk", "--config", str(toml_path)])
    opts = _common.make_solver_options(args)
    assert opts.pi_iters == 32
    assert opts.neumann_terms == 96
    # untouched field falls back to the config's (== dataclass default here) value
    assert opts.e_max_iter == 128


def test_explicit_flag_overrides_config(tmp_path):
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text("[solver]\npi_iters = 32\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["reconcile", "--species", "sp.nwk", "--gene", "g.nwk",
         "--config", str(toml_path), "--pi-iters", "16"])
    opts = _common.make_solver_options(args)
    assert opts.pi_iters == 16  # explicit flag beats config
