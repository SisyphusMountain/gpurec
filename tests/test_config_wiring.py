"""Tests for Task 11: wiring ``GpurecConfig`` into ``GeneReconModel`` and the CLI.

Covers:
  (a) ``GeneReconModel(..., config=...)`` uses ``config.solver`` as ``solver_options`` when no
      explicit ``solver_options`` is passed, and that an explicit ``solver_options`` still wins.
  (b) A TOML file round-trips through ``--config`` + ``make_solver_options`` to a ``SolverOptions``,
      with explicit ``--pi-iters``/``--neumann-terms``/``--e-max-iter`` flags overriding it.
  (c) With neither ``--config`` nor any explicit solver flag, ``make_solver_options`` returns
      exactly today's default ``SolverOptions()`` (pi_iters=64, neumann_terms=64, e_max_iter=128).

Also covers Task 13: threading ``GpurecConfig`` into the three fit entry points
(``fit_genewise``/``optimize``/``map_cv``) -- see the section near the bottom of this file.
"""
from pathlib import Path
import tempfile

import pytest
import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.cli import _common
from gpurec.cli.main import build_parser
from gpurec.config import GpurecConfig, PrecisionOptions
from gpurec.config.rates import RateBounds
from gpurec.solver.penalties import OriginationPenalty, PenaltyOptions


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


def test_model_uses_config_precision_and_copies_it_to_batch_statics():
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float64")
    )
    model = _build_cpu_model(config=cfg)

    assert model.theta.dtype == torch.float64
    assert model.dtype == torch.float64
    assert model.accumulator_dtype == torch.float64
    assert model.precision_options == cfg.precision
    assert all(static.precision_options == cfg.precision for static in model.batch_statics)
    assert all(static.accumulator_dtype == torch.float64 for static in model.batch_statics)
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float64
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float64


def test_model_float32_accumulator_reaches_preprocessing_statics():
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float32", accumulator_dtype="float32")
    )
    model = _build_cpu_model(config=cfg)

    assert model.accumulator_dtype == torch.float32
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float32
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float32
                assert set(meta["_log_split_probs_by_dtype"]) == {
                    torch.float32,
                    torch.float64,
                }


def test_model_mixed_precision_materializes_each_preprocessing_value_in_its_owner_dtype():
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float32", accumulator_dtype="float64")
    )
    model = _build_cpu_model(config=cfg)

    # Transfer normalization metadata belongs to accumulator-domain reductions.
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float64
    # Split probabilities are direct dense-kernel inputs and must arrive in the
    # model dtype rather than being narrowed inside every Triton program.
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float32


def test_model_explicit_dtype_overrides_config_precision():
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float64")
    )
    model = _build_cpu_model(config=cfg, dtype=torch.float32)

    assert model.theta.dtype == torch.float32
    assert model.precision_options.model_dtype == "float32"
    # Resolving an explicit override must not mutate the caller's config.
    assert cfg.precision.model_dtype == "float64"


def test_model_rejects_accumulator_narrower_than_effective_model_dtype():
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float32")
    )
    with pytest.raises(ValueError, match="at least as wide"):
        _build_cpu_model(config=cfg)


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


def test_cli_config_model_dtype_is_effective_and_explicit_dtype_wins(tmp_path):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "family_0.nwk"
    config = tmp_path / "precision.toml"
    species.write_text("(A:1,B:1)Root:1;\n", encoding="utf-8")
    gene.write_text("(A_1:1,B_1:1)GeneRoot:1;\n", encoding="utf-8")
    config.write_text(
        '[precision]\nmodel_dtype = "float64"\naccumulator_dtype = "float64"\n',
        encoding="utf-8",
    )

    common = [
        "reconcile",
        "--species",
        str(species),
        "--gene",
        str(gene),
        "--device",
        "cpu",
        "--config",
        str(config),
    ]
    configured, _ = _common.build_model(build_parser().parse_args(common))
    overridden, _ = _common.build_model(
        build_parser().parse_args([*common, "--dtype", "float32"])
    )

    assert configured.theta.dtype == torch.float64
    assert overridden.theta.dtype == torch.float32
    assert configured.accumulator_dtype == overridden.accumulator_dtype == torch.float64


# ================================================================================================
# Task 13: thread GpurecConfig into fit_genewise / optimize / map_cv
#
# Each helper below monkeypatches the ONE heavy call (GeneReconModel construction / first_order /
# the model-building + inner-fit helpers) so the wiring can be asserted WITHOUT a CUDA/Triton
# solve -- these entry points are otherwise expensive end-to-end recipes. The golden/HVP gate
# (test_optim_golden.py / test_genewise_hvp.py) is the real end-to-end value-additive proof; these
# are fast unit-level checks of the resolution precedence itself.
# ================================================================================================

# --- fit_genewise ----------------------------------------------------------

def _run_fit_genewise_capture(monkeypatch, **kwargs):
    """Run ``fit_genewise`` with a stubbed ``GeneReconModel`` (never actually touched -- adam_steps=0,
    max_iter=0, certify=False mean the tier loop builds a model but never calls into it) and a spy
    on ``RateBounds`` (a plain dataclass; wrapping its constructor just records the resolved
    min_rate/max_rate without changing behavior). ``parse_families`` (which reads every .ale file
    once per fit) is stubbed for the same reason as the model: these paths do not exist on disk.
    Returns ``(solver_options_used, bounds_used)``.
    """
    import gpurec.fit.genewise_fit as gf

    captured = {}

    class _CaptureModel:
        def __init__(self, *args, **mkwargs):
            captured["kwargs"] = mkwargs
            self.receiver_weights = torch.zeros(1)
            # fit_genewise reads the per-family clade counts to decide when a re-plan pays off
            self.families = [{"C": 1}]

    real_rate_bounds = gf.RateBounds
    bounds_seen = []

    def _spy_bounds(*args, **bkwargs):
        rb = real_rate_bounds(*args, **bkwargs)
        bounds_seen.append(rb)
        return rb

    monkeypatch.setattr(gf, "GeneReconModel", _CaptureModel)
    monkeypatch.setattr(gf, "parse_families", lambda *_a, **_k: object())
    monkeypatch.setattr(gf, "RateBounds", _spy_bounds)
    gf.fit_genewise(
        "sp.nwk", ["g.nwk"], device="cpu", adam_steps=0, pi_tiers=(16,),
        max_iter=0, min_drop=32, rebuild_frac=0.25, hessian_refresh=15,
        init_curvature="exact", certify=False,
        certify_curvature=False, verbose=False, **kwargs,
    init_log2_rates=(0.0, 0.0, 0.0),
    )
    return captured["kwargs"]["solver_options"], bounds_seen[0]


def test_fit_genewise_config_solver_field_threaded(monkeypatch):
    cfg = GpurecConfig(solver=SolverOptions(e_max_iter=999))
    so, _bounds = _run_fit_genewise_capture(monkeypatch, config=cfg)
    assert so.e_max_iter == 999
    # pi_iters/neumann_terms are NOT part of _BASE_SOLVER's key subset -- fit_genewise's per-tier
    # `sopts()` always overrides them from `pi_tiers`/`neu_opt`, so config.solver.pi_iters has no
    # effect here (this mirrors the map_cv test below, where pi_iters IS threaded).
    assert so.pi_iters == 16


def test_fit_genewise_config_rates_sets_effective_min_rate(monkeypatch):
    cfg = GpurecConfig(rates=RateBounds(min_rate=1e-4))
    _so, bounds = _run_fit_genewise_capture(monkeypatch, config=cfg)
    assert bounds.min_rate == 1e-4
    # max_rate is ALSO substituted from config.rates (RateBounds(min_rate=1e-4)'s own max_rate,
    # i.e. None) since it too is still at its genewise-preset signature default -- config.rates is
    # threaded field-by-field, each guarded independently, not "only the field the caller set".
    assert bounds.max_rate is None


def test_fit_genewise_explicit_min_rate_beats_config(monkeypatch):
    cfg = GpurecConfig(rates=RateBounds(min_rate=1e-4))
    _so, bounds = _run_fit_genewise_capture(monkeypatch, config=cfg, min_rate=5e-3)
    assert bounds.min_rate == 5e-3


def test_fit_genewise_config_none_reproduces_today_defaults(monkeypatch):
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    so, bounds = _run_fit_genewise_capture(monkeypatch)
    assert so == SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    assert bounds.min_rate == RateBounds.genewise().min_rate
    assert bounds.max_rate == RateBounds.genewise().max_rate


# --- optimize ----------------------------------------------------------

def _run_optimize_capture(monkeypatch, **kwargs):
    """Run ``optimize`` with a stubbed ``first_order`` (returns ``theta0`` unchanged) and
    ``polish_mode="none"`` (skips the Newton polish stage entirely) so no CUDA/Triton solve ever
    runs. Returns the ``origination_penalty`` ``first_order`` was actually called with.
    """
    import gpurec.fit.optimize as opt_mod

    captured = {}

    def _fake_first_order(batch_statics, theta0, receiver_weights, **fkwargs):
        captured["origination_penalty"] = fkwargs.get("origination_penalty")
        return theta0, [], None

    monkeypatch.setattr(opt_mod, "first_order", _fake_first_order)
    opt_mod.optimize(None, torch.zeros(3), torch.zeros(3), verbose=False, polish_mode="none", **kwargs)
    return captured["origination_penalty"]


def test_optimize_config_sets_origination_penalty(monkeypatch):
    op = OriginationPenalty(l2=5.0)
    cfg = GpurecConfig(regularizer=PenaltyOptions(origination=op))
    got = _run_optimize_capture(monkeypatch, config=cfg)
    assert got is op


def test_optimize_explicit_origination_penalty_wins_over_config(monkeypatch):
    op_cfg = OriginationPenalty(l2=5.0)
    op_explicit = OriginationPenalty(l2=9.0)
    cfg = GpurecConfig(regularizer=PenaltyOptions(origination=op_cfg))
    got = _run_optimize_capture(monkeypatch, config=cfg, origination_penalty=op_explicit)
    assert got is op_explicit


def test_optimize_config_none_reproduces_today_default_origination_penalty(monkeypatch):
    got = _run_optimize_capture(monkeypatch)
    assert got is None


def test_default_origination_penalty_is_noop_equivalent_to_none():
    """VERIFY the brief's claimed equivalence: a default (all-zero, disabled) ``OriginationPenalty``
    -- what ``config.regularizer.origination`` resolves to for a default ``GpurecConfig()`` -- is a
    byte-exact no-op, identical to never applying a penalty at all (``origination_penalty=None``)."""
    from gpurec.solver.penalties import origination_penalty_and_grad

    default_pen = GpurecConfig().regularizer.origination
    assert isinstance(default_pen, OriginationPenalty)
    assert not default_pen.any_active()
    omega = torch.randn(5, dtype=torch.float64)
    pen, grad = origination_penalty_and_grad(omega, default_pen)
    assert float(pen) == 0.0
    assert torch.equal(grad, torch.zeros_like(omega))


# --- map_cv --------------------------------------------------------------

def _run_map_cv_capture(monkeypatch, **kwargs):
    """Run ``map_cv`` with stubbed ``_build``/``fit_specieswise``/``heldout_nll`` (each replaced by a
    cheap, deterministic fake) so the k-fold loop completes instantly without any real gene/species
    tree files or CUDA/Triton solve. Returns ``(result_dict, captured)`` where
    ``captured["solver_options"]`` is the ``SolverOptions`` every ``_build`` call actually received.
    """
    import gpurec.fit.map_cv as mc

    captured = {}

    class _FakeModel:
        def __init__(self):
            self.species_helpers = {"S": 2}
            self.receiver_weights = torch.zeros(2)
            self.batch_statics = None

    def _fake_build(species_tree, paths, *, mode, device, solver_options):
        captured["solver_options"] = solver_options
        return _FakeModel()

    def _fake_fit_specieswise(batch_statics, theta0, receiver_weights, *, lam, theta_ref, **fkwargs):
        return {"theta": theta0}

    def _fake_heldout_nll(batch_statics, theta, receiver_weights):
        return 0.0

    monkeypatch.setattr(mc, "_build", _fake_build)
    monkeypatch.setattr(mc, "fit_specieswise", _fake_fit_specieswise)
    monkeypatch.setattr(mc, "heldout_nll", _fake_heldout_nll)
    result = mc.map_cv("sp.nwk", ["g1.nwk", "g2.nwk"], device="cpu", k=2, verbose=False, **kwargs)
    return result, captured


def test_map_cv_config_solver_field_threaded(monkeypatch):
    cfg = GpurecConfig(solver=SolverOptions(e_max_iter=999, pi_iters=32))
    _result, captured = _run_map_cv_capture(monkeypatch, config=cfg)
    assert captured["solver_options"].e_max_iter == 999
    assert captured["solver_options"].pi_iters == 32  # pi_iters IS in _CV_SO's key subset


def test_map_cv_config_lambdas_threaded(monkeypatch):
    cfg = GpurecConfig(regularizer=PenaltyOptions(lambdas=(0.0, 2.0)))
    result, _captured = _run_map_cv_capture(monkeypatch, config=cfg)
    assert result["lambdas"] == [2.0, 0.0]


def test_map_cv_explicit_lambdas_wins_over_config(monkeypatch):
    cfg = GpurecConfig(regularizer=PenaltyOptions(lambdas=(0.0, 2.0)))
    result, _captured = _run_map_cv_capture(monkeypatch, config=cfg, lambdas=(3.0, 4.0))
    assert result["lambdas"] == [4.0, 3.0]


def test_map_cv_config_none_reproduces_today_defaults(monkeypatch):
    from gpurec.fit.map_cv import _CV_SO

    result, captured = _run_map_cv_capture(monkeypatch)
    assert captured["solver_options"] == SolverOptions(**_CV_SO)
    assert result["lambdas"] == sorted({0.0, 1.0, 10.0, 100.0, 1000.0}, reverse=True)
