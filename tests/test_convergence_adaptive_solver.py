"""Tests for per-family solver-convergence diagnostics + explicit rebatching.

Covers:
  - explicit ``family_group_assignments`` rebatching (Rust + Python round-trip),
  - per-batch ``SolverOptions`` assignment without clobbering the global options,
  - forward Pi residual diagnostic (converged at high ``pi_iters``, large at low),
  - backward Neumann relres diagnostic (monotone in ``neumann_terms``).
"""

from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.scheduling import batching


_TINY_ALE = """#constructor_string
A_1,B_1,C_1
#observations
120
#Bip_counts
4 120
5 120
6 120
#Bip_bls
1 1
2 1
3 1
4 1
5 1
6 1
#Dip_counts
4 2 3 120
5 1 3 120
6 1 2 120
#last_leafset_id
6
#leaf-id
A_1 1
B_1 2
C_1 3
#set-id
1 : 3
2 : 1
3 : 2
4 : 1 2
5 : 2 3
6 : 1 3
#END
"""


def _require_native() -> None:
    try:
        batching._load_native_module()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


def _require_cuda() -> torch.device:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


def _write_multi(tmp_path: Path, k: int) -> tuple[Path, list[Path]]:
    species = tmp_path / "species.nwk"
    species.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    families = []
    for i in range(k):
        fam = tmp_path / f"family_{i}.ale"
        fam.write_text(_TINY_ALE, encoding="utf-8")
        families.append(fam)
    return species, families


def _build(tmp_path: Path, k: int, *, device, neumann_terms: int = 16) -> GeneReconModel:
    species, families = _write_multi(tmp_path, k)
    return GeneReconModel(
        species,
        families,
        mode="specieswise",
        device=device,
        solver_options=SolverOptions(
            e_max_iter=2000, e_tol=1e-10, pi_iters=16, neumann_terms=neumann_terms
        ),
    )


_ARCHAEA = Path(__file__).resolve().parent / "data/alerax_archaea_davin2017"


def _build_archaea(k: int, *, device, neumann_terms: int = 16) -> GeneReconModel:
    """Build a specieswise model on a few real archaea families (non-trivial trees).

    The tiny 3-species synthetic family is trivially converged, so it cannot exercise
    the Neumann-convergence diagnostics; the real data can.
    """
    species = _ARCHAEA / "species_reference/reference_species_tree.newick"
    fam_dir = _ARCHAEA / "ale_gene_tree_distributions/main_families_ge4seq"
    if not species.exists() or not fam_dir.exists():
        pytest.skip("archaea reference dataset is unavailable")
    families = [str(p) for p in sorted(fam_dir.glob("*.ale"))[: k + 10]]
    options = SolverOptions(
        e_max_iter=4000, e_tol=1e-10, pi_iters=16, neumann_terms=neumann_terms
    )
    while families:
        try:
            return GeneReconModel(
                species, families[:k], mode="specieswise", device=device, solver_options=options
            )
        except ValueError as exc:
            bad = next((f for f in families[:k] if Path(f).name in str(exc)), None)
            if bad is None:
                raise
            families.remove(bad)
    pytest.skip("no parseable archaea families available")


def _set_moderate_theta(model: GeneReconModel) -> None:
    # A non-trivial rate point so the self-loop has rho ~0.3-0.6 (real Neumann behaviour);
    # the default theta makes these small families converge in a single term.
    rates = torch.log2(torch.tensor([0.3, 0.5, 0.2], device=model.theta.device))
    with torch.no_grad():
        model.theta.copy_(rates.expand_as(model.theta).contiguous())


def test_family_group_assignments_partition_batches(tmp_path: Path) -> None:
    _require_native()
    model = _build(tmp_path, 6, device="cpu")
    labels = [0, 1, 2, 0, 1, 2]
    model.replan_batches(family_group_assignments=labels)
    # every batch's families share one label, and all families covered exactly once
    for batch in model.family_batches:
        assert len({labels[i] for i in batch}) == 1
    covered = sorted(i for b in model.family_batches for i in b)
    assert covered == list(range(6))
    assert model.family_group_assignments == labels


def test_set_batch_solver_options_is_per_batch_and_keeps_global(tmp_path: Path) -> None:
    _require_native()
    model = _build(tmp_path, 6, device="cpu")
    model.replan_batches(family_group_assignments=[0, 1, 2, 0, 1, 2])
    opts = {
        0: SolverOptions(pi_iters=16, neumann_terms=4),
        1: SolverOptions(pi_iters=32, neumann_terms=20),
        2: SolverOptions(pi_iters=64, neumann_terms=32),
    }
    model.set_batch_solver_options(opts)
    for batch, static in zip(model.family_batches, model.batch_statics):
        label = [0, 1, 2, 0, 1, 2][batch[0]]
        assert static.solver_options.neumann_terms == opts[label].neumann_terms
    # distinct objects (no aliasing) and global options untouched
    assert len({id(s.solver_options) for s in model.batch_statics}) == len(model.batch_statics)
    assert model.solver_options.neumann_terms == 16


def test_set_batch_solver_options_requires_grouping(tmp_path: Path) -> None:
    _require_native()
    model = _build(tmp_path, 3, device="cpu")
    with pytest.raises(ValueError):
        model.set_batch_solver_options({0: SolverOptions()})


def test_forward_residual_converges_with_pi_iters(tmp_path: Path) -> None:
    device = _require_cuda()
    _require_native()
    from gpurec.core.inference.solver import solve_forward_residual

    model = _build_archaea(8, device=device)
    static = model.batch_statics[0]
    hi = solve_forward_residual(static, model.theta.detach(), model.receiver_weights, pi_iters=400)
    lo = solve_forward_residual(static, model.theta.detach(), model.receiver_weights, pi_iters=2)
    assert float(hi.max()) < 1e-2
    assert float(lo.max()) > float(hi.max())


def test_backward_relres_monotone_in_neumann_terms(tmp_path: Path) -> None:
    device = _require_cuda()
    _require_native()
    from gpurec.api._execution import evaluate_static_convergence

    model = _build_archaea(16, device=device)
    _set_moderate_theta(model)
    static = model.batch_statics[0]
    theta = model.theta.detach()
    medians = []
    for nt in (2, 8, 16):
        _f, relres, _vk = evaluate_static_convergence(
            static, theta, model.receiver_weights, pi_iters_high=400, neumann_terms=nt
        )
        finite = relres[torch.isfinite(relres)]
        medians.append(float(finite.median()))
    # more Neumann terms -> smaller residual
    assert medians[0] > medians[1] > medians[2]
