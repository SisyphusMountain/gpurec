"""Parity tests for documented uniform backward ancestor-batching flags."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.gradient,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "test_trees_1000"


_PROTOTYPE_CASES = [
    ("proposal0_2d_triton", {
        "GPUREC_SELF_LOOP_2D_TRITON": "1",
        "GPUREC_SELF_LOOP_2D_BLOCK_W": "1",
    }),
    ("proposal1_tree_staged", {
        "GPUREC_SELF_LOOP_TREE_STAGED": "1",
        "GPUREC_SELF_LOOP_TREE_TILE_W": "2",
        "GPUREC_SELF_LOOP_TREE_TILE_S": "128",
    }),
    ("proposal2_tree_transposed", {
        "GPUREC_SELF_LOOP_TREE_TRANSPOSED": "1",
        "GPUREC_SELF_LOOP_TREE_TILE_W": "16",
    }),
    ("proposal3_hybrid_tree_router", {
        "GPUREC_SELF_LOOP_TREE_STAGED": "1",
        "GPUREC_SELF_LOOP_TREE_MIN_W": "0",
        "GPUREC_SELF_LOOP_TREE_SPLIT_WAVES": "1",
    }),
    ("proposal4_forward_path_prefix", {
        "GPUREC_FORWARD_UNIFORM_PATH_PREFIX": "1",
        "GPUREC_FORWARD_UNIFORM_PATH_TILE_W": "2",
    }),
    ("proposal5_cuda_nosplit_tree", {
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT": "1",
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION": "tree",
    }),
]

_PROTOTYPE_ENV_KEYS = sorted({
    key for _name, env in _PROTOTYPE_CASES for key in env
} | {
    "GPUREC_CUDA_WAVE_NOSPLIT",
})


def _gene_paths(n_families: int) -> list[str]:
    if not _DATA_DIR.exists():
        pytest.skip("test_trees_1000 dataset not present")
    paths = sorted(_DATA_DIR.glob("g_*.nwk"))[:n_families]
    if len(paths) < n_families:
        pytest.skip(f"Need {n_families} gene families in test_trees_1000")
    return [str(path) for path in paths]


def _set_clean_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key in _PROTOTYPE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _loss_and_grad(n_families: int) -> tuple[torch.Tensor, torch.Tensor]:
    model = GeneReconModel.from_trees(
        species_tree=str(_DATA_DIR / "sp.nwk"),
        gene_trees=_gene_paths(n_families),
        mode="global",
        pibar_mode="uniform",
        device=torch.device("cuda"),
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=6,
        max_iters_E=128,
        tol_E=0.0,
        neumann_terms=3,
        use_pruning=False,
        cg_tol=1e-8,
        cg_maxiter=500,
        max_wave_size=4096,
    )
    model.zero_grad(set_to_none=True)
    model.static.warm_E = None
    loss = model()
    loss.backward()
    torch.cuda.synchronize()
    return loss.detach(), model.theta.grad.detach().clone()


@pytest.mark.parametrize(
    "n_families",
    [
        10,
        pytest.param(50, marks=pytest.mark.slow),
    ],
)
def test_documented_uniform_backward_prototypes_match_default_on_test_trees_1000(
    monkeypatch,
    n_families,
):
    """Baseline parity for documented opt-in prototypes on high-S subsets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if any(
        env.get("GPUREC_CUDA_SELF_LOOP_NOSPLIT") == "1"
        for _name, env in _PROTOTYPE_CASES
    ):
        pytest.importorskip("cuda.bindings")

    with monkeypatch.context() as base_env:
        _set_clean_env(base_env, {})
        baseline_loss, baseline_grad = _loss_and_grad(n_families)

    grad_inf = baseline_grad.abs().max().clamp_min(torch.tensor(
        1e-12, device=baseline_grad.device, dtype=baseline_grad.dtype,
    ))
    failures: list[str] = []

    for case_name, env in _PROTOTYPE_CASES:
        with monkeypatch.context() as proto_env:
            _set_clean_env(proto_env, env)
            loss, grad = _loss_and_grad(n_families)

        loss_diff = (loss - baseline_loss).abs().item()
        grad_abs = (grad - baseline_grad).abs().max().item()
        grad_rel = (grad - baseline_grad).abs().max().div(grad_inf).item()
        print(
            f"{n_families} families {case_name}: "
            f"loss_abs={loss_diff:.3e} grad_abs={grad_abs:.3e} "
            f"grad_rel_inf={grad_rel:.3e}"
        )

        # The fp32 backward path uses CUDA atomics and opt-in prototype kernels
        # with different accumulation orders; compare gradients primarily by
        # the documented max-relative-to-inf metric and keep max-absolute as a
        # scale diagnostic.
        grad_rel_limit = 2e-3
        grad_abs_limit = max(1e-2, grad_rel_limit * float(grad_inf.item()))
        if loss_diff > 5e-3 or grad_abs > grad_abs_limit or grad_rel > grad_rel_limit:
            failures.append(
                f"{case_name}: loss_abs={loss_diff:.3e}, "
                f"grad_abs={grad_abs:.3e}, grad_rel_inf={grad_rel:.3e}"
            )

    assert not failures, "\n".join(failures)
