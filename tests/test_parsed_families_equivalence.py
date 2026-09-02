"""The resident-parse path must reproduce the JSON path tensor for tensor.

``GeneReconModel(parsed_families=..., family_indices=...)`` parses each ``.ale`` file once and
receives numpy arrays from Rust instead of a JSON string. This test pins that the model it builds
is indistinguishable from the one the legacy ``preprocess_dataset`` path builds: same number of
batches, same static tensors (dtype, shape, values), same species helpers, same wave metadata, and
-- when a GPU is available -- the same genewise loss and gradient.
"""
import glob
import os

import pytest
import torch

from gpurec.api.model import GeneReconModel
from gpurec.core.scheduling.batching import parse_families

_SAMPLE_DIR = os.path.join(
    "/tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec",
    "00daa94b-c320-49e3-b103-a174ae643650/scratchpad/coleman_sample",
)
_SPECIES_TREE = os.path.join(_SAMPLE_DIR, "ReferenceTree.nwk")
_N_FAMILIES = 40
_CLADE_BUDGET = 60_000
_SUBSET = slice(5, 25)


def _sample_paths():
    return sorted(glob.glob(os.path.join(_SAMPLE_DIR, "ALEs", "*.ale")))[:_N_FAMILIES]


pytestmark = pytest.mark.skipif(
    not os.path.isfile(_SPECIES_TREE) or len(_sample_paths()) < _N_FAMILIES,
    reason=f"sample ALE dataset not present under {_SAMPLE_DIR}",
)

# The tensor attributes of _BatchStatic that must match element for element.
_STATIC_TENSOR_FIELDS = ("rate_family_idx", "family_index_tensor")
# Per-wave metadata keys holding tensors.
_META_TENSOR_KEYS = (
    "sl", "sr", "reduce_idx", "log_split_probs", "eq1_reduce_idx",
    "ge2_ptr", "ge2_parent_ids",
)
_META_SCALAR_KEYS = ("start", "end", "W", "has_splits", "phase", "n_eq1", "ge2_max_fanout")
_LAYOUT_TENSOR_KEYS = ("perm", "leaf_species_index", "root_clade_ids", "family_idx")


def _assert_tensor_equal(left, right, what):
    assert type(left) is type(right), f"{what}: {type(left)} vs {type(right)}"
    if not torch.is_tensor(left):
        assert left == right, f"{what}: {left!r} vs {right!r}"
        return
    assert left.dtype == right.dtype, f"{what}: dtype {left.dtype} vs {right.dtype}"
    assert left.shape == right.shape, f"{what}: shape {left.shape} vs {right.shape}"
    assert torch.equal(left, right), f"{what}: values differ"


def _assert_models_equal(model_a, model_b):
    assert len(model_a.batch_statics) == len(model_b.batch_statics)
    assert model_a.family_batches == model_b.family_batches
    assert len(model_a.families) == len(model_b.families)

    for key, value_a in model_a.species_helpers.items():
        value_b = model_b.species_helpers[key]
        _assert_tensor_equal(value_a, value_b, f"species_helpers[{key!r}]")

    for batch_index, (static_a, static_b) in enumerate(
        zip(model_a.batch_statics, model_b.batch_statics)
    ):
        tag = f"batch {batch_index}"
        assert static_a.family_indices == static_b.family_indices, tag
        for field in _STATIC_TENSOR_FIELDS:
            _assert_tensor_equal(
                getattr(static_a, field), getattr(static_b, field), f"{tag}.{field}"
            )
        layout_a, layout_b = static_a.wave_layout, static_b.wave_layout
        for key in _LAYOUT_TENSOR_KEYS:
            _assert_tensor_equal(layout_a[key], layout_b[key], f"{tag}.wave_layout[{key!r}]")
        assert len(layout_a["wave_metas"]) == len(layout_b["wave_metas"]), tag
        for wave_index, (meta_a, meta_b) in enumerate(
            zip(layout_a["wave_metas"], layout_b["wave_metas"])
        ):
            what = f"{tag}.wave {wave_index}"
            assert set(meta_a) == set(meta_b), f"{what}: keys {set(meta_a) ^ set(meta_b)}"
            for key in _META_SCALAR_KEYS:
                if key in meta_a:
                    assert meta_a[key] == meta_b[key], f"{what}[{key!r}]"
            for key in _META_TENSOR_KEYS:
                if key in meta_a:
                    _assert_tensor_equal(meta_a[key], meta_b[key], f"{what}[{key!r}]")
            for dtype, values_a in meta_a.get("_log_split_probs_by_dtype", {}).items():
                _assert_tensor_equal(
                    values_a,
                    meta_b["_log_split_probs_by_dtype"][dtype],
                    f"{what}._log_split_probs_by_dtype[{dtype}]",
                )


def _build_pair(paths, indices, parsed, device):
    legacy = GeneReconModel(
        _SPECIES_TREE, paths, mode="genewise", device=device, clade_budget=_CLADE_BUDGET
    )
    from_parsed = GeneReconModel(
        _SPECIES_TREE,
        paths,
        mode="genewise",
        device=device,
        clade_budget=_CLADE_BUDGET,
        parsed_families=parsed,
        family_indices=indices,
    )
    return legacy, from_parsed


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_parsed_path_matches_legacy_full_set():
    paths = _sample_paths()
    parsed = parse_families(_SPECIES_TREE, paths)
    assert parsed.n_families() == len(paths)
    legacy, from_parsed = _build_pair(paths, list(range(len(paths))), parsed, _device())
    _assert_models_equal(legacy, from_parsed)


def test_parsed_path_matches_legacy_subset():
    paths = _sample_paths()
    parsed = parse_families(_SPECIES_TREE, paths)
    indices = list(range(len(paths)))[_SUBSET]
    subset_paths = [paths[index] for index in indices]
    legacy, from_parsed = _build_pair(subset_paths, indices, parsed, _device())
    _assert_models_equal(legacy, from_parsed)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parsed_path_matches_legacy_loss_and_grad():
    """Loss is bitwise equal; the gradient matches to the backward kernel's own run-to-run noise.

    The per-family NLL is reproducible, so it is compared with ``torch.equal``. The genewise
    gradient is NOT: the backward accumulates with atomics, so calling it twice on the SAME model
    with the SAME theta already gives differing last bits (measured ~5e-4 on 40 coleman families,
    peak |grad| ~4e2). The test therefore asserts the legacy-vs-parsed difference is no worse than
    1e-5 of the gradient's own scale, and records the legacy self-difference for comparison.
    """
    paths = _sample_paths()
    parsed = parse_families(_SPECIES_TREE, paths)
    device = torch.device("cuda")
    legacy, from_parsed = _build_pair(paths, list(range(len(paths))), parsed, device)
    theta = torch.full((len(paths), 3), -1.5, dtype=legacy.theta.dtype, device=device)
    loss_a, grad_a, _ = legacy.genewise_loss_vector_and_grad(theta=theta)
    _loss_a2, grad_a2, _ = legacy.genewise_loss_vector_and_grad(theta=theta)
    loss_b, grad_b, _ = from_parsed.genewise_loss_vector_and_grad(theta=theta)

    assert torch.equal(loss_a, loss_b), f"max |dloss| = {(loss_a - loss_b).abs().max()}"
    scale = float(grad_a.abs().max())
    self_noise = float((grad_a - grad_a2).abs().max())
    cross = float((grad_b - grad_a).abs().max())
    assert cross <= 1e-5 * scale, (
        f"legacy-vs-parsed max |dgrad| = {cross:.3e} exceeds 1e-5 * |grad|max = {1e-5 * scale:.3e} "
        f"(legacy self run-to-run noise = {self_noise:.3e})"
    )
