from types import SimpleNamespace

import pytest
import torch

import gpurec.api.model as api_model
import gpurec.api.uniform_chunked as uniform_chunked_api
from gpurec.core.origination import (
    OriginationPrior,
    PreparedOriginationPrior,
    prepare_origination_prior,
)


def _cpu() -> torch.device:
    return torch.device("cpu")


def test_default_origination_prior_keeps_implicit_uniform_distribution():
    prior = OriginationPrior().prepare(
        S=3,
        device=_cpu(),
        dtype=torch.float64,
    )

    assert prior.probs is None
    assert prior.species_count == 3
    assert prior.family_count is None
    assert prior.is_default_uniform
    assert prior.is_shared
    assert not prior.is_family_specific
    assert prior.log_weights is None
    assert prior.select_families([2, 0]) is prior


def test_scalar_origination_prior_is_rejected_like_existing_probs_helper():
    with pytest.raises(ValueError, match=r"None, \[S\], or \[families, S\]"):
        OriginationPrior(0.5).prepare(
            S=3,
            device=_cpu(),
            dtype=torch.float64,
        )


def test_shared_vector_origination_prior_normalizes_and_logs_weights():
    prior = OriginationPrior([2.0, 1.0, 1.0]).prepare(
        S=3,
        device=_cpu(),
        dtype=torch.float64,
    )

    expected = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    torch.testing.assert_close(prior.probs, expected)
    torch.testing.assert_close(prior.log_weights, torch.log2(expected))
    assert prior.species_count == 3
    assert prior.family_count is None
    assert prior.is_shared
    assert not prior.is_family_specific
    assert prior.select_families([1, 0]) is prior


def test_family_matrix_origination_prior_normalizes_and_selects_subset():
    raw = torch.tensor(
        [
            [1.0, 1.0, 2.0],
            [0.0, 4.0, 4.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    prior = prepare_origination_prior(
        raw,
        S=3,
        device=_cpu(),
        dtype=torch.float64,
        family_count=3,
    )
    expected = raw.to(dtype=torch.float64)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    assert isinstance(prior, PreparedOriginationPrior)
    assert prior.species_count == 3
    assert prior.family_count == 3
    assert prior.is_family_specific
    torch.testing.assert_close(prior.probs, expected)

    selected = prior.select_families([2, 0])

    assert selected is not prior
    assert selected.species_count == 3
    assert selected.family_count == 2
    assert selected.is_family_specific
    torch.testing.assert_close(selected.probs, expected[[2, 0]])
    torch.testing.assert_close(selected.log_weights, torch.log2(expected[[2, 0]]))


def test_family_matrix_subset_validates_indices():
    prior = OriginationPrior(
        [
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    ).prepare(S=2, device=_cpu(), dtype=torch.float64, family_count=2)

    with pytest.raises(IndexError, match="family index 2 out of range"):
        prior.select_families([2])
    with pytest.raises(ValueError, match="family_indices entries"):
        prior.select_families([True])
    with pytest.raises(ValueError, match="family_indices entries"):
        prior.select_families([1.5])  # type: ignore[list-item]


def test_prepared_origination_prior_uses_existing_trust_boundary():
    raw = torch.tensor([0.0, float("nan"), -1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        OriginationPrior(raw).prepare(
            S=3,
            device=_cpu(),
            dtype=torch.float64,
        )

    prepared = OriginationPrior.prepared(raw).prepare(
        S=3,
        device=_cpu(),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        prepared.probs,
        raw.to(dtype=torch.float64),
        equal_nan=True,
    )

    same = prepare_origination_prior(
        prepared,
        S=3,
        device=_cpu(),
        dtype=torch.float64,
    )

    assert same is prepared


def test_prepared_origination_prior_rejects_incompatible_reprepare_shape():
    prepared = OriginationPrior([1.0, 1.0, 2.0]).prepare(
        S=3,
        device=_cpu(),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="prepared origination prior has S=3"):
        prepare_origination_prior(
            prepared,
            S=2,
            device=_cpu(),
            dtype=torch.float64,
        )


def test_gene_recon_model_threads_prepared_origination_prior(monkeypatch):
    dataset = SimpleNamespace(
        dtype=torch.float64,
        genewise=False,
        specieswise=False,
        device=_cpu(),
        S=3,
        families=[object(), object()],
        unnorm_row_max=torch.zeros(3, dtype=torch.float64),
    )
    dataset._species_helpers_for_mode = lambda *, device, dtype: ({}, None)
    static = SimpleNamespace()
    build_calls: list[PreparedOriginationPrior] = []

    def fake_build_static_state(_dataset, **kwargs):
        build_calls.append(kwargs["origination_prior"])
        return static

    monkeypatch.setattr(
        api_model,
        "require_cuda_device",
        lambda device, *, owner: torch.device(device),
    )
    monkeypatch.setattr(
        api_model,
        "_build_static_state_impl",
        fake_build_static_state,
    )
    monkeypatch.setattr(
        api_model,
        "_metadata_for_full_static",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    model = api_model.GeneReconModel(
        dataset=dataset,
        mode="global",
        origination_probs=OriginationPrior(
            [
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        ),
    )

    expected = torch.tensor(
        [
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    assert isinstance(model._origination_prior, PreparedOriginationPrior)
    assert model._origination_prior.is_family_specific
    torch.testing.assert_close(model.origination_probs, expected)
    assert model._origination_prior.probs is model.origination_probs
    assert len(build_calls) == 1
    assert build_calls[0] is model._origination_prior
    assert build_calls[0].probs is model.origination_probs


def test_retained_shared_theta_layout_omits_family_idx():
    calls = []

    class FakeRustPreprocessed:
        def build_chunked_layouts(self, **kwargs):
            calls.append(dict(kwargs))
            return [
                {
                    "indices": [0],
                    "clades": 2,
                    "splits": 0,
                    "waves": 1,
                    "max_wave": 2,
                    "wave_layout": {},
                }
            ]

    dataset = SimpleNamespace(
        _rust_preprocessed=FakeRustPreprocessed(),
        _preprocess_cpu_cores=16,
        dtype=torch.float32,
        S=3,
        families=[{"root_clade_id": 1, "C": 2}],
        family_names=["fam0"],
        gene_tree_paths=[["g0.nwk"]],
    )

    specs = api_model._build_batch_specs_from_retained_rust(
        dataset,
        mode="specieswise",
        family_chunk_size=500,
        clade_budget=315000,
        batch_packing="clade_first_fit",
        max_wave_size=8192,
        max_root_wave_size=None,
        max_dts_partial_rows=None,
        small_family_max_leaves=0,
    )

    assert specs is not None
    assert calls[0]["include_family_idx"] is False
    assert specs[0].wave_layout == {}


def test_uniform_chunked_model_threads_prepared_origination_prior(
    monkeypatch,
    tmp_path,
):
    build_kwargs = []

    class FakeDataset:
        def __init__(self, **kwargs):
            self.dtype = kwargs.get("dtype", torch.float64)
            self.device = kwargs.get("device", torch.device("cpu"))
            self.S = 3
            self.families = [
                {"C": 2, "N_splits": 1},
                {"C": 3, "N_splits": 2},
            ]
            self.gene_tree_paths = [("g0.nwk",), ("g1.nwk",)]
            self.family_names = ("fam0", "fam1")
            self.unnorm_row_max = torch.zeros(3, dtype=self.dtype)

        def _species_helpers_for_mode(self, *, device, dtype):
            return {}, None

        @classmethod
        def _from_preprocessed_raw(cls, **_kwargs):
            return cls()

    spec = uniform_chunked_api._UniformChunkSpec(indices=[0, 1], clades=5, splits=3)
    built = uniform_chunked_api._UniformBuiltChunk(
        spec=spec,
        wave_layout={},
        waves=1,
        max_wave=5,
        split_rows=3,
        max_wave_split_rows=3,
    )

    monkeypatch.setattr(
        uniform_chunked_api,
        "require_cuda_device",
        lambda device, *, owner: torch.device("cpu"),
    )
    class FakeRustPreprocessed:
        def to_torch(self):
            return {}

        def family_basic_counts(self):
            return {
                "clade_counts": [2, 3],
                "split_counts": [1, 2],
            }

        def build_chunked_layouts(self, **kwargs):
            build_kwargs.append(dict(kwargs))
            return [
                {
                    "indices": spec.indices,
                    "clades": spec.clades,
                    "splits": spec.splits,
                    "wave_layout": built.wave_layout,
                    "waves": built.waves,
                    "max_wave": built.max_wave,
                    "split_rows": built.split_rows,
                    "max_wave_split_rows": built.max_wave_split_rows,
                }
            ]

    class FakeRustExtension:
        def preprocess_dataset(self, *_args, **_kwargs):
            return FakeRustPreprocessed()

    import gpurec.core.preprocess_rust as preprocess_rust

    monkeypatch.setattr(uniform_chunked_api, "GeneDataset", FakeDataset)
    monkeypatch.setattr(
        uniform_chunked_api.GeneDataset,
        "_from_preprocessed_raw",
        classmethod(lambda cls, **_kwargs: cls()),
    )
    monkeypatch.setattr(preprocess_rust, "RustPreprocessExtension", FakeRustExtension)

    model = uniform_chunked_api.UniformChunkedReconModel(
        species_tree=tmp_path / "species.nwk",
        gene_trees=[tmp_path / "g0.nwk", tmp_path / "g1.nwk"],
        device="cpu",
        dtype=torch.float64,
        family_chunk_size=2,
        max_wave_size=8,
        origination_probs=OriginationPrior(
            [
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        ),
    )

    assert isinstance(model._origination_prior, PreparedOriginationPrior)
    assert model._origination_prior.is_family_specific
    assert model._origination_prior.probs is model.origination_probs
    assert model._state.origination_prior is model._origination_prior
    assert model._state.origination_probs is model.origination_probs
    assert build_kwargs[0]["include_family_idx"] is False
