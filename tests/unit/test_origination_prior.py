import pytest
import torch

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
