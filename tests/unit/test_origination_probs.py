import pytest
import torch

from gpurec.core.likelihood import (
    E_fixed_point,
    E_step,
    compute_origination_denominator,
    compute_nll,
    compute_nll_root_rows,
    gather_root_rows,
    prepare_origination_probs,
)


def _manual_weighted_nll(root_rows, E, weights):
    weights = weights / weights.sum(dim=-1, keepdim=True)
    numerator = torch.log2((weights * torch.exp2(root_rows)).sum(dim=-1))
    denominator = torch.log2((weights * (1.0 - torch.exp2(E))).sum(dim=-1))
    return -(numerator - denominator)


@pytest.mark.parametrize(
    "mode",
    ["shared", "vector", "prepared_vector", "family_specific"],
)
def test_root_row_nll_matches_full_pi_for_origination_probability_modes(mode):
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0, -3.5],
            [-1.5, -3.0, -2.5, -4.0],
            [-3.0, -1.0, -4.5, -2.0],
            [-2.0, -2.5, -3.5, -1.75],
            [-4.25, -2.25, -1.5, -3.25],
            [-2.75, -3.5, -2.0, -4.5],
        ],
        dtype=dtype,
    )
    roots = torch.tensor([4, 1, 3], dtype=torch.long)
    E = torch.tensor(
        [
            [-3.0, -2.0, -4.0, -3.5],
            [-2.5, -2.25, -3.25, -4.0],
            [-3.75, -2.75, -2.5, -3.0],
        ],
        dtype=dtype,
    )

    kwargs = {}
    if mode == "vector":
        kwargs["origination_probs"] = torch.tensor([2.0, 5.0, 1.0, 3.0], dtype=dtype)
    elif mode == "prepared_vector":
        kwargs["origination_probs"] = prepare_origination_probs(
            torch.tensor([2.0, 5.0, 1.0, 3.0], dtype=dtype),
            S=4,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        kwargs["origination_probs_prepared"] = True
    elif mode == "family_specific":
        kwargs["origination_probs"] = torch.tensor(
            [
                [2.0, 5.0, 1.0, 3.0],
                [1.0, 4.0, 2.0, 6.0],
                [3.0, 1.0, 7.0, 2.0],
            ],
            dtype=dtype,
        )

    full_pi_nll = compute_nll(Pi, E, roots, **kwargs)
    root_row_nll = compute_nll_root_rows(gather_root_rows(Pi, roots), E, **kwargs)

    torch.testing.assert_close(root_row_nll, full_pi_nll, rtol=1e-12, atol=1e-12)


def test_gather_root_rows_matches_direct_indexing_for_vector_and_scalar_roots():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
            [-3.0, -1.0, -4.5],
        ],
        dtype=dtype,
    )

    vector_roots = torch.tensor([2, 0], dtype=torch.long)
    torch.testing.assert_close(
        gather_root_rows(Pi, vector_roots),
        Pi[vector_roots, :],
    )
    torch.testing.assert_close(
        gather_root_rows(Pi, torch.tensor(1)),
        Pi[1, :],
    )


def test_root_row_nll_accepts_precomputed_denominator():
    dtype = torch.float64
    root_rows = torch.tensor(
        [[-4.0, -2.0, -5.0], [-1.5, -3.0, -2.5]],
        dtype=dtype,
    )
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    denominator = torch.log2(1 - torch.exp2(E).mean(dim=-1))

    expected = compute_nll_root_rows(root_rows, E)
    actual = compute_nll_root_rows(root_rows, E, denominator=denominator)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_origination_denominator_accepts_shared_and_family_specific_shapes():
    dtype = torch.float64
    shared_E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    family_E = torch.tensor(
        [[-3.0, -2.0, -4.0], [-2.5, -2.25, -3.25]],
        dtype=dtype,
    )
    shared_weights = torch.tensor([2.0, 5.0, 1.0], dtype=dtype)
    family_weights = torch.tensor(
        [[2.0, 5.0, 1.0], [1.0, 4.0, 2.0]],
        dtype=dtype,
    )

    shared_denominator = compute_origination_denominator(shared_E, shared_weights)
    broadcast_denominator = compute_origination_denominator(
        shared_E,
        family_weights,
        family_count=2,
    )
    family_denominator = compute_origination_denominator(
        family_E,
        family_weights,
        family_count=2,
    )

    assert shared_denominator.shape == ()
    assert broadcast_denominator.shape == (2,)
    assert family_denominator.shape == (2,)
    torch.testing.assert_close(
        shared_denominator,
        torch.log2(
            (
                (shared_weights / shared_weights.sum(dim=-1, keepdim=True))
                * (1.0 - torch.exp2(shared_E))
            ).sum(dim=-1)
        ),
    )
    torch.testing.assert_close(
        broadcast_denominator,
        torch.log2(
            (
                (family_weights / family_weights.sum(dim=-1, keepdim=True))
                * (1.0 - torch.exp2(shared_E))
            ).sum(dim=-1)
        ),
    )
    torch.testing.assert_close(
        family_denominator,
        torch.log2(
            (
                (family_weights / family_weights.sum(dim=-1, keepdim=True))
                * (1.0 - torch.exp2(family_E))
            ).sum(dim=-1)
        ),
    )


@pytest.mark.parametrize(
    ("E", "weights", "family_count", "message"),
    [
        (
            torch.zeros(1, 2, 3, dtype=torch.float64),
            None,
            None,
            "E must have shape",
        ),
        (
            torch.zeros(3, 3, dtype=torch.float64),
            None,
            2,
            "E family dimension",
        ),
        (
            torch.zeros(2, 3, dtype=torch.float64),
            torch.ones(1, 3, dtype=torch.float64),
            None,
            "origination_probs.*E family dimension",
        ),
        (
            torch.zeros(3, dtype=torch.float64),
            torch.ones(3, 3, dtype=torch.float64),
            2,
            "one row per family",
        ),
    ],
)
def test_origination_denominator_rejects_ambiguous_family_shapes(
    E,
    weights,
    family_count,
    message,
):
    with pytest.raises(ValueError, match=message):
        compute_origination_denominator(
            E,
            weights,
            family_count=family_count,
        )


def test_root_row_nll_accepts_vector_precomputed_denominator():
    dtype = torch.float64
    root_rows = torch.tensor(
        [[-4.0, -2.0, -5.0], [-1.5, -3.0, -2.5]],
        dtype=dtype,
    )
    E = torch.tensor(
        [[-3.0, -2.0, -4.0], [-2.5, -2.25, -3.25]],
        dtype=dtype,
    )
    denominator = torch.log2(1 - torch.exp2(E).mean(dim=-1))

    expected = compute_nll_root_rows(root_rows, E)
    actual = compute_nll_root_rows(root_rows, E, denominator=denominator)

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "denominator",
    [
        torch.zeros(2, 1, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64),
    ],
)
def test_root_row_nll_rejects_broadcasting_denominator_shapes(denominator):
    dtype = torch.float64
    root_rows = torch.tensor(
        [[-4.0, -2.0, -5.0], [-1.5, -3.0, -2.5]],
        dtype=dtype,
    )
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)

    with pytest.raises(ValueError, match="denominator.*per-family NLL shape"):
        compute_nll_root_rows(root_rows, E, denominator=denominator)


def test_scalar_root_row_nll_rejects_vector_denominator():
    dtype = torch.float64
    root_row = torch.tensor([-4.0, -2.0, -5.0], dtype=dtype)
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)

    with pytest.raises(ValueError, match="denominator.*per-family NLL shape"):
        compute_nll_root_rows(root_row, E, denominator=torch.zeros(1, dtype=dtype))


@pytest.mark.parametrize(
    ("root_rows", "E", "message"),
    [
        (
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            "share the species dimension",
        ),
        (
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(3, 3, dtype=torch.float64),
            "family dimension",
        ),
        (
            torch.zeros(1, 2, 3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            "Pi_root_rows",
        ),
        (
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            "when Pi_root_rows has shape",
        ),
        (
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(1, 2, 3, dtype=torch.float64),
            "E must have shape",
        ),
    ],
)
def test_root_row_nll_rejects_invalid_root_or_e_shapes(root_rows, E, message):
    with pytest.raises(ValueError, match=message):
        compute_nll_root_rows(root_rows, E)


def test_compute_nll_scalar_root_uses_root_row_contract_with_weighted_prior():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
        ],
        dtype=dtype,
    )
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    weights = torch.tensor([2.0, 5.0, 1.0], dtype=dtype)

    actual = compute_nll(Pi, E, torch.tensor(1), origination_probs=weights)
    expected = compute_nll_root_rows(Pi[1], E, origination_probs=weights)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_e_step_requires_ancestors_t():
    dtype = torch.float64
    E = torch.full((3,), -1.0, dtype=dtype)

    with pytest.raises(ValueError, match="ancestors_T"):
        E_step(
            E=E,
            sp_P_idx=torch.tensor([0], dtype=torch.long),
            sp_child12_idx=torch.tensor([1, 2], dtype=torch.long),
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            ancestors_T=None,
        )


@pytest.mark.parametrize(
    ("ancestors_T", "message"),
    [
        ([[0.0]], "ancestors_T must be a tensor"),
        (torch.ones(3, dtype=torch.float64), "two-dimensional"),
        (torch.eye(2, dtype=torch.float64), r"ancestors_T shape must be \(3, 3\)"),
        (torch.eye(3, dtype=torch.float32), "ancestors_T dtype must match"),
    ],
)
def test_e_step_rejects_invalid_ancestors_t_before_matmul(
    ancestors_T: object,
    message: str,
):
    dtype = torch.float64
    E = torch.full((3,), -1.0, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        E_step(
            E=E,
            sp_P_idx=torch.tensor([0], dtype=torch.long),
            sp_child12_idx=torch.tensor([1, 2], dtype=torch.long),
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            ancestors_T=ancestors_T,
        )


def test_e_fixed_point_requires_ancestors_t():
    dtype = torch.float64
    species_helpers = {
        "S": 3,
        "s_P_indexes": torch.tensor([0], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="ancestors_T"):
        E_fixed_point(
            species_helpers=species_helpers,
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            max_iters=1,
            tolerance=-1.0,
            warm_start_E=None,
            dtype=dtype,
            device=torch.device("cpu"),
            ancestors_T=None,
            e_shape=(3,),
        )


def test_e_fixed_point_rejects_invalid_ancestors_t_before_iterations():
    dtype = torch.float64
    species_helpers = {
        "S": 3,
        "s_P_indexes": torch.tensor([0], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
    }
    progress_calls = []

    with pytest.raises(ValueError, match=r"ancestors_T shape must be \(3, 3\)"):
        E_fixed_point(
            species_helpers=species_helpers,
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            max_iters=1,
            tolerance=-1.0,
            warm_start_E=None,
            dtype=dtype,
            device=torch.device("cpu"),
            ancestors_T=torch.eye(2, dtype=dtype),
            progress_callback=lambda *_args: progress_calls.append(_args),
            e_shape=(3,),
        )

    assert progress_calls == []


def test_e_fixed_point_requires_explicit_e_shape_before_iterations():
    dtype = torch.float64
    species_helpers = {
        "S": 3,
        "s_P_indexes": torch.tensor([0], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
    }
    progress_calls = []

    with pytest.raises(ValueError, match="e_shape is required"):
        E_fixed_point(
            species_helpers=species_helpers,
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            max_iters=1,
            tolerance=-1.0,
            warm_start_E=None,
            dtype=dtype,
            device=torch.device("cpu"),
            ancestors_T=torch.eye(3, dtype=dtype),
            progress_callback=lambda *_args: progress_calls.append(_args),
        )

    assert progress_calls == []


def test_e_fixed_point_uses_explicit_e_shape_for_initial_state():
    dtype = torch.float64
    species_helpers = {
        "S": 3,
        "s_P_indexes": torch.tensor([0, 3], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
    }

    result = E_fixed_point(
        species_helpers=species_helpers,
        log_pS=torch.tensor(0.0, dtype=dtype),
        log_pD=torch.tensor(0.0, dtype=dtype),
        log_pL=torch.tensor(0.0, dtype=dtype),
        max_transfer_mat=torch.zeros(3, dtype=dtype),
        max_iters=1,
        tolerance=-1.0,
        warm_start_E=None,
        dtype=dtype,
        device=torch.device("cpu"),
        ancestors_T=torch.eye(3, dtype=dtype),
        e_shape=(2, 3),
    )

    assert result["E"].shape == (2, 3)
    assert result["E_bar"].shape == (2, 3)


def test_e_fixed_point_rejects_warm_start_that_conflicts_with_explicit_e_shape():
    dtype = torch.float64
    species_helpers = {
        "S": 3,
        "s_P_indexes": torch.tensor([0], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
    }
    progress_calls = []

    with pytest.raises(ValueError, match="warm_start_E shape"):
        E_fixed_point(
            species_helpers=species_helpers,
            log_pS=torch.zeros(3, dtype=dtype),
            log_pD=torch.zeros(3, dtype=dtype),
            log_pL=torch.zeros(3, dtype=dtype),
            max_transfer_mat=torch.zeros(3, dtype=dtype),
            max_iters=1,
            tolerance=-1.0,
            warm_start_E=torch.zeros(2, 3, dtype=dtype),
            dtype=dtype,
            device=torch.device("cpu"),
            ancestors_T=torch.eye(3, dtype=dtype),
            progress_callback=lambda *_args: progress_calls.append(_args),
            e_shape=(3,),
        )

    assert progress_calls == []


def test_weighted_origination_likelihood_matches_manual_formula():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
            [-3.0, -1.0, -4.5],
            [-2.0, -2.5, -3.5],
        ],
        dtype=dtype,
    )
    roots = torch.tensor([1, 3], dtype=torch.long)
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    weights = torch.tensor([0.15, 0.60, 0.25], dtype=dtype)

    expected = _manual_weighted_nll(Pi[roots], E, weights)
    actual = compute_nll(Pi, E, roots, origination_probs=weights)
    root_actual = compute_nll_root_rows(Pi[roots], E, origination_probs=weights)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(root_actual, expected, rtol=1e-12, atol=1e-12)


def test_family_specific_origination_likelihood_matches_manual_formula():
    dtype = torch.float64
    root_rows = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-2.0, -2.5, -3.5],
        ],
        dtype=dtype,
    )
    E = torch.tensor(
        [
            [-3.0, -2.0, -4.0],
            [-2.5, -2.25, -3.25],
        ],
        dtype=dtype,
    )
    weights = torch.tensor(
        [
            [0.15, 0.60, 0.25],
            [3.0, 1.0, 2.0],
        ],
        dtype=dtype,
    )

    expected = _manual_weighted_nll(root_rows, E, weights)
    actual = compute_nll_root_rows(root_rows, E, origination_probs=weights)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_uniform_origination_probs_preserve_default_likelihood():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
            [-3.0, -1.0, -4.5],
        ],
        dtype=dtype,
    )
    roots = torch.tensor([0, 2], dtype=torch.long)
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    uniform = torch.ones(3, dtype=dtype)

    default = compute_nll(Pi, E, roots)
    weighted = compute_nll(Pi, E, roots, origination_probs=uniform)
    root_weighted = compute_nll_root_rows(
        Pi[roots],
        E,
        origination_probs=uniform,
    )

    torch.testing.assert_close(weighted, default, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(root_weighted, default, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([0.0, 0.0, 0.0]), "positive mass"),
        (torch.tensor([1.0, -1.0, 2.0]), "non-negative"),
        (torch.tensor([1.0, float("nan"), 2.0]), "finite"),
    ],
)
def test_prepared_origination_probs_are_internal_trust_boundary(
    weights: torch.Tensor,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        prepare_origination_probs(
            weights,
            S=3,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    prepared = prepare_origination_probs(
        weights,
        S=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
        assume_prepared=True,
    )

    torch.testing.assert_close(prepared, weights.to(dtype=torch.float64), equal_nan=True)
