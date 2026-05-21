import torch

import gpurec.core.backward as backward
from gpurec.core.extract_parameters import as_family_param


DEVICE = torch.device("cpu")
DTYPE = torch.float64


def test_optional_cuda_self_loop_fallback_handles_validation_failures():
    captured = None

    try:
        raise ValueError("optional CUDA prototype validation failed")
    except backward._OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS as exc:
        captured = exc

    assert isinstance(captured, ValueError)


def test_optional_cuda_self_loop_fallback_exceptions_stay_narrow():
    assert backward._OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS == (
        ImportError,
        RuntimeError,
        ValueError,
    )
    assert Exception not in backward._OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS


def test_backward_auto_wrap_records_family_idx_none_as_single_shared_row():
    C = 4
    S = 3
    E = torch.arange(S, dtype=DTYPE)
    Ebar = E + 10.0
    E_s1 = E + 20.0
    E_s2 = E + 30.0
    log_pS = torch.linspace(-0.5, -0.3, S, dtype=DTYPE)
    log_pD = torch.tensor(-1.25, dtype=DTYPE)
    log_pL = torch.tensor(-1.5, dtype=DTYPE)
    max_transfer_mat = torch.arange(S, dtype=DTYPE).reshape(S, 1)

    (
        auto_wrapped,
        family_idx,
        wrapped_E,
        wrapped_Ebar,
        wrapped_E_s1,
        wrapped_E_s2,
        wrapped_log_pS,
        wrapped_log_pD,
        wrapped_log_pL,
        wrapped_mt,
    ) = backward._auto_wrap_backward_inputs(
        C=C,
        device=DEVICE,
        family_idx=None,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_mat,
    )

    assert auto_wrapped is True
    assert family_idx.dtype == torch.long
    assert family_idx.device == DEVICE
    assert family_idx.tolist() == [0, 0, 0, 0]
    assert wrapped_E.shape == (1, S)
    assert wrapped_Ebar.shape == (1, S)
    assert wrapped_E_s1.shape == (1, S)
    assert wrapped_E_s2.shape == (1, S)
    assert wrapped_log_pS.shape == (1, S)
    assert wrapped_log_pD.shape == (1,)
    assert wrapped_log_pL.shape == (1,)
    assert wrapped_mt.shape == (1, S, 1)
    torch.testing.assert_close(wrapped_E[0], E)
    torch.testing.assert_close(wrapped_mt[0], max_transfer_mat)


def test_backward_explicit_family_idx_preserves_g_equals_s_row_intent():
    G = S = 3
    C = 5
    family_idx = torch.tensor(
        [[2, 99], [0, 99], [1, 99], [2, 99], [1, 99]],
        dtype=torch.int32,
    )[:, 0]
    family_species = torch.arange(G * S, dtype=DTYPE).reshape(G, S)
    ambiguous_row_values = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)

    (
        auto_wrapped,
        normalized_family_idx,
        wrapped_E,
        _wrapped_Ebar,
        _wrapped_E_s1,
        _wrapped_E_s2,
        _wrapped_log_pS,
        wrapped_log_pD,
        _wrapped_log_pL,
        _wrapped_mt,
    ) = backward._auto_wrap_backward_inputs(
        C=C,
        device=DEVICE,
        family_idx=family_idx,
        E=family_species,
        Ebar=family_species + 10.0,
        E_s1=family_species + 20.0,
        E_s2=family_species + 30.0,
        log_pS=ambiguous_row_values + 1.0,
        log_pD=ambiguous_row_values,
        log_pL=ambiguous_row_values + 2.0,
        max_transfer_mat=family_species + 40.0,
    )

    assert auto_wrapped is False
    assert normalized_family_idx.dtype == torch.long
    assert normalized_family_idx.is_contiguous()
    assert normalized_family_idx.tolist() == [2, 0, 1, 2, 1]
    assert wrapped_E.shape == (G, S)
    assert wrapped_log_pD.shape == (G,)
    torch.testing.assert_close(
        as_family_param(
            wrapped_log_pD,
            S=S,
            device=DEVICE,
            dtype=DTYPE,
            family_rows=wrapped_E.shape[0],
        ),
        ambiguous_row_values.reshape(G, 1),
    )


def test_backward_auto_wrap_treats_g_equals_s_vector_as_shared_species_row():
    S = 3
    C = 5
    species_values = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)

    (
        auto_wrapped,
        _family_idx,
        wrapped_E,
        _wrapped_Ebar,
        _wrapped_E_s1,
        _wrapped_E_s2,
        _wrapped_log_pS,
        wrapped_log_pD,
        _wrapped_log_pL,
        _wrapped_mt,
    ) = backward._auto_wrap_backward_inputs(
        C=C,
        device=DEVICE,
        family_idx=None,
        E=torch.arange(S, dtype=DTYPE),
        Ebar=torch.arange(S, dtype=DTYPE) + 10.0,
        E_s1=torch.arange(S, dtype=DTYPE) + 20.0,
        E_s2=torch.arange(S, dtype=DTYPE) + 30.0,
        log_pS=species_values + 1.0,
        log_pD=species_values,
        log_pL=species_values + 2.0,
        max_transfer_mat=torch.arange(S, dtype=DTYPE),
    )

    assert auto_wrapped is True
    assert wrapped_E.shape == (1, S)
    assert wrapped_log_pD.shape == (1, S)
    torch.testing.assert_close(
        as_family_param(
            wrapped_log_pD,
            S=S,
            device=DEVICE,
            dtype=DTYPE,
            family_rows=wrapped_E.shape[0],
        ),
        species_values.reshape(1, S),
    )
