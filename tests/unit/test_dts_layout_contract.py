from __future__ import annotations

import torch

from gpurec.core.kernels._dts_layout_contract import (
    DtsBackwardLayoutCode,
    DtsForwardAddressMode,
    dts_backward_param_layout,
    dts_forward_param_layout,
)
from gpurec.core.kernels.dts_fused import _prepare_param
from gpurec.core.kernels.wave_backward import (
    _dts_grad_layout,
    _dts_layout_param_args,
)


def test_forward_contract_keeps_family_indexed_1d_length_s_as_species_vector() -> None:
    param = torch.arange(6, dtype=torch.float32)[::2]

    layout = dts_forward_param_layout(param, S=3, family_indexed=True)
    prepared, mode, row_stride, species_stride = _prepare_param(
        param,
        n_splits=7,
        S=3,
        family_indexed=True,
    )

    assert layout.mode is DtsForwardAddressMode.FAMILY_INDEXED
    assert layout.intent == "shared_species"
    assert layout.ambiguous_1d_family_species is True
    assert prepared is param
    assert (mode, row_stride, species_stride) == (4, 0, 2)


def test_backward_contract_keeps_family_indexed_1d_length_s_as_family_scalar() -> None:
    family_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    log_pD = torch.arange(3, dtype=torch.float32)
    log_pS = torch.arange(10, 13, dtype=torch.float32)

    layout = dts_backward_param_layout(log_pD, S=3, family_indexed=True)
    pD_arg, pS_arg, layout_code = _dts_layout_param_args(
        log_pD,
        log_pS,
        family_idx=family_idx,
        S=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert layout.code is DtsBackwardLayoutCode.FAMILY_SCALAR
    assert layout.intent == "family_scalar"
    assert layout.ambiguous_1d_family_species is True
    assert layout_code == 2
    assert torch.equal(pD_arg, log_pD)
    assert torch.equal(pS_arg, log_pS)
    assert _dts_grad_layout(log_pD, family_idx=family_idx, S=3) == 2


def test_family_scalar_rows_are_unambiguous_when_kept_two_dimensional() -> None:
    family_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    param = torch.arange(3, dtype=torch.float32).reshape(3, 1)

    forward_layout = dts_forward_param_layout(param, S=3, family_indexed=True)
    backward_layout = dts_backward_param_layout(param, S=3, family_indexed=True)
    prepared, mode, row_stride, species_stride = _prepare_param(
        param,
        n_splits=5,
        S=3,
        family_indexed=True,
    )
    pD_arg, pS_arg, layout_code = _dts_layout_param_args(
        param,
        param + 10,
        family_idx=family_idx,
        S=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert forward_layout.intent == "family_scalar"
    assert forward_layout.ambiguous_1d_family_species is False
    assert backward_layout.code is DtsBackwardLayoutCode.FAMILY_SCALAR
    assert backward_layout.ambiguous_1d_family_species is False
    assert prepared is param
    assert (mode, row_stride, species_stride) == (4, 1, 0)
    assert layout_code == 2
    assert pD_arg.shape == (3,)
    assert pS_arg.shape == (3,)
    assert _dts_grad_layout(param, family_idx=family_idx, S=3) == 2


def test_family_species_rows_share_the_same_contract_codes() -> None:
    family_idx = torch.tensor([0, 1], dtype=torch.long)
    param = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    forward_layout = dts_forward_param_layout(param, S=3, family_indexed=True)
    backward_layout = dts_backward_param_layout(param, S=3, family_indexed=True)
    prepared, mode, row_stride, species_stride = _prepare_param(
        param,
        n_splits=4,
        S=3,
        family_indexed=True,
    )
    pD_arg, pS_arg, layout_code = _dts_layout_param_args(
        param,
        param + 10,
        family_idx=family_idx,
        S=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert forward_layout.intent == "family_species"
    assert backward_layout.code is DtsBackwardLayoutCode.FAMILY_SPECIES
    assert prepared is param
    assert (mode, row_stride, species_stride) == (4, 3, 1)
    assert layout_code == 3
    assert torch.equal(pD_arg, param)
    assert torch.equal(pS_arg, param + 10)
    assert _dts_grad_layout(param, family_idx=family_idx, S=3) == 3
