import math
from types import SimpleNamespace

import pytest
import torch

from gpurec.core.parameters.extract_parameters import (
    extract_parameters_uniform,
    origination_log_probs_from_weights,
    receiver_log_probs_from_weights,
)


def test_kernel_dtype_mapping_rejects_unsupported_types():
    from gpurec.core.kernels.e_step import _tl_float_dtype as e_step_dtype
    from gpurec.core.kernels.pi_forward import _tl_float_dtype as forward_dtype
    from gpurec.core.kernels.wave_backward import _tl_float_dtype as backward_dtype

    for mapper in (e_step_dtype, forward_dtype, backward_dtype):
        with pytest.raises(TypeError, match="kernel state|backward kernel state"):
            mapper(torch.float16)

    # The retained backward diagnostics deliberately accumulate bf16 state in
    # fp32; the forward and E-step recurrences do not accept bf16 state.
    assert backward_dtype(torch.bfloat16) is backward_dtype(torch.float32)
    with pytest.raises(TypeError, match="kernel state"):
        forward_dtype(torch.bfloat16)


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_fp32_rate_softmax_uses_configured_accumulator(accumulator_dtype):
    generator = torch.Generator().manual_seed(17)
    theta = torch.randn((257, 3), generator=generator, dtype=torch.float32) * 4.0
    zeros = torch.zeros((257,), dtype=torch.float32)

    log_pS, log_pD, log_pL, _ = extract_parameters_uniform(
        theta,
        zeros,
        specieswise=True,
        accumulator_dtype=accumulator_dtype,
    )
    logits = torch.cat(
        (
            torch.zeros((257, 1), dtype=accumulator_dtype),
            theta.to(dtype=accumulator_dtype),
        ),
        dim=-1,
    )
    reference = (
        torch.log_softmax(logits * math.log(2.0), dim=-1) / math.log(2.0)
    ).float()

    assert torch.equal(log_pS, reference[:, 0])
    assert torch.equal(log_pD, reference[:, 1])
    assert torch.equal(log_pL, reference[:, 2])


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_fp32_receiver_and_origination_softmaxes_use_configured_accumulator(
    accumulator_dtype,
):
    weights = torch.linspace(-12.5, 9.25, 1331, dtype=torch.float32)
    reference = (
        torch.log_softmax(weights.to(dtype=accumulator_dtype), dim=-1)
        / math.log(2.0)
    ).float()

    assert torch.equal(
        receiver_log_probs_from_weights(
            weights,
            accumulator_dtype=accumulator_dtype,
        ),
        reference,
    )
    assert torch.equal(
        origination_log_probs_from_weights(
            weights,
            accumulator_dtype=accumulator_dtype,
        ),
        reference,
    )


def test_small_softmax_default_derives_from_input_dtype():
    weights = torch.linspace(-2.0, 1.0, 17, dtype=torch.float32)
    assert torch.equal(
        receiver_log_probs_from_weights(weights),
        receiver_log_probs_from_weights(
            weights,
            accumulator_dtype=torch.float32,
        ),
    )


def test_weighted_exact_hvp_promotes_head_logits_before_small_softmax(monkeypatch):
    """The exact-HVP orchestration must not round a configured fp64 head back to fp32."""
    from gpurec.core.parameters import extract_parameters as parameter_module
    from gpurec.solver.hvp_exact import make_exact_hvp_single

    class _HeadObserved(RuntimeError):
        pass

    observed = {}

    def _observe(weights, *, accumulator_dtype=None):
        observed["weights_dtype"] = weights.dtype
        observed["accumulator_dtype"] = accumulator_dtype
        raise _HeadObserved

    monkeypatch.setattr(
        parameter_module,
        "origination_log_probs_from_weights",
        _observe,
    )

    pi_state = SimpleNamespace(
        pi_offset=torch.zeros(1, dtype=torch.float64),
        pibar_offset=torch.zeros(1, dtype=torch.float64),
        validate=lambda *_args, **_kwargs: None,
    )
    static = SimpleNamespace(
        accumulator_dtype=torch.float64,
        solver_options=SimpleNamespace(pi_iters=1),
        species_helpers={
            "S": 2,
            "sp_child1": torch.tensor([-1, -1]),
            "sp_child2": torch.tensor([-1, -1]),
            "sp_parent": torch.tensor([-1, -1]),
            "max_ancestor_depth": 1,
        },
        wave_layout={
            "leaf_species_index": torch.tensor([0], dtype=torch.int32),
            "root_clade_ids": torch.tensor([0]),
        },
        rate_family_idx=torch.tensor([0]),
    )
    saved = {
        "pi_wave": torch.zeros((1, 2), dtype=torch.float32),
        "pibar_wave": torch.zeros((1, 2), dtype=torch.float32),
        "pibar_row_max": torch.zeros(1, dtype=torch.float32),
        "pi_state": pi_state,
        "E": torch.zeros((1, 2), dtype=torch.float32),
    }

    with pytest.raises(_HeadObserved):
        make_exact_hvp_single(
            static,
            torch.zeros((2, 3), dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            saved,
            origination_weights=torch.tensor([-0.25, 0.25], dtype=torch.float32),
        )

    assert observed == {
        "weights_dtype": torch.float64,
        "accumulator_dtype": torch.float64,
    }
