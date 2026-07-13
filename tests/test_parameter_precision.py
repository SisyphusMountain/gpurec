import math

import torch

from gpurec.core.parameters.extract_parameters import (
    extract_parameters_uniform,
    origination_log_probs_from_weights,
    receiver_log_probs_from_weights,
)


def test_fp32_rate_softmax_is_correctly_rounded_from_fp64():
    generator = torch.Generator().manual_seed(17)
    theta = torch.randn((257, 3), generator=generator, dtype=torch.float32) * 4.0
    zeros = torch.zeros((257,), dtype=torch.float32)

    log_pS, log_pD, log_pL, _ = extract_parameters_uniform(
        theta, zeros, specieswise=True
    )
    logits64 = torch.cat(
        (torch.zeros((257, 1), dtype=torch.float64), theta.double()), dim=-1
    )
    reference = (
        torch.log_softmax(logits64 * math.log(2.0), dim=-1) / math.log(2.0)
    ).float()

    assert torch.equal(log_pS, reference[:, 0])
    assert torch.equal(log_pD, reference[:, 1])
    assert torch.equal(log_pL, reference[:, 2])


def test_fp32_receiver_and_origination_softmaxes_are_correctly_rounded():
    weights = torch.linspace(-12.5, 9.25, 1331, dtype=torch.float32)
    reference = (torch.log_softmax(weights.double(), dim=-1) / math.log(2.0)).float()

    assert torch.equal(receiver_log_probs_from_weights(weights), reference)
    assert torch.equal(origination_log_probs_from_weights(weights), reference)
