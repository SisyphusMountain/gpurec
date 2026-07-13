import pytest
import torch

from gpurec.core.pi_state import PiState


def test_centered_rows_reconstruct_in_fp64_and_preserve_impossible_lanes():
    residual = torch.tensor([[0.25, -1.5, float("-inf")]], dtype=torch.float32)
    offset = torch.tensor([1200.125], dtype=torch.float64)

    absolute = PiState.reconstruct_rows(residual, offset)

    assert absolute.dtype == torch.float64
    torch.testing.assert_close(
        absolute[:, :2],
        torch.tensor([[1200.375, 1198.625]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isneginf(absolute[0, 2]).item()


def test_centered_rows_reject_ambiguous_offset_layout():
    residual = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="one value per residual row"):
        PiState.reconstruct_rows(
            residual, torch.zeros(1, dtype=torch.float64)
        )
    with pytest.raises(TypeError, match="fp64"):
        PiState.reconstruct_rows(
            residual, torch.zeros(2, dtype=torch.float32)
        )


@pytest.mark.gpu
def test_complete_centered_state_validates_frames_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pi = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
    state = PiState(
        pi_offset=torch.zeros(2, device="cuda", dtype=torch.float64),
        pibar_offset=torch.ones(2, device="cuda", dtype=torch.float64),
    )
    pibar = pi.clone()
    row_max = torch.zeros(2, device="cuda", dtype=torch.float32)

    state.validate(pi, pibar, row_max)
    torch.testing.assert_close(
        state.reconstruct_pibar(pibar), torch.ones_like(pi, dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.reconstruct_pibar_row_max(row_max),
        torch.zeros(2, device="cuda", dtype=torch.float64),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_centered_state_rejects_nonfinite_offsets_on_explicit_audit(bad_value: float):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pi = torch.zeros((1, 2), device="cuda", dtype=torch.float32)
    state = PiState(
        pi_offset=torch.tensor([bad_value], device="cuda", dtype=torch.float64),
        pibar_offset=torch.zeros(1, device="cuda", dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="finite"):
        state.validate(pi, pi.clone(), torch.zeros(1, device="cuda"))


@pytest.mark.gpu
def test_centered_state_rejects_noncanonical_all_impossible_row():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    impossible = torch.full((1, 2), float("-inf"), device="cuda")
    state = PiState(
        pi_offset=torch.ones(1, device="cuda", dtype=torch.float64),
        pibar_offset=torch.zeros(1, device="cuda", dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="canonical zero offset"):
        state.validate(impossible, impossible.clone(), torch.full((1,), float("-inf"), device="cuda"))
