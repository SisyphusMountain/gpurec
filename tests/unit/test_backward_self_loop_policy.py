from pathlib import Path

import gpurec.core.backward as backward


def test_backward_module_omits_native_cuda_self_loop_helpers():
    for name in (
        "_OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS",
        "_cuda_self_loop_options_from_env",
        "_cuda_self_loop_wave_backend",
        "_cuda_self_loop_fallback_after_optional_failure",
    ):
        assert not hasattr(backward, name)


def test_native_cuda_self_loop_module_is_removed():
    root = Path(__file__).resolve().parents[2]

    assert not (root / "gpurec" / "core" / "kernels" / "wave_backward_cuda.py").exists()
