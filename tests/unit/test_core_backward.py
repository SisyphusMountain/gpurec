import gpurec.core.backward as backward


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
