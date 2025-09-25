"""
JIT loader for the C++ backtracking extension using PyTorch's cpp_extension.

If the prebuilt module _ale_backtrack cannot be imported, this will compile it
on the fly the first time it's used.
"""
from pathlib import Path
from typing import Optional

def get_ale_backtrack(verbose: bool = False):
    try:
        import _ale_backtrack as mod  # noqa: F401
        return mod
    except Exception:
        pass
    try:
        from torch.utils.cpp_extension import load
    except Exception as e:
        raise RuntimeError("PyTorch cpp_extension not available to JIT-compile _ale_backtrack") from e
    # Source lives at repo_root/cpp/ale_backtrack.cpp
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # src/reconciliation -> src -> repo_root
    src_path = repo_root / 'cpp' / 'ale_backtrack.cpp'
    if not src_path.exists():
        raise FileNotFoundError(f"Missing C++ source: {src_path}")
    mod = load(
        name="_ale_backtrack",
        sources=[str(src_path)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return mod