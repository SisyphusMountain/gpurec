"""C++-accelerated preprocessing pipeline for CCP construction."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

from torch.utils.cpp_extension import load


_CPP_DIR = pathlib.Path(__file__).resolve().parent / "cpp"
_CPP_SRC = _CPP_DIR / "preprocess.cpp"


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    sources = [
        str(_CPP_SRC),
        str(_CPP_DIR / "tree_utils.cpp"),
        str(_CPP_DIR / "clade_utils.cpp"),
    ]
    return load(
        name="preprocess_cpp",
        sources=sources,
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )
