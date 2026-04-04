"""Adapter for importing the logmatmul library without sys.modules hacking.

logmatmul lives under ``<repo>/logmatmul/src/`` and uses relative imports
(``from .dense import ...``).  Its top-level package is called ``src``,
which collides with this project's own ``src`` package.

This module performs the import *once* with a clean sys.modules swap and
exposes the symbols that the rest of the codebase needs.  Every other
module should do::

    from ._logmatmul_compat import HAS_LOGMATMUL, LogspaceMatmulFn, ...
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_logmatmul_dir = str(Path(__file__).resolve().parents[2] / "logmatmul")

HAS_LOGMATMUL: bool = False
LogspaceMatmulFn = None
streaming_topk = None
logspace_matmul_compressed = None

try:
    _saved_src = sys.modules.get("src")

    # Temporarily remove this project's ``src`` so importlib can load
    # logmatmul's own ``src`` package.
    if "src" in sys.modules:
        del sys.modules["src"]

    if _logmatmul_dir not in sys.path:
        sys.path.insert(0, _logmatmul_dir)

    importlib.import_module("src")
    LogspaceMatmulFn = importlib.import_module("src.autograd").LogspaceMatmulFn
    streaming_topk = importlib.import_module("src.sparse").streaming_topk
    logspace_matmul_compressed = importlib.import_module("src.compressed").logspace_matmul_compressed
    HAS_LOGMATMUL = True

    # Restore this project's ``src`` package.
    if _saved_src is not None:
        sys.modules["src"] = _saved_src

except (ImportError, FileNotFoundError, ModuleNotFoundError):
    HAS_LOGMATMUL = False
    LogspaceMatmulFn = None
    streaming_topk = None
    logspace_matmul_compressed = None
