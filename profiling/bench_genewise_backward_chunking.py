#!/usr/bin/env python3
"""Dedicated genewise uniform backward chunking benchmark entrypoint.

This Proposal 7 wrapper delegates to the Proposal 2 benchmark implementation
so the profiling code stays in one place.  The defaults here are intentionally
strict: optimized genewise backward is selected, optimized guards are required,
and active-path preflight validation is enabled before stats or timings run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("BACKWARD_PATH", "optimized-genewise")
    os.environ.setdefault("STRICT_OPTIMIZED_KERNELS", "1")
    os.environ.setdefault("FAMILY_CHUNK_SIZE", "50")
    os.environ.setdefault("GPUREC_BENCH_VALIDATE_OPTIMIZED_FLAGS", "1")

    from profiling.proposal2.bench_genewise_backward import main as proposal2_main

    proposal2_main()


if __name__ == "__main__":
    main()
