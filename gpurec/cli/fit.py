"""`gpurec fit` — optimize DTL rates (implemented in Task 3)."""
from __future__ import annotations

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--init-rate", type=float, default=None)
    parser.add_argument("--out", default=None)


def run_fit(args) -> int:
    raise NotImplementedError("gpurec fit is implemented in Task 3")
