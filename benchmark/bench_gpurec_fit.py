#!/usr/bin/env python
"""Fit gpurec DTL rates on a dataset and write rates (AleRax format) + a JSON sidecar.

Thin wrapper over the `gpurec fit` library path so the scale kit and the CLI share code.
"""
import argparse
import sys


def build_parser():
    from gpurec.cli import _common     # import-safe: _common has no heavy/torch imports at module scope
    p = argparse.ArgumentParser(description="gpurec rate-fit benchmark driver")
    _common.add_common_args(p)         # --species --gene --mode --device --dtype --pi-iters --neumann-terms --e-max-iter
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--init-rate", type=float, default=None)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpurec.cli.fit import run_fit
    return run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
