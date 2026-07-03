#!/usr/bin/env python
"""Fit gpurec DTL rates on a dataset and write rates (AleRax format) + a JSON sidecar.

Thin wrapper over the `gpurec fit` library path so the scale kit and the CLI share code.
"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description="gpurec rate-fit benchmark driver")
    p.add_argument("--species", required=True)
    p.add_argument("--gene", required=True, nargs="+")
    p.add_argument("--mode", choices=["global", "specieswise", "genewise"], default="global")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--init-rate", type=float, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpurec.cli.fit import run_fit
    return run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
