"""gpurec command-line entry point."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpurec", description="GPU DTL gene/species-tree reconciliation")
    sub = parser.add_subparsers(dest="command")
    from . import reconcile, fit
    reconcile.add_args(sub.add_parser("reconcile", help="marginal log-likelihood at fixed rates"))
    fit.add_args(sub.add_parser("fit", help="optimize DTL rates"))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reconcile":
        from . import reconcile
        return reconcile.run_reconcile(args)
    if args.command == "fit":
        from . import fit
        return fit.run_fit(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
