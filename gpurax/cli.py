"""Command-line interface mirroring GeneRax's argument set, wrapping the
gpurax driver (parse families -> Step 0 -> warm-up -> rates/SPR loop ->
final reconciliation -> export)."""

import argparse

from gpurax.driver import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpurax")
    parser.add_argument("-f", "--families", required=True)
    parser.add_argument("-s", "--species-tree", required=True)
    parser.add_argument("-r", "--rec-model", default="UndatedDTL")
    parser.add_argument("--max-spr-radius", type=int, default=5)
    parser.add_argument("--rec-radius", type=int, default=1)
    parser.add_argument("--rec-weight", type=float, default=1.0)
    parser.add_argument("-p", "--prefix", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    run(
        args.families,
        args.species_tree,
        args.prefix,
        rec_model=args.rec_model,
        max_spr_radius=args.max_spr_radius,
        rec_radius=args.rec_radius,
        rec_weight=args.rec_weight,
        device=args.device,
    )
    return 0
