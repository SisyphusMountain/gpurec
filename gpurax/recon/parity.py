import math
import sys
import pathlib

import torch

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)  # repo gpurec, not .venv (project rule)

from gpurec.api.model import GeneReconModel


def recon_loglk(species_path, gene_tree_path, D, L, T, *, device="cuda"):
    model = GeneReconModel(
        species_path, [gene_tree_path], mode="global", device=device, dtype=torch.float64
    )
    triple = torch.tensor(
        [math.log2(D), math.log2(L), math.log2(T)],
        dtype=model.theta.dtype,
        device=model.theta.device,
    )
    with torch.no_grad():
        model.theta.copy_(triple)
        nll_bits = float(model())  # summed scalar NLL in bits
    return -nll_bits * math.log(2.0)  # -> nats logL (gpurec/AleRax convention)
