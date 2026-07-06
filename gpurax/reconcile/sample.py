import math
import pathlib
import sys

import torch

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gpurec.api.model import GeneReconModel
from gpurec import sample_reconciliations


def reconcile_family(species_path, tree_path, rates, *, seed=0, device="cuda"):
    model = GeneReconModel(species_path, [tree_path], mode="global",
                            device=device, dtype=torch.float64)
    triple = torch.tensor([math.log2(rates[0]), math.log2(rates[1]), math.log2(rates[2])],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        model.theta.copy_(triple)
    return sample_reconciliations(model, family_index=0, seed=seed)
