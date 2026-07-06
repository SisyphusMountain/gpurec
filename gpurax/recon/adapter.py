import math
import sys
import pathlib

import torch

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)  # repo gpurec, not .venv (project rule)

from gpurec.api.model import GeneReconModel


class ReconBatch:
    def __init__(self, species_path, device="cuda"):
        self._species = species_path
        self._device = device

    def score(self, topology_paths, rates):
        model = GeneReconModel(
            self._species, list(topology_paths),
            mode="genewise", device=self._device, dtype=torch.float64,
        )
        theta = torch.empty_like(model.theta)  # (F, 3)
        for i, (D, L, T) in enumerate(rates):
            theta[i, 0] = math.log2(D)
            theta[i, 1] = math.log2(L)
            theta[i, 2] = math.log2(T)
        with torch.no_grad():
            model.theta.copy_(theta)
            nll_bits = model.genewise_loss_vector()  # [F] NLL bits
        return [(-float(b) * math.log(2.0)) for b in nll_bits]
