import math
import sys
import pathlib

import torch

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)  # repo gpurec, not .venv (project rule)

from gpurec.api.model import GeneReconModel
from gpurec.optim.optimize import optimize


def optimize_global_rates(species_path, tree_paths, init=(0.2, 0.3, 0.1), device="cuda"):
    """Fit global (D, L, T) rates that maximize the summed reconciliation likelihood over the
    given (fixed) gene trees -- GeneRax "Step 1" (optimize rates at fixed gene trees).

    Mirrors gpurec/cli/fit.py's global-mode optimizer call: Adam basin-entry (adaptive LR
    schedule) followed by a ridge-regularized Newton polish (``gpurec.optim.optimize.optimize``,
    ``polish_mode="ridge"``); ``newton_lanczos`` is theta-shape aware, so the global ``(3,)``
    theta polishes via the FD-Hessian path without special-casing here.
    """
    model = GeneReconModel(species_path, list(tree_paths), mode="global", device=device,
                           dtype=torch.float64)
    theta0 = torch.tensor(
        [math.log2(init[0]), math.log2(init[1]), math.log2(init[2])],
        dtype=model.theta.dtype, device=model.theta.device,
    )
    with torch.no_grad():
        model.theta.copy_(theta0)
    theta_hat, _hist = optimize(
        model.batch_statics, model.theta.detach(), model.receiver_weights.detach(),
        optimizer="adam", schedule="adaptive", max_steps=300, polish_mode="ridge",
        verbose=False,
    )
    D, L, T = (2.0 ** theta_hat.detach().cpu()).tolist()
    return D, L, T
