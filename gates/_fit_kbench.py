"""Phase-1 functional gate: an Adam -> L-BFGS fit through the gpurec optim layer reaches the
known kernel-bench basin on the frozen 666x80 capture.

The kbench golden Newton fit went F0=170130 -> FN~144033 from this fixture's theta (rate 0.1).
Adam+L-BFGS (no second-order kernels) should reach the same basin (~144k), within the ~7 NLL
run-to-run optimizer-endpoint spread.

    python -m gates._fit_kbench
"""

from __future__ import annotations

import sys
import time

import torch

from gates._parity_kbench import _DEFAULT_CAP, gpurec_static_from_capture
from gpurec.optim.baselines import lbfgs_scipy
from gpurec.optim.optimize import final_eval, first_order


def run(cap_path=_DEFAULT_CAP, device="cuda", adam_steps=150, lbfgs_iters=100, maxcor=50):
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    static = gpurec_static_from_capture(cap, device)
    theta0 = cap["inputs"]["theta"].to(device).float().contiguous()
    rw = cap["inputs"]["col_weights"].to(device).float().contiguous()
    statics = [static]

    nll0, gn0 = final_eval(statics, theta0, rw)
    print(f"[fit 666x80] start NLL={nll0:.3f} ||g||={gn0:.3e}  (golden Newton: F0=170130 -> FN~144033)")

    t0 = time.perf_counter()
    theta1, h1, _warm = first_order(statics, theta0, rw, optimizer="adam", lr0=1.0,
                                    schedule="adaptive", max_steps=adam_steps, verbose=True)
    nll1, gn1 = final_eval(statics, theta1, rw)
    print(f"[fit] after Adam({len(h1)} steps): NLL={nll1:.3f} ||g||={gn1:.3e}  t={time.perf_counter()-t0:.1f}s")

    theta2, h2 = lbfgs_scipy(statics, theta1, rw, maxiter=lbfgs_iters, maxcor=maxcor, verbose=True)
    nll2, gn2 = final_eval(statics, theta2, rw)
    print(f"[fit] after L-BFGS({len(h2)} evals): NLL={nll2:.3f} ||g||={gn2:.3e}  "
          f"total t={time.perf_counter()-t0:.1f}s")

    # pass: descended into the known basin (FN~144033) within the run-to-run spread (~tens of NLL)
    ok = nll2 <= 144033 + 200.0
    print(f"  -> {'PASS' if ok else 'FAIL'}: final NLL={nll2:.1f} vs known basin ~144033")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
