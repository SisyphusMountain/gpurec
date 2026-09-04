"""Which path is the fixed point at a rate corner: the exact solve or the converged log sweeps?

For a set of waves, apply ONE log-space sweep (the production log kernel) to each path's published
answer for that wave and measure how much the row moves. A true fixed point of the log kernel's
equation does not move (residual ~ 1e-12 in fp64). Runs on fp64 models; the exact path's sweep is
run right after its solve inside the forward (monkeypatched), so the sweep sees the same
upstream waves the solve saw.

Usage: corner_residual_probe.py --species S --families LIST --corner=D,L,T --sweeps 1024 --waves 29,51,60
"""
from __future__ import annotations

import argparse
import sys

import torch

import gpurec.core.inference.forward as fwd

WINDOW = 60.0


def _build(species, paths, forward, adjoint, pi_iters):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": pi_iters, "neumann_terms": 16,
                          "forward_self_loop": forward, "adjoint_self_loop": adjoint})
    m = GeneReconModel(species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    return m


def _abs_rows(pi, off, ws, W):
    return pi[ws:ws + W].double() + off[ws:ws + W].double().unsqueeze(1)


class Recorder:
    def __init__(self, targets):
        self.targets = targets          # set of wave starts
        self.step_args = {}             # ws -> (args, kwargs) of the prologue log step
        self.out = {}                   # ws -> dict(published=abs rows, swept=abs rows after one log sweep)
        self.orig_step = fwd.compute_wave_step
        self.orig_exact = fwd.compute_exact_tree_self_loop

    def one_sweep_from(self, ws, pi, pi_off, pibar_template, pibar_off_template):
        a, k = self.step_args[ws]
        W = a[7]
        out = pibar_template.clone()
        out_off = pibar_off_template.clone()
        k = dict(k)
        k.update(store_final_pibar=False, input_ws=None, pi_residual_out=None, row_mask=None,
                 pibar_row_max=k["pibar_row_max"].clone())
        self.orig_step(pi, pi_off, out, out_off, out, out_off, *a[6:], **k)
        return _abs_rows(out, out_off, ws, W)

    def patched_step(self, *a, **k):
        ws = a[6]
        if k.get("input_ws") == 0 and ws in self.targets:
            self.step_args[ws] = (a, k)
        self.orig_step(*a, **k)
        if k.get("store_final_pibar") and ws in self.targets and ws in self.step_args:
            # log path: a[2]/a[3] hold the final iterate for this wave
            W = a[7]
            published = _abs_rows(a[2], a[3], ws, W)
            swept = self.one_sweep_from(ws, a[2], a[3], a[4], a[5])
            self.out[ws] = dict(published=published.clone(), swept=swept)

    def patched_exact(self, *a, **k):
        self.orig_exact(*a, **k)
        ws, W = a[7], a[8]
        if ws in self.targets and ws in self.step_args:
            published = _abs_rows(a[2], a[3], ws, W)
            swept = self.one_sweep_from(ws, a[2], a[3], a[4], a[5])
            self.out[ws] = dict(published=published.clone(), swept=swept)

    def __enter__(self):
        fwd.compute_wave_step = self.patched_step
        fwd.compute_exact_tree_self_loop = self.patched_exact
        return self

    def __exit__(self, *exc):
        fwd.compute_wave_step = self.orig_step
        fwd.compute_exact_tree_self_loop = self.orig_exact


def _forward(model, theta):
    from gpurec.core.inference.solver import solve_resident_e_pi
    static = model.batch_statics[0]
    with torch.no_grad():
        solve_resident_e_pi(static, model._theta_for_static(static, theta),
                            model.receiver_weights.detach(), warm_start_E=None,
                            pi_iters=None, pi_residual_out=None)
    return static


def _report(label, rec, ws, other=None):
    pub, sw = rec.out[ws]["published"], rec.out[ws]["swept"]
    fin = torch.isfinite(pub) & torch.isfinite(sw)
    row_max = torch.where(torch.isfinite(pub), pub, torch.full_like(pub, -float("inf"))).amax(dim=1, keepdim=True)
    inwin = fin & (pub >= row_max - WINDOW)
    res = torch.where(inwin, (sw - pub).abs(), torch.zeros_like(pub))
    r, s = divmod(int(res.argmax()), pub.shape[1])
    lost = int((torch.isfinite(pub) & ~torch.isfinite(sw) & (pub >= row_max - WINDOW)).sum())
    line = (f"[res] {label:5s} wave start {ws}: rows={pub.shape[0]} max in-window |one sweep - published| = {float(res.max()):.3e} "
            f"at row+{r} lane {s} (published {float(pub[r, s]):.6f}, swept {float(sw[r, s]):.6f}, row max {float(row_max[r]):.4f}); "
            f"in-window lanes the sweep sends to -inf: {lost}")
    if other is not None:
        opub = other.out[ws]["published"]
        both = torch.isfinite(pub) & torch.isfinite(opub) & inwin
        d = torch.where(both, (pub - opub).abs(), torch.zeros_like(pub))
        r2, s2 = divmod(int(d.argmax()), pub.shape[1])
        line += (f"\n[res]       exact-vs-log at this wave: max in-window |dPi| = {float(d.max()):.3e} at row+{r2} lane {s2} "
                 f"(this path {float(pub[r2, s2]):.6f}, other {float(opub[r2, s2]):.6f}); "
                 f"this path's residual there {float((sw - pub)[r2, s2]):+.3e}, other's {float((other.out[ws]['swept'] - opub)[r2, s2]):+.3e}")
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--corner", required=True)
    ap.add_argument("--sweeps", required=True, type=int)
    ap.add_argument("--waves", required=True, help="comma-separated wave indices")
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    d, l, t = (float(x) for x in args.corner.split(","))
    waves = [int(x) for x in args.waves.split(",")]
    m_ex = _build(args.species, paths, "exact", "exact", 16)
    theta = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(len(paths), 1).contiguous()
    metas = m_ex.batch_statics[0].wave_layout["wave_metas"]
    targets = {int(metas[w]["start"]) for w in waves}
    print(f"[res] corner D={d} L={l} T={t}; waves {waves} -> starts {sorted(targets)} (of {len(metas)} waves)", flush=True)
    with Recorder(targets) as rec_ex:
        _forward(m_ex, theta)
    del m_ex
    torch.cuda.empty_cache()
    m_log = _build(args.species, paths, "log", "series", args.sweeps)
    with Recorder(targets) as rec_log:
        _forward(m_log, theta)
    for w in waves:
        ws = int(metas[w]["start"])
        print(f"[res] --- wave {w} ---", flush=True)
        _report("exact", rec_ex, ws, other=rec_log)
        _report("log", rec_log, ws, other=rec_ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
