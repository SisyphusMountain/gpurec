"""Find where the exact forward departs from the converged log iteration at a rate corner (fp64).

Builds two fp64 genewise models over the same families (log@N sweeps vs exact), evaluates the
forward at one shared (D, L, T) log2-rate corner, and reports, per batch: the per-family NLL
difference, the number of clade rows whose absolute log2 Pi differs by more than `--tol`, the first
such row in wave order (wave index, row, worst lane, both values), and the exact path's flag count.

Usage: corner_row_probe.py --species S --families LIST --limit N --corner=D,L,T --sweeps 16384 --tol 1e-6
"""
from __future__ import annotations

import argparse
import sys

import torch


def _build(species, paths, forward, adjoint, pi_iters, neumann_terms):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": pi_iters, "neumann_terms": neumann_terms,
                          "forward_self_loop": forward, "adjoint_self_loop": adjoint})
    m = GeneReconModel(species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    return m


def _rows(model, theta):
    from gpurec.core.inference.solver import solve_resident_e_pi
    out = []
    for static in model.batch_statics:
        with torch.no_grad():
            res = solve_resident_e_pi(static, model._theta_for_static(static, theta),
                                      model.receiver_weights.detach(), warm_start_E=None,
                                      pi_iters=None, pi_residual_out=None)
        st = static.pi_forward_state
        pi_abs = res[5].double() + st.pi_offset.double().unsqueeze(1)
        out.append((pi_abs.clone(), int(getattr(st, "wide_row_total", 0)), static))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--corner", required=True)
    ap.add_argument("--sweeps", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--rows", required=True, type=int, help="1 = also do the row-level comparison (a second forward per model)")
    ap.add_argument("--window", required=True, type=float, help="only lanes within this many log2 of the row max count (1e9 = whole row)")
    ap.add_argument("--first-waves", required=True, type=int, help="print a windowless per-wave summary for this many leading waves")
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    d, l, t = (float(x) for x in args.corner.split(","))
    m_log = _build(args.species, paths, "log", "series", args.sweeps, 512)
    m_ex = _build(args.species, paths, "exact", "exact", 16, 16)
    theta = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(len(paths), 1).contiguous()
    nll_log = m_log.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].double()
    nll_ex = m_ex.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].double()
    dn = nll_ex - nll_log
    print(f"[row] corner D={d} L={l} T={t} families={len(paths)}  sum NLL log={float(nll_log.sum()):.6f} exact={float(nll_ex.sum()):.6f}  "
          f"per-family dNLL: max {float(dn.abs().max()):.3e} (family #{int(dn.abs().argmax())}), n families |d|>1e-6: {int((dn.abs() > 1e-6).sum())}", flush=True)
    bad = (dn.abs() > 1e-6).nonzero().flatten().tolist()
    print(f"[row] differing families (index: dNLL bits, path): " +
          "; ".join(f"{i}: {float(dn[i]):+.4f} {paths[i].split('/')[-1][:12]}" for i in bad), flush=True)
    if args.rows == 0:
        return 0
    rows_log = _rows(m_log, theta)
    rows_ex = _rows(m_ex, theta)
    for b, ((pl, _, st), (pe, flagged, _)) in enumerate(zip(rows_log, rows_ex)):
        both = torch.isfinite(pl) & torch.isfinite(pe)
        diff = torch.where(both, (pe - pl).abs(), torch.zeros_like(pl))
        row_max = torch.where(torch.isfinite(pl), pl, torch.full_like(pl, -float("inf"))).amax(dim=1, keepdim=True)
        inwin = both & (pl >= row_max - args.window)
        bad_rows = (torch.where(inwin, diff, torch.zeros_like(diff)) > args.tol).any(dim=1)
        lost = int((torch.isfinite(pl) & ~torch.isfinite(pe) & (pl >= row_max - args.window)).sum())
        print(f"[row] batch {b}: rows={pl.shape[0]} flagged(exact)={flagged} rows with in-window |dPi|>{args.tol:g}: {int(bad_rows.sum())}  "
              f"in-window lanes lost by exact: {lost}  max in-window |dPi|={float(torch.where(inwin, diff, torch.zeros_like(diff)).max()):.3e}", flush=True)
        wl = st.wave_layout
        metas = wl["wave_metas"]
        print(f"[row]   leading waves, whole row (wave, rows, split?, max|dPi| both-finite, depth below row max of that lane, "
              f"n lanes finite only in log, finite only in exact, worst depth-below-max among finite-only-in-log lanes):", flush=True)
        for i, mt in enumerate(metas[: args.first_waves]):
            a, z = mt["start"], mt["end"]
            L, X = pl[a:z], pe[a:z]
            bothf = torch.isfinite(L) & torch.isfinite(X)
            dd = torch.where(bothf, (X - L).abs(), torch.zeros_like(L))
            r, c = divmod(int(dd.argmax()), L.shape[1])
            only_log = torch.isfinite(L) & ~torch.isfinite(X)
            only_ex = ~torch.isfinite(L) & torch.isfinite(X)
            rm = torch.where(torch.isfinite(L), L, torch.full_like(L, -float("inf"))).amax(dim=1, keepdim=True)
            depth_only_log = float((rm - L)[only_log].min()) if bool(only_log.any()) else float("nan")
            print(f"[row]     ({i}, {z-a}, {bool(mt.get('has_splits'))}, {float(dd.max()):.3e}, {float(rm[r] - L[r, c]):.1f}, "
                  f"{int(only_log.sum())}, {int(only_ex.sum())}, {depth_only_log:.1f})", flush=True)
        if bool(bad_rows.any()):
            first = int(bad_rows.nonzero()[0])
            wave = next((i for i, mt in enumerate(metas) if mt["start"] <= first < mt["end"]), -1)
            lane = int(torch.where(inwin[first], diff[first], torch.zeros_like(diff[first])).argmax())
            print(f"[row]   first bad row {first} is in wave {wave} of {len(metas)} (wave start={metas[wave]['start'] if wave>=0 else '?'}, "
                  f"has_splits={metas[wave].get('has_splits') if wave>=0 else '?'}); worst lane {lane}: log={float(pl[first, lane]):.6f} exact={float(pe[first, lane]):.6f}  "
                  f"row max (log)={float(row_max[first]):.4f}", flush=True)
            dwin = torch.where(inwin, diff, torch.zeros_like(diff))
            worst_row = int(dwin.amax(dim=1).argmax()); worst_lane = int(dwin[worst_row].argmax())
            wave_w = next((i for i, mt in enumerate(metas) if mt["start"] <= worst_row < mt["end"]), -1)
            print(f"[row]   WORST row {worst_row} (wave {wave_w}, has_splits={metas[wave_w].get('has_splits')}), lane {worst_lane}: "
                  f"log={float(pl[worst_row, worst_lane]):.4f} exact={float(pe[worst_row, worst_lane]):.4f} row max (log)={float(row_max[worst_row]):.4f} "
                  f"(exact row max={float(torch.where(torch.isfinite(pe[worst_row]), pe[worst_row], torch.full_like(pe[worst_row], -float('inf'))).max()):.4f})", flush=True)
            per_wave = []
            for i, mt in enumerate(metas):
                seg = dwin[mt["start"]:mt["end"]]
                if seg.numel():
                    per_wave.append((i, float(seg.max()), int((seg.amax(dim=1) > args.tol).sum()), mt["end"] - mt["start"], bool(mt.get("has_splits"))))
            print("[row]   per-wave (wave, max|dPi|, n bad rows, rows, split?):", flush=True)
            for rec in per_wave:
                if rec[1] > args.tol:
                    print(f"[row]     {rec}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
