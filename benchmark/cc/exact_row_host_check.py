"""Host-side (fp64, CPU) check of the exact tree solve on one captured clade row.

Captures the exact kernel's inputs for one wave (monkeypatch), then for one row of that wave:
  1. reproduces the kernel's CURRENT scheme (alpha/gamma, U0/U1 with subtraction, M/R rebuild),
  2. runs the ADDITIVE scheme (subtree masses carried as affine functions of the incoming mass),
  3. scores every candidate row (kernel output, 1, 2) by the per-lane RELATIVE residual of the
     fixed-point equation, evaluated with additions only (bottom-up subtree masses, top-down
     off-chain masses), so a lane 300 binary orders under the row maximum is judged on its own
     scale.
Usage: exact_row_host_check.py --species S --families LIST --corner=D,L,T --wave 25 --row 121
"""
from __future__ import annotations

import argparse
import sys

import torch

import gpurec.core.inference.forward as fwd


def _build(species, paths):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16,
                          "forward_self_loop": "exact", "adjoint_self_loop": "exact"})
    m = GeneReconModel(species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    return m


class Capture:
    def __init__(self, ws):
        self.ws = ws
        self.orig = fwd.compute_exact_tree_self_loop
        self.data = None

    def __call__(self, *a, **k):
        if a[7] == self.ws:
            W = a[8]
            snap = dict(Pi_in=a[0][self.ws:self.ws + W].clone(), Pi_in_offset=a[1][self.ws:self.ws + W].clone(),
                        diag=a[10].clone(), q=a[11].clone(), sl1=a[12].clone(), sl2=a[13].clone(), mt=a[14].clone(),
                        recv=a[15].clone(), child1=a[16].clone(), child2=a[17].clone(), level_ptr=a[18].clone(),
                        level_parents=a[19].clone(), level_child1=a[20].clone(), level_child2=a[21].clone(),
                        dts=a[22].clone(), dts_offset=a[23].clone(), dts_center=a[24].clone(),
                        family_idx=k["family_idx"][self.ws:self.ws + W].clone(), has_leaf=k["has_leaf_term"],
                        use_recv=k["use_receiver_weights"], S=a[9], W=W)
            self.orig(*a, **k)
            snap["Pi_out"] = a[2][self.ws:self.ws + W].clone()
            snap["Pi_out_offset"] = a[3][self.ws:self.ws + W].clone()
            snap["Pibar_out"] = a[4][self.ws:self.ws + W].clone()
            snap["Pibar_offset"] = a[5][self.ws:self.ws + W].clone()
            self.data = snap
        else:
            self.orig(*a, **k)


def levels_of(d):
    lp = d["level_ptr"].tolist()
    out = []
    for i in range(len(lp) - 1):
        sl = slice(lp[i], lp[i + 1])
        out.append((d["level_parents"][sl].long(), d["level_child1"][sl].long(), d["level_child2"][sl].long()))
    return out


def solve_current(src, diag, q, sl1, sl2, recv, c1, c2, S, levels):
    """The kernel's 4-walk scheme, transcribed."""
    alpha = torch.zeros(S, dtype=torch.float64); gamma = torch.zeros_like(alpha)
    leaf = c1 >= S
    alpha[leaf] = src[leaf] / diag[leaf]; gamma[leaf] = q[leaf] / diag[leaf]
    for par, k1, k2 in levels:
        m1, m2 = k1 < S, k2 < S
        a1 = torch.where(m1, alpha[k1.clamp(max=S - 1)], torch.zeros_like(alpha[par])); g1 = torch.where(m1, gamma[k1.clamp(max=S - 1)], 0.0)
        a2 = torch.where(m2, alpha[k2.clamp(max=S - 1)], torch.zeros_like(alpha[par])); g2 = torch.where(m2, gamma[k2.clamp(max=S - 1)], 0.0)
        Aa = sl1[par] * a1 + sl2[par] * a2; G = sl1[par] * g1 + sl2[par] * g2
        pivot = diag[par] + G * recv[par]
        alpha[par] = (src[par] + Aa) / pivot; gamma[par] = (q[par] + G) / pivot
    u0 = torch.zeros(S, dtype=torch.float64); u1 = torch.ones(S, dtype=torch.float64)
    for par, k1, k2 in reversed(levels):
        cu0 = u0[par] - recv[par] * (alpha[par] + gamma[par] * u0[par])
        cu1 = u1[par] - recv[par] * (gamma[par] * u1[par])
        for kk in (k1, k2):
            m = kk < S
            u0[kk[m]] = cu0[m]; u1[kk[m]] = cu1[m]
    p0 = recv * (alpha + gamma * u0); p1 = recv * gamma * u1
    T = p0.sum() / (1.0 - p1.sum())
    first = alpha + gamma * (u0 + u1 * T)
    M = recv * torch.where(first > 0, first, torch.zeros_like(first))
    for par, k1, k2 in levels:
        M[par] = M[par] + torch.where(k1 < S, M[k1.clamp(max=S - 1)], 0.0) + torch.where(k2 < S, M[k2.clamp(max=S - 1)], 0.0)
    R = torch.zeros(S, dtype=torch.float64)
    for par, k1, k2 in reversed(levels):
        M1 = torch.where(k1 < S, M[k1.clamp(max=S - 1)], 0.0); M2 = torch.where(k2 < S, M[k2.clamp(max=S - 1)], 0.0)
        m1, m2 = k1 < S, k2 < S
        R[k1[m1]] = (R[par] + M2)[m1]; R[k2[m2]] = (R[par] + M1)[m2]
    p = alpha + gamma * (R + M)
    return p, T


def solve_additive(src, diag, q, sl1, sl2, recv, c1, c2, S, levels):
    """Subtree receiver mass as an affine function of the incoming available mass: M = mA + mG*u."""
    alpha = torch.zeros(S, dtype=torch.float64); gamma = torch.zeros_like(alpha)
    mA = torch.zeros_like(alpha); mG = torch.zeros_like(alpha)
    leaf = c1 >= S
    alpha[leaf] = src[leaf] / diag[leaf]; gamma[leaf] = q[leaf] / diag[leaf]
    mA[leaf] = recv[leaf] * alpha[leaf]; mG[leaf] = recv[leaf] * gamma[leaf]
    for par, k1, k2 in levels:
        m1, m2 = k1 < S, k2 < S
        i1, i2 = k1.clamp(max=S - 1), k2.clamp(max=S - 1)
        a1 = torch.where(m1, alpha[i1], 0.0); g1 = torch.where(m1, gamma[i1], 0.0)
        a2 = torch.where(m2, alpha[i2], 0.0); g2 = torch.where(m2, gamma[i2], 0.0)
        A1 = torch.where(m1, mA[i1], 0.0); G1 = torch.where(m1, mG[i1], 0.0)
        A2 = torch.where(m2, mA[i2], 0.0); G2 = torch.where(m2, mG[i2], 0.0)
        Aa = sl1[par] * a1 + sl2[par] * a2; G = sl1[par] * g1 + sl2[par] * g2
        pivot = diag[par] + G * recv[par]
        alpha[par] = (src[par] + Aa) / pivot; gamma[par] = (q[par] + G) / pivot
        rg = recv[par] * gamma[par]
        mG[par] = rg + (G1 + G2) * (1.0 - rg)
        mA[par] = recv[par] * alpha[par] * (1.0 - G1 - G2) + A1 + A2
    root = levels[-1][0]
    assert root.numel() == 1
    T = mA[root] / (1.0 - mG[root])
    u = torch.zeros(S, dtype=torch.float64); R = torch.zeros(S, dtype=torch.float64)
    u[root] = T
    valid = torch.zeros(S, dtype=torch.float64)
    for par, k1, k2 in reversed(levels):
        m1, m2 = k1 < S, k2 < S
        i1, i2 = k1.clamp(max=S - 1), k2.clamp(max=S - 1)
        A1 = torch.where(m1, mA[i1], 0.0); G1 = torch.where(m1, mG[i1], 0.0)
        A2 = torch.where(m2, mA[i2], 0.0); G2 = torch.where(m2, mG[i2], 0.0)
        v = (R[par] + A1 + A2) / (1.0 - G1 - G2)
        valid[par] = v
        u[k1[m1]] = v[m1]; u[k2[m2]] = v[m2]
        R[k1[m1]] = (R[par] + A2 + G2 * v)[m1]; R[k2[m2]] = (R[par] + A1 + G1 * v)[m2]
    valid[leaf] = R[leaf]
    p = alpha + gamma * u
    return p, T, valid


def residual(p, src, diag, q, sl1, sl2, recv, c1, c2, S, levels):
    """Per-lane relative residual of the fixed point, with the transfer mass built by additions."""
    M = recv * p
    for par, k1, k2 in levels:
        M[par] = M[par] + torch.where(k1 < S, M[k1.clamp(max=S - 1)], 0.0) + torch.where(k2 < S, M[k2.clamp(max=S - 1)], 0.0)
    R = torch.zeros(S, dtype=torch.float64)
    for par, k1, k2 in reversed(levels):
        M1 = torch.where(k1 < S, M[k1.clamp(max=S - 1)], 0.0); M2 = torch.where(k2 < S, M[k2.clamp(max=S - 1)], 0.0)
        m1, m2 = k1 < S, k2 < S
        R[k1[m1]] = (R[par] + M2)[m1]; R[k2[m2]] = (R[par] + M1)[m2]
    i1, i2 = c1.clamp(max=S - 1), c2.clamp(max=S - 1)
    M1 = torch.where(c1 < S, M[i1], 0.0); M2 = torch.where(c2 < S, M[i2], 0.0)
    p1 = torch.where(c1 < S, p[i1], 0.0); p2 = torch.where(c2 < S, p[i2], 0.0)
    valid = R + M1 + M2
    stay = 1.0 - diag + q * recv                      # dl + ebar
    p_new = src + stay * p + q * valid + sl1 * p1 + sl2 * p2
    return (p_new - p).abs() / p.abs().clamp_min(torch.finfo(torch.float64).tiny), valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--corner", required=True); ap.add_argument("--wave", required=True, type=int)
    ap.add_argument("--row", required=True, type=int, help="row within the wave")
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    d, l, t = (float(x) for x in args.corner.split(","))
    m = _build(args.species, paths)
    theta = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(len(paths), 1).contiguous()
    static = m.batch_statics[0]
    metas = static.wave_layout["wave_metas"]
    ws = int(metas[args.wave]["start"])
    cap = Capture(ws)
    fwd.compute_exact_tree_self_loop = cap
    from gpurec.core.inference.solver import solve_resident_e_pi
    with torch.no_grad():
        solve_resident_e_pi(static, m._theta_for_static(static, theta), m.receiver_weights.detach(),
                            warm_start_E=None, pi_iters=None, pi_residual_out=None)
    fwd.compute_exact_tree_self_loop = cap.orig
    dd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cap.data.items()}
    S, w = dd["S"], args.row
    assert not dd["has_leaf"], "this check handles split waves"
    fam = int(dd["family_idx"][w])
    pin = dd["Pi_in"][w].double(); pin_off = float(dd["Pi_in_offset"][w])
    scale = max(float(pin[torch.isfinite(pin)].max()) + pin_off, float(dd["dts_center"][w]))
    src = torch.exp2(dd["dts"][w].double() + (float(dd["dts_offset"][w]) - scale))
    src = torch.where(torch.isfinite(dd["dts"][w]), src, torch.zeros_like(src))
    cst = lambda name: dd[name][fam].double() if dd[name].dim() == 2 else dd[name].double()
    diag, q, sl1, sl2, mt = cst("diag"), cst("q"), cst("sl1"), cst("sl2"), cst("mt")
    recv = dd["recv"].double() if dd["use_recv"] else torch.ones(S, dtype=torch.float64)
    if recv.dim() == 2:
        recv = recv[fam]
    c1, c2 = dd["child1"].long(), dd["child2"].long()
    levels = levels_of(dd)
    kern = torch.exp2(dd["Pi_out"][w].double() + float(dd["Pi_out_offset"][w]) - scale)
    kern_pibar = torch.exp2(dd["Pibar_out"][w].double() + float(dd["Pibar_offset"][w]) - scale)
    cur, T_cur = solve_current(src, diag, q, sl1, sl2, recv, c1, c2, S, levels)
    add, T_add, valid_add = solve_additive(src, diag, q, sl1, sl2, recv, c1, c2, S, levels)
    print(f"[host] wave {args.wave} (start {ws}) row +{w}: S={S} scale={scale:.4f} src>0 lanes={int((src>0).sum())} "
          f"T current={float(T_cur):.17g} additive={float(T_add):.17g}", flush=True)
    l2 = lambda x: torch.log2(x.clamp_min(torch.finfo(torch.float64).tiny))
    depth = l2(kern).max() - l2(kern)
    for label, cand in (("kernel", kern), ("host-current", cur), ("host-additive", add)):
        r, valid = residual(cand, src, diag, q, sl1, sl2, recv, c1, c2, S, levels)
        worst = int(r.argmax())
        print(f"[host] {label:14s}: max relative residual {float(r.max()):.3e} at lane {worst} (depth {float(depth[worst]):.1f} below max); "
              f"lanes with rel. residual > 1e-9: {int((r > 1e-9).sum())}; negative lanes: {int((cand < 0).sum())}; zero lanes: {int((cand == 0).sum())}", flush=True)
    dk = (l2(cur) - l2(kern)).abs()
    print(f"[host] host-current vs kernel: max |dlog2| {float(dk.max()):.3e} (kernel transcription check)", flush=True)
    da = (l2(add) - l2(kern)).abs()
    print(f"[host] host-additive vs kernel: max |dlog2| {float(da.max()):.3e}; by depth below row max:", flush=True)
    for lo, hi in ((0, 30), (30, 50), (50, 60), (60, 80), (80, 120), (120, 200), (200, 400), (400, 2000)):
        sel = (depth >= lo) & (depth < hi)
        if bool(sel.any()):
            print(f"[host]     depth [{lo},{hi}): {int(sel.sum()):5d} lanes, max |dlog2| {float(da[sel].max()):.3e}, "
                  f"median {float(da[sel].median()):.3e}", flush=True)
    dv = (l2(valid_add * mt) - l2(kern_pibar)).abs()
    print(f"[host] additive valid mass*mt vs kernel Pibar: max |dlog2| {float(dv.max()):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
