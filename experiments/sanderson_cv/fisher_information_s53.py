"""Observed-information / identifiability analysis for the gpurec paper (paper section 5.3).

Forms the exact dense Hessian H_F = H_data + lam*L (p = 3*S = 357) at a CERTIFIED specieswise
optimum (certified_v2/archaea_lam{lam}_certified.pt, theta_newton) and derives the three section-5.3
deliverables:
  (A) the observed-information spectrum    -- H_data is INDEFINITE (a near-zero/negative tail =
                                              the directions the data cannot constrain);
  (B) the lambda-homotopy of H_F spectra   -- the GBM prior lifts the tail until H_F is PD
                                              (regularization restores identifiability);
  (C) the Duplication-Loss confounding     -- within-species 3x3 block: (D-L) net-growth STIFF,
                                              (D+L) turnover SOFT; Transfer decoupled & identified;
  (D) posterior SEs/CIs from H_F(lam*)^-1  -- D,L confounded => large SE; T identified => small SE.

p=357 so the exact 357x357 eigh is cheap. The near-zero eigenvalues need fp64 (use --dtype float64);
fp32 is fine for block/sign STRUCTURE only. Single batch at 256 families (the certified scope); the
multi-batch HVP (build_hvp_once sums over batch_statics) also handles full archaea (much slower in fp64).

Run (GPU; needs the gpurec native ext + archaea data; archaea root defaults to the in-repo test data):
  GPUREC_PREPROCESS_PATH=<.../libgpurec_preprocess.so> PYTHONPATH=<worktree> \
  python fisher_information_s53.py [--families 256] [--dtype float64] [--lambdas 0.03,0.1,0.3,1.0,3.0,10.0]

Writes _artifacts/fisher_s53/ (cached Hessians + results.pt + console log) and paper-ready figures to
both _artifacts/fisher_s53/ and ../../../kernel-bench/paper/figures/.
"""
from __future__ import annotations
import os, sys, time, argparse, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = os.path.join(HERE, "_artifacts", "fisher_s53")
PAPER_FIG = "/home/enzo/Documents/git/gpurec/kernel-bench/paper/figures"
CERT_DIR = os.path.join(HERE, "_artifacts", "certified_v2")
NM = ["D", "L", "T"]   # theta[:,0]=Duplication, [:,1]=Loss, [:,2]=Transfer (log2-odds vs speciation)
DEV = "cuda"


def _resolve_theta(source, lam, lamf):
    """(cert_path, key) for the certified theta at this (source, lam).
    certified_v2 = 256-fam fp64-certified homotopy; full_homotopy = full-archaea (5446) polished fits."""
    if source == "certified_v2":
        return os.path.join(CERT_DIR, f"archaea_lam{lam}_certified.pt"), "theta_newton"
    if source == "full_homotopy":
        return os.path.join(HERE, "archaea_homotopy_out", f"polish_lam{int(lamf)}.pt"), "theta_newton"
    raise ValueError(f"unknown source {source!r}")


def build_hessian(N, lam, dtype, *, source="certified_v2", cache=True):
    """Dense H_F = H_data + lam*L and L at the certified specieswise optimum (theta_newton).

    `lam` is the string key (preserves the exact cert filename, e.g. '1.0'); `lamf` is the float used
    for the Laplacian / arithmetic. N<=0 (or None) => all families."""
    lamf = float(lam)
    nfam = None if (N is None or int(N) <= 0) else int(N)
    ntag = "all" if nfam is None else str(nfam)
    tag = f"H_{source}_N{ntag}_lam{lam}_{('f64' if dtype==torch.float64 else 'f32')}.pt"
    cpath = os.path.join(OUTDIR, tag)
    if cache and os.path.exists(cpath):
        d = torch.load(cpath, map_location="cpu", weights_only=False)
        print(f"[build] cache hit {tag}  (S={d['S']}, lam={d['lam']})")
        return d
    import saddle_escape
    saddle_escape.DTYPE = dtype
    from run_cv import DATASETS, _CV_SO
    from gpurec import GeneReconModel, SolverOptions
    cert, key = _resolve_theta(source, lam, lamf)
    theta_src = torch.load(cert, map_location=DEV, weights_only=False)[key]
    so = SolverOptions(**_CV_SO); so.validate()
    ds = DATASETS["archaea"]
    t0 = time.time()
    m = GeneReconModel(str(ds["species_tree"]), [str(x) for x in ds["families"](nfam)],
                       mode="specieswise", device=DEV, solver_options=so, clade_budget=80000)
    S = int(m.species_helpers["S"]); p = 3 * S
    rw = m.receiver_weights.detach().to(dtype)
    sp = m.species_helpers["sp_parent"].detach().long()
    child = (sp >= 0).nonzero(as_tuple=True)[0].contiguous(); par = sp[child].contiguous()
    lap = saddle_escape.make_lap(child, par, lamf); lapL = saddle_escape.make_lap(child, par, 1.0)
    theta = theta_src.to(DEV).to(dtype)
    Av = saddle_escape.build_hvp_once(m.batch_statics, theta, rw, lap, p)   # H_F = H_data + lam*L
    I = torch.eye(p, device=DEV, dtype=dtype)
    H = torch.stack([Av(I[:, i]) for i in range(p)], 1); H = 0.5 * (H + H.T)
    L = torch.stack([lapL(I[:, i]) for i in range(p)], 1); L = 0.5 * (L + L.T)
    Hd = H - lamf * L
    nb = len(m.batch_statics)
    d = dict(H=H.cpu(), H_data=Hd.cpu(), L=L.cpu(), theta=theta.reshape(S, 3).cpu(),
             sp_parent=sp.cpu(), child=child.cpu(), par=par.cpu(), lam=lamf, S=S, N=N,
             dtype=str(dtype), n_batches=nb,
             muH=torch.linalg.eigvalsh(H.double()).cpu(), muHd=torch.linalg.eigvalsh(Hd.double()).cpu())
    if cache:
        os.makedirs(OUTDIR, exist_ok=True); torch.save(d, cpath)
    print(f"[build] N={N} lam={lam} dtype={d['dtype']} S={S} p={p} batches={nb}  "
          f"lam_min(H_F)={float(d['muH'][0]):+.5f}  ({time.time()-t0:.1f}s)")
    return d


# ---------------------------------------------------------------------------------------------------
def analyze_confounding(d):
    """Reproduce + extend the curvature-structure numbers (README) at this optimum. fp64 throughout."""
    H = d["H"].double(); Hd = d["H_data"].double(); L = d["L"].double()
    S = int(d["S"]); p = 3 * S; lam = float(d["lam"])
    muH = torch.linalg.eigvalsh(H); muHd = torch.linalg.eigvalsh(Hd); muL = torch.linalg.eigvalsh(L)
    VH = torch.linalg.eigh(H)[1]
    kappa = float(muH[-1] / muH[0]) if muH[0] > 0 else float("inf")
    Hb = H.reshape(S, 3, S, 3); Hdb = Hd.reshape(S, 3, S, 3)
    eye = torch.eye(S, dtype=torch.bool)

    def frob(B, mask=None):
        M = torch.zeros(3, 3)
        for a in range(3):
            for b in range(3):
                blk = B[:, a, :, b]; M[a, b] = (blk[mask] if mask is not None else blk).norm()
        return M
    rt = frob(Hb); rt_s = frob(Hb, eye); rt_c = frob(Hb, ~eye)
    norm_rt = torch.stack([torch.stack([rt[a, b] / (rt[a, a] * rt[b, b]).sqrt() for b in range(3)])
                           for a in range(3)])
    avg = sum(H[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(S)) / S
    dl = avg[:2, :2]; ev, evec = torch.linalg.eigh(dl)
    soft_v, stiff_v = evec[:, 0], evec[:, 1]            # eig order ascending: [0]=soft, [1]=stiff
    vmin = VH[:, 0]
    rough = float((vmin @ (L @ vmin)) / (vmin @ vmin))
    pct = 100 * float((muL < rough).float().mean())
    # per-species loadings of the global bottom eigenvector (D,L,T)
    vmin_sp = vmin.reshape(S, 3)
    # per-species D-L confounding: softest eigenvector of each species' own (D,L) 2x2 sub-block
    # (the turnover-vs-net-growth split; Transfer excluded -- it is a separate gently-curved axis),
    # sign-aligned so the Duplication component >= 0. D,L co-signed => soft dir = turnover (D+L).
    soft_sp = torch.zeros(S, 2)      # (D,L) loadings of the per-species soft D-L direction
    dl_coup_sp = torch.zeros(S)      # per-species normalized |H_DL|/sqrt(H_DD H_LL)
    for i in range(S):
        blk = H[3 * i:3 * i + 3, 3 * i:3 * i + 3][:2, :2]
        ev_i, evec_i = torch.linalg.eigh(blk)
        v = evec_i[:, 0]
        if v[0] < 0:
            v = -v
        soft_sp[i] = v
        dl_coup_sp[i] = blk[0, 1].abs() / (blk[0, 0] * blk[1, 1]).clamp_min(1e-12).sqrt()
    turn = torch.tensor([1.0, 1.0], dtype=soft_sp.dtype) / (2 ** 0.5)
    cos_turn = (soft_sp @ turn).abs()      # soft_sp rows already unit-norm (eigenvectors)
    res = dict(
        lam=lam, S=S, p=p, kappa=kappa,
        muH=muH, muHd=muHd, muL=muL,
        H_min=float(muH[0]), H_med=float(muH[p // 2]), H_max=float(muH[-1]),
        Hd_min=float(muHd[0]), Hd_nneg=int((muHd < -1e-3).sum()),
        Hd_below=[int((muHd < t).sum()) for t in (1e-3, 1e-2, 1e-1)],
        norm_rt=norm_rt, dl_coupling=float(norm_rt[0, 1]),
        within_pct=float(100 * rt_s[0, 1] ** 2 / (rt_s[0, 1] ** 2 + rt_c[0, 1] ** 2)),
        avg_block=avg, dl_eig=(float(ev[0]), float(ev[1])), soft_v=soft_v, stiff_v=stiff_v,
        T_self=float(avg[2, 2]), T_off=(float(avg[2, 0]), float(avg[2, 1])),
        vmin=vmin, vmin_sp=vmin_sp, soft_sp=soft_sp, cos_turn=cos_turn, dl_coup_sp=dl_coup_sp,
        soft_turn_med=float(cos_turn.median()), soft_turn_frac=float((cos_turn > 0.9).float().mean()),
        soft_roughness=rough, soft_pct=pct)
    return res


def analyze_SE(d):
    """Posterior SEs/CIs from C = H_F^-1 at a PD optimum (the MAP covariance). fp64."""
    H = d["H"].double(); S = int(d["S"]); lam = float(d["lam"])
    mu = torch.linalg.eigvalsh(H)
    assert float(mu[0]) > 0, f"H_F not PD at lam={lam} (lam_min={float(mu[0])}); SEs undefined"
    C = torch.linalg.inv(H)
    se = C.diagonal().clamp_min(0).sqrt().reshape(S, 3)            # log2-rate SE per (species, rate)
    # variance along the within-species turnover (D+L) vs net-growth (D-L) axes
    a_turn = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64) / (2 ** 0.5)
    a_net = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64) / (2 ** 0.5)
    vt, vn = [], []
    for i in range(S):
        Cb = C[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        vt.append(float(a_turn @ Cb @ a_turn)); vn.append(float(a_net @ Cb @ a_net))
    vt = torch.tensor(vt); vn = torch.tensor(vn)
    return dict(lam=lam, C=C, se=se, var_turnover=vt, var_netgrowth=vn,
                se_med=[float(se[:, k].median()) for k in range(3)],
                se_q=[(float(se[:, k].quantile(.1)), float(se[:, k].quantile(.9))) for k in range(3)],
                turn_over_net=float((vt / vn).median()))


# ---------------------------------------------------------------------------------------------------
def make_figures(conf, se, homotopy, theta, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})
    for fdir in (OUTDIR, PAPER_FIG):
        os.makedirs(fdir, exist_ok=True)

    def save(fig, name):
        name = name.replace(".pdf", f"{suffix}.pdf")
        for fdir in (OUTDIR, PAPER_FIG):
            fig.savefig(os.path.join(fdir, name), bbox_inches="tight")
        plt.close(fig); print(f"[fig] {name}")

    # ---- Fig 3a: observed-information spectrum + lambda-homotopy -----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    muHd = conf["muHd"].numpy(); muH = conf["muH"].numpy()
    idx = np.arange(len(muHd))
    ax[0].axhline(0, color="0.6", lw=.7)
    ax[0].scatter(idx, muHd, s=7, c="#c44", label=r"$H_{\rm data}$ (observed info)")
    ax[0].scatter(idx, muH, s=7, c="#357", marker="x",
                  label=fr"$H_F=H_{{\rm data}}+\lambda L,\ \lambda={conf['lam']}$")
    ax[0].set_yscale("symlog", linthresh=1e-2)
    ax[0].set_xlabel("eigenvalue index (ascending)"); ax[0].set_ylabel("eigenvalue")
    ax[0].set_title(f"(a) spectrum at the $\\lambda={conf['lam']}$ optimum\n"
                    fr"$H_{{\rm data}}$: {conf['Hd_nneg']} negative, "
                    fr"{conf['Hd_below'][2]}/{conf['p']} below 0.1; $\kappa(H_F)\approx${conf['kappa']:.0f}")
    ax[0].legend(loc="lower right", fontsize=7)

    lams = [h["lam"] for h in homotopy]; lmins = [h["lam_min"] for h in homotopy]
    ax[1].loglog(lams, lmins, "o-", color="#357")
    ax[1].set_xlabel(r"prior strength $\lambda$"); ax[1].set_ylabel(r"$\lambda_{\min}(H_F)>0$ (PD)")
    ax[1].set_title("(b) regularization restores identifiability\n"
                    r"$\lambda_{\min}(H_F)$ vs prior strength")
    ax[1].grid(True, which="both", ls=":", lw=.4)
    save(fig, "fig_s53_spectrum.pdf")

    # ---- Fig 3b: D-L confounding (within-species block + bottom-eigenvector loadings) --------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    avg = conf["avg_block"].numpy()
    im = ax[0].imshow(avg, cmap="RdBu_r", vmin=-abs(avg).max(), vmax=abs(avg).max())
    ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
    ax[0].set_xticklabels(NM); ax[0].set_yticklabels(NM)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f"{avg[i, j]:.1f}", ha="center", va="center", fontsize=8)
    es, eh = conf["dl_eig"]
    ax[0].set_title(f"(a) avg within-species block\nstiff(D$-$L net growth)={eh:.1f}, "
                    f"soft(D$+$L turnover)={es:.1f}")
    fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04)

    # per-species softest eigenvector of the within-species (D,L) 2x2 block, sign-aligned D>=0:
    # co-signed D,L => points cluster on the turnover diagonal (D=L), the confounded direction.
    # color = per-species D-L coupling (1 => collinear/confounded); strongly-coupled species sit on D=L.
    vs = conf["soft_sp"].numpy()        # (S,2)
    sc = ax[1].scatter(vs[:, 0], vs[:, 1], s=16, c=conf["dl_coup_sp"].numpy(), cmap="plasma",
                       vmin=0, vmax=1, label="per-species soft D-L dir")
    ax[1].plot([0, 1], [0, 1], "k--", lw=.8, label="D=L (turnover, soft)")
    ax[1].plot([0, 1], [0, -1], "0.6", ls=":", lw=.8, label="D=$-$L (net, stiff)")
    ax[1].set_xlim(-0.05, 1.05); ax[1].set_ylim(-1.05, 1.05); ax[1].set_aspect("equal")
    ax[1].set_xlabel("Duplication loading"); ax[1].set_ylabel("Loss loading")
    ax[1].set_title(f"(b) per-species soft D-L direction\n"
                    fr"{100*conf['soft_turn_frac']:.0f}% align w/ turnover, |cos|med={conf['soft_turn_med']:.2f}")
    fig.colorbar(sc, ax=ax[1], fraction=.046, pad=.04, label="D-L coupling")
    ax[1].legend(fontsize=7, loc="lower right")
    save(fig, "fig_s53_dl_confounding.pdf")

    # ---- Fig 3c: posterior SEs at lambda* ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    se_np = se["se"].numpy()
    ax[0].boxplot([se_np[:, k] for k in range(3)], tick_labels=NM, showfliers=False)
    ax[0].set_ylabel(r"posterior SE (log$_2$-rate)")
    ax[0].set_title(f"(a) per-rate posterior SE at $\\lambda$={se['lam']}\n"
                    f"(MAP cov $H_F^{{-1}}$)")
    ax[1].scatter(se["var_netgrowth"].sqrt().numpy(), se["var_turnover"].sqrt().numpy(),
                  s=10, c="#357")
    m = max(float(se["var_netgrowth"].sqrt().max()), float(se["var_turnover"].sqrt().max())) * 1.1
    ax[1].plot([0, m], [0, m], "k--", lw=.7)
    ax[1].set_xlim(0, m); ax[1].set_ylim(0, m); ax[1].set_aspect("equal")
    ax[1].set_xlabel("SE of net growth (D$-$L)"); ax[1].set_ylabel("SE of turnover (D$+$L)")
    ax[1].set_title(f"(b) turnover is less identified\nmedian var ratio (turn/net)={se['turn_over_net']:.1f}")
    save(fig, "fig_s53_se.pdf")


# ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=256, help="N families; <=0 => all (full archaea)")
    ap.add_argument("--source", choices=["certified_v2", "full_homotopy"], default="certified_v2",
                    help="certified_v2 = 256-fam fp64-certified homotopy; full_homotopy = full-archaea polished fits")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--lambdas", default="0.03,0.1,0.3,1.0,3.0,10.0",
                    help="comma list for the homotopy figure (must exist in the chosen source)")
    ap.add_argument("--conf-lam", default="0.03", help="lambda for the confounding spectrum/eigvecs")
    ap.add_argument("--se-lam", default="1.0", help="lambda* (PD) for posterior SEs")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--fig-suffix", default="", help="appended to figure filenames (avoid overwrite)")
    a = ap.parse_args()
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    os.makedirs(OUTDIR, exist_ok=True)
    lams = a.lambdas.split(",")

    print(f"=== section 5.3 observed-information / identifiability  "
          f"(source={a.source}, N={a.families}, {a.dtype}) ===\n")
    builds = {lm: build_hessian(a.families, lm, dtype, source=a.source)
              for lm in dict.fromkeys(lams + [a.conf_lam, a.se_lam])}

    conf = analyze_confounding(builds[a.conf_lam])
    print(f"\n[A/C] CONFOUNDING at lam={a.conf_lam}  (kappa={conf['kappa']:.0f})")
    print(f"  H_F spectrum: min={conf['H_min']:+.4f} med={conf['H_med']:+.3f} max={conf['H_max']:.1f}")
    print(f"  H_data INDEFINITE: min={conf['Hd_min']:+.4f}  #neg={conf['Hd_nneg']}  "
          f"#<0.1={conf['Hd_below'][2]}/{conf['p']}")
    print(f"  normalized D-L coupling={conf['dl_coupling']:.2f} (~1 confounded); within-species={conf['within_pct']:.0f}%")
    print(f"  within-species avg block (D,L,T):\n{conf['avg_block'].numpy()}")
    print(f"  D-L 2x2 eig: stiff(D-L net)={conf['dl_eig'][1]:.1f}  soft(D+L turnover)={conf['dl_eig'][0]:.1f}  "
          f"soft_vec~{conf['soft_v'].numpy().round(2)}")
    print(f"  T self-curv={conf['T_self']:.2f} off~{conf['T_off'][0]:.2f},{conf['T_off'][1]:.2f} (decoupled)")
    print(f"  global soft dir lam_min={conf['H_min']:+.4f} roughness={conf['soft_roughness']:.3f} "
          f"({conf['soft_pct']:.0f}th pctile of L)")
    print(f"  per-species soft dir: {100*conf['soft_turn_frac']:.0f}% align with turnover(D+L) "
          f"(|cos|>0.9); median |cos to turnover|={conf['soft_turn_med']:.2f}")

    se = analyze_SE(builds[a.se_lam])
    print(f"\n[D] POSTERIOR SEs at lam*={a.se_lam}  (H_F PD, C=H_F^-1)")
    for k in range(3):
        lo, hi = se["se_q"][k]
        print(f"  SE[{NM[k]}] median={se['se_med'][k]:.3f} log2  (10-90%: {lo:.3f}-{hi:.3f})  "
              f"=> rate x/ {2**(1.96*se['se_med'][k]):.2f} (95% CI factor)")
    print(f"  turnover(D+L) vs net-growth(D-L) posterior variance ratio (median) = {se['turn_over_net']:.1f}  "
          f"(>1 => turnover less identified)")

    homotopy = [dict(lam=float(lm), lam_min=float(builds[lm]["muH"][0]),
                     loss=None, S=builds[lm]["S"]) for lm in lams]
    print(f"\n[B] LAMBDA-HOMOTOPY lam_min(H_F):")
    for h in homotopy:
        print(f"   lam={h['lam']:<6} lam_min={h['lam_min']:+.5f}  {'PD' if h['lam_min']>0 else 'INDEF'}")

    res = dict(families=a.families, source=a.source, dtype=a.dtype, conf=conf, se=se, homotopy=homotopy,
               conf_lam=float(a.conf_lam), se_lam=float(a.se_lam))
    rname = f"results_{a.source}_N{a.families}_{a.dtype}.pt"
    torch.save(res, os.path.join(OUTDIR, rname))
    print(f"\n[saved] {rname}")

    if not a.no_figs:
        make_figures(conf, se, homotopy, builds[a.conf_lam]["theta"], suffix=a.fig_suffix)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
