"""DEFINITIVE headline check: is AleRax's reported per-family NLL (the speed-table baseline,
sum of per_fam_likelihoods.txt) the SAME quantity as gpurec's amalgamated likelihood?

Eval AleRax's FITTED Hogenom rates (a chunk) in gpurec (converged pi64/neu64) and compare per-family
to AleRax's per_fam_likelihoods. If diff ~ 0 -> same objective, headline apples-to-apples. If
diff ~ several nat/family -> the headline compares different quantities (margin is only 0.42 nat/fam).

Run from experiments/alerax_speed with the mapcv-merge gpurec:
  PYTHONPATH=$WT:$WT/experiments/sanderson_cv GPUREC_PREPROCESS_PATH=... CHUNK=07 python xcheck_headline.py
"""
import os, sys, glob, math, torch
HERE = os.path.dirname(os.path.abspath(__file__))
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"
ST = str(DATASETS["hogenom"]["species_tree"])
ln2 = math.log(2.0)
GPUREC_OWN_NLL_NATS = 1_906_464 * ln2     # gpurec at its OWN fitted rates (paper headline, bits->nats)

chunks = sorted(c.split("/")[-1] for c in glob.glob("/tmp/alerax_hogenom_chunked_out/chunk_*")
                if os.path.isdir(c + "/model_parameters"))
so = SolverOptions(**{**_CV_SO, "pi_iters": 64, "neumann_terms": 64})
all_d, tot_gp, tot_al, nfam = [], 0.0, 0.0, 0
print(f"[xcheck-all] {len(chunks)} chunks; gpurec-own NLL (paper) = {GPUREC_OWN_NLL_NATS:.0f} nats", flush=True)

for C in chunks:
    ALERAX_OUT = f"/tmp/alerax_hogenom_chunked_out/{C}/model_parameters"
    FAMFILE = f"{HERE}/alerax_chunks/{C}.families.txt"
    PERFAM = f"{HERE}/alerax_chunks/{C}.per_fam_likelihoods.txt"
    names, paths, cur = [], [], None
    for line in open(FAMFILE):
        s = line.strip()
        if s.startswith("- "): cur = s[2:].strip()
        elif s.startswith("starting_gene_tree"):
            names.append(cur); paths.append(s.split("=", 1)[1].strip())
    def read_rate(n):
        p = f"{ALERAX_OUT}/{n}_rates.txt"
        return [float(x) for x in open(p).read().split("\n")[1].split()[:3]] if os.path.exists(p) else None
    alerax_ll = {}
    for line in open(PERFAM):
        p = line.split()
        if len(p) >= 2:
            try: alerax_ll[p[0]] = float(p[-1])
            except ValueError: pass
    keep = [(n, pa) for n, pa in zip(names, paths) if read_rate(n) is not None and n in alerax_ll]
    names = [n for n, _ in keep]; paths = [pa for _, pa in keep]
    rates = torch.tensor([read_rate(n) for n in names], dtype=torch.float32, device=DEV)
    theta = torch.log2(rates)
    mc = GeneReconModel(ST, paths, mode="genewise", device=DEV, solver_options=so, clade_budget=900_000)
    mc.receiver_weights.requires_grad_(False)
    nll_bits = mc.genewise_loss_vector(theta=theta).detach().cpu()
    gpurec_ll = (-nll_bits * ln2)
    cd = [float(gpurec_ll[i]) - alerax_ll[n] for i, n in enumerate(names)]
    all_d += cd
    tot_gp += float(gpurec_ll.sum()); tot_al += sum(alerax_ll[n] for n in names); nfam += len(names)
    del mc; torch.cuda.empty_cache()
    print(f"  {C}: {len(names)} fam  gpurec-alerax mean={sum(cd)/len(cd):+.4f} nat/fam  (running n={nfam})", flush=True)

d = torch.tensor(all_d)
gpurec_at_alerax_nll = -tot_gp     # apples-to-apples AleRax NLL (AleRax rates, gpurec likelihood)
alerax_reported_nll = -tot_al      # AleRax reported NLL (the paper baseline)
print(f"\n{'='*64}\n=== FULL-SET apples-to-apples headline ({nfam} families, nats) ===\n{'='*64}", flush=True)
print(f"  gpurec @ gpurec rates (paper)        NLL = {GPUREC_OWN_NLL_NATS:11.0f}", flush=True)
print(f"  AleRax  @ AleRax rates (reported)    NLL = {alerax_reported_nll:11.0f}   <- current headline baseline", flush=True)
print(f"  gpurec  @ AleRax rates (apples)      NLL = {gpurec_at_alerax_nll:11.0f}   <- consistent-likelihood baseline", flush=True)
print(f"\n  REPORTED margin  (gpurec vs AleRax-reported) = {alerax_reported_nll-GPUREC_OWN_NLL_NATS:8.0f} nats "
      f"= {(alerax_reported_nll-GPUREC_OWN_NLL_NATS)/nfam:+.3f} /fam", flush=True)
print(f"  APPLES-TO-APPLES margin (both in gpurec)     = {gpurec_at_alerax_nll-GPUREC_OWN_NLL_NATS:8.0f} nats "
      f"= {(gpurec_at_alerax_nll-GPUREC_OWN_NLL_NATS)/nfam:+.3f} /fam", flush=True)
print(f"  eval discrepancy (AleRax under-reports)      = {alerax_reported_nll-gpurec_at_alerax_nll:8.0f} nats "
      f"= {(alerax_reported_nll-gpurec_at_alerax_nll)/nfam:+.3f} /fam (median {float(d.median()):+.4f})", flush=True)
print(f"\n  -> gpurec lower NLL: REPORTED by {(alerax_reported_nll-GPUREC_OWN_NLL_NATS)/nfam:.3f}/fam, "
      f"APPLES-TO-APPLES by {(gpurec_at_alerax_nll-GPUREC_OWN_NLL_NATS)/nfam:.3f}/fam "
      f"({'STILL WINS' if gpurec_at_alerax_nll>GPUREC_OWN_NLL_NATS else 'FLIPS!'})", flush=True)
