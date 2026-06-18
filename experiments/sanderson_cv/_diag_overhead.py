"""Diagnose the genewise per-gradient overhead: split forward (Pi self-loop + E-step) vs backward (adjoint),
and measure how each scales with pi_iters and neumann_terms. Tells us where the wall-clock actually goes so
we know which knob (pi, neumann, warm-start, clade_budget) to attack. Run on a freed GPU."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.core.inference.solver import solve_resident_e_pi
DEV = "cuda"
NF = int(os.environ.get("FAMILIES", "1500"))
paths = DATASETS["hogenom_full"]["families"](None)[:NF]
sp = str(DATASETS["hogenom_full"]["species_tree"])

def build(pi, neu, budget=80000):
    m = GeneReconModel(sp, [str(x) for x in paths], mode="genewise", device=DEV,
                       solver_options=SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu}), clade_budget=budget)
    m.receiver_weights.requires_grad_(False); return m

def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()                      # warm/JIT
    t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps

m = build(64, 16)
F = int(m.theta.shape[0]); nb = len(m.batch_statics)
tc = sum(int(s.wave_layout["leaf_species_index"].numel()) for s in m.batch_statics)
th = torch.zeros(F, 3, device=DEV)
print(f"hogenom {F} families, {nb} batches, {tc:,} clades", flush=True)

def fullgrad(): m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
def lossonly(): m.genewise_loss_vector_and_grad(theta=th, need_grad=False)

print("\n[A] forward(loss-only) vs full(loss+grad)  -> backward = full - forward, at pi=64 neu=16:", flush=True)
tf = timed(lossonly); tg = timed(fullgrad)
print(f"  loss-only (forward) = {tf*1000:.0f} ms   full (fwd+bwd) = {tg*1000:.0f} ms   backward = {(tg-tf)*1000:.0f} ms "
      f"({100*(tg-tf)/tg:.0f}% of grad)", flush=True)

print("\n[B] forward scaling with pi (loss-only): is the Pi self-loop the cost?", flush=True)
for pi in [8, 16, 32, 64, 128]:
    m.solver_options = SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": 16})
    print(f"  pi={pi:3d}: loss-only = {timed(lossonly)*1000:.0f} ms", flush=True)

print("\n[C] backward scaling with neumann (full - forward), pi fixed 64:", flush=True)
mf = SolverOptions(**{**_CV_SO, "pi_iters": 64, "neumann_terms": 16}); m.solver_options = mf
tf64 = timed(lossonly)
for neu in [8, 16, 32, 64]:
    m.solver_options = SolverOptions(**{**_CV_SO, "pi_iters": 64, "neumann_terms": neu})
    tgn = timed(fullgrad)
    print(f"  neu={neu:3d}: full = {tgn*1000:.0f} ms   backward = {(tgn-tf64)*1000:.0f} ms", flush=True)

print("\n[D] clade_budget effect on forward (launch overhead), pi=64:", flush=True)
del m; torch.cuda.empty_cache()
for budget in [40000, 80000, 160000]:
    mb = build(64, 16, budget=budget)
    th2 = torch.zeros(int(mb.theta.shape[0]), 3, device=DEV)
    def lo(): mb.genewise_loss_vector_and_grad(theta=th2, need_grad=False)
    print(f"  budget={budget:6d} ({len(mb.batch_statics)} batches): loss-only = {timed(lo)*1000:.0f} ms", flush=True)
    del mb; torch.cuda.empty_cache()
