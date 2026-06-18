"""Feasibility probe for HOGENOM 5000-family genewise: build cost, memory, per-grad time, and whether the
adjoint warm-start cache (~total_clades * S * 4 bytes) fits on this GPU before committing to a long run."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
DEV = "cuda"; GB = 1024**3
PI = int(os.environ.get("PI", "64")); NEU = int(os.environ.get("NEU", "16"))
paths = [l.strip() for l in open("/tmp/hogenom_5000_paths.txt") if l.strip()]
print(f"hogenom 5000-family probe: {len(paths)} families, pi={PI} neumann={NEU}", flush=True)
torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
m = GeneReconModel(str(DATASETS["hogenom"]["species_tree"]), paths, mode="genewise", device=DEV,
                   solver_options=SolverOptions(**{**_CV_SO, "pi_iters": PI, "neumann_terms": NEU}), clade_budget=80000)
m.receiver_weights.requires_grad_(False)
F = int(m.theta.shape[0]); S = int(m.species_helpers["S"]); nb = len(m.batch_statics)
total_clades = sum(int(s.wave_layout["leaf_species_index"].numel()) for s in m.batch_statics)
torch.cuda.synchronize()
free_b, tot_b = torch.cuda.mem_get_info()
print(f"[build] F={F} S={S} batches={nb} total_clades={total_clades:,}  build={time.perf_counter()-t0:.1f}s", flush=True)
print(f"[mem] after build: peak_reserved={torch.cuda.max_memory_reserved()/GB:.2f}GB  free={free_b/GB:.2f}GB / {tot_b/GB:.1f}GB", flush=True)
warm_cache_gb = total_clades * S * 4 / GB
print(f"[warm cache estimate] total_clades*S*4 = {warm_cache_gb:.1f}GB  (fp32 adjoint over every wave-row, all batches)", flush=True)

th = torch.zeros(F, 3, device=DEV)
# cold gradient timing
torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
_, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); torch.cuda.synchronize()
print(f"[cold grad] {time.perf_counter()-t:.1f}s  |g|inf={float(g.abs().max()):.2f}  peak_reserved={torch.cuda.max_memory_reserved()/GB:.2f}GB", flush=True)

if warm_cache_gb < free_b/GB - 3.0:
    os.environ["GPUREC_WARM_ADJOINT"] = "1"
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    try:
        _, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); torch.cuda.synchronize()
        cache_b = sum(v.numel()*v.element_size() for s in m.batch_statics for v in (s.warm_v or {}).values())
        print(f"[warm grad] {time.perf_counter()-t:.1f}s  peak_reserved={torch.cuda.max_memory_reserved()/GB:.2f}GB  "
              f"actual warm_v cache={cache_b/GB:.2f}GB", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[warm grad] OOM: {str(e)[:120]}", flush=True)
else:
    print(f"[warm grad] SKIPPED -- predicted cache {warm_cache_gb:.1f}GB won't fit in {free_b/GB:.1f}GB free", flush=True)
