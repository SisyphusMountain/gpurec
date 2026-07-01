"""Manual family-sharded data-parallel execution (NOT torch DDP).

The gradient is a hand-written implicit VJP assigned directly to the parameter
.grad tensors (no autograd graph reaches theta), so DDP hooks never fire. Manual
SUM all-reduce is correct because loss/grad are a SUM over families and the
E-adjoint solve is linear with a family-independent operator:
    Σ_shard solve(q_shard) = solve(Σ_shard q_shard).
All collectives are no-ops when not launched under torchrun.

Usage (torchrun --nproc_per_node=N -m your_train_script):
    rank, world, local, device = maybe_init_distributed()
    my_families = shard_families(all_families, rank, world)   # before building the model
    model = GeneReconModel(species_tree, my_families, ..., device=device)
    for step in range(max_steps):
        opt.zero_grad(); loss = model(); loss.backward()      # per-rank partial grad
        all_reduce_model_grads_(model)                        # SUM the 3 param grads
        # priors/regularizers are rank-identical and already applied once inside the
        # per-rank loss; do NOT re-add them post-reduction. warm_E/warm_v stay per-rank.
        opt.step()
"""
from __future__ import annotations
import os
import torch

try:
    import torch.distributed as dist
    _HAVE_DIST = True
except Exception:  # pragma: no cover
    _HAVE_DIST = False


def ddp_enabled() -> bool:
    return _HAVE_DIST and dist.is_available() and dist.is_initialized()


def maybe_init_distributed(device_pref: str = "cuda"):
    if not _HAVE_DIST or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, torch.device(device_pref)
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    ndev = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    local = int(os.environ.get("LOCAL_RANK", rank % ndev))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}"); backend = "nccl"
    else:
        device = torch.device("cpu"); backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world, local, device


def all_reduce_sum_(t):
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_max_(t):
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def broadcast_(t, src: int = 0):
    if ddp_enabled():
        dist.broadcast(t, src=src)
    return t


def barrier():
    if ddp_enabled():
        dist.barrier()


def cleanup():
    if ddp_enabled():
        dist.destroy_process_group()


def _family_work(f) -> int:
    h = f.get("ccp_helpers", f) if isinstance(f, dict) else f
    def _g(k, d=0):
        v = h.get(k, d) if isinstance(h, dict) else getattr(h, k, d)
        try:
            return int(v)
        except Exception:
            return int(d)
    return _g("C", 1) + _g("N_splits", 0)


def shard_families(families, rank: int, world: int):
    """Deterministic work-balanced partition (greedy longest-processing-time)."""
    if world <= 1:
        return list(families)
    if world > len(families):
        raise ValueError(f"world_size {world} > n_families {len(families)}")
    order = sorted(range(len(families)), key=lambda i: -_family_work(families[i]))
    buckets = [[] for _ in range(world)]
    load = [0] * world
    for i in order:
        j = min(range(world), key=lambda k: load[k])
        buckets[j].append(i); load[j] += _family_work(families[i])
    return [families[i] for i in sorted(buckets[rank])]


def all_reduce_model_grads_(model):
    """SUM all-reduce the three learnable-parameter grads in place (NaN-sanitized)."""
    for name in ("theta", "receiver_weights", "origination_weights"):
        p = getattr(model, name, None)
        if p is not None and p.grad is not None:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            all_reduce_sum_(p.grad)
