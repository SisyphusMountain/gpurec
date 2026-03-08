#!/usr/bin/env python3
"""
Quick standalone runner to exercise batched likelihood.

Usage:
  python scripts/try_batched_likelihood.py \
    --species /path/to/sp.nwk \
    --gene /path/to/g.nwk --gene /path/to/g.nwk \
    --device cuda --dtype float32 --genewise 1 --specieswise 0 --pairwise 0

Defaults point to tests/data/test_mixed_200.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch


def parse_args():
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_sp = repo_root / "tests" / "data" / "test_mixed_200" / "sp.nwk"
    default_g = repo_root / "tests" / "data" / "test_mixed_200" / "g.nwk"
    p.add_argument("--species", type=str, default=str(default_sp))
    p.add_argument("--gene", type=str, action="append", default=[str(default_g), str(default_g)]*20, help="Repeat to add multiple gene trees")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    p.add_argument("--genewise", type=int, default=0)
    p.add_argument("--specieswise", type=int, default=0)
    p.add_argument("--pairwise", type=int, default=0)
    p.add_argument("--itersE", type=int, default=2000)
    p.add_argument("--itersPi", type=int, default=2000)
    p.add_argument("--memstats", type=int, default=1, help="Report CUDA memory stats and NVTX markers (1/0)")
    p.add_argument("--nvtx_ops", type=int, default=0, help="Emit NVTX ranges for every PyTorch op (1/0)")
    p.add_argument("--torch-prof", type=int, default=0, help="Enable PyTorch profiler and export Chrome trace (1/0)")
    p.add_argument("--torch-prof-out", type=str, default="nsys_reports/torch_trace.json", help="Chrome trace output path")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.core.model import GeneDataset
    from src.core.extract_parameters import extract_parameters
    from src.core.batching import collate_gene_families

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    genewise = bool(args.genewise)
    specieswise = bool(args.specieswise)
    pairwise = bool(args.pairwise)

    print(f"Device: {device}, dtype: {dtype}, genewise={genewise}, specieswise={specieswise}, pairwise={pairwise}")
    try:
        import triton
        print(f"triton version: {triton.__version__}")
    except Exception:
        print("triton not found; will rely on PyTorch fallbacks if needed")
    
    # Optional memory instrumentation
    def _format_mib(x: int) -> str:
        return f"{x / (1024**2):.2f} MiB"
    
    if device.type == 'cuda' and args.memstats:
        try:
            props = torch.cuda.get_device_properties(device)
            print(f"GPU: {props.name}, total memory: {_format_mib(props.total_memory)}")
        except Exception:
            pass

    sp = args.species
    gs = args.gene
    print(f"Species: {sp}")
    print(f"Genes ({len(gs)}): first={gs[0]}")

    ds = GeneDataset(sp, gs, genewise=genewise, specieswise=specieswise, pairwise=pairwise, device=device, dtype=dtype)
    print(f"Families: {ds.num_families}, S={ds.S}")

    # Inspect parameter shapes pre-likelihood
    theta_stack = torch.stack([fam['theta'] for fam in ds.families], dim=0)
    log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta_stack.to(device=device, dtype=dtype), ds.tr_mat_unnormalized.to(device=device, dtype=dtype), genewise=True, specieswise=specieswise, pairwise=pairwise)
    print(f"param shapes: log_pS={tuple(log_pS.shape)}, log_pD={tuple(log_pD.shape)}, log_pL={tuple(log_pL.shape)}, T={tuple(T.shape)}, T_max={tuple(T_max.shape)}")

    # Inspect collation to see C totals and seg_ptr
    items = []
    for fam in ds.families:
        items.append({
            'ccp': fam['ccp_helpers'],
            'leaf_row_index': fam['leaf_row_index'],
            'leaf_col_index': fam['leaf_col_index'],
            'root_clade_id': int(fam['root_clade_id']),
        })
    batched = collate_gene_families(items, dtype=dtype, device=device)
    C_total = int(batched['ccp']['C'])
    clades_per_gene = torch.tensor([m['C'] for m in batched['family_meta']], dtype=torch.long, device=device)
    seg_ptr = torch.nn.functional.pad(torch.cumsum(clades_per_gene, dim=0), (1, 0))
    print(f"Batched C={C_total}, clades_per_gene={clades_per_gene.tolist()}, seg_ptr={seg_ptr}")
    for i in range(ds.num_families):
        ds.set_params(idx=i,
                    D=0.16103,
                    L=1e-10,
                    T=0.156391)
    # Run batched likelihood (optionally capture max CUDA memory usage)
    if device.type == 'cuda' and args.memstats:
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        # Add an NVTX range to wrap the whole batched run
        nvtx = getattr(getattr(torch, 'cuda', None), 'nvtx', None)
        if nvtx is not None and hasattr(nvtx, 'range_push'):
            try:
                nvtx.range_push("likelihood_batch")
            except Exception:
                nvtx = None

    def _run_likelihood():
        return ds.compute_likelihood_batch(indices=list(range(ds.num_families)), max_iters_E=args.itersE, max_iters_Pi=args.itersPi)

    if int(args.torch_prof):
        print(f"PyTorch profiler enabled; exporting Chrome trace to {args.torch_prof_out}")
        try:
            from torch.profiler import profile, ProfilerActivity
        except Exception as e:
            print(f"Failed to import torch.profiler; running without it. Error: {e}")
            ll = _run_likelihood()
        else:
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                if int(args.nvtx_ops):
                    try:
                        from torch.autograd.profiler import emit_nvtx
                    except Exception:
                        emit_nvtx = None
                    if emit_nvtx is not None:
                        with emit_nvtx(record_shapes=True):
                            ll = _run_likelihood()
                    else:
                        ll = _run_likelihood()
                else:
                    ll = _run_likelihood()
            # Ensure all CUDA work is complete before exporting
            if device.type == 'cuda':
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
            try:
                Path(args.torch_prof_out).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                from pathlib import Path as _Path
                _Path(args.torch_prof_out).parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(args.torch_prof_out)
                print(f"Chrome trace written to {args.torch_prof_out}")
            except Exception as e:
                print(f"Failed to write Chrome trace: {e}")
            # Optional: quick summary in stdout
            try:
                print(prof.key_averages().table(sort_by=("self_cuda_time_total" if device.type == 'cuda' else "self_cpu_time_total"), row_limit=30))
            except Exception:
                pass
    else:
        if int(args.nvtx_ops):
            print("Enabling NVTX per-op ranges via torch.autograd.profiler.emit_nvtx")
            try:
                from torch.autograd.profiler import emit_nvtx
            except Exception:
                emit_nvtx = None
            if emit_nvtx is None:
                ll = _run_likelihood()
            else:
                with emit_nvtx(record_shapes=True):
                    ll = _run_likelihood()
        else:
            ll = _run_likelihood()

    if device.type == 'cuda' and args.memstats:
        try:
            torch.cuda.synchronize(device)
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            cur_alloc = torch.cuda.memory_allocated(device)
            cur_reserved = torch.cuda.memory_reserved(device)
            print("CUDA memory usage:")
            print(f"  peak allocated: {_format_mib(peak_alloc)} | peak reserved: {_format_mib(peak_reserved)}")
            print(f"  current allocated: {_format_mib(cur_alloc)} | current reserved: {_format_mib(cur_reserved)}")
            # Emit an NVTX mark with peak values so they appear on the Nsight timeline
            nvtx = getattr(getattr(torch, 'cuda', None), 'nvtx', None)
            if nvtx is not None:
                try:
                    msg = f"Peak alloc: {_format_mib(peak_alloc)}, Peak reserved: {_format_mib(peak_reserved)}"
                    if hasattr(nvtx, 'mark'):
                        nvtx.mark(msg)
                    elif hasattr(nvtx, 'range_push') and hasattr(nvtx, 'range_pop'):
                        nvtx.range_push(msg); nvtx.range_pop()
                except Exception:
                    pass
        finally:
            # Close the NVTX range if opened
            nvtx = getattr(getattr(torch, 'cuda', None), 'nvtx', None)
            if nvtx is not None and hasattr(nvtx, 'range_pop'):
                try:
                    nvtx.range_pop()
                except Exception:
                    pass
    print(f"Per-family log-likelihoods ({len(ll)}): {ll}")


if __name__ == "__main__":
    main()
