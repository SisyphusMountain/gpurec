"""CUDA and kernel warmup helpers used by :class:`GeneReconModel`."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch


def _warm_cuda_context(device: torch.device) -> None:
    with torch.cuda.device(device):
        # Exercise cuBLAS and common elementwise kernels while CPU preprocessing runs.
        probe = torch.ones((64, 64), device=device, dtype=torch.float32)
        warmed = torch.log2(torch.exp2((probe @ probe) * 0.001) + 1.0)
        warmed.sum()
        torch.cuda.synchronize(device)


def _start_cuda_context_warmup(
    device: torch.device,
) -> tuple[ThreadPoolExecutor, Future] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gpurec-cuda-warmup",
    )
    return executor, executor.submit(_warm_cuda_context, device)


def _finish_cuda_context_warmup(
    handle: tuple[ThreadPoolExecutor, Future] | None,
) -> None:
    if handle is None:
        return
    executor, future = handle
    try:
        try:
            future.result()
        except Exception:
            pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _warm_resident_uniform_kernels(
    species_helpers: dict[str, any],
    ancestors_T: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if ancestors_T is None:
        return
    try:
        from gpurec.core.kernels.dts_fused import compute_dts_forward
        from gpurec.core.kernels.wave_step import (
            compute_leaf_initial_wave_step,
            compute_pibar,
            compute_wave_step,
        )
        from gpurec.core.likelihood import E_fixed_point
        from gpurec.core.species import species_wave_topology

        with torch.cuda.device(device):
            S = int(species_helpers["S"])
            topology = species_wave_topology(species_helpers, S=S, device=device)
            sp_child1 = topology["sp_child1"]
            sp_child2 = topology["sp_child2"]
            sp_parent = topology["sp_parent"]
            sp_subtree_start = topology["sp_subtree_start"]
            sp_subtree_end = topology["sp_subtree_end"]
            max_ancestor_depth = int(topology["max_ancestor_depth"])
            W = 1
            C = 2
            Pi = torch.empty((C, S), dtype=dtype, device=device)
            Pibar = torch.empty_like(Pi)
            row_max = torch.empty((C,), dtype=dtype, device=device)
            max_transfer = torch.zeros((S,), dtype=dtype, device=device)
            E = torch.full((S,), -1.0, dtype=dtype, device=device)
            Ebar = torch.full((S,), -2.0, dtype=dtype, device=device)
            DL = torch.full((S,), -3.0, dtype=dtype, device=device)
            SL1 = torch.full((S,), -4.0, dtype=dtype, device=device)
            SL2 = torch.full((S,), -5.0, dtype=dtype, device=device)
            leaf_species = torch.zeros((C,), dtype=torch.long, device=device)
            leaf_logp = torch.zeros((S,), dtype=dtype, device=device)
            dts = torch.full((W, S), -10.0, dtype=dtype, device=device)

            E_fixed_point(
                species_helpers=species_helpers,
                log_pS=leaf_logp,
                log_pD=leaf_logp,
                log_pL=leaf_logp,
                max_transfer_mat=max_transfer,
                max_iters=1,
                tolerance=-1.0,
                warm_start_E=None,
                dtype=dtype,
                device=device,
                ancestors_T=ancestors_T,
                e_shape=(S,),
            )
            compute_wave_step(
                Pi, Pibar, Pibar, 0, W, S,
                max_transfer, DL, Ebar, E, SL1, SL2,
                sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                None, None,
                leaf_species_idx=leaf_species,
                leaf_logp=leaf_logp,
                has_leaf_term=True,
                initial_state=True,
            )
            compute_leaf_initial_wave_step(
                Pibar, 0, W, S,
                max_transfer, DL, Ebar, E, SL1, SL2,
                sp_child1, sp_child2,
                sp_subtree_start, sp_subtree_end,
                leaf_species,
                leaf_logp,
            )
            compute_wave_step(
                Pibar, Pi, Pibar, 0, W, S,
                max_transfer, DL, Ebar, E, SL1, SL2,
                sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                None, dts,
                leaf_species_idx=leaf_species,
                leaf_logp=leaf_logp,
                has_leaf_term=False,
            )
            compute_wave_step(
                Pibar, Pi, Pibar, 0, W, S,
                max_transfer, DL, Ebar, E, SL1, SL2,
                sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                None, dts,
                leaf_species_idx=leaf_species,
                leaf_logp=leaf_logp,
                has_leaf_term=False,
                store_final_pibar=True,
                final_pibar_row_max=row_max,
            )
            compute_pibar(
                Pi, Pibar, 0, W, S,
                max_transfer, sp_parent, max_ancestor_depth,
                row_max_out=row_max,
            )
            lefts = torch.zeros((1,), dtype=torch.long, device=device)
            rights = torch.ones((1,), dtype=torch.long, device=device)
            split_logp = torch.zeros((1,), dtype=dtype, device=device)
            eq1_parent = torch.zeros((1,), dtype=torch.long, device=device)
            ge2_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
            ge2_parent = torch.zeros((1,), dtype=torch.long, device=device)
            compute_dts_forward(
                Pi, Pibar,
                lefts, rights,
                sp_child1, sp_child2,
                leaf_logp, leaf_logp, split_logp,
                W, 1, eq1_parent, ge2_ptr[:1], ge2_parent[:0],
            )
            compute_dts_forward(
                Pi, Pibar,
                lefts, rights,
                sp_child1, sp_child2,
                leaf_logp, leaf_logp, split_logp,
                W, 0, eq1_parent[:0], ge2_ptr, ge2_parent,
            )
            torch.cuda.synchronize(device)
    except Exception:
        return


def _start_resident_uniform_kernel_warmup(
    species_helpers: dict[str, any],
    ancestors_T: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[ThreadPoolExecutor, Future] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gpurec-resident-warmup",
    )
    return executor, executor.submit(
        _warm_resident_uniform_kernels,
        species_helpers,
        ancestors_T,
        dtype=dtype,
        device=device,
    )
