#!/bin/bash
# Nsight Compute on the four per-wave BACKWARD kernels, one representative launch each.
#
# Usage (inside a job, after sourcing env.sh):  bash benchmark/cc/run_ncu_bwd.sh THETA
# THETA is "fitted" or "flat-6". One ncu process per kernel; each runs one gradient on 100
# families and profiles the 41st launch of the named kernel, first with --set full and then with
# a short list of pipe/instruction counters.
#
# WHAT THIS MEASURED (H100 NVL, 100 families, fitted theta from $CC_RUNS/results/full_v3.pt,
# forward and adjoint self-loops both "exact", 2026-09-03):
#
#   kernel                                    grid x block  dur us  occupancy  SM%   DRAM%  regs
#   _prepare_reconciliation_self_loop_vjp        462 x 256   174.4  12.5/12.5  28.9   0.58   187
#   _accumulate_transfer_subtree_vjp            2870 x 128   106.1  75.0/43.6  34.5   5.54    40
#   _accumulate_gene_split_event_vjp            1435 x 256    60.3  62.5/45.7  38.5  10.21    48
#   _accumulate_transfer_receiver_log_prob_vjp   462 x  64    64.5  43.8/ 9.4  10.4   1.09    72
#
#   None of the four is close to the memory roofline and none is compute-bound: DRAM throughput is
#   0.6-10%, the top pipe is ALU at 22-29%, and all four are LATENCY-bound.
#     prepare:  1 clade row per program, 2048 species lanes, 8 warps. 187 registers per thread cap
#               it at ONE block per SM (2 warps per scheduler, 12.5% occupancy) and 66.8% of cycles
#               have no eligible warp. 46 exp2/log2 per element, 34 of them from the ancestor walk;
#               486 ALU and 161 load/store ops per element, again mostly that walk. It costs 57%
#               excessive sectors but only 21 GB/s, and 94% of its loads hit L1 -- the walk re-reads
#               the same 8 KB Pi row 34 times, so it is L1-LATENCY bound, not bandwidth bound.
#     subtree:  33 tl.debug_barrier()s walking the species tree bottom-up through global scratch;
#               the top stall (31.9% of 15.0 cycles per issued instruction) is the shared-memory
#               scoreboard those barriers sit on. 65% excessive sectors.
#     receiver: grid is one program of 64 threads per clade row -- 0.25 waves per SM, 9.4% achieved
#               occupancy, 85.7% of cycles with no eligible warp. Starved, not busy.
#     gene split: the healthiest of the four (IPC 2.05, 38.5% SM, 39.8% memory); its remaining
#               losses are 30% excessive sectors from the species-child gathers and load imbalance
#               from parent rows that the active mask lets return early.
#
# WHERE A GRADIENT ACTUALLY GOES (nsys over ONE gradient, 500 families, fitted theta; total GPU
# 4.90 s, gradient wall 5.45 s). Take this, not a grad+Hessian capture, as the weighting for
# gradient work -- the Hessian re-runs the same kernels and roughly doubles their launch counts:
#     17.2%  843 ms  _prepare_reconciliation_self_loop_vjp_kernel        (1704 launches, 495 us)
#     17.2%  840 ms  _exact_tree_pi_self_loop_kernel                     (forward)
#     13.4%  655 ms  _accumulate_transfer_subtree_vjp_kernel
#     10.1%  494 ms  _exact_tree_self_loop_transpose_kernel              (adjoint)
#      8.5%  414 ms  _accumulate_gene_split_event_vjp_kernel
#      7.1%  348 ms  _stage_multiple_gene_split_event_reduction_kernel
#      6.1%  300 ms  at::native::indexFuncLargeIndex                     (8437 launches)
#      5.8%  284 ms  _update_reconciliation_likelihood_kernel
#      5.4%  267 ms  _accumulate_transfer_receiver_log_probability_vjp_kernel
#      2.5%  124 ms  _accumulate_reconciliation_event_vjp_kernel
#
# WHAT WAS TRIED AND REVERTED (all measured, none kept):
#   1. Replacing the ancestor walk's sp_parent hop chain by a precomputed [34, S] table of each
#      species' d-th ancestor. Bit-identical, removes one scattered gather per depth and the
#      serial dependency between depths. MEASURED 227 us (from 174): the table is 267 KB and the
#      loop streams all of it per row, which evicts the 8 KB Pi row the gather depends on
#      (L1 hit rate 94% -> 79%). Marking the table reads ".cg" so they skip L1 made it 243 us.
#   2. The same table plus reading the ancestor masses back out of the receiver_mass scratch row
#      instead of recomputing exp2 from Pi_star. Removes 74% of the kernel's transcendental
#      instructions and 22% of all its instructions; MEASURED 172.7 us (from 174.4) and, over a
#      whole gradient at 500 families, 827 ms (from 843) with the gradient wall time unchanged at
#      5.45 s. Negligible, so reverted.
#   3. Sizing the subtree kernel's level walk by nodes (128) instead of by species (256), which
#      cuts the padded lane-slots of the 33 levels from 8704 to 4608 for 1006 internal nodes.
#      Bit-identical. MEASURED 663 ms (from 655) over a gradient. Reverted.
#   4. num_warps 2 -> 8 on the starved receiver-log-probability kernel: 282 ms (from 267).
#   5. A full num_warps sweep (benchmark/cc/bwd_kernels.py --mode warps, 500 families, fitted
#      theta). prepare 4/16/32 warps: 1.04x/1.04x/1.07x; subtree 8 warps: 1.04x. ALL REJECTED on
#      correctness: every non-default value moved the gradient by 3 to 7 against a run-to-run
#      atomics noise floor of 1.2e-3 and a gradient whose own largest entry is 0.59.
#
# THE BLOCKER, and the one change that would matter. The prepare kernel's cost IS the 34 gathers
# per species per row, and they exist only to build ancestor_sum for
# ``valid_receiver_mass = total_receiver_mass - ancestor_sum``. Any cheaper way to get that mass --
# a top-down level walk, a depth-first prefix sum, or simply a different warp count -- reorders one
# of those two sums. That subtraction cancels catastrophically at the fitted theta: a ~1e-7
# relative change in total_receiver_mass moves the finished gradient by O(3), 3000x the atomics
# noise, because donor_adjoint_coefficient divides by the difference. The FORWARD already avoids
# this exact cancellation -- see _valid_receiver_index_tables in gpurec/core/inference/forward.py,
# which builds the same mass as two running sums of non-negative terms for precisely this reason.
# Converting the backward to that form would remove the 34 gathers AND the cancellation together,
# but it changes the gradient on purpose and needs a decision, not a kernel tweak.
set -uo pipefail
THETA=${1:-fitted}
KERNELS=(
  _prepare_reconciliation_self_loop_vjp_kernel
  _accumulate_transfer_subtree_vjp_kernel
  _accumulate_gene_split_event_vjp_kernel
  _accumulate_transfer_receiver_log_probability_vjp_kernel
)
EXTRA_METRICS=launch__registers_per_thread,smsp__inst_executed.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_lsu.sum,launch__waves_per_multiprocessor
for K in "${KERNELS[@]}"; do
  echo "############ NCU FULL $K ($THETA) ############"
  $NCU --target-processes all --kernel-name "regex:$K" --launch-skip 40 --launch-count 1 --set full \
    $CC_PY benchmark/cc/bwd_kernels.py --mode probe --species $CC_SPECIES --families $CC_FAMILIES \
    --limit 100 --theta-pt $CC_RUNS/results/full_v3.pt --theta "$THETA" 2>&1
  echo "############ NCU PIPES $K ($THETA) ############"
  $NCU --target-processes all --kernel-name "regex:$K" --launch-skip 40 --launch-count 1 \
    --metrics $EXTRA_METRICS \
    $CC_PY benchmark/cc/bwd_kernels.py --mode probe --species $CC_SPECIES --families $CC_FAMILIES \
    --limit 100 --theta-pt $CC_RUNS/results/full_v3.pt --theta "$THETA" 2>&1
done
