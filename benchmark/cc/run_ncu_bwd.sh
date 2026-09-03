#!/bin/bash
# Nsight Compute on the four per-wave BACKWARD kernels, one representative launch each.
#
# Usage (inside a job, after sourcing env.sh):  bash benchmark/cc/run_ncu_bwd.sh THETA
# THETA is "fitted" or "flat-6". One ncu process per kernel; each runs one gradient on 100
# families and profiles the 41st launch of the named kernel, first with --set full and then with
# a short list of pipe/instruction counters.
#
# WHAT THIS MEASURED (H100 NVL, 100 families, fitted theta from $CC_RUNS/results/full_v3.pt,
# forward and adjoint self-loops both "exact", 2026-09-03), BEFORE the self-loop VJP kernel was
# rebuilt around the additive valid-receiver mass:
#
#   kernel                                    grid x block  dur us  occupancy  SM%   DRAM%  regs
#   _prepare_reconciliation_self_loop_vjp        462 x 256   174.4  12.5/12.5  28.9   0.58   187
#   _accumulate_transfer_subtree_vjp            2870 x 128   106.1  75.0/43.6  34.5   5.54    40
#   _accumulate_gene_split_event_vjp            1435 x 256    60.3  62.5/45.7  38.5  10.21    48
#   _accumulate_transfer_receiver_log_prob_vjp   462 x  64    64.5  43.8/ 9.4  10.4   1.09    72
#
#   None of the four was close to the memory roofline and none was compute-bound: DRAM throughput
#   0.6-10%, top pipe ALU at 22-29%, all four LATENCY-bound.
#     prepare:  1 clade row per program, 2048 species lanes, 8 warps. 187 registers per thread cap
#               it at ONE block per SM (2 warps per scheduler, 12.5% occupancy) and 66.8% of cycles
#               had no eligible warp. 46 exp2/log2 per element, 34 of them from an ancestor walk;
#               486 ALU and 161 load/store per element, again mostly that walk. 57% excessive
#               sectors at 21 GB/s with 94% of loads hitting L1 -- the walk re-read the same 8 KB
#               Pi row 34 times, so it was L1-LATENCY bound, not bandwidth bound.
#     subtree:  33 tl.debug_barrier()s walking the species tree bottom-up through global scratch;
#               top stall (31.9% of 15.0 cycles per issued instruction) the shared-memory
#               scoreboard those barriers sit on. 65% excessive sectors.
#     receiver: one program of 64 threads per clade row -- 0.25 waves per SM, 9.4% achieved
#               occupancy, 85.7% of cycles with no eligible warp. Starved, not busy.
#     gene split: the healthiest (IPC 2.05, 38.5% SM, 39.8% memory); its losses are 30% excessive
#               sectors from the species-child gathers and imbalance from early-returning rows.
#
# WHERE A GRADIENT GOES (nsys over ONE gradient, 500 families, fitted theta; total GPU 4.90 s,
# gradient wall 5.45 s). Take this, not a grad+Hessian capture, as the weighting for gradient
# work -- the Hessian re-runs the same kernels and roughly doubles their launch counts:
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
# WHAT DID NOT WORK, all measured on the OLD kernel and all reverted. Every one of these attacks
# the ancestor walk's cost without removing the walk, and none of them paid:
#   1. A precomputed [34, S] table of each species' d-th ancestor, replacing the sp_parent hop
#      chain. Bit-identical, removes one scattered gather per depth and the serial dependency
#      between depths. MEASURED 227 us (from 174): the table is 267 KB and the loop streams all of
#      it per row, which evicts the 8 KB Pi row the gather depends on (L1 hit 94% -> 79%). Marking
#      the table reads ".cg" so they skip L1 made it 243 us.
#   2. That table plus reading the ancestor masses out of the receiver_mass scratch row instead of
#      recomputing exp2 from Pi_star: -74% transcendental instructions, -22% instructions,
#      172.7 us, and 827 ms (from 843) over a gradient with the wall time unchanged.
#   3. Sizing the subtree kernel's level walk by nodes (128) instead of species (256), cutting the
#      33 levels' padded lane-slots from 8704 to 4608 for 1006 internal nodes: 663 ms (from 655).
#   4. num_warps 2 -> 8 on the starved receiver-log-probability kernel: 282 ms (from 267).
#   5. A num_warps sweep on the old kernel: prepare 4/16/32 gave 1.04x/1.04x/1.07x and subtree 8
#      gave 1.04x, but EVERY non-default value moved the gradient by 3 to 7 against a run-to-run
#      atomics noise floor of 1.2e-3 and a gradient whose largest entry is 0.59. All rejected.
#
# WHAT DID WORK: building the valid receiver mass additively. The walk existed only to form
# ancestor_sum for ``valid_receiver_mass = total_receiver_mass - ancestor_sum``, and that
# subtraction is also why finding 5 above happened -- a ~1e-7 relative change in the row total came
# back magnified through the cancellation. The kernel now builds the same mass the way the forward
# self-loop already did, as two running sums of non-negative terms over the depth-first interval
# order (gpurec/core/valid_receivers.py). Both the walk and the row-wide reduction are gone.
#   prepare kernel, same launch: 174.4 -> 61.4 us (2.84x). Instructions 28.16 -> 10.65 M, ALU
#   14.38 -> 4.22 M, load/store 4.76 -> 1.60 M, exp2/log2 1.368 -> 0.362 M (46.2 -> 12.2 per
#   element). Registers 187 -> 175, occupancy still 12.4% -- it is simply doing far less work.
#   With the cancellation gone, subtree num_warps 8 became correctness-neutral (gradient moves
#   5.1e-3 against an 8.4e-3 noise floor, versus 3.5 before) and its 1.052x was taken.
#   One gradient, 500 families, mean of 3, same job: fitted 5.389 -> 4.765 s (1.131x),
#   theta = -6  4.759 -> 4.083 s (1.166x).
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
