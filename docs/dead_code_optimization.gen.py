#!/usr/bin/env python3
"""Generate the optimization dead-code manifest for gpurec.

Dead = not reachable from ANY of the three optimization entry points
(fit_global / fit_genewise / fit_specieswise+map_cv), per the static
reachability analysis. For each dead FILE we also grep the whole repo to
classify whether it is *safe to delete* or blocked by tests/gates/scripts/
public-API references (dead-in-optimization != safe-to-delete).

Re-runnable: `python gen_manifest.py` writes docs/dead_code_optimization.{json,md}.
"""
from __future__ import annotations
import json, subprocess, re
from pathlib import Path

# Repo root = parent of docs/ (this script lives in docs/). Portable across clones.
REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "gpurec"
OUT_JSON = REPO / "docs" / "dead_code_optimization.json"
OUT_MD = REPO / "docs" / "dead_code_optimization.md"

# Trees that are NOT the library itself (references here don't affect the optimization path
# but DO block a whole-file deletion).
EXTERNAL_TREES = ["tests", "gates", "scripts", "experiments", "benchmark",
                  "cv-results-consolidated", "consolidate-release-untracked", "notebooks"]

# ---- Dead whole files (dead across all 3 optimization modes) -------------------------
# module_path (import form) -> note
DEAD_FILES = {
    "gpurec/solver/curvature.py":              "analytic gauge-projected joint-curvature core",
    "gpurec/solver/genewise_curvature.py":     "analytic genewise joint curvature",
    "gpurec/solver/origination_curvature.py":  "analytic origination joint curvature",
    "gpurec/solver/receiver_curvature.py":     "analytic receiver (theta,alpha) joint curvature",
    "gpurec/fit/map_fit.py":                   "old MAP fit driver (spectrum_min/_deflate_step/fit)",
    "gpurec/fit/baselines.py":                 "gd / lbfgs_scipy calibration baselines",
    "gpurec/batched_lbfgs.py":                 "BatchedLBFGS optimizer (re-exported as public API)",
    "gpurec/core/kernels/centered_pi_forward.py": "centered-Pi forward kernels (env-gated path, never executes)",
    "gpurec/api/_autograd.py":                 "nn.Module forward autograd Functions (model.forward path)",
    "gpurec/distributed.py":                   "multi-GPU driver (not part of any fit recipe)",
}
# NOTE: gpurec/core/backtracking/input.py is NOT listed: it is a separate PUBLIC API
# (sample_reconciliations, exported in gpurec/__init__.py), not optimization dead code.

# ---- Dead symbols inside files that are otherwise live -------------------------------
# file -> list of (line, name, kind)
DEAD_SYMBOLS = {
    "gpurec/fit/newton_cg.py": [(20,"_fd_hessian_hvp","func"),(274,"newton_tr","func"),(366,"newton_cg","func")],
    "gpurec/fit/optimize.py": [(108,"first_order","func"),(226,"newton_polish","func"),
        (277,"_exact_ridge_lambda","func"),(297,"ridge_anneal","func"),(490,"optimize","func")],
    "gpurec/fit/map_cv.py": [(170,"_smoke","func"),(34,"_DEFAULT_SO","var"),(78,"MAP_CV_REFERENCE","var")],
    "gpurec/solver/cg.py": [(76,"lanczos_min_eigpair","func"),(101,"steihaug_cg","func"),(182,"cg_solve","func")],
    "gpurec/solver/ggn.py": [(74,"make_ggn_hvp","func")],
    "gpurec/solver/hvp_exact.py": [(96,"_head_seed_tangents","func")],
    "gpurec/solver/forward_tangent.py": [(52,"param_jvp_weighted","func")],
    "gpurec/solver/penalties.py": [(18,"_root_mask","func"),(29,"tv_prior_and_grad","func"),
        (58,"_logsumexp2","func"),(62,"origination_log_pO","func"),(67,"origination_log_pO_floored","func"),
        (77,"species_node_depths","func"),(90,"species_root_index","func"),(98,"OriginationPenalty","class"),
        (129,"origination_penalty_value","func"),(175,"origination_penalty_and_grad","func")],
    "gpurec/api/model.py": [(192,"configure_solver","method"),(200,"clear_warm_starts","method"),
        (208,"_stream_batches","method"),(274,"replan_batches","method"),(311,"set_batch_solver_options","method"),
        (339,"convergence_report","method"),(377,"classify_families","method"),
        (395,"_default_group_options","method"),(413,"adapt_solver_to_convergence","method"),(451,"forward","method")],
    "gpurec/api/_execution.py": [(133,"evaluate_static_convergence","func")],
    "gpurec/api/_implicit_grad.py": [(39,"_bicgstab","func")],
    "gpurec/core/inference/solver.py": [(188,"origination_grad_from_root_rows","func")],
    "gpurec/core/inference/forward.py": [(16,"CenteredPiForwardState","class"),(155,"pi_wave_forward_centered","func")],
    "gpurec/core/parameters/extract_parameters.py": [(47,"origination_log_probs_from_weights","func")],
    "gpurec/core/kernels/wave_backward.py": [(187,"_gmres_solve_wave_self_loop","func"),
        (210,"_gmres_solve_wave_self_loop_fixed_cgs2","func")],
    "gpurec/core/kernels/wave_backward_kernels.py": [(1145,"_dts_grad_mt_two_stage_reduce_kernel","func"),
        (1177,"_pibar_ud_side_active_kernel","func")],
    "gpurec/core/kernels/_dts_layout_contract.py": [(23,"DtsForwardAddressMode","class"),
        (40,"DtsForwardParamLayout","class"),(59,"dts_forward_param_layout","func"),
        (211,"_stride_tuple","func"),(237,"_contiguous_strides","func")],
    "gpurec/core/kernels/wave_tangent.py": [(35,"_wave_step_tangent_kernel","func"),(411,"compute_wave_step_tangent","func")],
    "gpurec/core/scheduling/batching.py": [(139,"plan_batch_wave_layouts","func"),(327,"plan_family_batches","func"),
        (166,"_split_rows_for_family","func"),(186,"build_wave_layout","func")],
    "gpurec/config/gpurec_config.py": [(72,"_merge_into","method"),(121,"validate","method"),(128,"to_dict","method"),
        (133,"from_dict","method"),(175,"optimize_reference","method"),(182,"from_toml","method"),(201,"load_config","func")],
    "gpurec/config/newton.py": [(40,"validate","method")],
}

# Symbols flagged uncertain (agents disagreed) — not asserted dead.
UNCERTAIN = [("gpurec/core/kernels/wave_backward.py",35,"_device_scalar_param",
    "uniform-mode agent listed reached; genewise+specieswise derived dead (params are tensors). Verify before removing.")]

# Branches never taken in ANY optimization mode (file:line -> condition)
DEAD_BRANCHES = [
    ("gpurec/core/inference/forward.py",43,"GPUREC_CENTERED_PI_FORWARD env gate — never set"),
    ("gpurec/core/inference/solver.py",38,"if use_receiver_weights: — all modes use uniform receiver weights (forward weighted branch)"),
    ("gpurec/api/_execution.py",40,"non-uniform origination branch — origination weights always uniform"),
    ("gpurec/api/_implicit_grad.py",771,"_bicgstab arm of E-adjoint selector never chosen (modes use neumann or gmres)"),
    ("gpurec/core/kernels/wave_backward.py",445,'if self_loop_solver=="gmres": — self-loop always Neumann'),
    ("gpurec/core/kernels/wave_backward.py",977,"use_grad_mt_two_stage — grad_mt layout never 0/shared[S]"),
    ("gpurec/core/kernels/wave_backward.py",1114,"elif skip_zero_sides: — side_active tensor always supplied"),
    ("gpurec/api/model.py",451,"model.forward / _GeneReconFunction.apply — nn.Module autograd path never entered"),
    ("gpurec/api/_execution.py",82,"GPUREC_WARM_ADJOINT env gate — never set (warm-adjoint cache path)"),
    ("gpurec/fit/newton_cg.py",79,"if with_receiver: — never True; sole door to the solver/*curvature cluster"),
]

# Dead function PARAMETER
DEAD_PARAMS = [("gpurec/fit/global_fit.py",42,"fit_global(solver_options=...)",
    "documented 'accepted for API compatibility but not used'; recipe hard-codes its own tiers")]

# Code that is dead in SOME modes but live in others (do NOT delete) — informational
MODE_PARTITIONED = [
    "solver/hvp_exact.py + ggn.vjp_root_to_theta + solver/forward_tangent.py + solver/cg.py witness + "
    "kernels dts_so/dts_tangent/e_step_so/e_step_tangent/wave_so/wave_tangent-selfloop: LIVE only in specieswise "
    "(exact-HVP Newton); dead in uniform/genewise (FD Hessian).",
    "_neumann_e_adjoint: LIVE only in uniform. _gmres E-adjoint: LIVE in genewise+specieswise. "
    "(Only _bicgstab is dead in all three.)",
    "genewise_loss_vector[_and_grad] / evaluate_static_loss_vector_grad / stream_genewise_loss_vector_grad: "
    "LIVE in uniform+genewise, dead in specieswise. Conversely stream_batches / evaluate_static_loss_grad / "
    "nll_from_root_rows: LIVE only in specieswise.",
]

def grep_import_refs(stem: str, exclude_module_file: str | None = None) -> dict[str, list[str]]:
    """Find IMPORT references to a module `stem` across trees. Greps the plain stem (portable),
    then keeps only lines that look like imports, using Python's regex (reliable, unlike the
    shell grep whose \\b handling is aliased). Returns {tree: [sample import lines]}."""
    imp = re.compile(rf"(from\s+[\w.]*\b{re.escape(stem)}\b\s+import|import\s+[\w.]*\b{re.escape(stem)}\b)")
    hits: dict[str, list[str]] = {}
    for tree in EXTERNAL_TREES + ["gpurec"]:
        d = REPO / tree
        if not d.exists():
            continue
        try:
            r = subprocess.run(["grep","-rIn",stem,str(d),"--include=*.py"],
                               capture_output=True,text=True,timeout=120)
        except Exception:
            continue
        lines = []
        for ln in r.stdout.splitlines():
            if "__pycache__" in ln or "/.venv/" in ln or "/build/" in ln:
                continue
            if exclude_module_file and f"/{exclude_module_file}:" in ln:
                continue
            # ln = path:lineno:content ; test only the content against the import regex
            parts = ln.split(":", 2)
            content = parts[2] if len(parts) == 3 else ln
            if imp.search(content):
                lines.append(ln)
        if lines:
            hits[tree] = lines[:6]
    return hits

def classify_file(module_path: str):
    stem = Path(module_path).stem
    refs = grep_import_refs(stem, exclude_module_file=Path(module_path).name)
    # separate package (import within library, excluding self and dead siblings) from external
    ext = {t: v for t, v in refs.items() if t != "gpurec"}
    pkg = refs.get("gpurec", [])
    # public API re-export check
    public = any("__init__.py" in ln for ln in pkg)
    if public:
        safety = "blocked:public-api"
    elif ext:
        safety = "blocked:" + ",".join(sorted(ext.keys()))
    elif pkg:
        # only referenced within the (dead) cluster or by a dead symbol in a live file
        safety = "review:pkg-refs"
    else:
        safety = "safe"
    return safety, ext, pkg

def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    files_out = []
    for mod, note in DEAD_FILES.items():
        safety, ext, pkg = classify_file(mod)
        loc = len((PKG.parent / mod).read_text().splitlines()) if (PKG.parent / mod).exists() else None
        files_out.append({"file": mod, "loc": loc, "note": note, "deletion_safety": safety,
                          "external_refs": ext, "package_refs": pkg})

    manifest = {
        "analysis": "gpurec optimization dead-code (static reachability from fit_global / fit_genewise / fit_specieswise+map_cv)",
        "definition": "dead == not reachable from ANY of the three optimization entry points",
        "caveat": "dead-in-optimization != safe-to-delete; see deletion_safety per file",
        "dead_files": files_out,
        "dead_symbols_in_live_files": [
            {"file": f, "symbols": [{"line": l, "name": n, "kind": k} for (l, n, k) in syms]}
            for f, syms in DEAD_SYMBOLS.items()],
        "uncertain": [{"file": f, "line": l, "name": n, "reason": r} for (f, l, n, r) in UNCERTAIN],
        "dead_branches": [{"file": f, "line": l, "condition": c} for (f, l, c) in DEAD_BRANCHES],
        "dead_params": [{"file": f, "line": l, "param": p, "note": n} for (f, l, p, n) in DEAD_PARAMS],
        "mode_partitioned_keep": MODE_PARTITIONED,
    }
    OUT_JSON.write_text(json.dumps(manifest, indent=2))

    # ---- Markdown companion ----
    L = []
    L.append("# gpurec optimization dead-code manifest\n")
    L.append("Generated by `scratchpad/gen_manifest.py` (re-runnable). **Dead** = not reachable from any of the three")
    L.append("optimization entry points: `fit_global` (uniform), `fit_genewise` (genewise), `fit_specieswise`+`map_cv` (specieswise).\n")
    L.append("> **dead-in-optimization != safe-to-delete.** Many dead modules are still referenced by tests, gates,")
    L.append("> out-of-tree scripts, or the public API. The `deletion_safety` column reflects a whole-repo grep.\n")
    L.append("## Dead whole files\n")
    L.append("| file | LOC | deletion safety | blockers |")
    L.append("|------|-----|-----------------|----------|")
    for f in files_out:
        blockers = "; ".join(f"{t}({len(v)})" for t, v in f["external_refs"].items()) or \
                   ("public re-export" if f["deletion_safety"]=="blocked:public-api" else
                    ("pkg refs" if f["deletion_safety"].startswith("review") else "—"))
        L.append(f"| `{f['file']}` | {f['loc']} | **{f['deletion_safety']}** | {blockers} |")
    L.append("\n## Dead symbols in otherwise-live files\n")
    for f, syms in DEAD_SYMBOLS.items():
        names = ", ".join(f"`{n}`({l})" for (l, n, k) in syms)
        L.append(f"- **{f}**: {names}")
    L.append("\n## Dead control-flow branches (never taken in any mode)\n")
    for f, l, c in DEAD_BRANCHES:
        L.append(f"- `{f}:{l}` — {c}")
    L.append("\n## Dead function parameter\n")
    for f, l, p, n in DEAD_PARAMS:
        L.append(f"- `{f}:{l}` — `{p}`: {n}")
    L.append("\n## Uncertain (verify before removing)\n")
    for f, l, n, r in UNCERTAIN:
        L.append(f"- `{f}:{l}` `{n}` — {r}")
    L.append("\n## Mode-partitioned — dead in some modes, LIVE in others (do NOT delete)\n")
    for s in MODE_PARTITIONED:
        L.append(f"- {s}")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print("\nDeletion-safety summary:")
    for f in files_out:
        print(f"  {f['deletion_safety']:24s} {f['file']}")

if __name__ == "__main__":
    main()
