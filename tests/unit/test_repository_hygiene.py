"""Repository-level source and documentation hygiene checks."""

import ast
import json
import re
import subprocess
import tomllib
from pathlib import Path

import gpurec
import gpurec.workflow as workflow

SUBPROCESS_TIMEOUT = 30


def _tracked_files(root: Path, *patterns: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    return [root / line for line in result.stdout.splitlines() if line]


def _is_subprocess_run(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "run"
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    parts: list[str] = []
    while isinstance(decorator, ast.Attribute):
        parts.append(decorator.attr)
        decorator = decorator.value
    if isinstance(decorator, ast.Name):
        parts.append(decorator.id)
    return ".".join(reversed(parts))


def _has_pytest_mark(function: ast.FunctionDef, marker: str) -> bool:
    return any(
        _decorator_name(decorator).endswith(f"mark.{marker}")
        for decorator in function.decorator_list
    )


def _is_pytest_fixture(function: ast.FunctionDef) -> bool:
    return any(
        _decorator_name(decorator).endswith("pytest.fixture")
        or _decorator_name(decorator) == "fixture"
        for decorator in function.decorator_list
    )


def _function_mentions_name(function: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name
        for node in ast.walk(function)
    )


def test_workflow_and_backtracking_use_public_model_surface():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "gpurec" / "backtracking.py"]
    paths.extend(sorted((root / "gpurec" / "workflow").glob("*.py")))
    forbidden = (
        "model._dataset",
        "model._active_static",
        "model._active_theta",
        "model._current_batch_index",
        "model._ensure_batch_static",
        "model._batch_statics",
        "model._static",
        'getattr(model, "_',
        "from gpurec.core.preprocess_cpp",
        "_load_species_gene_ext",
        "from gpurec.api.autograd import _extract_parameters",
        "Pi_wave_forward",
        "E_fixed_point",
    )

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(root)} contains {token}")

    assert offenders == []


def test_workflow_sampling_uses_public_config_constants():
    root = Path(__file__).resolve().parents[2]
    sampling_text = (root / "gpurec" / "workflow" / "sampling.py").read_text(
        encoding="utf-8"
    )

    assert "_UINT64_MAX" not in sampling_text
    assert "UINT64_MAX" in sampling_text


def test_hogenom_scripts_use_public_model_surface():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "scripts" / "export_hogenom_rates_from_checkpoint.py",
        root / "scripts" / "fast_optimize_hogenom_ccp.py",
        root / "scripts" / "hogenom_ccp_wandb_opt.py",
        root / "scripts" / "hogenom_opt_helpers.py",
        root / "scripts" / "make_hogenom_branchscale_penalty_report.py",
        root / "scripts" / "optimize_hogenom_ccp_global_uniform.py",
        root / "scripts" / "optimize_hogenom_ccp_hydra.py",
        root / "scripts" / "optimize_hogenom_ccp_specieswise_uniform.py",
        root / "scripts" / "optimize_hogenom_ccp_wandb.py",
        root / "scripts" / "optimize_hogenom_penalty316_kkt.py",
        root / "scripts" / "profile_hogenom_ccp_pass.py",
    ]
    forbidden = (
        "model._",
        'getattr(model, "_',
        "from gpurec.api.model import _",
        "from gpurec.api.model import _GeneReconFullLossFunction",
        "_stream_full_batches",
        "_ensure_batch_static",
        "_current_batch_index",
        "_schedule_prefetch",
        "_theta_for_batch_index",
        "_evaluate_static_state",
        "from gpurec.core.preprocess_cpp import _load_extension",
    )

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(root)} contains {token}")

    assert offenders == []


def test_python_scripts_do_not_use_runtime_asserts():
    root = Path(__file__).resolve().parents[2]
    offenders = [
        str(path.relative_to(root))
        for path in sorted((root / "scripts").glob("*.py"))
        if "assert " in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_tests_subprocess_calls_have_explicit_timeouts():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in sorted((root / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_subprocess_run(node):
                continue
            if any(keyword.arg == "timeout" for keyword in node.keywords):
                continue
            offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert offenders == []


def test_checked_fixture_contracts_are_documented():
    root = Path(__file__).resolve().parents[2]
    tests_readme = (root / "tests" / "README.md").read_text(encoding="utf-8")
    normalized_tests_readme = " ".join(tests_readme.split())
    stochastic_readme = (
        root / "tests" / "data" / "test_trees_3" / "README.md"
    ).read_text(encoding="utf-8")
    normalized_stochastic_readme = " ".join(stochastic_readme.split())
    rust_readme = (
        root / "tests" / "fixtures" / "backtracking" / "README.md"
    ).read_text(encoding="utf-8")
    normalized_rust_readme = " ".join(rust_readme.split())
    stochastic_test = (
        root / "tests" / "integration" / "test_stochastic_backtracking.py"
    ).read_text(encoding="utf-8")
    rust_fixture = json.loads(
        (
            root / "tests" / "fixtures" / "backtracking" / "speciation.json"
        ).read_text(encoding="utf-8")
    )

    for token in (
        "tests/fixtures/backtracking/README.md",
        "tests/data/test_trees_3/README.md",
        "35 x 15",
        "CPU-only Rust JSON fixture",
    ):
        assert token in normalized_tests_readme

    for token in (
        "smallest checked CUDA stochastic-backtracking fixture",
        "`sp.nwk`: one rooted binary species tree with 15 postorder species nodes",
        "`g.nwk`: one rooted binary gene tree",
        "`pi` payload with 35 clade rows and 15 species columns",
        "The Rust sampler should emit one `recGeneTree`",
        "10 leaf events",
        "CPU-only Rust CLI checks should use `tests/fixtures/backtracking/speciation.json`",
    ):
        assert token in normalized_stochastic_readme

    for token in (
        "`speciation.json` is a hand-authored CPU-only smoke fixture",
        "Species order: postorder `A`, `B`, `Root`",
        "Split set: one deterministic split",
        "row-major `3 x 3` base-2 log matrix",
        "impossible states use the `-1.0e300` sentinel",
        "all mass on the root species",
        "visible speciation at `Root`",
    ):
        assert token in normalized_rust_readme

    for token in (
        "TEST_TREES_3_EXPECTED_CLADES = 35",
        "TEST_TREES_3_EXPECTED_SPECIES = 15",
        "TEST_TREES_3_EXPECTED_LEAVES = 10",
    ):
        assert token in stochastic_test

    assert rust_fixture["species_names_postorder"] == ["A", "B", "Root"]
    assert rust_fixture["root_clade"] == 2
    assert rust_fixture["leaf_species"] == [0, 1, None]
    assert rust_fixture["pi"]["rows"] == 3
    assert rust_fixture["pi"]["cols"] == 3
    assert rust_fixture["origination_probs"] == [0.0, 0.0, 1.0]


def test_hogenom_scripts_are_marked_as_legacy_experiment_surface():
    root = Path(__file__).resolve().parents[2]
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    project_readme = (root / "README.md").read_text(encoding="utf-8")

    assert "gpurec optimize" in scripts_readme
    assert "legacy" in scripts_readme
    assert "HOGENOM reproducers" in scripts_readme
    assert "legacy checkout-local experiment launchers" in project_readme
    assert "Legacy HOGENOM W&B wrapper" in project_readme
    assert "Curated HOGENOM W&B run" not in project_readme
    for script_name in (
        "hogenom_ccp_wandb_opt.py",
        "fast_optimize_hogenom_ccp.py",
    ):
        script_text = (root / "scripts" / script_name).read_text(encoding="utf-8")
        assert script_name in scripts_readme
        assert "Legacy checkout-local HOGENOM experiment launcher" in script_text


def test_gpu_tests_use_explicit_module_level_markers():
    root = Path(__file__).resolve().parents[2]
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")

    assert "GPU_TEST_FILES" not in conftest
    assert "SLOW_TEST_PATTERNS" not in conftest
    assert "item.path.name" not in conftest
    assert "item.nodeid" not in conftest
    assert "kernels" in conftest
    offenders: list[str] = []
    this_file = Path(__file__).resolve()
    for path in sorted((root / "tests").rglob("test_*.py")):
        if path == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "not torch.cuda.is_available()" not in text
            and 'reason="CUDA required"' not in text
            and 'pytest.skip("CUDA required")' not in text
        ):
            continue
        if "pytestmark" not in text or "pytest.mark.gpu" not in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_unit_tests_using_large_cuda_fixture_are_marked_slow():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    this_file = Path(__file__).resolve()

    for path in sorted((root / "tests" / "unit").glob("test_*.py")):
        if path == this_file:
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [
            node for node in module.body if isinstance(node, ast.FunctionDef)
        ]
        large_fixture_names: set[str] = {"data_dir_1000"}

        changed = True
        while changed:
            changed = False
            for function in functions:
                if not _is_pytest_fixture(function):
                    continue
                arg_names = {arg.arg for arg in function.args.args}
                if (
                    arg_names & large_fixture_names
                    or any(
                        _function_mentions_name(function, fixture_name)
                        for fixture_name in large_fixture_names
                    )
                ) and function.name not in large_fixture_names:
                    large_fixture_names.add(function.name)
                    changed = True

        for function in functions:
            if not function.name.startswith("test_"):
                continue
            arg_names = {arg.arg for arg in function.args.args}
            uses_large_fixture = bool(arg_names & large_fixture_names) or any(
                _function_mentions_name(function, fixture_name)
                for fixture_name in large_fixture_names
            )
            if uses_large_fixture and not _has_pytest_mark(function, "slow"):
                offenders.append(f"{path.relative_to(root)}::{function.name}")

    assert offenders == []


def test_pytest_marker_taxonomy_matches_current_layout():
    root = Path(__file__).resolve().parents[2]
    pytest_ini = (root / "pytest.ini").read_text(encoding="utf-8")
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    tests_readme = (root / "tests" / "README.md").read_text(encoding="utf-8")

    for marker in ("unit", "integration", "kernel", "gpu", "slow"):
        assert f"    {marker}:" in pytest_ini
    assert "gradient:" not in pytest_ini
    assert "autograd:" not in pytest_ini

    assert "kernel_marker = pytest.mark.kernel" in conftest
    assert 'test_section == "kernels"' in conftest
    assert "item.add_marker(kernel_marker)" in conftest
    assert "both\n`integration` and `kernel` to `tests/kernels`" in tests_readme


def test_tests_use_pytest_managed_temporary_paths():
    root = Path(__file__).resolve().parents[2]
    forbidden = "/" + "tmp/"
    offenders = [
        str(path.relative_to(root))
        for path in _tracked_files(root, "tests/**/*.py")
        if forbidden in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_generated_artifact_roots_stay_ignored_and_untracked():
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    expected_ignored_patterns = (
        ".preprocess_cache/",
        "build/",
        "dist/",
        "*.egg-info/",
        "output_*/",
        "wandb/",
        "crates/*/target/",
    )
    for pattern in expected_ignored_patterns:
        assert pattern in gitignore

    tracked_artifacts = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(
            root,
            ".preprocess_cache",
            "build",
            "dist",
            "*.egg-info",
            "output_*",
            "wandb",
            "crates/*/target",
        )
    ]

    assert tracked_artifacts == []


def test_no_tracked_root_test_modules_bypass_pytest_collection():
    root = Path(__file__).resolve().parents[2]
    offenders = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(root, "test_*.py")
        if path.parent == root
    ]

    assert offenders == []


def test_unit_gpu_tree_data_fixtures_are_centralized():
    root = Path(__file__).resolve().parents[2]
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "def data_dir_100" in conftest
    assert "def data_dir_1000" in conftest

    dataset_names = ("test_trees_100", "test_trees_1000")
    offenders: list[str] = []
    this_file = Path(__file__).resolve()
    for path in sorted((root / "tests" / "unit").glob("test_*.py")):
        if path == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in dataset_names):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_project_readme_documents_preprocess_cache_refresh_guidance():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    assert "--preprocess-cache output_gpurec/preprocess_cache" in normalized
    assert "--refresh-preprocess-cache" in normalized
    assert "safe-loading or cache-shape validation errors" in normalized
    assert "original tree inputs" in normalized


def test_project_readme_documents_workflow_config_aliases():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")

    for token in (
        "`dtype`",
        "`fp32`",
        "`single`",
        "`torch.float64`",
        "`family_chunk_size`",
        "JSON `null`",
        "`batch_packing`",
        "`clade_first_fit`",
        "`depth_first_fit`",
        "`first_fit_decreasing`",
        "`wave_first_fit`",
        "`clade_budget`",
    ):
        assert token in project_readme


def test_bfloat16_policy_is_documented_as_direct_api_only():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())
    uniform_module = ast.parse(
        (root / "gpurec" / "api" / "uniform_chunked.py").read_text(
            encoding="utf-8"
        )
    )
    uniform_class = next(
        node
        for node in uniform_module.body
        if isinstance(node, ast.ClassDef) and node.name == "UniformChunkedReconModel"
    )
    uniform_doc = ast.get_docstring(uniform_class) or ""
    cli_text = (root / "gpurec" / "cli.py").read_text(encoding="utf-8")
    config_text = (root / "gpurec" / "workflow" / "config.py").read_text(
        encoding="utf-8"
    )

    for token in (
        "workflow CLI intentionally supports only float32 and float64",
        "direct `UniformChunkedReconModel` constructor also accepts `torch.bfloat16`",
        "experimental CUDA-only path",
        "forward/NLL probes",
        "not a supported workflow configuration dtype",
        "not supported by the retained Pi backward/gradient path",
        "release smokes, optimizer checkpoints, or Hessian/second-order diagnostics",
    ):
        assert token in normalized
    for token in (
        "Both native CUDA prototypes compute dynamic shared-memory requirements",
        "`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`",
        "pass that same byte count to `cuLaunchKernel`",
        "preflight the requested scratch size against the device shared-memory limit",
    ):
        assert token in normalized
    for token in (
        "torch.bfloat16",
        "experimental",
        "forward/NLL probes",
        "workflow configuration",
        "CLI runs intentionally expose only fp32/fp64",
        "retained Pi",
    ):
        assert token in uniform_doc
    assert "bfloat16" not in cli_text
    assert "bf16" not in cli_text
    assert "bfloat16" not in config_text
    assert "bf16" not in config_text


def test_uniform_chunked_loss_and_grad_documents_e_adjoint_stats():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())
    uniform_module = ast.parse(
        (root / "gpurec" / "api" / "uniform_chunked.py").read_text(
            encoding="utf-8"
        )
    )
    uniform_class = next(
        node
        for node in uniform_module.body
        if isinstance(node, ast.ClassDef) and node.name == "UniformChunkedReconModel"
    )
    class_doc = ast.get_docstring(uniform_class) or ""
    loss_and_grad = next(
        node
        for node in uniform_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "loss_and_grad"
    )
    method_doc = ast.get_docstring(loss_and_grad) or ""

    for token in (
        "`UniformChunkedReconModel.loss_and_grad()` returns `(loss, grad, stats)`",
        "selected chunk/family counts",
        "E-adjoint solve telemetry",
        "`e_adjoint_method`",
        "`e_adjoint_iterations`",
        "`e_adjoint_rel_res`",
        "`e_adjoint_success`",
    ):
        assert token in normalized
    for token in (
        "stats dictionary",
        "selected chunk/family counts",
        "E-adjoint solve telemetry",
    ):
        assert token in class_doc
    for token in (
        "e_adjoint_method",
        "e_adjoint_iterations",
        "e_adjoint_rel_res",
        "e_adjoint_success",
    ):
        assert token in method_doc


def test_project_readme_documents_sampling_output_layout():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    assert "--sample-out-dir output_gpurec" in project_readme
    assert "--family-start" in project_readme
    assert "--sample-max-families" in project_readme
    assert "--seed" in project_readme
    assert "--max-events" in project_readme
    assert "output_gpurec/reconciliations/all/" in project_readme
    assert "event_counts.tsv" in project_readme
    assert "totalSpeciesEventCounts.txt" in project_readme
    assert "totalTransfers.txt" in project_readme
    assert "Successful sampling reruns replace prior gpurec-generated" in project_readme
    assert "use a separate `--sample-out-dir` to keep multiple windows" in project_readme
    for token in (
        "`event_counts.tsv` is tab-separated",
        "`totalSpeciesEventCounts.txt` is the AleRax-compatible comma-space text format",
        "`totalTransfers.txt` uses whitespace-separated source species, destination species, and average transfer count",
        "Aggregate values are averaged over the requested sample count for each retained family",
        "not over all families in the original checkpoint",
        "The sampled RecPhyloXML subset expected by gpurec contains `recGeneTree` blocks",
        "`eventsRec` event containers",
        "`branchingOut`",
        "`transferBack`",
        "gpurec-generated sample XML files are expected to contain one `recGeneTree` per file",
        "shared event-count traversal can still read multiple `recGeneTree` blocks",
    ):
        assert token in normalized


def test_project_readme_documents_gpurec_run_end_to_end_workflow():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    assert "To optimize and sample in one supported CLI workflow" in normalized
    assert "gpurec run \\" in project_readme
    assert "--config run.json" in project_readme
    assert "--samples 100" in project_readme
    assert "`gpurec run` does not accept `--checkpoint`" in project_readme
    assert "reported by the optimizer" in normalized
    assert "falling back to `checkpoints/best.pt` or `checkpoints/latest.pt`" in normalized
    assert "exits without sampling if optimization fails" in normalized
    assert "Use `gpurec sample --checkpoint ...` to sample an existing run" in project_readme


def test_project_readme_documents_workflow_optimizer_modes():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "| `adam` |",
        "| `adagrad` |",
        "| `lbfgs` |",
        "| `adam-lbfgs` |",
        "`lbfgs_line_search` is `none` or `strong_wolfe`",
        "LBFGS runtime errors stop the run with a failed status",
        "`adam_warmup_steps` controls the phase switch",
        "incompatible resumed optimizer state is discarded",
    ):
        assert token in normalized


def test_project_readme_documents_completed_resume_status():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "Resume starts from the checkpoint `next_step`",
        "If `next_step` already equals the configured `steps`",
        "performs only the final evaluation/artifact refresh",
        "returns the same `not_converged`/`max_steps` status",
        "Increase `steps` beyond the checkpoint `next_step`",
    ):
        assert token in normalized


def test_project_readme_documents_e_adjoint_diagnostics():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "History rows include aggregate `solver/*` telemetry",
        "E-adjoint nonconvergence is diagnostic-only",
        "BiCGSTAB solve returns its best iterate with `success=False`",
        "optimization continues unless the objective or gradient becomes nonfinite",
        "`solver/e_adjoint_failed_batches`",
        "relative-residual and iteration summaries",
    ):
        assert token in normalized


def test_implicit_gradient_documents_bicgstab_failure_policy():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse(
        (root / "gpurec" / "optimization" / "implicit_grad.py").read_text(
            encoding="utf-8"
        )
    )
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }

    bicgstab_doc = " ".join(
        (ast.get_docstring(functions["_bicgstab"]) or "").split()
    )
    e_adjoint_doc = " ".join(
        (ast.get_docstring(functions["_e_adjoint_and_theta_vjp"]) or "").split()
    )

    for token in (
        "Nonconvergence is reported through ``_SolveStats(success=False)``",
        "current best iterate is still returned",
        "E-adjoint gradient path currently treats it as diagnostic telemetry",
    ):
        assert token in bicgstab_doc
    for token in (
        "BiCGSTAB nonconvergence is diagnostic-only",
        "best returned E-adjoint iterate is consumed",
        "workflow history can surface failed batches",
    ):
        assert token in e_adjoint_doc


def test_strict_json_serializer_documents_sanitizing_contract():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse(
        (root / "gpurec" / "workflow" / "diagnostics.py").read_text(
            encoding="utf-8"
        )
    )
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "json_dumps_strict"
    )
    docstring = " ".join((ast.get_docstring(function) or "").split())

    for token in (
        "standards-compliant JSON",
        "Non-finite floats",
        "JSON null",
        "strict JSON readers",
    ):
        assert token in docstring


def test_prepared_origination_probs_trust_boundary_is_documented():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse(
        (root / "gpurec" / "core" / "likelihood.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_origination_probs"
    )
    docstring = " ".join((ast.get_docstring(function) or "").split())

    for token in (
        "assume_prepared=True",
        "internal trust boundary",
        "already passed this helper",
        "skips finite, nonnegative, positive-mass, and normalization checks",
        "Public entry points should keep the default",
    ):
        assert token in docstring


def test_project_readme_documents_genewise_per_family_api_contract():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "`model.nll_per_family()` and `model.full_nll_per_family()` are genewise-only",
        "row-wise optimizers",
        "`model(reduce=\"per_family\")` under `torch.no_grad()`",
        "independent per-family gradients are not defined",
    ):
        assert token in normalized


def test_project_readme_documents_leaf_species_mapping_contract():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "For direct `from_trees` inputs",
        "legacy prefix fallback",
        "`Species_gene` maps to species `Species`",
        "leaf without `_` maps to the full leaf label",
        "AleRax family files with `mapping` entries",
        "`UniformChunkedReconModel(..., leaf_species_maps=...)`",
        "`GeneDataset(..., leaf_species_maps=...)`",
    ):
        assert token in normalized


def test_newick_input_subset_is_documented_on_public_surfaces():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(project_readme.split())

    for token in (
        "retained C++ preprocessing parser supports a deliberately small Newick subset",
        "unquoted labels",
        "optional internal labels",
        "optional numeric branch lengths",
        "Branch lengths are ignored",
        "numeric text must immediately follow `:`",
        "final semicolon is optional",
        "quotes, escaping, comments, NHX or BEAST-style metadata",
        "embedded `:`, `,`, `(`, `)`, or `;` delimiters",
        "species-tree file must contain exactly one rooted binary tree",
        "gene-tree file may contain one or more semicolon-delimited records",
        "Gene multifurcations are right-binarized",
        "unary gene nodes and non-binary species nodes are rejected",
    ):
        assert token in normalized_readme

    api_model = ast.parse(
        (root / "gpurec" / "api" / "model.py").read_text(encoding="utf-8")
    )
    gene_model = next(
        node
        for node in api_model.body
        if isinstance(node, ast.ClassDef) and node.name == "GeneReconModel"
    )
    from_trees = next(
        node
        for node in gene_model.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_trees"
    )
    from_alerax = next(
        node
        for node in gene_model.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_alerax_families"
    )
    model_docs = " ".join(
        (
            f"{ast.get_docstring(from_trees) or ''} "
            f"{ast.get_docstring(from_alerax) or ''}"
        ).split()
    )
    for token in (
        "supported simple Newick subset",
        "semicolon-delimited records",
        "final record may omit its terminal semicolon",
        "Branch lengths are ignored",
        "right-binarized",
    ):
        assert token in model_docs

    core_model = ast.parse(
        (root / "gpurec" / "core" / "model.py").read_text(encoding="utf-8")
    )
    gene_dataset = next(
        node
        for node in core_model.body
        if isinstance(node, ast.ClassDef) and node.name == "GeneDataset"
    )
    gene_dataset_doc = " ".join((ast.get_docstring(gene_dataset) or "").split())
    assert "supported simple Newick subset" in gene_dataset_doc

    uniform_module = ast.parse(
        (root / "gpurec" / "api" / "uniform_chunked.py").read_text(
            encoding="utf-8"
        )
    )
    uniform_class = next(
        node
        for node in uniform_module.body
        if isinstance(node, ast.ClassDef) and node.name == "UniformChunkedReconModel"
    )
    uniform_doc = " ".join((ast.get_docstring(uniform_class) or "").split())
    assert "simple Newick subset" in uniform_doc

    preprocess_cpp = (
        root / "gpurec" / "core" / "cpp" / "preprocess.cpp"
    ).read_text(encoding="utf-8")
    for token in (
        "rooted binary species Newick tree",
        "simple-Newick gene-tree files",
        "multiple semicolon-delimited records",
    ):
        assert token in preprocess_cpp


def test_solver_reconfiguration_docs_cover_lazy_prefetch_contract():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(project_readme.split())
    model_module = ast.parse(
        (root / "gpurec" / "api" / "model.py").read_text(encoding="utf-8")
    )
    model_class = next(
        node
        for node in model_module.body
        if isinstance(node, ast.ClassDef) and node.name == "GeneReconModel"
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "configure_solver_iterations"
    )
    normalized_doc = " ".join((ast.get_docstring(method) or "").split())

    for token in (
        "lazy preprocessing or resident-batch prefetching",
        "`model.configure_solver_iterations()` updates the model defaults",
        "resident batch static states that are already built",
        "does not cancel or rewrite pending background prefetch work",
        "`model.materialize_batches()` and configure again",
    ):
        assert token in normalized_readme

    for token in (
        "model defaults and resident batch static states",
        "already built",
        "does not cancel or rewrite pending background prefetch work",
        "configure before scheduling lazy prefetch",
        "materialize resident batches and configure again",
    ):
        assert token in normalized_doc


def test_project_readme_top_level_import_examples_match_public_exports():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    imported_names: list[str] = []

    for block in re.findall(r"```python\n(.*?)```", project_readme, flags=re.S):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module == "gpurec"
            ):
                imported_names.extend(
                    alias.name for alias in node.names if alias.name != "*"
                )

    assert imported_names
    assert sorted({name for name in imported_names if name not in gpurec.__all__}) == []


def test_project_readme_documents_top_level_workflow_exports():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")

    assert sorted(name for name in workflow.__all__ if name not in project_readme) == []


def test_project_readme_documents_top_level_backtracking_helpers():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    public_helpers = sorted(
        name
        for name, module_name in gpurec._LAZY_EXPORTS.items()
        if module_name == "gpurec.backtracking" and name != "EVENT_KEYS"
    )

    for name in public_helpers:
        assert name in project_readme


def test_project_readme_documents_package_environment_flags():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    env_pattern = re.compile(r"\bGPUREC_[A-Z0-9_]+\b")
    package_env_flags = sorted(
        {
            match.group(0)
            for path in _tracked_files(root, "gpurec/**/*.py")
            for match in env_pattern.finditer(path.read_text(encoding="utf-8"))
        }
    )

    assert package_env_flags
    undocumented = [name for name in package_env_flags if name not in project_readme]

    assert undocumented == []


def test_project_readme_documents_cuda_prototype_fallback_policy():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "native CUDA prototype flags are experimental diagnostics",
        "`auto` and `enabled` fall back to the retained Triton paths",
        "required modes re-raise failures only after",
        "dtype/device/layout eligibility checks select that path",
        "`GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1`",
        "self-loop prototype fallback is silent",
        "Pibar prototype fallback is silent in `auto` but warns once in `enabled`",
        "self-loop loader preloads wheel NVRTC builtins",
        "Pibar loader currently does not",
    ):
        assert token in normalized


def test_cuda_prototype_source_documents_loader_and_fallback_policy():
    root = Path(__file__).resolve().parents[2]
    helper_module = ast.parse(
        (root / "gpurec" / "core" / "_helpers.py").read_text(encoding="utf-8")
    )
    wave_backward = ast.parse(
        (root / "gpurec" / "core" / "kernels" / "wave_backward.py").read_text(
            encoding="utf-8"
        )
    )

    helper_functions = {
        node.name: node
        for node in helper_module.body
        if isinstance(node, ast.FunctionDef)
    }
    wave_functions = {
        node.name: node for node in wave_backward.body if isinstance(node, ast.FunctionDef)
    }

    helper_doc = " ".join(
        (
            ast.get_docstring(helper_functions["_env_mode_enabled_required"])
            or ""
        ).split()
    )
    pibar_options_doc = " ".join(
        (
            ast.get_docstring(wave_functions["_cuda_pibar_from_ud_options"])
            or ""
        ).split()
    )
    backward_doc = " ".join(
        (
            ast.get_docstring(
                ast.parse(
                    (root / "gpurec" / "core" / "backward.py").read_text(
                        encoding="utf-8"
                    )
                )
            )
            or ""
        ).split()
    )
    self_loop_doc = " ".join(
        (
            ast.get_docstring(
                ast.parse(
                    (
                        root
                        / "gpurec"
                        / "core"
                        / "kernels"
                        / "wave_backward_cuda.py"
                    ).read_text(encoding="utf-8")
                )
            )
            or ""
        ).split()
    )
    pibar_doc = " ".join(
        (
            ast.get_docstring(
                ast.parse(
                    (
                        root
                        / "gpurec"
                        / "core"
                        / "kernels"
                        / "pibar_vjp_cuda.py"
                    ).read_text(encoding="utf-8")
                )
            )
            or ""
        ).split()
    )

    for token in (
        "``auto`` and ``enabled`` are best-effort modes",
        "call sites still apply their own eligibility checks",
    ):
        assert token in helper_doc
    for token in (
        "``auto`` is silent best-effort",
        "``enabled`` is best-effort with the caller's warning-on-fallback path",
        "``GPUREC_CUDA_PIBAR_FROM_UD_STRICT``",
    ):
        assert token in pibar_options_doc
    for token in (
        "fall back to the retained Triton self-loop path",
        "``ImportError``, ``RuntimeError``, or ``ValueError``",
        "required modes re-raise failures after the wave is eligible",
    ):
        assert token in backward_doc
    assert "preloads wheel-provided NVRTC builtins" in self_loop_doc
    assert "preflights dynamic shared-memory scratch size" in self_loop_doc
    assert "compiles directly without the self-loop loader's wheel NVRTC builtins preload" in pibar_doc
    assert "preflights dynamic shared-memory scratch size" in pibar_doc


def test_cuda_prototype_launchers_preflight_dynamic_shared_memory():
    root = Path(__file__).resolve().parents[2]

    for relative_path in (
        "gpurec/core/kernels/wave_backward_cuda.py",
        "gpurec/core/kernels/pibar_vjp_cuda.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        shared_pos = source.index("shared_bytes =")
        props_pos = source.index("torch.cuda.get_device_properties")
        optin_pos = source.index("shared_memory_per_block_optin")
        fallback_pos = source.index(
            "shared_memory_per_block",
            optin_pos + len("shared_memory_per_block_optin"),
        )
        check_pos = source.index("shared_bytes > int(max_shared)")
        attr_pos = source.index("CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES")
        launch_pos = source.index("cuLaunchKernel(")

        assert shared_pos < props_pos < optin_pos
        assert props_pos < fallback_pos < check_pos < attr_pos < launch_pos
        assert "shared_bytes," in source[attr_pos:launch_pos]
        assert "shared_bytes," in source[launch_pos:]


def test_retired_leaf_hit_env_flag_stays_out_of_runtime_surface():
    root = Path(__file__).resolve().parents[2]
    retired = "GPUREC_" + "LEAF" + "_HIT_ONLY" + "_LOGP"
    runtime_paths = [
        root / "README.md",
        root / "gpurec" / "core" / "kernels" / "wave_backward.py",
    ]

    offenders = [
        str(path.relative_to(root))
        for path in runtime_paths
        if retired in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_rust_backtracking_does_not_declare_quick_xml_directly():
    root = Path(__file__).resolve().parents[2]
    crate = root / "crates" / "gpurec-backtrack"
    manifest = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads((crate / "Cargo.lock").read_text(encoding="utf-8"))

    assert "quick-xml" not in manifest.get("dependencies", {})
    assert "quick-xml" not in manifest.get("dev-dependencies", {})

    package_deps = {
        package["name"]: package.get("dependencies", [])
        for package in lockfile.get("package", [])
    }
    assert "quick-xml" not in package_deps["gpurec-backtrack"]


def test_rust_backtracking_work_items_use_concrete_clade_state():
    root = Path(__file__).resolve().parents[2]
    lib_rs = (
        root / "crates" / "gpurec-backtrack" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")

    assert "clade: Option<usize>" not in lib_rs
    assert "clade: Some(" not in lib_rs
    assert "if let Some(clade) = item.clade" not in lib_rs
    assert "clade: usize" in lib_rs


def test_second_order_docs_reference_current_public_loss_apis():
    root = Path(__file__).resolve().parents[2]
    note = (
        root / "docs" / "second-order-optimization-opportunities.md"
    ).read_text(encoding="utf-8")

    assert "_forward_inference" not in note
    assert "GeneReconModel._" not in note
    assert "model.nll()" not in note
    assert "GeneReconModel.full_loss_for_theta(theta)" in note
    assert "UniformChunkedReconModel.nll()" in note


def test_hogenom_alerax_rate_evaluator_documents_local_file_contract():
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "profiling" / "evaluate_hogenom_alerax_rates.py"
    ).read_text(encoding="utf-8")

    for token in (
        "Checkout-local HOGENOM AleRax rate validation helper",
        "not a general AleRax rate-file parser",
        "second line",
        "first three whitespace-separated values",
        "checkpoint/<family>.txt",
        "D/L/T rates in the first three columns of its second line",
    ):
        assert token in script


def test_tracked_notebooks_are_documented_as_checkout_local_artifacts():
    root = Path(__file__).resolve().parents[2]
    note = (root / "notebooks" / "README.md").read_text(encoding="utf-8")
    notebooks = [
        path.name
        for path in _tracked_files(root, "notebooks/*.ipynb")
    ]

    assert notebooks
    for token in (
        "checkout-local HOGENOM analysis artifacts",
        "not portable examples",
        "CUDA",
        "reconcile_hogenom_ccp_gpurec.ipynb",
        "optimize_hogenom_ccp_adam_oscillation.ipynb",
    ):
        assert token in note
    assert [name for name in notebooks if name not in note] == []


def test_scripts_readme_lists_tracked_scripts_in_ownership_matrix():
    root = Path(__file__).resolve().parents[2]
    note = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    script_names = [
        path.name
        for path in _tracked_files(root, "scripts/*")
        if path.suffix in {".py", ".R"}
    ]

    assert script_names
    for token in (
        "Ownership Matrix",
        "Candidate for deletion or migration",
        "checkout-local",
        "not as a portable benchmark",
        "check_release_metadata.py",
    ):
        assert token in note
    assert [name for name in script_names if name not in note] == []


def test_missing_inline_path_references_are_explicitly_historical_or_optional():
    root = Path(__file__).resolve().parents[2]
    path_ref_pattern = re.compile(
        r"`((?:docs|profiling|scripts|tests|gpurec)/"
        r"[A-Za-z0-9_.\-/]+"
        r"(?:\.md|\.py|\.ipynb|\.json|\.yaml|\.yml|\.txt|\.R)?)`"
    )
    allowed_context_tokens = (
        "historical",
        "missing",
        "not all tracked",
        "not reproducible",
        "not distributed",
        "optional",
        "skipped-when-missing",
        "superseded",
        "untracked",
        "no longer points",
        "no longer names",
    )
    offenders: list[str] = []

    for path in _tracked_files(root, "README.md", "docs/*.md"):
        text = path.read_text(encoding="utf-8")
        for match in path_ref_pattern.finditer(text):
            reference = match.group(1)
            if "..." in reference or (root / reference).exists():
                continue
            context_start = text.rfind("\n\n", 0, match.start()) + 2
            context_end = text.find("\n\n", match.end())
            if context_end == -1:
                context_end = len(text)
            context = text[context_start:context_end].lower()
            file_text = text.lower()
            if any(token in context or token in file_text for token in allowed_context_tokens):
                continue
            offenders.append(f"{path.relative_to(root)}: {reference}")

    assert offenders == []


def test_cpp_wave_stat_exports_validate_positive_max_wave_size():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "gpurec" / "core" / "cpp" / "preprocess.cpp"
    ).read_text(encoding="utf-8")
    exported_functions = [
        "compute_phased_waves",
        "compute_wave_stats",
        "compute_packet_wave_stats",
        "compute_phased_wave_stats",
        "compute_phased_cross_family_wave_stats",
        "compute_cross_family_wave_stats",
    ]

    assert "void require_positive_max_wave_size" in source
    for name in exported_functions:
        pattern = (
            rf"{name}\([^)]*max_wave_size\)\s*\{{\s*"
            r"require_positive_max_wave_size\("
            r'"max_wave_size", max_wave_size\);'
        )
        assert re.search(pattern, source, flags=re.S), name


def test_preprocess_cpp_declares_direct_standard_includes():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "gpurec" / "core" / "cpp" / "preprocess.cpp"
    ).read_text(encoding="utf-8")

    required_includes = {
        "std::chrono": "#include <chrono>",
        "std::set": "#include <set>",
    }
    for symbol, include in required_includes.items():
        if symbol in source:
            assert include in source, f"{symbol} requires direct {include}"


def test_preprocess_cpp_does_not_export_unowned_bench_parse():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "gpurec" / "core" / "cpp" / "preprocess.cpp"
    ).read_text(encoding="utf-8")

    assert "bench_parse" not in source
    assert "std::chrono" not in source
    assert "#include <chrono>" not in source


def test_pi_wave_backward_signature_omits_unused_ancestors_t():
    root = Path(__file__).resolve().parents[2]
    backward_path = root / "gpurec" / "core" / "backward.py"
    module = ast.parse(backward_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "Pi_wave_backward"
    )
    kwonly_names = {arg.arg for arg in function.args.kwonlyargs}
    assert "ancestors_T" not in kwonly_names
    assert "ancestors_T" not in (ast.get_docstring(function) or "")

    for path in (
        root / "gpurec" / "api" / "uniform_chunked.py",
        root / "gpurec" / "optimization" / "implicit_grad.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        pi_backward_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Pi_wave_backward"
        ]
        assert pi_backward_calls
        for call in pi_backward_calls:
            assert "ancestors_T" not in {keyword.arg for keyword in call.keywords}


def test_tests_package_docstring_matches_current_layout():
    root = Path(__file__).resolve().parents[2]
    doc = (root / "tests" / "__init__.py").read_text(encoding="utf-8")

    assert "gradients/" not in doc
    assert "performance/" not in doc
    assert "unit/" in doc
    assert "integration/" in doc
    assert "kernels/" in doc
    assert "tests.unit.alerax_helpers" in doc


def test_pytest_warning_filters_are_not_blanket_ignores():
    root = Path(__file__).resolve().parents[2]
    config = (root / "pytest.ini").read_text(encoding="utf-8")
    broad_ignores = {
        "ignore::DeprecationWarning",
        "ignore::PendingDeprecationWarning",
    }

    for broad_ignore in broad_ignores:
        assert broad_ignore not in config
