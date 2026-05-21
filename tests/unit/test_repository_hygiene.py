"""Repository-level source and documentation hygiene checks."""

import ast
import re
import subprocess
from pathlib import Path

import gpurec
import gpurec.workflow as workflow


def _tracked_files(root: Path, *patterns: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in result.stdout.splitlines() if line]


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


def test_project_readme_documents_sampling_output_layout():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")

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
