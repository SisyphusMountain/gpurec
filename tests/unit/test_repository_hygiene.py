"""Repository-level source and documentation hygiene checks."""

import subprocess
from pathlib import Path


def _tracked_files(root: Path, *patterns: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in result.stdout.splitlines() if line]


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


def test_project_readme_documents_sampling_output_layout():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")

    assert "output_gpurec/reconciliations/all/" in project_readme
    assert "event_counts.tsv" in project_readme
    assert "totalSpeciesEventCounts.txt" in project_readme
    assert "totalTransfers.txt" in project_readme
