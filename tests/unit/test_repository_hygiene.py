"""Repository-level source and documentation hygiene checks."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import gpurec
import gpurec.workflow as workflow
import gpurec.workflow.checkpoint as workflow_checkpoint

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

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
    return [
        path
        for line in result.stdout.splitlines()
        if line and (path := root / line).exists()
    ]


def _tracked_package_python_files(root: Path) -> list[Path]:
    return _tracked_files(root, "gpurec/*.py", "gpurec/**/*.py")


def _markdown_table_rows_by_first_cell(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or not cells[0].startswith("`") or not cells[0].endswith("`"):
            continue
        rows[cells[0].strip("`")] = cells[1:]
    return rows


_GPUREC_ENV_PATTERN = re.compile(r"\bGPUREC_[A-Z0-9_]+\b")
_ENV_OWNER_CATEGORIES = frozenset(
    {
        "User-facing",
        "User-facing compatibility",
        "User-facing discovery",
    }
)
_SUPPORTED_ENV_FLAGS = frozenset(
    {
        "GPUREC_BACKTRACK_BIN",
        "GPUREC_BACKTRACK_NATIVE_LIB",
        "GPUREC_ALERAX_COMPAT",
        "GPUREC_MEMORY_POLICY_FRACTION",
        "GPUREC_MEMORY_POLICY_RESERVE_GIB",
        "GPUREC_PREPROCESS_BIN",
        "GPUREC_PREPROCESS_NATIVE_LIB",
    }
)
_SUPPORTED_ENV_CATEGORIES = frozenset(
    {
        "User-facing",
        "User-facing compatibility",
        "User-facing discovery",
    }
)
_UNSUPPORTED_ENV_DOC_FLAGS = frozenset(
    {
        "GPUREC_PREPROCESS_BACKEND",
        "GPUREC_SCHEDULER_BACKEND",
        "GPUREC_FUSE_FINAL_PIBAR",
        "GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM",
        "GPUREC_WAVE_STEP_BLOCK_S",
        "GPUREC_WAVE_STEP_NUM_WARPS",
        "GPUREC_DTS_BLOCK_S",
        "GPUREC_DTS_NUM_WARPS",
        "GPUREC_DTS_GRAD_MT_TILE_SPLITS",
        "GPUREC_DTS_PARENT_BLOCK_S",
        "GPUREC_DTS_PARENT_NUM_WARPS",
        "GPUREC_DTS_PARENT_TILE_SPLITS",
        "GPUREC_PIBAR_UD_BLOCK_S",
        "GPUREC_PIBAR_UD_NUM_WARPS",
        "GPUREC_SELF_LOOP_2D_BLOCK_W",
        "GPUREC_SELF_LOOP_2D_BLOCK_NODES",
        "GPUREC_SELF_LOOP_2D_NUM_WARPS",
        "GPUREC_SELF_LOOP_2D_JT_NUM_WARPS",
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT",
        "GPUREC_CUDA_SELF_LOOP_SPLIT",
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION",
        "GPUREC_CUDA_PIBAR_FROM_UD",
        "GPUREC_CUDA_PIBAR_FROM_UD_STRICT",
        "GPUREC_BACKWARD_NO_CPU_PRUNING",
        "GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO",
        "GPUREC_TRITON_CHILD_EDGE_SELF_LOOP",
        "GPUREC_TRITON_SELF_LOOP_DIRECT_GRADS",
    }
)


def _package_env_flags(root: Path) -> list[str]:
    return sorted(
        {
            match.group(0)
            for path in _tracked_package_python_files(root)
            for match in _GPUREC_ENV_PATTERN.finditer(
                path.read_text(encoding="utf-8")
            )
        }
    )


def _runtime_read_env_flags(root: Path) -> list[str]:
    read_flags: set[str] = set()
    env_read_helpers: set[str] = set()

    for path in _tracked_package_python_files(root):
        module = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _GPUREC_ENV_PATTERN.fullmatch(node.value.value)
        }
        for function in (
            node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)
        ):
            defaults = function.args.defaults
            args_with_defaults = function.args.args[-len(defaults) :] if defaults else []
            for arg, default in zip(args_with_defaults, defaults, strict=True):
                if (
                    isinstance(default, ast.Constant)
                    and isinstance(default.value, str)
                    and _GPUREC_ENV_PATTERN.fullmatch(default.value)
                ):
                    constants[arg.arg] = default.value
            for arg, default in zip(
                function.args.kwonlyargs, function.args.kw_defaults, strict=True
            ):
                if (
                    isinstance(default, ast.Constant)
                    and isinstance(default.value, str)
                    and _GPUREC_ENV_PATTERN.fullmatch(default.value)
                ):
                    constants[arg.arg] = default.value

        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            is_os_environ_get = (
                isinstance(func, ast.Attribute)
                and func.attr in {"get", "setdefault"}
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            )
            is_os_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            )
            is_env_read_helper = (
                isinstance(func, ast.Name) and func.id in env_read_helpers
            )
            if not (is_os_environ_get or is_os_getenv or is_env_read_helper):
                continue

            key_arg = node.args[0]
            if (
                isinstance(key_arg, ast.Constant)
                and isinstance(key_arg.value, str)
                and _GPUREC_ENV_PATTERN.fullmatch(key_arg.value)
            ):
                read_flags.add(key_arg.value)
            elif isinstance(key_arg, ast.Name) and key_arg.id in constants:
                read_flags.add(constants[key_arg.id])

    return sorted(read_flags)


def _runtime_environment_owner_manifest(root: Path) -> dict[str, str]:
    pruning_plan = (
        root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
    ).read_text(encoding="utf-8")
    manifest_match = re.search(
        r"### Environment Owner Manifest\n\n(?P<table>\| Variable\(s\).*?)\n\n"
        r"Supported environment flags are limited to",
        pruning_plan,
        flags=re.S,
    )
    assert manifest_match is not None

    owners: dict[str, str] = {}
    for line in manifest_match.group("table").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"Variable(s)", "---"}:
            continue
        flags_cell, owner, _notes = cells
        assert owner in _ENV_OWNER_CATEGORIES
        for flag in _GPUREC_ENV_PATTERN.findall(flags_cell):
            assert flag not in owners
            owners[flag] = owner

    return owners


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


def test_package_docs_do_not_advertise_core_as_public_surface():
    root = Path(__file__).resolve().parents[2]
    package_module = ast.parse(
        (root / "gpurec" / "__init__.py").read_text(encoding="utf-8")
    )
    core_package_module = ast.parse(
        (root / "gpurec" / "core" / "__init__.py").read_text(encoding="utf-8")
    )
    package_doc = " ".join((ast.get_docstring(package_module) or "").split())
    core_package_doc = " ".join(
        (ast.get_docstring(core_package_module) or "").split()
    )
    docs_readme = " ".join(
        (root / "docs" / "README.md").read_text(encoding="utf-8").split()
    )
    public_export_modules = sorted(set(gpurec._LAZY_EXPORTS.values()))

    for token in (
        "High-level public API",
        "from gpurec import GeneReconModel",
        "from gpurec import RunConfig, optimize, sample",
        "``gpurec.core`` namespace is an implementation namespace",
        "unstable unless a helper is explicitly documented as supported",
    ):
        assert token in package_doc
    for module_name in public_export_modules:
        assert f"``{module_name}``" in package_doc

    for token in (
        "Low-Level API Stability",
        "The supported package entry points are the high-level classes, workflow helpers, backtracking helpers, and entropy helpers",
        "`gpurec.core` is an implementation namespace",
        "Direct imports from `gpurec.core` are unstable",
        "tests may use internals to guard behavior",
        "not public API evidence",
    ):
        assert token in docs_readme
    for module_name in public_export_modules:
        assert f"`{module_name}`" in docs_readme

    for token in (
        "Internal implementation namespace",
        "Public user-facing imports are exported from :mod:`gpurec`",
        "``gpurec.core`` are unstable",
        "explicitly documented as a supported low-level exception",
    ):
        assert token in core_package_doc
    for module_name in public_export_modules:
        assert f":mod:`{module_name}`" in core_package_doc

    for token in (
        "Lower-level access",
        "from gpurec.core.model",
        "from gpurec.core.likelihood",
        "from gpurec.core.forward",
        "GeneDataset",
        "E_fixed_point",
        "Pi_wave_forward",
    ):
        assert token not in package_doc


def test_internal_api_helper_modules_document_support_boundary():
    root = Path(__file__).resolve().parents[2]
    family_layout_module = ast.parse(
        (root / "gpurec" / "api" / "_family_layout.py").read_text(
            encoding="utf-8"
        )
    )
    validation_module = ast.parse(
        (root / "gpurec" / "api" / "_validation.py").read_text(
            encoding="utf-8"
        )
    )
    docs_readme = " ".join(
        (root / "docs" / "README.md").read_text(encoding="utf-8").split()
    )
    family_layout_doc = " ".join(
        (ast.get_docstring(family_layout_module) or "").split()
    )
    validation_doc = " ".join(
        (ast.get_docstring(validation_module) or "").split()
    )
    family_layout_docstrings = {
        node.name: " ".join((ast.get_docstring(node) or "").split())
        for node in family_layout_module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    validation_docstrings = {
        node.name: " ".join((ast.get_docstring(node) or "").split())
        for node in validation_module.body
        if isinstance(node, ast.FunctionDef)
    }

    for token in (
        "Internal family-layout support",
        "not a public import surface",
        "GeneReconModel",
        "UniformChunkedReconModel",
        "one runtime layout contract",
    ):
        assert token in family_layout_doc
    for name, token in (
        ("FamilyWaveInputs", "Validated family preprocessing slices"),
        ("FamilyWaveLayout", "Built family layout tensors"),
        (
            "origination_probs_for_family_indices",
            "family-specific origination rows",
        ),
        ("family_wave_inputs", "validated family preprocessing dictionaries"),
        ("schedule_family_waves", "already-collected family input metadata"),
        ("build_family_wave_layout", "model internals"),
    ):
        assert token in family_layout_docstrings[name]

    for token in (
        "Internal validation helpers",
        "support code for ``gpurec.api`` and ``gpurec.workflow``",
        "not standalone public API",
    ):
        assert token in validation_doc
    assert (
        "active sharing mode"
        in validation_docstrings["validate_theta_shape"]
    )
    assert "Direct imports from `gpurec.core` are unstable" in docs_readme


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
        root / "scripts" / "visualize_hogenom_loss_landscape.py",
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


def test_package_runtime_does_not_use_assert_statements():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in _tracked_package_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
            elif isinstance(node, ast.Raise):
                exc = node.exc
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "AssertionError"
                ) or (isinstance(exc, ast.Name) and exc.id == "AssertionError"):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert offenders == []


def test_disabled_adaptive_neumann_terms_message_is_professional():
    from gpurec._validation import ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE

    assert "not part of the supported production optimization route" in (
        ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE
    )
    normalized = ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE.lower()
    assert "terrible" not in normalized
    assert "must be fixed" not in normalized


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
        "`--samples 2 --output-dir <dir>`",
        "`sample_0.xml` and `sample_1.xml`",
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


def test_rust_backtracking_source_documents_payload_schema_and_validation():
    root = Path(__file__).resolve().parents[2]
    source = (root / "crates" / "gpurec-backtrack" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )

    for token in (
        "The JSON schema mirrors the Python exporter",
        "base-2 log values",
        "`-1e300` used as the practical negative-infinity",
        "Species indices use the postorder order",
        "Row-major matrix used by the JSON backtracking schema",
        "data[row * cols + col]",
        "rows*cols overflows usize",
        "parent`, `left`, and `right` are clade indices",
        "`log_prob` is the base-2 log conditional split probability",
        "`origination_probs`, when present, are ordinary",
        "nonnegative weights over species",
        "zero weights are treated as impossible",
        "{name} has length {got}, expected {expected}",
        "root_clade {} is out of bounds",
        "leaf_species[{idx}] is out of bounds",
        "origination_probs contains negative value",
        "max_events must be positive",
        "split {idx} has clade outside",
        "split {idx} log_prob is non-finite",
    ):
        assert token in source


def test_hogenom_scripts_are_marked_as_legacy_experiment_surface():
    root = Path(__file__).resolve().parents[2]
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    project_readme = (root / "README.md").read_text(encoding="utf-8")

    assert "gpurec optimize" in scripts_readme
    assert "--device cuda" in scripts_readme
    assert "optimized workflow currently requires CUDA" in scripts_readme
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


def test_scripts_readme_ownership_matrix_covers_tracked_script_surface():
    root = Path(__file__).resolve().parents[2]
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    rows = _markdown_table_rows_by_first_cell(scripts_readme)
    tracked_scripts = {
        path.name
        for path in _tracked_files(root, "scripts/*.py", "scripts/*.R")
    }
    script_surface = tracked_scripts | {
        name
        for name in rows
        if (root / "scripts" / name).exists()
    }
    allowed_statuses = {
        "Checkout-local AleRax comparison helper.",
        "Checkout-local HOGENOM counts-init route benchmark.",
        "Checkout-local HOGENOM landscape visualizer.",
        "Checkout-local HOGENOM optimizer benchmark.",
        "Checkout-local HOGENOM route benchmark.",
        "Checkout-local HOGENOM route replay.",
        "Compatibility wrapper.",
        "Fixed-dataset global-uniform reproducer.",
        "Fixed-dataset specieswise-uniform reproducer.",
        "HOGENOM checkpoint rate exporter.",
        "Hydra adapter for the legacy W&B optimizer.",
        "Internal checkout-local profiler.",
        "Legacy fast HOGENOM launcher.",
        "Legacy full HOGENOM W&B optimizer.",
        "One-off branch-scale penalty/KKT analysis.",
        "One-off LaTeX report builder.",
        "Optional plotting helper.",
        "Release metadata gate.",
        "Shared helper for legacy uniform launchers.",
    }
    deprecation_or_migration_candidates = {
        "fast_optimize_hogenom_ccp.py",
        "hogenom_ccp_wandb_opt.py",
        "hogenom_opt_helpers.py",
        "make_hogenom_branchscale_penalty_report.py",
        "optimize_hogenom_ccp_global_uniform.py",
        "optimize_hogenom_ccp_hydra.py",
        "optimize_hogenom_ccp_specieswise_uniform.py",
        "optimize_hogenom_penalty316_kkt.py",
        "profile_hogenom_ccp_pass.py",
    }

    assert set(rows) == script_surface
    for name, (status, ownership) in rows.items():
        assert status in allowed_statuses, name
        assert ownership, name
        normalized_ownership = ownership.lower()
        assert any(
            token in normalized_ownership
            for token in (
                "supported cli",
                "supported workflow",
                "`gpurec.workflow`",
                "release hygiene",
                "hogenom",
                "alerax",
                "plotting",
                "fixed-dataset",
                "historical",
                "legacy",
            )
        ), name

    for name in deprecation_or_migration_candidates:
        ownership = rows[name][1]
        normalized_ownership = ownership.lower()
        assert any(
            token in normalized_ownership
            for token in (
                "candidate for deletion or migration",
                "archive/delete or migrate",
                "delete or migrate",
                "keep only as",
                "keep only until",
                "keep only while",
                "migrate once",
                "migrate reusable behavior",
                "until its",
            )
        ), name


def test_tests_readme_documents_white_box_and_legacy_script_test_ownership():
    root = Path(__file__).resolve().parents[2]
    tests_readme = " ".join(
        (root / "tests" / "README.md").read_text(encoding="utf-8").split()
    )

    for token in (
        "Test Surface Ownership",
        "`tests/unit/test_legacy_scripts.py` owns executable guards for",
        "checkout-local script/profiling helpers",
        "cleanup on success and failure",
        "parser-level validation",
        "safe checkpoint loading",
        "White-box tests for internal helpers are allowed only when they guard a documented deletion-prone contract",
        "replacement behavior test",
        "deliberate deletion note",
    ):
        assert token in tests_readme


def test_docs_map_distinguishes_cuda_smoke_from_checkout_local_config():
    root = Path(__file__).resolve().parents[2]
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    configs_readme = (root / "configs" / "README.md").read_text(encoding="utf-8")
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized_docs_readme = " ".join(docs_readme.split())
    normalized_configs_readme = " ".join(configs_readme.split())

    for token in (
        "../examples/README.md",
        "genewise `hessian-sgd`",
        "specieswise `adagrad-restarts`",
        "source-checkout/source-archive CUDA",
        "config/parser fixture",
        "not a CPU fallback",
        "not an end-to-end optimizer smoke",
        "S > 256",
        "../configs/hogenom_ccp_wandb.yaml",
        "checkout-local HOGENOM Hydra/W&B",
        "not a portable example",
        "../configs/README.md",
        "config ownership note",
        "installed flat JSON workflow configs",
        "checkout-local Hydra/HOGENOM experiment inputs",
    ):
        assert token in normalized_docs_readme

    for token in (
        "Config Ownership",
        "not installed package templates unless explicitly documented as such",
        "`hogenom_ccp_wandb.yaml`",
        "Hydra YAML consumed by `python scripts/optimize_hogenom_ccp_hydra.py`",
        "not by `gpurec optimize --config`",
        "Checkout-local only",
        "untracked `tests/data/HOGENOM/...` inputs",
        "online W&B defaults",
        "`examples/README.md`",
        "`examples/minimal-run-config.json`",
        "`examples/specieswise-adagrad-restarts-config.json`",
        "genewise `hessian-sgd` auto defaults",
        "specieswise `adagrad-restarts` auto defaults",
        "not CPU fallbacks",
        "end-to-end optimizer smokes",
        "New tracked configs should state",
        "which command consumes the file",
        "parser fixture",
        "historical experiment input",
    ):
        assert token in normalized_configs_readme
    assert "see `configs/README.md` for config ownership" in project_readme


def test_run_config_reference_covers_current_config_surface():
    from dataclasses import fields

    from gpurec.workflow.config import RunConfig

    root = Path(__file__).resolve().parents[2]
    reference = (root / "docs" / "run-config-reference.md").read_text(
        encoding="utf-8"
    )
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    rows = _markdown_table_rows_by_first_cell(reference)
    config_fields = {field.name for field in fields(RunConfig)}

    assert set(rows) == config_fields
    for name in config_fields:
        assert f"`{name}`" in reference
        assert f"--{name.replace('_', '-')}" in rows[name][0]

    for token in (
        "flat JSON and Python dataclass contract",
        "`RunConfig.from_dict(...)` before model construction",
        "resolves to `hessian-sgd`",
        "resolves to `adagrad-restarts`",
        "Disabled compatibility flag",
        "not part of the supported production optimization route",
        "Non-default values require specieswise `adagrad-restarts`",
        "Requires genewise `hessian-sgd`",
        "Resume starts at checkpoint `next_step`",
        "`not_converged`/`max_steps`",
        "`--require-final-check-ok`",
    ):
        assert token in reference
    assert "docs/run-config-reference.md" in project_readme
    assert "run-config-reference.md" in docs_readme
    normalized_reference = reference.lower()
    assert "terrible" not in normalized_reference
    assert "must be fixed" not in normalized_reference


def test_simplification_opportunity_index_is_mapped_and_gate_oriented():
    root = Path(__file__).resolve().parents[2]
    docs_readme = " ".join(
        (root / "docs" / "README.md").read_text(encoding="utf-8").split()
    )
    refactor_plan = " ".join(
        (
            root / "docs" / "refactor-simplification-plan-2026-05-21.md"
        ).read_text(encoding="utf-8").split()
    )
    index = " ".join(
        (
            root / "docs" / "simplification-opportunity-index-2026-05-21.md"
        ).read_text(encoding="utf-8").split()
    )

    for token in (
        "`simplification-opportunity-index-2026-05-21.md`",
        "direct inventory of removable or mergeable alternative paths",
        "source-file evidence",
        "retained behavior",
        "deletion gates",
    ):
        assert token in docs_readme

    for token in (
        "what specific paths can be simplified or removed",
        "`simplification-opportunity-index-2026-05-21.md`",
        "implementation plan behind that index",
    ):
        assert token in refactor_plan

    for token in (
        "which documents and code paths show concrete opportunities to simplify",
        "Current alternatives",
        "Simplification",
        "Keep",
        "Gate",
        "EVAL-01 - Collapse The Four Evaluation Pipelines",
        "MODE-01 - Replace Shape-Driven Mode Inference With `ParameterLayout`",
        "LIK-02 - Remove Or Deprecate Misleading Log-Likelihood Aliases",
        "CPP-02 - Remove Or Rehome C++ Diagnostic Exports",
        "SCRIPT-01 - Delete Or Reclassify Fixed-Dataset Scripts",
        "TEST-01 - Stop Preserving Dead Internals Through Tests",
        "Search confirms diagnostics are not part of supported user workflows",
    ):
        assert token in index


def test_simplification_execution_log_mentions_every_index_proposal():
    root = Path(__file__).resolve().parents[2]
    index = (
        root / "docs" / "simplification-opportunity-index-2026-05-21.md"
    ).read_text(encoding="utf-8")
    execution_log = (
        root / "docs" / "simplification-execution-log-2026-05-21.md"
    ).read_text(encoding="utf-8")

    proposal_ids = re.findall(r"^### ([A-Z]+-\d+) - ", index, flags=re.M)
    assert proposal_ids

    missing = [
        proposal_id
        for proposal_id in proposal_ids
        if proposal_id not in execution_log
    ]

    assert missing == []


def test_release_readiness_gpu_smoke_matches_small_species_limitation():
    root = Path(__file__).resolve().parents[2]
    release_readiness = (root / "docs" / "release-readiness.md").read_text(
        encoding="utf-8"
    )

    for token in (
        "small family subset",
        "S > 256",
        "tiny source",
        "config/parser fixture",
        "not an end-to-end optimizer smoke",
        "HOGENOM or 1000-tree benchmark checks",
    ):
        assert token in release_readiness
    assert "small species tree" not in release_readiness


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


def test_project_readme_documents_preprocess_cpu_core_guidance():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    assert "--preprocess-cpu-cores 8" in normalized
    assert "worker thread count for CPU preprocessing" in normalized
    assert "Rust preprocessing uses its runtime default" in normalized


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
        "`small_family_max_leaves`",
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
        "Backward CUDA/Triton prototype selectors have been removed",
        "retained backward path is Triton-only",
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


def test_active_mask_bfloat16_boundary_is_documented_as_private_helper():
    root = Path(__file__).resolve().parents[2]
    backward_source = (root / "gpurec" / "core" / "backward.py").read_text(
        encoding="utf-8"
    )
    wave_backward_source = (
        root / "gpurec" / "core" / "kernels" / "wave_backward.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(wave_backward_source)
    supported_dtype_assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_SUPPORTED_FLOAT_DTYPES"
            for target in node.targets
        )
    )
    module_docstring = " ".join((ast.get_docstring(module) or "").split())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "active_mask_from_rhs_absmax_fused"
    )
    function_docstring = " ".join((ast.get_docstring(function) or "").split())

    for token in (
        "accepts bf16 inputs for standalone row-mask experiments",
        "Pi_wave_backward",
        "rejects bf16 before this helper is reached",
    ):
        assert token in module_docstring

    for token in (
        "private retained-kernel helper",
        "not a public dtype policy",
        "fp32/fp64/bf16 CUDA tensors",
        "standalone mask experiments",
        "supports only fp32/fp64",
        "rejects bf16 before calling this helper",
    ):
        assert token in function_docstring

    assignment_source = ast.get_source_segment(
        wave_backward_source,
        supported_dtype_assignment,
    )
    assert assignment_source is not None
    assert "torch.bfloat16" in assignment_source
    assert "_SUPPORTED_BACKWARD_FLOAT_DTYPES = (torch.float32, torch.float64)" in backward_source
    assert "dtype not in _SUPPORTED_BACKWARD_FLOAT_DTYPES" in backward_source


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


def test_uniform_chunked_nll_per_family_documents_diagnostic_contract():
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
    class_doc = " ".join((ast.get_docstring(uniform_class) or "").split())
    nll_per_family = next(
        node
        for node in uniform_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "nll_per_family"
    )
    method_doc = " ".join((ast.get_docstring(nll_per_family) or "").split())

    for token in (
        "`UniformChunkedReconModel.nll_per_family(chunk_indices=...)`",
        "no-grad global/uniform diagnostic",
        "one shared-theta NLL per selected family",
        "independent per-family gradients",
    ):
        assert token in normalized
    for token in (
        "no-grad global/uniform diagnostic",
        "one shared-theta NLL per selected family",
        "chunk filtering",
        "does not define independent per-family gradients",
    ):
        assert token in class_doc
    for token in (
        "no-grad per-family NLL diagnostics",
        "selected chunk order",
        "global/uniform shared-theta diagnostic",
        "not an independent per-family gradient surface",
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
        "origination is a file-level event",
        "the first speciation reached by the shared traversal is counted as origination",
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
    assert "Failed optimization still prints the optimization status line" in project_readme
    assert "unless the run reached `status=converged`" in normalized
    assert "resolved `mode`, `optimizer`, family/species/batch counts" in normalized
    assert "batch packing, family chunk size, clade budget" in normalized
    assert "solver iteration budgets" in normalized
    assert (
        "`configured_steps`, `optimizer_step_cap`, `optimizer_step_cap_reason`"
        in project_readme
    )
    assert (
        "`final_check_iters`, `steps_completed`, `elapsed_s`, `best_step`"
        in project_readme
    )
    assert "objective, gradient route, rate parameterization" in normalized
    assert "optimizer-specific route fields" in normalized
    assert "Add `--require-converged` to `gpurec optimize`" in project_readme
    assert "`--require-final-check-ok`" in project_readme
    assert "`final_check_status=ok`" in project_readme
    assert "`final_grad_inf`" in project_readme
    assert "`final_projected_grad_inf`" in project_readme
    assert "`final_solver_e_adjoint_failed_batches`" in project_readme
    assert "`final_solver_e_adjoint_rel_res_max`" in project_readme
    assert "`sampled_families`, `samples`, `xml`, and `sample_out_dir`" in project_readme
    assert "gpurec summary-info --summary output_gpurec/summary.json" in project_readme
    assert "`--require-converged`" in project_readme
    assert "`last_final_check_status`" in project_readme
    assert "Use `gpurec sample --checkpoint ...` to sample an existing run" in normalized


def test_project_readme_documents_workflow_optimizer_modes():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "| `auto` |",
        "| `adam` |",
        "| `adagrad` |",
        "| `adagrad-restarts` |",
        "| `projected-sgd` |",
        "| `lbfgs` |",
        "| `adam-lbfgs` |",
        "| `projected-lbfgs` |",
        "| `lbfgsb` |",
        "| `batched-lbfgs` |",
        "| `adam-fd-newton` |",
        "If omitted, `auto` resolves to `hessian-sgd` for `mode=genewise`",
        "`adagrad-restarts` for `mode=specieswise`",
        "Workflow rate bounds default to `min_rate=2^-30` and `max_rate=2`",
        "`lbfgs_line_search` is `none` or `strong_wolfe`",
        "LBFGS runtime errors stop the run with a failed status",
        "`adam_warmup_steps` controls the phase switch",
        "Armijo line-search probes use loss-only evaluations",
        "`fd_adam_warmup_steps` controls per-batch Adam warmup",
        "`fd_hessian_refresh_steps` controls",
        "`fd_newton_damping` controls Hessian regularization",
        "there is no separate log-rate movement cap",
        "incompatible resumed optimizer state is discarded",
        "Requires `mode=genewise`",
        "projected gradients at rate bounds",
        "vectorized row-wise port of PyTorch's bracket/zoom line search",
        "`adagrad_restart_schedule` is validated before model loading",
        "`E/Pi[/Neumann]:lr:steps`",
        "Later phases must not decrease `fixed_iters_E`, `fixed_iters_Pi`, or `neumann_terms`",
        "`adagrad_restart_final_check_iters` must be zero to disable",
    ):
        assert token in normalized
    assert "fd_newton_max_step" not in normalized
    assert "--fd-newton-max-step" not in (
        root / "gpurec" / "cli.py"
    ).read_text(encoding="utf-8")


def test_production_optimization_guide_is_linked_and_documents_routes():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "production-optimization-guide.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(guide.split())

    assert "docs/production-optimization-guide.md" in project_readme
    assert "production-optimization-guide.md" in docs_readme
    for token in (
        "negative log-likelihood in bits",
        "`theta` stores base-2 log rates",
        "`objective=negative_log_likelihood_bits`",
        "`gradient_route=implicit_first_order_adjoint`",
        "`rate_parameterization=base2_log_dlt_rates`",
        "`production_default_basis=hogenom_and_test_trees_1000`",
        "`hessian-sgd`",
        "`adagrad-restarts`",
        "8:1.0:60,16:0.5:35,32:0.5:30",
        "`adagrad_restart_total_steps=125`",
        "`optimizer_step_cap=125`",
        "`optimizer_step_cap_reason=adagrad_restart_schedule`",
        "`adagrad_restart_final_check_iters=128`",
        "same solver budget with a new LR",
        "must not reduce `fixed_iters_E`, `fixed_iters_Pi`, or `neumann_terms`",
        "`final_check_iters`",
        "optimizer-specific route fields",
        "Hessian-SGD warmup/refresh/normal-stage solver controls",
        "HOGENOM",
        "`tests/data/test_trees_1000`",
        "likelihood/gradient parity",
        "`history.jsonl`",
        "elapsed seconds",
        "`per_fam_likelihoods.tsv`",
        "`--check-preprocess`",
    ):
        assert token in normalized


def test_second_order_notes_do_not_contradict_production_optimizer_defaults():
    root = Path(__file__).resolve().parents[2]
    guide = (root / "docs" / "production-optimization-guide.md").read_text(
        encoding="utf-8"
    )
    notes = (root / "docs" / "second-order-optimization-opportunities.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(notes.split())

    assert "second-order-optimization-opportunities.md" in (
        root / "docs" / "README.md"
    ).read_text(encoding="utf-8")
    assert "`hessian-sgd` for genewise batches" in normalized
    assert "`adagrad-restarts` for specieswise runs" in normalized
    assert "explicit `optimizer=batched-lbfgs`" in normalized
    assert "genewise runs to `hessian-sgd`" in normalized
    assert "specieswise runs to `adagrad-restarts`" in normalized
    assert "to the batched optimizer" not in normalized
    assert "`hessian-sgd`" in guide


def test_output_artifact_reference_is_linked_and_documents_contract():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "production-optimization-guide.md").read_text(
        encoding="utf-8"
    )
    reference = (root / "docs" / "output-artifacts.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(reference.split())

    assert "docs/output-artifacts.md" in project_readme
    assert "output-artifacts.md" in docs_readme
    assert "`docs/output-artifacts.md`" in guide
    for token in (
        "staged publish step",
        "`history.jsonl`",
        "`optimization_history.csv`",
        "`summary.json`",
        "effective optimizer/batch/solver route",
        "objective/gradient/parameterization metadata",
        "`objective=negative_log_likelihood_bits`",
        "`gradient_route=implicit_first_order_adjoint`",
        "`rate_parameterization=base2_log_dlt_rates`",
        "`production_default_basis=hogenom_and_test_trees_1000`",
        "resolved `mode` and `optimizer`",
        "`families`",
        "`species`",
        "`batches`",
        "`batch_packing`",
        "`family_chunk_size`",
        "`clade_budget`",
        "`fixed_iters_e`",
        "`fixed_iters_pi`",
        "`neumann_terms`",
        "`solver/pi_adjoint_residual_absmax_max`",
        "`solver/pi_adjoint_residual_relmax_max`",
        "`solver/pi_adjoint_residual_checked_batches`",
        "`final_log_likelihood_bits`",
        "`final_grad_inf`",
        "`final_projected_grad_inf`",
        "`best_log_likelihood_bits`",
        "`steps_completed`",
        "`sampling_checkpoint`",
        "`elapsed_s`",
        "`best_step`",
        "`final_check_iters`",
        "`solver_warmup_iters`",
        "`fd_adam_warmup_steps`",
        "`fd_hessian_refresh_steps`",
        "`hessian_sgd_normal_fixed_iters_pi`",
        "`hessian_sgd_normal_neumann_terms`",
        "`hessian_sgd_pi_adjoint_warmstart`",
        "`pi_fixed_point_relaxation`",
        "`hessian_sgd_validation_interval`",
        "`hessian_sgd_validation_fixed_iters_pi`",
        "`hessian_sgd_validation_neumann_terms`",
        "`adagrad_restart_schedule`",
        "`adagrad_restart_total_steps`",
        "`adagrad_restart_final_check_iters`",
        "`configured_steps`",
        "`optimizer_step_cap`",
        "`optimizer_step_cap_reason`",
        "`final_check_status`",
        "`final_check_source`",
        "`final_check_reason`",
        "`final_check_fallback_clade_budget`",
        "`final_check_loss_abs_delta_bits`",
        "`final_check_grad_max_abs_delta`",
        "`final_check_grad_rel_inf_delta`",
        "`final_solver_e_adjoint_failed_batches`",
        "`final_solver_e_adjoint_success_batches`",
        "`final_solver_e_adjoint_rel_res_max`",
        "`gpurec optimize` status line",
        "JSON strings with spaces escaped as `\\u0020`",
        "`sampled_families`",
        "`samples`",
        "`xml`",
        "`sample_out_dir`",
        "exits without sampling fields",
        "anything other than `converged`",
        "`gpurec run`",
        "`rates_final.tsv`",
        "`per_fam_likelihoods.tsv`",
        "`theta_final.pt`",
        "`checkpoints/latest.pt`",
        "`checkpoints/best.pt`",
        "`route_metadata`",
        "`adagrad_restart_total_steps`",
        "`optimizer_step_cap`",
        "gpurec summary-info --summary output_gpurec/summary.json",
        "`--require-converged`",
        "`--require-final-check-ok`",
        "`final_check_status=ok`",
        "gpurec optimize` when the command should print",
        "gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt",
        "`last_final_check_status`",
        "`last_solver_e_adjoint_failed_batches`",
        "`last_solver_e_adjoint_success_batches`",
        "`last_solver_e_adjoint_rel_res_max`",
        "`optimizer/final_check_status=ok`",
        "`likelihood/data_nll_bits`",
        "`grad/projected_inf`",
        "`optimizer/adagrad_restart_*`",
        "Strict JSON",
        "family ordering",
        "`reconciliations/event_counts.tsv`",
        "`reconciliations/totalSpeciesEventCounts.txt`",
        "`reconciliations/all/*_sample_*.xml`",
    ):
        assert token in normalized


def test_troubleshooting_guide_documents_operator_failure_triage():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "troubleshooting.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(guide.split())

    assert "docs/troubleshooting.md" in project_readme
    assert "troubleshooting.md" in docs_readme
    for token in (
        "gpurec validate-config --config run.json",
        "`--check-preprocess`",
        "`unknown RunConfig field`",
        "`missing gene-tree path`",
        "`cuda_backward_ready=false`",
        "`--require-cuda-backward-ready`",
        "`summary.json`",
        "gpurec summary-info --summary output/summary.json",
        "`--require-converged`",
        "`--require-final-check-ok`",
        "gpurec optimize --require-converged",
        "gpurec run --require-converged",
        "`final_check_status=ok`",
        "`optimizer/final_check_status=ok`",
        "final-check likelihood/gradient deltas",
        "gpurec checkpoint-info --checkpoint output/checkpoints/latest.pt",
        "`history.jsonl`",
        "`grad/projected_inf`",
        "`nonfinite_objective_or_gradient`",
        "`adagrad_restart_schedule_complete`",
        (
            "gpurec optimize --config run.json --resume-from "
            "output/checkpoints/latest.pt"
        ),
        "`clade_budget`",
        "`family_chunk_size`",
        "`GPUREC_BACKTRACK_BIN`",
        "`theta_final.pt`",
    ):
        assert token in normalized


def test_input_preparation_guide_documents_alerax_data_contract():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "input-preparation.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(guide.split())

    assert "docs/input-preparation.md" in project_readme
    assert "input-preparation.md" in docs_readme
    for token in (
        "`[FAMILIES]`",
        "`starting_gene_tree`",
        "`gene_tree`",
        "`mapping`",
        "Species:gene1;gene2",
        "relative to the family file",
        "relative to the JSON config file",
        "duplicate gene assignments",
        "gpurec validate-config --config run.json --check-preprocess",
        "`cuda_backward_ready`",
        "`--require-cuda-backward-ready`",
        "`optimizer=auto`",
        "`hessian-sgd`",
        "`adagrad-restarts`",
        "`S > 256`",
        "GeneDataset(..., leaf_species_maps=...)",
    ):
        assert token in normalized


def test_optimization_workflow_call_graph_documents_current_cli_and_optimizers():
    root = Path(__file__).resolve().parents[2]
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    call_graph = (
        root / "docs" / "optimization-workflow-call-graph.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(call_graph.split())

    assert "optimization-workflow-call-graph.md" in docs_readme
    for token in (
        "`gpurec validate-config`",
        "stops at the preflight node",
        "`--check-preprocess`",
        "CPU `GeneDataset` preprocessing",
        "passes the retained CUDA backward `S > 256` gate",
        "`--require-cuda-backward-ready`",
        "without constructing `GeneReconModel`",
        "without constructing `GeneReconModel`, loading the Rust preprocessing extension, or touching CUDA",
        "loads the retained Rust parser",
        "`optimizer=auto` resolves to `hessian-sgd`",
        "`adagrad-restarts` for `mode=specieswise`",
        "`adagrad_restart_total_steps`",
        "`optimizer_step_cap`",
        "`optimizer_step_cap_reason`",
        "`adagrad_restart_final_check_iters`",
        "records `optimizer/adagrad_restart_*` fields",
        "`hessian-sgd` is genewise-only",
        "finite-difference Hessians",
        "projected gradients",
        "loss-only probes",
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
        "Opt-in Pi-adjoint warmstart runs also report",
        "`solver/pi_adjoint_residual_absmax_max`",
        "`solver/pi_adjoint_residual_relmax_max`",
        "one extra fixed-point self-loop application",
        "`pi_fixed_point_relaxation`",
    ):
        assert token in normalized


def test_implicit_gradient_documents_bicgstab_failure_policy():
    root = Path(__file__).resolve().parents[2]
    bridge_name = "implicit_grad_" + "loglik_vjp_wave"
    module = ast.parse(
        (root / "gpurec" / "optimization" / "implicit_grad.py").read_text(
            encoding="utf-8"
        )
    )
    optimization_init = ast.parse(
        (root / "gpurec" / "optimization" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    all_assignment = next(
        node
        for node in optimization_init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    exported_names = ast.literal_eval(all_assignment.value)

    bicgstab_doc = " ".join(
        (ast.get_docstring(functions["_bicgstab"]) or "").split()
    )
    bridge_doc = " ".join(
        (ast.get_docstring(functions[bridge_name]) or "").split()
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
    for token in (
        "Internal API bridge",
        "called by ``gpurec.api.model`` and ``gpurec.api.autograd``",
        "not exported from ``gpurec.optimization.__all__``",
        "external callers should use ``GeneReconModel`` or ``UniformChunkedReconModel``",
    ):
        assert token in bridge_doc
    assert bridge_name not in exported_names

    allowed_paths = {
        "gpurec/api/autograd.py",
        "gpurec/api/model.py",
        "gpurec/optimization/implicit_grad.py",
    }
    offenders = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(root, "gpurec/**/*.py")
        if bridge_name in path.read_text(encoding="utf-8")
        and path.relative_to(root).as_posix() not in allowed_paths
    ]
    assert offenders == []


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
        "`GeneReconModel.nll_per_family()` and `GeneReconModel.full_nll_per_family()`",
        "are genewise-only",
        "row-wise optimizers",
        "`model(reduce=\"per_family\")` under `torch.no_grad()`",
        "independent per-family gradients are not defined",
    ):
        assert token in normalized


def test_removed_likelihood_aliases_stay_out_of_runtime_surface():
    root = Path(__file__).resolve().parents[2]
    legacy_full = "compute_log_" + "likelihood"
    legacy_root_rows = f"{legacy_full}_root_rows"
    likelihood_source = (
        root / "gpurec" / "core" / "likelihood.py"
    ).read_text(encoding="utf-8")
    runtime_plan = " ".join(
        (
            root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
        ).read_text(encoding="utf-8").split()
    )
    simplification_index = " ".join(
        (
            root / "docs" / "simplification-opportunity-index-2026-05-21.md"
        ).read_text(encoding="utf-8").split()
    )

    assert "def compute_nll" in likelihood_source
    assert "def compute_nll_root_rows" in likelihood_source
    assert f"def {legacy_full}" not in likelihood_source
    assert f"def {legacy_root_rows}" not in likelihood_source

    for token in (
        f"`{legacy_full}()`",
        f"`{legacy_root_rows}()`",
        "aliases have been removed",
        "`compute_nll()` and `compute_nll_root_rows()`",
        "prevents tracked Python surfaces from reintroducing the removed aliases",
    ):
        assert token in runtime_plan

    for token in (
        "LIK-02 - Remove Or Deprecate Misleading Log-Likelihood Aliases",
        "Remove the misleading aliases after public usage is checked",
        "Repository hygiene coverage proving tracked Python surfaces use `compute_nll*`",
    ):
        assert token in simplification_index

    offenders = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(
            root,
            "gpurec/**/*.py",
            "tests/**/*.py",
            "scripts/**/*.py",
            "profiling/**/*.py",
        )
        if (
            legacy_full in path.read_text(encoding="utf-8")
            or legacy_root_rows in path.read_text(encoding="utf-8")
        )
    ]

    assert offenders == []


def test_compute_nll_is_thin_root_row_adapter():
    root = Path(__file__).resolve().parents[2]
    likelihood_module = ast.parse(
        (root / "gpurec" / "core" / "likelihood.py").read_text(encoding="utf-8")
    )
    compute_nll = next(
        node
        for node in likelihood_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_nll"
    )
    called_names = {
        node.func.id
        for node in ast.walk(compute_nll)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "compute_nll_root_rows" in called_names
    assert "logsumexp2" not in called_names
    assert "_weighted_logsumexp2" not in called_names
    assert "compute_origination_denominator" not in called_names


def test_api_runtime_uses_root_row_likelihood_contract():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in _tracked_files(root, "gpurec/api/**/*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "compute_nll"
            ):
                offenders.append(
                    f"{path.relative_to(root).as_posix()}:{node.lineno}"
                )

    assert offenders == []


def test_runtime_e_fixed_point_calls_pass_explicit_e_shape():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in _tracked_files(
        root,
        "gpurec/**/*.py",
        "scripts/**/*.py",
        "profiling/**/*.py",
    ):
        relative = path.relative_to(root).as_posix()
        if relative == "gpurec/core/likelihood.py":
            continue
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            is_e_call = (
                isinstance(node.func, ast.Name)
                and node.func.id == "E_fixed_point"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "E_fixed_point"
            )
            if not is_e_call:
                continue
            if not any(keyword.arg == "e_shape" for keyword in node.keywords):
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == []


def test_dts_shape_precedence_is_documented_before_runtime_change():
    root = Path(__file__).resolve().parents[2]
    readme = " ".join((root / "README.md").read_text(encoding="utf-8").split())
    forward_module = ast.parse(
        (root / "gpurec" / "core" / "forward.py").read_text(encoding="utf-8")
    )
    dts_module = ast.parse(
        (root / "gpurec" / "core" / "kernels" / "dts_fused.py").read_text(
            encoding="utf-8"
        )
    )
    backward_module = ast.parse(
        (root / "gpurec" / "core" / "kernels" / "wave_backward.py").read_text(
            encoding="utf-8"
        )
    )
    docstrings = {
        "Pi_wave_forward": " ".join(
            (
                ast.get_docstring(
                    next(
                        node
                        for node in forward_module.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "Pi_wave_forward"
                    )
                )
                or ""
            ).split()
        ),
        "_prepare_param": " ".join(
            (
                ast.get_docstring(
                    next(
                        node
                        for node in dts_module.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_prepare_param"
                    )
                )
                or ""
            ).split()
        ),
        "_dts_layout_param_args": " ".join(
            (
                ast.get_docstring(
                    next(
                        node
                        for node in backward_module.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_dts_layout_param_args"
                    )
                )
                or ""
            ).split()
        ),
    }

    for token in (
        "avoid bare `[G]` DTS parameter vectors when `G == S`",
        "`[G, 1]` for family scalar rows",
        "`[G, S]` for family/species rows",
        "direct DTS forward helper treats a one-dimensional length-`S` tensor as shared species-indexed",
        "retained backward helper with `family_idx` treats a one-dimensional tensor as family-indexed",
    ):
        assert token in readme

    for token in (
        "Direct callers should pass [G, 1] or [G, S]",
        "forward/backward DTS paths cannot disagree",
    ):
        assert token in docstrings["Pi_wave_forward"]

    for token in (
        "1-D tensor with ``numel() == S`` is treated as a shared species vector",
        "normalize genewise scalar rows to ``[G, 1]``",
        "when ``G == S``",
    ):
        assert token in docstrings["_prepare_param"]

    for token in (
        "With ``family_idx`` present",
        "one-dimensional tensor as family scalar rows",
        "when ``G == S``",
    ):
        assert token in docstrings["_dts_layout_param_args"]


def test_log_every_docs_distinguish_stdout_from_history():
    root = Path(__file__).resolve().parents[2]
    readme = " ".join((root / "README.md").read_text(encoding="utf-8").split())
    cli_source = (root / "gpurec" / "cli.py").read_text(encoding="utf-8")
    config_source = (
        root / "gpurec" / "workflow" / "config.py"
    ).read_text(encoding="utf-8")
    optimize_source = (
        root / "gpurec" / "workflow" / "optimize.py"
    ).read_text(encoding="utf-8")

    for token in (
        "History JSONL is recorded for every optimizer step",
        "`log_every` and `--log-every` only throttle console progress prints",
    ):
        assert token in readme

    assert "History logging interval" not in cli_source
    assert "Console progress print interval in optimization steps" in cli_source
    assert "history is recorded every step" in cli_source
    assert "History rows are recorded every optimizer step" in config_source
    assert "step % config.log_every" in optimize_source
    assert "self._record(row)" in optimize_source
    assert optimize_source.index("self._record(row)") < optimize_source.index(
        "step % config.log_every"
    )


def test_project_readme_and_model_docstrings_document_full_batch_helpers():
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
    docstrings = {
        node.name: " ".join((ast.get_docstring(node) or "").split())
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"materialize_batches", "full_loss_for_theta"}
    }

    for token in (
        "`model.materialize_batches()` builds every resident batch static state",
        "returns a copy of the batch metadata list",
        "`model.full_loss_for_theta(theta)` streams all resident batches",
        "differentiable probes use the gradient-producing streaming path",
        "calls made under `torch.no_grad()` use the loss-only streaming path",
    ):
        assert token in normalized_readme

    for token in (
        "Build all resident batch static states",
        "returned list is a copy",
        "without mutating model bookkeeping",
    ):
        assert token in docstrings["materialize_batches"]

    for token in (
        "Stream every resident batch using an explicit theta tensor",
        "uses the gradient-producing full-batch streaming path",
        "uses the loss-only streaming path",
    ):
        assert token in docstrings["full_loss_for_theta"]


def test_public_properties_and_batched_lbfgs_knobs_are_documented():
    root = Path(__file__).resolve().parents[2]
    model_module = ast.parse(
        (root / "gpurec" / "api" / "model.py").read_text(encoding="utf-8")
    )
    uniform_module = ast.parse(
        (root / "gpurec" / "api" / "uniform_chunked.py").read_text(
            encoding="utf-8"
        )
    )
    lbfgs_module = ast.parse(
        (root / "gpurec" / "optimization" / "batched_lbfgs.py").read_text(
            encoding="utf-8"
        )
    )

    def class_docstrings(module: ast.Module, class_name: str) -> dict[str, str]:
        class_node = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return {
            node.name: " ".join((ast.get_docstring(node) or "").split())
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }

    model_docs = class_docstrings(model_module, "GeneReconModel")
    uniform_docs = class_docstrings(uniform_module, "UniformChunkedReconModel")
    lbfgs_class = next(
        node
        for node in lbfgs_module.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchedLBFGS"
    )
    lbfgs_doc = " ".join((ast.get_docstring(lbfgs_class) or "").split())

    for name, token in (
        ("current_batch_metadata", "Metadata for the resident batch"),
        ("current_batch_index", "Index of the resident batch"),
        ("mode", "Parameter-sharing mode"),
        ("family_names", "Family names in dataset order"),
        ("species_tree_path", "Path to the species tree"),
        ("n_families", "Number of gene families"),
        ("species_names", "internal species-index order"),
        ("n_species", "Number of species"),
    ):
        assert token in model_docs[name]

    for name, token in (
        ("n_families", "Number of gene families"),
        ("family_count", "Alias for"),
        ("chunk_count", "Number of built uniform chunks"),
        ("fixed_iters_Pi", "Fixed Pi iteration count"),
        ("fixed_iters_E", "adaptive E is active"),
        ("chunk_metadata", "Immutable per-chunk metadata"),
    ):
        assert token in uniform_docs[name]

    for token in (
        "max_eval",
        "Maximum closure evaluations",
        "max_iter * (max_ls + 1) + 1",
        "accepted probed loss",
        "tolerance_grad",
        "tolerance_change",
        "max_ls",
        "c1",
        "shrink",
    ):
        assert token in lbfgs_doc


def test_small_species_backward_limitation_is_documented_publicly():
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    source = (root / "gpurec" / "core" / "backward.py").read_text(
        encoding="utf-8"
    )

    for text in (readme, docs_readme):
        for token in (
            "not a CPU fallback",
            "S > 256",
            "not an end-to-end optimizer smoke",
        ):
            assert token in text

    for token in (
        "currently requires ``S > 256`` species",
        "small-species backward fallback",
        "requires CUDA, ``float32``/``float64``, and ``S > 256``",
        'raise RuntimeError("Pi_wave_backward fused path requires S > 256")',
    ):
        assert token in source


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
        "narrow low-level `GeneDataset(..., leaf_species_maps=...)` exception",
        "the rest of `gpurec.core` should be treated as unstable implementation surface",
    ):
        assert token in normalized


def test_project_readme_documents_checkpoint_config_metadata_surface():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())
    checkpoint_module = ast.parse(
        (root / "gpurec" / "workflow" / "checkpoint.py").read_text(
            encoding="utf-8"
        )
    )
    module_docstring = " ".join((ast.get_docstring(checkpoint_module) or "").split())
    supported_checkpoint_names = {
        "CHECKPOINT_VERSION",
        "load_checkpoint",
        "restore_model_theta",
        "save_checkpoint",
        "validate_checkpoint_model_compatibility",
    }

    for token in (
        "`load_checkpoint(path)[\"config\"]`",
        "`load_checkpoint(path)[\"route_metadata\"]`",
        "`gpurec.workflow.checkpoint`",
        "`RunConfig.from_dict(...)`",
        "no separate public `load_checkpoint_config`",
    ):
        assert token in project_readme

    for token in (
        "lower-level `gpurec.workflow.checkpoint` submodule explicitly supports",
        "`save_checkpoint`",
        "`load_checkpoint`",
        "`restore_model_theta`",
        "`validate_checkpoint_model_compatibility`",
        "`CHECKPOINT_VERSION`",
        "not top-level `gpurec.workflow` shortcuts",
        "versioned checkpoint payload",
    ):
        assert token in normalized

    for token in (
        "explicit lower-level support boundary",
        "stable shortcut surface is ``gpurec.workflow``/top-level ``gpurec``",
        "payload schema is versioned",
        "``route_metadata``",
    ):
        assert token in module_docstring

    assert set(workflow_checkpoint.__all__) == supported_checkpoint_names
    for name in supported_checkpoint_names:
        assert name not in workflow.__all__
        assert name not in gpurec.__all__
        assert hasattr(workflow_checkpoint, name)


def test_checkpoint_identity_and_config_validation_boundary_is_documented():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    checkpoint_source = (root / "gpurec" / "workflow" / "checkpoint.py").read_text(
        encoding="utf-8"
    )
    checkpoint_module = ast.parse(checkpoint_source)
    module_docstring = " ".join((ast.get_docstring(checkpoint_module) or "").split())
    normalized_readme = " ".join(project_readme.split())
    config_identity_keys = {
        "species_tree",
        "families_file",
        "mode",
        "start",
        "max_families",
    }

    assert (
        set(getattr(workflow_checkpoint, "_CHECKPOINT_CONFIG_IDENTITY_KEYS"))
        == config_identity_keys
    )
    for token in (
        "Version-1 workflow checkpoints carry identity metadata for safe restore",
        "`family_names`",
        "`species_names`",
        "`species_tree`",
        "`families_file`",
        "`mode`",
        "`start`",
        "`max_families`",
        "`load_checkpoint()` requires those fields to be present",
        "`validate_checkpoint_model_compatibility()` first validates the stored config",
        "with `RunConfig.from_dict(...)`",
        "then compares identity and route metadata with the active `RunConfig`",
        "before `restore_model_theta()` copies parameters",
        "Path identity fields are normalized during comparison",
        "lower-level payload reader",
    ):
        assert token in normalized_readme

    for token in (
        "Version-1 checkpoints carry identity metadata for safe restore",
        "``family_names``",
        "``species_names``",
        "``species_tree``",
        "``families_file``",
        "``mode``",
        "``start``",
        "``max_families``",
        "``load_checkpoint()`` requires those fields to exist",
        "``validate_checkpoint_model_compatibility()`` compares them",
        "before ``restore_model_theta()`` copies parameters",
        "first validating the stored config with ``RunConfig.from_dict(...)``",
        "normalizing only path identity fields during comparison",
        "lower-level payload reader",
    ):
        assert token in module_docstring

    assert "_require_config_identity_fields(path, payload[\"config\"])" in checkpoint_source
    assert "_validate_checkpoint_run_config(checkpoint_path, checkpoint_config)" in checkpoint_source
    assert "checkpoint_string_list(path, \"family_names\"" in checkpoint_source
    assert "checkpoint_string_list(path, \"species_names\"" in checkpoint_source
    assert "_normalize_checkpoint_identity_value(" in checkpoint_source


def test_newick_input_subset_is_documented_on_public_surfaces():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(project_readme.split())

    for token in (
        "retained preprocessing parser supports a deliberately small Newick subset",
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

    rust_preprocess = (
        root / "crates" / "gpurec-preprocess" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    for token in (
        "GeneNewickParser",
        "skip_branch_length",
        "preprocess_multiple_families_with_threads",
    ):
        assert token in rust_preprocess


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


def test_project_readme_documents_top_level_entropy_helpers():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    public_helpers = sorted(
        name
        for name, module_name in gpurec._LAZY_EXPORTS.items()
        if module_name == "gpurec.entropy"
    )

    for name in public_helpers:
        assert name in project_readme
    for token in (
        "analytical reconciliation entropy",
        "`compute_reconciliation_entropy()` works from a solved `GeneReconModel` family",
        "`reconciliation_entropy_from_payload()` works from an `export_backtracking_input()`",
        "`collapsed` result",
        "stochastic backtracker",
        "`expanded` also includes hidden extinction histories",
        "`E` and\n`Ebar`",
    ):
        assert token in project_readme


def test_project_readme_documents_supported_environment_flags_only():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    package_env_flags = _package_env_flags(root)

    assert sorted(package_env_flags) == sorted(_SUPPORTED_ENV_FLAGS)
    assert sorted(name for name in _SUPPORTED_ENV_FLAGS if name not in project_readme) == []
    assert sorted(name for name in _UNSUPPORTED_ENV_DOC_FLAGS if name in project_readme) == []


def test_runtime_surface_plan_records_package_environment_owners():
    root = Path(__file__).resolve().parents[2]
    pruning_plan = (
        root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
    ).read_text(encoding="utf-8")
    package_env_flags = _package_env_flags(root)
    runtime_read_flags = _runtime_read_env_flags(root)
    owner_by_flag = _runtime_environment_owner_manifest(root)

    assert _SUPPORTED_ENV_FLAGS.issubset(package_env_flags)
    assert _SUPPORTED_ENV_FLAGS.issubset(runtime_read_flags)
    assert sorted(runtime_read_flags) == sorted(_SUPPORTED_ENV_FLAGS)
    assert sorted(owner_by_flag) == sorted(_SUPPORTED_ENV_FLAGS)
    assert sorted(
        flag
        for flag, owner in owner_by_flag.items()
        if owner in _SUPPORTED_ENV_CATEGORIES
    ) == sorted(_SUPPORTED_ENV_FLAGS)
    normalized_plan = " ".join(pruning_plan.split())
    assert "Supported environment flags are limited to" in normalized_plan
    assert "Scheduler backend selection" in normalized_plan
    assert "preprocess cache locations" in normalized_plan
    assert "backward CUDA/Triton selectors" in normalized_plan
    assert "kernel launch tuning" in normalized_plan
    assert sorted(
        name for name in _UNSUPPORTED_ENV_DOC_FLAGS if name in pruning_plan
    ) == []


def test_project_readme_excludes_removed_diagnostic_environment_matrix():
    root = Path(__file__).resolve().parents[2]
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(project_readme.split())

    for token in (
        "Kernel routing, scheduler selection, cache locations, and launch tuning "
        "are not supported environment contracts",
        "Rust preprocessing discovery",
        "`GPUREC_BACKTRACK_NATIVE_LIB`",
        "Python helper calls with `backend=\"native\"`",
    ):
        assert token in normalized
    assert sorted(name for name in _UNSUPPORTED_ENV_DOC_FLAGS if name in project_readme) == []


def test_backward_source_omits_native_cuda_prototype_surface():
    root = Path(__file__).resolve().parents[2]
    backward_source = (root / "gpurec" / "core" / "backward.py").read_text(
        encoding="utf-8"
    )
    wave_backward_source = (
        root / "gpurec" / "core" / "kernels" / "wave_backward.py"
    ).read_text(encoding="utf-8")

    for token in (
        "_cuda_self_loop",
        "_OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS",
        "wave_backward_cuda",
        "pibar_vjp_cuda",
        "GPUREC_CUDA_SELF_LOOP",
        "GPUREC_CUDA_PIBAR_FROM_UD",
        "GPUREC_TRITON_CHILD_EDGE_SELF_LOOP",
        "GPUREC_TRITON_SELF_LOOP_DIRECT_GRADS",
        "GPUREC_BACKWARD_NO_CPU_PRUNING",
        "GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO",
        "GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO",
    ):
        assert token not in backward_source
        assert token not in wave_backward_source

    for relative_path in (
        "gpurec/core/kernels/wave_backward_cuda.py",
        "gpurec/core/kernels/pibar_vjp_cuda.py",
    ):
        assert not (root / relative_path).exists()

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


def test_rust_backtracking_apply_term_variants_have_direct_unit_tests():
    root = Path(__file__).resolve().parents[2]
    lib_rs = (
        root / "crates" / "gpurec-backtrack" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")

    for token in (
        "fn hidden_transfer_donor_emits_transfer_loss_and_requeues_recipient",
        "fn hidden_speciation_left_emits_loss_on_right_species",
        "fn hidden_speciation_right_emits_loss_on_left_species",
        "fn split_transfer_right_keeps_left_child_on_donor_branch",
        "fn split_transfer_left_keeps_right_child_on_donor_branch",
        "fn split_speciation_assigns_left_and_right_clades_to_species_children",
        "fn swapped_split_speciation_swaps_clades_not_species_children",
        "Term::HiddenTransferLossDonor",
        "Term::HiddenSpeciationLeft",
        "Term::HiddenSpeciationRight",
        "Term::SplitTransferRight",
        "Term::SplitTransferLeft",
        "Term::SplitSpeciation(0, true)",
        "assert_work_item(",
    ):
        assert token in lib_rs


def test_private_family_tree_path_alias_is_not_in_source_surface():
    root = Path(__file__).resolve().parents[2]
    private_alias = "_normalize_" + "family_tree_paths"
    offenders = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(
            root,
            "gpurec/**/*.py",
            "scripts/**/*.py",
            "profiling/**/*.py",
        )
        if private_alias in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_uniform_chunked_state_container_is_internal():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse(
        (root / "gpurec" / "api" / "uniform_chunked.py").read_text(
            encoding="utf-8"
        )
    )
    class_names = {
        node.name for node in module.body if isinstance(node, ast.ClassDef)
    }
    name_refs = [
        node.lineno
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and node.id == "UniformChunkedState"
    ]
    all_assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    exported_names = ast.literal_eval(all_assignment.value)

    assert "UniformChunkedState" not in class_names
    assert "_UniformChunkedState" in class_names
    assert name_refs == []
    assert "UniformChunkedState" not in exported_names
    assert "_UniformChunkedState" not in exported_names


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
    for stale_line_reference in (
        "gpurec/api/autograd.py:280",
        "gpurec/optimization/implicit_grad.py:20",
        "gpurec/api/uniform_chunked.py:627",
        "gpurec/api/uniform_chunked.py:640",
    ):
        assert stale_line_reference not in note
    assert "exact source line" in note
    assert "numbers because the optimization internals move frequently" in note


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


def test_documented_uniform_pipeline_benchmark_help_imports_current_api():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "profiling/bench_uniform_forward_backward_pipeline.py",
            "--help",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    source = (
        root / "profiling" / "bench_uniform_forward_backward_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "--family-chunk-size" in result.stdout
    assert "--stats-only" in result.stdout
    assert "_UniformBuiltChunk as BuiltChunk" in source
    assert "_built_chunks_from_rust" in source
    assert "RustPreprocessExtension" in source
    assert "compute_nll" in source
    assert ("compute_log_" + "likelihood") not in source


def test_profiling_readme_documents_entrypoints_and_artifact_policy():
    root = Path(__file__).resolve().parents[2]
    profiling_readme = (root / "profiling" / "README.md").read_text(
        encoding="utf-8"
    )
    normalized_profiling_readme = " ".join(profiling_readme.split())
    docs_map = (root / "docs" / "README.md").read_text(encoding="utf-8")
    project_readme = (root / "README.md").read_text(encoding="utf-8")
    tracked_profiling_files = [
        path.name for path in _tracked_files(root, "profiling/*.py")
    ]

    for token in (
        "The tracked profiling directory is source-checkout tooling",
        "Supported Entrypoints",
        "bench_uniform_forward_backward_pipeline.py",
        "Maintained full-pipeline benchmark",
        "evaluate_hogenom_alerax_rates.py",
        "Checkout-local HOGENOM/AleRax validation helper",
        "New tracked profiling entrypoints should have a `--help` smoke",
        "adaptive fixed-point convergence",
        "Artifact Policy",
        "profiling/ancestor_batching/",
        "profiling/bf16_backward_nsys/",
        "profiling/bf16_handoff_prod/",
        "profiling/hogenom_ccp/",
        "profiling/specieswise_worker3/",
        "profiling/proposal2/",
        "profiling/proposal8/",
        "Python bytecode cache",
        "not source, fixtures, or retained benchmark results",
    ):
        assert token in normalized_profiling_readme
    assert [
        name for name in tracked_profiling_files
        if name not in normalized_profiling_readme
    ] == []
    assert "../profiling/README.md" in docs_map
    assert "See `profiling/README.md`" in project_readme


def test_full_benchmark_status_records_timed_evidence_and_diagnostic_boundary():
    root = Path(__file__).resolve().parents[2]
    lean_fast_path = (root / "docs" / "lean-fast-path.md").read_text(
        encoding="utf-8"
    )
    execution_log = (
        root / "docs" / "simplification-execution-log-2026-05-21.md"
    ).read_text(encoding="utf-8")
    benchmark_source = (
        root / "profiling" / "bench_uniform_forward_backward_pipeline.py"
    ).read_text(encoding="utf-8")
    benchmark_tests = (
        root / "tests" / "unit" / "test_bench_uniform_forward_backward_pipeline.py"
    ).read_text(encoding="utf-8")
    lean_fast_path_status = " ".join(lean_fast_path.split())
    execution_log_status = " ".join(execution_log.split())
    combined_status = " ".join((lean_fast_path + "\n" + execution_log).split())

    for proposal in ("ENV-01", "SCHED-01", "BWD-01", "BWD-02"):
        assert proposal in combined_status
    for token in (
        "valid full 1000-family timed run exists",
        "Windowed preflight remains a setup diagnostic only",
        "`performance_evidence 0`",
        "Triton-only for backward kernels",
        "Rust-only for scheduling/chunk layout",
        "does not use preprocessing result caches",
        "After deleting the generated caches and removing inclusion-DAG materialization",
        "pipeline_policy families 1000 chunks 40 family_chunk_size 25 max_wave_size 8192",
        "strict_optimized_verdict pass",
        "pipeline_summary reps 3",
        "forward_median_ms 2169.380",
        "backward_median_ms 3699.282",
        "total_median_ms 5867.989",
        "grad_finite 1",
    ):
        assert token in lean_fast_path_status
    for token in (
        "superseded by the follow-up cleanup",
        "Triton-only for backward kernels",
        "diagnostic env toggles and scheduler alternatives have been removed",
        "1000-family benchmark now has a valid timed run",
    ):
        assert token in execution_log_status
    assert '"performance_evidence", 0' in benchmark_source
    assert "performance_evidence 0" in benchmark_tests


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


def test_ignored_local_workspace_inventory_documents_notebooks_and_profiles():
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    notebooks_readme = (root / "notebooks" / "README.md").read_text(encoding="utf-8")
    pruning_plan = (
        root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
    ).read_text(encoding="utf-8")
    profiling_readme = (root / "profiling" / "README.md").read_text(
        encoding="utf-8"
    )
    combined_docs = notebooks_readme + "\n" + pruning_plan + "\n" + profiling_readme

    for token in (
        "*.ipynb",
        "profiling/ancestor_batching/",
        "profiling/bf16_backward_nsys/",
        "profiling/bf16_handoff_prod/",
        "profiling/hogenom_ccp/",
        "profiling/specieswise_worker3/",
    ):
        assert token in gitignore

    for token in (
        "Ignored/local workspace inventory",
        "Ignored Local Notebooks",
        "evaluate_gpurec_at_alerax_params.ipynb",
        "hogenom_adam_bfgs_schedule.ipynb",
        "optimize_hogenom_ccp_specieswise_origination.ipynb",
        "pi_iteration_bound_diagnostic.ipynb",
        "gpurec.optimization.optimize_scheduled",
        "pi_iteration_bound_diagnostic_impl",
        "profiling/ancestor_batching/",
        "missing local harnesses",
        "profiling/bf16_backward_nsys/",
        "profiling/bf16_handoff_prod/",
        "bf16 is now documented as direct-API-only",
        "profiling/hogenom_ccp/",
        "profiling/specieswise_worker3/",
        "profiling/proposal2/",
        "profiling/proposal8/",
        "archive/delete",
    ):
        assert token in combined_docs


def test_ignored_local_test_data_inventory_is_documented():
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    tests_readme = (root / "tests" / "README.md").read_text(encoding="utf-8")
    pruning_plan = (
        root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
    ).read_text(encoding="utf-8")
    combined_docs = tests_readme + "\n" + pruning_plan

    ignored_data_patterns = (
        "tests/data/test_trees_20/",
        "tests/data/test_trees_100/",
        "tests/data/test_trees_1000/",
        "tests/data/test_trees_10000/",
        "tests/data/test_trees_dtl01/",
        "tests/data/HOGENOM/",
        "tests/data/davin/",
        "tests/data/hogenom_bench/",
        "tests/data.tar.gz",
        "tests/data/**/output/",
    )
    for token in ignored_data_patterns:
        assert token in gitignore
        assert token in combined_docs

    for token in (
        "Ignored local test data roots",
        "distributable fixture",
        "Ignored test-data and cache inventory",
        "Generated tree-scale fixtures",
        "External or checkout-local biological datasets",
        "Do not treat as source of truth",
        "Delete/regenerate as needed",
        "before any required workflow depends on it",
    ):
        assert token in combined_docs


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


def test_fixed_dataset_hogenom_launchers_document_unique_contracts():
    root = Path(__file__).resolve().parents[2]
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    script_paths = [
        root / "scripts" / "optimize_hogenom_ccp_global_uniform.py",
        root / "scripts" / "optimize_hogenom_ccp_specieswise_uniform.py",
        root / "scripts" / "profile_hogenom_ccp_pass.py",
    ]
    source_paths = [
        *script_paths,
        root / "scripts" / "hogenom_opt_helpers.py",
    ]
    sources = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    for token in (
        "Checkout-local HOGENOM run with one shared D/T/L theta row",
        "`global_optimization_history.csv`",
        "`global_rate_distribution_history.csv`",
        "`optimized_global_rates.csv`",
        "`uniform_origination_distribution.csv`",
        "square/l1/huber/elastic-net/gaussian/beta-pS regularizers",
        "Checkout-local HOGENOM run with one D/T/L theta row per species",
        "`specieswise_optimization_history.csv`",
        "`specieswise_parameter_history.csv`",
        "`optimized_specieswise_rates.csv`",
        "helper-owned optimizer schedules",
        "HOGENOM-only CUDA/Nsight harness",
        "`config`, `model`, `warmup`, `measured`, and `summary` events",
        "CUDA profiler API calls and NVTX ranges",
    ):
        assert token in scripts_readme

    for token in (
        "Checkout-local global-uniform HOGENOM optimizer reproducer",
        "output_gpurec_global_uniform_opt_max100",
        "Checkout-local specieswise-uniform HOGENOM optimizer reproducer",
        "output_gpurec_specieswise_uniform_opt_max100",
        "Shared helpers for checkout-local fixed-dataset HOGENOM launchers",
        "Migrate reusable optimizer schedules",
        "schemas into ``gpurec.workflow``",
        "Checkout-local HOGENOM CCP CUDA/Nsight profiling harness",
        "config/model/warmup/measured/summary events",
        "argparse.RawDescriptionHelpFormatter",
    ):
        assert token in sources

    for script in script_paths:
        result = subprocess.run(
            [sys.executable, str(script.relative_to(root)), "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert "Checkout-local contract:" in result.stdout


def test_branchscale_penalty_report_documents_legacy_layout_and_staleness():
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "scripts" / "make_hogenom_branchscale_penalty_report.py"
    ).read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")

    for token in (
        "Checkout-local branchscale penalty report builder",
        "penalty_*",
        "history.jsonl",
        "branchscaled_node_rates_final.tsv",
        "tree_plots/rates_final.png",
        "timestamped",
        "May 18, 2026",
        "1325 branch multipliers",
        "archive/delete",
    ):
        assert token in script

    for token in (
        "Expected layout: `penalty_*` child directories",
        "`history.jsonl`",
        "`branchscaled_node_rates_final.tsv`",
        "`tree_plots/rates_final.png`",
        "timestamped launcher outputs",
        "Delete or migrate",
        "supported CLI",
    ):
        assert token in scripts_readme


def test_penalty316_kkt_script_documents_checkout_local_contract():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "optimize_hogenom_penalty316_kkt.py"
    script = script_path.read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    help_result = subprocess.run(
        [sys.executable, str(script_path.relative_to(root)), "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    for token in (
        "Checkout-local branchscale penalty/KKT analysis launcher",
        "legacy HOGENOM W&B optimizer",
        "branchscale penalty-316.22776601683796",
        "CUDA optimized likelihood path",
        "Branchscaled mode with W&B disabled",
        "100 Adam warmup steps",
        "Strong-Wolfe LBFGS",
        "L1 KKT residual",
        "Timestamped output directories",
        "latest_run.txt",
        "Archive or migrate",
        "tiny fixture guard",
    ):
        assert token in script

    for token in (
        "Checkout-local contract",
        "default penalty 316.22776601683796",
        "timestamped run directories",
        "latest_run.txt",
        "Archive/delete or migrate",
        "supported gpurec CLI",
    ):
        assert token in help_result.stdout

    for token in (
        "Checkout-local branchscaled penalty-316.22776601683796 reproducer",
        "W&B disabled",
        "100 Adam warmup steps",
        "Strong-Wolfe LBFGS",
        "L1 KKT residual checks",
        "timestamped output directories",
        "`latest_run.txt`",
        "supported CLI",
        "tiny fixture guard",
    ):
        assert token in scripts_readme


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


def test_rust_preprocess_extension_exports_match_runtime_surface():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "crates" / "gpurec-preprocess" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")

    expected_exports = {
        "preprocess_dataset",
        "preprocess_request_binary",
        "preprocess_request_numpy",
        "preprocess_request_torch",
        "schedule_global_phased_waves_json",
        "family_schedule_summary_json",
        "plan_family_batches_json",
        "build_wave_layout_plan_json",
    }
    actual_exports = set(re.findall(r"wrap_pyfunction!\(([^,\s]+)", source))

    assert expected_exports <= actual_exports
    assert "PYBIND11_MODULE" not in source
    assert 'm.def("' not in source


def test_removed_preprocess_diagnostic_exports_have_no_runtime_callers():
    root = Path(__file__).resolve().parents[2]
    runtime_files = _tracked_package_python_files(root)
    legacy_exports = {"preprocess"}
    diagnostic_exports = {
        "compute_phased_waves",
        "compute_wave_stats",
        "compute_packet_wave_stats",
        "compute_phased_wave_stats",
        "compute_phased_cross_family_wave_stats",
        "compute_cross_family_wave_stats",
    }
    blocked_exports = legacy_exports | diagnostic_exports
    call_offenders: list[str] = []
    preprocess_family_calls: list[tuple[str, ast.Call]] = []

    for path in runtime_files:
        relative = path.relative_to(root).as_posix()
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "preprocess_multiple_families":
                preprocess_family_calls.append((relative, node))
            if node.func.attr in blocked_exports:
                call_offenders.append(f"{relative}:{node.lineno}:{node.func.attr}")

    assert call_offenders == []
    assert [path for path, _node in preprocess_family_calls] == [
        "gpurec/core/model.py",
        "gpurec/core/model.py",
    ]

    detailed_calls = 0
    species_only_default_calls = 0
    for _path, node in preprocess_family_calls:
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        families_arg = node.args[1] if len(node.args) > 1 else None
        include_details = keywords.get("include_details")

        if isinstance(include_details, ast.Constant) and include_details.value is True:
            detailed_calls += 1
            assert isinstance(
                keywords.get("include_species_matrices"), ast.Constant
            )
            assert keywords["include_species_matrices"].value is False
            continue

        assert "include_details" not in keywords
        assert isinstance(families_arg, ast.Dict)
        assert families_arg.keys == []
        assert isinstance(keywords.get("include_species_matrices"), ast.Constant)
        assert keywords["include_species_matrices"].value is False
        species_only_default_calls += 1

    assert detailed_calls == 2
    assert species_only_default_calls == 0


def test_rust_preprocess_does_not_materialize_inclusion_dag_payloads():
    root = Path(__file__).resolve().parents[2]
    rust_source = (root / "crates/gpurec-preprocess/src/lib.rs").read_text(
        encoding="utf-8"
    )
    model_source = (root / "gpurec/core/model.py").read_text(encoding="utf-8")

    for token in (
        "include_inclusion",
        "inclusion_children",
        "inclusion_parents",
        "ubiquitous_clade_id",
    ):
        assert token not in rust_source

    for token in (
        'ccp.pop("inclusion_children", None)',
        'ccp.pop("inclusion_parents", None)',
        'ccp.pop("ubiquitous_clade_id", None)',
    ):
        assert token in model_source


def test_test_only_scheduler_helpers_stay_out_of_runtime_source():
    root = Path(__file__).resolve().parents[2]
    helper_names = (
        "compute_clade_" + "waves",
        "collate_" + "wave",
        "split_phase_" + "waves",
    )
    batching_module = ast.parse(
        (root / "gpurec" / "core" / "batching.py").read_text(encoding="utf-8")
    )
    function_names = {
        node.name for node in batching_module.body if isinstance(node, ast.FunctionDef)
    }
    scheduling_module = root / "gpurec" / "core" / "scheduling.py"
    offenders = [
        path.relative_to(root).as_posix()
        for path in _tracked_files(root, "gpurec/**/*.py")
        if any(name in path.read_text(encoding="utf-8") for name in helper_names)
    ]

    assert function_names.isdisjoint(helper_names)
    assert not scheduling_module.exists()
    assert offenders == []


def test_runtime_surface_plan_records_refresh_findings_before_behavior_changes():
    root = Path(__file__).resolve().parents[2]
    pruning_plan = (
        root / "docs" / "runtime-surface-pruning-plan-2026-05-21.md"
    ).read_text(encoding="utf-8")
    audit = (
        root / "docs" / "repo-wide-audit-2026-05-21.md"
    ).read_text(encoding="utf-8")
    normalized_plan = " ".join(pruning_plan.split())
    normalized_audit = " ".join(audit.split())

    for token in (
        "Core/API Refresh Findings",
        "`finite_float()`, `positive_float()`, and `nonnegative_float()`",
        "reject Python bools and bool tensors",
        "`tol_E`, `pi_max_diff_tol`, and `min_rate`",
        "`as_family_param()`, `as_family_species()`, and `extract_parameters_uniform()`",
        "`family_rows` precedence when `G == S`",
        "bare length-`G` ambiguity",
        "`_normalize_family_tree_paths()`",
        "private alias is absent from tracked runtime/script/profiling Python sources",
        "`gpurec.core.batch_planning.__all__`",
        "`FamilyBatchPlan`, `normalize_batch_packing`, `normalize_clade_budget`, `normalize_family_chunk_size`, and `plan_family_batches`",
        "narrow shared low-level planning boundary",
        "not a promise that the rest of `gpurec.core` is stable",
        "`UniformChunkedState`",
        "`_UniformChunkedState`",
        "absent as a class/name reference",
        "`UniformChunkedReconModel.nll_per_family()`",
        "no-grad global/uniform diagnostic",
        "exact `chunk_indices`",
        "`implicit_grad_loglik_vjp_wave()`",
        "documented as an internal bridge",
        "source guard limits tracked runtime references",
        "Direct `build_wave_layout()` family-index inputs",
        "`family_clade_counts` and `family_clade_offsets`",
        "Explicit theta tensors in `gpurec/api/model.py`",
        "Shared `validate_theta_shape()`",
        "wrong row counts now fail before CUDA checks",
        "rejects non-floating or nonfinite theta values before CUDA work",
        "invalid `theta_init` shapes/values and invalid explicit `theta` shapes/values",
        "`collate_gene_families()` docstring",
        "instead of removed `preprocess_gene_with_species`",
        "Workflow/Backtracking Refresh Findings",
        "`_BACKTRACK_HELP_MARKERS`",
        "`--seed`, `--output-dir`, and `--max-events`",
        "LBFGS branch evaluates the current theta",
        "`nonfinite_objective_or_gradient`",
        "Final optimization evaluation in `gpurec/workflow/optimize.py`",
        "mandatory `final_eval`",
        "`profiling/evaluate_hogenom_alerax_rates.py`",
        "`scripts/compare_backtracking_alerax_events.py`",
        "Local validation/profiling CLI count controls",
        "`gpurec/_argparse_types.py` helpers",
        "parser-level positive, non-negative, or positive-even errors",
        "`optimizer.load_state_dict`",
        "`ValueError`, `RuntimeError`, and `TypeError`",
        "`_RUN_CONFIG_CLI_OVERRIDE_FIELDS`",
        "dynamic `_RUN_CONFIG_CLI_OVERRIDE_FIELDS` attribute is absent",
        "`HiddenTransferLossDonor`",
        "`Sampler::apply_term`",
        "queued `WorkItem` clade/species state",
        "`_parse_minimal_pyproject()`",
        "Python 3.10 unit collection for TOML-reading tests",
        "Rust backtracking CLI multi-sample output",
    ):
        assert token in normalized_plan

    for token in (
        "follow-up core/API explorer",
        "direct API float-bool validation finding is now fixed",
        "private `_normalize_family_tree_paths()` compatibility alias is now deleted",
        "`normalize_family_chunk_size()` export-intent finding is now fixed",
        "whole `gpurec.core.batch_planning.__all__` export set is now documented",
        "`UniformChunkedState` ownership finding is now fixed",
        "explicit theta tensor shape validation finding is now fixed",
        "parameter extraction shape-contract finding is now guarded",
        "`UniformChunkedReconModel.nll_per_family()` diagnostic-contract finding is now guarded",
        "`implicit_grad_loglik_vjp_wave()` ownership finding is now guarded",
        "test-only scheduler helper deletion finding is now fixed",
        "Python `compute_clade_waves()` adapter deletion finding is now fixed",
        "follow-up workflow/backtracking explorer",
        "stale Rust sampler help-marker finding is now fixed",
        "Python 3.10 TOML fallback finding is now guarded",
        "tests/Rust/docs follow-up explorer",
        "TOML-reading unit modules use a `tomllib`/`tomli` conditional import",
        "workflow/scripts follow-up explorer",
        "mandatory final optimization evaluation nonfinite gap is now fixed",
        "Resume optimizer-state restore now also discards",
        "local HOGENOM script count-control finding is now fixed",
        "dynamic CLI compatibility attribute finding is now fixed",
        "Rust sampler term-variant finding is now fixed",
        "core/API follow-up explorer",
        "direct `build_wave_layout()` family-index gap is now fixed",
    ):
        assert token in normalized_audit


def test_collate_gene_families_docstring_uses_current_layout_owner():
    root = Path(__file__).resolve().parents[2]
    source = (root / "gpurec" / "core" / "batching.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    docstring = ast.get_docstring(functions["collate_gene_families"]) or ""

    assert "preprocessed gene-family CCP payloads" in docstring
    assert "build_wave_layout()" in docstring
    assert "preprocess_gene_with_species" not in docstring
    assert "likelihood_2.py" not in docstring


def test_removed_cpp_preprocessing_sources_do_not_return():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "gpurec" / "core" / "cpp").exists()
    assert not (root / "gpurec" / "core" / "preprocess_cpp.py").exists()


def test_repo_audit_headline_metrics_are_marked_as_initial_snapshot():
    root = Path(__file__).resolve().parents[2]
    audit = (
        root / "docs" / "repo-wide-audit-2026-05-21.md"
    ).read_text(encoding="utf-8")

    for token in (
        "initial read-only audit snapshot",
        "live repository",
        "Later verification entries record current collection counts",
        "Initial tracked scope snapshot",
        "Initial source-like size snapshot",
        "Initial test inventory snapshot",
    ):
        assert token in audit


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
        root / "profiling" / "bench_uniform_forward_backward_pipeline.py",
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
