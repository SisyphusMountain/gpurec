import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate_paths(module_name: str, library_filename: str) -> list[Path]:
    env_key = f"{module_name.upper()}_PATH"
    paths = []
    if env_path := os.environ.get(env_key):
        paths.append(Path(env_path).expanduser())
    paths.append(_REPO_ROOT / "gpurec" / library_filename)
    crate_name = module_name.replace("_", "-")
    paths.append(_REPO_ROOT / "crates" / crate_name / "target/release" / library_filename)
    paths.append(_REPO_ROOT / "crates" / crate_name / "target/debug" / library_filename)
    return paths


@lru_cache(maxsize=None)
def load_native_module(module_name: str, library_filename: str):
    """Load a bundled extension first, then fall back to local Cargo targets."""
    import_errors = []
    for import_name in (f"gpurec.{module_name}", module_name):
        try:
            return importlib.import_module(import_name)
        except ImportError as exc:
            import_errors.append(f"{import_name}: {exc}")

    tried = []
    for path in _candidate_paths(module_name, library_filename):
        tried.append(str(path))
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    tried_text = "\n  - ".join(tried)
    imports_text = "\n  - ".join(import_errors)
    raise ImportError(
        f"could not find native extension {module_name!r}. Imports tried:\n  - {imports_text}\n"
        f"Paths tried:\n  - {tried_text}\n"
        "Install the package with its Rust extensions, run `cargo build --release` "
        "inside the matching crate, or set the *_PATH environment override."
    )
