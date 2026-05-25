"""Python bridges for the Rust preprocessing prototype."""

from __future__ import annotations

import json
import os
import importlib.machinery
import importlib.util
import shlex
import shutil
import sys
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREPROCESS_MANIFEST = _REPO_ROOT / "crates" / "gpurec-preprocess" / "Cargo.toml"
_PREPROCESS_BINARY_ENV = "GPUREC_PREPROCESS_BIN"
_PREPROCESS_NATIVE_LIB_ENV = "GPUREC_PREPROCESS_NATIVE_LIB"
_PREPROCESS_RUN_TIMEOUT_SECONDS = 3600
_BINARY_MAGIC = b"GPREP001"
_NATIVE_MODULE_NAME = "gpurec_preprocess"


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _preprocess_command(
    *,
    cargo_manifest: str | Path = _PREPROCESS_MANIFEST,
    preprocess_binary: str | Path | None = None,
) -> list[str]:
    source = "preprocess_binary"
    if preprocess_binary is None:
        preprocess_binary = os.environ.get(_PREPROCESS_BINARY_ENV)
        source = _PREPROCESS_BINARY_ENV
    if preprocess_binary is not None:
        text = os.fspath(preprocess_binary)
        path = Path(text).expanduser()
        has_separator = any(sep and sep in text for sep in (os.sep, os.altsep))
        if (
            not isinstance(preprocess_binary, str)
            or path.is_absolute()
            or path.parent != Path(".")
            or has_separator
        ):
            resolved = path.resolve()
            if not resolved.is_file():
                raise RuntimeError(
                    f"gpurec preprocessing binary from {source} does not exist "
                    f"or is not a file: {resolved}"
                )
            if not os.access(resolved, os.X_OK):
                raise RuntimeError(
                    f"gpurec preprocessing binary from {source} is not executable: "
                    f"{resolved}"
                )
            return [str(resolved)]
        if shutil.which(text) is None:
            raise RuntimeError(
                f"gpurec preprocessing binary {text!r} from {source} was not found on PATH"
            )
        return [text]

    manifest = Path(cargo_manifest)
    if not manifest.exists():
        raise RuntimeError(
            "gpurec Rust preprocessing needs a binary or source-tree cargo "
            f"fallback; default source manifest not found at {manifest}"
        )
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "gpurec Rust preprocessing fallback requires cargo on PATH. "
            f"Set {_PREPROCESS_BINARY_ENV} or install Rust/Cargo."
        )
    return [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
    ]


def _is_cargo_fallback_command(command: list[str]) -> bool:
    return (
        len(command) >= 7
        and command[0] == "cargo"
        and command[1] == "run"
        and "--manifest-path" in command
        and command[-1] == "--"
    )


def _native_library_path(cargo_manifest: str | Path = _PREPROCESS_MANIFEST) -> Path:
    override = os.environ.get(_PREPROCESS_NATIVE_LIB_ENV)
    if override:
        return Path(override).expanduser().resolve()
    manifest = Path(cargo_manifest)
    return (manifest.parent / "target" / "release" / "libgpurec_preprocess.so").resolve()


def _build_native_extension(cargo_manifest: str | Path = _PREPROCESS_MANIFEST) -> None:
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "gpurec Rust preprocessing native backend requires cargo to build "
            f"the extension, or set {_PREPROCESS_NATIVE_LIB_ENV} to a built library"
        )
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "python-extension",
            "--manifest-path",
            str(cargo_manifest),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )


@lru_cache(maxsize=1)
def _load_native_module(cargo_manifest: str | Path = _PREPROCESS_MANIFEST):
    path = _native_library_path(cargo_manifest)
    if not path.exists() and os.environ.get(_PREPROCESS_NATIVE_LIB_ENV) is None:
        _build_native_extension(cargo_manifest)
    if not path.exists():
        raise RuntimeError(f"Rust preprocessing native library not found: {path}")

    existing = sys.modules.get(_NATIVE_MODULE_NAME)
    if existing is not None:
        return existing

    loader = importlib.machinery.ExtensionFileLoader(_NATIVE_MODULE_NAME, str(path))
    spec = importlib.util.spec_from_file_location(
        _NATIVE_MODULE_NAME,
        str(path),
        loader=loader,
    )
    if spec is None:
        raise RuntimeError(f"could not create import spec for Rust preprocessing library: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NATIVE_MODULE_NAME] = module
    try:
        loader.exec_module(module)
    except Exception:
        sys.modules.pop(_NATIVE_MODULE_NAME, None)
        raise
    return module


class _BinaryReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def _read(self, size: int) -> memoryview:
        end = self._offset + size
        if end > len(self._data):
            raise RuntimeError("truncated Rust preprocessing binary output")
        chunk = memoryview(self._data)[self._offset:end]
        self._offset = end
        return chunk

    def read_magic(self) -> None:
        magic = bytes(self._read(len(_BINARY_MAGIC)))
        if magic != _BINARY_MAGIC:
            raise RuntimeError("invalid Rust preprocessing binary output header")

    def read_u8(self) -> int:
        return int(self._read(1)[0])

    def read_u64(self) -> int:
        return int(struct.unpack_from("<Q", self._read(8))[0])

    def read_i64(self) -> int:
        return int(struct.unpack_from("<q", self._read(8))[0])

    def read_string(self) -> str:
        size = self.read_u64()
        return bytes(self._read(size)).decode("utf-8")

    def read_strings(self) -> list[str]:
        return [self.read_string() for _ in range(self.read_u64())]

    def read_sparse_strings(self) -> list[str]:
        values = [""] * self.read_u64()
        for _ in range(self.read_u64()):
            index = self.read_u64()
            values[index] = self.read_string()
        return values

    def read_i64_tensor(self) -> torch.Tensor:
        length = self.read_u64()
        if length == 0:
            return torch.empty(0, dtype=torch.int64)
        data = bytearray(self._read(length * 8))
        return torch.frombuffer(data, dtype=torch.int64)

    def read_f64_tensor(self, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        length = self.read_u64()
        if length == 0:
            tensor = torch.empty(0, dtype=torch.float64)
        else:
            data = bytearray(self._read(length * 8))
            tensor = torch.frombuffer(data, dtype=torch.float64)
        if shape is not None:
            tensor = tensor.reshape(shape)
        return tensor

    def read_optional_f64_tensor(
        self,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        present = self.read_u8()
        if present == 0:
            return None
        if present == 1:
            return self.read_f64_tensor(shape=shape)
        raise RuntimeError("invalid optional tensor marker in Rust preprocessing output")

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise RuntimeError("unexpected trailing bytes in Rust preprocessing output")


def _parse_binary_output(data: bytes) -> dict[str, Any]:
    reader = _BinaryReader(data)
    reader.read_magic()
    species = _read_binary_species(reader)
    families = {}
    for _ in range(reader.read_u64()):
        name = reader.read_string()
        families[name] = _read_binary_family(reader)
    reader.finish()
    return {"species": species, "families": families}


def _torchify_native_output(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {key: _torchify_native_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_torchify_native_output(item) for item in value]
    return value


def _canonical_torch_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _i64_numpy_view(value: Any):
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device="cpu", dtype=torch.long).contiguous().numpy()


def _species_index_arrays(
    species_helpers: dict[str, Any],
    *,
    S: int | None = None,
) -> tuple[int, Any, Any]:
    s = int(species_helpers["S"] if S is None else S)
    return (
        s,
        _i64_numpy_view(species_helpers["s_P_indexes"]),
        _i64_numpy_view(species_helpers["s_C12_indexes"]),
    )


def rust_species_parent_from_helpers(
    species_helpers: dict[str, Any],
    *,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    s, s_p_indexes, s_c12_indexes = _species_index_arrays(species_helpers)
    module = _load_native_module()
    parent = module.species_parent_from_indexes_torch(
        s,
        s_p_indexes,
        s_c12_indexes,
        torch.from_numpy,
    )
    return parent.to(dtype=dtype)


def rust_species_wave_topology(
    species_helpers: dict[str, Any],
    *,
    device: torch.device | str,
    S: int | None = None,
) -> dict[str, Any]:
    target_device = _canonical_torch_device(device)
    s, s_p_indexes, s_c12_indexes = _species_index_arrays(species_helpers, S=S)
    module = _load_native_module()
    topology = dict(
        module.species_wave_topology_torch(
            s,
            s_p_indexes,
            s_c12_indexes,
            torch.from_numpy,
        )
    )
    topology["S"] = int(topology["S"])
    topology["sp_child1"] = topology["sp_child1_cpu"].to(
        device=target_device,
        dtype=torch.int32,
    )
    topology["sp_child2"] = topology["sp_child2_cpu"].to(
        device=target_device,
        dtype=torch.int32,
    )
    topology["sp_parent"] = topology["sp_parent_cpu"].to(
        device=target_device,
        dtype=torch.int32,
    )
    topology["max_ancestor_depth"] = int(topology["max_ancestor_depth"])
    topology["compact_level_ptr"] = topology["compact_level_ptr"].to(
        device=target_device,
        dtype=torch.long,
    ).contiguous()
    topology["compact_level_parents"] = topology["compact_level_parents"].to(
        device=target_device,
        dtype=torch.int32,
    ).contiguous()
    topology["compact_level_child1"] = topology["compact_level_child1"].to(
        device=target_device,
        dtype=torch.int32,
    ).contiguous()
    topology["compact_level_child2"] = topology["compact_level_child2"].to(
        device=target_device,
        dtype=torch.int32,
    ).contiguous()
    return topology


def rust_uniform_ancestors_t_from_topology(
    species_helpers: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_device = _canonical_torch_device(device)
    s, s_p_indexes, s_c12_indexes = _species_index_arrays(species_helpers)
    module = _load_native_module()
    indices = module.uniform_ancestors_t_indices_torch(
        s,
        s_p_indexes,
        s_c12_indexes,
        torch.from_numpy,
    ).to(device=target_device, dtype=torch.long).contiguous()
    values = torch.ones(
        (int(indices.shape[1]),),
        dtype=dtype,
        device=target_device,
    )
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(
            indices,
            values,
            (s, s),
            device=target_device,
            dtype=dtype,
            is_coalesced=True,
        )


def _read_binary_species(reader: _BinaryReader) -> dict[str, Any]:
    s = reader.read_u64()
    species = {
        "S": s,
        "names": reader.read_strings(),
        "s_P_indexes": reader.read_i64_tensor(),
        "s_C12_indexes": reader.read_i64_tensor(),
        "unnorm_row_max": reader.read_f64_tensor(),
    }
    ancestors = reader.read_optional_f64_tensor(shape=(s, s))
    recipients = reader.read_optional_f64_tensor(shape=(s, s))
    if ancestors is not None:
        species["ancestors_dense"] = ancestors
    if recipients is not None:
        species["Recipients_mat"] = recipients
    species_name_to_index = {}
    for _ in range(reader.read_u64()):
        name = reader.read_string()
        species_name_to_index[name] = reader.read_u64()
    species["species_name_to_index"] = species_name_to_index
    return species


def _read_binary_family(reader: _BinaryReader) -> dict[str, Any]:
    ccp = {
        "split_counts": reader.read_i64_tensor(),
        "split_parents_sorted": reader.read_i64_tensor(),
        "split_leftrights_sorted": reader.read_i64_tensor(),
        "log_split_probs_sorted": reader.read_f64_tensor(),
        "num_segs_ge2": reader.read_i64(),
        "num_segs_eq1": reader.read_i64(),
        "end_rows_ge2": reader.read_i64(),
        "C": reader.read_i64(),
        "N_splits": reader.read_i64(),
        "root_clade_id": reader.read_i64(),
        "clade_leaf_labels": reader.read_sparse_strings(),
    }
    return {
        "ccp": ccp,
        "root_clade_id": reader.read_i64(),
        "leaf_row_index": reader.read_i64_tensor(),
        "leaf_col_index": reader.read_i64_tensor(),
    }


class RustPreprocessExtension:
    """Raw Python wrapper around the native Rust preprocessing module."""

    def __init__(
        self,
        *,
        cargo_manifest: str | Path = _PREPROCESS_MANIFEST,
    ) -> None:
        self.cargo_manifest = Path(cargo_manifest)
        self._from_numpy = torch.from_numpy

    def preprocess_multiple_families(
        self,
        species_path: str,
        families: dict[str, list[str]],
        *,
        leaf_species_maps: dict[str, dict[str, str]] | None = None,
        include_details: bool = True,
        include_species_matrices: bool = True,
        include_debug_details: bool = False,
        include_scheduler_details: bool = False,
        include_legacy_ccp_details: bool = False,
        num_threads: int = 0,
    ) -> dict[str, Any]:
        del (
            include_details,
            include_debug_details,
            include_scheduler_details,
            include_legacy_ccp_details,
        )
        request = {
            "species_path": str(species_path),
            "families": {
                str(name): [str(path) for path in paths]
                for name, paths in families.items()
            },
            "leaf_species_maps": leaf_species_maps or {},
            "include_species_matrices": bool(include_species_matrices),
            "num_threads": int(num_threads),
        }
        module = _load_native_module(self.cargo_manifest)
        return module.preprocess_request_torch(json.dumps(request), self._from_numpy)

    def preprocess_dataset(
        self,
        species_path: str,
        families: dict[str, list[str]],
        *,
        leaf_species_maps: dict[str, dict[str, str]] | None = None,
        include_species_matrices: bool = True,
        num_threads: int = 0,
    ) -> "RustPreprocessedDataset":
        request = {
            "species_path": str(species_path),
            "families": {
                str(name): [str(path) for path in paths]
                for name, paths in families.items()
            },
            "family_order": [str(name) for name in families],
            "leaf_species_maps": leaf_species_maps or {},
            "include_species_matrices": bool(include_species_matrices),
            "num_threads": int(num_threads),
        }
        module = _load_native_module(self.cargo_manifest)
        native = module.preprocess_dataset(json.dumps(request))
        return RustPreprocessedDataset(native, self._from_numpy)


class RustPreprocessedDataset:
    """Native Rust preprocessing result retained for fused chunk/layout planning."""

    def __init__(self, native, from_numpy) -> None:
        self._native = native
        self._from_numpy = from_numpy

    def family_counts(self) -> dict[str, Any]:
        return dict(self._native.family_counts())

    def family_basic_counts(self) -> dict[str, Any]:
        return dict(self._native.family_basic_counts())

    def to_torch(self) -> dict[str, Any]:
        return self._native.to_torch(self._from_numpy)

    def to_torch_compact(self) -> dict[str, Any]:
        compact = getattr(self._native, "to_torch_compact", None)
        if not callable(compact):
            return self.to_torch()
        return compact(self._from_numpy)

    def build_chunked_layouts(
        self,
        *,
        family_chunk_size: int,
        clade_budget: int | None,
        batch_packing: str,
        max_wave_size: int | None,
        max_root_wave_size: int | None,
        max_dts_partial_rows: int | None = None,
        dtype: str = "float32",
        num_threads: int = 0,
    ) -> list[dict[str, Any]]:
        request = {
            "family_chunk_size": int(family_chunk_size),
            "clade_budget": None if clade_budget is None else int(clade_budget),
            "batch_packing": str(batch_packing),
            "max_wave_size": None if max_wave_size is None else int(max_wave_size),
            "max_root_wave_size": (
                None if max_root_wave_size is None else int(max_root_wave_size)
            ),
            "max_dts_partial_rows": (
                None if max_dts_partial_rows is None else int(max_dts_partial_rows)
            ),
            "dtype": str(dtype),
            "num_threads": int(num_threads),
        }
        return list(
            self._native.build_chunked_layouts_torch(
                json.dumps(request),
                self._from_numpy,
            )
        )


class RustPreprocessSubprocessExtension:
    """Raw Python wrapper around the Rust preprocessing CLI."""

    def __init__(
        self,
        *,
        cargo_manifest: str | Path = _PREPROCESS_MANIFEST,
        preprocess_binary: str | Path | None = None,
    ) -> None:
        self.cargo_manifest = Path(cargo_manifest)
        self.preprocess_binary = preprocess_binary

    def preprocess_multiple_families(
        self,
        species_path: str,
        families: dict[str, list[str]],
        *,
        leaf_species_maps: dict[str, dict[str, str]] | None = None,
        include_details: bool = True,
        include_species_matrices: bool = True,
        include_debug_details: bool = False,
        include_scheduler_details: bool = False,
        include_legacy_ccp_details: bool = False,
        num_threads: int = 0,
    ) -> dict[str, Any]:
        del (
            include_details,
            include_debug_details,
            include_scheduler_details,
            include_legacy_ccp_details,
        )
        request = {
            "species_path": str(species_path),
            "families": {
                str(name): [str(path) for path in paths]
                for name, paths in families.items()
            },
            "leaf_species_maps": leaf_species_maps or {},
            "include_species_matrices": bool(include_species_matrices),
            "num_threads": int(num_threads),
        }
        command = _preprocess_command(
            cargo_manifest=self.cargo_manifest,
            preprocess_binary=self.preprocess_binary,
        )
        cwd = str(_REPO_ROOT) if _is_cargo_fallback_command(command) else None
        with tempfile.TemporaryDirectory(prefix="gpurec-preprocess-") as tmp:
            request_path = Path(tmp) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            full_command = command + ["--binary-output", str(request_path)]
            try:
                result = subprocess.run(
                    full_command,
                    check=False,
                    capture_output=True,
                    cwd=cwd,
                    timeout=_PREPROCESS_RUN_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "gpurec Rust preprocessing command timed out after "
                    f"{_PREPROCESS_RUN_TIMEOUT_SECONDS} seconds; "
                    f"command: {_command_text(full_command)}"
                ) from exc
        if result.returncode != 0:
            details = [
                "gpurec Rust preprocessing command failed",
                f"exit code {result.returncode}",
                f"command: {_command_text(full_command)}",
            ]
            stderr = result.stderr.decode("utf-8", errors="replace")
            stdout = result.stdout.decode("utf-8", errors="replace")
            if stderr:
                details.append(f"stderr: {stderr.strip()}")
            if result.stdout:
                details.append(f"stdout: {stdout.strip()}")
            raise RuntimeError("; ".join(details))
        return _parse_binary_output(result.stdout)
