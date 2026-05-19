import hashlib
import math
import os
import pickle
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from .preprocess_cpp import _load_extension as _load_species_gene_ext
from .species import uniform_ancestors_t_from_topology


def _resolve_family_path(text: str, base_dir: Path) -> str:
    value = text.strip().strip('"').strip("'")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def parse_alerax_mapping_file(path: str | os.PathLike) -> dict[str, str]:
    """Parse AleRax ``species:gene1;gene2`` mapping files."""
    mapping: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid AleRax mapping line in {path}: {raw!r}")
        species, genes = line.split(":", 1)
        species = species.strip()
        for gene in genes.split(";"):
            gene = gene.strip()
            if gene:
                previous = mapping.get(gene)
                if previous is not None:
                    raise ValueError(
                        f"duplicate AleRax mapping for gene {gene!r} in {path}: "
                        f"{previous!r} and {species!r}"
                    )
                mapping[gene] = species
    return mapping


def _normalize_family_index(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number)
    raise ValueError(f"{name} must be an integer")


def normalize_family_selection(
    start: int,
    max_families: int | None,
) -> tuple[int, int | None]:
    start_int = _normalize_family_index("start", start)
    if start_int < 0:
        raise ValueError("start must be non-negative")
    if max_families is None:
        return start_int, None
    max_int = _normalize_family_index("max_families", max_families)
    if max_int <= 0:
        raise ValueError("max_families must be positive when provided")
    return start_int, max_int


def parse_alerax_family_file(
    families_file: str | os.PathLike,
    *,
    start: int = 0,
    max_families: int | None = None,
) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    """Parse an AleRax families file.

    Returns family names, per-family tree-distribution paths, and optional
    leaf-to-species maps. ``starting_gene_tree`` and ``gene_tree`` entries are
    both accepted because AleRax uses both forms in local fixtures.
    """
    start, max_families = normalize_family_selection(start, max_families)

    base_dir = Path(families_file).resolve().parent
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    seen_names: set[str] = set()

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        if not current["tree_paths"]:
            raise ValueError(f"family {current['name']!r} has no gene tree path")
        if current["name"] in seen_names:
            raise ValueError(
                f"duplicate AleRax family name {current['name']!r} in {families_file}"
            )
        seen_names.add(current["name"])
        records.append(current)
        current = None

    allowed_keys = {"starting_gene_tree", "gene_tree", "mapping"}

    for raw in Path(families_file).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if line.startswith("-"):
            finish()
            name = line[1:].strip()
            if not name:
                raise ValueError(f"empty family name in {families_file}")
            current = {"name": name, "tree_paths": [], "mapping": None}
            continue
        if current is None:
            continue
        if "=" not in line:
            raise ValueError(
                f"invalid AleRax family line for family {current['name']!r}: {raw!r}"
            )
        key, value = [part.strip() for part in line.split("=", 1)]
        if key not in allowed_keys:
            raise ValueError(
                f"unknown AleRax family key {key!r} in family {current['name']!r}"
            )
        if not value:
            raise ValueError(
                f"empty AleRax family value for key {key!r} in family {current['name']!r}"
            )
        if key in {"starting_gene_tree", "gene_tree"}:
            current["tree_paths"].append(_resolve_family_path(value, base_dir))
        elif key == "mapping":
            current["mapping"] = _resolve_family_path(value, base_dir)
    finish()

    stop = None if max_families is None else start + max_families
    selected = records[start:stop]
    if not selected:
        raise ValueError(
            f"empty family selection: start={start}, max_families={max_families}, "
            f"available={len(records)}"
        )

    names = [str(rec["name"]) for rec in selected]
    tree_paths = [list(rec["tree_paths"]) for rec in selected]
    leaf_maps = [
        parse_alerax_mapping_file(rec["mapping"]) if rec["mapping"] else {}
        for rec in selected
    ]
    return names, tree_paths, leaf_maps


def _normalize_family_tree_paths(
    gene_tree_paths: Sequence[str | os.PathLike | Sequence[str | os.PathLike]],
) -> list[list[str]]:
    normalized: list[list[str]] = []
    for entry in gene_tree_paths:
        if isinstance(entry, (str, os.PathLike)):
            paths = [str(entry)]
        else:
            paths = [str(path) for path in entry]
        if not paths:
            raise ValueError("each family must have at least one tree path")
        normalized.append(paths)
    return normalized


def _load_preprocess_cache(
    path: str | os.PathLike,
    *,
    label: str,
    required_keys: Sequence[str],
    validator: Callable[[dict[str, Any], Path, str], None] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"could not safely load {label} preprocess cache {path}; "
            "delete it or rerun with refresh_preprocess_cache=True"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} preprocess cache {path} must contain a dictionary")
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise RuntimeError(
            f"{label} preprocess cache {path} is missing key(s): {', '.join(missing)}"
        )
    if validator is not None:
        validator(payload, path, label)
    return payload


def _invalid_preprocess_cache(path: Path, label: str, reason: str) -> RuntimeError:
    return RuntimeError(
        f"{label} preprocess cache {path} is invalid: {reason}; "
        "delete it or rerun with refresh_preprocess_cache=True"
    )


def _cache_int(
    payload: dict[str, Any],
    key: str,
    *,
    path: Path,
    label: str,
    minimum: int | None = None,
) -> int:
    if key not in payload:
        raise _invalid_preprocess_cache(path, label, f"missing key {key!r}")
    value = payload[key]
    if isinstance(value, bool):
        raise _invalid_preprocess_cache(path, label, f"{key!r} must be an integer")
    if torch.is_tensor(value):
        if value.ndim != 0 or value.dtype == torch.bool:
            raise _invalid_preprocess_cache(path, label, f"{key!r} must be an integer")
        value = value.item()
    if isinstance(value, float) and not value.is_integer():
        raise _invalid_preprocess_cache(path, label, f"{key!r} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} must be an integer",
        ) from exc
    if minimum is not None and parsed < minimum:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} must be at least {minimum}",
        )
    return parsed


def _cache_tensor(
    payload: dict[str, Any],
    key: str,
    *,
    path: Path,
    label: str,
    dtype: torch.dtype | None = None,
    floating: bool = False,
    ndim: int | None = None,
    length: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if key not in payload:
        raise _invalid_preprocess_cache(path, label, f"missing key {key!r}")
    tensor = payload[key]
    if not torch.is_tensor(tensor):
        raise _invalid_preprocess_cache(path, label, f"{key!r} must be a tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} must have dtype {dtype}, got {tensor.dtype}",
        )
    if floating and not tensor.dtype.is_floating_point:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} must be a floating-point tensor",
        )
    if ndim is not None and tensor.ndim != ndim:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} must have {ndim} dimension(s), got {tensor.ndim}",
        )
    if length is not None and tensor.numel() != length:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} has length {tensor.numel()} but expected {length}",
        )
    if shape is not None and tuple(tensor.shape) != shape:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"{key!r} has shape {tuple(tensor.shape)} but expected {shape}",
        )
    return tensor


def _validate_species_preprocess_cache(
    payload: dict[str, Any],
    path: Path,
    label: str,
) -> None:
    S = _cache_int(payload, "S", path=path, label=label, minimum=1)
    names = payload.get("names")
    if not isinstance(names, (list, tuple)) or len(names) != S:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"'names' must contain {S} species labels",
        )
    _cache_tensor(
        payload,
        "s_P_indexes",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
    )
    _cache_tensor(
        payload,
        "s_C12_indexes",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
    )
    if payload["s_P_indexes"].numel() != payload["s_C12_indexes"].numel():
        raise _invalid_preprocess_cache(
            path,
            label,
            "'s_P_indexes' and 's_C12_indexes' must have the same length",
        )
    _cache_tensor(
        payload,
        "unnorm_row_max",
        path=path,
        label=label,
        floating=True,
        ndim=1,
        length=S,
    )
    if "Recipients_mat" in payload:
        _cache_tensor(
            payload,
            "Recipients_mat",
            path=path,
            label=label,
            floating=True,
            shape=(S, S),
        )
    species_name_to_index = payload.get("species_name_to_index")
    if not isinstance(species_name_to_index, dict):
        raise _invalid_preprocess_cache(
            path,
            label,
            "'species_name_to_index' must be a dictionary",
        )


def _validate_family_preprocess_cache(
    payload: dict[str, Any],
    path: Path,
    label: str,
) -> None:
    ccp = payload.get("ccp")
    if not isinstance(ccp, dict):
        raise _invalid_preprocess_cache(path, label, "'ccp' must be a dictionary")
    C = _cache_int(ccp, "C", path=path, label=label, minimum=1)
    N = _cache_int(ccp, "N_splits", path=path, label=label, minimum=0)
    root = _cache_int(ccp, "root_clade_id", path=path, label=label, minimum=0)
    if root >= C:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"'root_clade_id'={root} outside clade range [0, {C})",
        )
    num_eq1 = _cache_int(ccp, "num_segs_eq1", path=path, label=label, minimum=0)
    end_rows_ge2 = _cache_int(ccp, "end_rows_ge2", path=path, label=label, minimum=0)
    if num_eq1 + end_rows_ge2 != N:
        raise _invalid_preprocess_cache(
            path,
            label,
            "'num_segs_eq1' + 'end_rows_ge2' must equal 'N_splits'",
        )
    _cache_tensor(
        ccp,
        "split_counts",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
        length=C,
    )
    _cache_tensor(
        ccp,
        "split_parents_sorted",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
        length=N,
    )
    _cache_tensor(
        ccp,
        "split_leftrights_sorted",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
        length=2 * N,
    )
    _cache_tensor(
        ccp,
        "log_split_probs_sorted",
        path=path,
        label=label,
        floating=True,
        ndim=1,
        length=N,
    )
    clade_leaf_labels = ccp.get("clade_leaf_labels")
    if not isinstance(clade_leaf_labels, (list, tuple)) or len(clade_leaf_labels) != C:
        raise _invalid_preprocess_cache(
            path,
            label,
            f"'clade_leaf_labels' must contain {C} labels",
        )

    leaf_rows = _cache_tensor(
        payload,
        "leaf_row_index",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
    )
    leaf_cols = _cache_tensor(
        payload,
        "leaf_col_index",
        path=path,
        label=label,
        dtype=torch.long,
        ndim=1,
        length=int(leaf_rows.numel()),
    )
    if leaf_rows.numel() > 0:
        if int(leaf_rows.min().item()) < 0 or int(leaf_rows.max().item()) >= C:
            raise _invalid_preprocess_cache(
                path,
                label,
                "'leaf_row_index' contains clade rows outside the CCP range",
            )
    if leaf_cols.numel() > 0 and int(leaf_cols.min().item()) < 0:
        raise _invalid_preprocess_cache(
            path,
            label,
            "'leaf_col_index' contains negative species indexes",
        )


class GeneDataset:
    @staticmethod
    def _drop_unused_family_details(raw: dict[str, Any]) -> dict[str, Any]:
        ccp = raw.get("ccp")
        if isinstance(ccp, dict):
            ccp.pop("clade_leaves", None)
            ccp.pop("clade_is_leaf", None)
        return raw

    def __init__(
        self,
        species_tree_path,
        gene_tree_paths,
        genewise, # whether to have a different theta for each gene family
        specieswise, # no need for genewise: it only appear when collating or gradient steps
        dtype=torch.float32,
        device=None,
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.genewise = genewise
        self.specieswise = specieswise
        self.device = device
        self.dtype = dtype
        ext = _load_species_gene_ext()

        family_tree_paths = _normalize_family_tree_paths(gene_tree_paths)
        if family_names is None:
            family_names = [f"family_{i:06d}" for i in range(len(family_tree_paths))]
        else:
            family_names = [str(name) for name in family_names]
        if len(family_names) != len(family_tree_paths):
            raise ValueError("family_names must match gene_tree_paths length")
        if leaf_species_maps is None:
            leaf_species_maps = [{} for _ in family_tree_paths]
        else:
            leaf_species_maps = [dict(m) for m in leaf_species_maps]
        if len(leaf_species_maps) != len(family_tree_paths):
            raise ValueError("leaf_species_maps must match gene_tree_paths length")

        if preprocess_cache_dir is not None:
            self.species_helpers, raw_by_family = self._preprocess_with_cache(
                ext,
                species_tree_path,
                family_tree_paths,
                family_names,
                leaf_species_maps,
                preprocess_cache_dir=preprocess_cache_dir,
                refresh=refresh_preprocess_cache,
            )
        else:
            families_input = {
                name: paths
                for name, paths in zip(family_names, family_tree_paths)
            }
            leaf_species_input = {
                name: mapping
                for name, mapping in zip(family_names, leaf_species_maps)
                if mapping
            }
            raw_all = ext.preprocess_multiple_families(
                str(species_tree_path),
                families_input,
                leaf_species_maps=leaf_species_input,
                include_details=True,
                include_species_matrices=False,
            )
            raw_by_family = {
                name: self._drop_unused_family_details(raw)
                for name, raw in raw_all['families'].items()
            }
            self.species_helpers = raw_all['species']
        if "Recipients_mat" in self.species_helpers:
            self.unnorm_row_max = torch.log2(self.species_helpers["Recipients_mat"]).max(dim=-1).values
        else:
            self.unnorm_row_max = self.species_helpers["unnorm_row_max"]
        self.S = int(self.species_helpers['S'])

        _INV_LN2 = 1.0 / math.log(2.0)
        families = []
        for family_name in family_names:
            raw = raw_by_family[family_name]
            ccp = raw['ccp']
            clade_leaf_labels = list(ccp.pop('clade_leaf_labels', []))
            # Convert log_split_probs from ln (C++ output) to log2
            ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
            C = int(ccp['C'])
            if len(clade_leaf_labels) != C:
                clade_leaf_labels = [""] * C
            families.append({
                'ccp_helpers': ccp,
                'root_clade_id': int(ccp['root_clade_id']),
                'leaf_row_index': raw['leaf_row_index'],
                'leaf_col_index': raw['leaf_col_index'],
                'C': C,
                'N_splits': int(ccp['N_splits']),
                'log_split_probs': ccp['log_split_probs_sorted'],
                'clade_leaf_labels': clade_leaf_labels,
            })
        # stored on CPU. Only move when computing likelihood and optimizing.
        self.families = families
        self.family_names = list(family_names)
        self.gene_tree_paths = family_tree_paths
        self.leaf_species_maps = [dict(m) for m in leaf_species_maps]
        self.species_tree_path = species_tree_path

        self.num_families = len(families)

    @staticmethod
    def _hash_file(path: str | os.PathLike) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _preprocess_with_cache(
        cls,
        ext,
        species_tree_path,
        gene_tree_paths,
        family_names,
        leaf_species_maps,
        *,
        preprocess_cache_dir: str | os.PathLike,
        refresh: bool,
    ):
        cache_dir = Path(preprocess_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        version = "light-v5:compact-species:alerax-maps:leaf-labels"
        species_hash = cls._hash_file(species_tree_path)
        species_key = hashlib.sha256(
            f"{version}:species:{species_hash}".encode("utf-8")
        ).hexdigest()
        species_cache = cache_dir / f"species-{species_key}.pt"

        species_helpers = None
        if species_cache.exists() and not refresh:
            species_helpers = _load_preprocess_cache(
                species_cache,
                label="species",
                required_keys=("S",),
                validator=_validate_species_preprocess_cache,
            )

        raw_by_family = {}
        missing = {}
        missing_maps = {}
        family_cache_paths = {}
        for name, paths, leaf_map in zip(family_names, gene_tree_paths, leaf_species_maps):
            h = hashlib.sha256()
            for path in paths:
                h.update(cls._hash_file(path).encode("utf-8"))
                h.update(b"\0")
            for gene, species in sorted(leaf_map.items()):
                h.update(gene.encode("utf-8"))
                h.update(b"\t")
                h.update(species.encode("utf-8"))
                h.update(b"\0")
            gene_hash = h.hexdigest()
            family_key = hashlib.sha256(
                f"{version}:family:{species_hash}:{gene_hash}".encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / f"family-{family_key}.pt"
            family_cache_paths[name] = cache_path
            if cache_path.exists() and not refresh:
                raw_by_family[name] = cls._drop_unused_family_details(
                    _load_preprocess_cache(
                        cache_path,
                        label=f"family {name!r}",
                        required_keys=("ccp", "leaf_row_index", "leaf_col_index"),
                        validator=_validate_family_preprocess_cache,
                    )
                )
            else:
                missing[name] = paths
                if leaf_map:
                    missing_maps[name] = leaf_map

        if missing:
            raw_all = ext.preprocess_multiple_families(
                str(species_tree_path),
                missing,
                leaf_species_maps=missing_maps,
                include_details=True,
                include_species_matrices=False,
            )
            if species_helpers is None:
                species_helpers = raw_all["species"]
                torch.save(species_helpers, species_cache)

            for name, raw in raw_all["families"].items():
                raw = cls._drop_unused_family_details(raw)
                raw_by_family[name] = raw
                torch.save(raw, family_cache_paths[name])

        if species_helpers is None:
            raw_species = ext.preprocess_multiple_families(
                str(species_tree_path),
                {},
                include_species_matrices=False,
            )
            species_helpers = raw_species["species"]
            torch.save(species_helpers, species_cache)

        return species_helpers, raw_by_family
    
    @staticmethod
    def _move_tensor(t: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)

    def _species_helpers_for_mode(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        species_helpers = {
            k: (self._move_tensor(v, device=device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in self.species_helpers.items()
            if k not in {'Recipients_mat', 'ancestors_dense'}
        }

        ancestors_T = uniform_ancestors_t_from_topology(
            self.species_helpers,
            device=device,
            dtype=dtype,
        )
        return species_helpers, ancestors_T
