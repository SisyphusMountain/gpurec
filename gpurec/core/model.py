import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from .preprocess_rust import RustPreprocessExtension
from .species import uniform_ancestors_t_from_topology


_DEFAULT_PREPROCESS_UNCACHED_BATCH_SIZE = 1024
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


def normalize_family_tree_paths(
    gene_tree_paths: Iterable[str | os.PathLike | Iterable[str | os.PathLike]],
) -> list[list[str]]:
    """Normalize public gene-tree inputs into one path list per family.

    Each path is later parsed by the retained Rust simple-Newick reader.  A file
    may contain one or more semicolon-delimited gene-tree records, with an
    optional terminal semicolon on the final record; every record from every
    path in one family is amalgamated into that family's CCP.
    """
    if isinstance(gene_tree_paths, (str, os.PathLike)):
        raise ValueError(
            "gene_trees must be a non-empty sequence of paths, not a single path"
        )
    normalized: list[list[str]] = []
    try:
        entries = iter(gene_tree_paths)
    except TypeError as exc:
        raise ValueError("gene_trees must be a non-empty sequence of paths") from exc
    for entry in entries:
        if isinstance(entry, (str, os.PathLike)):
            paths = [str(entry)]
        else:
            try:
                paths = [str(path) for path in entry]
            except TypeError as exc:
                raise ValueError(
                    "each gene_trees entry must be a path or a non-empty "
                    "sequence of paths"
                ) from exc
        if not paths:
            raise ValueError("each family must have at least one gene_trees path")
        normalized.append(paths)
    if not normalized:
        raise ValueError("gene_trees must not be empty")
    return normalized


def normalize_family_inputs(
    gene_tree_paths: Iterable[str | os.PathLike | Iterable[str | os.PathLike]],
    family_names: Sequence[str] | None = None,
    leaf_species_maps: Sequence[dict[str, str]] | None = None,
) -> tuple[list[list[str]], list[str], list[dict[str, str]]]:
    """Normalize family paths plus optional names and leaf-species maps."""
    family_tree_paths = normalize_family_tree_paths(gene_tree_paths)
    if family_names is None:
        normalized_names = [
            f"family_{i:06d}" for i in range(len(family_tree_paths))
        ]
    else:
        normalized_names = [str(name) for name in family_names]
    if len(normalized_names) != len(family_tree_paths):
        raise ValueError("family_names must match gene_tree_paths length")
    seen_family_names: set[str] = set()
    for name in normalized_names:
        if name in seen_family_names:
            raise ValueError(f"duplicate family name {name!r} in family_names")
        seen_family_names.add(name)
    if leaf_species_maps is None:
        normalized_maps = [{} for _ in family_tree_paths]
    else:
        normalized_maps = [dict(m) for m in leaf_species_maps]
    if len(normalized_maps) != len(family_tree_paths):
        raise ValueError("leaf_species_maps must match gene_tree_paths length")
    return family_tree_paths, normalized_names, normalized_maps


def _normalize_preprocess_cpu_cores(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("preprocess_cpu_cores must be a positive integer")
    if isinstance(value, Integral):
        cores = int(value)
    elif isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError("preprocess_cpu_cores must be a positive integer")
        cores = int(number)
    else:
        raise ValueError("preprocess_cpu_cores must be a positive integer")
    if cores <= 0:
        raise ValueError("preprocess_cpu_cores must be a positive integer")
    return cores


def _load_preprocess_extension():
    return RustPreprocessExtension()


class GeneDataset:
    """Preprocessed reconciliation dataset backed by the Rust parser.

    The species path must contain one rooted binary tree in the supported simple
    Newick subset.  Gene-tree files may contain one or more semicolon-delimited
    records, with an optional terminal semicolon on the final record; records
    listed for one family are pooled into that family's CCP.
    """

    @staticmethod
    def _drop_unused_family_details(raw: dict[str, Any]) -> dict[str, Any]:
        ccp = raw.get("ccp")
        if isinstance(ccp, dict):
            ccp.pop("clade_leaves", None)
            ccp.pop("clade_is_leaf", None)
            ccp.pop("inclusion_children", None)
            ccp.pop("inclusion_parents", None)
            ccp.pop("ubiquitous_clade_id", None)
        return raw

    @staticmethod
    def _family_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
        _INV_LN2 = 1.0 / math.log(2.0)
        ccp = raw['ccp']
        clade_leaf_labels = list(ccp.pop('clade_leaf_labels', []))
        # Convert log_split_probs from ln (Rust preprocessing output) to log2.
        ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
        C = int(ccp['C'])
        if len(clade_leaf_labels) != C:
            clade_leaf_labels = [""] * C
        return {
            'ccp_helpers': ccp,
            'root_clade_id': int(ccp['root_clade_id']),
            'leaf_row_index': raw['leaf_row_index'],
            'leaf_col_index': raw['leaf_col_index'],
            'C': C,
            'N_splits': int(ccp['N_splits']),
            'log_split_probs': ccp['log_split_probs_sorted'],
            'clade_leaf_labels': clade_leaf_labels,
        }

    def __init__(
        self,
        species_tree_path,
        gene_tree_paths,
        genewise, # whether to have a different theta for each gene family
        specieswise, # no need for genewise: it only appear when collating or gradient steps
        dtype=torch.float32,
        device=None,
        preprocess_cpu_cores: int | None = None,
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
        _preprocess_progress: Callable[..., None] | None = None,
        _uncached_preprocess_batch_size: int = _DEFAULT_PREPROCESS_UNCACHED_BATCH_SIZE,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocess_cpu_cores = _normalize_preprocess_cpu_cores(preprocess_cpu_cores)

        self.genewise = genewise
        self.specieswise = specieswise
        self.device = device
        self.dtype = dtype
        family_tree_paths, family_names, leaf_species_maps = normalize_family_inputs(
            gene_tree_paths,
            family_names,
            leaf_species_maps,
        )

        ext = _load_preprocess_extension()
        # Private benchmark hook: this intentionally stays out of the documented
        # dataset API because it observes setup stages without changing model
        # inputs, outputs, or preprocessing semantics.
        preprocess_progress = _preprocess_progress or (lambda *_args, **_kwargs: None)
        preprocess_progress(
            "init_start",
            families=len(family_tree_paths),
            preprocess_cpu_cores=preprocess_cpu_cores,
        )
        self.species_helpers, families = self._preprocess_without_cache(
            ext,
            species_tree_path,
            family_tree_paths,
            family_names,
            leaf_species_maps,
            progress=preprocess_progress,
            preprocess_cpu_cores=preprocess_cpu_cores,
            _batch_size=_uncached_preprocess_batch_size,
            _materialize_families=True,
        )

        if "Recipients_mat" in self.species_helpers:
            self.unnorm_row_max = torch.log2(self.species_helpers["Recipients_mat"]).max(dim=-1).values
        else:
            self.unnorm_row_max = self.species_helpers["unnorm_row_max"]
        self.S = int(self.species_helpers['S'])

        # stored on CPU. Only move when computing likelihood and optimizing.
        self.families = families
        self.family_names = list(family_names)
        self.gene_tree_paths = family_tree_paths
        self.leaf_species_maps = [dict(m) for m in leaf_species_maps]
        self.species_tree_path = species_tree_path
        self._rust_preprocessed = None
        self._preprocess_cpu_cores = preprocess_cpu_cores
        self._compact_families = False

        self.num_families = len(families)

    @classmethod
    def from_retained_preprocess(
        cls,
        species_tree_path,
        gene_tree_paths,
        genewise,
        specieswise,
        dtype=torch.float32,
        device=None,
        preprocess_cpu_cores: int | None = None,
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
        compact_families: bool = False,
    ) -> "GeneDataset":
        """Build a dataset while retaining native Rust preprocessing output."""
        preprocess_cpu_cores = _normalize_preprocess_cpu_cores(preprocess_cpu_cores)
        family_tree_paths, family_names, leaf_species_maps = normalize_family_inputs(
            gene_tree_paths,
            family_names,
            leaf_species_maps,
        )

        ext = _load_preprocess_extension()
        preprocess_dataset = getattr(ext, "preprocess_dataset", None)
        if not callable(preprocess_dataset):
            return cls(
                species_tree_path=species_tree_path,
                gene_tree_paths=family_tree_paths,
                genewise=genewise,
                specieswise=specieswise,
                dtype=dtype,
                device=device,
                preprocess_cpu_cores=preprocess_cpu_cores,
                family_names=family_names,
                leaf_species_maps=leaf_species_maps,
            )

        families_input = {
            name: paths
            for name, paths in zip(family_names, family_tree_paths)
        }
        leaf_species_input = {
            name: mapping
            for name, mapping in zip(family_names, leaf_species_maps)
            if mapping
        }
        rust_preprocessed = preprocess_dataset(
            str(species_tree_path),
            families_input,
            leaf_species_maps=leaf_species_input,
            include_species_matrices=False,
            num_threads=0 if preprocess_cpu_cores is None else preprocess_cpu_cores,
        )
        use_compact_families = compact_families and callable(
            getattr(getattr(rust_preprocessed, "_native", None), "to_torch_compact", None)
        )
        return cls._from_preprocessed_raw(
            raw=(
                rust_preprocessed.to_torch_compact()
                if use_compact_families
                else rust_preprocessed.to_torch()
            ),
            species_tree_path=species_tree_path,
            gene_tree_paths=family_tree_paths,
            genewise=genewise,
            specieswise=specieswise,
            dtype=dtype,
            device=device,
            family_names=family_names,
            leaf_species_maps=leaf_species_maps,
            rust_preprocessed=rust_preprocessed,
            preprocess_cpu_cores=preprocess_cpu_cores,
            compact_families=use_compact_families,
        )

    @classmethod
    def _preprocess_without_cache(
        cls,
        ext,
        species_tree_path,
        gene_tree_paths,
        family_names,
        leaf_species_maps,
        *,
        progress: Callable[..., None] | None = None,
        preprocess_cpu_cores: int | None = None,
        _batch_size: int = _DEFAULT_PREPROCESS_UNCACHED_BATCH_SIZE,
        _materialize_families: bool = False,
    ):
        progress = progress or (lambda *_args, **_kwargs: None)
        preprocess_cpu_cores = _normalize_preprocess_cpu_cores(preprocess_cpu_cores)
        if (
            isinstance(_batch_size, bool)
            or not isinstance(_batch_size, Integral)
            or int(_batch_size) <= 0
        ):
            raise ValueError("_batch_size must be a positive integer")
        batch_size = int(_batch_size)
        families_input = {
            name: paths
            for name, paths in zip(family_names, gene_tree_paths)
        }
        leaf_species_input = {
            name: mapping
            for name, mapping in zip(family_names, leaf_species_maps)
            if mapping
        }
        family_items = list(families_input.items())
        batches = math.ceil(len(family_items) / batch_size) if family_items else 0
        progress(
            "uncached_preprocess_start",
            families=len(families_input),
            families_with_leaf_maps=len(leaf_species_input),
            batch_size=batch_size,
            batches=batches,
            preprocess_cpu_cores=preprocess_cpu_cores,
        )

        if not family_items:
            raw_all = ext.preprocess_multiple_families(
                str(species_tree_path),
                {},
                include_details=True,
                include_species_matrices=False,
                include_debug_details=False,
                include_scheduler_details=False,
                include_legacy_ccp_details=False,
                num_threads=0 if preprocess_cpu_cores is None else preprocess_cpu_cores,
            )
            progress("uncached_preprocess_done", families=0, batches=0)
            return raw_all["species"], [] if _materialize_families else {}

        species_helpers = None
        families = []
        raw_by_family = {}
        built_families = 0
        for batch_idx, start in enumerate(range(0, len(family_items), batch_size)):
            batch_items = family_items[start:start + batch_size]
            batch_input = dict(batch_items)
            batch_maps = {
                name: leaf_species_input[name]
                for name, _paths in batch_items
                if name in leaf_species_input
            }
            batch_names = list(batch_input)
            progress(
                "uncached_preprocess_batch_start",
                batch_idx=batch_idx,
                batches=batches,
                families=len(batch_input),
                families_with_leaf_maps=len(batch_maps),
                first_family=batch_names[0],
                last_family=batch_names[-1],
            )
            raw_all = ext.preprocess_multiple_families(
                str(species_tree_path),
                batch_input,
                leaf_species_maps=batch_maps,
                include_details=True,
                include_species_matrices=False,
                include_debug_details=False,
                include_scheduler_details=False,
                include_legacy_ccp_details=False,
                num_threads=0 if preprocess_cpu_cores is None else preprocess_cpu_cores,
            )
            raw_families = raw_all["families"]
            missing_outputs = set(batch_input) - set(raw_families)
            if missing_outputs:
                raise RuntimeError(
                    "preprocess_multiple_families did not return family result(s): "
                    + ", ".join(sorted(missing_outputs))
                )
            if species_helpers is None:
                species_helpers = raw_all["species"]
            for name in batch_input:
                raw = raw_families[name]
                raw_by_family[name] = cls._drop_unused_family_details(raw)
                if _materialize_families:
                    families.append(cls._family_from_raw(raw_by_family.pop(name)))
                built_families += 1
            progress(
                "uncached_preprocess_batch_done",
                batch_idx=batch_idx,
                batches=batches,
                families=len(raw_families),
                total_built=built_families,
            )

        progress(
            "uncached_preprocess_done",
            families=built_families,
            batches=batches,
        )
        if _materialize_families:
            return species_helpers, families
        return species_helpers, raw_by_family

    @classmethod
    def _from_preprocessed_raw(
        cls,
        *,
        raw: dict[str, Any],
        species_tree_path,
        gene_tree_paths,
        genewise,
        specieswise,
        dtype=torch.float32,
        device=None,
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
        rust_preprocessed: Any | None = None,
        preprocess_cpu_cores: int | None = None,
        compact_families: bool = False,
    ) -> "GeneDataset":
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        family_tree_paths, family_names, leaf_species_maps = normalize_family_inputs(
            gene_tree_paths,
            family_names,
            leaf_species_maps,
        )

        raw_families = raw["families"]
        missing = [name for name in family_names if name not in raw_families]
        if missing:
            raise RuntimeError(
                "preprocess_multiple_families did not return family result(s): "
                + ", ".join(missing)
            )

        self = cls.__new__(cls)
        self.genewise = genewise
        self.specieswise = specieswise
        self.device = device
        self.dtype = dtype
        self.species_helpers = raw["species"]
        if "Recipients_mat" in self.species_helpers:
            self.unnorm_row_max = torch.log2(
                self.species_helpers["Recipients_mat"]
            ).max(dim=-1).values
        else:
            self.unnorm_row_max = self.species_helpers["unnorm_row_max"]
        self.S = int(self.species_helpers['S'])
        if compact_families:
            self.families = [dict(raw_families[name]) for name in family_names]
        else:
            self.families = [
                cls._family_from_raw(
                    cls._drop_unused_family_details(raw_families[name])
                )
                for name in family_names
            ]
        self.family_names = list(family_names)
        self.gene_tree_paths = family_tree_paths
        self.leaf_species_maps = [dict(m) for m in leaf_species_maps]
        self.species_tree_path = species_tree_path
        self._rust_preprocessed = rust_preprocessed
        self._preprocess_cpu_cores = preprocess_cpu_cores
        self._compact_families = compact_families
        self.num_families = len(self.families)
        return self

    def _ensure_full_families(self) -> None:
        if not getattr(self, "_compact_families", False):
            return
        rust_preprocessed = getattr(self, "_rust_preprocessed", None)
        if rust_preprocessed is None:
            raise RuntimeError("compact dataset cannot materialize family details")
        raw_families = rust_preprocessed.to_torch()["families"]
        self.families = [
            self._family_from_raw(
                self._drop_unused_family_details(raw_families[name])
            )
            for name in self.family_names
        ]
        self._compact_families = False
    
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
