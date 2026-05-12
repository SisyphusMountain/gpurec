import math
import os
import hashlib
from pathlib import Path
from typing import Any, Sequence

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
                mapping[gene] = species
    return mapping


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
    base_dir = Path(families_file).resolve().parent
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        if not current["tree_paths"]:
            raise ValueError(f"family {current['name']!r} has no gene tree path")
        records.append(current)
        current = None

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
        if current is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key in {"starting_gene_tree", "gene_tree"}:
            current["tree_paths"].append(_resolve_family_path(value, base_dir))
        elif key == "mapping":
            current["mapping"] = _resolve_family_path(value, base_dir)
    finish()

    if start < 0:
        raise ValueError("start must be non-negative")
    stop = None if max_families is None else start + int(max_families)
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


class GeneDataset:
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
                include_species_matrices=False,
            )
            raw_by_family = raw_all['families']
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
            # Convert log_split_probs from ln (C++ output) to log2
            ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
            families.append({
                'ccp_helpers': ccp,
                'root_clade_id': int(ccp['root_clade_id']),
                'leaf_row_index': raw['leaf_row_index'],
                'leaf_col_index': raw['leaf_col_index'],
                'C': int(ccp['C']),
                'N_splits': int(ccp['N_splits']),
                'log_split_probs': ccp['log_split_probs_sorted'],
            })
        # stored on CPU. Only move when computing likelihood and optimizing.
        self.families = families
        self.family_names = list(family_names)
        self.gene_tree_paths = family_tree_paths
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

        version = "light-v4:compact-species:alerax-maps"
        species_hash = cls._hash_file(species_tree_path)
        species_key = hashlib.sha256(
            f"{version}:species:{species_hash}".encode("utf-8")
        ).hexdigest()
        species_cache = cache_dir / f"species-{species_key}.pt"

        species_helpers = None
        if species_cache.exists() and not refresh:
            species_helpers = torch.load(
                species_cache,
                map_location="cpu",
                weights_only=False,
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
                raw_by_family[name] = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
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
                include_species_matrices=False,
            )
            if species_helpers is None:
                species_helpers = raw_all["species"]
                torch.save(species_helpers, species_cache)

            for name, raw in raw_all["families"].items():
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
