import math
import os
import hashlib
from pathlib import Path
from typing import Any

import torch
from .preprocess_cpp import _load_extension as _load_species_gene_ext
from .species import uniform_ancestors_t_from_topology


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
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.genewise = genewise
        self.specieswise = specieswise
        self.device = device
        self.dtype = dtype
        ext = _load_species_gene_ext()

        use_single_preprocess = os.environ.get("GPUREC_PREPROCESS_MODE", "").lower() == "single"
        family_names = [f"family_{i:06d}" for i in range(len(gene_tree_paths))]
        if use_single_preprocess:
            raw_by_family = {
                name: ext.preprocess(
                    species_tree_path,
                    [str(path)],
                    include_species_matrices=False,
                )
                for name, path in zip(family_names, gene_tree_paths)
            }
            self.species_helpers = raw_by_family[family_names[0]]['species']
        elif preprocess_cache_dir is not None:
            self.species_helpers, raw_by_family = self._preprocess_with_cache(
                ext,
                species_tree_path,
                gene_tree_paths,
                family_names,
                preprocess_cache_dir=preprocess_cache_dir,
                refresh=refresh_preprocess_cache,
            )
        else:
            families_input = {
                name: [str(path)]
                for name, path in zip(family_names, gene_tree_paths)
            }
            raw_all = ext.preprocess_multiple_families(
                species_tree_path,
                families_input,
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
        for i, (gpath, family_name) in enumerate(zip(gene_tree_paths, family_names)):
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
        self.gene_tree_paths = list(gene_tree_paths)
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
        *,
        preprocess_cache_dir: str | os.PathLike,
        refresh: bool,
    ):
        cache_dir = Path(preprocess_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        version = "light-v3:compact-species"
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
        family_cache_paths = {}
        for name, path in zip(family_names, gene_tree_paths):
            gene_hash = cls._hash_file(path)
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
                missing[name] = [str(path)]

        if missing:
            raw_all = ext.preprocess_multiple_families(
                species_tree_path,
                missing,
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
                species_tree_path,
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
