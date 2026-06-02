from functools import lru_cache
import importlib.util
import sys
from pathlib import Path

import torch

from gpurec.api.model import solve_resident_e_pi

_NATIVE_LIBRARY_PATH = Path(__file__).resolve().parents[3] / "crates/gpurec-backtrack/target/release/libgpurec_backtrack.so"


@lru_cache(maxsize=1)
def _load_native_module():
    spec = importlib.util.spec_from_file_location("gpurec_backtrack", str(_NATIVE_LIBRARY_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpurec_backtrack"] = module
    spec.loader.exec_module(module)
    return module


def _numpy(value, dtype: torch.dtype) -> object:
    return torch.as_tensor(value).detach().cpu().to(dtype=dtype).contiguous().numpy()


def _family_param(param: torch.Tensor, family_index: int) -> torch.Tensor:
    param = torch.as_tensor(param).detach()
    if param.ndim <= 1:
        return param.reshape(-1)[0 if int(param.numel()) == 1 else family_index]
    return param[0] if int(param.shape[0]) == 1 else param[family_index]


def _species_param(param: torch.Tensor, model, family_index: int, S: int) -> torch.Tensor:
    param = torch.as_tensor(param).detach()
    if param.ndim == 0:
        return param.reshape(1).expand(S)
    if param.ndim == 1:
        if int(param.numel()) == S:
            return param
        return param.reshape(-1)[0 if int(param.numel()) == 1 else family_index].reshape(1).expand(S)
    row = param[0] if int(param.shape[0]) == 1 else param[family_index]
    return row.reshape(-1)[0].reshape(1).expand(S) if int(row.numel()) == 1 else row


def sample_reconciliations(model, *, family_index: int = 0, seed: int = 0):
    family = model.families[family_index]
    batch_static = model.batch_statics[0]
    local_family_index = family_index
    for static in model.batch_statics:
        if family_index in static.family_indices:
            batch_static = static
            local_family_index = static.family_indices.index(family_index)
            break
    offset = sum(int(model.families[idx]["C"]) for idx in batch_static.family_indices[:local_family_index])
    theta = model._theta_for_static(batch_static, model.theta)
    E, _, _, Ebar, _, pi_wave, pibar_wave, _, log_p_s, log_p_d, *_ = solve_resident_e_pi(batch_static, theta)
    perm = batch_static.wave_layout["perm"]
    pi = pi_wave.index_select(0, perm)
    pibar = pibar_wave.index_select(0, perm)
    C = int(family["C"])
    species_helpers = model.species_helpers
    S = int(species_helpers["S"])

    leaf_species = torch.full((C,), -1, dtype=torch.int64)
    leaf_species[torch.as_tensor(family["leaf_row_index"], dtype=torch.int64)] = torch.as_tensor(family["leaf_col_index"], dtype=torch.int64)

    return _load_native_module().sample_reconciliations_torch(
        int(family.get("root_clade_id", C - 1)),
        _numpy(leaf_species, torch.int64),
        _numpy(family.get("split_parents_sorted", [clade for clade in range(C) if int(leaf_species[clade]) < 0]), torch.int64),
        _numpy(family["split_leftrights_sorted"], torch.int64),
        _numpy(family.get("log_split_probs_sorted", torch.zeros(len(family["split_leftrights_sorted"]) // 2)), torch.float64),
        _numpy(pi[offset : offset + C], torch.float64),
        _numpy(pibar[offset : offset + C], torch.float64),
        _numpy(_family_param(E, local_family_index), torch.float64),
        _numpy(_family_param(Ebar, local_family_index), torch.float64),
        _numpy(_species_param(log_p_s, model, local_family_index, S), torch.float64),
        _numpy(_species_param(log_p_d, model, local_family_index, S), torch.float64),
        _numpy(species_helpers["sp_child1"], torch.int64),
        _numpy(species_helpers["sp_child2"], torch.int64),
        _numpy(species_helpers["sp_subtree_start"], torch.int64),
        _numpy(species_helpers["sp_subtree_end"], torch.int64),
        seed,
    )
