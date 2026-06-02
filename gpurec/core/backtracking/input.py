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


def sample_reconciliations(model, *, family_index: int = 0, seed: int = 0):
    family = model.families[family_index]
    offset = sum(int(model.families[idx]["C"]) for idx in range(family_index))
    E, _, _, Ebar, _, pi_wave, pibar_wave, _, log_p_s, log_p_d, *_ = solve_resident_e_pi(model, model.theta)
    perm = model.wave_layout["perm"]
    pi = pi_wave.index_select(0, perm)
    pibar = pibar_wave.index_select(0, perm)
    C = int(family["C"])
    species_helpers = model.species_helpers
    S = int(species_helpers["S"])

    leaf_species = torch.full((C,), -1, dtype=torch.int64)
    leaf_species[torch.as_tensor(family["leaf_row_index"], dtype=torch.int64)] = torch.as_tensor(family["leaf_col_index"], dtype=torch.int64)

    return _load_native_module().sample_reconciliations_torch(
        _numpy(leaf_species, torch.int64),
        _numpy(family["split_leftrights_sorted"], torch.int64),
        _numpy(pi[offset : offset + C], torch.float64),
        _numpy(pibar[offset : offset + C], torch.float64),
        _numpy(_family_param(E, family_index), torch.float64),
        _numpy(_family_param(Ebar, family_index), torch.float64),
        _numpy(_family_param(log_p_s, family_index).reshape(1).expand(S), torch.float64),
        _numpy(_family_param(log_p_d, family_index).reshape(1).expand(S), torch.float64),
        _numpy(species_helpers["sp_child1"], torch.int64),
        _numpy(species_helpers["sp_child2"], torch.int64),
        _numpy(species_helpers["sp_subtree_start"], torch.int64),
        _numpy(species_helpers["sp_subtree_end"], torch.int64),
        seed,
    )
