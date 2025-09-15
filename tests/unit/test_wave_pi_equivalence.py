import math
from pathlib import Path

import pytest
import torch

from src.core.ccp import build_ccp_from_single_tree, build_ccp_helpers
from src.core.tree_helpers import build_species_helpers
from src.core.ccp import build_clade_species_mapping
from src.reconciliation.likelihood import E_fixed_point, Pi_fixed_point
from src.reconciliation.wave_likelihood import Pi_waves_log, build_waves_by_clade_size


def _load_dataset_helpers(dataset_dir: Path, device: torch.device, dtype: torch.dtype):
    sp_path = str(dataset_dir / "sp.nwk")
    g_path = str(dataset_dir / "g.nwk")

    ccp = build_ccp_from_single_tree(g_path, debug=False)
    species_helpers = build_species_helpers(sp_path, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)

    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    log_clade_species_map = torch.log(clade_species_map + 1e-45)
    log_clade_species_map[clade_species_map == 0] = float("-inf")

    return ccp_helpers, species_helpers, log_clade_species_map


@pytest.mark.parametrize("dataset", [
    "test_trees_1",
    "test_trees_2",
])
def test_wave_matches_fixed_point(dataset):
    base = Path(__file__).parent.parent / "data" / dataset
    assert (base / "sp.nwk").exists() and (base / "g.nwk").exists(), "Missing dataset files"

    device = torch.device("cpu")
    dtype = torch.float64

    # Build helpers
    ccp_helpers, species_helpers, log_clade_species_map = _load_dataset_helpers(base, device, dtype)

    # Choose event rates (balanced, non-trivial transfers)
    delta, tau, lam = 0.1, 0.2, 0.1
    theta = torch.tensor([
        math.log(max(delta, 1e-10)),
        math.log(max(tau, 1e-10)),
        math.log(max(lam, 1e-10)),
    ], device=device, dtype=dtype)

    # Compute E and components
    E_out = E_fixed_point(
        species_helpers=species_helpers,
        theta=theta,
        max_iters=200,
        tolerance=1e-12,
        return_components=True,
        warm_start_E=None,
        use_anderson=False,
    )
    E = E_out["E"]
    E_s1 = E_out["E_s1"]
    E_s2 = E_out["E_s2"]
    Ebar = E_out["E_bar"]

    # Legacy fixed-point Pi
    FP = Pi_fixed_point(
        ccp_helpers=ccp_helpers,
        species_helpers=species_helpers,
        clade_species_map=log_clade_species_map,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        theta=theta,
        max_iters=300,
        tolerance=1e-12,
        warm_start_Pi=None,
        use_anderson=False,
    )
    Pi_ref = FP["Pi"]

    # Wave-by-size Pi
    waves = build_waves_by_clade_size(ccp_helpers)
    Pi_wave, _ = Pi_waves_log(
        ccp_helpers=ccp_helpers,
        species_helpers=species_helpers,
        clade_species_map=log_clade_species_map,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        theta=theta,
        waves=waves,
        max_iters_wave=200,
        tol_wave=1e-12,
        warm_start_Pi=None,
    )

    # Compare
    assert Pi_wave.shape == Pi_ref.shape
    # Tight tolerances; allow tiny numerical differences
    assert torch.allclose(Pi_wave, Pi_ref, rtol=1e-7, atol=1e-9), (
        f"Pi mismatch: max abs diff = {(Pi_wave - Pi_ref).abs().max().item():.3e}"
    )

