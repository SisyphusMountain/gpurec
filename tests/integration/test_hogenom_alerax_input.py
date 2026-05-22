from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, UniformChunkedReconModel


_ROOT = Path(__file__).resolve().parents[1]
HOGENOM_DIR = _ROOT / "data" / "HOGENOM" / "hogenom"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (HOGENOM_DIR / "hogenom_families.local.txt").exists(),
        reason=f"HOGENOM local AleRax fixture not present: {HOGENOM_DIR}",
    ),
]


@pytest.mark.slow
def test_hogenom_alerax_family_file_likelihood_matches_reference(tmp_path):
    """Local AleRax CCP/tree-sample input gives AleRax-compatible likelihood."""
    families = HOGENOM_DIR / "hogenom_families.local.txt"
    species_tree = HOGENOM_DIR / "hogenom_S.tree"
    kwargs = dict(
        device="cuda",
        dtype=torch.float64,
        theta_init_rates=(0.1, 0.1, 0.1),
        max_families=2,
        fixed_iters_E=16,
        fixed_iters_Pi=6,
    )
    resident = GeneReconModel.from_alerax_families(
        str(species_tree),
        families,
        mode="global",
        max_wave_size=32768,
        **kwargs,
    )
    chunked = UniformChunkedReconModel.from_alerax_families(
        str(species_tree),
        families,
        family_chunk_size=2,
        max_wave_size=32768,
        **kwargs,
    )

    with torch.no_grad():
        nll_bits = resident(reduce="per_family").detach().cpu()
        chunked_bits = chunked.nll_per_family().detach().cpu()
    torch.testing.assert_close(chunked_bits, nll_bits, rtol=1e-10, atol=1e-10)

    log_likelihood_nats = torch.tensor(
        [-float(x) * math.log(2.0) for x in nll_bits],
        dtype=torch.float64,
    )
    reference = torch.tensor(
        [-43.441721421888033, -132.66416588113742],
        dtype=torch.float64,
    )
    torch.testing.assert_close(log_likelihood_nats, reference, rtol=0.0, atol=5e-3)
