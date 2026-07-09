import math
from pathlib import Path
import pytest

rustree = pytest.importorskip("rustree")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from gpurec.bench.simulate import simulate_dataset
from gpurec.api.model import GeneReconModel


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["global", "genewise", "specieswise"])
def test_simulated_dataset_is_gpurec_compatible(mode, tmp_path):
    sp, genes = simulate_dataset(mode, tmp_path, n_species=20, n_families=5, dtl=0.05, seed=1)
    assert Path(sp).exists() and len(genes) == 5 and all(Path(g).exists() for g in genes)
    model = GeneReconModel(sp, genes, mode=mode, device="cuda", dtype=torch.float32)
    loss = float(model())                      # NLL in bits
    assert math.isfinite(loss), f"{mode}: non-finite likelihood {loss}"


@pytest.mark.gpu
def test_simulation_is_deterministic(tmp_path):
    a = simulate_dataset("global", tmp_path / "a", n_species=20, n_families=5, dtl=0.05, seed=7)
    b = simulate_dataset("global", tmp_path / "b", n_species=20, n_families=5, dtl=0.05, seed=7)
    assert [Path(g).read_text() for g in a[1]] == [Path(g).read_text() for g in b[1]]
    assert Path(a[0]).read_text() == Path(b[0]).read_text()
