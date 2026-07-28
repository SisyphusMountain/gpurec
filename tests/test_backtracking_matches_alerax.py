# tests/test_backtracking_matches_alerax.py
"""External validation of gpurec's backtracking sampler against AleRax v1.4.0.

test_fraction_missing_alerax.py already confirms gpurec's forward LIKELIHOOD
(a single number per family) matches AleRax's, with and without fraction-missing.
This test goes one level deeper: it checks that gpurec's *sampler* -- the native
Rust code in crates/gpurec-backtrack that draws concrete reconciliations -- picks
species with the same PROBABILITIES AleRax's own sampler does, not just the same
total likelihood. Two implementations can agree on a sum while disagreeing on how
that sum is split up; this test would catch that.

AleRax supports this directly: ``-g N --seed S`` makes it sample N reconciled gene
trees itself (not just report a likelihood), and its output includes, per sample,
which species branch the gene family's history STARTS at (the ``origination``
column in ``family_0_speciesEventCounts_<k>.txt``). That's exactly what gpurec
calls the "root species" -- the species ``sample_root_species``/the first node of
``sample_reconciliations`` draws, weighted by ``Pi(root_clade, .)``.

Gene tree rooting -- do NOT pass ``--gene-tree-rooting ROOTED``
-----------------------------------------------------------------
gpurec's CCP preprocessing always marginalizes uniformly over every possible
rooting of the input gene tree (nothing in gpurec/ or crates/ can turn this
off). AleRax's default does the same thing -- its startup log says so
explicitly ("all gene tree root positions are considered with the same
probability") -- so the two are already apples-to-apples with NO rooting flag
passed. An earlier version of this test passed ``--gene-tree-rooting ROOTED``
to force AleRax onto the one given rooting, reasoning that this was needed to
match gpurec; that reasoning was backwards, and it manufactured a large,
reproducible discrepancy in the sampled origination distribution on any
fixture where duplication and transfer both compete for the same clade (e.g.
species "A" drawn ~14% of the time by gpurec vs ~28% by AleRax on the fixture
below) -- a genuinely different quantity being compared, not a model bug. Full
writeup, including the two other hypotheses tested and refuted along the way
(transfer-admissibility rule, uniform-receiver normalization, fixed-point
convergence -- all confirmed to match), is in
docs/gpurax/alerax_transfer_origination_discrepancy.md.

Receiver weights have no AleRax equivalent at all (AleRax's transfer receiver
is always uniform), so that piece can only be checked internally -- see
test_backtracking_statistical_consistency.py.
"""
from __future__ import annotations

import collections
import shutil
import subprocess

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.gpu

_ALERAX = shutil.which("alerax")

_SPECIES_NWK = "((A:1,B:1)AB:1,C:1)Root;"
_GENE_NWK = "(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;"  # A duplicated: forces
# real duplication/transfer competition (unlike a gene tree matching the species
# topology exactly, which has only one possible reconciliation).
_D = _L = _T = 0.3
_FM_HIGH = 0.8  # applied at B and C, the two species with a single observed copy
_N_SAMPLES = 2000
_SIGNIFICANCE = 0.01
_ALERAX_TIMEOUT_S = 60


def _write_fixture(workdir, fm_b, fm_c):
    (workdir / "sp.nwk").write_text(_SPECIES_NWK + "\n")
    (workdir / "g.nwk").write_text(_GENE_NWK + "\n")
    (workdir / "families.txt").write_text("[FAMILIES]\n- family_0\ngene_tree = g.nwk\n")
    (workdir / "fm.txt").write_text(f"A 0\nB {fm_b}\nC {fm_c}\n")


def _run_alerax_origination(workdir, out_name):
    """Sample _N_SAMPLES reconciliations with AleRax; return a Counter of the
    origination species (the species each sample's family history starts at)."""
    cmd = [
        _ALERAX,
        "-f", "families.txt",
        "-s", "sp.nwk",
        "-p", out_name,
        "-g", str(_N_SAMPLES),
        "--seed", "0",
        "--species-tree-search", "SKIP",
        "--rec-model", "UndatedDTL",
        "--model-parametrization", "GLOBAL",
        "--fraction-missing-file", "fm.txt",
        "--min-covered-species", "3",
        "--fix-rates", "--d", str(_D), "--l", str(_L), "--t", str(_T),
    ]
    proc = subprocess.run(
        cmd, cwd=str(workdir), capture_output=True, text=True, timeout=_ALERAX_TIMEOUT_S
    )
    recon_dir = workdir / out_name / "reconciliations" / "all"
    if proc.returncode != 0 or not recon_dir.exists():
        raise AssertionError(
            f"AleRax sampling run failed (rc={proc.returncode}); no {recon_dir}.\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    counts = collections.Counter()
    for k in range(_N_SAMPLES):
        path = recon_dir / f"family_0_speciesEventCounts_{k}.txt"
        with open(path) as f:
            next(f)  # header
            for line in f:
                fields = [p.strip() for p in line.split(",")]
                if fields[6] == "1":  # 'origination' column
                    counts[fields[0]] += 1
                    break
    assert sum(counts.values()) == _N_SAMPLES, (
        f"expected exactly one origination species per sample, got totals {counts}"
    )
    return counts


def _run_gpurec_origination(sp_path, g_path, fraction_missing):
    """Sample _N_SAMPLES reconciliations with gpurec's own backtracking sampler
    (the real public entry point); return a Counter of the root species."""
    import torch

    from gpurec.api.model import GeneReconModel
    from gpurec.core.backtracking.input import sample_reconciliations
    from gpurec.core.scheduling.batching import species_name_to_index

    model = GeneReconModel(
        str(sp_path), [str(g_path)], mode="global", device="cuda", dtype=torch.float64,
        fraction_missing=fraction_missing,
    )
    with torch.no_grad():
        model.theta.copy_(
            torch.log2(torch.tensor([_D, _L, _T], dtype=torch.float64, device=model.theta.device))
        )
    idx_to_name = {v: k for k, v in species_name_to_index(str(sp_path)).items()}

    counts = collections.Counter()
    for seed in range(_N_SAMPLES):
        nodes = sample_reconciliations(model, family_index=0, seed=seed)
        counts[idx_to_name[nodes[0][1]]] += 1
    return counts


def _pooled_two_sample_chisquare(counts_a, counts_b, min_cell=5.0):
    """chi-square test of homogeneity between two categorical count distributions,
    pooling any category whose column total is too small to be well-powered into
    a single 'other' bucket first."""
    categories = sorted(set(counts_a) | set(counts_b))
    big = [c for c in categories if counts_a.get(c, 0) + counts_b.get(c, 0) >= min_cell]
    small = [c for c in categories if c not in big]
    columns = big + (["other"] if small else [])
    table = np.zeros((2, len(columns)), dtype=float)
    for j, c in enumerate(big):
        table[0, j] = counts_a.get(c, 0)
        table[1, j] = counts_b.get(c, 0)
    if small:
        table[0, -1] = sum(counts_a.get(c, 0) for c in small)
        table[1, -1] = sum(counts_b.get(c, 0) for c in small)
    assert table.shape[1] >= 2, f"not enough well-populated categories to compare: {categories}"
    statistic, p_value, _dof, _expected = stats.chi2_contingency(table)
    return statistic, p_value, columns


@pytest.fixture(scope="module")
def origination_counts(tmp_path_factory):
    if _ALERAX is None:
        pytest.skip("alerax binary not found on PATH")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = {}
    for tag, fm_b, fm_c, gpurec_fm in (("fm0", 0, 0, None), ("fm_high", _FM_HIGH, _FM_HIGH, {"B": _FM_HIGH, "C": _FM_HIGH})):
        workdir = tmp_path_factory.mktemp(f"alerax_backtrack_{tag}")
        _write_fixture(workdir, fm_b, fm_c)
        results[f"alerax_{tag}"] = _run_alerax_origination(workdir, "out")
        results[f"gpurec_{tag}"] = _run_gpurec_origination(
            workdir / "sp.nwk", workdir / "g.nwk", gpurec_fm
        )

    print(
        "\n[backtracking vs AleRax v1.4.0] origination species over "
        f"{_N_SAMPLES} samples each, D=L=T={_D} (duplication+transfer both active):\n"
        f"  fm=0        gpurec={dict(results['gpurec_fm0'])}\n"
        f"              alerax={dict(results['alerax_fm0'])}\n"
        f"  fm={_FM_HIGH} (B,C)  gpurec={dict(results['gpurec_fm_high'])}\n"
        f"              alerax={dict(results['alerax_fm_high'])}\n"
    )
    return results


def test_origination_distribution_matches_alerax_fm0(origination_counts):
    """No fraction-missing: gpurec's sampled origination species distribution
    matches AleRax's own sampler, with duplication and transfer both active and
    genuinely competing to explain the family."""
    statistic, p_value, columns = _pooled_two_sample_chisquare(
        origination_counts["gpurec_fm0"], origination_counts["alerax_fm0"]
    )
    assert p_value > _SIGNIFICANCE, (
        f"origination distributions diverge at fm=0: chi2={statistic:.3f}, "
        f"p={p_value:.3g}, categories={columns}\n"
        f"gpurec={dict(origination_counts['gpurec_fm0'])}\n"
        f"alerax={dict(origination_counts['alerax_fm0'])}"
    )


def test_origination_distribution_matches_alerax_fraction_missing(origination_counts):
    """Large fraction-missing at the two species with a single observed copy:
    gpurec's sampler shifts its origination distribution the same way AleRax's
    does, on top of the same duplication+transfer competition."""
    statistic, p_value, columns = _pooled_two_sample_chisquare(
        origination_counts["gpurec_fm_high"], origination_counts["alerax_fm_high"]
    )
    assert p_value > _SIGNIFICANCE, (
        f"origination distributions diverge at fm={_FM_HIGH}: chi2={statistic:.3f}, "
        f"p={p_value:.3g}, categories={columns}\n"
        f"gpurec={dict(origination_counts['gpurec_fm_high'])}\n"
        f"alerax={dict(origination_counts['alerax_fm_high'])}"
    )
