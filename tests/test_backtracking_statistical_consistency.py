# tests/test_backtracking_statistical_consistency.py
"""Statistical consistency of the native backtracking sampler with the forward solve.

The Rust sampler (crates/gpurec-backtrack) draws concrete reconciliations by walking
the SAME Pi/Pibar/E/Ebar/receiver_log_probs arrays that ``solve_resident_e_pi``
produces for the actual reported likelihood -- it never recomputes anything from
theta itself (see gpurec/core/backtracking/input.py). That single-source-of-truth
design means a bug that mis-threads a feature into those arrays (receiver weights,
the fraction-missing extinction term) would show up as a real, detectable mismatch
between what the sampler draws and what the arrays say it should draw.

Fixture: a small species tree, a single-leaf gene family (so the whole sampling
trajectory stays in ONE clade -- see the SplitDup/SplitTransfer/SplitSpeciation
guard in ``sample_term``, which only fires when a clade has CCP children), with
BOTH non-uniform receiver weights and large fraction-missing values active. Both
tests draw many reconciliations with fixed seeds (fully deterministic -- no
run-to-run flakiness) and check the empirical distribution against the theoretical
one with a chi-square goodness-of-fit test:

  1. ``test_root_species_matches_pi_row``: the root species draw against the
     Pi(root_clade, .) row.
  2. ``test_transfer_recipient_matches_receiver_weighted_pi``: the transfer
     recipient draw, conditioned on donor species, against
     receiver_log_probs[r] + Pi(clade, r) over non-ancestor recipients r.

Both thresholds were sanity-checked against two deliberate mutations during
development (dropping receiver_log_probs from the theoretical recipient formula,
and disabling fraction_missing on one side only): both were rejected at p < 1e-7,
while the correct formulas pass with p > 0.15 -- so these tests have real power to
catch a regression in either feature, not just a coincidental pass.
"""
import numpy as np
import pytest
import torch
from scipy import stats

pytestmark = pytest.mark.gpu

from gpurec.api.model import GeneReconModel
from gpurec.core.backtracking.input import (
    _family_param,
    _load_native_module,
    _numpy,
    _species_param,
)
from gpurec.core.inference.solver import solve_resident_e_pi

_SPECIES = "(((A:1,B:1)AB:1,C:1)ABC:1,D:1)Root;"
_GENE = "A_1;"  # single-leaf gene family: the whole trajectory stays in one clade.
_SIGNIFICANCE = 0.01
_ROOT_SAMPLES = 5000
_RECIPIENT_SAMPLES = 300_000


def _build_model(tmp_path):
    sp = tmp_path / "species.nwk"
    sp.write_text(_SPECIES)
    g = tmp_path / "gene.nwk"
    g.write_text(_GENE)
    model = GeneReconModel(
        str(sp), [str(g)], mode="specieswise", device="cuda", dtype=torch.float64,
        fraction_missing={"C": 0.85, "D": 0.9},
    )
    S = int(model.species_helpers["S"])
    with torch.no_grad():
        # [dup, loss, transfer] logits relative to speciation's implicit 0 -- skewed
        # toward transfer so the recipient test has real transfer events to draw from.
        model.theta.copy_(
            torch.tensor([1.0, 1.0, 2.5], device="cuda", dtype=torch.float64).expand(S, 3)
        )
        model.receiver_weights.copy_(
            torch.linspace(-2.0, 2.0, S, device="cuda", dtype=torch.float64)
        )
    return model


def _family_arrays(model, family_index=0):
    """Mirror ``sample_reconciliations``'s setup (input.py) but return the raw
    arrays once, so many native-module calls can reuse them instead of re-running
    the forward solve per draw."""
    family = model.families[family_index]
    batch_static = model.batch_statics[0]
    theta = model._theta_for_static(batch_static, model.theta)
    (
        E, _, _, Ebar, _, pi_wave, pibar_wave, _, log_p_s, log_p_d, *_, receiver_log_probs,
    ) = solve_resident_e_pi(batch_static, theta, model.receiver_weights)
    perm = batch_static.wave_layout["perm"]
    pi = pi_wave.index_select(0, perm)
    pibar = pibar_wave.index_select(0, perm)
    C = int(family["C"])
    pi_state = batch_static.pi_forward_state
    pi_offset = pi_state.pi_offset.index_select(0, perm)[0:C]
    pibar_offset = pi_state.pibar_offset.index_select(0, perm)[0:C]
    pi_family = pi[0:C].to(pi_offset.dtype) + pi_offset[:, None]
    pibar_family = pibar[0:C].to(pibar_offset.dtype) + pibar_offset[:, None]

    S = int(model.species_helpers["S"])
    leaf_species = torch.full((C,), -1, dtype=torch.int64)
    leaf_species[torch.as_tensor(family["leaf_row_index"], dtype=torch.int64)] = (
        torch.as_tensor(family["leaf_col_index"], dtype=torch.int64)
    )

    native_args = (
        _numpy(leaf_species, torch.int64),
        _numpy(family["split_parents_sorted"], torch.int64),
        _numpy(family["split_leftrights_sorted"], torch.int64),
        _numpy(family["log_split_probs_sorted"], torch.float64),
        _numpy(pi_family, torch.float64),
        _numpy(pibar_family, torch.float64),
        _numpy(_family_param(E, family_index), torch.float64),
        _numpy(_family_param(Ebar, family_index), torch.float64),
        _numpy(_species_param(log_p_s, model, family_index, S), torch.float64),
        _numpy(_species_param(log_p_d, model, family_index, S), torch.float64),
        _numpy(receiver_log_probs, torch.float64),
        _numpy(model.species_helpers["sp_child1"], torch.int64),
        _numpy(model.species_helpers["sp_child2"], torch.int64),
        _numpy(model.species_helpers["sp_subtree_start"], torch.int64),
        _numpy(model.species_helpers["sp_subtree_end"], torch.int64),
    )
    root_clade = int(family.get("root_clade_id", C - 1))
    return native_args, root_clade, pi_family, receiver_log_probs


def _row_buckets(observed, expected):
    """Pool categories with expected count < 5 (the standard chi-square validity
    threshold) into one 'other' bucket, preserving the row's total."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    big = expected >= 5.0
    obs_b = list(observed[big])
    exp_b = list(expected[big])
    if (~big).any():
        obs_b.append(float(observed[~big].sum()))
        exp_b.append(float(expected[~big].sum()))
    return obs_b, exp_b


def _pooled_chisquare(rows):
    """Combine several independent multinomial rows (one per donor species) into
    one chi-square statistic. Rows without at least two well-powered buckets are
    skipped (not enough samples to say anything for that donor)."""
    obs_all, exp_all, dof = [], [], 0
    for observed, expected in rows:
        obs_b, exp_b = _row_buckets(observed, expected)
        if len(exp_b) < 2 or min(exp_b) < 5.0:
            continue
        obs_all.extend(obs_b)
        exp_all.extend(exp_b)
        dof += len(exp_b) - 1
    assert dof > 0, "no donor had enough samples for a well-powered test"
    statistic = float(np.sum((np.array(obs_all) - np.array(exp_all)) ** 2 / np.array(exp_all)))
    p_value = float(stats.chi2.sf(statistic, dof))
    return statistic, p_value, dof


def test_root_species_matches_pi_row(tmp_path):
    model = _build_model(tmp_path)
    native_args, root_clade, pi_family, _ = _family_arrays(model)
    native = _load_native_module()
    S = int(model.species_helpers["S"])

    root_counts = np.zeros(S)
    for seed in range(_ROOT_SAMPLES):
        nodes = native.sample_reconciliations_torch(root_clade, *native_args, seed)
        root_counts[nodes[0][1]] += 1

    pi_root = pi_family[root_clade].cpu().numpy()
    weight = pi_root - pi_root.max()
    p_theory = np.exp2(weight)
    p_theory /= p_theory.sum()
    expected = p_theory * _ROOT_SAMPLES
    assert expected.min() >= 5.0, "fixture too sparse for a valid chi-square test"

    statistic, p_value = stats.chisquare(root_counts, expected)
    assert p_value > _SIGNIFICANCE, (
        f"sampled root-species distribution diverges from Pi(root_clade, .): "
        f"chi2={statistic:.3f}, p={p_value:.3g}"
    )


def test_transfer_recipient_matches_receiver_weighted_pi(tmp_path):
    model = _build_model(tmp_path)
    native_args, root_clade, pi_family, receiver_log_probs = _family_arrays(model)
    native = _load_native_module()
    S = int(model.species_helpers["S"])

    _n_transfers, _n_dup, edges = native.sample_family_counts(
        root_clade, *native_args, _RECIPIENT_SAMPLES, 0
    )
    by_donor = {}
    for donor, recipient, count in edges:
        by_donor.setdefault(donor, {})[recipient] = count
    assert by_donor, "no transfers sampled -- fixture is not exercising the recipient path"

    subtree_start = model.species_helpers["sp_subtree_start"].cpu().numpy()
    subtree_end = model.species_helpers["sp_subtree_end"].cpu().numpy()

    def is_ancestor(a, node):
        return subtree_start[a] <= subtree_start[node] < subtree_end[a]

    pi_root = pi_family[root_clade].cpu().numpy()  # the only clade this fixture ever visits
    recv = receiver_log_probs.cpu().numpy()

    rows = []
    for donor, recips in by_donor.items():
        total = sum(recips.values())
        admissible = [r for r in range(S) if not is_ancestor(r, donor)]
        weight = np.array([recv[r] + pi_root[r] for r in admissible])
        weight = weight - weight.max()
        p_theory = np.exp2(weight)
        p_theory /= p_theory.sum()
        observed = np.array([recips.get(r, 0) for r in admissible], dtype=float)
        expected = p_theory * total
        rows.append((observed, expected))

    statistic, p_value, dof = _pooled_chisquare(rows)
    assert p_value > _SIGNIFICANCE, (
        f"sampled transfer-recipient distribution diverges from "
        f"receiver_log_probs + Pi(clade, .): chi2={statistic:.3f}, dof={dof}, p={p_value:.3g}"
    )
