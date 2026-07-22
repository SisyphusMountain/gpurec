import math, torch, pytest
pytestmark = pytest.mark.gpu

from gpurec.core.kernels.e_step import e_fixed_point_triton

def _tiny_species():
    # S=3: leaves 0,1 ; root 2. sentinel = S = 3.
    S = 3
    dev = "cuda"; dt = torch.float64
    sp_parent = torch.tensor([2, 2, -1], dtype=torch.int32, device=dev)
    sp_child1 = torch.tensor([3, 3, 0], dtype=torch.int32, device=dev)
    sp_child2 = torch.tensor([3, 3, 1], dtype=torch.int32, device=dev)
    return S, dev, dt, sp_parent, sp_child1, sp_child2

def _run(leaf_fm_log):
    S, dev, dt, sp_parent, sp_child1, sp_child2 = _tiny_species()
    logp = lambda v: torch.full((1, S), math.log2(v), dtype=dt, device=dev)
    E0 = torch.full((1, S), torch.finfo(dt).min, dtype=dt, device=dev)
    # max_transfer is a log2 transfer normalizer (real model: log_pT + receiver-set
    # normalization, always < 0). The brief's placeholder value of 0.0 (== transfer
    # weight 1.0, unnormalized) has no real extinction fixed point on this tiny tree
    # (the root self-amplifies to +inf -> NaN); that divergence is pre-existing and
    # unrelated to fraction-missing. A bounded value keeps the E-step convergent so
    # the leaf-injection assertions below are meaningful.
    max_transfer = torch.full((1, S), math.log2(0.05), dtype=dt, device=dev)
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0, log_pS=logp(0.2), log_pD=logp(0.2), log_pL=logp(0.2),
        max_transfer=max_transfer,
        receiver_log_probs=torch.full((S,), -math.log2(S), dtype=dt, device=dev),
        species_parent=sp_parent, species_child1=sp_child1, species_child2=sp_child2,
        max_ancestor_depth=3, max_iter=4000, tol=1e-13, use_receiver_weights=False,
        leaf_fm_log=leaf_fm_log,
    )
    return E, Ebar

def test_fm_zero_matches_no_fm_exactly():
    E_none, _ = _run(None)
    S = E_none.shape[-1]
    off = torch.full((S,), float("-inf"), dtype=E_none.dtype, device=E_none.device)  # all -inf => no missing
    E_fm0, _ = _run(off)
    assert torch.equal(E_none, E_fm0)

def test_leaf_self_consistency_with_fraction_missing():
    S, dev, dt, *_ = _tiny_species()
    fm = 0.3
    leaf_fm_log = torch.tensor([math.log2(fm), math.log2(fm), float("-inf")], dtype=dt, device=dev)
    E, Ebar = _run(leaf_fm_log)
    l = 0
    pS = math.log2(0.2); pD = math.log2(0.2); pL = math.log2(0.2)
    E_l = E[0, l]; Eb_l = Ebar[0, l]
    terms = torch.stack([
        torch.tensor(pS + math.log2(fm), dtype=dt, device=dev),  # p^S * fm_l  (single factor)
        pD + 2 * E_l,
        E_l + Eb_l,
        torch.tensor(pL, dtype=dt, device=dev),
    ])
    m = terms.max()
    rhs = m + torch.log2(torch.exp2(terms - m).sum())
    assert abs(float(rhs - E_l)) < 1e-9
