"""Streaming multi-batch exact HVP: H u = sum_b H_b u (memory-bounded, rebuild-per-HVP).

The total NLL is a sum over gene families; batches partition the families into DISJOINT subsets and
specieswise theta (S,3) is SHARED, so H = d2L/dtheta2 = sum_b H_b exactly. `make_exact_hvp` on a
`batch_statics` LIST of length > 1 streams: per hvp(u) call it loops the batches, rebuilds each
batch's forward saved-intermediates + point cache, evaluates H_b u, accumulates, and frees before
the next batch.

Gates here are FAST (fp64, tiny 20-species model, a few HVPs) and NOT @slow:
  * single-batch equivalence: make_exact_hvp([static]) == make_exact_hvp_single(static) bit-for-bit;
  * FD-Hessian parity (the key correctness gate): streamed H u == (g(t+eps u) - g(t-eps u))/(2 eps)
    where g is make_value_and_grad's multi-batch gradient;
  * additivity: streamed H_batchlist u == sum_b H_[b] u;
  * the streaming path runs with e_adjoint_solver="neumann" too.
"""
import math

import pytest

rustree = pytest.importorskip("rustree")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.bench.simulate import simulate_dataset
from gpurec.fit.newton_cg import _fd_hessian_hvp
from gpurec.solver.hvp_exact import make_exact_hvp, make_exact_hvp_single
from gpurec.solver.value_and_grad import forward_solve, make_value_and_grad

_TSI = 128
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=_TSI, neumann_terms=64, self_loop_solver="neumann",
           bicgstab_max_iter=500, bicgstab_tol=1e-10, bicgstab_breakdown_tol=1e-30,
           adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0)


def _build(n_species=20, n_families=8, family_chunk_size=3, e_adjoint_solver="gmres", seed=3,
           mode="specieswise"):
    so = SolverOptions(e_adjoint_solver=e_adjoint_solver, **_SO); so.validate()
    import tempfile
    d = tempfile.mkdtemp()
    sp, genes = simulate_dataset(mode, d, n_species=n_species, n_families=n_families,
                                 dtl=0.05, seed=seed)
    m = GeneReconModel(sp, genes, mode=mode, device="cuda", dtype=torch.float64,
                       solver_options=so, family_chunk_size=family_chunk_size)
    return m


@pytest.mark.gpu
def test_single_batch_equivalence():
    """A length-1 batch list dispatches to the single-batch primitive bit-for-bit (no behavior
    change from the pre-refactor code)."""
    m = _build(family_chunk_size=None)  # all families in one batch
    assert len(m.batch_statics) == 1
    st = m.batch_statics[0]
    S = int(m.species_helpers["S"])
    theta = m.theta.detach().clone()
    rw = m.receiver_weights.detach().clone()
    _l, sv = forward_solve([st], theta, rw)
    hvp_list = make_exact_hvp([st], theta, rw, sv, tangent_self_iters=_TSI)
    hvp_prim = make_exact_hvp_single(st, theta, rw, sv, tangent_self_iters=_TSI)
    torch.manual_seed(0)
    for _ in range(3):
        u = torch.randn(theta.numel(), device="cuda", dtype=torch.float64)
        a, b = hvp_list(u), hvp_prim(u)
        assert torch.equal(a, b), f"dispatcher not bit-identical: max diff {float((a-b).abs().max()):.2e}"


@pytest.mark.gpu
def test_streaming_fd_hessian_parity_specieswise():
    """KEY GATE: streamed multi-batch H u matches the central finite-difference of the multi-batch
    gradient for several random directions, in fp64.

    eps=1e-3 (NOT 1e-5): the central FD of this specieswise objective is gradient-NOISE-limited, not
    truncation-limited. The pi_iters/E-adjoint-truncated gradient carries a ~5e-7*scale evaluation
    noise, so the FD error is ~noise/eps and DECREASES monotonically as eps grows. An eps sweep
    (1e-5, 1e-4, 1e-3) gives worst-rel {4.4e-2, 4.3e-3, 4.5e-4} for BOTH the single-batch validated
    primitive AND this streamed multi-batch HVP -- err proportional to 1/eps, no error floor => no
    structural bug (a real curvature mismatch would show an eps-independent floor). The streamed
    curvature is additionally pinned EXACTLY (rel<1e-10) to the sum of per-single-batch HVPs in
    test_additivity_streamed_equals_sum_of_single and bit-for-bit to the single-batch primitive in
    test_single_batch_equivalence; this FD gate confirms that validated curvature also equals the
    Hessian of make_value_and_grad's independent multi-batch gradient. eps=1e-3 clears 1e-3 with
    ~2x margin on random dense directions (the noise-optimal step for this objective)."""
    m = _build(family_chunk_size=3)
    assert len(m.batch_statics) > 1, f"need multi-batch, got {len(m.batch_statics)}"
    S = int(m.species_helpers["S"])
    theta = m.theta.detach().clone()
    rw = m.receiver_weights.detach().clone()

    hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=_TSI)
    vg = make_value_and_grad(m.batch_statics, rw, theta_shape=(S, 3))
    fd = _fd_hessian_hvp(vg, theta.reshape(-1).contiguous(), None, eps=1e-3)

    torch.manual_seed(0)
    worst = 0.0
    for k in range(4):
        u = torch.randn(theta.numel(), device="cuda", dtype=torch.float64)
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all(), f"dir {k}: non-finite HVP"
        assert rel < 1e-3, f"dir {k}: FD-Hessian rel={rel:.2e}"
        worst = max(worst, rel)
    print(f"\n[fd-parity specieswise multibatch] worst rel over 4 dirs (eps=1e-3) = {worst:.2e}")


@pytest.mark.gpu
def test_additivity_streamed_equals_sum_of_single():
    """Streamed H(batchlist) u == sum_b H([b]) u (specieswise: theta shared -> pure sum)."""
    m = _build(family_chunk_size=3)
    assert len(m.batch_statics) > 1
    S = int(m.species_helpers["S"])
    theta = m.theta.detach().clone()
    rw = m.receiver_weights.detach().clone()
    hvp_stream = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=_TSI)

    def sum_of_single(u):
        acc = None
        for st in m.batch_statics:
            _l, sv_b = forward_solve([st], theta, rw)
            hb = make_exact_hvp([st], theta, rw, sv_b, tangent_self_iters=_TSI)
            c = hb(u)
            acc = c if acc is None else acc + c
        return acc

    torch.manual_seed(1)
    for _ in range(2):
        u = torch.randn(theta.numel(), device="cuda", dtype=torch.float64)
        a, b = hvp_stream(u).double(), sum_of_single(u).double()
        rel = float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)
        assert rel < 1e-10, f"streamed != sum_of_single: rel={rel:.2e}"


@pytest.mark.gpu
def test_streaming_fd_hessian_parity_genewise_theta():
    """The genewise branch of the streaming dispatcher (per-family gather/scatter, theta-only) must
    match the FD Hessian of the multi-batch genewise gradient. Locks the disjoint-family index_add
    scatter path in _make_exact_hvp_streaming (distinct from genewise_curvature's held-cache HVP)."""
    m = _build(family_chunk_size=3, mode="genewise")
    assert len(m.batch_statics) > 1, f"need multi-batch, got {len(m.batch_statics)}"
    G = len(m.families) if hasattr(m, "families") else int(m.theta.shape[0])
    S = int(m.species_helpers["S"])
    theta = m.theta.detach().clone()  # [G, 3]
    rw = m.receiver_weights.detach().clone()
    assert m.batch_statics[0].genewise and not m.batch_statics[0].specieswise

    hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=_TSI)
    vg = make_value_and_grad(m.batch_statics, rw, theta_shape=tuple(theta.shape))
    fd = _fd_hessian_hvp(vg, theta.reshape(-1).contiguous(), None, eps=1e-3)  # noise-optimal (see specieswise gate)
    torch.manual_seed(0)
    worst = 0.0
    for k in range(3):
        # broadcast e_j across families' theta component j (exercises every family block)
        u = torch.zeros(theta.numel(), device="cuda", dtype=torch.float64)
        u.view(-1)[k::3] = 1.0
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 1e-3, f"dir {k}: genewise streaming FD rel={rel:.2e}"
        worst = max(worst, rel)
    print(f"\n[fd-parity genewise multibatch streaming] worst rel over 3 dirs (eps=1e-3) = {worst:.2e}")


@pytest.mark.gpu
def test_streaming_with_neumann_e_adjoint():
    """The streaming HVP runs (finite, FD-consistent) with e_adjoint_solver='neumann'."""
    m = _build(family_chunk_size=3, e_adjoint_solver="neumann")
    assert len(m.batch_statics) > 1
    assert m.batch_statics[0].solver_options.e_adjoint_solver == "neumann"
    S = int(m.species_helpers["S"])
    theta = m.theta.detach().clone()
    rw = m.receiver_weights.detach().clone()
    hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=_TSI)
    vg = make_value_and_grad(m.batch_statics, rw, theta_shape=(S, 3))
    fd = _fd_hessian_hvp(vg, theta.reshape(-1).contiguous(), None, eps=1e-3)  # noise-optimal step (see FD-parity test)
    torch.manual_seed(2)
    u = torch.randn(theta.numel(), device="cuda", dtype=torch.float64)
    Ha, Hf = hvp(u).double(), fd(u).double()
    rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
    assert torch.isfinite(Ha).all() and rel < 1e-3, f"neumann streaming FD rel={rel:.2e}"
