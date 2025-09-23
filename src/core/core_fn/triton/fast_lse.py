# logsumexp, except we store the relevant outputs for backward

import torch
from torch import Tensor
from typing import Tuple
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

# =========================
# Triton kernels
# =========================
# logsumexp kernel which stores offset exponentials for backward
@triton.jit
def _lse4_kernel(
    Y, E0, E1, E2, E3,   # outputs
    X0, X1, X2, X3,      # inputs
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 128)
    mask = offs < N
    NEG_INF = tl.constexpr(float('-inf'))

    x0 = tl.load(X0 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")
    x1 = tl.load(X1 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")
    x2 = tl.load(X2 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")
    x3 = tl.load(X3 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(m01, m23)
    m_safe = tl.where(m == NEG_INF, 0.0, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)

    s01 = e0 + e1
    s23 = e2 + e3
    s   = s01 + s23

    y = tl.where(s == 0, NEG_INF, tl.log(s) + m)

    tl.store(Y  + offs, y,  mask=mask)
    tl.store(E0 + offs, e0, mask=mask)
    tl.store(E1 + offs, e1, mask=mask)
    tl.store(E2 + offs, e2, mask=mask)
    tl.store(E3 + offs, e3, mask=mask)


@triton.jit
def _lse4_backward_kernel(
    G0, G1, G2, G3,   # grads wrt inputs
    E0, E1, E2, E3,   # saved exponentials from forward
    GY,               # upstream grad
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 128)
    mask = offs < N

    e0 = tl.load(E0 + offs, mask=mask)
    e1 = tl.load(E1 + offs, mask=mask)
    e2 = tl.load(E2 + offs, mask=mask)
    e3 = tl.load(E3 + offs, mask=mask)
    gy = tl.load(GY + offs, mask=mask)

    s = e0 + e1 + e2 + e3
    inv_s = tl.where(s > 0, 1.0 / s, 0.0)

    tl.store(G0 + offs, gy * e0 * inv_s, mask=mask)
    tl.store(G1 + offs, gy * e1 * inv_s, mask=mask)
    tl.store(G2 + offs, gy * e2 * inv_s, mask=mask)
    tl.store(G3 + offs, gy * e3 * inv_s, mask=mask)


def _grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"]),)

# =========================
# Custom op returning (y, e0, e1, e2, e3)
# =========================

@triton_op("enzo::lse4_exps", mutates_args=())
def lse4_exps(x0: Tensor, x1: Tensor, x2: Tensor, x3: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _lse4_exps_cuda(x0, x1, x2, x3)


def _lse4_exps_cuda(x0: Tensor, x1: Tensor, x2: Tensor, x3: Tensor):
    # broadcast & promote
    b0, b1, b2, b3 = torch.broadcast_tensors(x0, x1, x2, x3)
    dtype = x0.dtype
    b0 = b0.contiguous().to(dtype)
    b1 = b1.contiguous().to(dtype)
    b2 = b2.contiguous().to(dtype)
    b3 = b3.contiguous().to(dtype)

    y  = torch.empty_like(b0)
    e0 = torch.empty_like(b0)
    e1 = torch.empty_like(b0)
    e2 = torch.empty_like(b0)
    e3 = torch.empty_like(b0)

    N = y.numel()
    BLOCK = 1024

    wrap_triton(_lse4_kernel)[_grid](
        y, e0, e1, e2, e3,
        b0, b1, b2, b3,
        N,
        BLOCK=BLOCK,
    )
    return (y, e0, e1, e2, e3)

@torch.library.register_kernel("enzo::lse4_exps", "cuda")
def _lse4_exps_cuda_impl(x0, x1, x2, x3):
    return _lse4_exps_cuda(x0, x1, x2, x3)

# =========================
# Autograd for enzo::lse4_exps
# =========================

def _setup_ctx_exps(ctx, inputs, output=None, **_):
    if output is None:
        raise RuntimeError("setup_context: missing 'output'")
    x0, x1, x2, x3 = inputs
    y, e0, e1, e2, e3 = output
    ctx.save_for_backward(x0, x1, x2, x3, e0, e1, e2, e3)


def _backward_exps(ctx, gy, *_unused_grads_for_e):
    x0, x1, x2, x3, e0, e1, e2, e3 = ctx.saved_tensors

    # Ensure contiguous, broadcastable shapes
    be0, be1, be2, be3, bgy = torch.broadcast_tensors(e0, e1, e2, e3, gy)
    be0 = be0.contiguous(); be1 = be1.contiguous(); be2 = be2.contiguous(); be3 = be3.contiguous()
    bgy = bgy.contiguous()

    g0 = torch.empty_like(be0); g1 = torch.empty_like(be1)
    g2 = torch.empty_like(be2); g3 = torch.empty_like(be3)

    N = be0.numel()
    BLOCK = 1024

    wrap_triton(_lse4_backward_kernel)[_grid](
        g0, g1, g2, g3,
        be0, be1, be2, be3,
        bgy,
        N,
        BLOCK=BLOCK,
    )

    # reduce back to original shapes (unbroadcast)
    def _unbroadcast(grad, ref):
        # collapse extra leading dims
        while grad.dim() > ref.dim():
            grad = grad.sum(dim=0)
        # sum along broadcasted axes
        for i, (gs, rs) in enumerate(zip(grad.shape, ref.shape)):
            if rs == 1 and gs != 1:
                grad = grad.sum(dim=i, keepdim=True)
        return grad

    return (
        _unbroadcast(g0, x0),
        _unbroadcast(g1, x1),
        _unbroadcast(g2, x2),
        _unbroadcast(g3, x3),
        # No grads wrt outputs (y, e0..e3) since we're defining backward for the op itself
    )

# Register autograd for the multi-output op
lse4_exps.register_autograd(_backward_exps, setup_context=_setup_ctx_exps)

# =========================
# Friendly wrapper that returns only y
# =========================

def lse4(x0: Tensor, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
    """
    User-facing API: returns just y = logsumexp(x0..x3)
    (Under the hood, the op also computes/stashes e0..e3 for a recompute-free backward.)
    """
    y, e0, e1, e2, e3 = lse4_exps(x0, x1, x2, x3)
    # We intentionally do NOT detach e0..e3 here; they are kept alive by ctx.save_for_backward.
    # Simply dropping Python references is fine—the autograd ctx holds them.
    return y

# =========================
# Quick test
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    for dtype in (torch.float32,):
        x0 = torch.randn(1_000, device="cuda", dtype=dtype, requires_grad=True)
        x1 = torch.randn_like(x0, requires_grad=True)
        x2 = torch.randn_like(x0, requires_grad=True)
        x3 = torch.randn_like(x0, requires_grad=True)

        y_ref = torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)
        y = lse4(x0, x1, x2, x3)

        print(dtype, "max abs diff:", (y - y_ref).abs().max().item())

        go = torch.randn_like(y)
        (gx0_ref, gx1_ref, gx2_ref, gx3_ref) = torch.autograd.grad(y_ref, (x0, x1, x2, x3), go, retain_graph=True)
        (gx0, gx1, gx2, gx3) = torch.autograd.grad(y,     (x0, x1, x2, x3), go)

        print(dtype, "grad max abs diffs:",
              (gx0 - gx0_ref).abs().max().item(),
              (gx1 - gx1_ref).abs().max().item(),
              (gx2 - gx2_ref).abs().max().item(),
              (gx3 - gx3_ref).abs().max().item())
