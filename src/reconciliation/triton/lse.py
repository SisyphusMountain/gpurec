import torch
import triton
import triton.language as tl

# Use a safe device default that doesn't depend on Triton internals at import time
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32

# --------------------------
# Utilities
# --------------------------
CONFIGS = [
    triton.Config({'BLOCK':  128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK':  256}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK':  512}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
]

# --------------------------
# Torch baselines
# --------------------------
def lse4_torch(x0, x1, x2, x3):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)

def lse5_torch(x0, x1, x2, x3, x4):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3, x4], dim=0), dim=0)

def lse7_torch(x0, x1, x2, x3, x4, x5, x6):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3, x4, x5, x6], dim=0), dim=0)


@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse4_pair_kernel(OUT, X0, X1, X2, X3,
                     N, BLOCK: tl.constexpr):
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
    tl.store(OUT + offs, y, mask=mask)

def lse4_triton_pair(x0, x1, x2, x3):
    return LSE4PairFn.apply(x0, x1, x2, x3)



@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse5_pair_kernel(OUT, X0, X1, X2, X3, X4,
                     N, BLOCK: tl.constexpr):
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
    x4 = tl.load(X4 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m    = tl.maximum(tl.maximum(m01, m23), x4)
    m_safe = tl.where(m == NEG_INF, 0.0, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
    e4 = tl.exp(x4 - m_safe)

    # pairwise tree for 5: ((e0+e1)+(e2+e3)) + e4
    s01 = e0 + e1
    s23 = e2 + e3
    s   = (s01 + s23) + e4

    y = tl.where(s == 0, NEG_INF, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


def lse5_triton_pair(x0, x1, x2, x3, x4):
    return LSE5PairFn.apply(x0, x1, x2, x3, x4)


@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse7_pair_kernel(OUT, X0, X1, X2, X3, X4, X5, X6,
                     N, BLOCK: tl.constexpr):
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
    x4 = tl.load(X4 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")
    x5 = tl.load(X5 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")
    x6 = tl.load(X6 + offs, mask=mask, other=NEG_INF, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m45 = tl.maximum(x4, x5)
    m   = tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, x6))
    
    # When all inputs are -inf, m is -inf and we should return -inf
    all_inf = (m == NEG_INF)
    m_safe = tl.where(all_inf, 0.0, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
    e4 = tl.exp(x4 - m_safe); e5 = tl.exp(x5 - m_safe)
    e6 = tl.exp(x6 - m_safe)

    # pairwise tree for 7: (((e0+e1)+(e2+e3)) + ((e4+e5)+e6))
    s01 = e0 + e1
    s23 = e2 + e3
    s45 = e4 + e5
    s   = (s01 + s23) + (s45 + e6)

    # When all inputs are -inf, return -inf directly
    y = tl.where(all_inf, NEG_INF, tl.where(s == 0, NEG_INF, tl.log(s) + m))
    tl.store(OUT + offs, y, mask=mask)


def lse7_triton_pair(x0,x1,x2,x3,x4,x5,x6):
    return LSE7PairFn.apply(x0, x1, x2, x3, x4, x5, x6)


# --------------------------
# Dim-reduction variants (K = 4,5,7)
# --------------------------

@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse4_reduce_kernel(OUT, X, N,
                       DTYPE: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ninf = -float('inf')
    ZERO = tl.zeros((), DTYPE)
    stride = N  # x shape = [4, N] contiguous -> stride along K is N

    x0 = tl.load(X + 0 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x1 = tl.load(X + 1 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x2 = tl.load(X + 2 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x3 = tl.load(X + 3 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(m01, m23)
    m_safe = tl.where(m == ninf, ZERO, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
    s01 = e0 + e1
    s23 = e2 + e3
    s   = s01 + s23
    y = tl.where(s == 0, ninf, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse5_reduce_kernel(OUT, X, N,
                       DTYPE: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ninf = -float('inf')
    ZERO = tl.zeros((), DTYPE)
    stride = N

    x0 = tl.load(X + 0 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x1 = tl.load(X + 1 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x2 = tl.load(X + 2 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x3 = tl.load(X + 3 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x4 = tl.load(X + 4 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(tl.maximum(m01, m23), x4)
    m_safe = tl.where(m == ninf, ZERO, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
    e4 = tl.exp(x4 - m_safe)
    s01 = e0 + e1
    s23 = e2 + e3
    s   = (s01 + s23) + e4
    y = tl.where(s == 0, ninf, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse7_reduce_kernel(OUT, X, N,
                       DTYPE: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ninf = -float('inf')
    ZERO = tl.zeros((), DTYPE)
    stride = N

    x0 = tl.load(X + 0 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x1 = tl.load(X + 1 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x2 = tl.load(X + 2 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x3 = tl.load(X + 3 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x4 = tl.load(X + 4 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x5 = tl.load(X + 5 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x6 = tl.load(X + 6 * stride + offs, mask=mask, other=ninf, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m45 = tl.maximum(x4, x5)
    m   = tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, x6))
    m_safe = tl.where(m == ninf, ZERO, m)

    e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
    e4 = tl.exp(x4 - m_safe); e5 = tl.exp(x5 - m_safe)
    e6 = tl.exp(x6 - m_safe)

    s01 = e0 + e1
    s23 = e2 + e3
    s45 = e4 + e5
    s   = (s01 + s23) + (s45 + e6)
    y = tl.where(s == 0, ninf, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


def _lseK_reduce(x: torch.Tensor, dim: int, K: int):
    if not x.is_cuda:
        return torch.logsumexp(x, dim=dim)
    dim = int(dim)
    if x.dim() == 0:
        raise ValueError("x must have at least 1 dimension")
    dim = dim % x.dim()
    if x.shape[dim] != K:
        raise ValueError(f"Expected size {K} at dim={dim}, got {x.shape[dim]}")

    xp = x.movedim(dim, 0).contiguous()  # [K, *rest]
    rest_shape = xp.shape[1:]
    N = int(xp[0].numel())
    x2d = xp.view(K, N)
    out = torch.empty((N,), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    D = tl.float64 if x.dtype == torch.float64 else tl.float32
    if K == 4:
        lse4_reduce_kernel[grid](out, x2d, N, DTYPE=D)
    elif K == 5:
        lse5_reduce_kernel[grid](out, x2d, N, DTYPE=D)
    elif K == 7:
        lse7_reduce_kernel[grid](out, x2d, N, DTYPE=D)
    else:
        raise ValueError("K must be 4, 5 or 7")
    return out.view(*rest_shape)


def lse4_reduce(x: torch.Tensor, dim: int):
    return _lseK_reduce(x, dim, K=4)


def lse5_reduce(x: torch.Tensor, dim: int):
    return _lseK_reduce(x, dim, K=5)


def lse7_reduce(x: torch.Tensor, dim: int):
    return _lseK_reduce(x, dim, K=7)

# --------------------------
# Autograd Functions
# --------------------------
class _LSEBaseFn(torch.autograd.Function):
    @staticmethod
    def _grad_weights(xs, y):
        # weights = exp(xi - y); guard y = -inf -> weights = 0
        finite = torch.isfinite(y)
        weights = [torch.where(finite, torch.exp(x - y), torch.zeros_like(x)) for x in xs]
        return weights


class LSE4PairFn(_LSEBaseFn):
    @staticmethod
    def forward(ctx, x0, x1, x2, x3):
        if any(not x.is_cuda for x in [x0, x1, x2, x3]):
            raise RuntimeError("LSE4PairFn only supports CUDA tensors")
        if any(not x.is_contiguous() for x in [x0, x1, x2, x3]):
            raise RuntimeError("LSE4PairFn only supports contiguous tensors")
        out = torch.empty_like(x0)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        lse4_pair_kernel[grid](out, x0, x1, x2, x3, n)
        ctx.save_for_backward(x0, x1, x2, x3, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x0, x1, x2, x3, y = ctx.saved_tensors
        if x0.is_cuda:
            go = grad_out
            if go.ndim == 0 or go.numel() == 1:
                go = go.expand_as(y).contiguous()
            elif not go.is_contiguous():
                go = go.contiguous()
            # Triton fused backward: gi = grad_out * exp(xi - y)
            g0 = torch.empty_like(x0)
            g1 = torch.empty_like(x1)
            g2 = torch.empty_like(x2)
            g3 = torch.empty_like(x3)
            n = x0.numel()
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
            D = tl.float64 if x0.dtype == torch.float64 else tl.float32
            lse4_bwd_kernel[grid](g0, g1, g2, g3, x0, x1, x2, x3, y, go, n, DTYPE=D)
            return g0, g1, g2, g3
        else:
            w0, w1, w2, w3 = _LSEBaseFn._grad_weights([x0, x1, x2, x3], y)
            return grad_out * w0, grad_out * w1, grad_out * w2, grad_out * w3


class LSE5PairFn(_LSEBaseFn):
    @staticmethod
    def forward(ctx, x0, x1, x2, x3, x4):
        out = torch.empty_like(x0)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        lse5_pair_kernel[grid](out, x0, x1, x2, x3, x4, n)
        ctx.save_for_backward(x0, x1, x2, x3, x4, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x0, x1, x2, x3, x4, y = ctx.saved_tensors
        if x0.is_cuda:
            go = grad_out
            if go.ndim == 0 or go.numel() == 1:
                go = go.expand_as(y).contiguous()
            elif not go.is_contiguous():
                go = go.contiguous()
            g0 = torch.empty_like(x0)
            g1 = torch.empty_like(x1)
            g2 = torch.empty_like(x2)
            g3 = torch.empty_like(x3)
            g4 = torch.empty_like(x4)
            n = x0.numel()
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
            D = tl.float64 if x0.dtype == torch.float64 else tl.float32
            lse5_bwd_kernel[grid](g0, g1, g2, g3, g4, x0, x1, x2, x3, x4, y, go, n, DTYPE=D)
            return g0, g1, g2, g3, g4
        else:
            w0, w1, w2, w3, w4 = _LSEBaseFn._grad_weights([x0, x1, x2, x3, x4], y)
            return grad_out * w0, grad_out * w1, grad_out * w2, grad_out * w3, grad_out * w4


class LSE7PairFn(_LSEBaseFn):
    @staticmethod
    def forward(ctx, x0, x1, x2, x3, x4, x5, x6):
        out = torch.empty_like(x0)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        lse7_pair_kernel[grid](out, x0, x1, x2, x3, x4, x5, x6, n)
        ctx.save_for_backward(x0, x1, x2, x3, x4, x5, x6, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x0, x1, x2, x3, x4, x5, x6, y = ctx.saved_tensors
        if x0.is_cuda:
            go = grad_out
            if go.ndim == 0 or go.numel() == 1:
                go = go.expand_as(y).contiguous()
            elif not go.is_contiguous():
                go = go.contiguous()
            g0 = torch.empty_like(x0)
            g1 = torch.empty_like(x1)
            g2 = torch.empty_like(x2)
            g3 = torch.empty_like(x3)
            g4 = torch.empty_like(x4)
            g5 = torch.empty_like(x5)
            g6 = torch.empty_like(x6)
            n = x0.numel()
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
            D = tl.float64 if x0.dtype == torch.float64 else tl.float32
            lse7_bwd_kernel[grid](g0, g1, g2, g3, g4, g5, g6,
                                  x0, x1, x2, x3, x4, x5, x6,
                                  y, go, n, DTYPE=D)
            return g0, g1, g2, g3, g4, g5, g6
        else:
            w = _LSEBaseFn._grad_weights([x0, x1, x2, x3, x4, x5, x6], y)
            return tuple(grad_out * wi for wi in w)


# --------------------------
# Triton backward kernels
# --------------------------
@triton.jit
def lse4_bwd_kernel(G0, G1, G2, G3,
                    X0, X1, X2, X3,
                    Y, GY,
                    N,
                    DTYPE: tl.constexpr,
                    BLOCK: tl.constexpr = 256):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    ZERO = tl.zeros((), DTYPE)
    NEG_INF = tl.constexpr(float("-inf"))
    x0 = tl.load(X0 + offs, mask=mask)
    x1 = tl.load(X1 + offs, mask=mask)
    x2 = tl.load(X2 + offs, mask=mask)
    x3 = tl.load(X3 + offs, mask=mask)
    y  = tl.load(Y  + offs, mask=mask)
    gy = tl.load(GY + offs, mask=mask)

    finite = y != NEG_INF
    w0 = tl.where(finite, tl.exp(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp(x3 - y), ZERO)

    tl.store(G0 + offs, gy * w0, mask=mask)
    tl.store(G1 + offs, gy * w1, mask=mask)
    tl.store(G2 + offs, gy * w2, mask=mask)
    tl.store(G3 + offs, gy * w3, mask=mask)


@triton.jit
def lse5_bwd_kernel(G0, G1, G2, G3, G4,
                    X0, X1, X2, X3, X4,
                    Y, GY,
                    N,
                    DTYPE: tl.constexpr,
                    BLOCK: tl.constexpr = 256):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    ZERO = tl.zeros((), DTYPE)
    NEG_INF = tl.constexpr(float("-inf"))
    x0 = tl.load(X0 + offs, mask=mask)
    x1 = tl.load(X1 + offs, mask=mask)
    x2 = tl.load(X2 + offs, mask=mask)
    x3 = tl.load(X3 + offs, mask=mask)
    x4 = tl.load(X4 + offs, mask=mask)
    y  = tl.load(Y  + offs, mask=mask)
    gy = tl.load(GY + offs, mask=mask)

    finite = y != NEG_INF
    w0 = tl.where(finite, tl.exp(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp(x3 - y), ZERO)
    w4 = tl.where(finite, tl.exp(x4 - y), ZERO)

    tl.store(G0 + offs, gy * w0, mask=mask)
    tl.store(G1 + offs, gy * w1, mask=mask)
    tl.store(G2 + offs, gy * w2, mask=mask)
    tl.store(G3 + offs, gy * w3, mask=mask)
    tl.store(G4 + offs, gy * w4, mask=mask)


@triton.jit
def lse7_bwd_kernel(G0, G1, G2, G3, G4, G5, G6,
                    X0, X1, X2, X3, X4, X5, X6,
                    Y, GY,
                    N,
                    DTYPE: tl.constexpr,
                    BLOCK: tl.constexpr = 256):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    ZERO = tl.zeros((), DTYPE)
    NEG_INF = tl.constexpr(float("-inf"))
    x0 = tl.load(X0 + offs, mask=mask)
    x1 = tl.load(X1 + offs, mask=mask)
    x2 = tl.load(X2 + offs, mask=mask)
    x3 = tl.load(X3 + offs, mask=mask)
    x4 = tl.load(X4 + offs, mask=mask)
    x5 = tl.load(X5 + offs, mask=mask)
    x6 = tl.load(X6 + offs, mask=mask)
    y  = tl.load(Y  + offs, mask=mask)
    gy = tl.load(GY + offs, mask=mask)

    finite = y != NEG_INF
    w0 = tl.where(finite, tl.exp(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp(x3 - y), ZERO)
    w4 = tl.where(finite, tl.exp(x4 - y), ZERO)
    w5 = tl.where(finite, tl.exp(x5 - y), ZERO)
    w6 = tl.where(finite, tl.exp(x6 - y), ZERO)

    tl.store(G0 + offs, gy * w0, mask=mask)
    tl.store(G1 + offs, gy * w1, mask=mask)
    tl.store(G2 + offs, gy * w2, mask=mask)
    tl.store(G3 + offs, gy * w3, mask=mask)
    tl.store(G4 + offs, gy * w4, mask=mask)
    tl.store(G5 + offs, gy * w5, mask=mask)
    tl.store(G6 + offs, gy * w6, mask=mask)


# --------------------------
# Lightweight tests (module-local)
# --------------------------
def _sprinkle_neginf(x: torch.Tensor, p=0.02):
    if p <= 0:
        return x
    mask = torch.rand_like(x) < p
    return x.masked_fill(mask, float('-inf'))


def test_lse5_reduce_2d():
    """Verify lse5_reduce works on 2D tensors along either dimension."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    torch.manual_seed(0)

    # dim=0 reduction (K at dim 0)
    N = 257
    x = torch.randn(5, N, device=device, dtype=dtype)
    x = _sprinkle_neginf(x, p=0.03)
    y_ref = torch.logsumexp(x, dim=0)
    y = lse5_reduce(x, dim=0)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)

    # dim=1 reduction (K at dim 1)
    M = 193
    x2 = torch.randn(M, 5, device=device, dtype=dtype)
    x2 = _sprinkle_neginf(x2, p=0.03)
    y2_ref = torch.logsumexp(x2, dim=1)
    y2 = lse5_reduce(x2, dim=1)
    assert torch.allclose(y2, y2_ref, rtol=1e-6, atol=1e-6)


def test_lse7_reduce_2d():
    """Verify lse7_reduce works on 2D tensors along either dimension."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    torch.manual_seed(1)

    # dim=0 reduction (K at dim 0)
    N = 129
    x = torch.randn(7, N, device=device, dtype=dtype)
    x = _sprinkle_neginf(x, p=0.03)
    y_ref = torch.logsumexp(x, dim=0)
    y = lse7_reduce(x, dim=0)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)

    # dim=1 reduction (K at dim 1)
    M = 211
    x2 = torch.randn(M, 7, device=device, dtype=dtype)
    x2 = _sprinkle_neginf(x2, p=0.03)
    y2_ref = torch.logsumexp(x2, dim=1)
    y2 = lse7_reduce(x2, dim=1)
    assert torch.allclose(y2, y2_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    # Run tests manually: python -m src.reconciliation.triton.lse
    test_lse5_reduce_2d()
    test_lse7_reduce_2d()
    print("OK: lse5_reduce and lse7_reduce 2D tests passed")

    # Optional benchmark for lse7_triton_pair on 7 tensors of shape [2000, 200]
    def benchmark_lse7_pair_2k_200(dtype=torch.float32):
        if not torch.cuda.is_available():
            print("CUDA not available; skipping lse7_triton_pair benchmark")
            return
        torch.manual_seed(0)
        shape = (2000, 200)
        xs = [torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(7)]
        # warm-up / compile
        _ = lse7_triton_pair(*xs)
        _ = lse7_torch(*xs)
        torch.cuda.synchronize()

        def _bytes_moved(shape, dtype):
            itemsize = torch.finfo(dtype).bits // 8
            elems = shape[0] * shape[1]
            return (7 * elems + elems) * itemsize  # 7 reads + 1 write

        def _gbps(bytes_moved, ms):
            return (bytes_moved / 1e9) / (ms / 1e3)

        bytes_mv = _bytes_moved(shape, dtype)
        q = [0.5, 0.2, 0.8]
        ms_tri, _, _ = triton.testing.do_bench(lambda: lse7_triton_pair(*xs), quantiles=q)
        ms_ref, _, _ = triton.testing.do_bench(lambda: lse7_torch(*xs), quantiles=q)
        gbps_tri = _gbps(bytes_mv, ms_tri)
        gbps_ref = _gbps(bytes_mv, ms_ref)

        # Accuracy check
        y_tri = lse7_triton_pair(*xs)
        y_ref = lse7_torch(*xs)
        max_abs = (y_tri - y_ref).abs().max().item()
        ok = torch.allclose(y_tri, y_ref, rtol=1e-6, atol=1e-6)

        print(f"\nBenchmark lse7_triton_pair on 7x{shape}:")
        print(f"  Triton: {ms_tri:.3f} ms  ({gbps_tri:.2f} GB/s)")
        print(f"  Torch : {ms_ref:.3f} ms  ({gbps_ref:.2f} GB/s)")
        print(f"  Allclose: {ok}, max |diff| = {max_abs:.3e}")

    benchmark_lse7_pair_2k_200()
