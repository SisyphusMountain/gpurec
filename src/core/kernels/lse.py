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

@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def _lse4_kernel(OUT, X0, X1, X2, X3,
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

    e0 = tl.exp2(x0 - m_safe); e1 = tl.exp2(x1 - m_safe)
    e2 = tl.exp2(x2 - m_safe); e3 = tl.exp2(x3 - m_safe)

    s01 = e0 + e1
    s23 = e2 + e3
    s   = s01 + s23

    y = tl.where(s == 0, NEG_INF, tl.log2(s) + m)
    tl.store(OUT + offs, y, mask=mask)
    
@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def _lse5_kernel(OUT, X0, X1, X2, X3, X4,
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
    m   = tl.maximum(tl.maximum(m01, m23), x4)
    m_safe = tl.where(m == NEG_INF, 0.0, m)

    e0 = tl.exp2(x0 - m_safe); e1 = tl.exp2(x1 - m_safe)
    e2 = tl.exp2(x2 - m_safe); e3 = tl.exp2(x3 - m_safe)
    e4 = tl.exp2(x4 - m_safe)

    s01 = e0 + e1
    s23 = e2 + e3
    s   = (s01 + s23) + e4

    y = tl.where(s == 0, NEG_INF, tl.log2(s) + m)
    tl.store(OUT + offs, y, mask=mask)

@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def _lse7_kernel(OUT, X0, X1, X2, X3, X4, X5, X6,
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

    e0 = tl.exp2(x0 - m_safe)
    e1 = tl.exp2(x1 - m_safe)
    e2 = tl.exp2(x2 - m_safe)
    e3 = tl.exp2(x3 - m_safe)
    e4 = tl.exp2(x4 - m_safe)
    e5 = tl.exp2(x5 - m_safe)
    e6 = tl.exp2(x6 - m_safe)

    s01 = e0 + e1
    s23 = e2 + e3
    s45 = e4 + e5
    s   = (s01 + s23) + (s45 + e6)

    # When all inputs are -inf, return -inf directly
    y = tl.where(all_inf, NEG_INF, tl.where(s == 0, NEG_INF, tl.log2(s) + m))
    tl.store(OUT + offs, y, mask=mask)


def lse4_triton_pair(x0, x1, x2, x3):
    return LSE4PairFn.apply(x0, x1, x2, x3)

def lse5_triton_pair(x0, x1, x2, x3, x4):
    return LSE5PairFn.apply(x0, x1, x2, x3, x4)

def lse7_triton_pair(x0,x1,x2,x3,x4,x5,x6):
    return LSE7PairFn.apply(x0, x1, x2, x3, x4, x5, x6)



# --------------------------
# Autograd Functions
# --------------------------
class _LSEBaseFn(torch.autograd.Function):
    @staticmethod
    def _grad_weights(xs, y):
        # weights = exp(xi - y); guard y = -inf -> weights = 0
        finite = torch.isfinite(y)
        weights = [torch.where(finite, torch.exp2(x - y), torch.zeros_like(x)) for x in xs]
        return weights


class LSE4PairFn(_LSEBaseFn):
    @staticmethod
    def forward(ctx, x0, x1, x2, x3):

        out = torch.empty_like(x0)
        n = out.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        _lse4_kernel[grid](out, x0, x1, x2, x3, n)
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
        _lse5_kernel[grid](out, x0, x1, x2, x3, x4, n)
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
        _lse7_kernel[grid](out, x0, x1, x2, x3, x4, x5, x6, n)
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
    w0 = tl.where(finite, tl.exp2(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp2(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp2(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp2(x3 - y), ZERO)

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
    w0 = tl.where(finite, tl.exp2(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp2(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp2(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp2(x3 - y), ZERO)
    w4 = tl.where(finite, tl.exp2(x4 - y), ZERO)

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
    w0 = tl.where(finite, tl.exp2(x0 - y), ZERO)
    w1 = tl.where(finite, tl.exp2(x1 - y), ZERO)
    w2 = tl.where(finite, tl.exp2(x2 - y), ZERO)
    w3 = tl.where(finite, tl.exp2(x3 - y), ZERO)
    w4 = tl.where(finite, tl.exp2(x4 - y), ZERO)
    w5 = tl.where(finite, tl.exp2(x5 - y), ZERO)
    w6 = tl.where(finite, tl.exp2(x6 - y), ZERO)

    tl.store(G0 + offs, gy * w0, mask=mask)
    tl.store(G1 + offs, gy * w1, mask=mask)
    tl.store(G2 + offs, gy * w2, mask=mask)
    tl.store(G3 + offs, gy * w3, mask=mask)
    tl.store(G4 + offs, gy * w4, mask=mask)
    tl.store(G5 + offs, gy * w5, mask=mask)
    tl.store(G6 + offs, gy * w6, mask=mask)


