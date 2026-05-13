"""Runtime CUDA kernels for opt-in no-split wave-backward paths."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import sysconfig

import torch


_CUDA_SRC = r"""
extern "C" __device__ __forceinline__ float gpurec_warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

extern "C" __device__ __forceinline__ float gpurec_block_sum(float val) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val = gpurec_warp_sum(val);
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();
    const int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        val = gpurec_warp_sum(val);
    }
    return val;
}

extern "C" __device__ __forceinline__ void gpurec_weights_nosplit(
    const float* __restrict__ Pi,
    const float* __restrict__ Pibar,
    const float* __restrict__ row_max_ptr,
    const float* __restrict__ mt,
    const float* __restrict__ DL,
    const float* __restrict__ Ebar,
    const float* __restrict__ E,
    const float* __restrict__ SL1,
    const float* __restrict__ SL2,
    const int* __restrict__ sp_child1,
    const int* __restrict__ sp_child2,
    const int* __restrict__ leaf_species,
    const float* __restrict__ leaf_logp,
    int ws,
    int w,
    int s,
    int S,
    int stride,
    float* q0,
    float* q1,
    float* q2,
    float* q3,
    float* q4,
    float* q5,
    float* pcoeff
) {
    const int global_row = ws + w;
    const int row_base = global_row * stride;
    const float neg = -1.0e30f;
    const float pi = Pi[row_base + s];
    const float pibar = Pibar[row_base + s];
    const int c1 = sp_child1[s];
    const int c2 = sp_child2[s];
    const float pi_c1 = (c1 >= 0 && c1 < S) ? Pi[row_base + c1] : neg;
    const float pi_c2 = (c2 >= 0 && c2 < S) ? Pi[row_base + c2] : neg;

    const float t0 = DL[s] + pi;
    const float t1 = pi + Ebar[s];
    const float t2 = pibar + E[s];
    const float t3 = SL1[s] + pi_c1;
    const float t4 = SL2[s] + pi_c2;
    const int leaf_s = leaf_species[global_row];
    const float t5 = (leaf_s == s) ? leaf_logp[s] : neg;

    float m = fmaxf(fmaxf(fmaxf(t0, t1), fmaxf(t2, t3)), fmaxf(t4, t5));
    const float m_safe = (m > -1.0e29f) ? m : 0.0f;
    const float e0 = exp2f(t0 - m_safe);
    const float e1 = exp2f(t1 - m_safe);
    const float e2 = exp2f(t2 - m_safe);
    const float e3 = exp2f(t3 - m_safe);
    const float e4 = exp2f(t4 - m_safe);
    const float e5 = exp2f(t5 - m_safe);
    const float sum = e0 + e1 + e2 + e3 + e4 + e5;
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    *q0 = e0 * inv_sum;
    *q1 = e1 * inv_sum;
    *q2 = e2 * inv_sum;
    *q3 = e3 * inv_sum;
    *q4 = e4 * inv_sum;
    *q5 = e5 * inv_sum;

    const float row_max = row_max_ptr[global_row];
    const float inv_denom = (pibar > -1.0e29f) ? exp2f(row_max + mt[s] - pibar) : 0.0f;
    *pcoeff = (*q2) * inv_denom;
}

extern "C" __global__ void gpurec_wave_backward_nosplit_uniform_fp32(
    const float* __restrict__ Pi,
    const float* __restrict__ Pibar,
    const float* __restrict__ pibar_row_max,
    const float* __restrict__ rhs,
    const unsigned char* __restrict__ active_mask,
    const float* __restrict__ mt,
    const float* __restrict__ DL,
    const float* __restrict__ Ebar,
    const float* __restrict__ E,
    const float* __restrict__ SL1,
    const float* __restrict__ SL2,
    const int* __restrict__ sp_child1,
    const int* __restrict__ sp_child2,
    const int* __restrict__ sp_parent,
    const int* __restrict__ leaf_species,
    const float* __restrict__ leaf_logp,
    const long long* __restrict__ level_ptr,
    const int* __restrict__ level_parent,
    const int* __restrict__ level_child1,
    const int* __restrict__ level_child2,
    float* __restrict__ v_out,
    float* __restrict__ grad_log_pD,
    float* __restrict__ grad_log_pS,
    float* __restrict__ grad_E,
    float* __restrict__ grad_Ebar,
    float* __restrict__ grad_E_s1,
    float* __restrict__ grad_E_s2,
    float* __restrict__ grad_mt,
    int ws,
    int W,
    int S,
    int stride,
    int neumann_terms,
    int n_levels,
    int use_active_mask,
    int correction_mode,
    int event_grad_mode
) {
    const int w = blockIdx.x;
    if (w >= W) {
        return;
    }

    extern __shared__ float sh[];
    float* term = sh;
    float* work = term + S;
    float* vacc = work + S;
    float* diag = vacc + S;
    float* pcoef = diag + S;
    float* sl1w = pcoef + S;
    float* sl2w = sl1w + S;

    __shared__ float A_shared;
    const int out_base = w * S;
    const int row_base = (ws + w) * stride;

    const bool active = (use_active_mask == 0) || (active_mask[w] != 0);
    if (!active) {
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            v_out[out_base + s] = 0.0f;
        }
        return;
    }

    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float q0, q1, q2, q3, q4, q5, pc;
        gpurec_weights_nosplit(
            Pi, Pibar, pibar_row_max, mt, DL, Ebar, E, SL1, SL2,
            sp_child1, sp_child2, leaf_species, leaf_logp,
            ws, w, s, S, stride,
            &q0, &q1, &q2, &q3, &q4, &q5, &pc
        );
        const float r = rhs[out_base + s];
        term[s] = r;
        vacc[s] = r;
        diag[s] = q0 + q1;
        pcoef[s] = pc;
        sl1w[s] = q3;
        sl2w[s] = q4;
    }
    __syncthreads();

    for (int iter = 0; iter < neumann_terms; ++iter) {
        float local_A = 0.0f;
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            const float u = term[s] * pcoef[s];
            work[s] = u;
            local_A += u;
        }
        const float A = gpurec_block_sum(local_A);
        if (threadIdx.x == 0) {
            A_shared = A;
        }
        __syncthreads();

        if (correction_mode != 0) {
            for (int level = 0; level < n_levels; ++level) {
                const long long start = level_ptr[level];
                const long long end = level_ptr[level + 1];
                for (long long idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
                    const int parent = level_parent[idx];
                    const int c1 = level_child1[idx];
                    const int c2 = level_child2[idx];
                    float value = (parent >= 0 && parent < S) ? work[parent] : 0.0f;
                    if (c1 >= 0 && c1 < S) {
                        value += work[c1];
                    }
                    if (c2 >= 0 && c2 < S) {
                        value += work[c2];
                    }
                    if (parent >= 0 && parent < S) {
                        work[parent] = value;
                    }
                }
                __syncthreads();
            }
        }

        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            const float row_max = pibar_row_max[ws + w];
            const float pprime = exp2f(Pi[row_base + s] - row_max);
            float result = term[s] * diag[s] + pprime * (A_shared - work[s]);
            const int parent = sp_parent[s];
            if (parent >= 0 && parent < S) {
                const int pc1 = sp_child1[parent];
                const float side_w = (pc1 == s) ? sl1w[parent] : sl2w[parent];
                result += term[parent] * side_w;
            }
            work[s] = result;
            vacc[s] += result;
        }
        __syncthreads();

        float* tmp = term;
        term = work;
        work = tmp;
        __syncthreads();
    }

    float local_pD = 0.0f;
    float local_pS = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const float v = vacc[s];
        v_out[out_base + s] = v;

        float q0, q1, q2, q3, q4, q5, pc;
        gpurec_weights_nosplit(
            Pi, Pibar, pibar_row_max, mt, DL, Ebar, E, SL1, SL2,
            sp_child1, sp_child2, leaf_species, leaf_logp,
            ws, w, s, S, stride,
            &q0, &q1, &q2, &q3, &q4, &q5, &pc
        );

        const float aw0 = v * q0;
        const float aw1 = v * q1;
        const float aw2 = v * q2;
        const float aw3 = v * q3;
        const float aw4 = v * q4;
        const float aw5 = v * q5;
        const float aw345 = aw3 + aw4 + aw5;

        if (event_grad_mode == 1) {
            atomicAdd(grad_log_pD + s, aw0);
            atomicAdd(grad_log_pS + s, aw345);
        } else {
            local_pD += aw0;
            local_pS += aw345;
        }
        atomicAdd(grad_E + s, aw0 + aw2);
        atomicAdd(grad_Ebar + s, aw1);
        atomicAdd(grad_E_s1 + s, aw4);
        atomicAdd(grad_E_s2 + s, aw3);
        atomicAdd(grad_mt + s, aw2);
    }
    if (event_grad_mode == 0) {
        const float sum_pD = gpurec_block_sum(local_pD);
        const float sum_pS = gpurec_block_sum(local_pS);
        if (threadIdx.x == 0) {
            atomicAdd(grad_log_pD, sum_pD);
            atomicAdd(grad_log_pS, sum_pS);
        }
    }
}
"""


def _check_driver(result, what: str):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"{what} failed with CUDA error {err}")
    return result[1:] if len(result) > 1 else ()


def _check_nvrtc(result, what: str, prog=None):
    err = result[0]
    if int(err) != 0:
        log = ""
        if prog is not None:
            try:
                from cuda.bindings import nvrtc

                size = _check_nvrtc(
                    nvrtc.nvrtcGetProgramLogSize(prog),
                    "nvrtcGetProgramLogSize",
                )[0]
                buf = bytearray(size)
                nvrtc.nvrtcGetProgramLog(prog, buf)
                log = buf.decode("utf-8", errors="replace")
            except Exception:
                log = ""
        raise RuntimeError(f"{what} failed with NVRTC error {err}\n{log}")
    return result[1:] if len(result) > 1 else ()


@lru_cache(maxsize=1)
def _preload_nvrtc_builtins() -> None:
    """Load wheel-provided NVRTC builtins when they are not on ld.so's path."""
    roots = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(Path(purelib))
    for root in roots:
        for pattern in (
            "nvidia/cu[0-9]*/lib/libnvrtc-builtins.so*",
            "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so*",
        ):
            for lib in sorted(root.glob(pattern), reverse=True):
                if lib.is_file():
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    return


@lru_cache(maxsize=4)
def _load_cuda_kernel(device_index: int, cc_major: int, cc_minor: int):
    from cuda.bindings import driver, nvrtc

    del device_index
    _preload_nvrtc_builtins()
    arch = f"--gpu-architecture=compute_{cc_major}{cc_minor}".encode()
    prog = _check_nvrtc(
        nvrtc.nvrtcCreateProgram(
            _CUDA_SRC.encode("utf-8"),
            b"gpurec_wave_backward_cuda.cu",
            0,
            [],
            [],
        ),
        "nvrtcCreateProgram",
    )[0]
    _check_nvrtc(
        nvrtc.nvrtcCompileProgram(
            prog,
            3,
            [arch, b"--std=c++14", b"--use_fast_math"],
        ),
        "nvrtcCompileProgram",
        prog,
    )
    ptx_size = _check_nvrtc(nvrtc.nvrtcGetPTXSize(prog), "nvrtcGetPTXSize")[0]
    ptx = bytearray(ptx_size)
    _check_nvrtc(nvrtc.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX", prog)

    _check_driver(driver.cuInit(0), "cuInit")
    module = _check_driver(driver.cuModuleLoadData(bytes(ptx)), "cuModuleLoadData")[0]
    func = _check_driver(
        driver.cuModuleGetFunction(
            module,
            b"gpurec_wave_backward_nosplit_uniform_fp32",
        ),
        "cuModuleGetFunction",
    )[0]
    return module, func


def _kernel_params(*values):
    params = (ctypes.c_void_p * len(values))()
    keepalive = []
    for i, (kind, value) in enumerate(values):
        if kind == "ptr":
            obj = ctypes.c_void_p(0 if value is None else int(value.data_ptr()))
        elif kind == "int":
            obj = ctypes.c_int(int(value))
        else:
            raise ValueError(f"unknown CUDA param kind {kind!r}")
        keepalive.append(obj)
        params[i] = ctypes.cast(ctypes.pointer(obj), ctypes.c_void_p)
    return params, keepalive


def wave_backward_uniform_nosplit_cuda(
    Pi_star,
    Pibar_star,
    ws,
    W,
    S,
    rhs,
    mt_squeezed,
    DL_const,
    Ebar,
    E,
    SL1_const,
    SL2_const,
    sp_child1,
    sp_child2,
    sp_parent,
    leaf_species_idx,
    leaf_logp,
    pibar_row_max,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    accum_param_grads,
    *,
    active_mask=None,
    neumann_terms=3,
    correction_mode="tree",
):
    """Run the opt-in fp32 CUDA no-split self-loop kernel."""
    if accum_param_grads is None or len(accum_param_grads) != 7:
        raise ValueError("CUDA no-split wave path requires seven gradient tensors")
    float_tensors = (
        Pi_star,
        Pibar_star,
        rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        leaf_logp,
        pibar_row_max,
        *accum_param_grads,
    )
    if any(t.dtype != torch.float32 for t in float_tensors):
        raise ValueError("CUDA no-split wave path currently supports fp32 only")
    if Pi_star.device.type != "cuda":
        raise ValueError("CUDA no-split wave path requires CUDA tensors")
    if active_mask is not None and active_mask.dtype is not torch.bool:
        raise ValueError("active_mask must be a bool tensor")
    if accum_param_grads[0].numel() not in (1, int(S)):
        raise ValueError("grad_log_pD must be scalar or [S]")
    if accum_param_grads[1].numel() != accum_param_grads[0].numel():
        raise ValueError("grad_log_pD and grad_log_pS must have matching layouts")
    for grad in accum_param_grads[2:]:
        if grad.numel() != int(S):
            raise ValueError("E/Ebar/transfer gradients must be [S]")

    tensors = [
        Pi_star,
        Pibar_star,
        rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        sp_parent,
        leaf_species_idx,
        leaf_logp,
        pibar_row_max,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        *accum_param_grads,
    ]
    if active_mask is not None:
        tensors.append(active_mask)
    for t in tensors:
        if not torch.is_tensor(t) or t.device != Pi_star.device or not t.is_contiguous():
            raise ValueError("all CUDA no-split wave tensors must be contiguous on the same device")

    if sp_child1.dtype != torch.int32 or sp_child2.dtype != torch.int32:
        raise ValueError("species child tensors must be int32")
    if sp_parent.dtype != torch.int32 or leaf_species_idx.dtype != torch.int32:
        raise ValueError("species parent and leaf index tensors must be int32")
    if compact_level_ptr.dtype != torch.int64:
        raise ValueError("compact level ptr must use int64 offsets")
    if (
        compact_level_parents.dtype != torch.int32
        or compact_level_child1.dtype != torch.int32
        or compact_level_child2.dtype != torch.int32
    ):
        raise ValueError("compact level tensors must use int32 node ids")

    v_k = torch.empty((int(W), int(S)), device=Pi_star.device, dtype=torch.float32)
    device_index = Pi_star.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(device_index)

    from cuda.bindings import driver

    block = 256
    shared_bytes = int(S) * 7 * 4

    correction_mode_value = 0 if str(correction_mode).lower() in ("self", "diagonal", "0") else 1
    event_grad_mode = 1 if accum_param_grads[0].numel() == int(S) else 0
    n_levels = int(compact_level_ptr.numel()) - 1
    active_arg = active_mask.contiguous() if active_mask is not None else rhs
    params, _keepalive = _kernel_params(
        ("ptr", Pi_star),
        ("ptr", Pibar_star),
        ("ptr", pibar_row_max),
        ("ptr", rhs),
        ("ptr", active_arg),
        ("ptr", mt_squeezed),
        ("ptr", DL_const),
        ("ptr", Ebar),
        ("ptr", E),
        ("ptr", SL1_const),
        ("ptr", SL2_const),
        ("ptr", sp_child1),
        ("ptr", sp_child2),
        ("ptr", sp_parent),
        ("ptr", leaf_species_idx),
        ("ptr", leaf_logp),
        ("ptr", compact_level_ptr),
        ("ptr", compact_level_parents),
        ("ptr", compact_level_child1),
        ("ptr", compact_level_child2),
        ("ptr", v_k),
        ("ptr", accum_param_grads[0]),
        ("ptr", accum_param_grads[1]),
        ("ptr", accum_param_grads[2]),
        ("ptr", accum_param_grads[3]),
        ("ptr", accum_param_grads[4]),
        ("ptr", accum_param_grads[5]),
        ("ptr", accum_param_grads[6]),
        ("int", ws),
        ("int", W),
        ("int", S),
        ("int", Pi_star.stride(0)),
        ("int", neumann_terms),
        ("int", n_levels),
        ("int", 1 if active_mask is not None else 0),
        ("int", correction_mode_value),
        ("int", event_grad_mode),
    )

    with torch.cuda.device(device_index):
        _, func = _load_cuda_kernel(device_index, cc_major, cc_minor)
        _check_driver(
            driver.cuFuncSetAttribute(
                func,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes,
            ),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
        )
        stream = torch.cuda.current_stream(Pi_star.device).cuda_stream
        _check_driver(
            driver.cuLaunchKernel(
                func,
                int(W),
                1,
                1,
                block,
                1,
                1,
                shared_bytes,
                stream,
                params,
                0,
            ),
            "cuLaunchKernel(gpurec_wave_backward_nosplit_uniform_fp32)",
        )
    return v_k
