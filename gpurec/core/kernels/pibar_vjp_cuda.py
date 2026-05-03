"""Runtime CUDA kernels for experimental uniform-Pibar VJP paths."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch


_CUDA_SRC = r"""
extern "C" __global__ void gpurec_pibar_from_ud_shared_padded_fp32(
    const float* __restrict__ Pi,
    const float* __restrict__ pibar_ud,
    const float* __restrict__ pibar_A,
    const unsigned char* __restrict__ side_active,
    const long long* __restrict__ sl,
    const long long* __restrict__ sr,
    const long long* __restrict__ reduce_idx,
    const unsigned char* __restrict__ active_mask,
    const float* __restrict__ pibar_row_max,
    const long long* __restrict__ sp_child1,
    const long long* __restrict__ sp_child2,
    const long long* __restrict__ level_parents,
    float* __restrict__ accumulated_rhs,
    int n_ws,
    int S,
    int pi_stride,
    int rhs_stride,
    int n_levels,
    int max_level_width,
    int use_active_mask,
    int use_side_active
) {
    const int row = blockIdx.x;
    if (row >= 2 * n_ws) {
        return;
    }
    if (use_side_active != 0 && side_active[row] == 0) {
        return;
    }

    const int split_i = (row < n_ws) ? row : (row - n_ws);
    if (use_active_mask != 0) {
        const long long parent_w = reduce_idx[split_i];
        if (active_mask[parent_w] == 0) {
            return;
        }
    }

    const long long child = (row < n_ws) ? sl[split_i] : sr[split_i];
    const int row_base = row * S;
    const long long pi_base = child * (long long)pi_stride;
    const long long rhs_base = child * (long long)rhs_stride;
    const float A = pibar_A[row];
    const float row_max = pibar_row_max[child];

    extern __shared__ float work[];

    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        work[s] = pibar_ud[row_base + s];
    }
    __syncthreads();

    for (int level = 0; level < n_levels; ++level) {
        const long long level_base = level * (long long)max_level_width;
        for (int offset = threadIdx.x; offset < max_level_width; offset += blockDim.x) {
            const long long parent = level_parents[level_base + offset];
            if (parent >= 0 && parent < S) {
                const long long c1 = sp_child1[parent];
                const long long c2 = sp_child2[parent];
                float value = work[parent];
                if (c1 >= 0 && c1 < S) {
                    value += work[c1];
                }
                if (c2 >= 0 && c2 < S) {
                    value += work[c2];
                }
                work[parent] = value;
            }
        }
        __syncthreads();
    }

    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const float p_prime = exp2f(Pi[pi_base + s] - row_max);
        const float contrib = p_prime * (A - work[s]);
        atomicAdd(accumulated_rhs + rhs_base + s, contrib);
    }
}

extern "C" __global__ void gpurec_pibar_from_ud_shared_fp32(
    const float* __restrict__ Pi,
    const float* __restrict__ pibar_ud,
    const float* __restrict__ pibar_A,
    const unsigned char* __restrict__ side_active,
    const long long* __restrict__ sl,
    const long long* __restrict__ sr,
    const long long* __restrict__ reduce_idx,
    const unsigned char* __restrict__ active_mask,
    const float* __restrict__ pibar_row_max,
    const long long* __restrict__ level_ptr,
    const int* __restrict__ level_parent,
    const int* __restrict__ level_child1,
    const int* __restrict__ level_child2,
    float* __restrict__ accumulated_rhs,
    int n_ws,
    int S,
    int pi_stride,
    int rhs_stride,
    int n_levels,
    int use_active_mask,
    int use_side_active
) {
    const int row = blockIdx.x;
    if (row >= 2 * n_ws) {
        return;
    }
    if (use_side_active != 0 && side_active[row] == 0) {
        return;
    }

    const int split_i = (row < n_ws) ? row : (row - n_ws);
    if (use_active_mask != 0) {
        const long long parent_w = reduce_idx[split_i];
        if (active_mask[parent_w] == 0) {
            return;
        }
    }

    const long long child = (row < n_ws) ? sl[split_i] : sr[split_i];
    const int row_base = row * S;
    const long long pi_base = child * (long long)pi_stride;
    const long long rhs_base = child * (long long)rhs_stride;
    const float A = pibar_A[row];
    const float row_max = pibar_row_max[child];

    extern __shared__ float work[];

    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        work[s] = pibar_ud[row_base + s];
    }
    __syncthreads();

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

    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const float p_prime = exp2f(Pi[pi_base + s] - row_max);
        const float contrib = p_prime * (A - work[s]);
        atomicAdd(accumulated_rhs + rhs_base + s, contrib);
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


@lru_cache(maxsize=8)
def _load_cuda_kernel(device_index: int, cc_major: int, cc_minor: int):
    from cuda.bindings import driver, nvrtc

    del device_index  # Part of the cache key; driver modules are context-specific.
    arch = f"--gpu-architecture=compute_{cc_major}{cc_minor}".encode()
    prog = _check_nvrtc(
        nvrtc.nvrtcCreateProgram(
            _CUDA_SRC.encode("utf-8"),
            b"gpurec_pibar_vjp_cuda.cu",
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
    compact_func = _check_driver(
        driver.cuModuleGetFunction(module, b"gpurec_pibar_from_ud_shared_fp32"),
        "cuModuleGetFunction",
    )[0]
    padded_func = _check_driver(
        driver.cuModuleGetFunction(module, b"gpurec_pibar_from_ud_shared_padded_fp32"),
        "cuModuleGetFunction",
    )[0]
    return module, compact_func, padded_func


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


def uniform_cross_pibar_vjp_tree_from_ud_cuda(
    Pi_star,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    accumulated_rhs,
    S,
    *,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
    side_active=None,
    level_parents=None,
    sp_child1=None,
    sp_child2=None,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
):
    """Apply uniform-Pibar VJP from DTS-staged u_d using shared row scratch."""
    n_ws = int(sl.shape[0])
    if n_ws == 0:
        return side_active
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for CUDA Pibar VJP")
    use_compact_levels = (
        compact_level_ptr is not None
        and compact_level_parents is not None
        and compact_level_child1 is not None
        and compact_level_child2 is not None
    )
    use_padded_levels = (
        level_parents is not None
        and sp_child1 is not None
        and sp_child2 is not None
    )
    if not use_compact_levels and not use_padded_levels:
        raise ValueError("compact or padded level tensors are required for CUDA Pibar VJP")

    float_tensors = (Pi_star, pibar_ud, pibar_A, pibar_row_max, accumulated_rhs)
    if any(t.dtype != torch.float32 for t in float_tensors):
        raise ValueError("CUDA Pibar VJP prototype currently supports fp32 only")
    if Pi_star.device.type != "cuda":
        raise ValueError("CUDA Pibar VJP prototype requires CUDA tensors")
    if active_mask is not None and active_mask.dtype is not torch.bool:
        raise ValueError("active_mask must be bool")
    if side_active is not None and side_active.dtype is not torch.bool:
        raise ValueError("side_active must be bool")
    if sl.dtype != torch.int64 or sr.dtype != torch.int64:
        raise ValueError("split child tensors must be int64")
    if reduce_idx is not None and reduce_idx.dtype != torch.int64:
        raise ValueError("reduce_idx must be int64")
    if use_compact_levels:
        if compact_level_ptr.dtype != torch.int64:
            raise ValueError("compact level ptr must be int64")
        if (
            compact_level_parents.dtype != torch.int32
            or compact_level_child1.dtype != torch.int32
            or compact_level_child2.dtype != torch.int32
        ):
            raise ValueError("compact level node tensors must be int32")
    if use_padded_levels:
        if level_parents.dtype != torch.int64:
            raise ValueError("level_parents must be int64")
        if sp_child1.dtype != torch.int64 or sp_child2.dtype != torch.int64:
            raise ValueError("species child tensors must be int64")

    tensors = [
        Pi_star,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        pibar_row_max,
        accumulated_rhs,
    ]
    if use_compact_levels:
        tensors.extend([
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
        ])
    if use_padded_levels:
        tensors.extend([level_parents, sp_child1, sp_child2])
    if reduce_idx is not None:
        tensors.append(reduce_idx)
    if active_mask is not None:
        tensors.append(active_mask)
    if side_active is not None:
        tensors.append(side_active)
    for t in tensors:
        if not torch.is_tensor(t) or t.device != Pi_star.device or not t.is_contiguous():
            raise ValueError("all CUDA Pibar VJP tensors must be contiguous on the same device")

    if pibar_ud.shape != (2 * n_ws, int(S)):
        raise ValueError("pibar_ud must have shape [2 * n_ws, S]")
    if pibar_A.numel() != 2 * n_ws:
        raise ValueError("pibar_A must have one value per split side")
    if side_active is not None and side_active.numel() != 2 * n_ws:
        raise ValueError("side_active must have one value per split side")

    device_index = Pi_star.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(device_index)

    from cuda.bindings import driver

    block = 256
    shared_bytes = int(S) * 4
    props = torch.cuda.get_device_properties(device_index)
    max_shared = getattr(
        props,
        "shared_memory_per_block_optin",
        getattr(props, "shared_memory_per_block", 49152),
    )
    if shared_bytes > int(max_shared):
        raise ValueError("CUDA Pibar VJP shared-memory row scratch is too large")
    active_arg = active_mask.contiguous() if active_mask is not None else pibar_A
    side_arg = side_active.contiguous() if side_active is not None else pibar_A
    reduce_arg = reduce_idx.contiguous() if reduce_idx is not None else sl

    with torch.cuda.device(device_index):
        _, compact_func, padded_func = _load_cuda_kernel(device_index, cc_major, cc_minor)
        if use_compact_levels:
            func = compact_func
            params, _keepalive = _kernel_params(
                ("ptr", Pi_star),
                ("ptr", pibar_ud),
                ("ptr", pibar_A),
                ("ptr", side_arg),
                ("ptr", sl),
                ("ptr", sr),
                ("ptr", reduce_arg),
                ("ptr", active_arg),
                ("ptr", pibar_row_max),
                ("ptr", compact_level_ptr),
                ("ptr", compact_level_parents),
                ("ptr", compact_level_child1),
                ("ptr", compact_level_child2),
                ("ptr", accumulated_rhs),
                ("int", n_ws),
                ("int", S),
                ("int", Pi_star.stride(0)),
                ("int", accumulated_rhs.stride(0)),
                ("int", int(compact_level_ptr.numel()) - 1),
                ("int", 1 if active_mask is not None else 0),
                ("int", 1 if side_active is not None else 0),
            )
            launch_name = "gpurec_pibar_from_ud_shared_fp32"
        else:
            func = padded_func
            params, _keepalive = _kernel_params(
                ("ptr", Pi_star),
                ("ptr", pibar_ud),
                ("ptr", pibar_A),
                ("ptr", side_arg),
                ("ptr", sl),
                ("ptr", sr),
                ("ptr", reduce_arg),
                ("ptr", active_arg),
                ("ptr", pibar_row_max),
                ("ptr", sp_child1),
                ("ptr", sp_child2),
                ("ptr", level_parents),
                ("ptr", accumulated_rhs),
                ("int", n_ws),
                ("int", S),
                ("int", Pi_star.stride(0)),
                ("int", accumulated_rhs.stride(0)),
                ("int", int(level_parents.shape[0])),
                ("int", int(level_parents.shape[1])),
                ("int", 1 if active_mask is not None else 0),
                ("int", 1 if side_active is not None else 0),
            )
            launch_name = "gpurec_pibar_from_ud_shared_padded_fp32"
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
                2 * n_ws,
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
            f"cuLaunchKernel({launch_name})",
        )
    return side_active
