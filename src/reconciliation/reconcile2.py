"""CLI entry point for likelihood_2 based reconciliation."""

import argparse
import math
from typing import Dict, Optional

import torch

from ..core.ccp import (
    build_ccp_from_single_tree,
    get_root_clade_id,
    build_ccp_helpers,
    build_clade_species_mapping,
)
from ..core.tree_helpers import build_species_helpers
from ..core.preprocess_cpp import (
    preprocess_fast,
    compare_with_python_precomputed,
)
from ..core.likelihood_2 import E_fixed_point, Pi_fixed_point, compute_log_likelihood


def setup_fixed_points_likelihood2(
    species_tree_path: str,
    gene_tree_path: str,
    delta: float = 1e-10,
    tau: float = 1e-10,
    lambda_param: float = 1e-10,
    max_iters_E: int = 1_000_000_000, # large numbers by default to ensure convergence
    max_iters_Pi: int = 1_000_000_000,
    tol_E: float = 1e-9,
    tol_Pi: float = 1e-9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
    use_theta: bool = False,
    theta: Optional[torch.Tensor] = None,
    use_cpp_preprocess: bool = True,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        print(f"Parameters: delta={delta}, tau={tau}, lambda={lambda_param}")
    import time
    start = time.time()
    if use_cpp_preprocess:
        pre = preprocess_fast(
            species_tree_path,
            gene_tree_path,
            device=device,
            dtype=dtype,
        )
        if debug:
            match, details = compare_with_python_precomputed(
                species_tree_path,
                gene_tree_path,
                pre,
                device=device,
                dtype=dtype,
            )
            if not match:
                print("C++ preprocessing mismatch detected against Python pipeline:")
                for key, entry in details.items():
                    status = "OK" if entry["match"] else "DIFF"
                    print(f"  {status:<4} {key}: {entry['info']}")
                raise ValueError("C++ preprocessing did not reproduce Python tensors")
        species_helpers = pre["species_helpers"]
        ccp_helpers = pre["ccp_helpers"]
        root_clade_id = pre["root_clade_id"]
        clade_species = pre["clade_species_map"]
        log_clade_species_map = torch.log(clade_species + 1e-45)
        log_clade_species_map = torch.where(
            clade_species > 0,
            log_clade_species_map,
            torch.full_like(log_clade_species_map, float("-inf")),
        )
    else:
        ccp = build_ccp_from_single_tree(gene_tree_path, debug=False)
        species_helpers = build_species_helpers(species_tree_path, device, dtype)
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        log_clade_species_map = torch.log(clade_species_map + 1e-45)
        log_clade_species_map[clade_species_map == 0] = float("-inf")
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        root_clade_id = get_root_clade_id(ccp, debug=False)
    t0 = time.time()
    print(f"preprocessing time: {t0 - start} s")
    if use_theta:
        if theta is None:
            raise ValueError("theta parameter required when use_theta is True")
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        else:
            theta = theta.to(device=device, dtype=dtype)
        param_tensor = theta
        if debug:
            exp_theta = torch.exp(theta)
            delta_val = exp_theta[0].item()
            tau_val = exp_theta[1].item()
            lambda_val = exp_theta[2].item()
            print(
                "Theta parameters: "
                f"delta={delta_val:.6f}, tau={tau_val:.6f}, lambda={lambda_val:.6f}"
            )
    else:
        param_tensor = torch.tensor(
            [
                math.log(max(delta, 1e-10)),
                math.log(max(tau, 1e-10)),
                math.log(max(lambda_param, 1e-10)),
            ],
            dtype=dtype,
            device=device,
        )
        if debug:
            print(
                "Individual parameters: "
                f"delta={delta}, tau={tau}, lambda={lambda_param}"
            )

    result_E = E_fixed_point(
        species_helpers=species_helpers,
        theta=param_tensor,
        max_iters=max_iters_E,
        tolerance=tol_E,
        return_components=True,
        warm_start_E=None,
        dtype=dtype,
    )
    t1 = time.time()
    print(f"E computation time: {t1 - t0} s")
    E = result_E["E"]
    E_s1 = result_E["E_s1"]
    E_s2 = result_E["E_s2"]
    Ebar = result_E["E_bar"]
    if debug:
        print(f"number of iterations E: {result_E['iterations']}")
    result_Pi = Pi_fixed_point(
        ccp_helpers=ccp_helpers,
        species_helpers=species_helpers,
        clade_species_map=log_clade_species_map,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        theta=param_tensor,
        max_iters=max_iters_Pi,
        tolerance=tol_Pi,
        warm_start_Pi=None,
    )
    t2 = time.time()
    print(f"Pi computation time {t2 - t1} s")
    Pi = result_Pi["Pi"]
    if debug:
        print(f"number of iterations for Pi: {result_Pi['iterations']}")
    log_likelihood = compute_log_likelihood(Pi, root_clade_id)
    if debug:
        print(f"Log-likelihood: {log_likelihood:.6f}")
    if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
        if debug:
            print("WARNING: numerical instability detected")
    else:
        if debug:
            print("No numerical instability detected")

    return {
        "root_clade_id": root_clade_id,
        "theta": param_tensor,
        "log_likelihood": float(log_likelihood),
        "Pi": Pi,
        "E": E,
        "Ebar": Ebar,
        "E_s1": E_s1,
        "E_s2": E_s2,
        "species_helpers": species_helpers,
        "clade_species_map": log_clade_species_map,
        "ccp_helpers": ccp_helpers,
    }


def _resolve_dtype(name: str) -> torch.dtype:
    attr = name.lower()
    if not hasattr(torch, attr):
        raise ValueError(f"Unsupported dtype '{name}'")
    dtype = getattr(torch, attr)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Attribute '{name}' is not a torch dtype")
    return dtype


def _resolve_device(name: Optional[str]) -> Optional[torch.device]:
    if name is None:
        return None
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run likelihood_2 fixed-point solver and print log-likelihood.",
    )
    parser.add_argument("species_tree_path")
    parser.add_argument("gene_tree_path")
    parser.add_argument("--delta", type=float, default=1e-10)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lambda-param", dest="lambda_param", type=float, default=1e-10)
    parser.add_argument("--max-iters-E", dest="max_iters_E", type=int, default=1000)
    parser.add_argument("--max-iters-Pi", dest="max_iters_Pi", type=int, default=1000)
    parser.add_argument("--tol-E", dest="tol_E", type=float, default=1e-9)
    parser.add_argument("--tol-Pi", dest="tol_Pi", type=float, default=1e-9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-theta", action="store_true")
    parser.add_argument(
        "--theta",
        nargs=3,
        type=float,
        metavar=("LOG_DELTA", "LOG_TAU", "LOG_LAMBDA"),
    )
    parser.add_argument("--no-use-triton", dest="use_triton", action="store_false")
    parser.add_argument("--compare-triton", action="store_true")
    parser.add_argument("--no-cpp-preprocess", dest="use_cpp_preprocess", action="store_false")
    parser.set_defaults(use_triton=True, compare_triton=False)
    parser.set_defaults(use_cpp_preprocess=True)

    args = parser.parse_args()
    dtype = _resolve_dtype(args.dtype)
    device = _resolve_device(args.device)

    if args.use_theta and args.theta is None:
        parser.error("--theta must be provided when --use-theta is set")
    if not args.use_theta and args.theta is not None:
        parser.error("--theta is only valid when --use-theta is set")

    result = setup_fixed_points_likelihood2(
        species_tree_path=args.species_tree_path,
        gene_tree_path=args.gene_tree_path,
        delta=args.delta,
        tau=args.tau,
        lambda_param=args.lambda_param,
        max_iters_E=args.max_iters_E,
        max_iters_Pi=args.max_iters_Pi,
        tol_E=args.tol_E,
        tol_Pi=args.tol_Pi,
        device=device,
        dtype=dtype,
        debug=args.debug,
        use_theta=args.use_theta,
        theta=args.theta,
        use_triton=args.use_triton,
        compare_triton=args.compare_triton,
        use_cpp_preprocess=args.use_cpp_preprocess,
    )

    print(result["log_likelihood"])


if __name__ == "__main__":
    main()
