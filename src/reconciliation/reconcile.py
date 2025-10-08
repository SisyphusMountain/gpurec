# """CLI entry point for likelihood_2 based reconciliation."""

# import argparse
# import math
# from typing import Dict, Optional

# import torch

# # from ..core.batching import build_clade_species_map_from_indices
# from ..core import (
#     build_ccp_from_single_tree,
#     get_root_clade_id,
#     build_ccp_helpers,
#     build_clade_species_mapping,
# )
# from ..core import build_species_helpers
# from ..core.preprocess_cpp import (
#     preprocess_fast,
#     compare_with_python_precomputed,
# )
# from ..core.likelihood import E_fixed_point, Pi_fixed_point, compute_log_likelihood


# def setup_fixed_points_likelihood(
#     species_tree_path: str,
#     gene_tree_path: str,
#     delta: float = 1e-10,
#     tau: float = 1e-10,
#     lambda_param: float = 1e-10,
#     max_iters_E: int = 1_000_000_000, # large numbers by default to ensure convergence
#     max_iters_Pi: int = 1_000_000_000,
#     tol_E: float = 1e-7,
#     tol_Pi: float = 1e-7,
#     device: Optional[torch.device] = None,
#     dtype: torch.dtype = torch.float32,
#     debug: bool = False,
#     theta: Optional[torch.Tensor] = None,
# ) -> Dict:
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if debug:
#         print(f"Parameters: delta={delta}, tau={tau}, lambda={lambda_param}")
#     import time
#     start = time.time()

#     pre = preprocess_fast(
#         species_tree_path,
#         gene_tree_path,
#         device=device,
#         dtype=dtype,
#     )
#     species_helpers = pre["species_helpers"]
#     ccp_helpers = pre["ccp_helpers"]
#     root_clade_id = pre["root_clade_id"]
#     C = ccp_helpers["C"]
#     S = species_helpers["S"]
#     leaf_row_index = pre["leaf_row_index"]
#     leaf_col_index = pre["leaf_col_index"]
#     t0 = time.time()
#     print(f"preprocessing time: {t0 - start} s")

#     if theta is None:
#         pass
#     else:
#         pass

#     result_E = E_fixed_point(
#         species_helpers=species_helpers,
#         theta=theta,
#         max_iters=max_iters_E,
#         tolerance=tol_E,
#         return_components=True,
#         warm_start_E=None,
#         dtype=dtype,
#     )
#     t1 = time.time()
#     print(f"E computation time: {t1 - t0} s")
#     E = result_E["E"]
#     E_s1 = result_E["E_s1"]
#     E_s2 = result_E["E_s2"]
#     Ebar = result_E["E_bar"]
#     result_Pi = Pi_fixed_point(
#         ccp_helpers=ccp_helpers,
#         species_helpers=species_helpers,
#         leaf_row_index=leaf_row_index,
#         leaf_col_index=leaf_col_index,
#         E=E,
#         Ebar=Ebar,
#         E_s1=E_s1,
#         E_s2=E_s2,
#         theta=theta,
#         max_iters=max_iters_Pi,
#         tolerance=tol_Pi,
#         warm_start_Pi=None,
#     )
#     t2 = time.time()
#     print(f"Pi computation time {t2 - t1} s")
#     Pi = result_Pi["Pi"]
#     clade_species_map = result_Pi["clade_species_map"]
#     log_likelihood = compute_log_likelihood(Pi, root_clade_id)


#     return {
#         "root_clade_id": root_clade_id,
#         "theta": theta,
#         "log_likelihood": float(log_likelihood),
#         "Pi": Pi,
#         "E": E,
#         "Ebar": Ebar,
#         "E_s1": E_s1,
#         "E_s2": E_s2,
#         "species_helpers": species_helpers,
#         "clade_species_map": clade_species_map,
#         "ccp_helpers": ccp_helpers,
#     }




# def _resolve_dtype(name: str) -> torch.dtype:
#     attr = name.lower()
#     if not hasattr(torch, attr):
#         raise ValueError(f"Unsupported dtype '{name}'")
#     dtype = getattr(torch, attr)
#     if not isinstance(dtype, torch.dtype):
#         raise ValueError(f"Attribute '{name}' is not a torch dtype")
#     return dtype


# def _resolve_device(name: Optional[str]) -> Optional[torch.device]:
#     if name is None:
#         return None
#     return torch.device(name)


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Run likelihood_2 fixed-point solver and print log-likelihood.",
#     )
#     parser.add_argument("species_tree_path")
#     parser.add_argument("gene_tree_path")
#     parser.add_argument("--delta", type=float, default=1e-10)
#     parser.add_argument("--tau", type=float, default=0.05)
#     parser.add_argument("--lambda-param", dest="lambda_param", type=float, default=1e-10)
#     parser.add_argument("--max-iters-E", dest="max_iters_E", type=int, default=1000)
#     parser.add_argument("--max-iters-Pi", dest="max_iters_Pi", type=int, default=1000)
#     parser.add_argument("--tol-E", dest="tol_E", type=float, default=1e-9)
#     parser.add_argument("--tol-Pi", dest="tol_Pi", type=float, default=1e-9)
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--dtype", type=str, default="float64")
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--use-theta", action="store_true")
#     parser.add_argument(
#         "--theta",
#         nargs=3,
#         type=float,
#         metavar=("LOG_DELTA", "LOG_TAU", "LOG_LAMBDA"),
#     )
#     parser.add_argument("--no-use-triton", dest="use_triton", action="store_false")
#     parser.add_argument("--compare-triton", action="store_true")
#     parser.add_argument("--no-cpp-preprocess", dest="use_cpp_preprocess", action="store_false")
#     parser.set_defaults(use_triton=True, compare_triton=False)
#     parser.set_defaults(use_cpp_preprocess=True)

#     args = parser.parse_args()
#     dtype = _resolve_dtype(args.dtype)
#     device = _resolve_device(args.device)

#     if args.use_theta and args.theta is None:
#         parser.error("--theta must be provided when --use-theta is set")
#     if not args.use_theta and args.theta is not None:
#         parser.error("--theta is only valid when --use-theta is set")

#     result = setup_fixed_points_likelihood2(
#         species_tree_path=args.species_tree_path,
#         gene_tree_path=args.gene_tree_path,
#         delta=args.delta,
#         tau=args.tau,
#         lambda_param=args.lambda_param,
#         max_iters_E=args.max_iters_E,
#         max_iters_Pi=args.max_iters_Pi,
#         tol_E=args.tol_E,
#         tol_Pi=args.tol_Pi,
#         device=device,
#         dtype=dtype,
#         debug=args.debug,
#         use_theta=args.use_theta,
#         theta=args.theta,
#         use_triton=args.use_triton,
#         compare_triton=args.compare_triton,
#         use_cpp_preprocess=args.use_cpp_preprocess,
#     )

#     print(result["log_likelihood"])


# if __name__ == "__main__":
#     main()
