"""
Command-line interface for CCP reconciliation.
Only evaluates the log-likelihood of the gene tree given the species tree and DTL parameters.

In the future: we will implement sampling of reconciliations, and optimization of DTL parameters.
"""

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reconciliation.reconcile import setup_fixed_points


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Log-space CCP reconciliation for numerical stability')
    parser.add_argument('--species', required=True, help='Species tree file (.nwk)')
    parser.add_argument('--gene', required=True, help='Gene tree file (.nwk)')
    parser.add_argument('--delta', type=float, default=1e-10, help='Duplication rate (default: 1e-10)')
    parser.add_argument('--tau', type=float, default=1e-10, help='Transfer rate (default: 1e-10)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=1e-10, help='Loss rate (default: 1e-10)')
    parser.add_argument('--iters', type=int, default=100, help='Maximum iterations (default: 100)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use (default: auto)')
    parser.add_argument('--dtype', choices=['float32', 'float64', 'fp32', 'fp64'], default='float32',
                        help='Computation dtype (default: float32)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output in Pi_update function')
    parser.add_argument('--use-triton', action='store_true', default=True, help='Use Triton LSE kernels when available (default: True)')
    parser.add_argument('--no-triton', dest='use_triton', action='store_false', help='Disable Triton LSE kernels')
    parser.add_argument('--compare-triton', action='store_true', default=False, help='Compute both Triton and Torch results where applicable and report differences')
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    try:
        # Resolve dtype
        if args.dtype in ('float32', 'fp32'):
            dtype = torch.float32
        elif args.dtype in ('float64', 'fp64'):
            dtype = torch.float64
        else:
            dtype = torch.float32

        result = setup_fixed_points(
            args.species, args.gene,
            delta=args.delta, tau=args.tau, lambda_param=args.lambda_param,
            max_iters_E=args.iters, max_iters_Pi=args.iters, 
            device=device, dtype=dtype, debug=args.debug, use_triton=args.use_triton,
            compare_triton=args.compare_triton
        )
        
        print(f"\n🎯 Final Results:")
        print(f"   Log-likelihood: {result['log_likelihood']:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
