#!/usr/bin/env python3
"""
Pytest-based test runner for GPU-accelerated phylogenetic reconciliation.

This script provides convenient ways to run different test suites:
- All tests
- Quick tests (excluding slow tests)  
- Integration tests only
- Gradient tests only
- Individual test files
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_pytest(args: List[str]) -> int:
    """Run pytest with the given arguments."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest"] + args,
            cwd=Path(__file__).parent.parent,  # Run from project root
            capture_output=False
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def main():
    """Main test runner entry point."""
    if len(sys.argv) < 2:
        print("🧪 GPU-Accelerated Phylogenetic Reconciliation Test Runner")
        print("=" * 60)
        print("\nUsage: python tests/run_tests.py <command> [options]")
        print("\nCommands:")
        print("  all          - Run all tests")
        print("  quick        - Run fast tests only (exclude slow tests)")
        print("  integration  - Run integration tests only") 
        print("  gradient     - Run gradient tests only")
        print("  autograd     - Run automatic differentiation tests only")
        print("  unit         - Run unit tests only")
        print("  slow         - Run slow tests only")
        print("  file <path>  - Run specific test file")
        print("  pytest <args> - Pass arguments directly to pytest")
        print("\nExamples:")
        print("  python tests/run_tests.py all")
        print("  python tests/run_tests.py quick")
        print("  python tests/run_tests.py integration")
        print("  python tests/run_tests.py file tests/gradients/test_scatter_rigorous.py")
        print("  python tests/run_tests.py pytest -k test_scatter_manual_verification -v")
        return 1
    
    command = sys.argv[1].lower()
    
    print("🧪 GPU-Accelerated Phylogenetic Reconciliation Tests")
    print("=" * 60)
    
    if command == "all":
        print("🚀 Running all tests...")
        return run_pytest([])
    
    elif command == "quick":
        print("⚡ Running quick tests (excluding slow tests)...")
        return run_pytest(["-m", "not slow"])
    
    elif command == "integration":
        print("🔗 Running integration tests...")
        return run_pytest(["-m", "integration"])
    
    elif command == "gradient":
        print("📊 Running gradient tests...")
        return run_pytest(["-m", "gradient"])
    
    elif command == "autograd":
        print("🤖 Running automatic differentiation tests...")
        return run_pytest(["-m", "autograd"])
    
    elif command == "unit":
        print("🧱 Running unit tests...")
        return run_pytest(["-m", "unit"])
    
    elif command == "slow":
        print("🐌 Running slow tests...")
        return run_pytest(["-m", "slow"])
    
    elif command == "file":
        if len(sys.argv) < 3:
            print("❌ Error: file command requires a test file path")
            print("Example: python tests/run_tests.py file tests/gradients/test_scatter_rigorous.py")
            return 1
        
        test_file = sys.argv[2]
        print(f"📄 Running test file: {test_file}")
        return run_pytest([test_file])
    
    elif command == "pytest":
        # Pass remaining arguments directly to pytest
        pytest_args = sys.argv[2:]
        print(f"🔧 Running pytest with args: {' '.join(pytest_args)}")
        return run_pytest(pytest_args)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python tests/run_tests.py' without arguments to see available commands.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)