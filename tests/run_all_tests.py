#!/usr/bin/env python3
"""
Master test runner for GPU-accelerated phylogenetic reconciliation.

This script runs all test suites organized by feature:
- Integration tests: Compare results with AleRax reference implementation
- Gradient tests: Verify automatic differentiation correctness
- Unit tests: Test individual components in isolation
- Performance tests: Benchmark scaling and efficiency (future)
"""

import sys
from pathlib import Path
import subprocess
import time

# Test categories and their corresponding files
TEST_SUITES = {
    'Integration Tests': [
        'tests/integration/test_likelihood_comparison.py'
    ],
    'Gradient Tests': [
        'tests/gradients/test_scatter_rigorous.py',
        'tests/gradients/test_pi_update_rigorous.py'
    ]
}

def run_test_suite(suite_name: str, test_files: list) -> dict:
    """Run a test suite and return results."""
    print(f"\\n{'='*60}")
    print(f"🧪 Running {suite_name}")
    print('='*60)
    
    results = {}
    
    for test_file in test_files:
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = 'MISSING'
            continue
        
        print(f"\\n📋 Running {test_path.name}...")
        start_time = time.time()
        
        try:
            # Run the test
            result = subprocess.run([
                sys.executable, str(test_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_path.name} PASSED ({elapsed:.1f}s)")
                results[test_file] = 'PASS'
            else:
                print(f"❌ {test_path.name} FAILED ({elapsed:.1f}s)")
                print("STDERR:")
                print(result.stderr)
                results[test_file] = 'FAIL'
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_path.name} TIMEOUT (>5min)")
            results[test_file] = 'TIMEOUT'
        except Exception as e:
            print(f"💥 {test_path.name} ERROR: {e}")
            results[test_file] = 'ERROR'
    
    return results


def print_summary(all_results: dict):
    """Print a comprehensive summary of all test results."""
    print("\\n" + "="*60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    total_passed = 0
    
    for suite_name, results in all_results.items():
        print(f"\\n{suite_name}:")
        suite_passed = 0
        suite_total = 0
        
        for test_file, status in results.items():
            test_name = Path(test_file).name
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌', 
                'TIMEOUT': '⏰',
                'ERROR': '💥',
                'MISSING': '⚠️'
            }.get(status, '❓')
            
            print(f"   {status_icon} {test_name}: {status}")
            
            suite_total += 1
            total_tests += 1
            if status == 'PASS':
                suite_passed += 1
                total_passed += 1
        
        print(f"   📈 {suite_name}: {suite_passed}/{suite_total} passed")
    
    print(f"\\n🎯 OVERALL RESULT: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\\n🎉 ALL TESTS PASSED! Your JVP implementation is working correctly!")
        return 0
    else:
        print(f"\\n⚠️  {total_tests - total_passed} tests failed or had issues")
        return 1


def main():
    """Run all test suites and provide comprehensive summary."""
    print("🚀 Starting comprehensive test suite for phylogenetic reconciliation")
    print("This will test your JVP implementation and verify gradient correctness")
    
    all_results = {}
    
    for suite_name, test_files in TEST_SUITES.items():
        suite_results = run_test_suite(suite_name, test_files)
        all_results[suite_name] = suite_results
    
    return print_summary(all_results)


if __name__ == "__main__":
    exit(main())