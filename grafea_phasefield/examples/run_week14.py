"""
Week 14 Runner: Three-Point Bending & L-Shaped Panel Benchmarks
================================================================

Runs both Week 14 benchmarks:
1. Three-Point Bending (TPB) - Mode-I fracture with bending stress gradient
2. L-Shaped Panel - Critical crack NUCLEATION test (no pre-crack!)

Usage:
    python run_week14.py                  # Run both benchmarks (quick mode)
    python run_week14.py --tpb            # Run only TPB
    python run_week14.py --lpanel         # Run only L-panel
    python run_week14.py --full           # Full resolution (slow)
    python run_week14.py --quick          # Quick validation (default)
    python run_week14.py --very-quick     # Very quick pipeline test
"""

import sys
import os
import argparse
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.tpb_benchmark import (
    TPB_PARAMS,
    run_tpb_benchmark,
    validate_tpb_results,
    TPBBenchmark,
    quick_tpb_test,
    very_quick_tpb_test,
)

from benchmarks.l_shaped_panel_benchmark import (
    L_PANEL_PARAMS,
    run_l_panel_benchmark,
    validate_l_panel_results,
    compare_with_original_grafea,
    LPanelBenchmark,
    quick_l_panel_test,
    very_quick_l_panel_test,
)


def run_tpb(mode: str = 'quick', save_plots: bool = True) -> dict:
    """
    Run Three-Point Bending benchmark.

    Args:
        mode: 'full', 'quick', or 'very-quick'
        save_plots: Whether to save result plots

    Returns:
        Dictionary with results and validation
    """
    print("\n" + "=" * 70)
    print("  PART A: THREE-POINT BENDING BENCHMARK")
    print("=" * 70)

    start = time.time()

    if mode == 'full':
        benchmark = TPBBenchmark()
        results = benchmark.run(verbose=True)
    elif mode == 'quick':
        results = quick_tpb_test(n_steps=50, verbose=True)
    else:
        results = very_quick_tpb_test(n_steps=20, verbose=True)

    runtime = time.time() - start

    # Validate
    params = TPB_PARAMS.copy()
    if mode == 'quick':
        params.update({'l0': 2.0})
    elif mode == 'very-quick':
        params.update({'l0': 4.0})

    validation = validate_tpb_results(results, params, verbose=True)

    # Summary
    print(f"\n  TPB Runtime: {runtime:.1f}s")
    n_passed = sum(1 for v in validation.values() if v['passed'])
    n_total = len(validation)
    print(f"  Validation: {n_passed}/{n_total} checks passed")

    if save_plots:
        try:
            os.makedirs('results/week14', exist_ok=True)
            if mode == 'full':
                benchmark.plot_results(save_path='results/week14/tpb_results.png')
        except Exception as e:
            print(f"  Plot saving skipped: {e}")

    return {
        'results': results,
        'validation': validation,
        'runtime': runtime,
    }


def run_lpanel(mode: str = 'quick', save_plots: bool = True) -> dict:
    """
    Run L-Shaped Panel benchmark (nucleation test).

    Args:
        mode: 'full', 'quick', or 'very-quick'
        save_plots: Whether to save result plots

    Returns:
        Dictionary with results and validation
    """
    print("\n" + "=" * 70)
    print("  PART B: L-SHAPED PANEL BENCHMARK (NUCLEATION TEST)")
    print("=" * 70)

    start = time.time()

    if mode == 'full':
        benchmark = LPanelBenchmark()
        results = benchmark.run(verbose=True)
    elif mode == 'quick':
        results = quick_l_panel_test(n_steps=50, verbose=True)
    else:
        results = very_quick_l_panel_test(n_steps=20, verbose=True)

    runtime = time.time() - start

    # Validate
    params = L_PANEL_PARAMS.copy()
    if mode == 'quick':
        params.update({'l0': 10.0})
    elif mode == 'very-quick':
        params.update({'l0': 20.0})

    validation = validate_l_panel_results(results, params, verbose=True)

    # Comparison with original GraFEA
    comparison = compare_with_original_grafea(results)

    print(f"\n  L-Panel Runtime: {runtime:.1f}s")
    n_passed = sum(1 for v in validation.values() if v['passed'])
    n_total = len(validation)
    print(f"  Validation: {n_passed}/{n_total} checks passed")

    if comparison['advantage_demonstrated']:
        print("  KEY RESULT: Phase-field nucleation advantage DEMONSTRATED!")
    else:
        print("  Note: Nucleation not detected in this run.")

    if save_plots:
        try:
            os.makedirs('results/week14', exist_ok=True)
            if mode == 'full':
                benchmark.plot_results(save_path='results/week14/lpanel_results.png')
        except Exception as e:
            print(f"  Plot saving skipped: {e}")

    return {
        'results': results,
        'validation': validation,
        'comparison': comparison,
        'runtime': runtime,
    }


def print_week14_summary(tpb_output: dict, lpanel_output: dict):
    """Print comprehensive Week 14 summary."""
    print("\n" + "=" * 70)
    print("  WEEK 14 SUMMARY REPORT")
    print("=" * 70)

    # TPB Summary
    print("\n--- Three-Point Bending ---")
    tpb_val = tpb_output['validation']
    for criterion, data in tpb_val.items():
        status = "PASS" if data['passed'] else "FAIL"
        print(f"  [{status}] {criterion}")
    print(f"  Peak force: {tpb_output['results'].peak_force:.4f}")
    print(f"  Runtime: {tpb_output['runtime']:.1f}s")

    # L-Panel Summary
    print("\n--- L-Shaped Panel (NUCLEATION) ---")
    lpanel_val = lpanel_output['validation']
    for criterion, data in lpanel_val.items():
        status = "PASS" if data['passed'] else "FAIL"
        print(f"  [{status}] {criterion}")
    print(f"  Nucleation: {'YES' if lpanel_output['results'].nucleation_detected else 'NO'}")
    print(f"  Peak force: {lpanel_output['results'].peak_force:.4f}")
    print(f"  Runtime: {lpanel_output['runtime']:.1f}s")

    # Cross-benchmark comparison
    print("\n--- Cross-Benchmark Comparison ---")
    print(f"  {'Benchmark':<25} {'Peak Force':<15} {'Key Result':<30}")
    print(f"  {'-'*25} {'-'*15} {'-'*30}")
    print(f"  {'Three-Point Bending':<25} "
          f"{tpb_output['results'].peak_force:<15.4f} "
          f"{'Vertical crack from notch':<30}")

    nuc_status = 'Nucleation at corner' if lpanel_output['results'].nucleation_detected else 'No nucleation'
    print(f"  {'L-Shaped Panel':<25} "
          f"{lpanel_output['results'].peak_force:<15.4f} "
          f"{nuc_status:<30}")

    # GraFEA comparison table
    print("\n--- Phase-Field vs Original GraFEA ---")
    print(f"  {'Capability':<25} {'Original GraFEA':<20} {'Edge-Based PF':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Crack nucleation':<25} {'No':<20} "
          f"{'Yes' if lpanel_output['results'].nucleation_detected else 'No':<20}")
    print(f"  {'Pre-crack required':<25} {'Yes':<20} {'No':<20}")
    print(f"  {'L-panel test':<25} {'Would fail':<20} "
          f"{'Passed' if lpanel_output['results'].nucleation_detected else 'Check params':<20}")

    total_time = tpb_output['runtime'] + lpanel_output['runtime']
    print(f"\n  Total runtime: {total_time:.1f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Week 14: Three-Point Bending & L-Shaped Panel Benchmarks'
    )
    parser.add_argument('--tpb', action='store_true',
                       help='Run only Three-Point Bending')
    parser.add_argument('--lpanel', action='store_true',
                       help='Run only L-Shaped Panel')
    parser.add_argument('--full', action='store_true',
                       help='Full resolution (slow)')
    parser.add_argument('--quick', action='store_true', default=True,
                       help='Quick validation (default)')
    parser.add_argument('--very-quick', action='store_true',
                       help='Very quick pipeline test')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Determine mode
    if args.full:
        mode = 'full'
    elif args.very_quick:
        mode = 'very-quick'
    else:
        mode = 'quick'

    save_plots = not args.no_plots
    run_both = not args.tpb and not args.lpanel

    print("=" * 70)
    print("  Week 14: Three-Point Bending & L-Shaped Panel")
    print(f"  Mode: {mode}")
    print("=" * 70)

    tpb_output = None
    lpanel_output = None

    if args.tpb or run_both:
        tpb_output = run_tpb(mode=mode, save_plots=save_plots)

    if args.lpanel or run_both:
        lpanel_output = run_lpanel(mode=mode, save_plots=save_plots)

    if tpb_output and lpanel_output:
        print_week14_summary(tpb_output, lpanel_output)


if __name__ == '__main__':
    main()
