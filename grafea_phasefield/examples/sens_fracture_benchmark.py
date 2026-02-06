"""
SENS Fracture Benchmark Example
================================

Single Edge Notched Shear (SENS) test - canonical Mode-II fracture benchmark.

This example demonstrates:
1. Setting up the SENS geometry and mesh
2. Initializing the pre-existing crack
3. Running the phase-field fracture simulation with shear loading
4. Validating crack angle against literature (~70 degrees)
5. Comparing SENS (shear) vs SENT (tension) results

Key difference from SENT:
- SENT: Vertical tension -> horizontal crack (Mode I)
- SENS: Horizontal shear -> curved crack upward ~70 deg (Mode II)

The SENS benchmark is critical for validating the tension-compression split:
if the split is incorrect, the crack will go straight (like SENT) instead of curving.

Reference:
- Miehe et al. (2010) "Thermodynamically consistent phase-field models of fracture"
- Ambati et al. (2015) "A review on phase-field models of brittle fracture"
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmarks.sens_benchmark import (
    SENS_PARAMS,
    run_sens_benchmark,
    validate_sens_results,
    compare_with_literature,
    quick_sens_test,
    SENSBenchmark,
    generate_sens_mesh,
    create_precrack_damage,
    analyze_stress_state,
    validate_tension_compression_split,
)


def run_full_benchmark():
    """
    Run the full SENS benchmark with default parameters.

    This is the complete simulation as specified in the benchmark,
    but can be slow due to the fine mesh required near the crack.
    """
    print("=" * 70)
    print("SENS Fracture Benchmark - Full Simulation")
    print("=" * 70)

    results = run_sens_benchmark(verbose=True)

    # Validate results
    print("\n")
    validation = validate_sens_results(results, verbose=True)

    # Compare with literature
    print("\n")
    comparison = compare_with_literature(results, verbose=True)

    # Try to plot if matplotlib is available
    try:
        plot_results(results)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")

    return results, validation


def run_quick_benchmark():
    """
    Run a quick version of the SENS benchmark for testing.

    Uses fewer load steps and a coarser mesh for faster execution.
    """
    print("=" * 70)
    print("SENS Fracture Benchmark - Quick Test")
    print("=" * 70)

    results = quick_sens_test(n_steps=30, verbose=True)

    # Try to plot if matplotlib is available
    try:
        plot_results(results)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")

    # Basic validation
    print("\n--- Quick Validation ---")
    print(f"Peak shear force: {results.peak_force:.4f}")
    print(f"Displacement at peak: {results.displacement_at_peak:.6f}")
    print(f"Final crack length: {results.crack_length[-1]:.4f}")
    print(f"Final crack angle: {results.final_crack_angle:.1f} deg")

    return results


def run_with_class_interface():
    """
    Demonstrate using the SENSBenchmark class interface.
    """
    print("=" * 70)
    print("SENS Benchmark - Using Class Interface")
    print("=" * 70)

    params = {
        "n_steps": 50,
        "h_fine": 0.0075,
        "h_coarse": 0.03,
    }

    benchmark = SENSBenchmark(params)
    results = benchmark.run(verbose=True)
    validation = benchmark.validate(verbose=True)

    try:
        benchmark.plot_results(save_path="sens_results.png")
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")

    return benchmark


def demonstrate_mesh_generation():
    """
    Demonstrate SENS mesh generation with visualization.
    """
    print("=" * 70)
    print("SENS Mesh Generation Demo")
    print("=" * 70)

    params_coarse = {
        "L": 1.0,
        "h_fine": 0.02,
        "h_coarse": 0.1,
        "refinement_band": 0.15,
    }

    params_fine = {
        "L": 1.0,
        "h_fine": 0.005,
        "h_coarse": 0.05,
        "refinement_band": 0.15,
    }

    print("\nCoarse mesh:")
    mesh_coarse = generate_sens_mesh(params_coarse)
    print(f"  Nodes: {mesh_coarse.n_nodes}")
    print(f"  Elements: {mesh_coarse.n_elements}")
    print(f"  Edges: {mesh_coarse.n_edges}")

    print("\nFine mesh:")
    mesh_fine = generate_sens_mesh(params_fine)
    print(f"  Nodes: {mesh_fine.n_nodes}")
    print(f"  Elements: {mesh_fine.n_elements}")
    print(f"  Edges: {mesh_fine.n_edges}")

    # Demonstrate pre-crack initialization
    print("\nPre-crack damage initialization:")
    d_coarse = create_precrack_damage(mesh_coarse, 0.5, 0.5, 0.015)
    print(f"  Max damage: {np.max(d_coarse):.4f}")
    print(f"  Damaged edges (d > 0.5): {np.sum(d_coarse > 0.5)}")
    print(f"  Damaged edges (d > 0.9): {np.sum(d_coarse > 0.9)}")

    try:
        import matplotlib.pyplot as plt
        from postprocess.visualization import plot_mesh, plot_damage_field

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        plot_mesh(mesh_coarse, ax=axes[0], show_edges=True)
        axes[0].set_title(f"Coarse Mesh ({mesh_coarse.n_elements} elements)")

        plot_damage_field(mesh_coarse, d_coarse, ax=axes[1])
        axes[1].set_title("Pre-crack Damage Field")

        plt.tight_layout()
        plt.savefig("sens_mesh_demo.png", dpi=150)
        print("\nMesh visualization saved to 'sens_mesh_demo.png'")
        plt.show()

    except ImportError:
        print("\nNote: matplotlib not available, skipping mesh visualization")


def plot_results(results):
    """
    Create comprehensive result plots for SENS.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Load-displacement curve (shear)
    ax = axes[0, 0]
    ax.plot(results.displacement * 1000, results.shear_force, "b-", lw=2)
    ax.axhline(
        y=results.peak_force,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Peak = {results.peak_force:.4f}",
    )
    ax.set_xlabel("Shear Displacement (mm x 10^3)")
    ax.set_ylabel("Shear Force")
    ax.set_title("Load-Displacement Curve (Shear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Crack angle evolution
    ax = axes[0, 1]
    ax.plot(results.displacement * 1000, results.crack_angle, "g-", lw=2)
    ax.axhline(y=70, color="r", linestyle="--", alpha=0.7, label="Expected (70 deg)")
    ax.set_xlabel("Shear Displacement (mm x 10^3)")
    ax.set_ylabel("Crack Angle (deg)")
    ax.set_title("Crack Angle Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Energy evolution
    ax = axes[1, 0]
    ax.plot(
        results.displacement * 1000, results.strain_energy, "b-", label="Strain", lw=2
    )
    ax.plot(
        results.displacement * 1000, results.surface_energy, "r-", label="Surface", lw=2
    )
    ax.plot(
        results.displacement * 1000, results.total_energy, "k--", label="Total", lw=2
    )
    ax.set_xlabel("Shear Displacement (mm x 10^3)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Crack path
    ax = axes[1, 1]
    if len(results.crack_path) > 0:
        ax.plot(
            results.crack_path[:, 0],
            results.crack_path[:, 1],
            "r.-",
            lw=2,
            ms=4,
            label="Computed path",
        )

    # Expected path
    tip_x = SENS_PARAMS["crack_length"]
    tip_y = SENS_PARAMS["crack_y"]
    expected_angle = np.radians(70)
    end_x = tip_x + 0.4 * np.cos(expected_angle)
    end_y = tip_y + 0.4 * np.sin(expected_angle)
    ax.plot([tip_x, end_x], [tip_y, end_y], "b--", lw=2, label="Expected (~70 deg)")

    # Initial crack
    ax.plot(
        [0, tip_x],
        [tip_y, tip_y],
        "k-",
        lw=3,
        alpha=0.3,
        label="Initial crack",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Crack Path")
    ax.set_xlim([0, SENS_PARAMS["L"]])
    ax.set_ylim([0, SENS_PARAMS["L"]])
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sens_benchmark_results.png", dpi=150)
    print("\nResults saved to 'sens_benchmark_results.png'")
    plt.show()


def print_benchmark_info():
    """
    Print information about the SENS benchmark parameters.
    """
    print("=" * 70)
    print("SENS Benchmark Parameters")
    print("=" * 70)

    print("\nGeometry:")
    print(f"  Domain size (L):        {SENS_PARAMS['L']} mm")
    print(f"  Initial crack length:   {SENS_PARAMS['crack_length']} mm")
    print(f"  Crack y-position:       {SENS_PARAMS['crack_y']} mm")

    print("\nMaterial (steel-like):")
    print(f"  Young's modulus (E):    {SENS_PARAMS['E']:.1e} MPa")
    print(f"  Poisson's ratio (nu):   {SENS_PARAMS['nu']}")
    print(f"  Plane condition:        {SENS_PARAMS['plane']}")

    print("\nPhase-field:")
    print(f"  Gc:                     {SENS_PARAMS['Gc']} N/mm")
    print(f"  l0:                     {SENS_PARAMS['l0']} mm")

    print("\nMesh:")
    print(f"  h_fine:                 {SENS_PARAMS['h_fine']} mm")
    print(f"  h_coarse:               {SENS_PARAMS['h_coarse']} mm")
    print(f"  Refinement band:        {SENS_PARAMS['refinement_band']} mm")

    print("\nLoading (SHEAR - key difference from SENT):")
    print(f"  Maximum shear disp:     {SENS_PARAMS['u_max']} mm")
    print(f"  Number of steps:        {SENS_PARAMS['n_steps']}")

    print("\nSolver:")
    print(f"  Displacement tolerance: {SENS_PARAMS['tol_u']:.0e}")
    print(f"  Damage tolerance:       {SENS_PARAMS['tol_d']:.0e}")
    print(f"  Max iterations/step:    {SENS_PARAMS['max_iter']}")

    print("\nExpected Results:")
    print("  Crack angle:            ~70 deg (upward from horizontal)")
    print("  Crack direction:        Upward (toward loading direction)")
    print("  Comparison with SENT:   Different path (SENT is horizontal)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SENS Fracture Benchmark")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "class", "mesh", "info"],
        default="quick",
        help="Run mode: full (complete benchmark), quick (fast test), "
        "class (class interface demo), mesh (mesh demo), info (parameters)",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full_benchmark()
    elif args.mode == "quick":
        run_quick_benchmark()
    elif args.mode == "class":
        run_with_class_interface()
    elif args.mode == "mesh":
        demonstrate_mesh_generation()
    elif args.mode == "info":
        print_benchmark_info()
