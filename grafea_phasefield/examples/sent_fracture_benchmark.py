"""
SENT Fracture Benchmark Example
===============================

Single Edge Notched Tension (SENT) test - canonical Mode-I fracture benchmark.

This example demonstrates:
1. Setting up the SENT geometry and mesh
2. Initializing the pre-existing crack
3. Running the phase-field fracture simulation
4. Validating results against literature
5. Visualizing the load-displacement curve and crack path

Reference:
- Miehe et al. (2010) "Thermodynamically consistent phase-field models of fracture"
- Ambati et al. (2015) "A review on phase-field models of brittle fracture"
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmarks.sent_benchmark import (
    SENT_PARAMS,
    run_sent_benchmark,
    validate_sent_results,
    compare_with_literature,
    quick_sent_test,
    SENTBenchmark,
    generate_sent_mesh,
    create_precrack_damage,
)


def run_full_benchmark():
    """
    Run the full SENT benchmark with default parameters.

    This is the complete simulation as specified in the benchmark,
    but can be slow due to the fine mesh required near the crack.
    """
    print("=" * 70)
    print("SENT Fracture Benchmark - Full Simulation")
    print("=" * 70)

    # Run with default parameters
    results = run_sent_benchmark(verbose=True)

    # Validate results
    print("\n")
    validation = validate_sent_results(results, verbose=True)

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
    Run a quick version of the SENT benchmark for testing.

    Uses fewer load steps and a coarser mesh for faster execution.
    """
    print("=" * 70)
    print("SENT Fracture Benchmark - Quick Test")
    print("=" * 70)

    results = quick_sent_test(n_steps=30, verbose=True)

    # Try to plot if matplotlib is available
    try:
        plot_results(results)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")
    # Basic validation
    print("\n--- Quick Validation ---")
    print(f"Peak force: {results.peak_force:.4f}")
    print(f"Displacement at peak: {results.displacement_at_peak:.6f}")
    print(f"Final crack length: {results.crack_length[-1]:.4f}")

    return results


def run_with_class_interface():
    """
    Demonstrate using the SENTBenchmark class interface.

    The class provides a clean interface for running benchmarks
    and creating plots.
    """
    print("=" * 70)
    print("SENT Benchmark - Using Class Interface")
    print("=" * 70)

    # Create benchmark with custom parameters
    params = {
        "n_steps": 50,
        "h_fine": 0.0075,  # Slightly coarser for speed
        "h_coarse": 0.03,
    }

    benchmark = SENTBenchmark(params)

    # Run simulation
    results = benchmark.run(verbose=True)

    # Validate
    validation = benchmark.validate(verbose=True)

    # Plot results (if matplotlib available)
    try:
        benchmark.plot_results(save_path="sent_results.png")
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")

    return benchmark


def demonstrate_mesh_generation():
    """
    Demonstrate SENT mesh generation with visualization.
    """
    print("=" * 70)
    print("SENT Mesh Generation Demo")
    print("=" * 70)

    # Generate meshes with different parameters
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
        "refinement_band": 0.1,
    }

    print("\nCoarse mesh:")
    mesh_coarse = generate_sent_mesh(params_coarse)
    print(f"  Nodes: {mesh_coarse.n_nodes}")
    print(f"  Elements: {mesh_coarse.n_elements}")
    print(f"  Edges: {mesh_coarse.n_edges}")

    print("\nFine mesh:")
    mesh_fine = generate_sent_mesh(params_fine)
    print(f"  Nodes: {mesh_fine.n_nodes}")
    print(f"  Elements: {mesh_fine.n_elements}")
    print(f"  Edges: {mesh_fine.n_edges}")

    # Demonstrate pre-crack initialization
    print("\nPre-crack damage initialization:")
    d_coarse = create_precrack_damage(mesh_coarse, 0.5, 0.5, 0.015)
    print(f"  Max damage: {np.max(d_coarse):.4f}")
    print(f"  Damaged edges (d > 0.5): {np.sum(d_coarse > 0.5)}")
    print(f"  Damaged edges (d > 0.9): {np.sum(d_coarse > 0.9)}")

    # Plot if available
    try:
        import matplotlib.pyplot as plt
        from postprocess.visualization import plot_mesh, plot_damage_field

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot coarse mesh
        plot_mesh(mesh_coarse, ax=axes[0], show_edges=True)
        axes[0].set_title(f"Coarse Mesh ({mesh_coarse.n_elements} elements)")

        # Plot with pre-crack damage
        plot_damage_field(mesh_coarse, d_coarse, ax=axes[1])
        axes[1].set_title("Pre-crack Damage Field")

        plt.tight_layout()
        plt.savefig("sent_mesh_demo.png", dpi=150)
        print("\nMesh visualization saved to 'sent_mesh_demo.png'")
        plt.show()

    except ImportError:
        print("\nNote: matplotlib not available, skipping mesh visualization")


def plot_results(results):
    """
    Create comprehensive result plots.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Load-displacement curve
    ax = axes[0, 0]
    ax.plot(results.displacement * 1000, results.force, "b-", lw=2)
    ax.axhline(
        y=results.peak_force,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Peak = {results.peak_force:.4f}",
    )
    ax.axvline(
        x=results.displacement_at_peak * 1000, color="r", linestyle="--", alpha=0.7
    )
    ax.set_xlabel("Displacement (mm × 10³)")
    ax.set_ylabel("Reaction Force")
    ax.set_title("Load-Displacement Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Crack length evolution
    ax = axes[0, 1]
    ax.plot(results.displacement * 1000, results.crack_length, "g-", lw=2)
    ax.axhline(
        y=SENT_PARAMS["L"],
        color="k",
        linestyle=":",
        alpha=0.7,
        label=f'Domain size = {SENT_PARAMS["L"]}',
    )
    ax.set_xlabel("Displacement (mm × 10³)")
    ax.set_ylabel("Crack Length")
    ax.set_title("Crack Propagation")
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
    ax.set_xlabel("Displacement (mm × 10³)")
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
    ax.axhline(
        y=SENT_PARAMS["crack_y"],
        color="b",
        linestyle="--",
        alpha=0.5,
        label="Expected (y=0.5)",
    )
    ax.plot(
        [0, SENT_PARAMS["crack_length"]],
        [SENT_PARAMS["crack_y"], SENT_PARAMS["crack_y"]],
        "b-",
        lw=3,
        alpha=0.3,
        label="Initial crack",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Crack Path")
    ax.set_xlim([0, SENT_PARAMS["L"]])
    ax.set_ylim([0, SENT_PARAMS["L"]])
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sent_benchmark_results.png", dpi=150)
    print("\nResults saved to 'sent_benchmark_results.png'")
    plt.show()


def print_benchmark_info():
    """
    Print information about the SENT benchmark parameters.
    """
    print("=" * 70)
    print("SENT Benchmark Parameters")
    print("=" * 70)

    print("\nGeometry:")
    print(f"  Domain size (L):        {SENT_PARAMS['L']} mm")
    print(f"  Initial crack length:   {SENT_PARAMS['crack_length']} mm")
    print(f"  Crack y-position:       {SENT_PARAMS['crack_y']} mm")

    print("\nMaterial (steel-like):")
    print(f"  Young's modulus (E):    {SENT_PARAMS['E']:.1e} MPa")
    print(f"  Poisson's ratio (ν):    {SENT_PARAMS['nu']}")
    print(f"  Plane condition:        {SENT_PARAMS['plane']}")

    print("\nPhase-field:")
    print(f"  Gc:                     {SENT_PARAMS['Gc']} N/mm")
    print(f"  l0:                     {SENT_PARAMS['l0']} mm")

    print("\nMesh:")
    print(f"  h_fine:                 {SENT_PARAMS['h_fine']} mm")
    print(f"  h_coarse:               {SENT_PARAMS['h_coarse']} mm")
    print(f"  Refinement band:        {SENT_PARAMS['refinement_band']} mm")

    print("\nLoading:")
    print(f"  Maximum displacement:   {SENT_PARAMS['u_max']} mm")
    print(f"  Number of steps:        {SENT_PARAMS['n_steps']}")

    print("\nSolver:")
    print(f"  Displacement tolerance: {SENT_PARAMS['tol_u']:.0e}")
    print(f"  Damage tolerance:       {SENT_PARAMS['tol_d']:.0e}")
    print(f"  Max iterations/step:    {SENT_PARAMS['max_iter']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SENT Fracture Benchmark")
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
