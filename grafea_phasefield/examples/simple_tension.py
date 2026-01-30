"""
Simple Tension Test Example
===========================

Demonstrates basic usage of the GraFEA phase-field framework
with a simple uniaxial tension test.
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh import create_rectangle_mesh, EdgeGraph
from elements import GraFEAElement
from physics import IsotropicMaterial
from assembly import create_bc_from_region, merge_bcs
from solvers import StaggeredSolver, SolverConfig


def run_tension_test():
    """Run a simple uniaxial tension test."""
    print("=" * 60)
    print("GraFEA Phase-Field: Simple Tension Test")
    print("=" * 60)

    # Create mesh
    Lx, Ly = 1.0, 0.5  # Domain dimensions
    nx, ny = 20, 10    # Mesh divisions
    mesh = create_rectangle_mesh(Lx, Ly, nx, ny)
    print(f"\nMesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, {mesh.n_edges} edges")

    # Material properties (steel-like)
    material = IsotropicMaterial(
        E=210e9,    # Young's modulus [Pa]
        nu=0.3,     # Poisson's ratio
        Gc=2700,    # Critical energy release rate [J/m²]
        l0=0.03     # Phase-field length scale [m]
    )
    print(f"Material: E={material.E/1e9:.0f} GPa, ν={material.nu}, Gc={material.Gc} J/m²")

    # Create GraFEA elements
    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                for e in range(mesh.n_elements)]

    # Create edge graph for damage regularization
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')

    # Boundary conditions
    # Bottom: fix y-displacement
    tol = 1e-6
    bc_bottom_dofs, bc_bottom_vals = create_bc_from_region(
        mesh, lambda x, y: y < tol, 'y', 0.0
    )

    # Fix one node in x to prevent rigid body motion
    bc_fix_x = (np.array([0]), np.array([0.0]))

    # Top nodes for prescribed displacement
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > Ly - tol)
    bc_top_dofs = np.array([2*n + 1 for n in top_nodes])

    # Merge fixed boundary conditions
    fixed_dofs, fixed_vals = merge_bcs(
        (bc_bottom_dofs, bc_bottom_vals),
        bc_fix_x
    )
    all_bc_dofs = np.concatenate([fixed_dofs, bc_top_dofs])

    def bc_values_func(load_factor):
        """Return BC values for given load factor."""
        vals = np.zeros(len(all_bc_dofs))
        vals[:len(fixed_vals)] = fixed_vals
        vals[len(fixed_vals):] = load_factor  # Top displacement
        return vals

    # Setup solver
    config = SolverConfig(
        tol_u=1e-6,
        tol_d=1e-6,
        max_stagger_iter=50,
        verbose=True
    )
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # Load stepping
    max_displacement = 0.0003  # Maximum applied displacement
    n_steps = 20
    load_factors = np.linspace(0, max_displacement, n_steps)

    print(f"\nRunning {n_steps} load steps up to u_max = {max_displacement*1000:.3f} mm")
    print("-" * 60)

    # Run simulation
    results = solver.solve(load_factors, all_bc_dofs, bc_values_func)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    summary = solver.get_results_summary()
    print(f"Total steps: {summary['n_steps']}")
    print(f"Converged steps: {summary['converged_steps']}")
    print(f"Maximum damage: {summary['max_damage']:.4f}")
    print(f"Final strain energy: {summary['final_strain_energy']:.6e} J")
    print(f"Final surface energy: {summary['final_surface_energy']:.6e} J")

    # Optional: Plot results if matplotlib available
    try:
        import matplotlib.pyplot as plt
        from postprocess import plot_damage_field, plot_energy_evolution

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot final damage field
        plot_damage_field(mesh, results[-1].damage, ax=axes[0])
        axes[0].set_title(f'Damage Field (step {results[-1].step})')

        # Plot energy evolution
        plot_energy_evolution(results, ax=axes[1])
        axes[1].set_title('Energy Evolution')

        plt.tight_layout()
        plt.savefig('tension_test_results.png', dpi=150)
        print("\nResults saved to 'tension_test_results.png'")
        plt.show()

    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")

    return results


if __name__ == "__main__":
    results = run_tension_test()
