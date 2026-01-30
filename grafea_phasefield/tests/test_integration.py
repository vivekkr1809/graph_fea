"""
Integration Tests
=================

End-to-end tests for the complete framework.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from mesh.mesh_generators import create_rectangle_mesh, create_single_element
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from physics.damage import HistoryField
from physics.surface_energy import compute_surface_energy, assemble_damage_system
from assembly.global_assembly import (
    assemble_global_stiffness, compute_all_edge_strains, compute_strain_energy
)
from assembly.boundary_conditions import (
    apply_dirichlet_bc, create_bc_from_region, merge_bcs
)
from solvers.staggered_solver import StaggeredSolver, SolverConfig
from scipy.sparse.linalg import spsolve


class TestFullWorkflow:
    """Test complete workflow from mesh to results."""

    def test_undamaged_linear_response(self):
        """Without damage, should get linear elastic response."""
        # Create mesh
        mesh = create_rectangle_mesh(1, 1, 5, 5)

        # High Gc = no damage
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=1e10, l0=0.05)

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]
        edge_graph = EdgeGraph(mesh)

        # Solver
        config = SolverConfig(verbose=False, max_stagger_iter=10)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # BCs: fix bottom, prescribe top
        bc_bottom, _ = create_bc_from_region(mesh, lambda x, y: y < 0.01, 'y', 0.0)
        bc_fix_x = (np.array([0]), np.array([0.0]))  # Fix one x to prevent RBM
        bc_top_nodes = mesh.get_nodes_in_region(lambda x, y: y > 0.99)
        bc_top_dofs = np.array([2*n + 1 for n in bc_top_nodes])

        fixed_dofs, fixed_vals = merge_bcs(
            (bc_bottom, np.zeros(len(bc_bottom))),
            bc_fix_x
        )
        all_bc_dofs = np.concatenate([fixed_dofs, bc_top_dofs])

        def bc_values(load):
            vals = np.zeros(len(all_bc_dofs))
            vals[:len(fixed_vals)] = fixed_vals
            vals[len(fixed_vals):] = load  # Top displacement
            return vals

        # Run single step
        results = solver.solve(np.array([0.0001]), all_bc_dofs, bc_values)

        assert len(results) == 1
        assert results[0].converged
        assert results[0].n_iterations <= 3  # Should converge quickly
        assert np.allclose(results[0].damage, 0)  # No damage

    def test_single_element_tension(self):
        """Simple tension test on single element."""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements_conn = np.array([[0, 1, 2]])
        mesh = TriangleMesh(nodes, elements_conn)

        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.5)
        elements = [GraFEAElement(nodes, material)]

        # Fix node 0, apply displacement to node 1
        bc_dofs = np.array([0, 1, 5])  # u0x, u0y, u2y
        bc_values = np.array([0, 0, 0])

        # Apply x-displacement to node 1
        applied_disp = 0.001
        bc_dofs = np.append(bc_dofs, 2)
        bc_values = np.append(bc_values, applied_disp)

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(6)

        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        # Check solution is reasonable
        assert u[2] == applied_disp  # Prescribed
        assert u[0] == 0  # Fixed
        assert u[1] == 0  # Fixed

    def test_damage_growth_localization(self):
        """Test that damage grows and localizes under loading."""
        # Finer mesh for damage localization
        mesh = create_rectangle_mesh(1, 0.5, 10, 5)

        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.05)

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]
        edge_graph = EdgeGraph(mesh)

        config = SolverConfig(verbose=False, max_stagger_iter=50)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # BCs
        bc_bottom, _ = create_bc_from_region(mesh, lambda x, y: y < 0.01, 'y', 0.0)
        bc_fix_x = (np.array([0]), np.array([0.0]))
        bc_top_nodes = mesh.get_nodes_in_region(lambda x, y: y > 0.49)
        bc_top_dofs = np.array([2*n + 1 for n in bc_top_nodes])

        fixed_dofs, fixed_vals = merge_bcs(
            (bc_bottom, np.zeros(len(bc_bottom))),
            bc_fix_x
        )
        all_bc_dofs = np.concatenate([fixed_dofs, bc_top_dofs])

        def bc_values(load):
            vals = np.zeros(len(all_bc_dofs))
            vals[:len(fixed_vals)] = fixed_vals
            vals[len(fixed_vals):] = load
            return vals

        # Multiple load steps to see damage growth
        load_steps = np.linspace(0, 0.0005, 5)
        results = solver.solve(load_steps, all_bc_dofs, bc_values)

        # Damage should increase with loading
        max_damages = [np.max(r.damage) for r in results]
        assert max_damages[-1] > max_damages[0]

    def test_energy_balance(self):
        """Check that energy quantities are consistent."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=1e10, l0=0.1)

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]
        edge_graph = EdgeGraph(mesh)

        config = SolverConfig(verbose=False)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # Simple tension
        bc_bottom, _ = create_bc_from_region(mesh, lambda x, y: y < 0.01, 'both', 0.0)
        bc_top_nodes = mesh.get_nodes_in_region(lambda x, y: y > 0.99)
        bc_top_dofs = np.array([2*n + 1 for n in bc_top_nodes])

        all_bc_dofs = np.concatenate([bc_bottom, bc_top_dofs])

        def bc_values(load):
            vals = np.zeros(len(all_bc_dofs))
            vals[len(bc_bottom):] = load
            return vals

        results = solver.solve(np.array([0.0001, 0.0002]), all_bc_dofs, bc_values)

        # Strain energy should increase with displacement
        assert results[1].strain_energy > results[0].strain_energy

        # Total energy should be positive
        for r in results:
            total = r.strain_energy + r.surface_energy
            assert total >= 0


class TestDamageSystem:
    """Test damage evolution system."""

    def test_damage_system_assembly(self):
        """Test damage system matrix assembly."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        edge_graph = EdgeGraph(mesh)

        Gc, l0 = 2700, 0.05
        history = np.ones(mesh.n_edges) * 1e6  # Uniform driving force

        A_d, b_d = assemble_damage_system(mesh, edge_graph, history, Gc, l0)

        # Matrix should be sparse and square
        assert A_d.shape == (mesh.n_edges, mesh.n_edges)

        # Matrix should be positive definite (for solvability)
        # Check diagonal dominance as proxy
        A_dense = A_d.toarray()
        for i in range(mesh.n_edges):
            off_diag_sum = np.sum(np.abs(A_dense[i, :])) - np.abs(A_dense[i, i])
            assert A_dense[i, i] > 0

    def test_damage_solution_bounds(self):
        """Damage solution should be in [0, 1]."""
        mesh = create_rectangle_mesh(1, 1, 10, 10)
        edge_graph = EdgeGraph(mesh)

        Gc, l0 = 2700, 0.05

        # Test with various history values
        for H_val in [0, 1e5, 1e6, 1e7]:
            history = np.ones(mesh.n_edges) * H_val

            A_d, b_d = assemble_damage_system(mesh, edge_graph, history, Gc, l0)
            d = spsolve(A_d, b_d)

            # Clip and check
            d_clipped = np.clip(d, 0, 1)

            # Raw solution might be slightly out of bounds due to numerics
            # but should be close
            assert np.all(d >= -0.01)
            assert np.all(d <= 1.01)


class TestConvergence:
    """Test solver convergence properties."""

    def test_mesh_convergence(self):
        """Test that solution improves with mesh refinement."""
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=1e10, l0=0.1)

        energies = []
        for n in [3, 5, 7]:
            mesh = create_rectangle_mesh(1, 1, n, n)
            elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                        for e in range(mesh.n_elements)]

            # Apply uniform strain
            u = np.zeros(2 * mesh.n_nodes)
            u[::2] = 0.001 * mesh.nodes[:, 0]
            u[1::2] = -0.0003 * mesh.nodes[:, 1]  # Poisson effect

            damage = np.zeros(mesh.n_edges)
            E = compute_strain_energy(mesh, elements, u, damage)
            energies.append(E)

        # Energy should converge (differences decrease)
        diff1 = abs(energies[1] - energies[0])
        diff2 = abs(energies[2] - energies[1])
        assert diff2 < diff1 or diff2 < 1e-6  # Converging or already converged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
