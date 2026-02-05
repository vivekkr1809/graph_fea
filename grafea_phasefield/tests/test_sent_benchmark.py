"""
Tests for SENT Benchmark
========================

Unit tests for the Single Edge Notched Tension (SENT) benchmark components.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.sent_benchmark import (
    SENT_PARAMS,
    generate_sent_mesh,
    create_precrack_damage,
    apply_sent_boundary_conditions,
    create_sent_bc_function,
    extract_crack_path,
    compute_crack_length,
    validate_sent_results,
    SENTResults,
)
from mesh.triangle_mesh import TriangleMesh
from mesh.mesh_generators import create_rectangle_mesh


class TestSENTMeshGeneration:
    """Tests for SENT mesh generation."""

    def test_generate_sent_mesh_default(self):
        """Test default mesh generation."""
        mesh = generate_sent_mesh()

        # Check mesh has reasonable size
        assert mesh.n_nodes > 100  # Should have many nodes
        assert mesh.n_elements > 100  # Should have many elements
        assert mesh.n_edges > mesh.n_elements  # Interior edges shared

    def test_generate_sent_mesh_custom_params(self):
        """Test mesh generation with custom parameters."""
        params = {
            'L': 2.0,
            'h_fine': 0.02,
            'h_coarse': 0.1,
            'refinement_band': 0.2,
        }
        mesh = generate_sent_mesh(params)

        # Check domain size
        x_min, y_min = mesh.nodes.min(axis=0)
        x_max, y_max = mesh.nodes.max(axis=0)

        assert x_min == pytest.approx(0.0, abs=1e-10)
        assert y_min == pytest.approx(0.0, abs=1e-10)
        assert x_max == pytest.approx(2.0, abs=0.1)
        assert y_max == pytest.approx(2.0, abs=0.1)

    def test_generate_sent_mesh_uniform(self):
        """Test uniform mesh generation (no grading)."""
        params = {'h_fine': 0.05}
        mesh = generate_sent_mesh(params, use_graded_mesh=False)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_mesh_quality(self):
        """Test that generated mesh has reasonable quality."""
        mesh = generate_sent_mesh()

        # Check no zero-area elements
        assert all(mesh.element_areas > 0)

        # Check aspect ratio is not too extreme
        min_edge = np.min(mesh.edge_lengths)
        max_edge = np.max(mesh.edge_lengths)
        aspect_ratio = max_edge / min_edge

        # With graded mesh, some aspect ratio variation is expected
        assert aspect_ratio < 20, f"Aspect ratio too extreme: {aspect_ratio}"


class TestPrecrackDamage:
    """Tests for pre-crack damage initialization."""

    def test_precrack_exponential(self):
        """Test exponential pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        crack_tip_x = 0.5
        crack_y = 0.5
        l0 = 0.05

        d = create_precrack_damage(mesh, crack_tip_x, crack_y, l0, method='exponential')

        # Check damage field shape
        assert d.shape == (mesh.n_edges,)

        # Check damage is bounded [0, 1]
        assert np.all(d >= 0)
        assert np.all(d <= 1)

        # Check maximum damage is near crack
        assert np.max(d) > 0.9

        # Check damage decays away from crack
        midpoints = mesh.compute_edge_midpoints()
        for i in range(mesh.n_edges):
            x, y = midpoints[i]
            if x > crack_tip_x + 0.2:  # Far ahead of crack tip
                assert d[i] < 0.5, f"Damage too high ahead of crack: d[{i}]={d[i]}"

    def test_precrack_sharp(self):
        """Test sharp pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='sharp')

        # Sharp method should have d=0 or d=1 mostly
        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_precrack_linear(self):
        """Test linear pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='linear')

        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_precrack_symmetry(self):
        """Test that pre-crack is approximately symmetric about crack line."""
        # Use alternating pattern for better symmetry
        mesh = create_rectangle_mesh(1.0, 1.0, 40, 40, pattern='alternating')
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.03, method='exponential')

        midpoints = mesh.compute_edge_midpoints()

        # Check approximate symmetry about y=0.5
        # Due to mesh discretization, perfect symmetry is not expected
        symmetric_pairs = []
        for i in range(mesh.n_edges):
            x, y = midpoints[i]
            if x < 0.3:  # Well inside the crack region
                # Find symmetric point
                y_mirror = 1.0 - y
                for j in range(mesh.n_edges):
                    xj, yj = midpoints[j]
                    if abs(xj - x) < 0.02 and abs(yj - y_mirror) < 0.02:
                        symmetric_pairs.append((d[i], d[j]))
                        break

        # Check that most pairs are reasonably symmetric
        if symmetric_pairs:
            errors = [abs(a - b) for a, b in symmetric_pairs]
            mean_error = np.mean(errors)
            # Allow for some asymmetry due to mesh discretization
            assert mean_error < 0.3, f"Mean symmetry error too large: {mean_error}"


class TestBoundaryConditions:
    """Tests for SENT boundary condition setup."""

    def test_apply_sent_boundary_conditions(self):
        """Test BC application function."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        L = 1.0
        u_applied = 0.001

        bc_dofs, bc_values = apply_sent_boundary_conditions(mesh, u_applied, L)

        # Should have BCs on bottom y, corner x, and top y
        n_bottom = len(mesh.get_nodes_in_region(lambda x, y: y < 1e-10))
        n_top = len(mesh.get_nodes_in_region(lambda x, y: y > L - 1e-10))

        # Expected: n_bottom (y) + 1 (corner x) + n_top (y)
        expected_bc_count = n_bottom + 1 + n_top
        assert len(bc_dofs) == expected_bc_count

        # Check values
        assert np.sum(bc_values == 0.0) == n_bottom + 1  # Bottom y and corner x
        assert np.sum(bc_values == u_applied) == n_top   # Top y

    def test_create_sent_bc_function(self):
        """Test BC function creation."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)

        bc_dofs, bc_func = create_sent_bc_function(mesh, L=1.0)

        # Test at zero displacement
        vals_0 = bc_func(0.0)
        assert len(vals_0) == len(bc_dofs)
        assert all(v == 0.0 for v in vals_0)

        # Test at non-zero displacement
        vals_1 = bc_func(0.001)
        n_top = len(mesh.get_nodes_in_region(lambda x, y: y > 0.999))

        # Top nodes should have non-zero displacement
        nonzero_count = np.sum(vals_1 != 0.0)
        assert nonzero_count == n_top


class TestCrackPathExtraction:
    """Tests for crack path extraction."""

    def test_extract_crack_path_no_damage(self):
        """Test crack path extraction with no damage."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        damage = np.zeros(mesh.n_edges)

        path = extract_crack_path(mesh, damage, threshold=0.9)

        assert path.shape[0] == 0  # No crack

    def test_extract_crack_path_horizontal(self):
        """Test crack path extraction for horizontal crack."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)

        # Create damage along horizontal line
        damage = create_precrack_damage(mesh, 0.8, 0.5, 0.03, method='exponential')

        path = extract_crack_path(mesh, damage, threshold=0.5)

        if len(path) > 0:
            # Path should be roughly horizontal
            y_values = path[:, 1]
            y_spread = np.max(y_values) - np.min(y_values)
            assert y_spread < 0.2, f"Crack path not horizontal: y spread = {y_spread}"

    def test_compute_crack_length(self):
        """Test crack length computation."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)

        # No damage -> zero length
        d0 = np.zeros(mesh.n_edges)
        a0 = compute_crack_length(mesh, d0)
        assert a0 == 0.0

        # With damage
        d1 = create_precrack_damage(mesh, 0.5, 0.5, 0.02, method='exponential')
        a1 = compute_crack_length(mesh, d1)

        # Should have some crack length
        assert a1 > 0.0
        assert a1 < 1.0  # Should not exceed domain size


class TestValidation:
    """Tests for validation functions."""

    def test_validate_sent_results_structure(self):
        """Test validation function with mock results."""
        # Create mock results
        n_steps = 10
        results = SENTResults(
            displacement=np.linspace(0, 0.01, n_steps),
            force=np.concatenate([np.linspace(0, 0.5, n_steps//2),
                                  np.linspace(0.5, 0.2, n_steps - n_steps//2)]),
            crack_length=np.linspace(0.5, 1.0, n_steps),
            strain_energy=np.linspace(0, 0.1, n_steps),
            surface_energy=np.linspace(0, 0.05, n_steps),
            final_damage=np.zeros(100),
            crack_path=np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]),
        )

        validation = validate_sent_results(results, verbose=False)

        # Check validation structure
        assert 'crack_direction' in validation
        assert 'peak_load' in validation
        assert 'energy_balance' in validation
        assert 'crack_propagation' in validation

        # Check crack direction
        assert validation['crack_direction']['passed']  # Horizontal path
        assert validation['crack_direction']['angle'] == pytest.approx(0.0, abs=1.0)


class TestSENTResultsClass:
    """Tests for SENTResults dataclass."""

    def test_sent_results_properties(self):
        """Test computed properties of SENTResults."""
        results = SENTResults(
            displacement=np.array([0.0, 0.001, 0.002, 0.003]),
            force=np.array([0.0, 0.3, 0.5, 0.4]),
            crack_length=np.array([0.5, 0.5, 0.6, 0.8]),
            strain_energy=np.array([0.0, 0.01, 0.02, 0.015]),
            surface_energy=np.array([0.0, 0.001, 0.003, 0.005]),
            final_damage=np.zeros(50),
            crack_path=np.zeros((0, 2)),
        )

        # Test properties
        assert results.peak_force == 0.5
        assert results.displacement_at_peak == 0.002

        # Test total energy
        expected_total = results.strain_energy + results.surface_energy
        np.testing.assert_array_almost_equal(results.total_energy, expected_total)


class TestIntegration:
    """Integration tests for SENT benchmark."""

    @pytest.mark.slow
    def test_sent_mesh_with_elements(self):
        """Test that SENT mesh works with GraFEA elements."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement

        params = {
            'L': 1.0,
            'h_fine': 0.02,
            'h_coarse': 0.05,
            'refinement_band': 0.1,
        }
        mesh = generate_sent_mesh(params, use_graded_mesh=False)

        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.015)

        # Create elements - this should not raise any errors
        elements = []
        for e in range(mesh.n_elements):
            try:
                elem = GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                elements.append(elem)
            except ValueError as err:
                pytest.fail(f"Failed to create element {e}: {err}")

        assert len(elements) == mesh.n_elements

    @pytest.mark.slow
    def test_sent_full_pipeline(self):
        """Test complete SENT setup without running full simulation."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from mesh.edge_graph import EdgeGraph
        from solvers.staggered_solver import StaggeredSolver, SolverConfig

        # Use coarse mesh for fast testing
        # Use larger l0 relative to h to avoid singular matrices
        params = {
            'L': 1.0,
            'h_fine': 0.05,
            'h_coarse': 0.1,
            'l0': 0.1,  # Larger l0 for numerical stability
        }

        # Generate mesh
        mesh = generate_sent_mesh(params, use_graded_mesh=False)

        # Create material
        material = IsotropicMaterial(
            E=params.get('E', 210e3),
            nu=params.get('nu', 0.3),
            Gc=params.get('Gc', 2.7),
            l0=params['l0']
        )

        # Create elements
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        # Create edge graph
        edge_graph = EdgeGraph(mesh)

        # Create solver
        config = SolverConfig(verbose=False, max_stagger_iter=50)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # Initialize pre-crack with smaller initial damage to avoid singular matrices
        # Cap maximum initial damage at 0.9 to keep matrix invertible
        d_init = create_precrack_damage(mesh, 0.5, 0.5, params['l0'])
        d_init = np.minimum(d_init, 0.9)  # Cap to avoid singularity
        solver.set_initial_damage(d_init)

        # Setup boundary conditions
        bc_dofs, bc_func = create_sent_bc_function(mesh, params['L'])

        # Just verify setup is complete - don't run full simulation
        assert solver.damage is not None
        assert len(bc_dofs) > 0

        # Run one step to verify it works (at zero displacement)
        results = solver.solve(np.array([0.0]), bc_dofs, bc_func)
        assert len(results) == 1
        # At zero displacement with pre-crack, should converge
        assert results[-1].converged or not np.any(np.isnan(results[-1].displacement))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_crack_length(self):
        """Test with zero initial crack length."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        d = create_precrack_damage(mesh, 0.0, 0.5, 0.05)

        # Should have damage near left edge only
        assert np.max(d) < 1.0
        assert np.sum(d > 0.1) < mesh.n_edges / 2

    def test_crack_at_boundary(self):
        """Test with crack at domain center."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.02)

        # Crack should be centered
        path = extract_crack_path(mesh, d, threshold=0.5)
        if len(path) > 0:
            mean_y = np.mean(path[:, 1])
            assert 0.4 < mean_y < 0.6

    def test_small_l0(self):
        """Test with very small length scale."""
        mesh = create_rectangle_mesh(1.0, 1.0, 30, 30)
        l0 = 0.005  # Very small

        d = create_precrack_damage(mesh, 0.5, 0.5, l0)

        # Should still work but produce sharper damage profile
        assert d.shape == (mesh.n_edges,)
        assert np.max(d) > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
