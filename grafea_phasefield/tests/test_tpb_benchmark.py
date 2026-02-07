"""
Tests for Three-Point Bending (TPB) Benchmark
==============================================

Unit tests for the TPB benchmark components.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.tpb_benchmark import (
    TPB_PARAMS,
    generate_tpb_mesh,
    create_notch_damage,
    apply_tpb_boundary_conditions,
    create_tpb_bc_function,
    extract_crack_path,
    track_vertical_crack,
    compute_crack_length,
    validate_tpb_results,
    TPBResults,
)
from mesh.triangle_mesh import TriangleMesh
from mesh.mesh_generators import create_rectangle_mesh


class TestTPBMeshGeneration:
    """Tests for TPB mesh generation."""

    def test_generate_tpb_mesh_default(self):
        """Test default mesh generation."""
        mesh = generate_tpb_mesh()

        # Check mesh has reasonable size
        assert mesh.n_nodes > 50
        assert mesh.n_elements > 50
        assert mesh.n_edges > mesh.n_elements

    def test_generate_tpb_mesh_custom_params(self):
        """Test mesh generation with custom parameters."""
        params = {
            'L': 80.0,
            'W': 30.0,
            'h_fine': 1.0,
            'h_coarse': 5.0,
            'notch_refine_radius': 8.0,
        }
        mesh = generate_tpb_mesh(params)

        # Check domain size
        x_min, y_min = mesh.nodes.min(axis=0)
        x_max, y_max = mesh.nodes.max(axis=0)

        assert x_min == pytest.approx(0.0, abs=1e-10)
        assert y_min == pytest.approx(0.0, abs=1e-10)
        assert x_max == pytest.approx(80.0, abs=1.0)
        assert y_max == pytest.approx(30.0, abs=1.0)

    def test_generate_tpb_mesh_uniform(self):
        """Test uniform (non-graded) mesh generation."""
        params = {'h_fine': 2.0}
        mesh = generate_tpb_mesh(params, use_graded_mesh=False)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_mesh_quality(self):
        """Test that generated mesh has reasonable quality."""
        params = {'h_fine': 1.0, 'h_coarse': 5.0}
        mesh = generate_tpb_mesh(params)

        # No zero-area elements
        assert all(mesh.element_areas > 0)

        # Check no degenerate edges
        assert all(mesh.edge_lengths > 0)

    def test_mesh_domain_is_rectangle(self):
        """Test mesh covers the rectangular domain."""
        params = {'L': 100.0, 'W': 40.0, 'h_fine': 2.0, 'h_coarse': 8.0}
        mesh = generate_tpb_mesh(params)

        x_min, y_min = mesh.nodes.min(axis=0)
        x_max, y_max = mesh.nodes.max(axis=0)

        assert x_min == pytest.approx(0.0, abs=0.1)
        assert y_min == pytest.approx(0.0, abs=0.1)
        assert x_max == pytest.approx(100.0, abs=1.0)
        assert y_max == pytest.approx(40.0, abs=1.0)


class TestNotchDamage:
    """Tests for notch damage initialization."""

    def test_notch_damage_exponential(self):
        """Test exponential notch damage profile."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        d = create_notch_damage(mesh, 50.0, 20.0, 1.0, method='exponential')

        # Check shape and bounds
        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)

        # Max damage should be near the notch line
        assert np.max(d) > 0.9

    def test_notch_damage_sharp(self):
        """Test sharp notch damage profile."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        d = create_notch_damage(mesh, 50.0, 20.0, 1.0, method='sharp')

        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_notch_damage_linear(self):
        """Test linear notch damage profile."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        d = create_notch_damage(mesh, 50.0, 20.0, 1.0, method='linear')

        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_notch_is_vertical(self):
        """Test that notch damage is centered vertically."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        d = create_notch_damage(mesh, 50.0, 20.0, 1.0, method='exponential')

        midpoints = mesh.compute_edge_midpoints()

        # Heavily damaged edges should be near x=50
        heavily_damaged = np.where(d > 0.5)[0]
        if len(heavily_damaged) > 0:
            damaged_x = midpoints[heavily_damaged, 0]
            assert np.mean(np.abs(damaged_x - 50.0)) < 5.0

    def test_notch_below_depth(self):
        """Test that damage is concentrated below notch depth."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        notch_depth = 20.0
        l0 = 1.0
        d = create_notch_damage(mesh, 50.0, notch_depth, l0, method='exponential')

        midpoints = mesh.compute_edge_midpoints()

        # Edges well above notch tip should have low damage
        far_above = np.where(midpoints[:, 1] > notch_depth + 5 * l0)[0]
        if len(far_above) > 0:
            assert np.max(d[far_above]) < 0.1


class TestTPBBoundaryConditions:
    """Tests for TPB boundary condition setup."""

    def test_apply_tpb_boundary_conditions(self):
        """Test BC application."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        params = TPB_PARAMS.copy()
        u_applied = 0.1

        bc_dofs, bc_values = apply_tpb_boundary_conditions(mesh, u_applied, params)

        # Should have BCs for supports and load point
        assert len(bc_dofs) > 0
        assert len(bc_dofs) == len(bc_values)

        # Load point should have negative displacement (downward)
        assert np.any(bc_values < 0)

    def test_create_tpb_bc_function(self):
        """Test BC function creation."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        params = TPB_PARAMS.copy()

        bc_dofs, bc_func = create_tpb_bc_function(mesh, params)

        # At zero displacement, all BCs should be zero
        vals_0 = bc_func(0.0)
        assert len(vals_0) == len(bc_dofs)
        assert np.all(vals_0 == 0.0)

        # At non-zero displacement, load point should have non-zero value
        vals_1 = bc_func(0.1)
        assert np.any(vals_1 != 0.0)

        # Load point should have negative value (downward)
        assert np.any(vals_1 < 0)

    def test_support_nodes_on_bottom(self):
        """Test that support nodes are on the bottom edge."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        params = TPB_PARAMS.copy()

        bc_dofs, bc_values = apply_tpb_boundary_conditions(mesh, 0.1, params)

        # All fixed y-DOFs at supports should correspond to bottom nodes
        for i, dof in enumerate(bc_dofs):
            if bc_values[i] == 0.0:
                node = dof // 2
                component = dof % 2
                if component == 1:  # y-DOF
                    assert mesh.nodes[node, 1] < 1.0  # Near bottom


class TestCrackPathExtraction:
    """Tests for crack path extraction."""

    def test_extract_crack_path_no_damage(self):
        """Test crack path with no damage."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        damage = np.zeros(mesh.n_edges)

        path = extract_crack_path(mesh, damage)
        assert path.shape[0] == 0

    def test_extract_crack_path_vertical(self):
        """Test crack path for vertical damage pattern."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)

        # Create vertical damage at x=50
        d = create_notch_damage(mesh, 50.0, 30.0, 1.0, method='exponential')

        path = extract_crack_path(mesh, d, threshold=0.5)

        if len(path) > 0:
            # Path should be roughly vertical (small x spread)
            x_spread = np.max(path[:, 0]) - np.min(path[:, 0])
            y_spread = np.max(path[:, 1]) - np.min(path[:, 1])
            if y_spread > 1.0:
                assert x_spread < y_spread

    def test_track_vertical_crack(self):
        """Test vertical crack tip tracking."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)
        d = create_notch_damage(mesh, 50.0, 20.0, 1.0)

        tip = track_vertical_crack(mesh, d, 50.0, 1.0, threshold=0.5)

        # Should find crack tip near notch
        if tip is not None:
            assert abs(tip[0] - 50.0) < 5.0  # Near center x
            assert tip[1] <= 25.0  # At or near notch depth

    def test_compute_crack_length(self):
        """Test crack length computation."""
        mesh = create_rectangle_mesh(100.0, 40.0, 50, 20)

        # No damage
        d0 = np.zeros(mesh.n_edges)
        assert compute_crack_length(mesh, d0) == 0.0

        # With notch damage
        d1 = create_notch_damage(mesh, 50.0, 20.0, 1.0)
        a = compute_crack_length(mesh, d1)
        assert a > 0.0


class TestTPBValidation:
    """Tests for validation functions."""

    def test_validate_tpb_results_structure(self):
        """Test validation function with mock results."""
        n_steps = 10
        results = TPBResults(
            displacement=np.linspace(0, 0.5, n_steps),
            reaction_force=np.concatenate([
                np.linspace(0, 100, n_steps // 2),
                np.linspace(100, 30, n_steps - n_steps // 2)
            ]),
            crack_length=np.linspace(20, 35, n_steps),
            crack_tip_y=np.linspace(20, 35, n_steps),
            strain_energy=np.linspace(0, 10, n_steps),
            surface_energy=np.linspace(0, 5, n_steps),
            final_damage=np.zeros(100),
            crack_path=np.array([[50.0, 20.0], [50.0, 30.0], [50.0, 40.0]]),
        )

        validation = validate_tpb_results(results, verbose=False)

        # Check structure
        assert 'vertical_crack' in validation
        assert 'starts_at_notch' in validation
        assert 'has_peak' in validation
        assert 'significant_propagation' in validation
        assert 'energy_balance' in validation

    def test_validate_vertical_crack(self):
        """Test vertical crack validation."""
        n_steps = 10
        # Create results with perfectly vertical crack path
        results = TPBResults(
            displacement=np.linspace(0, 0.5, n_steps),
            reaction_force=np.concatenate([
                np.linspace(0, 100, n_steps // 2),
                np.linspace(100, 30, n_steps - n_steps // 2)
            ]),
            crack_length=np.linspace(20, 35, n_steps),
            crack_tip_y=np.linspace(20, 35, n_steps),
            strain_energy=np.linspace(0, 10, n_steps),
            surface_energy=np.linspace(0, 5, n_steps),
            final_damage=np.zeros(100),
            crack_path=np.array([[50.0, 21.0], [50.0, 25.0],
                                 [50.0, 30.0], [50.0, 35.0]]),
        )

        validation = validate_tpb_results(results, verbose=False)

        # Vertical crack should pass
        assert validation['vertical_crack']['passed']
        assert validation['vertical_crack']['angle_from_vertical'] < 5


class TestTPBResultsClass:
    """Tests for TPBResults dataclass."""

    def test_tpb_results_properties(self):
        """Test computed properties of TPBResults."""
        results = TPBResults(
            displacement=np.array([0.0, 0.1, 0.2, 0.3]),
            reaction_force=np.array([0.0, 50.0, 100.0, 80.0]),
            crack_length=np.array([20.0, 20.0, 22.0, 28.0]),
            crack_tip_y=np.array([20.0, 20.0, 22.0, 28.0]),
            strain_energy=np.array([0.0, 1.0, 3.0, 2.5]),
            surface_energy=np.array([0.0, 0.1, 0.5, 1.0]),
            final_damage=np.zeros(50),
            crack_path=np.zeros((0, 2)),
        )

        assert results.peak_force == 100.0
        assert results.displacement_at_peak == 0.2

        expected_total = results.strain_energy + results.surface_energy
        np.testing.assert_array_almost_equal(results.total_energy, expected_total)


class TestIntegration:
    """Integration tests for TPB benchmark."""

    @pytest.mark.slow
    def test_tpb_mesh_with_elements(self):
        """Test TPB mesh works with GraFEA elements."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement

        params = {
            'h_fine': 2.0,
            'h_coarse': 8.0,
            'l0': 4.0,
        }
        mesh = generate_tpb_mesh(params)

        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=4.0)

        elements = []
        for e in range(mesh.n_elements):
            try:
                elem = GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                elements.append(elem)
            except ValueError as err:
                pytest.fail(f"Failed to create element {e}: {err}")

        assert len(elements) == mesh.n_elements

    @pytest.mark.slow
    def test_tpb_full_pipeline(self):
        """Test complete TPB setup without running full simulation."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from mesh.edge_graph import EdgeGraph
        from solvers.staggered_solver import StaggeredSolver, SolverConfig

        params = {
            'L': 100.0,
            'W': 40.0,
            'h_fine': 4.0,
            'h_coarse': 10.0,
            'l0': 8.0,
            'notch_depth': 20.0,
        }

        mesh = generate_tpb_mesh(params, use_graded_mesh=False)

        material = IsotropicMaterial(
            E=params.get('E', 210e3),
            nu=params.get('nu', 0.3),
            Gc=params.get('Gc', 2.7),
            l0=params['l0']
        )

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        edge_graph = EdgeGraph(mesh)

        config = SolverConfig(verbose=False, max_stagger_iter=50)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # Initialize notch
        d_init = create_notch_damage(mesh, 50.0, 20.0, params['l0'])
        d_init = np.minimum(d_init, 0.9)
        solver.set_initial_damage(d_init)

        bc_dofs, bc_func = create_tpb_bc_function(mesh, {**TPB_PARAMS, **params})

        assert solver.damage is not None
        assert len(bc_dofs) > 0

        # Run one step at zero displacement
        results = solver.solve(np.array([0.0]), bc_dofs, bc_func)
        assert len(results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
