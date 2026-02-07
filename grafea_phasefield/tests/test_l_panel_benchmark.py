"""
Tests for L-Shaped Panel Benchmark
===================================

Unit tests for the L-shaped panel benchmark components.
This is the critical nucleation test - no pre-existing crack.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.l_shaped_panel_benchmark import (
    L_PANEL_PARAMS,
    generate_l_panel_mesh,
    identify_l_panel_boundaries,
    apply_l_panel_boundary_conditions,
    create_l_panel_bc_function,
    extract_crack_path,
    check_nucleation,
    find_nucleation_location,
    validate_l_panel_results,
    compare_with_original_grafea,
    LPanelResults,
)
from mesh.triangle_mesh import TriangleMesh


class TestLPanelMeshGeneration:
    """Tests for L-shaped panel mesh generation."""

    def test_generate_l_panel_mesh_default(self):
        """Test default L-panel mesh generation."""
        mesh = generate_l_panel_mesh()

        # Should have nodes and elements
        assert mesh.n_nodes > 50
        assert mesh.n_elements > 50
        assert mesh.n_edges > mesh.n_elements

    def test_generate_l_panel_mesh_custom(self):
        """Test custom L-panel mesh generation."""
        params = {
            'outer_size': 200.0,
            'inner_size': 120.0,
            'h_fine': 5.0,
            'h_coarse': 20.0,
        }
        mesh = generate_l_panel_mesh(params)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_generate_l_panel_uniform(self):
        """Test uniform L-panel mesh generation."""
        params = {'h_fine': 10.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_l_shape_domain(self):
        """Test that mesh conforms to L-shape domain."""
        params = {
            'outer_size': 250.0,
            'inner_size': 150.0,
            'h_fine': 10.0,
            'h_coarse': 25.0,
        }
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        outer = params['outer_size']
        inner = params['inner_size']
        corner_x = outer - inner  # 100
        corner_y = outer - inner  # 100

        # All nodes should be within the L-shape domain
        for x, y in mesh.nodes:
            # A point is in the L-shape if:
            # (y <= corner_y) OR (x <= corner_x)
            in_bottom = y <= corner_y + 1.0  # tolerance
            in_left = x <= corner_x + 1.0
            assert in_bottom or in_left, \
                f"Node ({x}, {y}) outside L-shape domain"

    def test_mesh_quality(self):
        """Test mesh quality."""
        params = {'h_fine': 10.0, 'h_coarse': 25.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        # No zero-area elements
        assert all(mesh.element_areas > 0)

        # No zero-length edges
        assert all(mesh.edge_lengths > 0)

    def test_mesh_covers_both_legs(self):
        """Test mesh covers both horizontal and vertical legs."""
        params = {
            'outer_size': 250.0,
            'inner_size': 150.0,
            'h_fine': 10.0,
            'h_coarse': 25.0,
        }
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        outer = params['outer_size']
        corner_x = outer - 150.0  # 100
        corner_y = corner_x

        # Should have nodes in the horizontal leg (x > corner_x, y < corner_y)
        horiz_nodes = mesh.get_nodes_in_region(
            lambda x, y: x > corner_x + 10 and y < corner_y - 10
        )
        assert len(horiz_nodes) > 0, "No nodes in horizontal leg"

        # Should have nodes in the vertical leg (x < corner_x, y > corner_y)
        vert_nodes = mesh.get_nodes_in_region(
            lambda x, y: x < corner_x - 10 and y > corner_y + 10
        )
        assert len(vert_nodes) > 0, "No nodes in vertical leg"


class TestBoundaryIdentification:
    """Tests for boundary node identification."""

    def test_identify_boundaries(self):
        """Test boundary identification."""
        params = {
            'outer_size': 250.0,
            'inner_size': 150.0,
            'h_fine': 10.0,
            'h_coarse': 25.0,
        }
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        boundaries = identify_l_panel_boundaries(mesh, params)

        # Should find bottom nodes
        assert len(boundaries['bottom']) > 0

        # Should find left edge nodes
        assert len(boundaries['left']) > 0

        # Should find top of vertical leg
        assert len(boundaries['top_leg']) > 0

        # Should find inner corner
        assert len(boundaries['inner_corner']) == 1

    def test_bottom_nodes_at_y_zero(self):
        """Test bottom nodes are at y=0."""
        params = {'h_fine': 10.0, 'h_coarse': 25.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        boundaries = identify_l_panel_boundaries(mesh, {**L_PANEL_PARAMS, **params})

        for node in boundaries['bottom']:
            assert mesh.nodes[node, 1] < 1e-6

    def test_inner_corner_location(self):
        """Test inner corner node is at correct position."""
        params = {
            'outer_size': 250.0,
            'inner_size': 150.0,
            'h_fine': 10.0,
            'h_coarse': 25.0,
        }
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        boundaries = identify_l_panel_boundaries(mesh, params)

        corner_x = params['outer_size'] - params['inner_size']
        corner_y = corner_x

        corner_node = boundaries['inner_corner'][0]
        x, y = mesh.nodes[corner_node]

        # Should be reasonably close to theoretical corner
        dist = np.sqrt((x - corner_x) ** 2 + (y - corner_y) ** 2)
        assert dist < params['h_coarse'], \
            f"Corner node at ({x}, {y}), expected near ({corner_x}, {corner_y})"


class TestLPanelBoundaryConditions:
    """Tests for L-panel boundary conditions."""

    def test_apply_l_panel_bcs(self):
        """Test BC application."""
        params = {'h_fine': 10.0, 'h_coarse': 25.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        bc_dofs, bc_values = apply_l_panel_boundary_conditions(
            mesh, 0.5, {**L_PANEL_PARAMS, **params}
        )

        assert len(bc_dofs) > 0
        assert len(bc_dofs) == len(bc_values)

        # Some BCs should be zero (bottom fixed)
        assert np.any(bc_values == 0.0)

        # Some should be non-zero (top displacement)
        assert np.any(bc_values != 0.0)

    def test_create_l_panel_bc_function(self):
        """Test BC function creation."""
        params = {'h_fine': 10.0, 'h_coarse': 25.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        bc_dofs, bc_func = create_l_panel_bc_function(
            mesh, {**L_PANEL_PARAMS, **params}
        )

        # At zero displacement
        vals_0 = bc_func(0.0)
        assert len(vals_0) == len(bc_dofs)

        # At non-zero displacement
        vals_1 = bc_func(0.5)
        assert np.any(vals_1 != 0.0)

    def test_bottom_fixed(self):
        """Test that bottom edge is fully fixed."""
        params = {'h_fine': 10.0, 'h_coarse': 25.0}
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        bc_dofs, bc_values = apply_l_panel_boundary_conditions(
            mesh, 0.5, {**L_PANEL_PARAMS, **params}
        )

        # All bottom nodes should have both DOFs constrained to zero
        bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < 1e-6)
        for node in bottom_nodes:
            dof_x = 2 * node
            dof_y = 2 * node + 1
            if dof_x in bc_dofs:
                idx = np.where(bc_dofs == dof_x)[0][0]
                assert bc_values[idx] == 0.0
            if dof_y in bc_dofs:
                idx = np.where(bc_dofs == dof_y)[0][0]
                assert bc_values[idx] == 0.0


class TestNucleation:
    """Tests for nucleation detection."""

    def test_check_nucleation_no_damage(self):
        """Test nucleation check with zero damage."""
        damage = np.zeros(100)
        assert not check_nucleation(damage)

    def test_check_nucleation_with_damage(self):
        """Test nucleation check with significant damage."""
        damage = np.zeros(100)
        damage[50] = 0.5  # High damage at one edge
        assert check_nucleation(damage, threshold=0.1)

    def test_check_nucleation_below_threshold(self):
        """Test nucleation check below threshold."""
        damage = np.zeros(100)
        damage[50] = 0.05  # Below threshold
        assert not check_nucleation(damage, threshold=0.1)

    def test_find_nucleation_location_no_damage(self):
        """Test nucleation location with no damage."""
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(10.0, 10.0, 5, 5)
        damage = np.zeros(mesh.n_edges)

        loc = find_nucleation_location(mesh, damage)
        assert loc is None

    def test_find_nucleation_location_with_damage(self):
        """Test nucleation location with damage."""
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(10.0, 10.0, 5, 5)

        damage = np.zeros(mesh.n_edges)
        damage[10] = 0.5

        loc = find_nucleation_location(mesh, damage, threshold=0.1)
        assert loc is not None
        assert len(loc) == 2  # (x, y)


class TestCrackPathExtraction:
    """Tests for crack path extraction."""

    def test_extract_crack_path_no_damage(self):
        """Test with no damage."""
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(10.0, 10.0, 5, 5)
        damage = np.zeros(mesh.n_edges)

        path = extract_crack_path(mesh, damage)
        assert path.shape[0] == 0

    def test_extract_crack_path_with_damage(self):
        """Test with damaged edges."""
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(10.0, 10.0, 5, 5)

        damage = np.zeros(mesh.n_edges)
        damage[10:15] = 0.9

        path = extract_crack_path(mesh, damage, threshold=0.5)
        assert path.shape[0] > 0
        assert path.shape[1] == 2


class TestLPanelValidation:
    """Tests for validation functions."""

    def test_validate_structure(self):
        """Test validation returns correct structure."""
        results = LPanelResults(
            displacement=np.linspace(0, 1, 10),
            reaction_force=np.linspace(0, 100, 10),
            max_damage=np.linspace(0, 0.8, 10),
            strain_energy=np.linspace(0, 10, 10),
            surface_energy=np.linspace(0, 5, 10),
            final_damage=np.zeros(100),
            crack_path=np.zeros((0, 2)),
            nucleation_detected=False,
            nucleation_step=None,
            nucleation_location=None,
        )

        validation = validate_l_panel_results(results, verbose=False)

        assert 'nucleation_occurred' in validation
        assert 'nucleation_at_corner' in validation
        assert 'crack_propagated' in validation
        assert 'correct_direction' in validation
        assert 'reasonable_angle' in validation
        assert 'has_peak' in validation
        assert 'started_undamaged' in validation

    def test_validate_nucleation_detected(self):
        """Test validation with nucleation."""
        results = LPanelResults(
            displacement=np.linspace(0, 1, 10),
            reaction_force=np.concatenate([
                np.linspace(0, 50, 5),
                np.linspace(50, 20, 5)
            ]),
            max_damage=np.linspace(0, 0.8, 10),
            strain_energy=np.linspace(0, 10, 10),
            surface_energy=np.linspace(0, 5, 10),
            final_damage=np.zeros(100),
            crack_path=np.array([[100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]),
            nucleation_detected=True,
            nucleation_step=5,
            nucleation_location=(101.0, 101.0),
        )

        validation = validate_l_panel_results(results, verbose=False)

        # Nucleation should pass
        assert validation['nucleation_occurred']['passed']

    def test_compare_with_original_grafea(self):
        """Test GraFEA comparison function."""
        results = LPanelResults(
            displacement=np.linspace(0, 1, 10),
            reaction_force=np.linspace(0, 100, 10),
            max_damage=np.linspace(0, 0.5, 10),
            strain_energy=np.linspace(0, 10, 10),
            surface_energy=np.linspace(0, 5, 10),
            final_damage=np.zeros(100),
            crack_path=np.zeros((0, 2)),
            nucleation_detected=True,
            nucleation_step=5,
            nucleation_location=(100.0, 100.0),
        )

        comparison = compare_with_original_grafea(results)

        assert comparison['original_grafea']['can_nucleate'] is False
        assert comparison['edge_based_phasefield']['can_nucleate'] is True
        assert comparison['advantage_demonstrated'] is True


class TestLPanelResultsClass:
    """Tests for LPanelResults dataclass."""

    def test_properties(self):
        """Test computed properties."""
        results = LPanelResults(
            displacement=np.array([0.0, 0.2, 0.4, 0.6]),
            reaction_force=np.array([0.0, 40.0, 80.0, 60.0]),
            max_damage=np.array([0.0, 0.01, 0.2, 0.5]),
            strain_energy=np.array([0.0, 2.0, 6.0, 5.0]),
            surface_energy=np.array([0.0, 0.0, 0.5, 1.0]),
            final_damage=np.zeros(50),
            crack_path=np.zeros((0, 2)),
            nucleation_detected=True,
            nucleation_step=2,
            nucleation_location=(100.0, 100.0),
        )

        assert results.peak_force == 80.0
        assert results.displacement_at_peak == 0.4

        expected_total = results.strain_energy + results.surface_energy
        np.testing.assert_array_almost_equal(results.total_energy, expected_total)


class TestIntegration:
    """Integration tests for L-panel benchmark."""

    @pytest.mark.slow
    def test_l_panel_mesh_with_elements(self):
        """Test L-panel mesh works with GraFEA elements."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement

        params = {
            'h_fine': 10.0,
            'h_coarse': 25.0,
            'l0': 20.0,
        }
        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        material = IsotropicMaterial(
            E=25.85e3, nu=0.18, Gc=0.095, l0=20.0
        )

        elements = []
        for e in range(mesh.n_elements):
            try:
                elem = GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                elements.append(elem)
            except ValueError as err:
                pytest.fail(f"Failed to create element {e}: {err}")

        assert len(elements) == mesh.n_elements

    @pytest.mark.slow
    def test_l_panel_full_pipeline(self):
        """Test complete L-panel setup without running full simulation."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from mesh.edge_graph import EdgeGraph
        from solvers.staggered_solver import StaggeredSolver, SolverConfig

        params = {
            'outer_size': 250.0,
            'inner_size': 150.0,
            'h_fine': 20.0,
            'h_coarse': 40.0,
            'l0': 30.0,
        }

        mesh = generate_l_panel_mesh(params, use_graded_mesh=False)

        material = IsotropicMaterial(
            E=25.85e3, nu=0.18, Gc=0.095, l0=params['l0']
        )

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        edge_graph = EdgeGraph(mesh)

        config = SolverConfig(verbose=False, max_stagger_iter=50)
        solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

        # NO initial damage (nucleation test)
        assert np.max(solver.damage) == 0.0

        bc_dofs, bc_func = create_l_panel_bc_function(
            mesh, {**L_PANEL_PARAMS, **params}
        )

        assert len(bc_dofs) > 0

        # Run one step at zero displacement
        results = solver.solve(np.array([0.0]), bc_dofs, bc_func)
        assert len(results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
