"""
Tests for SENS Benchmark
========================

Unit tests for the Single Edge Notched Shear (SENS) benchmark components.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.sens_benchmark import (
    SENS_PARAMS,
    generate_sens_mesh,
    create_precrack_damage,
    apply_sens_boundary_conditions,
    create_sens_bc_function,
    extract_crack_path,
    compute_crack_length,
    compute_crack_angle,
    track_crack_tip,
    validate_sens_results,
    analyze_stress_state,
    validate_tension_compression_split,
    compute_shear_reaction_force,
    SENSResults,
)
from mesh.triangle_mesh import TriangleMesh
from mesh.mesh_generators import create_rectangle_mesh


class TestSENSMeshGeneration:
    """Tests for SENS mesh generation."""

    def test_generate_sens_mesh_default(self):
        """Test default mesh generation."""
        mesh = generate_sens_mesh()

        assert mesh.n_nodes > 100
        assert mesh.n_elements > 100
        assert mesh.n_edges > mesh.n_elements

    def test_generate_sens_mesh_custom_params(self):
        """Test mesh generation with custom parameters."""
        params = {
            'L': 2.0,
            'h_fine': 0.02,
            'h_coarse': 0.1,
            'refinement_band': 0.2,
        }
        mesh = generate_sens_mesh(params)

        x_min, y_min = mesh.nodes.min(axis=0)
        x_max, y_max = mesh.nodes.max(axis=0)

        assert x_min == pytest.approx(0.0, abs=1e-10)
        assert y_min == pytest.approx(0.0, abs=1e-10)
        assert x_max == pytest.approx(2.0, abs=0.1)
        assert y_max == pytest.approx(2.0, abs=0.1)

    def test_generate_sens_mesh_uniform(self):
        """Test uniform mesh generation (no grading)."""
        params = {'h_fine': 0.05}
        mesh = generate_sens_mesh(params, use_graded_mesh=False)

        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_mesh_quality(self):
        """Test that generated mesh has reasonable quality."""
        mesh = generate_sens_mesh()

        # No zero-area elements
        assert all(mesh.element_areas > 0)

        # Aspect ratio not too extreme
        min_edge = np.min(mesh.edge_lengths)
        max_edge = np.max(mesh.edge_lengths)
        aspect_ratio = max_edge / min_edge

        assert aspect_ratio < 20, f"Aspect ratio too extreme: {aspect_ratio}"

    def test_mesh_wider_refinement_above_crack(self):
        """Test that SENS mesh has wider refinement band above crack line."""
        params = {
            'L': 1.0,
            'h_fine': 0.01,
            'h_coarse': 0.05,
            'refinement_band': 0.15,
            'crack_y': 0.5,
        }
        mesh = generate_sens_mesh(params)

        # Count fine elements above and below crack
        centroids = np.zeros((mesh.n_elements, 2))
        for e in range(mesh.n_elements):
            centroids[e] = np.mean(mesh.nodes[mesh.elements[e]], axis=0)

        crack_y = params['crack_y']
        fine_above = np.sum(
            (centroids[:, 1] > crack_y) &
            (centroids[:, 1] < crack_y + params['refinement_band'] * 1.5)
        )
        fine_below = np.sum(
            (centroids[:, 1] < crack_y) &
            (centroids[:, 1] > crack_y - params['refinement_band'])
        )

        # Should have more fine elements above due to wider band
        # (for the curved crack path)
        assert fine_above >= fine_below * 0.8  # Allow some tolerance


class TestPrecrackDamage:
    """Tests for pre-crack damage initialization."""

    def test_precrack_exponential(self):
        """Test exponential pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='exponential')

        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)
        assert np.max(d) > 0.9

    def test_precrack_sharp(self):
        """Test sharp pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='sharp')

        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_precrack_linear(self):
        """Test linear pre-crack damage profile."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='linear')

        assert d.shape == (mesh.n_edges,)
        assert np.all(d >= 0)
        assert np.all(d <= 1)

    def test_precrack_damage_decays_from_crack(self):
        """Test that damage decays away from crack line."""
        mesh = create_rectangle_mesh(1.0, 1.0, 30, 30)
        d = create_precrack_damage(mesh, 0.5, 0.5, 0.05, method='exponential')

        midpoints = mesh.compute_edge_midpoints()
        for i in range(mesh.n_edges):
            x, y = midpoints[i]
            if x > 0.7:  # Far ahead of crack tip
                assert d[i] < 0.5, f"Damage too high ahead of crack: d[{i}]={d[i]}"


class TestSENSBoundaryConditions:
    """Tests for SENS boundary condition setup."""

    def test_apply_sens_boundary_conditions(self):
        """Test SENS BC application."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        L = 1.0
        u_applied = 0.001

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, u_applied, L)

        n_bottom = len(mesh.get_nodes_in_region(lambda x, y: y < 1e-10))
        n_top = len(mesh.get_nodes_in_region(lambda x, y: y > L - 1e-10))

        # Bottom: 2*n_bottom (x and y), Top: 2*n_top (x and y)
        expected_bc_count = 2 * n_bottom + 2 * n_top
        assert len(bc_dofs) == expected_bc_count

    def test_sens_bc_bottom_fully_fixed(self):
        """Test that bottom is fully fixed (u_x = u_y = 0)."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        L = 1.0
        u_applied = 0.005

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, u_applied, L)

        bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < 1e-10)
        for node in bottom_nodes:
            dof_x = 2 * node
            dof_y = 2 * node + 1
            # Both x and y DOFs should be constrained at bottom
            assert dof_x in bc_dofs
            assert dof_y in bc_dofs
            # Values should be zero
            idx_x = np.where(bc_dofs == dof_x)[0]
            idx_y = np.where(bc_dofs == dof_y)[0]
            assert bc_values[idx_x[0]] == 0.0
            assert bc_values[idx_y[0]] == 0.0

    def test_sens_bc_top_shear(self):
        """Test that top has shear displacement and vertical constraint."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        L = 1.0
        u_applied = 0.005

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, u_applied, L)

        top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - 1e-10)
        for node in top_nodes:
            dof_x = 2 * node
            dof_y = 2 * node + 1
            # Both x and y DOFs should be constrained at top
            assert dof_x in bc_dofs
            assert dof_y in bc_dofs
            # x should be u_applied (shear), y should be zero
            idx_x = np.where(bc_dofs == dof_x)[0]
            idx_y = np.where(bc_dofs == dof_y)[0]
            assert bc_values[idx_x[0]] == u_applied
            assert bc_values[idx_y[0]] == 0.0

    def test_create_sens_bc_function(self):
        """Test BC function creation for load stepping."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)

        bc_dofs, bc_func = create_sens_bc_function(mesh, L=1.0)

        # Test at zero displacement
        vals_0 = bc_func(0.0)
        assert len(vals_0) == len(bc_dofs)
        assert all(v == 0.0 for v in vals_0)

        # Test at non-zero shear displacement
        vals_1 = bc_func(0.005)
        n_top = len(mesh.get_nodes_in_region(lambda x, y: y > 0.999))

        # Top x-DOF nodes should have non-zero displacement
        nonzero_count = np.sum(vals_1 != 0.0)
        assert nonzero_count == n_top

    def test_sens_bc_differs_from_sent(self):
        """Test that SENS BCs differ from SENT BCs."""
        from benchmarks.sent_benchmark import apply_sent_boundary_conditions

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        u = 0.005

        sent_dofs, sent_vals = apply_sent_boundary_conditions(mesh, u, 1.0)
        sens_dofs, sens_vals = apply_sens_boundary_conditions(mesh, u, 1.0)

        # SENS should have more constrained DOFs (bottom fully fixed vs only y)
        assert len(sens_dofs) > len(sent_dofs)


class TestCrackPathExtraction:
    """Tests for crack path extraction and angle computation."""

    def test_extract_crack_path_no_damage(self):
        """Test crack path extraction with no damage."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        damage = np.zeros(mesh.n_edges)

        path = extract_crack_path(mesh, damage, threshold=0.9)
        assert path.shape[0] == 0

    def test_extract_crack_path_horizontal(self):
        """Test crack path extraction for horizontal crack."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        damage = create_precrack_damage(mesh, 0.8, 0.5, 0.03, method='exponential')

        path = extract_crack_path(mesh, damage, threshold=0.5)
        if len(path) > 0:
            y_values = path[:, 1]
            y_spread = np.max(y_values) - np.min(y_values)
            assert y_spread < 0.2

    def test_compute_crack_angle_horizontal(self):
        """Test angle for horizontal crack is ~0."""
        path = np.array([[0.0, 0.5], [0.2, 0.5], [0.4, 0.5], [0.6, 0.5]])
        initial_tip = (0.1, 0.5)
        angle = compute_crack_angle(path, initial_tip)
        assert abs(angle) < 5.0

    def test_compute_crack_angle_angled(self):
        """Test angle for angled crack."""
        # 45-degree crack
        n = 10
        x = np.linspace(0.5, 1.0, n)
        y = np.linspace(0.5, 1.0, n)
        path = np.column_stack([x, y])
        initial_tip = (0.5, 0.5)
        angle = compute_crack_angle(path, initial_tip)
        assert 40 < angle < 50, f"Expected ~45 deg, got {angle}"

    def test_compute_crack_angle_70_degrees(self):
        """Test angle for ~70 degree crack (expected SENS result)."""
        n = 20
        angle_rad = np.radians(70)
        dist = np.linspace(0, 0.4, n)
        x = 0.5 + dist * np.cos(angle_rad)
        y = 0.5 + dist * np.sin(angle_rad)
        path = np.column_stack([x, y])
        initial_tip = (0.5, 0.5)
        angle = compute_crack_angle(path, initial_tip)
        assert 65 < angle < 75, f"Expected ~70 deg, got {angle}"

    def test_compute_crack_angle_insufficient_data(self):
        """Test angle with insufficient data."""
        path = np.array([[0.5, 0.5]])
        initial_tip = (0.5, 0.5)
        angle = compute_crack_angle(path, initial_tip)
        assert angle == 0.0

    def test_track_crack_tip(self):
        """Test crack tip tracking."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        damage = create_precrack_damage(mesh, 0.5, 0.5, 0.03, method='exponential')

        tip = track_crack_tip(mesh, damage, threshold=0.5)
        # Tip should be near (0.5, 0.5)
        assert 0.3 < tip[0] < 0.7
        assert 0.3 < tip[1] < 0.7

    def test_compute_crack_length(self):
        """Test crack length computation."""
        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)

        d0 = np.zeros(mesh.n_edges)
        a0 = compute_crack_length(mesh, d0)
        assert a0 == 0.0

        d1 = create_precrack_damage(mesh, 0.5, 0.5, 0.02, method='exponential')
        a1 = compute_crack_length(mesh, d1)
        assert a1 > 0.0
        assert a1 < 2.0


class TestStressStateAnalysis:
    """Tests for stress state analysis under shear loading."""

    def test_analyze_stress_state_structure(self):
        """Test that stress state analysis returns correct structure."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        # Apply small shear displacement
        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, 0.001, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        analysis = analyze_stress_state(mesh, elements, u)

        assert 'n_tension_dominated' in analysis
        assert 'n_compression_dominated' in analysis
        assert 'n_mixed' in analysis
        assert 'n_total' in analysis
        assert analysis['n_total'] == mesh.n_elements

    def test_shear_produces_mixed_stress_state(self):
        """Test that shear loading produces mixed principal strains."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, 0.001, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        analysis = analyze_stress_state(mesh, elements, u)

        # Under shear, most elements should have mixed principal strains
        # (one positive, one negative)
        assert analysis['n_mixed'] > analysis['n_tension_dominated'], \
            "Shear loading should produce mostly mixed stress states"


class TestTensionCompressionSplitValidation:
    """Tests for tension-compression split validation."""

    def test_split_validation_structure(self):
        """Test that split validation returns correct structure."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, 0.001, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        validation = validate_tension_compression_split(
            mesh, elements, u, damage
        )

        assert 'checks' in validation
        assert 'details' in validation
        assert 'passed' in validation
        assert 'energy_conservation' in validation['checks']
        assert 'psi_plus_nonnegative' in validation['checks']
        assert 'strain_reconstruction' in validation['checks']

    def test_split_energy_conservation(self):
        """Test that psi = psi_plus + psi_minus under shear."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, 0.001, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        validation = validate_tension_compression_split(
            mesh, elements, u, damage
        )

        assert validation['checks']['energy_conservation'], \
            f"Energy conservation failed: Miehe error = {validation['details']['max_miehe_energy_error']}"

    def test_split_balanced_under_shear(self):
        """Test that split is balanced (both psi+ and psi- significant) under shear."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, 0.001, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        validation = validate_tension_compression_split(
            mesh, elements, u, damage
        )

        assert validation['checks']['balanced_split'], \
            f"Split not balanced: ratio = {validation['details']['split_ratio']}"


class TestValidation:
    """Tests for validation functions."""

    def test_validate_sens_results_structure(self):
        """Test validation function with mock results."""
        n_steps = 10
        results = SENSResults(
            displacement=np.linspace(0, 0.015, n_steps),
            shear_force=np.concatenate([np.linspace(0, 0.4, n_steps // 2),
                                        np.linspace(0.4, 0.1, n_steps - n_steps // 2)]),
            crack_length=np.linspace(0.5, 1.0, n_steps),
            crack_angle=np.linspace(0, 70, n_steps),
            crack_tip_x=np.linspace(0.5, 0.8, n_steps),
            crack_tip_y=np.linspace(0.5, 0.8, n_steps),
            strain_energy=np.linspace(0, 0.1, n_steps),
            surface_energy=np.linspace(0, 0.05, n_steps),
            final_damage=np.zeros(100),
            crack_path=np.column_stack([
                np.linspace(0.5, 0.8, 5),
                np.linspace(0.5, 0.8, 5),
            ]),
        )

        validation = validate_sens_results(results, verbose=False)

        assert 'crack_angle' in validation
        assert 'crack_upward' in validation
        assert 'peak_load' in validation
        assert 'energy_balance' in validation
        assert 'not_horizontal' in validation

    def test_validate_correct_angle(self):
        """Test that validation passes for correct ~70 deg angle."""
        n_steps = 20
        # Simulate a crack at ~70 degrees
        angle_rad = np.radians(70)
        n_path = 10
        path_dist = np.linspace(0, 0.3, n_path)
        crack_path = np.column_stack([
            0.5 + path_dist * np.cos(angle_rad),
            0.5 + path_dist * np.sin(angle_rad),
        ])

        results = SENSResults(
            displacement=np.linspace(0, 0.015, n_steps),
            shear_force=np.concatenate([np.linspace(0, 0.4, n_steps // 2),
                                        np.linspace(0.4, 0.1, n_steps - n_steps // 2)]),
            crack_length=np.linspace(0.5, 1.0, n_steps),
            crack_angle=np.linspace(0, 70, n_steps),
            crack_tip_x=np.linspace(0.5, 0.8, n_steps),
            crack_tip_y=np.linspace(0.5, 0.8, n_steps),
            strain_energy=np.linspace(0, 0.1, n_steps),
            surface_energy=np.linspace(0, 0.05, n_steps),
            final_damage=np.zeros(100),
            crack_path=crack_path,
        )

        validation = validate_sens_results(results, verbose=False)
        assert validation['crack_angle']['passed']
        assert validation['crack_upward']['passed']
        assert validation['not_horizontal']['passed']


class TestSENSResultsClass:
    """Tests for SENSResults dataclass."""

    def test_sens_results_properties(self):
        """Test computed properties of SENSResults."""
        results = SENSResults(
            displacement=np.array([0.0, 0.005, 0.01, 0.015]),
            shear_force=np.array([0.0, 0.3, 0.5, 0.4]),
            crack_length=np.array([0.5, 0.5, 0.6, 0.8]),
            crack_angle=np.array([0.0, 10.0, 50.0, 68.0]),
            crack_tip_x=np.array([0.5, 0.52, 0.55, 0.6]),
            crack_tip_y=np.array([0.5, 0.52, 0.6, 0.75]),
            strain_energy=np.array([0.0, 0.01, 0.02, 0.015]),
            surface_energy=np.array([0.0, 0.001, 0.003, 0.005]),
            final_damage=np.zeros(50),
            crack_path=np.zeros((0, 2)),
        )

        assert results.peak_force == 0.5
        assert results.displacement_at_peak == 0.01
        assert results.final_crack_angle == 68.0

        expected_total = results.strain_energy + results.surface_energy
        np.testing.assert_array_almost_equal(results.total_energy, expected_total)

    def test_sens_results_final_angle_no_propagation(self):
        """Test final angle when no crack propagation."""
        results = SENSResults(
            displacement=np.array([0.0, 0.001]),
            shear_force=np.array([0.0, 0.1]),
            crack_length=np.array([0.5, 0.5]),
            crack_angle=np.array([0.0, 0.0]),
            crack_tip_x=np.array([0.5, 0.5]),
            crack_tip_y=np.array([0.5, 0.5]),
            strain_energy=np.array([0.0, 0.001]),
            surface_energy=np.array([0.0, 0.0]),
            final_damage=np.zeros(10),
            crack_path=np.zeros((0, 2)),
        )

        assert results.final_crack_angle == 0.0


class TestIntegration:
    """Integration tests for SENS benchmark."""

    @pytest.mark.slow
    def test_sens_mesh_with_elements(self):
        """Test that SENS mesh works with GraFEA elements."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement

        params = {
            'L': 1.0,
            'h_fine': 0.02,
            'h_coarse': 0.05,
            'refinement_band': 0.1,
        }
        mesh = generate_sens_mesh(params, use_graded_mesh=False)

        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.015)

        elements = []
        for e in range(mesh.n_elements):
            try:
                elem = GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                elements.append(elem)
            except ValueError as err:
                pytest.fail(f"Failed to create element {e}: {err}")

        assert len(elements) == mesh.n_elements

    @pytest.mark.slow
    def test_sens_full_pipeline(self):
        """Test complete SENS setup without running full simulation."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from mesh.edge_graph import EdgeGraph
        from solvers.staggered_solver import StaggeredSolver, SolverConfig

        params = {
            'L': 1.0,
            'h_fine': 0.05,
            'h_coarse': 0.1,
            'l0': 0.1,
        }

        mesh = generate_sens_mesh(params, use_graded_mesh=False)

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

        # Initialize pre-crack
        d_init = create_precrack_damage(mesh, 0.5, 0.5, params['l0'])
        d_init = np.minimum(d_init, 0.9)
        solver.set_initial_damage(d_init)

        # Setup shear boundary conditions
        bc_dofs, bc_func = create_sens_bc_function(mesh, params['L'])

        assert solver.damage is not None
        assert len(bc_dofs) > 0

        # Run one step at zero displacement to verify setup
        results = solver.solve(np.array([0.0]), bc_dofs, bc_func)
        assert len(results) == 1
        assert results[-1].converged or not np.any(np.isnan(results[-1].displacement))

    @pytest.mark.slow
    def test_sens_shear_force_computation(self):
        """Test shear force computation under applied shear."""
        from physics.material import IsotropicMaterial
        from elements.grafea_element import GraFEAElement
        from assembly.boundary_conditions import apply_dirichlet_bc
        from assembly.global_assembly import assemble_global_stiffness
        from scipy.sparse.linalg import spsolve

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.05)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        u_applied = 0.001
        bc_dofs, bc_values = apply_sens_boundary_conditions(mesh, u_applied, 1.0)
        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        F_shear = compute_shear_reaction_force(mesh, elements, u, damage, 1.0)

        # Should produce non-zero shear force for non-zero displacement
        assert abs(F_shear) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_crack_length(self):
        """Test with zero initial crack length."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        d = create_precrack_damage(mesh, 0.0, 0.5, 0.05)

        assert np.max(d) < 1.0
        assert np.sum(d > 0.1) < mesh.n_edges / 2

    def test_small_l0(self):
        """Test with very small length scale."""
        mesh = create_rectangle_mesh(1.0, 1.0, 30, 30)
        l0 = 0.005

        d = create_precrack_damage(mesh, 0.5, 0.5, l0)
        assert d.shape == (mesh.n_edges,)
        assert np.max(d) > 0.5

    def test_crack_angle_no_propagation(self):
        """Test crack angle when crack hasn't propagated."""
        path = np.array([[0.1, 0.5], [0.3, 0.5], [0.5, 0.5]])
        initial_tip = (0.5, 0.5)
        angle = compute_crack_angle(path, initial_tip)
        assert angle == 0.0  # No points ahead of tip

    def test_track_crack_tip_no_damage(self):
        """Test crack tip tracking with no damage."""
        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        damage = np.zeros(mesh.n_edges)

        tip = track_crack_tip(mesh, damage, threshold=0.5)
        assert tip == (0.0, 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
