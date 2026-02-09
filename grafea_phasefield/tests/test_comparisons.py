"""
Tests for Comparison Studies Module
====================================

Tests for:
- FEM phase-field reference solver
- Original GraFEA solver
- Comparison metrics
- Integration: GraFEA-PF vs FEM-PF
- Integration: GraFEA-PF vs Original GraFEA
"""

import numpy as np
import pytest
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'src'))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from mesh.mesh_generators import create_rectangle_mesh, create_square_mesh
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from solvers.staggered_solver import SolverConfig, LoadStep


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_material():
    """Standard steel-like material for testing."""
    return IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.04)


@pytest.fixture
def coarse_mesh():
    """Coarse mesh for fast testing."""
    return create_rectangle_mesh(1.0, 1.0, 10, 10)


@pytest.fixture
def medium_mesh():
    """Medium mesh for more detailed tests."""
    return create_rectangle_mesh(1.0, 1.0, 20, 20)


@pytest.fixture
def simple_config():
    """Solver config for fast testing."""
    return SolverConfig(
        tol_u=1e-4,
        tol_d=1e-4,
        max_stagger_iter=20,
        verbose=False,
    )


# ============================================================
# Test: Comparison Metrics
# ============================================================

class TestComparisonMetrics:
    """Tests for comparison_metrics module."""

    def test_path_deviation_identical(self):
        """Identical paths should have zero deviation."""
        from comparisons.comparison_metrics import compute_path_deviation

        path = np.array([[0.0, 0.5], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [1.0, 0.5]])
        result = compute_path_deviation(path, path)

        assert result['hausdorff'] < 1e-10
        assert result['mean_deviation'] < 1e-10

    def test_path_deviation_offset(self):
        """Parallel offset paths should give deviation equal to offset."""
        from comparisons.comparison_metrics import compute_path_deviation

        path_a = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])
        offset = 0.1
        path_b = np.array([[0.0, 0.5 + offset], [0.5, 0.5 + offset],
                           [1.0, 0.5 + offset]])

        result = compute_path_deviation(path_a, path_b)

        assert abs(result['hausdorff'] - offset) < 1e-6
        assert abs(result['mean_deviation'] - offset) < 1e-6

    def test_path_deviation_empty(self):
        """Empty paths should return inf/nan."""
        from comparisons.comparison_metrics import compute_path_deviation

        path_a = np.array([[0.0, 0.5], [1.0, 0.5]])
        empty = np.array([]).reshape(0, 2)

        result = compute_path_deviation(path_a, empty)
        assert np.isinf(result['hausdorff'])

    def test_load_displacement_error_identical(self):
        """Identical curves should have zero error."""
        from comparisons.comparison_metrics import compute_load_displacement_error

        u = np.linspace(0, 0.01, 50)
        F = np.sin(u * 100) * 100

        result_a = {'displacement': u, 'force': F}
        result_b = {'displacement': u, 'force': F}

        metrics = compute_load_displacement_error(result_a, result_b)
        assert metrics['peak_load_error'] < 1e-10
        assert metrics['l2_error'] < 1e-10

    def test_load_displacement_error_scaled(self):
        """Scaled curve should give known relative error."""
        from comparisons.comparison_metrics import compute_load_displacement_error

        u = np.linspace(0, 0.01, 50)
        F_a = np.concatenate([u[:25] * 1e4, 1e4 * 0.01 / 4 * (1 - (u[25:] - 0.005) / 0.005)])
        scale = 1.1
        F_b = F_a * scale

        result_a = {'displacement': u, 'force': F_a}
        result_b = {'displacement': u, 'force': F_b}

        metrics = compute_load_displacement_error(result_a, result_b)

        # Peak load error should be ~10% (scale - 1)
        expected_err = abs(scale - 1) / scale
        assert abs(metrics['peak_load_error'] - expected_err) < 0.01

    def test_path_smoothness_straight(self):
        """Straight path should have near-zero curvature."""
        from comparisons.comparison_metrics import compute_path_smoothness

        path = np.column_stack([np.linspace(0, 1, 20), np.full(20, 0.5)])
        result = compute_path_smoothness(path)

        assert result['mean_curvature'] < 1e-10
        assert result['direction_changes'] == 0
        assert abs(result['straightness_ratio'] - 1.0) < 1e-6

    def test_path_smoothness_zigzag(self):
        """Zigzag path should have high curvature and many direction changes."""
        from comparisons.comparison_metrics import compute_path_smoothness

        x = np.linspace(0, 1, 20)
        y = 0.5 + 0.1 * np.sin(x * 8 * np.pi)
        path = np.column_stack([x, y])
        result = compute_path_smoothness(path)

        assert result['mean_curvature'] > 0
        assert result['direction_changes'] > 0
        assert result['straightness_ratio'] < 1.0

    def test_crack_angle_horizontal(self):
        """Horizontal crack should have ~0 degree angle."""
        from comparisons.comparison_metrics import compute_crack_angle

        path = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.5)])
        result = compute_crack_angle(path)

        assert abs(result['mean_angle_deg']) < 2.0

    def test_crack_angle_diagonal(self):
        """45-degree crack should give ~45 degree angle."""
        from comparisons.comparison_metrics import compute_crack_angle

        t = np.linspace(0, 1, 10)
        path = np.column_stack([0.5 + t * 0.3, 0.5 + t * 0.3])
        result = compute_crack_angle(path)

        assert abs(result['mean_angle_deg'] - 45.0) < 5.0

    def test_compare_efficiency(self):
        """Test efficiency comparison function."""
        from comparisons.comparison_metrics import compare_efficiency

        timing_a = {
            'assembly_displacement_mean': 0.1,
            'solve_displacement_mean': 0.2,
            'assembly_damage_mean': 0.05,
            'solve_damage_mean': 0.15,
            'total_per_step_mean': 0.5,
        }
        timing_b = {
            'assembly_displacement_mean': 0.12,
            'solve_displacement_mean': 0.18,
            'assembly_damage_mean': 0.08,
            'solve_damage_mean': 0.12,
            'total_per_step_mean': 0.5,
        }
        dofs_a = {'displacement': 2000, 'damage': 1500}
        dofs_b = {'displacement': 2000, 'damage': 1000}

        result = compare_efficiency(timing_a, timing_b, dofs_a, dofs_b)

        assert result['damage_dof_ratio'] == 1.5
        assert 'total_per_step_mean_ratio' in result

    def test_validate_comparison(self):
        """Test validation against criteria."""
        from comparisons.comparison_metrics import validate_comparison

        comparison = {
            'path_deviation': {'hausdorff': 0.005},
            'load_displacement': {'peak_load_error': 0.03, 'l2_error': 0.08},
            'h_fine': 0.01,
        }
        criteria = {
            'sent_hausdorff_h_ratio': 1.0,
            'sent_peak_load_error': 0.05,
            'sent_l2_error': 0.10,
        }

        result = validate_comparison(comparison, criteria)

        assert result['all_passed'] is True
        assert result['criteria']['sent_hausdorff_h_ratio']['passed'] is True
        assert result['criteria']['sent_peak_load_error']['passed'] is True
        assert result['criteria']['sent_l2_error']['passed'] is True


# ============================================================
# Test: FEM Phase-Field Reference Solver
# ============================================================

class TestFEMPhasefieldSolver:
    """Tests for the FEM phase-field reference solver."""

    def test_construction(self, coarse_mesh, simple_material, simple_config):
        """Solver should initialize without errors."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)

        assert solver.n_nodes == coarse_mesh.n_nodes
        assert solver.n_elements == coarse_mesh.n_elements
        assert len(solver.damage) == coarse_mesh.n_nodes
        assert len(solver.displacement) == 2 * coarse_mesh.n_nodes

    def test_damage_is_node_based(self, coarse_mesh, simple_material, simple_config):
        """Damage field should be node-based (n_nodes), not edge-based."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)

        assert solver.damage.shape == (coarse_mesh.n_nodes,)
        # This is the key difference from GraFEA-PF

    def test_stiffness_assembly(self, coarse_mesh, simple_material, simple_config):
        """Stiffness matrix should be SPD with no damage."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)
        K = solver._assemble_stiffness()

        assert K.shape == (2 * coarse_mesh.n_nodes, 2 * coarse_mesh.n_nodes)
        # Check symmetry
        diff = (K - K.T).data
        assert np.max(np.abs(diff)) < 1e-10 if len(diff) > 0 else True

    def test_damage_system_assembly(self, coarse_mesh, simple_material, simple_config):
        """Damage system should be assembled correctly."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)
        # Set some history to trigger nontrivial RHS
        solver.history[:] = simple_material.critical_strain_energy_density() * 0.5

        A_d, b_d = solver._assemble_damage_system()

        assert A_d.shape == (coarse_mesh.n_nodes, coarse_mesh.n_nodes)
        assert b_d.shape == (coarse_mesh.n_nodes,)
        assert np.all(b_d >= 0), "RHS should be non-negative"

    def test_initial_damage_from_edges(self, coarse_mesh, simple_material, simple_config):
        """Converting edge damage to node damage should work."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)

        # Create some edge damage
        edge_damage = np.zeros(coarse_mesh.n_edges)
        # Damage some edges
        for i in range(min(10, coarse_mesh.n_edges)):
            edge_damage[i] = 0.8

        solver.set_initial_damage_from_edges(edge_damage, coarse_mesh)

        # Some nodes should have nonzero damage
        assert np.max(solver.damage) > 0
        assert np.min(solver.damage) >= 0
        assert np.max(solver.damage) <= 1.0

    def test_timing_summary(self, coarse_mesh, simple_material, simple_config):
        """Timing summary should return correct keys."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver

        solver = FEMPhasefieldSolver(coarse_mesh, simple_material, simple_config)
        summary = solver.get_timing_summary()

        assert 'assembly_displacement' in summary
        assert 'solve_displacement' in summary
        assert 'assembly_damage' in summary
        assert 'solve_damage' in summary
        assert 'total_mean' in summary


# ============================================================
# Test: Original GraFEA Solver
# ============================================================

class TestOriginalGraFEASolver:
    """Tests for the original GraFEA solver."""

    def test_construction(self, coarse_mesh, simple_material, simple_config):
        """Solver should initialize without errors."""
        from comparisons.original_grafea import OriginalGraFEASolver

        solver = OriginalGraFEASolver(coarse_mesh, simple_material, simple_config)

        assert solver.n_edges == coarse_mesh.n_edges
        assert len(solver.damage) == coarse_mesh.n_edges
        assert len(solver.displacement) == 2 * coarse_mesh.n_nodes

    def test_binary_damage(self, coarse_mesh, simple_material, simple_config):
        """Damage should be binary {0, 1}."""
        from comparisons.original_grafea import OriginalGraFEASolver

        solver = OriginalGraFEASolver(coarse_mesh, simple_material, simple_config)

        # Initial damage should be all zeros
        assert np.all(solver.damage == 0)

        # After setting initial damage
        d_init = np.zeros(coarse_mesh.n_edges)
        d_init[:5] = 1.0
        solver.set_initial_damage(d_init)

        unique_vals = np.unique(solver.damage)
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_critical_strains(self, coarse_mesh, simple_material, simple_config):
        """Critical strains should be positive and finite."""
        from comparisons.original_grafea import OriginalGraFEASolver

        solver = OriginalGraFEASolver(coarse_mesh, simple_material, simple_config)

        assert len(solver.eps_crit) == coarse_mesh.n_edges
        assert np.all(solver.eps_crit > 0)
        assert np.all(np.isfinite(solver.eps_crit))

    def test_irreversibility(self, coarse_mesh, simple_material, simple_config):
        """Once broken, edges should stay broken."""
        from comparisons.original_grafea import OriginalGraFEASolver

        solver = OriginalGraFEASolver(coarse_mesh, simple_material, simple_config)

        # Break some edges
        d = np.zeros(coarse_mesh.n_edges)
        d[:3] = 1.0
        solver.set_initial_damage(d)

        # These edges should remain broken
        assert np.all(solver.damage[:3] == 1.0)

    def test_elements_created(self, coarse_mesh, simple_material, simple_config):
        """Elements should be GraFEAElements."""
        from comparisons.original_grafea import OriginalGraFEASolver

        solver = OriginalGraFEASolver(coarse_mesh, simple_material, simple_config)

        assert len(solver.elements) == coarse_mesh.n_elements
        assert all(isinstance(e, GraFEAElement) for e in solver.elements)


# ============================================================
# Test: Integration - Quick Benchmark Runs
# ============================================================

class TestQuickBenchmarks:
    """Quick integration tests that run actual simulations."""

    def test_fem_pf_simple_tension(self):
        """FEM-PF should solve a simple tension test."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver
        from assembly.boundary_conditions import create_bc_from_region, merge_bcs

        mesh = create_rectangle_mesh(1.0, 1.0, 8, 8)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.15)
        config = SolverConfig(
            tol_u=1e-4, tol_d=1e-4,
            max_stagger_iter=10, verbose=False,
        )

        solver = FEMPhasefieldSolver(mesh, material, config)

        # BCs: fix bottom, prescribe top
        tol = 1e-6
        bc_bot_dofs, bc_bot_vals = create_bc_from_region(
            mesh, lambda x, y: y < tol, 'y', 0.0
        )
        bot_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
        bc_corner = (np.array([2 * bot_nodes[0]]), np.array([0.0]))
        fixed_dofs, fixed_vals = merge_bcs(
            (bc_bot_dofs, bc_bot_vals), bc_corner
        )
        top_nodes = mesh.get_nodes_in_region(lambda x, y: y > 1.0 - tol)
        top_dofs = np.array([2 * n + 1 for n in top_nodes])
        all_bc_dofs = np.concatenate([fixed_dofs, top_dofs])
        n_fixed = len(fixed_vals)

        def bc_func(u_app):
            vals = np.zeros(len(all_bc_dofs))
            vals[:n_fixed] = fixed_vals
            vals[n_fixed:] = u_app
            return vals

        u_steps = np.linspace(0, 0.005, 10)
        results = solver.solve(u_steps, all_bc_dofs, bc_func)

        assert len(results) == len(u_steps)
        assert all(isinstance(r, LoadStep) for r in results)
        # Should have positive strain energy for nonzero displacement
        assert results[-1].strain_energy > 0

    def test_original_grafea_simple_tension(self):
        """Original GraFEA should solve a simple tension test."""
        from comparisons.original_grafea import OriginalGraFEASolver
        from assembly.boundary_conditions import create_bc_from_region, merge_bcs

        mesh = create_rectangle_mesh(1.0, 1.0, 8, 8)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.15)
        config = SolverConfig(
            tol_u=1e-4, tol_d=1e-4,
            max_stagger_iter=10, verbose=False,
        )

        solver = OriginalGraFEASolver(mesh, material, config)

        # BCs
        tol = 1e-6
        bc_bot_dofs, bc_bot_vals = create_bc_from_region(
            mesh, lambda x, y: y < tol, 'y', 0.0
        )
        bot_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
        bc_corner = (np.array([2 * bot_nodes[0]]), np.array([0.0]))
        fixed_dofs, fixed_vals = merge_bcs(
            (bc_bot_dofs, bc_bot_vals), bc_corner
        )
        top_nodes = mesh.get_nodes_in_region(lambda x, y: y > 1.0 - tol)
        top_dofs = np.array([2 * n + 1 for n in top_nodes])
        all_bc_dofs = np.concatenate([fixed_dofs, top_dofs])
        n_fixed = len(fixed_vals)

        def bc_func(u_app):
            vals = np.zeros(len(all_bc_dofs))
            vals[:n_fixed] = fixed_vals
            vals[n_fixed:] = u_app
            return vals

        u_steps = np.linspace(0, 0.005, 10)
        results = solver.solve(u_steps, all_bc_dofs, bc_func)

        assert len(results) == len(u_steps)
        assert all(isinstance(r, LoadStep) for r in results)

    def test_dof_difference(self):
        """GraFEA-PF should have n_edges damage DOFs, FEM-PF should have n_nodes."""
        from comparisons.fem_pf_reference import FEMPhasefieldSolver
        from comparisons.original_grafea import OriginalGraFEASolver

        mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
        material = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=0.1)
        config = SolverConfig(verbose=False)

        fem = FEMPhasefieldSolver(mesh, material, config)
        orig = OriginalGraFEASolver(mesh, material, config)

        assert len(fem.damage) == mesh.n_nodes, "FEM-PF damage should be node-based"
        assert len(orig.damage) == mesh.n_edges, "Original GraFEA damage should be edge-based"

        # For triangular mesh, n_edges > n_nodes (approximately 1.5x)
        assert mesh.n_edges > mesh.n_nodes


# ============================================================
# Test: Plotting (smoke test)
# ============================================================

class TestPlotting:
    """Smoke tests for plotting module (no visual validation)."""

    def test_import(self):
        """Plotting module should import."""
        from comparisons.plotting import FIGURE_PARAMS
        assert 'figsize' in FIGURE_PARAMS

    def test_plot_sent_crack_paths(self, tmp_path):
        """Should generate a plot without errors."""
        from comparisons.plotting import plot_sent_crack_paths

        path_a = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.5)])
        path_b = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.51)])

        fig = plot_sent_crack_paths(path_a, path_b,
                                    save_path=str(tmp_path / 'test.png'))
        # Should create a file
        assert (tmp_path / 'test.png').exists()


# ============================================================
# Test: Length Scale Study
# ============================================================

class TestLengthScaleStudy:
    """Tests for length scale study module."""

    def test_measure_damage_band_width(self):
        """Band width measurement should work on simple data."""
        from comparisons.length_scale_study import measure_damage_band_width

        mesh = create_rectangle_mesh(1.0, 1.0, 20, 20)
        damage = np.zeros(mesh.n_edges)

        # Set damage for edges near y=0.5
        for i in range(mesh.n_edges):
            mid_y = mesh.edge_midpoints[i, 1]
            mid_x = mesh.edge_midpoints[i, 0]
            if abs(mid_y - 0.5) < 0.1 and mid_x > 0.3:
                damage[i] = 0.8

        width = measure_damage_band_width(damage, mesh, threshold=0.5, sample_x=0.5)
        assert width > 0
        assert width < 0.3  # Should be around 0.2 (band is ±0.1)


# ============================================================
# Test: Computational Cost
# ============================================================

class TestComputationalCost:
    """Tests for computational cost module."""

    def test_estimate_h_for_node_count(self):
        """Should estimate reasonable h values."""
        from comparisons.computational_cost import estimate_h_for_node_count

        h = estimate_h_for_node_count(1000, L=1.0)
        assert 0.01 < h < 0.1

        h_large = estimate_h_for_node_count(100, L=1.0)
        assert h_large > h  # Fewer nodes = larger elements

    def test_fit_scaling_exponent(self):
        """Should fit correct exponent for known data."""
        from comparisons.computational_cost import fit_scaling_exponent

        # time = 0.001 * N^1.5
        N = np.array([100, 500, 1000, 5000])
        t = 0.001 * N ** 1.5

        result = fit_scaling_exponent(N, t)
        assert abs(result['alpha'] - 1.5) < 0.1
        assert result['r_squared'] > 0.99
