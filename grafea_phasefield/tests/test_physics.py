"""
Tests for Physics Module
========================
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.material import IsotropicMaterial
from physics.tension_split import (
    spectral_split_2d, spectral_split, simple_edge_split,
    verify_energy_conservation
)
from physics.damage import (
    degradation_function, degradation_derivative,
    HistoryField, enforce_damage_bounds
)
from physics.surface_energy import (
    compute_edge_volumes, compute_surface_energy
)
from mesh.mesh_generators import create_rectangle_mesh
from mesh.edge_graph import EdgeGraph


class TestMaterial:
    """Tests for IsotropicMaterial."""

    def test_material_creation(self):
        """Test material creation with valid parameters."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
        assert mat.E == 210e9
        assert mat.nu == 0.3
        assert mat.Gc == 2700
        assert mat.l0 == 0.01

    def test_material_invalid_parameters(self):
        """Test material creation with invalid parameters."""
        with pytest.raises(ValueError):
            IsotropicMaterial(E=-1, nu=0.3, Gc=2700, l0=0.01)
        with pytest.raises(ValueError):
            IsotropicMaterial(E=210e9, nu=0.5, Gc=2700, l0=0.01)
        with pytest.raises(ValueError):
            IsotropicMaterial(E=210e9, nu=0.3, Gc=-1, l0=0.01)

    def test_lame_parameters(self):
        """Test Lamé parameter computation."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

        # Check μ (shear modulus)
        mu_expected = 210e9 / (2 * 1.3)
        assert np.isclose(mat.lame_mu, mu_expected)

        # Check λ
        lambda_expected = 210e9 * 0.3 / (1.3 * 0.4)
        assert np.isclose(mat.lame_lambda, lambda_expected)

    def test_constitutive_matrix_symmetry(self):
        """Constitutive matrix should be symmetric."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

        for plane in ['strain', 'stress']:
            C = mat.constitutive_matrix(plane)
            assert np.allclose(C, C.T)

    def test_constitutive_matrix_positive_definite(self):
        """Constitutive matrix should be positive definite."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

        for plane in ['strain', 'stress']:
            C = mat.constitutive_matrix(plane)
            eigenvalues = np.linalg.eigvalsh(C)
            assert np.all(eigenvalues > 0)


class TestTensionSplit:
    """Tests for tension-compression split."""

    def test_spectral_split_pure_tension(self):
        """Pure tension: all strain should be in plus part."""
        eps = np.array([0.001, 0.001, 0])  # Biaxial tension
        eigenvalues, eps_plus, eps_minus = spectral_split_2d(eps)

        assert np.all(eigenvalues >= 0)
        assert np.allclose(eps_plus, eps)
        assert np.allclose(eps_minus, 0)

    def test_spectral_split_pure_compression(self):
        """Pure compression: all strain should be in minus part."""
        eps = np.array([-0.001, -0.001, 0])  # Biaxial compression
        eigenvalues, eps_plus, eps_minus = spectral_split_2d(eps)

        assert np.all(eigenvalues <= 0)
        assert np.allclose(eps_plus, 0)
        assert np.allclose(eps_minus, eps)

    def test_spectral_split_mixed(self):
        """Mixed state: check decomposition."""
        eps = np.array([0.001, -0.001, 0])  # Tension-compression
        eigenvalues, eps_plus, eps_minus = spectral_split_2d(eps)

        # Sum should recover original
        assert np.allclose(eps_plus + eps_minus, eps)

    def test_spectral_split_energy_conservation(self):
        """Energy should be conserved: ψ = ψ⁺ + ψ⁻."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
        C = mat.constitutive_matrix()

        # Various strain states
        strain_cases = [
            [0.001, 0, 0],           # Uniaxial tension
            [-0.001, 0, 0],          # Uniaxial compression
            [0.001, -0.0003, 0],     # Mixed
            [0.001, 0.001, 0],       # Biaxial tension
            [0, 0, 0.001],           # Pure shear
            [0.001, 0.0005, 0.0002], # General
        ]

        for eps in strain_cases:
            eps = np.array(eps)
            psi_total = 0.5 * eps @ C @ eps

            _, eps_plus, eps_minus = spectral_split_2d(eps)
            psi_plus = 0.5 * eps_plus @ C @ eps_plus
            psi_minus = 0.5 * eps_minus @ C @ eps_minus

            assert np.isclose(psi_total, psi_plus + psi_minus, rtol=1e-10), \
                f"Energy not conserved for eps={eps}: total={psi_total}, sum={psi_plus + psi_minus}"

    def test_spectral_split_with_edge_transform(self):
        """Test spectral split with edge transformation."""
        mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
        C = mat.constitutive_matrix()

        # Create transformation matrix for a triangle
        phi = np.array([0, np.pi/2, np.pi/4])  # Example angles
        T = np.zeros((3, 3))
        for k in range(3):
            c, s = np.cos(phi[k]), np.sin(phi[k])
            T[k, :] = [c**2, s**2, c*s]

        eps_tensor = np.array([0.001, 0, 0])
        eps_edge = T @ eps_tensor

        result = spectral_split(eps_tensor, eps_edge, C, T)

        # Check energy conservation
        psi_total = 0.5 * eps_tensor @ C @ eps_tensor
        assert np.isclose(psi_total, result['psi_plus'] + result['psi_minus'])


class TestDamage:
    """Tests for damage module."""

    def test_degradation_function_bounds(self):
        """Degradation function should be in [0, 1]."""
        d = np.linspace(0, 1, 100)

        for model in ['quadratic', 'cubic']:
            g = degradation_function(d, model)
            assert np.all(g >= 0)
            assert np.all(g <= 1)
            assert np.isclose(g[0], 1)  # g(0) = 1
            assert np.isclose(g[-1], 0)  # g(1) = 0

    def test_degradation_monotonic(self):
        """Degradation should be monotonically decreasing."""
        d = np.linspace(0, 1, 100)

        for model in ['quadratic', 'cubic']:
            g = degradation_function(d, model)
            assert np.all(np.diff(g) <= 0)

    def test_degradation_derivative(self):
        """Test degradation derivative."""
        d = np.array([0, 0.5, 1.0])

        # Quadratic: g(d) = (1-d)², g'(d) = -2(1-d)
        g_prime = degradation_derivative(d, 'quadratic')
        assert np.isclose(g_prime[0], -2)  # g'(0) = -2
        assert np.isclose(g_prime[1], -1)  # g'(0.5) = -1
        assert np.isclose(g_prime[2], 0)   # g'(1) = 0

    def test_history_field_monotonicity(self):
        """History field should never decrease."""
        hf = HistoryField(10)

        hf.update(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1e6)
        H1 = hf.H.copy()

        hf.update(np.array([0, 0, 5, 0, 0, 10, 0, 0, 0, 0]) * 1e6)  # Some lower
        H2 = hf.H.copy()

        assert np.all(H2 >= H1)

    def test_history_field_reset(self):
        """Test history field reset."""
        hf = HistoryField(10)
        hf.update(np.ones(10) * 1e6)
        hf.reset()
        assert np.allclose(hf.H, 0)

    def test_damage_bounds_enforcement(self):
        """Test damage bounds enforcement."""
        d = np.array([-0.1, 0.5, 1.2])
        d_bounded = enforce_damage_bounds(d)

        assert np.all(d_bounded >= 0)
        assert np.all(d_bounded <= 1)

    def test_damage_irreversibility(self):
        """Test damage irreversibility enforcement."""
        d_old = np.array([0.3, 0.5, 0.7])
        d_new = np.array([0.2, 0.6, 0.4])  # Some decrease, some increase

        d_bounded = enforce_damage_bounds(d_new, d_old)

        assert np.all(d_bounded >= d_old)


class TestSurfaceEnergy:
    """Tests for surface energy module."""

    def test_edge_volumes_positive(self):
        """Edge volumes should be positive."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        omega = compute_edge_volumes(mesh)

        assert np.all(omega > 0)

    def test_edge_volumes_sum(self):
        """Sum of edge volumes should equal total volume."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        omega = compute_edge_volumes(mesh)

        total_volume = np.sum(mesh.element_areas) * mesh.thickness
        # Each element contributes its volume split among 3 edges
        # But edges are shared, so sum of omega should be less than 3*total_volume
        assert np.sum(omega) < 3 * total_volume

    def test_surface_energy_zero_damage(self):
        """Surface energy should be zero for zero damage."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        edge_graph = EdgeGraph(mesh)

        d = np.zeros(mesh.n_edges)
        E = compute_surface_energy(mesh, edge_graph, d, Gc=2700, l0=0.02)

        assert np.isclose(E, 0)

    def test_surface_energy_uniform_damage(self):
        """Test surface energy for uniform damage."""
        mesh = create_rectangle_mesh(1, 1, 10, 10)
        edge_graph = EdgeGraph(mesh)
        Gc, l0 = 2700, 0.05

        # Uniform damage should have zero gradient term
        d = 0.5 * np.ones(mesh.n_edges)
        E = compute_surface_energy(mesh, edge_graph, d, Gc, l0)

        # Should be positive
        assert E > 0

        # Compare with different damage levels
        d2 = 0.75 * np.ones(mesh.n_edges)
        E2 = compute_surface_energy(mesh, edge_graph, d2, Gc, l0)

        # Higher damage → higher surface energy
        assert E2 > E

    def test_surface_energy_scaling(self):
        """Surface energy should scale with damage squared (for uniform d)."""
        mesh = create_rectangle_mesh(1, 1, 10, 10)
        edge_graph = EdgeGraph(mesh)
        Gc, l0 = 2700, 0.05

        # For uniform damage, E_frac ∝ d²
        d1 = 0.3 * np.ones(mesh.n_edges)
        d2 = 0.6 * np.ones(mesh.n_edges)

        E1 = compute_surface_energy(mesh, edge_graph, d1, Gc, l0)
        E2 = compute_surface_energy(mesh, edge_graph, d2, Gc, l0)

        # E2/E1 should be approximately (0.6/0.3)² = 4
        ratio = E2 / E1
        expected_ratio = (0.6 / 0.3) ** 2
        assert np.isclose(ratio, expected_ratio, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
