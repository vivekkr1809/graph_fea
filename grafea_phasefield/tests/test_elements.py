"""
Tests for Elements Module
=========================
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from elements.cst_element import CSTElement
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial


class TestCSTElement:
    """Tests for CST element."""

    @pytest.fixture
    def setup_element(self):
        """Create standard test element."""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
        return CSTElement(nodes, material)

    def test_area_computation(self, setup_element):
        """Test element area."""
        elem = setup_element
        assert np.isclose(elem.area, 0.5)

    def test_B_matrix_shape(self, setup_element):
        """Test B matrix shape."""
        elem = setup_element
        assert elem.B.shape == (3, 6)

    def test_stiffness_matrix_symmetry(self, setup_element):
        """Stiffness matrix should be symmetric."""
        elem = setup_element
        assert np.allclose(elem.K, elem.K.T)

    def test_stiffness_matrix_positive_definite(self, setup_element):
        """Stiffness matrix should be positive semi-definite."""
        elem = setup_element
        eigenvalues = np.linalg.eigvalsh(elem.K)
        # Should have 3 zero eigenvalues (rigid body modes) and 3 positive
        assert np.all(eigenvalues >= -1e-10)

    def test_strain_computation(self, setup_element):
        """Test strain computation."""
        elem = setup_element

        # Uniform x-displacement should give zero strain
        u_rigid = np.array([[0.1, 0], [0.1, 0], [0.1, 0]])
        eps = elem.compute_strain(u_rigid)
        assert np.allclose(eps, 0, atol=1e-10)

        # Linear x-displacement should give constant ε_xx
        u_linear = np.array([[0, 0], [0.001, 0], [0, 0]])
        eps = elem.compute_strain(u_linear)
        # ε_xx = ∂u_x/∂x ≈ 0.001 / 1 = 0.001
        assert eps[0] > 0  # ε_xx > 0

    def test_strain_energy_density(self, setup_element):
        """Test strain energy density computation."""
        elem = setup_element
        u = np.array([[0, 0], [0.001, 0], [0, 0]])

        psi = elem.compute_strain_energy_density(u)
        assert psi > 0


class TestGraFEAElement:
    """Tests for GraFEA element."""

    @pytest.fixture
    def setup_elements(self):
        """Create matching CST and GraFEA elements."""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
        cst = CSTElement(nodes, material)
        grafea = GraFEAElement(nodes, material)
        return cst, grafea

    def test_edge_lengths(self, setup_elements):
        """Test edge length computation."""
        _, grafea = setup_elements

        # For right triangle with legs 1, hypotenuse √2
        expected = sorted([1.0, 1.0, np.sqrt(2)])
        actual = sorted(grafea.L)
        assert np.allclose(actual, expected)

    def test_transformation_matrix_invertible(self, setup_elements):
        """T matrix should be invertible."""
        _, grafea = setup_elements
        assert np.linalg.cond(grafea.T) < 1e10

    def test_strain_energy_equivalence(self, setup_elements):
        """
        CRITICAL TEST: Verify tensor and edge energy formulations are equivalent.
        """
        cst, grafea = setup_elements

        # Test multiple displacement states
        test_cases = [
            np.array([[0, 0], [0.001, 0], [0, 0]]),       # Uniaxial X
            np.array([[0, 0], [0, 0], [0, 0.001]]),       # Uniaxial Y
            np.array([[0, 0], [0, 0.0005], [0.0005, 0]]), # Shear
            np.array([[0, 0], [0.001, 0], [0, 0.001]]),   # Biaxial
            np.array([[0, 0], [0.001, 0.0002], [0.0003, 0.001]]),  # General
        ]

        for u in test_cases:
            psi_tensor = cst.compute_strain_energy_density(u)
            eps_edge = grafea.compute_edge_strains(u)
            psi_edge = grafea.strain_energy_density(eps_edge)

            assert np.isclose(psi_tensor, psi_edge, rtol=1e-10), \
                f"Energy mismatch: tensor={psi_tensor}, edge={psi_edge}"

    def test_stiffness_matrix_equivalence(self, setup_elements):
        """Verify damaged stiffness reduces to standard when d=0."""
        cst, grafea = setup_elements

        d = np.zeros(3)  # No damage
        K_grafea = grafea.compute_stiffness_damaged(d)

        assert np.allclose(K_grafea, cst.K, rtol=1e-10)

    def test_force_equilibrium(self, setup_elements):
        """Internal forces should sum to zero."""
        _, grafea = setup_elements
        u = np.array([[0, 0], [0.001, 0], [0, 0]])

        eps_edge = grafea.compute_edge_strains(u)
        F = grafea.compute_internal_force_grafea(eps_edge, np.zeros(3))

        # Sum of forces should be zero (equilibrium)
        assert np.allclose(np.sum(F, axis=0), 0, atol=1e-10)

    def test_damage_reduces_stiffness(self, setup_elements):
        """Damage should reduce element stiffness."""
        _, grafea = setup_elements

        K0 = grafea.compute_stiffness_damaged(np.array([0, 0, 0]))
        K_partial = grafea.compute_stiffness_damaged(np.array([0.5, 0, 0]))
        K_full = grafea.compute_stiffness_damaged(np.array([1, 1, 1]))

        # Check stiffness decreases (use Frobenius norm)
        assert np.linalg.norm(K_partial) < np.linalg.norm(K0)
        assert np.allclose(K_full, 0)  # Fully damaged = zero stiffness

    def test_edge_strains_direct_vs_transform(self, setup_elements):
        """Compare direct and transformation-based edge strain computation."""
        _, grafea = setup_elements

        # Small displacement
        u = np.array([[0, 0], [0.001, 0], [0, 0]])

        eps_transform = grafea.compute_edge_strains(u)
        eps_direct = grafea.compute_edge_strains_direct(u)

        # Should be close for small strains
        assert np.allclose(eps_transform, eps_direct, rtol=0.01)

    def test_verify_energy_equivalence_method(self, setup_elements):
        """Test the built-in verification method."""
        _, grafea = setup_elements

        u = np.array([[0, 0], [0.001, 0], [0, 0]])
        match, psi_tensor, psi_edge = grafea.verify_energy_equivalence(u)

        assert match
        assert psi_tensor > 0
        assert np.isclose(psi_tensor, psi_edge)


class TestElementEquivalence:
    """Integration tests comparing CST and GraFEA."""

    def test_multiple_element_shapes(self):
        """Test energy equivalence for various element shapes."""
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

        # Various element shapes
        element_shapes = [
            np.array([[0, 0], [1, 0], [0, 1]]),  # Right triangle
            np.array([[0, 0], [1, 0], [0.5, 0.866]]),  # Equilateral
            np.array([[0, 0], [2, 0], [1, 1]]),  # Isosceles
            np.array([[0, 0], [3, 0], [1, 2]]),  # Scalene
        ]

        u = np.array([[0, 0], [0.001, 0], [0, 0.0005]])

        for nodes in element_shapes:
            cst = CSTElement(nodes, material)
            grafea = GraFEAElement(nodes, material)

            psi_cst = cst.compute_strain_energy_density(u)
            eps_edge = grafea.compute_edge_strains(u)
            psi_grafea = grafea.strain_energy_density(eps_edge)

            assert np.isclose(psi_cst, psi_grafea, rtol=1e-10), \
                f"Mismatch for nodes:\n{nodes}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
