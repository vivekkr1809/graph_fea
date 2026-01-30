"""
Constant Strain Triangle (CST) Element
======================================

Standard FEM element implementation for comparison with GraFEA.
"""

import numpy as np
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.material import IsotropicMaterial


class CSTElement:
    """
    Constant Strain Triangle element.

    This class handles standard FEM computations that will be compared
    against GraFEA results for verification.

    The strain is constant throughout the element (hence "constant strain").

    Attributes:
        nodes: shape (3, 2), node coordinates
        material: IsotropicMaterial instance
        thickness: element thickness
        plane: 'strain' or 'stress'
        area: element area
        B: shape (3, 6), strain-displacement matrix
        C: shape (3, 3), constitutive matrix
        K: shape (6, 6), element stiffness matrix
    """

    def __init__(self, nodes: np.ndarray, material: IsotropicMaterial,
                 thickness: float = 1.0, plane: str = 'strain'):
        """
        Initialize CST element.

        Args:
            nodes: shape (3, 2), coordinates of element nodes
            material: material properties
            thickness: element thickness for plane problems
            plane: 'strain' or 'stress'
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.material = material
        self.thickness = thickness
        self.plane = plane

        if self.nodes.shape != (3, 2):
            raise ValueError(f"nodes must have shape (3, 2), got {self.nodes.shape}")

        self._compute_geometry()
        self._compute_matrices()

    def _compute_geometry(self):
        """Compute area and shape function derivatives."""
        X = self.nodes

        # Area using cross product formula (Appendix B of derivation)
        # A = 0.5 * |det([X1-X0, X2-X0])|
        self.area = 0.5 * abs(
            (X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) -
            (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1])
        )

        if self.area < 1e-15:
            raise ValueError("Element has zero or negative area (degenerate triangle)")

        # Shape function derivatives (constant for CST)
        # N_i = (a_i + b_i*x + c_i*y) / (2*A)
        # where:
        #   b_i = y_j - y_k (derivative w.r.t. x)
        #   c_i = x_k - x_j (derivative w.r.t. y)
        # for cyclic (i, j, k) = (0, 1, 2), (1, 2, 0), (2, 0, 1)

        self.b = np.array([
            X[1, 1] - X[2, 1],  # b_0 = y_1 - y_2
            X[2, 1] - X[0, 1],  # b_1 = y_2 - y_0
            X[0, 1] - X[1, 1]   # b_2 = y_0 - y_1
        ])

        self.c = np.array([
            X[2, 0] - X[1, 0],  # c_0 = x_2 - x_1
            X[0, 0] - X[2, 0],  # c_1 = x_0 - x_2
            X[1, 0] - X[0, 0]   # c_2 = x_1 - x_0
        ])

    def _compute_matrices(self):
        """Compute B (strain-displacement) and K (stiffness) matrices."""
        # B matrix: ε = B @ u
        # ε = [ε_xx, ε_yy, γ_xy]^T
        # u = [u_0x, u_0y, u_1x, u_1y, u_2x, u_2y]^T

        self.B = np.zeros((3, 6))
        for i in range(3):
            self.B[0, 2 * i] = self.b[i]       # ∂u_x/∂x → ε_xx
            self.B[1, 2 * i + 1] = self.c[i]   # ∂u_y/∂y → ε_yy
            self.B[2, 2 * i] = self.c[i]       # ∂u_x/∂y → γ_xy
            self.B[2, 2 * i + 1] = self.b[i]   # ∂u_y/∂x → γ_xy
        self.B /= (2 * self.area)

        # Constitutive matrix
        self.C = self.material.constitutive_matrix(self.plane)

        # Stiffness matrix: K = ∫ B^T C B dV = B^T C B × A × h
        self.K = self.B.T @ self.C @ self.B * self.area * self.thickness

    def compute_strain(self, u: np.ndarray) -> np.ndarray:
        """
        Compute strain from nodal displacements.

        Args:
            u: shape (3, 2) or (6,), nodal displacements

        Returns:
            ε: shape (3,), strain tensor in Voigt notation [ε_xx, ε_yy, γ_xy]
        """
        u_flat = np.asarray(u).flatten()
        if len(u_flat) != 6:
            raise ValueError(f"u must have 6 components, got {len(u_flat)}")
        return self.B @ u_flat

    def compute_stress(self, u: np.ndarray) -> np.ndarray:
        """
        Compute stress from nodal displacements.

        Args:
            u: nodal displacements

        Returns:
            σ: shape (3,), stress tensor in Voigt notation [σ_xx, σ_yy, τ_xy]
        """
        eps = self.compute_strain(u)
        return self.C @ eps

    def compute_strain_energy_density(self, u: np.ndarray) -> float:
        """
        Compute strain energy density.

        ψ = ½ ε:C:ε = ½ ε^T C ε

        Args:
            u: nodal displacements

        Returns:
            ψ: strain energy density [J/m³]
        """
        eps = self.compute_strain(u)
        return 0.5 * eps @ self.C @ eps

    def compute_strain_energy(self, u: np.ndarray) -> float:
        """
        Compute total strain energy in element.

        E = ψ × V = ψ × A × h

        Args:
            u: nodal displacements

        Returns:
            E: strain energy [J]
        """
        psi = self.compute_strain_energy_density(u)
        return psi * self.area * self.thickness

    def compute_internal_force(self, u: np.ndarray) -> np.ndarray:
        """
        Compute internal force vector.

        F_int = ∫ B^T σ dV = B^T σ × A × h

        Args:
            u: nodal displacements

        Returns:
            F: shape (6,), internal force vector
        """
        sigma = self.compute_stress(u)
        return self.B.T @ sigma * self.area * self.thickness

    def compute_principal_strains(self, u: np.ndarray) -> tuple:
        """
        Compute principal strains and directions.

        Args:
            u: nodal displacements

        Returns:
            eigenvalues: principal strains (sorted)
            eigenvectors: principal directions
        """
        eps = self.compute_strain(u)
        eps_xx, eps_yy, gamma_xy = eps
        eps_xy = gamma_xy / 2

        # Build strain tensor
        E_mat = np.array([[eps_xx, eps_xy],
                          [eps_xy, eps_yy]])

        eigenvalues, eigenvectors = np.linalg.eigh(E_mat)
        # Sort by magnitude (descending)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    def compute_von_mises_stress(self, u: np.ndarray) -> float:
        """
        Compute von Mises equivalent stress.

        For 2D plane strain: σ_vm = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3τ_xy²)

        Args:
            u: nodal displacements

        Returns:
            σ_vm: von Mises stress
        """
        sigma = self.compute_stress(u)
        sig_xx, sig_yy, tau_xy = sigma
        return np.sqrt(sig_xx ** 2 + sig_yy ** 2 - sig_xx * sig_yy + 3 * tau_xy ** 2)

    def mass_matrix(self, density: float = 1.0, lumped: bool = False) -> np.ndarray:
        """
        Compute element mass matrix.

        Args:
            density: material density
            lumped: if True, return lumped (diagonal) mass matrix

        Returns:
            M: shape (6, 6), mass matrix
        """
        if lumped:
            # Lumped mass: total mass distributed equally to nodes
            total_mass = density * self.area * self.thickness
            node_mass = total_mass / 3
            M = np.diag([node_mass] * 6)
        else:
            # Consistent mass matrix
            # For CST: M_ij = ρ·A·h·(1+δ_ij)/12
            M = np.zeros((6, 6))
            factor = density * self.area * self.thickness / 12
            for i in range(3):
                for j in range(3):
                    if i == j:
                        val = 2 * factor
                    else:
                        val = factor
                    M[2 * i, 2 * j] = val
                    M[2 * i + 1, 2 * j + 1] = val
        return M
