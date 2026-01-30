"""
GraFEA Element
==============

Edge-based finite element for Graph-based FEA framework.

Key insight (Reddy-Srinivasa): Strain energy depends only on edge lengths,
and forces are directed along edges.
"""

import numpy as np
from typing import Optional, Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.material import IsotropicMaterial


class GraFEAElement:
    """
    GraFEA element with edge-based strain energy formulation.

    This element computes strain energy in terms of edge strains rather
    than the traditional tensor strain. The key transformation is:

        ε_edge = T @ ε_tensor

    where T is the transformation matrix based on edge angles.

    Edge Convention:
        Edge 0: connects nodes 1-2 (opposite to node 0)
        Edge 1: connects nodes 2-0 (opposite to node 1)
        Edge 2: connects nodes 0-1 (opposite to node 2)

    Attributes:
        nodes: shape (3, 2), node coordinates
        material: IsotropicMaterial instance
        thickness: element thickness
        plane: 'strain' or 'stress'
        area: element area
        L: shape (3,), reference edge lengths
        phi: shape (3,), edge angles with x-axis
        T: shape (3, 3), transformation matrix ε_edge = T @ ε_tensor
        A: shape (3, 3), edge stiffness matrix
    """

    def __init__(self, nodes: np.ndarray, material: IsotropicMaterial,
                 thickness: float = 1.0, plane: str = 'strain'):
        """
        Initialize GraFEA element.

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
        self._compute_transformation()
        self._compute_edge_stiffness()
        self._compute_B_matrix()

    def _compute_geometry(self):
        """Compute area, edge lengths, and edge angles."""
        X = self.nodes

        # Area
        self.area = 0.5 * abs(
            (X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) -
            (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1])
        )

        if self.area < 1e-15:
            raise ValueError("Element has zero or negative area")

        # Edge node pairs: edge k is opposite to node k
        # Edge 0: nodes 1-2, Edge 1: nodes 2-0, Edge 2: nodes 0-1
        self.edge_nodes = [(1, 2), (2, 0), (0, 1)]

        # Edge lengths and angles
        self.L = np.zeros(3)      # Reference lengths
        self.phi = np.zeros(3)    # Angles with x-axis
        self.e_hat = np.zeros((3, 2))  # Unit vectors along edges

        for k, (i, j) in enumerate(self.edge_nodes):
            dx = X[j, 0] - X[i, 0]
            dy = X[j, 1] - X[i, 1]
            self.L[k] = np.sqrt(dx ** 2 + dy ** 2)
            self.phi[k] = np.arctan2(dy, dx)
            self.e_hat[k] = np.array([dx, dy]) / self.L[k]

    def _compute_transformation(self):
        """
        Compute transformation matrix T: ε_edge = T @ ε_tensor

        From the derivation (Eq. 3.7), for edge k at angle φ_k:
            T_k = [cos²φ_k, sin²φ_k, cosφ_k sinφ_k]

        This transforms [ε_xx, ε_yy, γ_xy] to edge strain ε_k.
        """
        self.T = np.zeros((3, 3))
        for k in range(3):
            c = np.cos(self.phi[k])
            s = np.sin(self.phi[k])
            # ε_k = ε_xx cos²φ + ε_yy sin²φ + γ_xy cosφ sinφ
            self.T[k, :] = [c ** 2, s ** 2, c * s]

        # Check that T is invertible
        det_T = np.linalg.det(self.T)
        if abs(det_T) < 1e-10:
            raise ValueError("Transformation matrix T is singular (degenerate element)")

        self.T_inv = np.linalg.inv(self.T)

    def _compute_edge_stiffness(self):
        """
        Compute edge stiffness matrix A = T^{-T} C T^{-1}

        This transforms the constitutive relation to edge space.
        The strain energy can then be written as:
            ψ₀ = ½ ε_edge^T A ε_edge

        From derivation Eq. (4.5).
        """
        C = self.material.constitutive_matrix(self.plane)
        self.C = C
        self.A = self.T_inv.T @ C @ self.T_inv

    def _compute_B_matrix(self):
        """Compute B matrix for strain calculation."""
        X = self.nodes
        b = np.array([
            X[1, 1] - X[2, 1],
            X[2, 1] - X[0, 1],
            X[0, 1] - X[1, 1]
        ])
        c = np.array([
            X[2, 0] - X[1, 0],
            X[0, 0] - X[2, 0],
            X[1, 0] - X[0, 0]
        ])

        self.B = np.zeros((3, 6))
        for i in range(3):
            self.B[0, 2 * i] = b[i]
            self.B[1, 2 * i + 1] = c[i]
            self.B[2, 2 * i] = c[i]
            self.B[2, 2 * i + 1] = b[i]
        self.B /= (2 * self.area)

    def compute_tensor_strain(self, u: np.ndarray) -> np.ndarray:
        """
        Compute tensor strain from nodal displacements.

        Args:
            u: shape (3, 2) or (6,), nodal displacements

        Returns:
            ε: shape (3,), strain tensor in Voigt notation [ε_xx, ε_yy, γ_xy]
        """
        u_flat = np.asarray(u).flatten()
        return self.B @ u_flat

    def compute_edge_strains(self, u: np.ndarray) -> np.ndarray:
        """
        Compute edge strains from nodal displacements.

        Uses transformation: ε_edge = T @ ε_tensor

        Args:
            u: shape (3, 2) or (6,), nodal displacements

        Returns:
            ε_edge: shape (3,), strains along each edge
        """
        eps_tensor = self.compute_tensor_strain(u)
        return self.T @ eps_tensor

    def compute_edge_strains_direct(self, u: np.ndarray) -> np.ndarray:
        """
        Compute edge strains directly from length change.

        ε_k = (ℓ_k - L_k) / L_k

        This is more intuitive and matches the GraFEA philosophy.
        Note: For small strains, this matches compute_edge_strains().

        Args:
            u: shape (3, 2), nodal displacements

        Returns:
            ε_edge: shape (3,), strains along each edge
        """
        u = np.asarray(u).reshape(3, 2)
        x = self.nodes + u  # Current positions

        eps_edge = np.zeros(3)
        for k, (i, j) in enumerate(self.edge_nodes):
            ell_k = np.linalg.norm(x[j] - x[i])  # Current length
            eps_edge[k] = (ell_k - self.L[k]) / self.L[k]

        return eps_edge

    def strain_energy_density(self, eps_edge: np.ndarray) -> float:
        """
        Compute undamaged strain energy density.

        ψ₀ = ½ ε_edge^T A ε_edge

        Args:
            eps_edge: edge strains, shape (3,)

        Returns:
            ψ: strain energy density [J/m³]
        """
        return 0.5 * eps_edge @ self.A @ eps_edge

    def strain_energy_density_tensor(self, u: np.ndarray) -> float:
        """
        Compute strain energy density using tensor formulation.

        ψ₀ = ½ ε^T C ε

        For verification that edge and tensor formulations match.

        Args:
            u: nodal displacements

        Returns:
            ψ: strain energy density [J/m³]
        """
        eps = self.compute_tensor_strain(u)
        return 0.5 * eps @ self.C @ eps

    def strain_energy_density_degraded(self, eps_edge: np.ndarray,
                                        d: np.ndarray,
                                        split_result: Optional[Dict] = None) -> float:
        """
        Compute degraded strain energy density with tension-compression split.

        From derivation Eq. (5.6):
        ψ_d = ½ Σ_ij A_ij (1-d_i)(1-d_j) ε_i⁺ ε_j⁺ + ½ Σ_ij A_ij ε_i⁻ ε_j⁻

        Args:
            eps_edge: edge strains, shape (3,)
            d: damage on edges, shape (3,)
            split_result: pre-computed tension-compression split (optional)

        Returns:
            ψ_d: degraded strain energy density [J/m³]
        """
        if split_result is None:
            # Import here to avoid circular dependency
            from physics.tension_split import spectral_split
            eps_tensor = self.T_inv @ eps_edge
            split_result = spectral_split(eps_tensor, eps_edge, self.C, self.T)

        eps_plus = split_result['eps_edge_plus']
        eps_minus = split_result['eps_edge_minus']

        # Degradation matrix Φ = diag(1-d_i)
        Phi = np.diag(1 - d)

        # Degraded tensile energy
        # Uses Φ @ A @ Φ to degrade both i and j contributions
        psi_plus_d = 0.5 * eps_plus @ Phi @ self.A @ Phi @ eps_plus

        # Undegraded compressive energy
        psi_minus = 0.5 * eps_minus @ self.A @ eps_minus

        return psi_plus_d + psi_minus

    def compute_stiffness_undamaged(self) -> np.ndarray:
        """
        Compute undamaged element stiffness matrix.

        K = B^T C B × A × h

        Returns:
            K: shape (6, 6), element stiffness matrix
        """
        return self.B.T @ self.C @ self.B * self.area * self.thickness

    def compute_stiffness_damaged(self, d: np.ndarray) -> np.ndarray:
        """
        Compute damaged element stiffness matrix.

        K^d = B^T T^{-T} Φ A Φ T^{-1} B × Area × h

        where Φ = diag(1-d_i)

        For the tensile part only. In practice, we need the full
        stiffness accounting for tension-compression split.

        Args:
            d: damage on edges, shape (3,)

        Returns:
            K: shape (6, 6), damaged element stiffness matrix
        """
        # Degradation matrix
        Phi = np.diag(1 - d)

        # Damaged edge stiffness
        A_damaged = Phi @ self.A @ Phi

        # Transform back to tensor space
        C_damaged = self.T.T @ A_damaged @ self.T

        # Element stiffness
        K = self.B.T @ C_damaged @ self.B * self.area * self.thickness

        return K

    def compute_stiffness_with_split(self, d: np.ndarray,
                                      eps_tensor: np.ndarray) -> np.ndarray:
        """
        Compute damaged stiffness with proper tension-compression split.

        The stiffness depends on the current strain state because the
        split between tension and compression affects which part gets
        degraded.

        For the linearized tangent stiffness in the tensile regime:
        K^d = B^T C^+ B × g(d) × A × h + B^T C^- B × A × h

        Args:
            d: damage on edges, shape (3,)
            eps_tensor: current strain state

        Returns:
            K: shape (6, 6), damaged element stiffness
        """
        from physics.tension_split import spectral_split_2d

        # Get split in tensor space
        _, eps_plus, eps_minus = spectral_split_2d(eps_tensor)

        # Compute degradation factor (average over edges)
        g = np.mean((1 - d) ** 2)

        # Simplified approach: degrade based on whether strain is tensile
        # For a more sophisticated approach, use algorithmic tangent
        trace_eps = eps_tensor[0] + eps_tensor[1]

        if trace_eps > 0:
            # Predominantly tensile - degrade full stiffness
            return self.compute_stiffness_damaged(d)
        else:
            # Predominantly compressive - no degradation
            return self.compute_stiffness_undamaged()

    def compute_internal_force_grafea(self, eps_edge: np.ndarray,
                                       d: np.ndarray) -> np.ndarray:
        """
        Compute internal force in GraFEA form.

        From derivation Eq. (7.6), the force on node I from edge IJ:
        F_I = A·h · Σ_{J≠I} [Σ_j A_{(IJ),j} (1-d_IJ)(1-d_j) ε_j] · e_IJ / L_IJ

        Args:
            eps_edge: edge strains, shape (3,)
            d: damage on edges, shape (3,)

        Returns:
            F: shape (3, 2), nodal forces (or shape (6,) flattened)
        """
        F = np.zeros((3, 2))

        # Degradation factors
        g = 1 - d  # Shape (3,)

        for I in range(3):  # Loop over nodes
            for k in range(3):  # Loop over edges
                i, j = self.edge_nodes[k]

                # Check if node I is connected to this edge
                if I not in (i, j):
                    continue

                # Other node on this edge
                J = j if I == i else i

                # Unit vector from I to J
                e_IJ = self.e_hat[k] if I == i else -self.e_hat[k]

                # Force magnitude from edge-edge coupling
                # F_mag = Σ_m A[k,m] * g[k] * g[m] * eps_edge[m]
                mag = 0.0
                for m in range(3):
                    mag += self.A[k, m] * g[k] * g[m] * eps_edge[m]

                # Internal force contribution (negative: opposes deformation)
                F[I] -= self.area * self.thickness * mag * e_IJ / self.L[k]

        return F

    def compute_internal_force_standard(self, u: np.ndarray,
                                         d: np.ndarray) -> np.ndarray:
        """
        Compute internal force using standard FEM approach with damage.

        F_int = B^T σ^d × A × h

        where σ^d is the degraded stress.

        Args:
            u: nodal displacements
            d: damage on edges

        Returns:
            F: shape (6,), internal force vector
        """
        eps_tensor = self.compute_tensor_strain(u)
        eps_edge = self.T @ eps_tensor

        # Get split
        from physics.tension_split import spectral_split
        split = spectral_split(eps_tensor, eps_edge, self.C, self.T)

        # Degradation
        Phi = np.diag(1 - d)

        # Degraded stress in edge space
        sigma_edge_plus = self.A @ Phi @ split['eps_edge_plus']
        sigma_edge_minus = self.A @ split['eps_edge_minus']

        # Total degraded stress (in edge space, then transform)
        sigma_edge_d = Phi @ sigma_edge_plus + sigma_edge_minus

        # Transform to tensor space
        sigma_tensor_d = self.T.T @ sigma_edge_d

        # Internal force
        return self.B.T @ sigma_tensor_d * self.area * self.thickness

    def get_edge_info(self) -> Dict:
        """
        Return dictionary with edge information.

        Useful for debugging and visualization.
        """
        return {
            'nodes': self.edge_nodes,
            'lengths': self.L.copy(),
            'angles': self.phi.copy(),
            'unit_vectors': self.e_hat.copy(),
        }

    def verify_energy_equivalence(self, u: np.ndarray,
                                   tol: float = 1e-10) -> Tuple[bool, float, float]:
        """
        Verify that tensor and edge energy formulations are equivalent.

        Args:
            u: nodal displacements
            tol: relative tolerance

        Returns:
            (match, psi_tensor, psi_edge): whether they match and the values
        """
        psi_tensor = self.strain_energy_density_tensor(u)
        eps_edge = self.compute_edge_strains(u)
        psi_edge = self.strain_energy_density(eps_edge)

        if abs(psi_tensor) < 1e-15:
            match = abs(psi_edge) < 1e-15
        else:
            match = abs(psi_tensor - psi_edge) / abs(psi_tensor) < tol

        return match, psi_tensor, psi_edge
