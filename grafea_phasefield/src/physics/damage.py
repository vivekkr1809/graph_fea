"""
Damage Model
============

Damage evolution, degradation functions, and history field for phase-field fracture.
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh
    from elements.grafea_element import GraFEAElement


def degradation_function(d: np.ndarray, model: str = 'quadratic') -> np.ndarray:
    """
    Compute degradation function g(d).

    The degradation function reduces stiffness as damage increases:
        C_damaged = g(d) · C

    Common choices:
        - quadratic: g(d) = (1-d)²  (standard, used by Miehe et al.)
        - cubic: g(d) = (1-d)³

    Args:
        d: damage values in [0, 1], any shape
        model: 'quadratic' or 'cubic'

    Returns:
        g(d): degradation values, same shape as d
    """
    d = np.asarray(d)

    if model == 'quadratic':
        return (1 - d) ** 2
    elif model == 'cubic':
        return (1 - d) ** 3
    else:
        raise ValueError(f"Unknown degradation model: {model}")


def degradation_derivative(d: np.ndarray, model: str = 'quadratic') -> np.ndarray:
    """
    Compute derivative of degradation function g'(d).

    Used in damage evolution equations.

    For quadratic: g'(d) = -2(1-d)
    For cubic: g'(d) = -3(1-d)²

    Args:
        d: damage values in [0, 1]
        model: 'quadratic' or 'cubic'

    Returns:
        g'(d): derivative values
    """
    d = np.asarray(d)

    if model == 'quadratic':
        return -2 * (1 - d)
    elif model == 'cubic':
        return -3 * (1 - d) ** 2
    else:
        raise ValueError(f"Unknown degradation model: {model}")


class HistoryField:
    """
    History field for damage irreversibility.

    The history field enforces that damage can only grow (irreversibility):
        H_k^{n+1} = max(H_k^n, ψ_k^+)

    where ψ_k^+ is the tensile strain energy density.

    Attributes:
        H: history field values, shape (n_edges,)
        n_edges: number of edges
    """

    def __init__(self, n_edges: int):
        """
        Initialize history field to zero.

        Args:
            n_edges: number of edges in the mesh
        """
        self.n_edges = n_edges
        self.H = np.zeros(n_edges)

    def update(self, driving_force: np.ndarray) -> None:
        """
        Update history field: H = max(H, driving_force).

        Args:
            driving_force: new driving force values, shape (n_edges,)
        """
        if len(driving_force) != self.n_edges:
            raise ValueError(f"driving_force has wrong size: {len(driving_force)} != {self.n_edges}")
        self.H = np.maximum(self.H, driving_force)

    def reset(self) -> None:
        """Reset history field to zero."""
        self.H[:] = 0

    def get_values(self) -> np.ndarray:
        """Return copy of history field values."""
        return self.H.copy()

    def set_values(self, H: np.ndarray) -> None:
        """Set history field values directly."""
        if len(H) != self.n_edges:
            raise ValueError(f"H has wrong size: {len(H)} != {self.n_edges}")
        self.H = H.copy()


def compute_driving_force(mesh: 'TriangleMesh',
                          elements: List['GraFEAElement'],
                          edge_strains: np.ndarray,
                          material) -> np.ndarray:
    """
    Compute damage driving force for each edge.

    The driving force is the tensile strain energy contribution,
    averaged over elements sharing each edge.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        edge_strains: shape (n_elements, 3), edge strains for each element
        material: material properties (or list for heterogeneous)

    Returns:
        driving_force: shape (n_edges,), driving force for each edge
    """
    from .tension_split import spectral_split

    n_edges = mesh.n_edges
    driving_force = np.zeros(n_edges)
    edge_volumes = np.zeros(n_edges)  # For volume-weighted averaging

    for e_idx, elem in enumerate(elements):
        eps_edge = edge_strains[e_idx]

        # Get tension-compression split
        T = elem.T
        C = elem.C
        T_inv = elem.T_inv
        eps_tensor = T_inv @ eps_edge
        split = spectral_split(eps_tensor, eps_edge, C, T)

        # Element volume contribution per edge (divided by 3)
        elem_volume = elem.area * elem.thickness
        edge_volume_contrib = elem_volume / 3

        # Get global edge indices for this element
        edge_indices = mesh.element_to_edges[e_idx]

        for local_k, global_k in enumerate(edge_indices):
            # Driving force: tensile energy associated with this edge
            # Use the full tensile energy (psi_plus) for the driving force
            # This is a simplification; more sophisticated approaches
            # would decompose by edge contribution

            # For edge-based approach, we can use the diagonal term
            # of the tensile energy in edge space
            eps_k_plus = split['eps_edge_plus'][local_k]
            A_kk = elem.A[local_k, local_k]

            # Approximate edge contribution to tensile energy
            psi_k_plus = 0.5 * A_kk * eps_k_plus ** 2

            # Accumulate with volume weighting
            driving_force[global_k] += psi_k_plus * edge_volume_contrib
            edge_volumes[global_k] += edge_volume_contrib

    # Normalize by total edge volume
    nonzero = edge_volumes > 0
    driving_force[nonzero] /= edge_volumes[nonzero]

    return driving_force


def compute_driving_force_full(mesh: 'TriangleMesh',
                               elements: List['GraFEAElement'],
                               edge_strains: np.ndarray) -> np.ndarray:
    """
    Compute driving force using full tensile energy per element.

    Alternative approach that uses the total tensile energy
    distributed to edges based on their contribution.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        edge_strains: shape (n_elements, 3), edge strains for each element

    Returns:
        driving_force: shape (n_edges,)
    """
    from .tension_split import spectral_split

    n_edges = mesh.n_edges
    driving_force = np.zeros(n_edges)
    edge_counts = np.zeros(n_edges)

    for e_idx, elem in enumerate(elements):
        eps_edge = edge_strains[e_idx]

        # Get split
        eps_tensor = elem.T_inv @ eps_edge
        split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

        # Total tensile energy for this element
        psi_plus = split['psi_plus']

        # Distribute to edges (equally for simplicity)
        edge_indices = mesh.element_to_edges[e_idx]
        for global_k in edge_indices:
            driving_force[global_k] += psi_plus / 3
            edge_counts[global_k] += 1

    # Average over elements sharing each edge
    nonzero = edge_counts > 0
    driving_force[nonzero] /= edge_counts[nonzero]

    return driving_force


def compute_damage_energy_release_rate(d: np.ndarray,
                                        history: np.ndarray,
                                        Gc: float, l0: float,
                                        omega: np.ndarray) -> np.ndarray:
    """
    Compute local energy release rate for damage evolution.

    From the damage evolution equation:
        ∂ψ/∂d + Gc/l0 · d = 2H · (1-d)

    The energy release rate is:
        Y_k = -g'(d_k) · H_k = 2(1-d_k) · H_k

    Args:
        d: current damage, shape (n_edges,)
        history: history field values, shape (n_edges,)
        Gc: critical energy release rate
        l0: length scale
        omega: edge volumes, shape (n_edges,)

    Returns:
        Y: energy release rate, shape (n_edges,)
    """
    return 2 * (1 - d) * history


def enforce_damage_bounds(d: np.ndarray,
                           d_old: Optional[np.ndarray] = None,
                           d_min: float = 0.0,
                           d_max: float = 1.0) -> np.ndarray:
    """
    Enforce damage bounds and irreversibility.

    Args:
        d: new damage values
        d_old: previous damage values (for irreversibility)
        d_min: minimum damage (usually 0)
        d_max: maximum damage (usually 1)

    Returns:
        d_bounded: bounded damage values
    """
    d_bounded = np.clip(d, d_min, d_max)

    if d_old is not None:
        # Irreversibility: damage can only increase
        d_bounded = np.maximum(d_bounded, d_old)

    return d_bounded


def identify_fully_damaged_edges(d: np.ndarray,
                                  threshold: float = 0.99) -> np.ndarray:
    """
    Identify edges that are fully damaged (cracked).

    Args:
        d: damage values, shape (n_edges,)
        threshold: damage threshold for "fully damaged"

    Returns:
        cracked_edges: indices of fully damaged edges
    """
    return np.where(d >= threshold)[0]


def compute_crack_length(mesh: 'TriangleMesh',
                         d: np.ndarray,
                         threshold: float = 0.5) -> float:
    """
    Estimate crack length from damage field.

    Crack length is approximated as sum of edge lengths weighted by damage.

    Args:
        mesh: TriangleMesh instance
        d: damage values, shape (n_edges,)
        threshold: damage threshold to count as cracked

    Returns:
        crack_length: estimated crack length
    """
    cracked = d > threshold
    return np.sum(mesh.edge_lengths[cracked] * d[cracked])


def compute_dissipated_energy(d: np.ndarray,
                              d_old: np.ndarray,
                              omega: np.ndarray,
                              Gc: float, l0: float) -> float:
    """
    Compute energy dissipated due to damage growth.

    Dissipation = Gc · Σ_k (d_k - d_k_old) · ω_k / l0

    Args:
        d: new damage values
        d_old: previous damage values
        omega: edge volumes
        Gc: critical energy release rate
        l0: length scale

    Returns:
        dissipation: dissipated energy
    """
    d_increment = np.maximum(d - d_old, 0)  # Only count increases
    return Gc / l0 * np.sum(d_increment * omega)
