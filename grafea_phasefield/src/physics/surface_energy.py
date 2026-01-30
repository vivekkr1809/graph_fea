"""
Surface Energy
==============

Fracture surface energy computation using graph Laplacian regularization.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh
    from mesh.edge_graph import EdgeGraph


def compute_edge_volumes(mesh: 'TriangleMesh') -> np.ndarray:
    """
    Compute volume associated with each edge.

    Each edge gets a portion of the volume of elements it belongs to.
    For a triangle with 3 edges, each edge gets 1/3 of the element volume.

    From derivation Eq. (6.2):
        ω_i = h × Σ_{e ∋ i} A^e / 3

    Args:
        mesh: TriangleMesh instance

    Returns:
        omega: edge volumes, shape (n_edges,)
    """
    omega = np.zeros(mesh.n_edges)

    for e_idx in range(mesh.n_elements):
        elem_volume = mesh.element_areas[e_idx] * mesh.thickness
        edge_indices = mesh.element_to_edges[e_idx]

        for k in edge_indices:
            omega[k] += elem_volume / 3

    return omega


def compute_surface_energy(mesh: 'TriangleMesh',
                           edge_graph: 'EdgeGraph',
                           d: np.ndarray,
                           Gc: float, l0: float) -> float:
    """
    Compute fracture surface energy.

    The surface energy consists of two terms:
    1. Local term: penalizes damage itself
    2. Gradient term: regularizes damage (prevents sharp jumps)

    From derivation Eq. (6.11):
        E_frac = Σ_i (Gc/2l0) d_i² ω_i + (Gc l0/4) Σ_i Σ_{j∈N(i)} w_ij (d_j - d_i)²

    Args:
        mesh: TriangleMesh instance
        edge_graph: EdgeGraph for neighbor information
        d: damage values, shape (n_edges,)
        Gc: critical energy release rate [J/m²]
        l0: phase-field length scale [m]

    Returns:
        E_frac: fracture surface energy [J]
    """
    omega = compute_edge_volumes(mesh)

    # Local term: (Gc / 2l0) Σ d² ω
    E_local = np.sum(Gc / (2 * l0) * d ** 2 * omega)

    # Gradient term: (Gc l0 / 4) Σ_i Σ_j w_ij (d_j - d_i)²
    # Factor 1/4 accounts for double counting (i,j) and (j,i)
    E_gradient = 0.0
    for i in range(mesh.n_edges):
        for j, w in zip(edge_graph.neighbors[i], edge_graph.neighbor_weights[i]):
            E_gradient += w * (d[j] - d[i]) ** 2

    # The sum counts each pair twice, so multiply by 1/2
    # Combined with factor for phase-field: Gc * l0 / 4
    E_gradient *= Gc * l0 / 4

    return E_local + E_gradient


def compute_surface_energy_derivative(mesh: 'TriangleMesh',
                                       edge_graph: 'EdgeGraph',
                                       d: np.ndarray,
                                       Gc: float, l0: float) -> np.ndarray:
    """
    Compute derivative of surface energy with respect to damage.

    ∂E_frac/∂d_i = (Gc/l0) d_i ω_i - (Gc l0) Σ_j w_ij (d_j - d_i)

    The second term is related to the graph Laplacian.

    Args:
        mesh: TriangleMesh instance
        edge_graph: EdgeGraph instance
        d: damage values, shape (n_edges,)
        Gc: critical energy release rate
        l0: phase-field length scale

    Returns:
        dE_dd: derivative, shape (n_edges,)
    """
    omega = compute_edge_volumes(mesh)

    # Local term derivative
    dE_dd = Gc / l0 * d * omega

    # Gradient term derivative (negative of Laplacian)
    laplacian_d = edge_graph.apply_laplacian(d)
    dE_dd -= Gc * l0 * laplacian_d * omega

    return dE_dd


def assemble_damage_system(mesh: 'TriangleMesh',
                           edge_graph: 'EdgeGraph',
                           history: np.ndarray,
                           Gc: float, l0: float) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assemble linear system for damage evolution.

    The damage minimization problem leads to:
        (Gc/l0 I - Gc l0 L_graph + 2 diag(H)) d = 2 diag(H) 1

    where L_graph is the graph Laplacian (with appropriate scaling).

    In terms of edge volumes ω_i, the system becomes:
        (Gc/l0 ω_i δ_ij + Gc l0 K_ij + 2 H_i ω_i δ_ij) d_j = 2 H_i ω_i

    where K_ij comes from the gradient term discretization.

    Args:
        mesh: TriangleMesh instance
        edge_graph: EdgeGraph instance
        history: history field values, shape (n_edges,)
        Gc: critical energy release rate
        l0: phase-field length scale

    Returns:
        A_d: system matrix, sparse CSR, shape (n_edges, n_edges)
        b_d: right-hand side vector, shape (n_edges,)
    """
    n = mesh.n_edges
    omega = compute_edge_volumes(mesh)

    # Build sparse matrix
    A_d = lil_matrix((n, n))
    b_d = np.zeros(n)

    for i in range(n):
        # Diagonal contributions
        # From local term: Gc/l0 * ω_i
        # From history term: 2 * H_i * ω_i
        A_d[i, i] = Gc / l0 * omega[i] + 2 * history[i] * omega[i]

        # Off-diagonal from graph Laplacian gradient term
        # The gradient energy is (Gc*l0/2) Σ w_ij (d_j - d_i)²
        # Taking derivative w.r.t. d_i gives Gc*l0 Σ w_ij (d_i - d_j)
        # This contributes to the matrix as:
        #   +Gc*l0*w_ij on diagonal (from all j)
        #   -Gc*l0*w_ij off-diagonal
        for j, w in zip(edge_graph.neighbors[i], edge_graph.neighbor_weights[i]):
            # Scale weight by average edge volume for dimensional consistency
            w_scaled = w * (omega[i] + omega[j]) / 2

            A_d[i, i] += Gc * l0 * w_scaled
            A_d[i, j] -= Gc * l0 * w_scaled

        # Right-hand side: 2 * H_i * ω_i
        b_d[i] = 2 * history[i] * omega[i]

    return A_d.tocsr(), b_d


def assemble_damage_system_simplified(mesh: 'TriangleMesh',
                                       edge_graph: 'EdgeGraph',
                                       history: np.ndarray,
                                       Gc: float, l0: float) -> Tuple[csr_matrix, np.ndarray]:
    """
    Simplified damage system assembly.

    Uses a simpler formulation where the damage equation is:
        (1/l0 + l0 Σ_j w_ij + 2H_i/Gc) d_i - l0 Σ_j w_ij d_j = 2H_i/Gc

    This formulation assumes uniform edge volumes and is simpler to implement.

    Args:
        mesh: TriangleMesh instance
        edge_graph: EdgeGraph instance
        history: history field values
        Gc: critical energy release rate
        l0: phase-field length scale

    Returns:
        A_d: system matrix, sparse CSR
        b_d: right-hand side vector
    """
    n = mesh.n_edges

    A_d = lil_matrix((n, n))
    b_d = np.zeros(n)

    for i in range(n):
        # Sum of weights
        w_sum = np.sum(edge_graph.neighbor_weights[i])

        # Diagonal
        A_d[i, i] = 1.0 / l0 + l0 * w_sum + 2 * history[i] / Gc

        # Off-diagonal
        for j, w in zip(edge_graph.neighbors[i], edge_graph.neighbor_weights[i]):
            A_d[i, j] = -l0 * w

        # RHS
        b_d[i] = 2 * history[i] / Gc

    return A_d.tocsr(), b_d


def compute_surface_energy_rate(mesh: 'TriangleMesh',
                                 d: np.ndarray,
                                 d_old: np.ndarray,
                                 edge_graph: 'EdgeGraph',
                                 Gc: float, l0: float,
                                 dt: float = 1.0) -> float:
    """
    Compute rate of surface energy change.

    Useful for energy balance checks.

    Args:
        mesh: TriangleMesh instance
        d: current damage
        d_old: previous damage
        edge_graph: EdgeGraph instance
        Gc: critical energy release rate
        l0: phase-field length scale
        dt: time step (or load step increment)

    Returns:
        dE_dt: rate of surface energy change
    """
    E_new = compute_surface_energy(mesh, edge_graph, d, Gc, l0)
    E_old = compute_surface_energy(mesh, edge_graph, d_old, Gc, l0)

    return (E_new - E_old) / dt


def estimate_l0_from_mesh(mesh: 'TriangleMesh', factor: float = 2.0) -> float:
    """
    Estimate appropriate l0 from mesh size.

    The phase-field length scale l0 should be resolved by the mesh.
    Common choice is l0 = 2h where h is the characteristic element size.

    Args:
        mesh: TriangleMesh instance
        factor: multiplier for average edge length

    Returns:
        l0: recommended length scale
    """
    h = np.mean(mesh.edge_lengths)
    return factor * h
