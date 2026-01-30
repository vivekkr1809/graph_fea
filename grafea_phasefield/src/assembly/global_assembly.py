"""
Global Assembly
===============

Assembly of global stiffness matrix and internal force vector.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh
    from elements.grafea_element import GraFEAElement


def assemble_global_stiffness(mesh: 'TriangleMesh',
                               elements: List['GraFEAElement'],
                               damage: np.ndarray) -> csr_matrix:
    """
    Assemble global stiffness matrix with edge-based damage.

    K_global = Σ_e K^e_damaged

    where K^e_damaged accounts for damage on the element's edges.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        damage: edge damage values, shape (n_edges,)

    Returns:
        K: sparse stiffness matrix, shape (2*n_nodes, 2*n_nodes)
    """
    n_dof = 2 * mesh.n_nodes
    K = lil_matrix((n_dof, n_dof))

    for e_idx, elem in enumerate(elements):
        # Get local damage values for this element
        edge_indices = mesh.element_to_edges[e_idx]
        d_local = damage[edge_indices]

        # Compute element stiffness with damage
        K_e = elem.compute_stiffness_damaged(d_local)

        # Get global DOF indices for this element
        # DOF ordering: [u_0x, u_0y, u_1x, u_1y, u_2x, u_2y]
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])
        dof_indices = np.array(dof_indices)

        # Scatter element matrix to global matrix
        for i_local, i_global in enumerate(dof_indices):
            for j_local, j_global in enumerate(dof_indices):
                K[i_global, j_global] += K_e[i_local, j_local]

    return K.tocsr()


def assemble_global_stiffness_undamaged(mesh: 'TriangleMesh',
                                         elements: List['GraFEAElement']) -> csr_matrix:
    """
    Assemble global stiffness matrix without damage.

    K_global = Σ_e K^e

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances

    Returns:
        K: sparse stiffness matrix, shape (2*n_nodes, 2*n_nodes)
    """
    return assemble_global_stiffness(mesh, elements, np.zeros(mesh.n_edges))


def assemble_internal_force(mesh: 'TriangleMesh',
                            elements: List['GraFEAElement'],
                            u: np.ndarray,
                            damage: np.ndarray) -> np.ndarray:
    """
    Assemble global internal force vector.

    F_int = Σ_e F^e_int

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector, shape (2*n_nodes,)
        damage: edge damage values, shape (n_edges,)

    Returns:
        F: internal force vector, shape (2*n_nodes,)
    """
    n_dof = 2 * mesh.n_nodes
    F = np.zeros(n_dof)

    u = np.asarray(u).flatten()

    for e_idx, elem in enumerate(elements):
        # Get nodal displacements for this element
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])
        dof_indices = np.array(dof_indices)

        u_e = u[dof_indices]

        # Compute edge strains
        eps_edge = elem.compute_edge_strains(u_e)

        # Get local damage
        edge_indices = mesh.element_to_edges[e_idx]
        d_local = damage[edge_indices]

        # Compute element internal force
        F_e = elem.compute_internal_force_standard(u_e.reshape(3, 2), d_local)

        # Scatter to global force vector
        for i_local, i_global in enumerate(dof_indices):
            F[i_global] += F_e[i_local]

    return F


def compute_all_edge_strains(mesh: 'TriangleMesh',
                             elements: List['GraFEAElement'],
                             u: np.ndarray) -> np.ndarray:
    """
    Compute edge strains for all elements.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector, shape (2*n_nodes,)

    Returns:
        edge_strains: shape (n_elements, 3), edge strains for each element
    """
    u = np.asarray(u).flatten()
    edge_strains = np.zeros((mesh.n_elements, 3))

    for e_idx, elem in enumerate(elements):
        # Get nodal displacements for this element
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])

        u_e = u[dof_indices]
        edge_strains[e_idx] = elem.compute_edge_strains(u_e)

    return edge_strains


def compute_all_tensor_strains(mesh: 'TriangleMesh',
                               elements: List['GraFEAElement'],
                               u: np.ndarray) -> np.ndarray:
    """
    Compute tensor strains for all elements.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector

    Returns:
        tensor_strains: shape (n_elements, 3), [ε_xx, ε_yy, γ_xy] for each element
    """
    u = np.asarray(u).flatten()
    tensor_strains = np.zeros((mesh.n_elements, 3))

    for e_idx, elem in enumerate(elements):
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])

        u_e = u[dof_indices]
        tensor_strains[e_idx] = elem.compute_tensor_strain(u_e)

    return tensor_strains


def compute_all_stresses(mesh: 'TriangleMesh',
                         elements: List['GraFEAElement'],
                         u: np.ndarray,
                         damage: np.ndarray) -> np.ndarray:
    """
    Compute stress tensors for all elements with damage.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector
        damage: edge damage values

    Returns:
        stresses: shape (n_elements, 3), [σ_xx, σ_yy, τ_xy] for each element
    """
    from physics.tension_split import spectral_split

    u = np.asarray(u).flatten()
    stresses = np.zeros((mesh.n_elements, 3))

    for e_idx, elem in enumerate(elements):
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])

        u_e = u[dof_indices]
        eps_tensor = elem.compute_tensor_strain(u_e)
        eps_edge = elem.T @ eps_tensor

        # Get local damage
        edge_indices = mesh.element_to_edges[e_idx]
        d_local = damage[edge_indices]

        # Get split
        split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

        # Degradation
        Phi = np.diag(1 - d_local)

        # Degraded stress in edge space
        sigma_edge_plus = elem.A @ Phi @ split['eps_edge_plus']
        sigma_edge_minus = elem.A @ split['eps_edge_minus']

        # Total stress (in edge space)
        sigma_edge_d = Phi @ sigma_edge_plus + sigma_edge_minus

        # Transform to tensor space
        stresses[e_idx] = elem.T.T @ sigma_edge_d

    return stresses


def compute_strain_energy(mesh: 'TriangleMesh',
                          elements: List['GraFEAElement'],
                          u: np.ndarray,
                          damage: np.ndarray) -> float:
    """
    Compute total strain energy.

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector
        damage: edge damage values

    Returns:
        E_strain: total strain energy [J]
    """
    from physics.tension_split import spectral_split

    u = np.asarray(u).flatten()
    total_energy = 0.0

    for e_idx, elem in enumerate(elements):
        node_indices = mesh.elements[e_idx]
        dof_indices = []
        for n in node_indices:
            dof_indices.extend([2 * n, 2 * n + 1])

        u_e = u[dof_indices]
        eps_edge = elem.compute_edge_strains(u_e)
        eps_tensor = elem.T_inv @ eps_edge

        # Get local damage
        edge_indices = mesh.element_to_edges[e_idx]
        d_local = damage[edge_indices]

        # Get split
        split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

        # Compute degraded energy
        psi = elem.strain_energy_density_degraded(eps_edge, d_local, split)

        # Add to total
        total_energy += psi * elem.area * elem.thickness

    return total_energy


def compute_reaction_forces(mesh: 'TriangleMesh',
                            elements: List['GraFEAElement'],
                            u: np.ndarray,
                            damage: np.ndarray,
                            bc_dofs: np.ndarray) -> np.ndarray:
    """
    Compute reaction forces at constrained DOFs.

    R = K @ u - F_ext = F_int

    Args:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        u: global displacement vector
        damage: edge damage values
        bc_dofs: indices of constrained DOFs

    Returns:
        reactions: reaction forces at bc_dofs
    """
    F_int = assemble_internal_force(mesh, elements, u, damage)
    return F_int[bc_dofs]


def compute_nodal_damage(mesh: 'TriangleMesh',
                          damage: np.ndarray) -> np.ndarray:
    """
    Map edge damage to nodes for visualization.

    Node damage is average of connected edge damages.

    Args:
        mesh: TriangleMesh instance
        damage: edge damage values

    Returns:
        nodal_damage: shape (n_nodes,)
    """
    nodal_damage = np.zeros(mesh.n_nodes)
    node_counts = np.zeros(mesh.n_nodes)

    for edge_idx, (n1, n2) in enumerate(mesh.edges):
        d_edge = damage[edge_idx]
        nodal_damage[n1] += d_edge
        nodal_damage[n2] += d_edge
        node_counts[n1] += 1
        node_counts[n2] += 1

    nonzero = node_counts > 0
    nodal_damage[nonzero] /= node_counts[nonzero]

    return nodal_damage


def compute_element_damage(mesh: 'TriangleMesh',
                            damage: np.ndarray) -> np.ndarray:
    """
    Map edge damage to elements for visualization.

    Element damage is average of its edge damages.

    Args:
        mesh: TriangleMesh instance
        damage: edge damage values

    Returns:
        element_damage: shape (n_elements,)
    """
    element_damage = np.zeros(mesh.n_elements)

    for e_idx in range(mesh.n_elements):
        edge_indices = mesh.element_to_edges[e_idx]
        element_damage[e_idx] = np.mean(damage[edge_indices])

    return element_damage
