"""
Boundary Conditions
==================

Application of Dirichlet and Neumann boundary conditions.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh


def apply_dirichlet_bc(K: csr_matrix, F: np.ndarray,
                       bc_dofs: np.ndarray, bc_values: np.ndarray,
                       method: str = 'elimination') -> Tuple[csr_matrix, np.ndarray]:
    """
    Apply Dirichlet boundary conditions to the system.

    Args:
        K: stiffness matrix, shape (n_dof, n_dof)
        F: force vector, shape (n_dof,)
        bc_dofs: indices of constrained DOFs
        bc_values: prescribed values at bc_dofs
        method: 'elimination' or 'penalty'

    Returns:
        K_bc, F_bc: modified system with BCs applied
    """
    if len(bc_dofs) != len(bc_values):
        raise ValueError("bc_dofs and bc_values must have same length")

    if method == 'elimination':
        return _apply_bc_elimination(K, F, bc_dofs, bc_values)
    elif method == 'penalty':
        return _apply_bc_penalty(K, F, bc_dofs, bc_values)
    else:
        raise ValueError(f"Unknown method: {method}")


def _apply_bc_elimination(K: csr_matrix, F: np.ndarray,
                          bc_dofs: np.ndarray, bc_values: np.ndarray
                          ) -> Tuple[csr_matrix, np.ndarray]:
    """
    Apply BCs by row/column elimination.

    More accurate than penalty method. The system is modified so that:
    - For each constrained DOF i: K[i,i] = 1, K[i,j] = 0, F[i] = bc_value
    - The RHS is adjusted to account for known values

    Args:
        K: stiffness matrix
        F: force vector
        bc_dofs: constrained DOF indices
        bc_values: prescribed values

    Returns:
        K_bc, F_bc: modified system
    """
    K = K.tolil()  # Convert to LIL for efficient row/column modifications
    F = F.copy()

    bc_dofs = np.asarray(bc_dofs, dtype=np.int64)
    bc_values = np.asarray(bc_values, dtype=np.float64)

    # Modify RHS for non-zero prescribed values
    # F_free -= K_free,constrained @ u_constrained
    for dof, val in zip(bc_dofs, bc_values):
        if val != 0:
            # Subtract contribution of known displacement
            F -= K[:, dof].toarray().flatten() * val

    # Set rows and columns to identity for constrained DOFs
    for dof, val in zip(bc_dofs, bc_values):
        K[dof, :] = 0  # Clear row
        K[:, dof] = 0  # Clear column
        K[dof, dof] = 1  # Set diagonal to 1
        F[dof] = val  # Set RHS to prescribed value

    return K.tocsr(), F


def _apply_bc_penalty(K: csr_matrix, F: np.ndarray,
                      bc_dofs: np.ndarray, bc_values: np.ndarray,
                      penalty: float = 1e20) -> Tuple[csr_matrix, np.ndarray]:
    """
    Apply BCs using penalty method.

    Adds large penalty to diagonal for constrained DOFs:
        K[i,i] += penalty
        F[i] += penalty * bc_value

    Less accurate than elimination but simpler.

    Args:
        K: stiffness matrix
        F: force vector
        bc_dofs: constrained DOF indices
        bc_values: prescribed values
        penalty: penalty coefficient

    Returns:
        K_bc, F_bc: modified system
    """
    K = K.tolil()
    F = F.copy()

    for dof, val in zip(bc_dofs, bc_values):
        K[dof, dof] += penalty
        F[dof] += penalty * val

    return K.tocsr(), F


def create_bc_from_region(mesh: 'TriangleMesh',
                          region_func: Callable[[float, float], bool],
                          component: str,
                          value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary conditions from a region function.

    Args:
        mesh: TriangleMesh instance
        region_func: function(x, y) -> bool, returns True for nodes in BC region
        component: 'x', 'y', or 'both'
        value: prescribed value

    Returns:
        bc_dofs: DOF indices
        bc_values: prescribed values
    """
    bc_dofs = []
    bc_values = []

    for i, (x, y) in enumerate(mesh.nodes):
        if region_func(x, y):
            if component in ('x', 'both'):
                bc_dofs.append(2 * i)
                bc_values.append(value)
            if component in ('y', 'both'):
                bc_dofs.append(2 * i + 1)
                bc_values.append(value)

    return np.array(bc_dofs, dtype=np.int64), np.array(bc_values, dtype=np.float64)


def create_fixed_bc(mesh: 'TriangleMesh',
                    node_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fully fixed (u_x = u_y = 0) boundary conditions.

    Args:
        mesh: TriangleMesh instance
        node_indices: indices of nodes to fix

    Returns:
        bc_dofs: DOF indices
        bc_values: zeros
    """
    bc_dofs = []
    for n in node_indices:
        bc_dofs.extend([2 * n, 2 * n + 1])

    return np.array(bc_dofs, dtype=np.int64), np.zeros(len(bc_dofs))


def create_roller_bc(mesh: 'TriangleMesh',
                     node_indices: np.ndarray,
                     direction: str = 'y') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create roller boundary conditions (free to slide in one direction).

    Args:
        mesh: TriangleMesh instance
        node_indices: indices of nodes with roller BC
        direction: 'x' (free in x, fixed in y) or 'y' (free in y, fixed in x)

    Returns:
        bc_dofs: DOF indices
        bc_values: zeros
    """
    bc_dofs = []

    if direction == 'y':
        # Free in y, fixed in x
        for n in node_indices:
            bc_dofs.append(2 * n)  # Fix x
    elif direction == 'x':
        # Free in x, fixed in y
        for n in node_indices:
            bc_dofs.append(2 * n + 1)  # Fix y
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return np.array(bc_dofs, dtype=np.int64), np.zeros(len(bc_dofs))


def create_prescribed_displacement_bc(mesh: 'TriangleMesh',
                                       node_indices: np.ndarray,
                                       component: str,
                                       value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prescribed displacement boundary conditions.

    Args:
        mesh: TriangleMesh instance
        node_indices: indices of nodes with prescribed displacement
        component: 'x' or 'y'
        value: prescribed displacement value

    Returns:
        bc_dofs: DOF indices
        bc_values: prescribed values
    """
    bc_dofs = []
    bc_values = []

    for n in node_indices:
        if component == 'x':
            bc_dofs.append(2 * n)
        elif component == 'y':
            bc_dofs.append(2 * n + 1)
        else:
            raise ValueError(f"Unknown component: {component}")
        bc_values.append(value)

    return np.array(bc_dofs, dtype=np.int64), np.array(bc_values, dtype=np.float64)


def merge_bcs(*bc_pairs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple boundary condition specifications.

    Args:
        *bc_pairs: tuples of (bc_dofs, bc_values)

    Returns:
        merged_dofs: combined DOF indices
        merged_values: combined values
    """
    all_dofs = []
    all_values = []

    for dofs, values in bc_pairs:
        all_dofs.extend(dofs)
        all_values.extend(values)

    # Remove duplicates (keep last value for each DOF)
    dof_to_value = {}
    for dof, val in zip(all_dofs, all_values):
        dof_to_value[dof] = val

    merged_dofs = np.array(sorted(dof_to_value.keys()), dtype=np.int64)
    merged_values = np.array([dof_to_value[d] for d in merged_dofs], dtype=np.float64)

    return merged_dofs, merged_values


def apply_point_load(F: np.ndarray,
                     node_idx: int,
                     fx: float = 0.0,
                     fy: float = 0.0) -> np.ndarray:
    """
    Apply point load to force vector.

    Args:
        F: force vector, shape (n_dof,)
        node_idx: node index where load is applied
        fx: force in x direction
        fy: force in y direction

    Returns:
        F: modified force vector
    """
    F = F.copy()
    F[2 * node_idx] += fx
    F[2 * node_idx + 1] += fy
    return F


def apply_distributed_load(mesh: 'TriangleMesh',
                           F: np.ndarray,
                           edge_indices: np.ndarray,
                           traction: np.ndarray) -> np.ndarray:
    """
    Apply distributed load (traction) on edges.

    The traction is integrated along the edge and distributed
    to the edge's nodes.

    Args:
        mesh: TriangleMesh instance
        F: force vector, shape (n_dof,)
        edge_indices: indices of edges with traction
        traction: shape (2,), traction vector [t_x, t_y] per unit length

    Returns:
        F: modified force vector
    """
    F = F.copy()
    traction = np.asarray(traction)

    for edge_idx in edge_indices:
        n1, n2 = mesh.edges[edge_idx]
        L = mesh.edge_lengths[edge_idx]

        # Total force on edge = traction × length × thickness
        total_force = traction * L * mesh.thickness

        # Distribute equally to both nodes
        F[2 * n1:2 * n1 + 2] += total_force / 2
        F[2 * n2:2 * n2 + 2] += total_force / 2

    return F


def get_free_dofs(n_dof: int, bc_dofs: np.ndarray) -> np.ndarray:
    """
    Get indices of free (unconstrained) DOFs.

    Args:
        n_dof: total number of DOFs
        bc_dofs: constrained DOF indices

    Returns:
        free_dofs: indices of free DOFs
    """
    all_dofs = set(range(n_dof))
    constrained = set(bc_dofs)
    return np.array(sorted(all_dofs - constrained), dtype=np.int64)


class BoundaryConditionManager:
    """
    Manager for handling boundary conditions in a simulation.

    Provides convenient interface for defining and applying BCs.
    """

    def __init__(self, mesh: 'TriangleMesh'):
        """
        Initialize BC manager.

        Args:
            mesh: TriangleMesh instance
        """
        self.mesh = mesh
        self.n_dof = 2 * mesh.n_nodes
        self.bc_dofs = np.array([], dtype=np.int64)
        self.bc_values = np.array([], dtype=np.float64)
        self._bc_sources = []  # Track where BCs come from

    def fix_region(self, region_func: Callable[[float, float], bool],
                   component: str = 'both', name: str = None) -> None:
        """
        Fix nodes in a region.

        Args:
            region_func: function(x, y) -> bool
            component: 'x', 'y', or 'both'
            name: optional name for this BC (for debugging)
        """
        dofs, values = create_bc_from_region(self.mesh, region_func, component, 0.0)
        self._add_bc(dofs, values, name or "fix_region")

    def prescribe_displacement(self, region_func: Callable[[float, float], bool],
                                component: str, value: float,
                                name: str = None) -> None:
        """
        Prescribe displacement in a region.

        Args:
            region_func: function(x, y) -> bool
            component: 'x' or 'y'
            value: prescribed displacement
            name: optional name
        """
        dofs, values = create_bc_from_region(self.mesh, region_func, component, value)
        self._add_bc(dofs, values, name or "prescribed_disp")

    def fix_nodes(self, node_indices: np.ndarray, name: str = None) -> None:
        """
        Fully fix specific nodes.

        Args:
            node_indices: indices of nodes to fix
            name: optional name
        """
        dofs, values = create_fixed_bc(self.mesh, node_indices)
        self._add_bc(dofs, values, name or "fix_nodes")

    def _add_bc(self, dofs: np.ndarray, values: np.ndarray, source: str) -> None:
        """Internal method to add BCs."""
        self.bc_dofs, self.bc_values = merge_bcs(
            (self.bc_dofs, self.bc_values),
            (dofs, values)
        )
        self._bc_sources.append((source, len(dofs)))

    def apply(self, K: csr_matrix, F: np.ndarray,
              method: str = 'elimination') -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply all boundary conditions to the system.

        Args:
            K: stiffness matrix
            F: force vector
            method: 'elimination' or 'penalty'

        Returns:
            K_bc, F_bc: modified system
        """
        return apply_dirichlet_bc(K, F, self.bc_dofs, self.bc_values, method)

    def update_values(self, new_values: np.ndarray) -> None:
        """
        Update BC values (e.g., for load stepping).

        Args:
            new_values: new values for all constrained DOFs
        """
        if len(new_values) != len(self.bc_values):
            raise ValueError("new_values must have same length as existing BCs")
        self.bc_values = np.asarray(new_values, dtype=np.float64)

    def scale_values(self, factor: float) -> None:
        """
        Scale all BC values by a factor.

        Args:
            factor: scaling factor
        """
        self.bc_values *= factor

    def get_free_dofs(self) -> np.ndarray:
        """Get indices of unconstrained DOFs."""
        return get_free_dofs(self.n_dof, self.bc_dofs)

    def summary(self) -> str:
        """Return summary of boundary conditions."""
        lines = [f"Boundary Conditions Summary ({len(self.bc_dofs)} constrained DOFs):"]
        for source, count in self._bc_sources:
            lines.append(f"  - {source}: {count} DOFs")
        return "\n".join(lines)
