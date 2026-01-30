"""
Mesh Generators
===============

Simple mesh generators for testing and benchmarks.
"""

import numpy as np
from typing import Optional, Tuple
from .triangle_mesh import TriangleMesh


def create_rectangle_mesh(Lx: float, Ly: float, nx: int, ny: int,
                          pattern: str = 'right',
                          thickness: float = 1.0) -> TriangleMesh:
    """
    Create structured triangular mesh on rectangle [0, Lx] × [0, Ly].

    Args:
        Lx, Ly: domain dimensions
        nx, ny: number of divisions in x and y
        pattern: diagonal pattern
            'right': diagonals go from lower-left to upper-right
            'left': diagonals go from lower-right to upper-left
            'crossed': both diagonals (4 triangles per quad)
            'alternating': alternating diagonal direction
        thickness: element thickness

    Returns:
        TriangleMesh instance
    """
    # Create nodes
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y

    nodes = np.zeros((n_nodes, 2))
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            idx = j * n_nodes_x + i
            nodes[idx, 0] = i * Lx / nx
            nodes[idx, 1] = j * Ly / ny

    # Create elements
    elements = []

    def node_idx(i, j):
        return j * n_nodes_x + i

    if pattern == 'right':
        for j in range(ny):
            for i in range(nx):
                # Lower-left triangle
                n0 = node_idx(i, j)
                n1 = node_idx(i + 1, j)
                n2 = node_idx(i + 1, j + 1)
                elements.append([n0, n1, n2])

                # Upper-right triangle
                n0 = node_idx(i, j)
                n1 = node_idx(i + 1, j + 1)
                n2 = node_idx(i, j + 1)
                elements.append([n0, n1, n2])

    elif pattern == 'left':
        for j in range(ny):
            for i in range(nx):
                # Lower-right triangle
                n0 = node_idx(i, j)
                n1 = node_idx(i + 1, j)
                n2 = node_idx(i, j + 1)
                elements.append([n0, n1, n2])

                # Upper-left triangle
                n0 = node_idx(i + 1, j)
                n1 = node_idx(i + 1, j + 1)
                n2 = node_idx(i, j + 1)
                elements.append([n0, n1, n2])

    elif pattern == 'alternating':
        for j in range(ny):
            for i in range(nx):
                if (i + j) % 2 == 0:
                    # Right diagonal
                    elements.append([node_idx(i, j), node_idx(i + 1, j),
                                     node_idx(i + 1, j + 1)])
                    elements.append([node_idx(i, j), node_idx(i + 1, j + 1),
                                     node_idx(i, j + 1)])
                else:
                    # Left diagonal
                    elements.append([node_idx(i, j), node_idx(i + 1, j),
                                     node_idx(i, j + 1)])
                    elements.append([node_idx(i + 1, j), node_idx(i + 1, j + 1),
                                     node_idx(i, j + 1)])

    elif pattern == 'crossed':
        # Add center node for each quad
        center_nodes = []
        for j in range(ny):
            for i in range(nx):
                cx = (i + 0.5) * Lx / nx
                cy = (j + 0.5) * Ly / ny
                center_nodes.append([cx, cy])

        nodes = np.vstack([nodes, np.array(center_nodes)])

        for j in range(ny):
            for i in range(nx):
                center_idx = n_nodes + j * nx + i
                n00 = node_idx(i, j)
                n10 = node_idx(i + 1, j)
                n11 = node_idx(i + 1, j + 1)
                n01 = node_idx(i, j + 1)

                elements.append([n00, n10, center_idx])
                elements.append([n10, n11, center_idx])
                elements.append([n11, n01, center_idx])
                elements.append([n01, n00, center_idx])

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return TriangleMesh(nodes, np.array(elements), thickness=thickness)


def create_notched_rectangle(L: float, H: float, a: float, h: float,
                             h_fine: Optional[float] = None,
                             notch_height: Optional[float] = None,
                             thickness: float = 1.0) -> TriangleMesh:
    """
    Create mesh for single-edge notched specimen (SENT).

    The notch extends from the left edge toward the center.

    Args:
        L: specimen width
        H: specimen height
        a: notch length (typically L/2)
        h: characteristic element size
        h_fine: element size near notch tip (default: h/2)
        notch_height: y-coordinate of notch (default: H/2)
        thickness: element thickness

    Returns:
        TriangleMesh instance
    """
    if h_fine is None:
        h_fine = h / 2
    if notch_height is None:
        notch_height = H / 2

    # Compute approximate number of divisions
    nx = max(int(np.ceil(L / h)), 4)
    ny = max(int(np.ceil(H / h)), 4)

    # Create a finer mesh that we'll then modify to include the notch
    # For simplicity, use a structured approach with local refinement

    # First, create basic structured mesh
    mesh = create_rectangle_mesh(L, H, nx, ny, pattern='right',
                                 thickness=thickness)

    # Find edges that are along the notch line
    # (This is a simplified approach - a production code would use
    # proper mesh generation with refinement)

    # For SENT, we identify edges on the notch and mark them for damage
    # The mesh itself remains structured, but we'll identify crack edges
    notch_edges = []
    tol = h / 4

    for edge_idx, (n1, n2) in enumerate(mesh.edges):
        p1, p2 = mesh.nodes[n1], mesh.nodes[n2]
        mid = 0.5 * (p1 + p2)

        # Check if edge is on notch line
        if (abs(mid[1] - notch_height) < tol and
                mid[0] < a + tol and
                p1[0] < a + tol and p2[0] < a + tol):
            # Check if edge is roughly horizontal
            if abs(p1[1] - p2[1]) < tol:
                notch_edges.append(edge_idx)

    # Store notch edge info as attribute
    mesh.notch_edges = np.array(notch_edges, dtype=np.int64)

    return mesh


def create_square_mesh(L: float, n: int, pattern: str = 'right',
                       thickness: float = 1.0) -> TriangleMesh:
    """
    Create structured triangular mesh on square [0, L] × [0, L].

    Convenience function that calls create_rectangle_mesh.

    Args:
        L: side length
        n: number of divisions per side
        pattern: diagonal pattern
        thickness: element thickness

    Returns:
        TriangleMesh instance
    """
    return create_rectangle_mesh(L, L, n, n, pattern, thickness)


def create_single_element(node_coords: Optional[np.ndarray] = None,
                          thickness: float = 1.0) -> TriangleMesh:
    """
    Create mesh with a single triangle element.

    Useful for unit testing.

    Args:
        node_coords: shape (3, 2), node coordinates
            Default: right triangle with vertices at (0,0), (1,0), (0,1)
        thickness: element thickness

    Returns:
        TriangleMesh instance
    """
    if node_coords is None:
        node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    elements = np.array([[0, 1, 2]])
    return TriangleMesh(node_coords, elements, thickness=thickness)


def create_two_element_patch(thickness: float = 1.0) -> TriangleMesh:
    """
    Create mesh with two triangles sharing an edge.

    Useful for testing edge connectivity and patch tests.

    Returns:
        TriangleMesh instance
    """
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    return TriangleMesh(nodes, elements, thickness=thickness)


def perturb_interior_nodes(mesh: TriangleMesh, magnitude: float = 0.1,
                           seed: Optional[int] = None) -> TriangleMesh:
    """
    Randomly perturb interior nodes for patch test verification.

    Args:
        mesh: input mesh
        magnitude: perturbation magnitude as fraction of min edge length
        seed: random seed for reproducibility

    Returns:
        New TriangleMesh with perturbed nodes
    """
    if seed is not None:
        np.random.seed(seed)

    # Get minimum edge length
    min_length = np.min(mesh.edge_lengths)
    pert = magnitude * min_length

    # Copy nodes
    new_nodes = mesh.nodes.copy()

    # Perturb interior nodes
    boundary_set = set(mesh.boundary_nodes)
    for i in range(mesh.n_nodes):
        if i not in boundary_set:
            new_nodes[i] += np.random.uniform(-pert, pert, 2)

    return TriangleMesh(new_nodes, mesh.elements.copy(),
                        thickness=mesh.thickness)


def refine_mesh(mesh: TriangleMesh) -> TriangleMesh:
    """
    Uniformly refine mesh by splitting each triangle into 4.

    Each edge is split at its midpoint, creating 4 sub-triangles
    per original triangle.

    Args:
        mesh: input mesh

    Returns:
        Refined TriangleMesh
    """
    # New nodes = original nodes + edge midpoints
    new_nodes = list(mesh.nodes)
    edge_midpoint_idx = {}

    for edge_idx, (n1, n2) in enumerate(mesh.edges):
        mid = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])
        new_idx = len(new_nodes)
        new_nodes.append(mid)
        edge_midpoint_idx[edge_idx] = new_idx

    new_nodes = np.array(new_nodes)

    # Create new elements
    new_elements = []
    for elem_idx, elem_nodes in enumerate(mesh.elements):
        n0, n1, n2 = elem_nodes
        e0, e1, e2 = mesh.element_to_edges[elem_idx]

        # Midpoint indices
        m0 = edge_midpoint_idx[e0]  # midpoint of edge opposite n0
        m1 = edge_midpoint_idx[e1]  # midpoint of edge opposite n1
        m2 = edge_midpoint_idx[e2]  # midpoint of edge opposite n2

        # Create 4 sub-triangles
        new_elements.append([n0, m2, m1])
        new_elements.append([n1, m0, m2])
        new_elements.append([n2, m1, m0])
        new_elements.append([m0, m1, m2])

    return TriangleMesh(new_nodes, np.array(new_elements),
                        thickness=mesh.thickness)
