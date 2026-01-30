"""
Triangle Mesh with Edge-Based Data Structures
=============================================

Core mesh class for GraFEA phase-field framework.
"""

import numpy as np
from typing import Tuple, List, Optional


class TriangleMesh:
    """
    Triangle mesh with edge-based data structures for GraFEA.

    The mesh supports edge-based computations required for the phase-field
    fracture framework, where damage is defined on edges rather than at
    integration points.

    Attributes:
        nodes: np.ndarray, shape (n_nodes, 2)
            Node coordinates in reference configuration
        elements: np.ndarray, shape (n_elements, 3)
            Node indices for each element (counterclockwise)
        edges: np.ndarray, shape (n_edges, 2)
            Node indices for each unique edge
        element_to_edges: np.ndarray, shape (n_elements, 3)
            Edge indices for each element [edge_opp_node0, edge_opp_node1, edge_opp_node2]
        edge_to_elements: list of lists
            Elements containing each edge (1 or 2 elements)
        boundary_edges: np.ndarray
            Indices of edges on the boundary
        boundary_nodes: np.ndarray
            Indices of nodes on the boundary

    Edge Convention:
        For element with nodes (n0, n1, n2):
        - Edge 0: connects n1-n2 (opposite to n0)
        - Edge 1: connects n2-n0 (opposite to n1)
        - Edge 2: connects n0-n1 (opposite to n2)
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 thickness: float = 1.0):
        """
        Initialize mesh and compute all connectivity.

        Args:
            nodes: shape (n_nodes, 2), node coordinates
            elements: shape (n_elements, 3), node indices for each element
            thickness: element thickness for plane problems
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.thickness = thickness

        # Validate input
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n_nodes, 2)")
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError("elements must have shape (n_elements, 3)")

        # Build connectivity
        self._build_edge_connectivity()
        self._identify_boundary()
        self._compute_geometric_quantities()

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the mesh."""
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """Number of elements in the mesh."""
        return len(self.elements)

    @property
    def n_edges(self) -> int:
        """Number of unique edges in the mesh."""
        return len(self.edges)

    def _build_edge_connectivity(self) -> None:
        """
        Create edge list and element-edge mappings.

        Uses canonical ordering (smaller node index first) to avoid duplicates.
        """
        # Edge node pairs for local edges (opposite to each node)
        local_edge_nodes = [(1, 2), (2, 0), (0, 1)]

        # Dictionary to map edge tuple to global edge index
        edge_dict = {}
        edges_list = []
        element_to_edges = np.zeros((self.n_elements, 3), dtype=np.int64)
        edge_to_elements = []

        for elem_idx, elem_nodes in enumerate(self.elements):
            for local_edge, (i, j) in enumerate(local_edge_nodes):
                n1, n2 = elem_nodes[i], elem_nodes[j]
                # Canonical ordering
                edge_key = (min(n1, n2), max(n1, n2))

                if edge_key not in edge_dict:
                    # New edge
                    global_edge_idx = len(edges_list)
                    edge_dict[edge_key] = global_edge_idx
                    edges_list.append(edge_key)
                    edge_to_elements.append([elem_idx])
                else:
                    # Existing edge - add this element
                    global_edge_idx = edge_dict[edge_key]
                    edge_to_elements[global_edge_idx].append(elem_idx)

                element_to_edges[elem_idx, local_edge] = edge_dict[edge_key]

        self.edges = np.array(edges_list, dtype=np.int64)
        self.element_to_edges = element_to_edges
        self.edge_to_elements = edge_to_elements

        # Also store the edge orientation for each element
        # (whether the canonical edge direction matches element orientation)
        self._compute_edge_orientations()

    def _compute_edge_orientations(self) -> None:
        """
        Compute edge orientations relative to element local numbering.

        edge_orientations[elem, local_edge] = +1 if edge direction matches
        element convention, -1 otherwise.
        """
        local_edge_nodes = [(1, 2), (2, 0), (0, 1)]
        self.edge_orientations = np.zeros((self.n_elements, 3), dtype=np.int64)

        for elem_idx, elem_nodes in enumerate(self.elements):
            for local_edge, (i, j) in enumerate(local_edge_nodes):
                n1, n2 = elem_nodes[i], elem_nodes[j]
                # Canonical is (min, max), element order is (n1, n2)
                if n1 < n2:
                    self.edge_orientations[elem_idx, local_edge] = 1
                else:
                    self.edge_orientations[elem_idx, local_edge] = -1

    def _identify_boundary(self) -> None:
        """
        Find boundary edges and nodes.

        An edge is on the boundary if it belongs to exactly one element.
        """
        # Boundary edges: those with only one adjacent element
        boundary_edge_mask = np.array([len(elems) == 1
                                       for elems in self.edge_to_elements])
        self.boundary_edges = np.where(boundary_edge_mask)[0]

        # Boundary nodes: nodes that belong to boundary edges
        boundary_node_set = set()
        for edge_idx in self.boundary_edges:
            boundary_node_set.add(self.edges[edge_idx, 0])
            boundary_node_set.add(self.edges[edge_idx, 1])
        self.boundary_nodes = np.array(sorted(boundary_node_set), dtype=np.int64)

    def _compute_geometric_quantities(self) -> None:
        """Compute areas and edge lengths."""
        self.element_areas = self.compute_element_areas()
        self.edge_lengths = self.compute_edge_lengths()
        self.edge_midpoints = self.compute_edge_midpoints()

    def compute_element_areas(self) -> np.ndarray:
        """
        Compute area of all elements.

        Uses the cross product formula:
        A = 0.5 * |det([x1-x0, y1-y0; x2-x0, y2-y0])|

        Returns:
            areas: shape (n_elements,)
        """
        areas = np.zeros(self.n_elements)
        for i, elem_nodes in enumerate(self.elements):
            X = self.nodes[elem_nodes]
            areas[i] = 0.5 * abs(
                (X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) -
                (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1])
            )
        return areas

    def compute_edge_lengths(self) -> np.ndarray:
        """
        Compute length of all edges.

        Returns:
            lengths: shape (n_edges,)
        """
        lengths = np.zeros(self.n_edges)
        for i, (n1, n2) in enumerate(self.edges):
            lengths[i] = np.linalg.norm(self.nodes[n2] - self.nodes[n1])
        return lengths

    def compute_edge_midpoints(self) -> np.ndarray:
        """
        Compute midpoint of all edges.

        Returns:
            midpoints: shape (n_edges, 2)
        """
        midpoints = np.zeros((self.n_edges, 2))
        for i, (n1, n2) in enumerate(self.edges):
            midpoints[i] = 0.5 * (self.nodes[n1] + self.nodes[n2])
        return midpoints

    def get_element_nodes(self, elem_idx: int) -> np.ndarray:
        """
        Return coordinates of element nodes.

        Args:
            elem_idx: element index

        Returns:
            coordinates: shape (3, 2)
        """
        return self.nodes[self.elements[elem_idx]]

    def get_edge_nodes(self, edge_idx: int) -> np.ndarray:
        """
        Return coordinates of edge endpoints.

        Args:
            edge_idx: edge index

        Returns:
            coordinates: shape (2, 2)
        """
        return self.nodes[self.edges[edge_idx]]

    def get_element_edges(self, elem_idx: int) -> np.ndarray:
        """
        Return edge indices for element.

        Args:
            elem_idx: element index

        Returns:
            edge_indices: shape (3,)
        """
        return self.element_to_edges[elem_idx]

    def get_edge_elements(self, edge_idx: int) -> List[int]:
        """
        Return element indices containing edge.

        Args:
            edge_idx: edge index

        Returns:
            element_indices: list of 1 or 2 element indices
        """
        return self.edge_to_elements[edge_idx]

    def get_element_edge_nodes_local(self, elem_idx: int) -> List[Tuple[int, int]]:
        """
        Get local node indices for each edge of an element.

        Returns:
            List of (i, j) tuples where edge k connects local nodes i and j.
        """
        # Edge k is opposite to node k
        return [(1, 2), (2, 0), (0, 1)]

    def get_edge_vector(self, edge_idx: int) -> np.ndarray:
        """
        Get the vector along an edge (from first to second node).

        Args:
            edge_idx: edge index

        Returns:
            vector: shape (2,)
        """
        n1, n2 = self.edges[edge_idx]
        return self.nodes[n2] - self.nodes[n1]

    def get_edge_unit_vector(self, edge_idx: int) -> np.ndarray:
        """
        Get unit vector along an edge.

        Args:
            edge_idx: edge index

        Returns:
            unit_vector: shape (2,)
        """
        vec = self.get_edge_vector(edge_idx)
        return vec / np.linalg.norm(vec)

    def is_boundary_edge(self, edge_idx: int) -> bool:
        """Check if edge is on the boundary."""
        return edge_idx in self.boundary_edges

    def is_boundary_node(self, node_idx: int) -> bool:
        """Check if node is on the boundary."""
        return node_idx in self.boundary_nodes

    def get_nodes_in_region(self, region_func) -> np.ndarray:
        """
        Get indices of nodes satisfying a condition.

        Args:
            region_func: function(x, y) -> bool

        Returns:
            node_indices: array of node indices
        """
        indices = []
        for i, (x, y) in enumerate(self.nodes):
            if region_func(x, y):
                indices.append(i)
        return np.array(indices, dtype=np.int64)

    def get_edges_in_region(self, region_func) -> np.ndarray:
        """
        Get indices of edges whose midpoints satisfy a condition.

        Args:
            region_func: function(x, y) -> bool

        Returns:
            edge_indices: array of edge indices
        """
        indices = []
        for i, (x, y) in enumerate(self.edge_midpoints):
            if region_func(x, y):
                indices.append(i)
        return np.array(indices, dtype=np.int64)
