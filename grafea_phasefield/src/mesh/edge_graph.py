"""
Edge Graph for Damage Regularization
=====================================

Graph structure where mesh edges become nodes, used for
computing the graph Laplacian in damage regularization.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import List, Optional

from .triangle_mesh import TriangleMesh


class EdgeGraph:
    """
    Graph structure where edges of the mesh become nodes.

    Used for damage regularization via graph Laplacian. Two mesh edges
    are "neighbors" in the edge graph if they share a mesh vertex.

    Attributes:
        n_edges: int
            Number of edges (= nodes in edge graph)
        neighbors: list of np.ndarray
            neighbors[i] = array of edge indices sharing a vertex with edge i
        neighbor_distances: list of np.ndarray
            Distance between edge midpoints for weighted Laplacian
        neighbor_weights: list of np.ndarray
            Weights for graph Laplacian (various schemes)
    """

    def __init__(self, mesh: TriangleMesh, weight_scheme: str = 'distance'):
        """
        Build edge graph from mesh.

        Args:
            mesh: TriangleMesh instance
            weight_scheme: 'uniform', 'distance', or 'area'
        """
        self.mesh = mesh
        self.n_edges = mesh.n_edges
        self.weight_scheme = weight_scheme

        self._build_edge_neighbors()
        self._compute_neighbor_distances()
        self.compute_weights(weight_scheme)

    def _build_edge_neighbors(self) -> None:
        """
        Find neighboring edges for each edge.

        Two edges are neighbors if they share a mesh vertex.
        """
        # Build node-to-edges mapping
        node_to_edges = [[] for _ in range(self.mesh.n_nodes)]
        for edge_idx, (n1, n2) in enumerate(self.mesh.edges):
            node_to_edges[n1].append(edge_idx)
            node_to_edges[n2].append(edge_idx)

        # Find neighbors for each edge
        self.neighbors = []
        for edge_idx, (n1, n2) in enumerate(self.mesh.edges):
            neighbor_set = set()
            # Add edges connected to n1
            for other_edge in node_to_edges[n1]:
                if other_edge != edge_idx:
                    neighbor_set.add(other_edge)
            # Add edges connected to n2
            for other_edge in node_to_edges[n2]:
                if other_edge != edge_idx:
                    neighbor_set.add(other_edge)
            self.neighbors.append(np.array(sorted(neighbor_set), dtype=np.int64))

    def _compute_neighbor_distances(self) -> None:
        """
        Compute distance between edge midpoints for each neighbor pair.
        """
        midpoints = self.mesh.edge_midpoints
        self.neighbor_distances = []

        for i in range(self.n_edges):
            distances = np.zeros(len(self.neighbors[i]))
            for k, j in enumerate(self.neighbors[i]):
                distances[k] = np.linalg.norm(midpoints[j] - midpoints[i])
            self.neighbor_distances.append(distances)

    def compute_weights(self, scheme: str = 'distance') -> None:
        """
        Compute graph Laplacian weights.

        Args:
            scheme:
                'uniform': w_ij = 1/|N(i)|
                'distance': w_ij ∝ 1/δ_ij² (distance between midpoints)
                'area': w_ij ∝ shared element area (if applicable)

        The weights are normalized so that sum_j w_ij ≈ 1 for interior edges.
        """
        self.weight_scheme = scheme
        self.neighbor_weights = []

        if scheme == 'uniform':
            for i in range(self.n_edges):
                n_neighbors = len(self.neighbors[i])
                if n_neighbors > 0:
                    weights = np.ones(n_neighbors) / n_neighbors
                else:
                    weights = np.array([])
                self.neighbor_weights.append(weights)

        elif scheme == 'distance':
            for i in range(self.n_edges):
                distances = self.neighbor_distances[i]
                if len(distances) > 0:
                    # Weight inversely proportional to distance squared
                    weights = 1.0 / (distances ** 2 + 1e-10)
                    # Normalize
                    weights = weights / np.sum(weights)
                else:
                    weights = np.array([])
                self.neighbor_weights.append(weights)

        elif scheme == 'area':
            # Area-based weighting: weight by shared element area
            self._compute_area_weights()

        else:
            raise ValueError(f"Unknown weight scheme: {scheme}")

    def _compute_area_weights(self) -> None:
        """
        Compute area-based weights for graph Laplacian.

        Weight is proportional to the area of elements sharing both edges.
        """
        self.neighbor_weights = []

        for i in range(self.n_edges):
            weights = np.zeros(len(self.neighbors[i]))
            elems_i = set(self.mesh.edge_to_elements[i])

            for k, j in enumerate(self.neighbors[i]):
                elems_j = set(self.mesh.edge_to_elements[j])
                shared_elems = elems_i.intersection(elems_j)

                # Sum areas of shared elements
                shared_area = sum(self.mesh.element_areas[e] for e in shared_elems)
                weights[k] = shared_area if shared_area > 0 else 1e-10

            # Normalize
            if len(weights) > 0 and np.sum(weights) > 0:
                weights = weights / np.sum(weights)

            self.neighbor_weights.append(weights)

    def apply_laplacian(self, d: np.ndarray) -> np.ndarray:
        """
        Apply graph Laplacian to damage field d.

        Computes: (Δ_graph d)_i = Σ_{j ∈ N(i)} w_ij (d_j - d_i)

        Args:
            d: damage values on edges, shape (n_edges,)

        Returns:
            Laplacian values, shape (n_edges,)
        """
        result = np.zeros(self.n_edges)
        for i in range(self.n_edges):
            for j, w in zip(self.neighbors[i], self.neighbor_weights[i]):
                result[i] += w * (d[j] - d[i])
        return result

    def get_laplacian_matrix(self) -> csr_matrix:
        """
        Return sparse graph Laplacian matrix.

        The Laplacian matrix L has:
        - L[i,i] = -Σ_j w_ij (negative sum of weights)
        - L[i,j] = w_ij (off-diagonal)

        Note: This is the combinatorial Laplacian, so L @ d gives
        the same result as apply_laplacian(d).

        Returns:
            L: sparse CSR matrix, shape (n_edges, n_edges)
        """
        L = lil_matrix((self.n_edges, self.n_edges))

        for i in range(self.n_edges):
            diag_sum = 0.0
            for j, w in zip(self.neighbors[i], self.neighbor_weights[i]):
                L[i, j] = w
                diag_sum += w
            L[i, i] = -diag_sum

        return L.tocsr()

    def get_squared_gradient_matrix(self) -> csr_matrix:
        """
        Return matrix for computing sum of squared gradient terms.

        For the surface energy term: Σ_i Σ_{j∈N(i)} w_ij (d_j - d_i)²

        This can be written as d^T G d where G is returned by this function.
        Note: G accounts for the double-counting factor.

        Returns:
            G: sparse CSR matrix, shape (n_edges, n_edges)
        """
        G = lil_matrix((self.n_edges, self.n_edges))

        for i in range(self.n_edges):
            for j, w in zip(self.neighbors[i], self.neighbor_weights[i]):
                # Contribution from (d_j - d_i)²
                # = d_i² - 2 d_i d_j + d_j²
                # Diagonal contributions
                G[i, i] += w
                G[j, j] += w
                # Off-diagonal
                G[i, j] -= w
                G[j, i] -= w

        # Factor of 1/2 for double counting (i,j) and (j,i)
        G = G * 0.5

        return G.tocsr()

    def get_neighbor_count(self) -> np.ndarray:
        """
        Get number of neighbors for each edge.

        Returns:
            counts: shape (n_edges,)
        """
        return np.array([len(n) for n in self.neighbors])

    def get_interior_edges(self) -> np.ndarray:
        """
        Get indices of interior (non-boundary) edges.

        Returns:
            interior_edge_indices: array of edge indices
        """
        boundary_set = set(self.mesh.boundary_edges)
        return np.array([i for i in range(self.n_edges)
                         if i not in boundary_set], dtype=np.int64)
