"""
Tests for Mesh Module
=====================
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from mesh.mesh_generators import (
    create_rectangle_mesh, create_single_element,
    create_two_element_patch, create_square_mesh
)


class TestTriangleMesh:
    """Tests for TriangleMesh class."""

    def test_simple_triangle(self):
        """Single element - verify connectivity."""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]])
        mesh = TriangleMesh(nodes, elements)

        assert mesh.n_nodes == 3
        assert mesh.n_elements == 1
        assert mesh.n_edges == 3
        assert len(mesh.boundary_edges) == 3  # All edges are boundary
        assert len(mesh.boundary_nodes) == 3  # All nodes are boundary

    def test_two_triangles(self):
        """Two elements sharing an edge."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = TriangleMesh(nodes, elements)

        assert mesh.n_nodes == 4
        assert mesh.n_elements == 2
        assert mesh.n_edges == 5  # 4 boundary + 1 internal

        # Find the internal edge (shared edge should have 2 elements)
        internal_edges = [i for i, elems in enumerate(mesh.edge_to_elements)
                         if len(elems) == 2]
        assert len(internal_edges) == 1

        # The shared edge connects nodes 0 and 2
        internal_edge = internal_edges[0]
        edge_nodes = set(mesh.edges[internal_edge])
        assert edge_nodes == {0, 2}

    def test_edge_convention(self):
        """Verify edge convention (edge k opposite to node k)."""
        mesh = create_single_element()

        # For element with nodes (0, 1, 2):
        # Edge 0: connects 1-2 (opposite to 0)
        # Edge 1: connects 2-0 (opposite to 1)
        # Edge 2: connects 0-1 (opposite to 2)

        elem_edges = mesh.element_to_edges[0]

        # Check edge 2 (connects nodes 0 and 1)
        edge_2_nodes = set(mesh.edges[elem_edges[2]])
        assert edge_2_nodes == {0, 1}, f"Edge 2 should connect nodes 0-1, got {edge_2_nodes}"

    def test_area_computation(self):
        """Test element area computation."""
        # Right triangle with legs of length 1
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]])
        mesh = TriangleMesh(nodes, elements)

        assert np.isclose(mesh.element_areas[0], 0.5)

        # Equilateral triangle with side 2
        s = 2.0
        h = s * np.sqrt(3) / 2
        nodes = np.array([[0, 0], [s, 0], [s/2, h]], dtype=float)
        elements = np.array([[0, 1, 2]])
        mesh = TriangleMesh(nodes, elements)

        expected_area = s**2 * np.sqrt(3) / 4
        assert np.isclose(mesh.element_areas[0], expected_area, rtol=1e-10)

    def test_edge_lengths(self):
        """Test edge length computation."""
        nodes = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)  # 3-4-5 triangle
        elements = np.array([[0, 1, 2]])
        mesh = TriangleMesh(nodes, elements)

        lengths = sorted(mesh.edge_lengths)
        assert np.allclose(lengths, [3, 4, 5])

    def test_boundary_detection(self):
        """Test boundary edge and node detection."""
        mesh = create_rectangle_mesh(1, 1, 3, 3)

        # All boundary edges should have only one adjacent element
        for edge_idx in mesh.boundary_edges:
            assert len(mesh.edge_to_elements[edge_idx]) == 1

        # Interior edges should have exactly two adjacent elements
        for edge_idx in range(mesh.n_edges):
            if edge_idx not in mesh.boundary_edges:
                assert len(mesh.edge_to_elements[edge_idx]) == 2


class TestEdgeGraph:
    """Tests for EdgeGraph class."""

    def test_edge_neighbors_single_element(self):
        """Test edge neighbors in single element."""
        mesh = create_single_element()
        edge_graph = EdgeGraph(mesh)

        # Each edge should have exactly 2 neighbors (the other edges)
        for i in range(mesh.n_edges):
            assert len(edge_graph.neighbors[i]) == 2

    def test_edge_graph_neighbors_two_elements(self):
        """Verify edge neighbor detection in two-element mesh."""
        mesh = create_two_element_patch()
        edge_graph = EdgeGraph(mesh)

        # The internal edge should have 4 neighbors
        internal_edges = [i for i, elems in enumerate(mesh.edge_to_elements)
                         if len(elems) == 2]
        assert len(internal_edges) == 1

        internal_edge = internal_edges[0]
        # Internal edge connects to all boundary edges through shared vertices
        assert len(edge_graph.neighbors[internal_edge]) >= 4

    def test_graph_laplacian_constant(self):
        """Laplacian of constant field should be zero."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        edge_graph = EdgeGraph(mesh)

        d = np.ones(mesh.n_edges) * 0.5
        lap_d = edge_graph.apply_laplacian(d)

        assert np.allclose(lap_d, 0, atol=1e-10)

    def test_graph_laplacian_linear(self):
        """Test Laplacian on linear function."""
        mesh = create_rectangle_mesh(1, 1, 10, 10)
        edge_graph = EdgeGraph(mesh)

        # Linear function: d = x (where x is edge midpoint x-coordinate)
        midpoints = mesh.edge_midpoints
        d = midpoints[:, 0]

        lap_d = edge_graph.apply_laplacian(d)

        # For a uniform mesh with distance-based weights,
        # Laplacian of linear function should be approximately 0
        # (up to boundary effects)
        interior_edges = edge_graph.get_interior_edges()
        assert np.allclose(lap_d[interior_edges], 0, atol=0.1)

    def test_weight_schemes(self):
        """Test different weight schemes."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)

        for scheme in ['uniform', 'distance', 'area']:
            edge_graph = EdgeGraph(mesh, weight_scheme=scheme)

            # Weights should be normalized (sum to ~1 for interior edges)
            for i in edge_graph.get_interior_edges():
                weight_sum = np.sum(edge_graph.neighbor_weights[i])
                assert 0 < weight_sum <= 1.5, f"Weight sum for scheme {scheme}: {weight_sum}"


class TestMeshGenerators:
    """Tests for mesh generators."""

    def test_rectangle_mesh_dimensions(self):
        """Test rectangle mesh has correct dimensions."""
        Lx, Ly = 2.0, 3.0
        nx, ny = 5, 7
        mesh = create_rectangle_mesh(Lx, Ly, nx, ny)

        # Check bounds
        assert np.isclose(mesh.nodes[:, 0].min(), 0)
        assert np.isclose(mesh.nodes[:, 0].max(), Lx)
        assert np.isclose(mesh.nodes[:, 1].min(), 0)
        assert np.isclose(mesh.nodes[:, 1].max(), Ly)

        # Check number of nodes and elements
        assert mesh.n_nodes == (nx + 1) * (ny + 1)
        assert mesh.n_elements == 2 * nx * ny

    def test_rectangle_mesh_patterns(self):
        """Test different mesh patterns."""
        for pattern in ['right', 'left', 'alternating']:
            mesh = create_rectangle_mesh(1, 1, 5, 5, pattern=pattern)
            assert mesh.n_elements == 50  # 2 * 5 * 5

    def test_single_element_default(self):
        """Test default single element creation."""
        mesh = create_single_element()
        assert mesh.n_nodes == 3
        assert mesh.n_elements == 1
        assert mesh.n_edges == 3

    def test_single_element_custom(self):
        """Test single element with custom coordinates."""
        nodes = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        mesh = create_single_element(nodes)

        assert np.allclose(mesh.nodes, nodes)

    def test_square_mesh(self):
        """Test square mesh convenience function."""
        mesh = create_square_mesh(1.0, 5)
        assert mesh.n_nodes == 36  # (5+1)^2
        assert mesh.n_elements == 50  # 2 * 5 * 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
