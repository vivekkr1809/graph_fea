"""
Mesh Module
===========

Triangle mesh with edge-based data structures for GraFEA.
"""

from .triangle_mesh import TriangleMesh
from .edge_graph import EdgeGraph
from .mesh_io import read_gmsh, write_vtk, read_triangle
from .mesh_generators import (
    create_rectangle_mesh,
    create_square_mesh,
    create_single_element,
    create_two_element_patch,
    create_notched_rectangle,
    perturb_interior_nodes,
    refine_mesh,
)

__all__ = [
    "TriangleMesh",
    "EdgeGraph",
    "read_gmsh",
    "write_vtk",
    "read_triangle",
    "create_rectangle_mesh",
    "create_square_mesh",
    "create_single_element",
    "create_two_element_patch",
    "create_notched_rectangle",
    "perturb_interior_nodes",
    "refine_mesh",
]
