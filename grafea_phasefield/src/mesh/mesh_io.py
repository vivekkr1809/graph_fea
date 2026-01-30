"""
Mesh I/O Functions
==================

Read and write mesh files in various formats.
"""

import numpy as np
from typing import Dict, Optional
from .triangle_mesh import TriangleMesh


def read_gmsh(filename: str) -> TriangleMesh:
    """
    Read Gmsh .msh file and return TriangleMesh.

    Supports Gmsh format version 2.2 and 4.1.

    Args:
        filename: path to .msh file

    Returns:
        TriangleMesh instance
    """
    try:
        import meshio
        mesh_data = meshio.read(filename)

        # Extract nodes
        nodes = mesh_data.points[:, :2]  # Take x, y coordinates

        # Find triangle cells
        elements = None
        for cell_block in mesh_data.cells:
            if cell_block.type == "triangle":
                elements = cell_block.data
                break

        if elements is None:
            raise ValueError("No triangle elements found in mesh file")

        return TriangleMesh(nodes, elements)

    except ImportError:
        # Fallback: manual parsing of Gmsh format
        return _read_gmsh_manual(filename)


def _read_gmsh_manual(filename: str) -> TriangleMesh:
    """
    Manual Gmsh file parser (format 2.2).

    Args:
        filename: path to .msh file

    Returns:
        TriangleMesh instance
    """
    nodes = []
    elements = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line = line.strip()

            if line == "$Nodes":
                n_nodes = int(f.readline().strip())
                for _ in range(n_nodes):
                    parts = f.readline().split()
                    # Format: node_id x y z
                    x, y = float(parts[1]), float(parts[2])
                    nodes.append([x, y])

            elif line == "$Elements":
                n_elems = int(f.readline().strip())
                for _ in range(n_elems):
                    parts = f.readline().split()
                    # Format: elem_id elem_type n_tags [tags...] node_ids...
                    elem_type = int(parts[1])
                    if elem_type == 2:  # 3-node triangle
                        n_tags = int(parts[2])
                        node_ids = [int(parts[3 + n_tags + i]) - 1
                                    for i in range(3)]
                        elements.append(node_ids)

            line = f.readline()

    if not nodes or not elements:
        raise ValueError("Failed to parse Gmsh file")

    return TriangleMesh(np.array(nodes), np.array(elements))


def write_vtk(mesh: TriangleMesh, filename: str,
              node_data: Optional[Dict[str, np.ndarray]] = None,
              edge_data: Optional[Dict[str, np.ndarray]] = None,
              cell_data: Optional[Dict[str, np.ndarray]] = None) -> None:
    """
    Write VTK file for ParaView visualization.

    Args:
        mesh: TriangleMesh instance
        filename: output filename (should end with .vtk or .vtu)
        node_data: dict of node-based scalar/vector fields
        edge_data: dict of edge-based scalar fields (will be mapped to cells)
        cell_data: dict of element-based scalar/vector fields
    """
    try:
        import meshio

        points = np.column_stack([mesh.nodes, np.zeros(mesh.n_nodes)])
        cells = [("triangle", mesh.elements)]

        # Prepare data dictionaries
        point_data = node_data if node_data else {}
        cell_data_dict = {}

        if cell_data:
            for name, data in cell_data.items():
                cell_data_dict[name] = [data]

        # Edge data needs special handling - map to visualization
        if edge_data:
            # For edge data, we can create a separate mesh of line segments
            _write_edge_vtk(mesh, filename.replace('.vtk', '_edges.vtk'),
                           filename.replace('.vtu', '_edges.vtu'),
                           edge_data)

        meshio_mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data_dict if cell_data_dict else None
        )
        meshio.write(filename, meshio_mesh)

    except ImportError:
        # Fallback: manual VTK writing
        _write_vtk_manual(mesh, filename, node_data, cell_data)


def _write_edge_vtk(mesh: TriangleMesh, filename_vtk: str, filename_vtu: str,
                    edge_data: Dict[str, np.ndarray]) -> None:
    """
    Write edge data as line segments in VTK format.
    """
    try:
        import meshio

        points = np.column_stack([mesh.nodes, np.zeros(mesh.n_nodes)])
        cells = [("line", mesh.edges)]

        cell_data_dict = {}
        for name, data in edge_data.items():
            cell_data_dict[name] = [data]

        meshio_mesh = meshio.Mesh(
            points=points,
            cells=cells,
            cell_data=cell_data_dict
        )

        # Use .vtu extension for better compatibility
        out_file = filename_vtu if filename_vtu.endswith('.vtu') else filename_vtk
        meshio.write(out_file, meshio_mesh)

    except ImportError:
        pass  # Skip if meshio not available


def _write_vtk_manual(mesh: TriangleMesh, filename: str,
                      node_data: Optional[Dict] = None,
                      cell_data: Optional[Dict] = None) -> None:
    """
    Manual VTK legacy format writer.
    """
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("GraFEA Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {mesh.n_nodes} double\n")
        for x, y in mesh.nodes:
            f.write(f"{x} {y} 0.0\n")

        # Cells
        n_cells = mesh.n_elements
        cell_size = n_cells * 4  # 3 nodes + 1 count per cell
        f.write(f"CELLS {n_cells} {cell_size}\n")
        for elem in mesh.elements:
            f.write(f"3 {elem[0]} {elem[1]} {elem[2]}\n")

        # Cell types (5 = VTK_TRIANGLE)
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("5\n")

        # Point data
        if node_data:
            f.write(f"POINT_DATA {mesh.n_nodes}\n")
            for name, data in node_data.items():
                if data.ndim == 1:
                    f.write(f"SCALARS {name} double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val}\n")
                else:
                    f.write(f"VECTORS {name} double\n")
                    for vec in data:
                        f.write(f"{vec[0]} {vec[1]} 0.0\n")

        # Cell data
        if cell_data:
            f.write(f"CELL_DATA {n_cells}\n")
            for name, data in cell_data.items():
                f.write(f"SCALARS {name} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{val}\n")


def read_triangle(basename: str) -> TriangleMesh:
    """
    Read Triangle format (.node, .ele files).

    Triangle is a mesh generator by Jonathan Shewchuk.

    Args:
        basename: base filename (without extension)

    Returns:
        TriangleMesh instance
    """
    # Read nodes
    nodes = []
    with open(f"{basename}.node", 'r') as f:
        first_line = f.readline().split()
        n_nodes = int(first_line[0])
        for _ in range(n_nodes):
            parts = f.readline().split()
            x, y = float(parts[1]), float(parts[2])
            nodes.append([x, y])

    # Read elements
    elements = []
    with open(f"{basename}.ele", 'r') as f:
        first_line = f.readline().split()
        n_elems = int(first_line[0])
        for _ in range(n_elems):
            parts = f.readline().split()
            # Triangle uses 1-based indexing by default
            node_ids = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
            elements.append(node_ids)

    return TriangleMesh(np.array(nodes), np.array(elements))
