Mesh Module
===========

The mesh module provides data structures and functions for creating and manipulating
triangular meshes for the GraFEA framework.

.. contents:: Contents
   :local:
   :depth: 2

TriangleMesh Class
------------------

.. autoclass:: mesh.TriangleMesh
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

EdgeGraph Class
---------------

.. autoclass:: mesh.EdgeGraph
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Mesh Generation Functions
-------------------------

create_rectangle_mesh
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.create_rectangle_mesh

create_square_mesh
~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.create_square_mesh

create_single_element
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.create_single_element

create_two_element_patch
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.create_two_element_patch

create_notched_rectangle
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.create_notched_rectangle

Mesh Manipulation
-----------------

refine_mesh
~~~~~~~~~~~

.. autofunction:: mesh.refine_mesh

perturb_interior_nodes
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mesh.perturb_interior_nodes

Mesh I/O
--------

read_gmsh
~~~~~~~~~

.. autofunction:: mesh.read_gmsh

write_vtk
~~~~~~~~~

.. autofunction:: mesh.write_vtk

read_triangle
~~~~~~~~~~~~~

.. autofunction:: mesh.read_triangle

Usage Examples
--------------

Creating a Simple Mesh
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mesh import create_rectangle_mesh, EdgeGraph

   # Create a rectangular mesh
   mesh = create_rectangle_mesh(
       width=1.0,
       height=0.5,
       nx=20,
       ny=10
   )

   # Access mesh properties
   print(f"Number of nodes: {mesh.n_nodes}")
   print(f"Number of elements: {mesh.n_elements}")
   print(f"Number of edges: {mesh.n_edges}")

   # Get boundary information
   boundary_edges = mesh.boundary_edges
   boundary_nodes = mesh.boundary_nodes

   # Create edge graph for damage regularization
   edge_graph = EdgeGraph(mesh)

Working with Mesh Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Element areas
   areas = mesh.element_areas
   print(f"Total area: {np.sum(areas)}")

   # Edge lengths
   lengths = mesh.edge_lengths
   print(f"Average edge length: {np.mean(lengths)}")

   # Find nodes in a region
   bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < 0.01)

Mesh Refinement
~~~~~~~~~~~~~~~

.. code-block:: python

   from mesh import create_square_mesh, refine_mesh

   # Create coarse mesh
   coarse = create_square_mesh(1.0, 5)

   # Refine uniformly
   fine = refine_mesh(coarse)

   print(f"Coarse: {coarse.n_elements} elements")
   print(f"Fine: {fine.n_elements} elements")  # 4x more
