GraFEA Phase-Field Framework Documentation
==========================================

**GraFEA Phase-Field Framework** is a Python-based finite element library implementing
edge-based phase-field fracture mechanics using Graph-based Finite Element Analysis (GraFEA).

.. note::
   This framework implements a novel approach where damage is defined on mesh edges
   rather than at integration points, providing a natural way to model crack propagation.

Key Features
------------

- **Edge-based damage**: Damage variables are associated with mesh edges, providing a discrete representation of crack surfaces
- **Spectral tension-compression split**: Only tensile strain energy drives damage evolution, preventing crack interpenetration
- **Graph Laplacian regularization**: Smooth damage fields via edge graph connectivity
- **Staggered solver**: Robust alternating minimization algorithm for coupled displacement-damage problems

Quick Example
-------------

.. code-block:: python

   from mesh import create_rectangle_mesh, EdgeGraph
   from elements import GraFEAElement
   from physics import IsotropicMaterial
   from solvers import StaggeredSolver

   # Create mesh and material
   mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)
   material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.05)

   # Setup elements and edge graph
   elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
               for e in range(mesh.n_elements)]
   edge_graph = EdgeGraph(mesh)

   # Create solver and run analysis
   solver = StaggeredSolver(mesh, elements, material, edge_graph)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   theory
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/simple_tension
   tutorials/advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/mesh
   api/elements
   api/physics
   api/assembly
   api/solvers
   api/postprocess

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
