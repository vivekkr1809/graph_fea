Quick Start Guide
=================

This guide will walk you through setting up and running your first simulation
with the GraFEA Phase-Field Framework.

Basic Workflow
--------------

A typical simulation follows these steps:

1. Create a mesh
2. Define material properties
3. Build elements and edge graph
4. Set up boundary conditions
5. Configure and run the solver
6. Visualize results

Step 1: Create a Mesh
---------------------

The framework provides convenient mesh generation functions:

.. code-block:: python

   from mesh import create_rectangle_mesh, create_notched_rectangle

   # Simple rectangular mesh
   mesh = create_rectangle_mesh(
       width=1.0,      # Width of domain
       height=0.5,     # Height of domain
       nx=20,          # Elements in x-direction
       ny=10           # Elements in y-direction
   )

   print(f"Mesh has {mesh.n_nodes} nodes and {mesh.n_elements} elements")
   print(f"Number of edges: {mesh.n_edges}")

For fracture simulations, you might want a pre-notched specimen:

.. code-block:: python

   # Single Edge Notched Tension (SENT) specimen
   mesh = create_notched_rectangle(
       width=1.0,
       height=0.5,
       notch_length=0.5,  # 50% of width
       nx=40,
       ny=20
   )

Step 2: Define Material Properties
----------------------------------

Create an isotropic material with phase-field fracture parameters:

.. code-block:: python

   from physics import IsotropicMaterial

   # Steel-like material
   material = IsotropicMaterial(
       E=210e9,       # Young's modulus [Pa]
       nu=0.3,        # Poisson's ratio [-]
       Gc=2700,       # Critical energy release rate [J/m^2]
       l0=0.02        # Phase-field length scale [m]
   )

   # Check derived properties
   print(f"Shear modulus: {material.shear_modulus:.2e} Pa")
   print(f"Bulk modulus: {material.bulk_modulus:.2e} Pa")

.. note::
   The length scale ``l0`` should typically be chosen as 2-4 times the mesh size
   to ensure proper regularization.

Step 3: Build Elements and Edge Graph
-------------------------------------

Create GraFEA elements for each mesh element and the edge graph for damage regularization:

.. code-block:: python

   from elements import GraFEAElement
   from mesh import EdgeGraph

   # Create GraFEA elements
   elements = []
   for e in range(mesh.n_elements):
       node_coords = mesh.nodes[mesh.elements[e]]
       elem = GraFEAElement(node_coords, material)
       elements.append(elem)

   # Create edge graph for damage regularization
   edge_graph = EdgeGraph(mesh)

Step 4: Set Up Boundary Conditions
----------------------------------

Define Dirichlet boundary conditions for the problem:

.. code-block:: python

   from assembly import create_bc_from_region, merge_bcs
   import numpy as np

   # Fix bottom edge (y = 0)
   bc_bottom = create_bc_from_region(
       mesh,
       region=lambda x, y: y < 1e-10,
       component='both',  # Fix both x and y
       value=0.0
   )

   # Prescribe displacement on top edge
   bc_top = create_bc_from_region(
       mesh,
       region=lambda x, y: y > mesh.nodes[:, 1].max() - 1e-10,
       component='y',
       value=0.001  # 1mm displacement
   )

   # Merge boundary conditions
   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top])

Step 5: Configure and Run the Solver
------------------------------------

Set up the staggered solver with appropriate parameters:

.. code-block:: python

   from solvers import StaggeredSolver, SolverConfig

   # Solver configuration
   config = SolverConfig(
       tol_u=1e-6,           # Displacement tolerance
       tol_d=1e-6,           # Damage tolerance
       max_stagger_iter=100,  # Max iterations per load step
       verbose=True           # Print convergence info
   )

   # Create solver
   solver = StaggeredSolver(
       mesh=mesh,
       elements=elements,
       material=material,
       edge_graph=edge_graph,
       config=config
   )

   # Define load steps (gradually increase loading)
   load_factors = np.linspace(0, 1, 20)

   # Function to compute BC values at each load factor
   def bc_value_func(load_factor):
       return bc_values * load_factor

   # Run simulation
   results = solver.solve(
       load_factors=load_factors,
       bc_dofs=bc_dofs,
       bc_value_func=bc_value_func
   )

Step 6: Visualize Results
-------------------------

Plot the results using the postprocessing module:

.. code-block:: python

   from postprocess import plot_damage_field, plot_deformed_mesh, plot_energy_evolution
   import matplotlib.pyplot as plt

   # Get final result
   final = results[-1]

   # Plot damage field
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   plot_damage_field(mesh, final.damage, ax=axes[0])
   axes[0].set_title('Damage Field')

   plot_deformed_mesh(mesh, final.displacement, scale=10, ax=axes[1])
   axes[1].set_title('Deformed Mesh (10x scale)')

   plt.tight_layout()
   plt.show()

   # Plot energy evolution
   plot_energy_evolution(results)
   plt.show()

Complete Example
----------------

Here's the complete code in one block:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from mesh import create_rectangle_mesh, EdgeGraph
   from elements import GraFEAElement
   from physics import IsotropicMaterial
   from assembly import create_bc_from_region, merge_bcs
   from solvers import StaggeredSolver, SolverConfig
   from postprocess import plot_damage_field, plot_energy_evolution

   # 1. Create mesh
   mesh = create_rectangle_mesh(1.0, 0.5, 20, 10)

   # 2. Define material
   material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.02)

   # 3. Build elements and edge graph
   elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
               for e in range(mesh.n_elements)]
   edge_graph = EdgeGraph(mesh)

   # 4. Boundary conditions
   bc_bottom = create_bc_from_region(mesh, lambda x, y: y < 1e-10, 'both', 0.0)
   bc_top = create_bc_from_region(mesh, lambda x, y: y > 0.5 - 1e-10, 'y', 0.001)
   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top])

   # 5. Solve
   config = SolverConfig(verbose=True)
   solver = StaggeredSolver(mesh, elements, material, edge_graph, config)
   results = solver.solve(
       load_factors=np.linspace(0, 1, 20),
       bc_dofs=bc_dofs,
       bc_value_func=lambda lf: bc_values * lf
   )

   # 6. Visualize
   plot_damage_field(mesh, results[-1].damage)
   plt.title('Final Damage Field')
   plt.show()

Next Steps
----------

- See :doc:`tutorials/simple_tension` for a detailed walkthrough
- Read :doc:`theory` to understand the underlying mathematics
- Explore the :doc:`api/index` for detailed function references
