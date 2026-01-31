Simple Tension Test Tutorial
=============================

This tutorial walks through a complete simple tension test simulation, from mesh
generation to results visualization.

.. contents:: Contents
   :local:
   :depth: 2

Problem Description
-------------------

We simulate a rectangular specimen under uniaxial tension:

- **Geometry**: 1.0 m × 0.5 m rectangle
- **Loading**: Fixed bottom, prescribed displacement on top
- **Material**: Steel-like properties

The goal is to observe damage initiation and propagation under tensile loading.

Step 1: Import Modules
----------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Mesh module
   from mesh import create_rectangle_mesh, EdgeGraph

   # Element module
   from elements import GraFEAElement

   # Physics module
   from physics import IsotropicMaterial

   # Assembly module
   from assembly import (create_bc_from_region, merge_bcs,
                         assemble_global_stiffness)

   # Solver module
   from solvers import StaggeredSolver, SolverConfig

   # Postprocessing module
   from postprocess import (plot_mesh, plot_damage_field,
                            plot_energy_evolution, plot_deformed_mesh)

Step 2: Create the Mesh
-----------------------

Generate a structured triangular mesh:

.. code-block:: python

   # Mesh dimensions
   width = 1.0    # m
   height = 0.5   # m
   nx = 20        # elements in x-direction
   ny = 10        # elements in y-direction

   # Create mesh
   mesh = create_rectangle_mesh(width, height, nx, ny)

   print(f"Mesh statistics:")
   print(f"  Nodes: {mesh.n_nodes}")
   print(f"  Elements: {mesh.n_elements}")
   print(f"  Edges: {mesh.n_edges}")

   # Visualize mesh
   fig, ax = plt.subplots(figsize=(10, 5))
   plot_mesh(mesh, ax=ax)
   ax.set_title('Mesh')
   ax.set_aspect('equal')
   plt.show()

Expected output::

   Mesh statistics:
     Nodes: 231
     Elements: 400
     Edges: 630

Step 3: Define Material Properties
----------------------------------

Create an isotropic material with phase-field parameters:

.. code-block:: python

   # Material properties
   E = 210e9      # Young's modulus [Pa]
   nu = 0.3       # Poisson's ratio [-]
   Gc = 2700      # Critical energy release rate [J/m^2]

   # Estimate length scale from mesh
   h = np.mean(mesh.edge_lengths)
   l0 = 2 * h     # Rule of thumb: l0 ≈ 2h

   print(f"Average element size: {h:.4f} m")
   print(f"Phase-field length scale: {l0:.4f} m")

   # Create material
   material = IsotropicMaterial(E=E, nu=nu, Gc=Gc, l0=l0)

   print(f"\nMaterial properties:")
   print(f"  Shear modulus: {material.shear_modulus:.2e} Pa")
   print(f"  Bulk modulus: {material.bulk_modulus:.2e} Pa")
   print(f"  Critical stress: {material.critical_stress():.2e} Pa")

Step 4: Create Elements and Edge Graph
--------------------------------------

Build the element objects and edge graph for damage regularization:

.. code-block:: python

   # Create GraFEA elements for each mesh element
   elements = []
   for e in range(mesh.n_elements):
       node_coords = mesh.nodes[mesh.elements[e]]
       elem = GraFEAElement(node_coords, material)
       elements.append(elem)

   print(f"Created {len(elements)} GraFEA elements")

   # Create edge graph for damage regularization
   edge_graph = EdgeGraph(mesh)

   print(f"Edge graph:")
   print(f"  Nodes (edges): {edge_graph.n_edges}")
   print(f"  Average neighbors: {np.mean([len(n) for n in edge_graph.neighbors]):.1f}")

Step 5: Set Up Boundary Conditions
----------------------------------

Define the mechanical boundary conditions:

.. code-block:: python

   # Tolerance for boundary detection
   tol = 1e-10

   # Bottom edge: fixed (u_x = 0, u_y = 0)
   bc_bottom = create_bc_from_region(
       mesh,
       region=lambda x, y: y < tol,
       component='both',
       value=0.0
   )

   # Top edge: prescribed vertical displacement
   max_displacement = 0.005  # 5 mm maximum displacement

   bc_top = create_bc_from_region(
       mesh,
       region=lambda x, y: y > height - tol,
       component='y',
       value=max_displacement
   )

   # Fix horizontal movement on top (optional, for symmetry)
   bc_top_x = create_bc_from_region(
       mesh,
       region=lambda x, y: y > height - tol,
       component='x',
       value=0.0
   )

   # Merge all boundary conditions
   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top, bc_top_x])

   print(f"Boundary conditions:")
   print(f"  Total constrained DOFs: {len(bc_dofs)}")
   print(f"  Bottom nodes: {np.sum(mesh.nodes[:, 1] < tol)}")
   print(f"  Top nodes: {np.sum(mesh.nodes[:, 1] > height - tol)}")

Step 6: Configure the Solver
----------------------------

Set up the staggered solver with appropriate parameters:

.. code-block:: python

   # Solver configuration
   config = SolverConfig(
       tol_u=1e-6,            # Displacement tolerance
       tol_d=1e-6,            # Damage tolerance
       max_stagger_iter=100,  # Maximum staggered iterations
       min_stagger_iter=2,    # Minimum iterations (ensure coupling)
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

Step 7: Define Load Schedule
----------------------------

Create a load schedule with finer steps near expected fracture:

.. code-block:: python

   # Load factors (0 to 1 = 0 to max_displacement)
   load_factors = np.concatenate([
       np.linspace(0, 0.5, 10),    # Coarse steps initially
       np.linspace(0.5, 0.8, 15),  # Medium steps
       np.linspace(0.8, 1.0, 25),  # Fine steps near failure
   ])

   print(f"Load schedule: {len(load_factors)} steps")

   # BC value function
   def bc_value_func(load_factor):
       return bc_values * load_factor

Step 8: Run the Simulation
--------------------------

Execute the solver:

.. code-block:: python

   print("Starting simulation...")
   print("=" * 50)

   results = solver.solve(
       load_factors=load_factors,
       bc_dofs=bc_dofs,
       bc_value_func=bc_value_func
   )

   print("=" * 50)
   print("Simulation complete!")

   # Check convergence
   n_converged = sum(1 for r in results if r.converged)
   print(f"Converged steps: {n_converged}/{len(results)}")

Step 9: Analyze Results
-----------------------

Extract and examine the results:

.. code-block:: python

   # Get summary
   load_factors_out = [r.load_factor for r in results]
   strain_energies = [r.strain_energy for r in results]
   surface_energies = [r.surface_energy for r in results]
   max_damages = [r.damage.max() for r in results]

   print("\nResults summary:")
   print(f"  Final load factor: {results[-1].load_factor:.3f}")
   print(f"  Final strain energy: {results[-1].strain_energy:.4e} J")
   print(f"  Final surface energy: {results[-1].surface_energy:.4e} J")
   print(f"  Maximum damage: {max_damages[-1]:.3f}")

   # Find when damage initiates
   damage_threshold = 0.01
   for i, r in enumerate(results):
       if r.damage.max() > damage_threshold:
           print(f"\nDamage initiation at step {i}, load factor = {r.load_factor:.3f}")
           break

Step 10: Visualize Results
--------------------------

Create plots to visualize the simulation:

.. code-block:: python

   # Plot 1: Damage evolution at different load steps
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   steps_to_plot = [
       len(results) // 4,
       len(results) // 2,
       3 * len(results) // 4,
       -1
   ]

   for ax, step_idx in zip(axes.flatten(), steps_to_plot):
       result = results[step_idx]
       plot_damage_field(mesh, result.damage, ax=ax, cmap='hot_r')
       ax.set_title(f'Load Factor = {result.load_factor:.3f}')
       ax.set_aspect('equal')

   plt.suptitle('Damage Evolution', fontsize=14)
   plt.tight_layout()
   plt.savefig('damage_evolution.png', dpi=150)
   plt.show()

.. code-block:: python

   # Plot 2: Energy evolution
   fig, ax = plt.subplots(figsize=(10, 6))
   plot_energy_evolution(results, ax=ax)
   ax.set_xlabel('Load Factor')
   ax.set_ylabel('Energy [J]')
   ax.set_title('Energy Evolution')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.savefig('energy_evolution.png', dpi=150)
   plt.show()

.. code-block:: python

   # Plot 3: Final deformed mesh
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   # Undeformed with damage
   plot_damage_field(mesh, results[-1].damage, ax=axes[0], cmap='hot_r')
   axes[0].set_title('Final Damage Field')
   axes[0].set_aspect('equal')

   # Deformed mesh
   plot_deformed_mesh(mesh, results[-1].displacement, scale=20, ax=axes[1])
   axes[1].set_title('Deformed Mesh (20x scale)')
   axes[1].set_aspect('equal')

   plt.tight_layout()
   plt.savefig('final_state.png', dpi=150)
   plt.show()

.. code-block:: python

   # Plot 4: Maximum damage vs load factor
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.plot(load_factors_out, max_damages, 'b-', linewidth=2)
   ax.set_xlabel('Load Factor')
   ax.set_ylabel('Maximum Damage')
   ax.set_title('Damage Growth')
   ax.grid(True, alpha=0.3)
   ax.set_ylim([0, 1.05])
   plt.savefig('damage_growth.png', dpi=150)
   plt.show()

Complete Script
---------------

Here's the complete script in one block:

.. literalinclude:: ../../grafea_phasefield/examples/simple_tension.py
   :language: python
   :caption: simple_tension.py

Expected Results
----------------

For the parameters used in this tutorial, you should observe:

1. **Linear elastic response** initially with no damage
2. **Damage initiation** when strain energy reaches critical level
3. **Damage localization** forming a band across the specimen
4. **Softening response** as damage grows

The energy plot should show:

- Increasing strain energy during loading
- Surface energy activation once damage begins
- Total energy increasing monotonically (thermodynamic consistency)

Exercises
---------

Try modifying the tutorial to explore:

1. **Mesh refinement**: Increase ``nx`` and ``ny`` and observe convergence
2. **Length scale**: Vary ``l0`` and see how crack width changes
3. **Pre-notched specimen**: Use ``create_notched_rectangle`` instead
4. **Different materials**: Try glass or aluminum properties
5. **Cyclic loading**: Modify load factors to include unloading

Next Steps
----------

- :doc:`advanced_usage` - More complex examples
- :doc:`../theory` - Understand the underlying mathematics
- :doc:`../api/solvers` - Detailed solver documentation
