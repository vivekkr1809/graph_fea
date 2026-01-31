Advanced Usage Tutorial
=======================

This tutorial covers advanced topics including pre-notched specimens, custom
boundary conditions, and parameter studies.

.. contents:: Contents
   :local:
   :depth: 2

Single Edge Notched Tension (SENT) Test
---------------------------------------

The SENT test is a standard benchmark for phase-field fracture models.

Problem Setup
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from mesh import create_notched_rectangle, EdgeGraph
   from elements import GraFEAElement
   from physics import IsotropicMaterial
   from assembly import create_bc_from_region, merge_bcs
   from solvers import StaggeredSolver, SolverConfig
   from postprocess import plot_damage_field, plot_energy_evolution

   # Create notched specimen
   width = 1.0     # mm
   height = 1.0    # mm
   notch_length = 0.5  # 50% of width
   nx = 50
   ny = 50

   mesh = create_notched_rectangle(width, height, notch_length, nx, ny)

   print(f"SENT mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

Material and Elements
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Material (normalized units)
   material = IsotropicMaterial(
       E=210e3,      # MPa
       nu=0.3,
       Gc=2.7,       # N/mm
       l0=0.02       # mm (2x mesh size)
   )

   # Create elements
   elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
               for e in range(mesh.n_elements)]

   # Edge graph
   edge_graph = EdgeGraph(mesh)

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

For SENT, we typically:

- Fix the bottom edge completely
- Apply vertical displacement on top
- Allow free horizontal expansion

.. code-block:: python

   tol = 1e-10

   # Bottom: fully fixed
   bc_bottom = create_bc_from_region(mesh, lambda x, y: y < tol, 'both', 0.0)

   # Top: prescribed y-displacement only
   bc_top = create_bc_from_region(mesh, lambda x, y: y > height - tol, 'y', 0.01)

   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top])

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~

Use finer load stepping near crack propagation:

.. code-block:: python

   config = SolverConfig(tol_u=1e-6, tol_d=1e-6, max_stagger_iter=100)
   solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

   # Load factors with refinement
   load_factors = np.concatenate([
       np.linspace(0, 0.4, 10),
       np.linspace(0.4, 0.6, 30),
       np.linspace(0.6, 1.0, 60),
   ])

   results = solver.solve(load_factors, bc_dofs, lambda lf: bc_values * lf)

Crack Path Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find crack path (edges with damage > 0.95)
   from physics import identify_fully_damaged_edges

   final = results[-1]
   cracked = identify_fully_damaged_edges(final.damage, threshold=0.95)

   fig, ax = plt.subplots(figsize=(8, 8))
   plot_damage_field(mesh, final.damage, ax=ax, cmap='hot_r')

   # Highlight fully cracked edges
   for edge_idx in cracked:
       n1, n2 = mesh.edges[edge_idx]
       coords = mesh.nodes[[n1, n2]]
       ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)

   ax.set_title('SENT: Damage Field with Crack Path')
   ax.set_aspect('equal')
   plt.show()

Custom Boundary Conditions
--------------------------

Mixed Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Fixed corner with applied traction

.. code-block:: python

   from assembly import create_fixed_bc, create_roller_bc

   # Fix bottom-left corner
   corner_nodes = mesh.get_nodes_in_region(
       lambda x, y: (x < 0.01) & (y < 0.01)
   )
   bc_corner = create_fixed_bc(mesh, corner_nodes)

   # Roller on bottom (free in x, fixed in y)
   bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
   bc_bottom_roller = create_roller_bc(mesh, bottom_nodes, direction='x')

   # Combine
   bc_dofs, bc_values = merge_bcs([bc_corner, bc_bottom_roller])

Time-Varying Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply non-proportional loading:

.. code-block:: python

   def complex_bc_func(load_factor):
       """Non-proportional loading with two phases."""
       values = bc_values.copy()

       if load_factor < 0.5:
           # Phase 1: Apply horizontal compression
           mask_x = (bc_dofs % 2 == 0)  # x-components
           values[mask_x] = -0.002 * (load_factor / 0.5)
       else:
           # Phase 2: Apply vertical tension
           mask_y = (bc_dofs % 2 == 1)  # y-components
           values[mask_y] = 0.005 * ((load_factor - 0.5) / 0.5)

       return values

   results = solver.solve(load_factors, bc_dofs, complex_bc_func)

Parameter Studies
-----------------

Length Scale Study
~~~~~~~~~~~~~~~~~~

Investigate mesh-independence by varying :math:`l_0`:

.. code-block:: python

   l0_values = [0.01, 0.02, 0.04, 0.08]
   results_l0 = {}

   for l0 in l0_values:
       print(f"\nRunning with l0 = {l0}")

       # Create material with new l0
       mat = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=l0)

       # Create new elements
       elems = [GraFEAElement(mesh.nodes[mesh.elements[e]], mat)
                for e in range(mesh.n_elements)]

       # Solve
       solver = StaggeredSolver(mesh, elems, mat, edge_graph, config)
       results_l0[l0] = solver.solve(load_factors, bc_dofs,
                                     lambda lf: bc_values * lf)

   # Compare load-displacement curves
   fig, ax = plt.subplots(figsize=(10, 6))

   for l0, res in results_l0.items():
       displacements = [r.displacement.max() for r in res]
       energies = [r.strain_energy + r.surface_energy for r in res]
       ax.plot(displacements, energies, label=f'l0 = {l0}')

   ax.set_xlabel('Maximum Displacement [mm]')
   ax.set_ylabel('Total Energy [mJ]')
   ax.legend()
   ax.set_title('Length Scale Sensitivity')
   plt.show()

Mesh Refinement Study
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mesh import refine_mesh

   mesh_levels = [mesh]
   for _ in range(2):
       mesh_levels.append(refine_mesh(mesh_levels[-1]))

   results_mesh = {}

   for i, m in enumerate(mesh_levels):
       print(f"\nMesh level {i}: {m.n_elements} elements")

       # Adjust l0 with mesh
       h = np.mean(m.edge_lengths)
       l0 = 2 * h

       mat = IsotropicMaterial(E=210e3, nu=0.3, Gc=2.7, l0=l0)
       eg = EdgeGraph(m)
       elems = [GraFEAElement(m.nodes[m.elements[e]], mat)
                for e in range(m.n_elements)]

       # Setup BCs for this mesh
       bc = create_bc_from_region(m, lambda x, y: y < tol, 'both', 0.0)
       bc_top = create_bc_from_region(m, lambda x, y: y > height - tol, 'y', 0.01)
       bc_dofs_m, bc_vals_m = merge_bcs([bc, bc_top])

       solver = StaggeredSolver(m, elems, mat, eg, config)
       results_mesh[i] = solver.solve(load_factors, bc_dofs_m,
                                      lambda lf: bc_vals_m * lf)

Energy Release Rate Computation
-------------------------------

Compute the energy release rate during crack propagation:

.. code-block:: python

   def compute_energy_release_rate(results, mesh, material):
       """Compute G from energy balance."""
       G_values = []
       crack_lengths = []

       for i in range(1, len(results)):
           # Change in energies
           dE_strain = results[i].strain_energy - results[i-1].strain_energy
           dE_surface = results[i].surface_energy - results[i-1].surface_energy
           dW_ext = results[i].external_work - results[i-1].external_work

           # Estimate crack length change
           from physics import compute_crack_length
           L_prev = compute_crack_length(results[i-1].damage, mesh)
           L_curr = compute_crack_length(results[i].damage, mesh)
           dL = L_curr - L_prev

           if dL > 1e-10:
               # G = -dΠ/dA where Π = E_strain - W_ext
               G = -(dE_strain - dW_ext) / dL
               G_values.append(G)
               crack_lengths.append(L_curr)

       return crack_lengths, G_values

   # Compute and plot
   L, G = compute_energy_release_rate(results, mesh, material)

   fig, ax = plt.subplots()
   ax.plot(L, G, 'o-')
   ax.axhline(material.Gc, color='r', linestyle='--', label=f'Gc = {material.Gc}')
   ax.set_xlabel('Crack Length [mm]')
   ax.set_ylabel('Energy Release Rate [N/mm]')
   ax.legend()
   plt.show()

Adaptive Load Stepping
----------------------

Implement adaptive stepping based on damage increment:

.. code-block:: python

   def adaptive_solve(solver, bc_dofs, bc_values, max_steps=200):
       """Solve with adaptive load stepping."""
       results = []
       load_factor = 0.0
       dlf = 0.05  # Initial step size

       while load_factor < 1.0 and len(results) < max_steps:
           # Try current step
           trial_lf = min(load_factor + dlf, 1.0)

           # Solve single step
           solver.reset_state()  # If available
           res = solver.solve(
               [trial_lf],
               bc_dofs,
               lambda lf: bc_values * lf
           )[0]

           if not res.converged:
               # Reduce step size and retry
               dlf *= 0.5
               print(f"Step failed, reducing dlf to {dlf}")
               continue

           # Check damage increment
           if results:
               dd_max = np.max(np.abs(res.damage - results[-1].damage))
               if dd_max > 0.1:
                   # Too much damage growth, reduce step
                   dlf *= 0.5
                   continue
               elif dd_max < 0.01 and dlf < 0.1:
                   # Too little growth, increase step
                   dlf *= 1.5

           results.append(res)
           load_factor = trial_lf
           print(f"Step {len(results)}: lf={load_factor:.4f}, max_d={res.damage.max():.4f}")

       return results

Parallel Execution
------------------

For parameter studies, use parallel processing:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   import multiprocessing

   def run_simulation(params):
       """Run single simulation with given parameters."""
       l0, Gc = params

       # Setup (recreate everything in worker process)
       mesh = create_rectangle_mesh(1.0, 0.5, 20, 10)
       material = IsotropicMaterial(E=210e9, nu=0.3, Gc=Gc, l0=l0)
       elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                   for e in range(mesh.n_elements)]
       edge_graph = EdgeGraph(mesh)

       # ... setup BCs ...

       solver = StaggeredSolver(mesh, elements, material, edge_graph, config)
       results = solver.solve(load_factors, bc_dofs, lambda lf: bc_values * lf)

       # Return summary
       return {
           'l0': l0,
           'Gc': Gc,
           'max_damage': results[-1].damage.max(),
           'final_energy': results[-1].strain_energy + results[-1].surface_energy
       }

   # Parameter combinations
   params_list = [(l0, Gc) for l0 in [0.01, 0.02, 0.04]
                           for Gc in [2000, 2700, 3500]]

   # Run in parallel
   n_workers = min(len(params_list), multiprocessing.cpu_count())
   with ProcessPoolExecutor(max_workers=n_workers) as executor:
       summaries = list(executor.map(run_simulation, params_list))

   # Analyze results
   for s in summaries:
       print(f"l0={s['l0']}, Gc={s['Gc']}: max_d={s['max_damage']:.3f}")

Next Steps
----------

- :doc:`../theory` - Mathematical details of the formulation
- :doc:`../api/solvers` - Full solver API documentation
- :doc:`../api/physics` - Physics module reference
