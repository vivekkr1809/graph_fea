Solvers Module
==============

The solvers module provides the staggered solution algorithm for coupled
displacement-damage problems in phase-field fracture.

.. contents:: Contents
   :local:
   :depth: 2

Configuration
-------------

SolverConfig
~~~~~~~~~~~~

.. autoclass:: solvers.SolverConfig
   :members:
   :undoc-members:
   :show-inheritance:

Result Classes
--------------

LoadStep
~~~~~~~~

.. autoclass:: solvers.LoadStep
   :members:
   :undoc-members:
   :show-inheritance:

Staggered Solver
----------------

StaggeredSolver
~~~~~~~~~~~~~~~

.. autoclass:: solvers.StaggeredSolver
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Basic Solver Setup
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solvers import StaggeredSolver, SolverConfig
   import numpy as np

   # Create solver configuration
   config = SolverConfig(
       tol_u=1e-6,            # Displacement tolerance
       tol_d=1e-6,            # Damage tolerance
       max_stagger_iter=100,  # Max iterations per step
       min_stagger_iter=2,    # Min iterations
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

Running a Simulation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define load factors
   load_factors = np.linspace(0, 1, 50)

   # Boundary condition DOFs and values
   bc_dofs = [...]  # Constrained DOF indices
   bc_values_max = np.array([...])  # Values at load_factor = 1

   # BC value function (called for each load step)
   def bc_value_func(lf):
       return bc_values_max * lf

   # Run simulation
   results = solver.solve(
       load_factors=load_factors,
       bc_dofs=bc_dofs,
       bc_value_func=bc_value_func
   )

   print(f"Completed {len(results)} load steps")

Analyzing Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Iterate through results
   for step in results:
       print(f"Step {step.step}:")
       print(f"  Load factor: {step.load_factor:.3f}")
       print(f"  Converged: {step.converged}")
       print(f"  Iterations: {step.n_iterations}")
       print(f"  Strain energy: {step.strain_energy:.4e}")
       print(f"  Surface energy: {step.surface_energy:.4e}")

   # Check for non-convergence
   non_converged = [s for s in results if not s.converged]
   if non_converged:
       print(f"Warning: {len(non_converged)} steps did not converge")

   # Get final state
   final = results[-1]
   u_final = final.displacement
   d_final = final.damage

Extracting Summary Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get summary from solver
   summary = solver.get_results_summary(results)

   print(f"Total steps: {summary['n_steps']}")
   print(f"Converged steps: {summary['n_converged']}")
   print(f"Max damage: {summary['max_damage']:.3f}")
   print(f"Peak strain energy: {summary['peak_strain_energy']:.4e}")
   print(f"Final surface energy: {summary['final_surface_energy']:.4e}")

Exporting Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export all results to file
   solver.export_results(results, 'simulation_results.npz')

   # Load later
   data = np.load('simulation_results.npz')
   print(data.files)

Custom Load Protocols
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Non-uniform load stepping
   load_factors = np.concatenate([
       np.linspace(0, 0.5, 10),      # Coarse steps initially
       np.linspace(0.5, 0.8, 20),    # Medium steps
       np.linspace(0.8, 1.0, 50),    # Fine steps near failure
   ])

   # Cyclic loading
   up = np.linspace(0, 1, 20)
   down = np.linspace(1, 0.5, 10)
   up2 = np.linspace(0.5, 1.5, 30)
   load_factors = np.concatenate([up, down, up2])

   results = solver.solve(load_factors, bc_dofs, bc_value_func)

Using External Forces
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # External force vector (point load)
   f_ext = np.zeros(2 * mesh.n_nodes)
   top_node = mesh.nodes[:, 1].argmax()
   f_ext[2 * top_node + 1] = -1000  # 1000N downward

   # Force scaling function
   def force_func(lf):
       return f_ext * lf

   results = solver.solve(
       load_factors=load_factors,
       bc_dofs=bc_dofs,
       bc_value_func=bc_value_func,
       external_force_func=force_func
   )

Convergence Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # More tolerant settings for difficult problems
   relaxed_config = SolverConfig(
       tol_u=1e-4,             # Looser tolerance
       tol_d=1e-4,
       max_stagger_iter=200,   # More iterations allowed
       min_stagger_iter=5,     # Ensure good coupling
       verbose=True
   )

   # Finer load stepping
   fine_loads = np.linspace(0, 1, 200)

   # Run with relaxed settings
   solver_relaxed = StaggeredSolver(mesh, elements, material, edge_graph,
                                     relaxed_config)
   results = solver_relaxed.solve(fine_loads, bc_dofs, bc_value_func)

Algorithm Details
-----------------

The staggered solver implements alternating minimization:

1. **Initialize**: Set :math:`u^0 = 0`, :math:`d^0 = 0`, :math:`H^0 = 0`

2. **For each load step** :math:`n`:

   a. Set :math:`u^{n,0} = u^{n-1}`, :math:`d^{n,0} = d^{n-1}`

   b. **Staggered iterations** :math:`k = 0, 1, ...`:

      i. **Mechanical step**: Solve :math:`K(d^{n,k}) u^{n,k+1} = f`

      ii. **History update**: :math:`H^{n,k+1} = \max(H^n, \psi^+(u^{n,k+1}))`

      iii. **Damage step**: Solve for :math:`d^{n,k+1}` from damage equation

      iv. **Check convergence**: :math:`\|u^{n,k+1} - u^{n,k}\| < \text{tol}_u` and
          :math:`\|d^{n,k+1} - d^{n,k}\| < \text{tol}_d`

   c. Set :math:`u^n = u^{n,k+1}`, :math:`d^n = d^{n,k+1}`

3. **Return** results for all load steps

The algorithm ensures:

- Damage irreversibility through history field
- Energy minimization at each step
- Stable crack propagation through load incrementation
