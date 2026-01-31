User Guide
==========

This guide provides detailed information on using the GraFEA Phase-Field Framework
for fracture mechanics simulations.

Mesh Generation and Handling
----------------------------

The framework provides several mesh generation functions and supports mesh import.

Built-in Mesh Generators
~~~~~~~~~~~~~~~~~~~~~~~~

**Rectangular meshes:**

.. code-block:: python

   from mesh import create_rectangle_mesh, create_square_mesh

   # Rectangle mesh with specified dimensions and element counts
   mesh = create_rectangle_mesh(width=2.0, height=1.0, nx=40, ny=20)

   # Square mesh (convenience function)
   mesh = create_square_mesh(size=1.0, n=20)  # 20x20 elements

**Pre-notched specimens:**

.. code-block:: python

   from mesh import create_notched_rectangle

   # Single Edge Notched Tension (SENT) specimen
   mesh = create_notched_rectangle(
       width=1.0,
       height=0.5,
       notch_length=0.5,   # Half the width
       nx=50,
       ny=25
   )

**Test meshes:**

.. code-block:: python

   from mesh import create_single_element, create_two_element_patch

   # Single triangle for testing
   mesh = create_single_element()

   # Two triangles sharing an edge
   mesh = create_two_element_patch()

Mesh Refinement
~~~~~~~~~~~~~~~

Refine an existing mesh by subdividing each triangle into four:

.. code-block:: python

   from mesh import refine_mesh

   coarse_mesh = create_square_mesh(1.0, 5)
   fine_mesh = refine_mesh(coarse_mesh)  # 4x more elements

Mesh Properties
~~~~~~~~~~~~~~~

The ``TriangleMesh`` class provides access to geometric quantities:

.. code-block:: python

   # Basic properties
   print(f"Nodes: {mesh.n_nodes}")
   print(f"Elements: {mesh.n_elements}")
   print(f"Edges: {mesh.n_edges}")

   # Geometric quantities (computed on demand)
   areas = mesh.element_areas      # Element areas
   lengths = mesh.edge_lengths     # Edge lengths
   midpoints = mesh.edge_midpoints # Edge midpoint coordinates

   # Boundary information
   boundary_edges = mesh.boundary_edges   # Indices of boundary edges
   boundary_nodes = mesh.boundary_nodes   # Indices of boundary nodes

Material Definition
-------------------

The ``IsotropicMaterial`` class encapsulates elastic and fracture properties.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from physics import IsotropicMaterial

   # Define material with all parameters
   steel = IsotropicMaterial(
       E=210e9,      # Young's modulus [Pa]
       nu=0.3,       # Poisson's ratio [-]
       Gc=2700,      # Critical energy release rate [J/m^2]
       l0=0.01       # Phase-field length scale [m]
   )

Derived Properties
~~~~~~~~~~~~~~~~~~

The material class computes derived quantities:

.. code-block:: python

   # Lamé parameters
   lambda_ = steel.lame_lambda
   mu = steel.lame_mu

   # Bulk and shear moduli
   K = steel.bulk_modulus
   G = steel.shear_modulus

   # Wave speeds (for dynamics)
   c_p, c_s = steel.wave_speeds()

   # Critical values
   sigma_c = steel.critical_stress()
   psi_c = steel.critical_strain_energy_density()

Constitutive Matrix
~~~~~~~~~~~~~~~~~~~

Get the constitutive matrix for plane strain or plane stress:

.. code-block:: python

   C_strain = steel.constitutive_matrix(plane_stress=False)  # Default
   C_stress = steel.constitutive_matrix(plane_stress=True)

Pre-defined Materials
~~~~~~~~~~~~~~~~~~~~~

Convenience functions for common materials:

.. code-block:: python

   from physics import create_steel_material, create_aluminum_material, create_glass_material

   steel = create_steel_material(Gc=2700, l0=0.01)
   aluminum = create_aluminum_material(Gc=1000, l0=0.01)
   glass = create_glass_material(Gc=10, l0=0.001)

Choosing the Length Scale
~~~~~~~~~~~~~~~~~~~~~~~~~

The phase-field length scale :math:`l_0` controls the width of the damage zone:

- Smaller :math:`l_0` → sharper cracks (requires finer mesh)
- Larger :math:`l_0` → more diffuse cracks

**Rule of thumb**: :math:`l_0 \approx 2h` to :math:`4h` where :math:`h` is the mesh size.

.. code-block:: python

   from physics import estimate_l0_from_mesh

   # Automatically estimate l0 from mesh
   l0 = estimate_l0_from_mesh(mesh)
   print(f"Recommended l0: {l0}")

Element Types
-------------

The framework provides two element types.

CSTElement (Standard FEM)
~~~~~~~~~~~~~~~~~~~~~~~~~

The Constant Strain Triangle for standard finite element analysis:

.. code-block:: python

   from elements import CSTElement

   # Create element
   node_coords = mesh.nodes[mesh.elements[0]]  # (3, 2) array
   elem = CSTElement(node_coords, material)

   # Element stiffness matrix
   K = elem.K  # 6x6 matrix

   # Compute strain and stress
   u_elem = displacement[mesh.elements[0]].flatten()  # 6-DOF vector
   strain = elem.compute_strain(u_elem)
   stress = elem.compute_stress(u_elem)

GraFEAElement (Edge-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The novel edge-based element for phase-field fracture:

.. code-block:: python

   from elements import GraFEAElement

   elem = GraFEAElement(node_coords, material)

   # Edge information
   L = elem.L          # Reference edge lengths
   phi = elem.phi      # Edge angles with x-axis
   T = elem.T          # Tensor-to-edge transformation
   A = elem.A          # Edge stiffness matrix

   # Compute edge strains
   edge_strains = elem.compute_edge_strains(u_elem)

   # Or directly from deformed configuration
   edge_strains_direct = elem.compute_edge_strains_direct(u_elem)

   # Stiffness with damage
   d_edges = [0.1, 0.0, 0.2]  # Damage on element's three edges
   K_damaged = elem.compute_stiffness_damaged(d_edges)

Boundary Conditions
-------------------

Flexible boundary condition specification using regions or node indices.

Region-Based BCs
~~~~~~~~~~~~~~~~

Define BCs using spatial predicates:

.. code-block:: python

   from assembly import create_bc_from_region

   # Fix bottom edge
   bc_bottom = create_bc_from_region(
       mesh,
       region=lambda x, y: y < 1e-10,  # Predicate function
       component='both',                 # 'x', 'y', or 'both'
       value=0.0
   )

   # Prescribe top displacement
   bc_top = create_bc_from_region(
       mesh,
       region=lambda x, y: y > 0.99,
       component='y',
       value=0.001
   )

Merging BCs
~~~~~~~~~~~

Combine multiple boundary conditions:

.. code-block:: python

   from assembly import merge_bcs

   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top])

Roller and Fixed BCs
~~~~~~~~~~~~~~~~~~~~

Convenience functions for common BC types:

.. code-block:: python

   from assembly import create_fixed_bc, create_roller_bc

   # Fix specific nodes completely
   bc_fixed = create_fixed_bc(mesh, node_indices=[0, 1, 2])

   # Roller BC (free in one direction)
   bc_roller = create_roller_bc(mesh, node_indices=[0, 1], direction='x')

Solver Configuration
--------------------

The staggered solver supports various configuration options.

SolverConfig Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solvers import SolverConfig

   config = SolverConfig(
       tol_u=1e-6,            # Displacement convergence tolerance
       tol_d=1e-6,            # Damage convergence tolerance
       max_stagger_iter=100,  # Maximum iterations per load step
       min_stagger_iter=2,    # Minimum iterations (ensures coupling)
       verbose=True,          # Print convergence information
       damage_bounds=(0.0, 1.0)  # Enforce damage bounds
   )

Running Simulations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from solvers import StaggeredSolver
   import numpy as np

   # Create solver
   solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

   # Define load schedule
   load_factors = np.concatenate([
       np.linspace(0, 0.8, 10),   # Fast initial loading
       np.linspace(0.8, 1.0, 20)  # Finer steps near fracture
   ])

   # BC value function (called for each load step)
   def bc_value_func(lf):
       return bc_values * lf

   # Solve
   results = solver.solve(
       load_factors=load_factors,
       bc_dofs=bc_dofs,
       bc_value_func=bc_value_func
   )

Accessing Results
~~~~~~~~~~~~~~~~~

Each load step returns a ``LoadStep`` object:

.. code-block:: python

   for result in results:
       print(f"Step {result.step}: load={result.load_factor:.3f}")
       print(f"  Strain energy: {result.strain_energy:.2e}")
       print(f"  Surface energy: {result.surface_energy:.2e}")
       print(f"  Converged: {result.converged} ({result.n_iterations} iterations)")

   # Final state
   final = results[-1]
   u_final = final.displacement  # (n_nodes, 2) array
   d_final = final.damage        # (n_edges,) array

Postprocessing and Visualization
--------------------------------

Damage Field Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_damage_field
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   plot_damage_field(mesh, damage, ax=ax, cmap='hot_r')
   ax.set_title('Damage Field')
   plt.colorbar(ax.collections[0], label='Damage')
   plt.show()

Deformed Mesh
~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_deformed_mesh

   plot_deformed_mesh(mesh, displacement, scale=10)
   plt.title('Deformed Mesh (10x scale)')
   plt.show()

Energy Evolution
~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_energy_evolution

   plot_energy_evolution(results)
   plt.show()

Load-Displacement Curves
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_load_displacement

   # Extract reaction force at a node
   plot_load_displacement(results)
   plt.show()

Energy Tracking
~~~~~~~~~~~~~~~

Track and export energy data:

.. code-block:: python

   from postprocess import EnergyTracker

   tracker = EnergyTracker()
   tracker.from_results(results)

   # Summary statistics
   summary = tracker.get_energy_summary()
   print(f"Peak strain energy: {summary['max_strain_energy']:.2e}")
   print(f"Total dissipation: {summary['total_dissipation']:.2e}")

   # Export to CSV
   tracker.export_csv('energy_history.csv')

Tips for Successful Simulations
-------------------------------

Mesh Quality
~~~~~~~~~~~~

- Use meshes where elements are close to equilateral
- Avoid highly distorted elements (aspect ratio > 3)
- Refine mesh in regions where cracks are expected

Length Scale Selection
~~~~~~~~~~~~~~~~~~~~~~

- Start with :math:`l_0 = 2h` (twice the mesh size)
- Increase if solution is too brittle or mesh-dependent
- Decrease for sharper cracks (requires finer mesh)

Load Stepping
~~~~~~~~~~~~~

- Use more load steps during crack propagation
- Consider adaptive stepping based on damage increment
- Start with coarse steps, refine where needed

Convergence Issues
~~~~~~~~~~~~~~~~~~

If the solver doesn't converge:

1. Increase ``max_stagger_iter``
2. Reduce load step size
3. Check boundary conditions for conflicts
4. Verify material parameters are physically reasonable
5. Ensure mesh quality is adequate

Performance
~~~~~~~~~~~

- The solver uses sparse matrices for efficiency
- Larger meshes may require more memory
- Consider using iterative solvers for very large problems
