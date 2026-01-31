Assembly Module
===============

The assembly module provides functions for global matrix assembly and boundary
condition application.

.. contents:: Contents
   :local:
   :depth: 2

Global Assembly
---------------

Stiffness Matrix Assembly
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.assemble_global_stiffness

.. autofunction:: assembly.assemble_global_stiffness_undamaged

Force Assembly
~~~~~~~~~~~~~~

.. autofunction:: assembly.assemble_internal_force

Strain Computation
~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.compute_all_edge_strains

.. autofunction:: assembly.compute_all_tensor_strains

Energy Computation
~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.compute_strain_energy

Damage Field Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.compute_nodal_damage

.. autofunction:: assembly.compute_element_damage

Boundary Conditions
-------------------

Applying Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.apply_dirichlet_bc

Creating Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: assembly.create_bc_from_region

.. autofunction:: assembly.create_fixed_bc

.. autofunction:: assembly.create_roller_bc

.. autofunction:: assembly.merge_bcs

BoundaryConditionManager
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: assembly.BoundaryConditionManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Global Stiffness Assembly
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import (assemble_global_stiffness,
                         assemble_global_stiffness_undamaged)
   import numpy as np

   # Assuming mesh, elements, and damage are defined

   # Undamaged assembly
   K_undamaged = assemble_global_stiffness_undamaged(mesh, elements)

   # Damaged assembly
   damage = np.zeros(mesh.n_edges)
   K_damaged = assemble_global_stiffness(mesh, elements, damage)

   # Compare stiffness reduction
   print(f"Undamaged: cond(K) = {np.linalg.cond(K_undamaged.toarray()):.2e}")
   print(f"Damaged: cond(K) = {np.linalg.cond(K_damaged.toarray()):.2e}")

Boundary Condition Setup
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import (create_bc_from_region, create_fixed_bc,
                         create_roller_bc, merge_bcs, apply_dirichlet_bc)
   import numpy as np

   # Fix bottom edge (y = 0)
   bc_bottom = create_bc_from_region(
       mesh,
       region=lambda x, y: y < 1e-10,
       component='both',
       value=0.0
   )

   # Prescribe displacement on top edge
   bc_top = create_bc_from_region(
       mesh,
       region=lambda x, y: y > mesh.nodes[:, 1].max() - 1e-10,
       component='y',
       value=0.01  # 10mm displacement
   )

   # Left edge roller (free in y)
   left_nodes = mesh.get_nodes_in_region(lambda x, y: x < 1e-10)
   bc_left = create_roller_bc(mesh, left_nodes, direction='y')

   # Merge all BCs
   bc_dofs, bc_values = merge_bcs([bc_bottom, bc_top, bc_left])

   print(f"Total constrained DOFs: {len(bc_dofs)}")

Applying BCs to System
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import apply_dirichlet_bc

   # Assemble global system
   K = assemble_global_stiffness(mesh, elements, damage)
   f = np.zeros(2 * mesh.n_nodes)

   # Apply boundary conditions (elimination method)
   K_bc, f_bc = apply_dirichlet_bc(K, f, bc_dofs, bc_values, method='elimination')

   # Solve
   from scipy.sparse.linalg import spsolve
   u = spsolve(K_bc, f_bc)

   # Alternative: penalty method
   K_pen, f_pen = apply_dirichlet_bc(K, f, bc_dofs, bc_values, method='penalty')

Computing Strains
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import compute_all_edge_strains, compute_all_tensor_strains

   # Displacement solution
   u = ...  # (n_nodes, 2) array

   # Compute edge strains for all elements
   edge_strains = compute_all_edge_strains(mesh, elements, u)
   # Returns (n_elements, 3) array

   # Compute tensor strains
   tensor_strains = compute_all_tensor_strains(mesh, elements, u)
   # Returns (n_elements, 3) array in Voigt notation

Energy Computation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import compute_strain_energy

   # Total strain energy
   energy = compute_strain_energy(mesh, elements, u, damage)
   print(f"Strain energy: {energy:.4e} J")

Damage Field Visualization Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from assembly import compute_nodal_damage, compute_element_damage

   # Average damage to nodes (for plotting)
   nodal_damage = compute_nodal_damage(mesh, damage)

   # Average damage per element
   elem_damage = compute_element_damage(mesh, damage)

   # Use for visualization
   import matplotlib.pyplot as plt
   import matplotlib.tri as tri

   triangulation = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                                     mesh.elements)
   plt.tripcolor(triangulation, facecolors=elem_damage, cmap='hot_r')
   plt.colorbar(label='Element Damage')
   plt.show()
