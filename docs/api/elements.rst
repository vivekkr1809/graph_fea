Elements Module
===============

The elements module provides finite element implementations for the GraFEA framework,
including both standard CST elements and the novel edge-based GraFEA elements.

.. contents:: Contents
   :local:
   :depth: 2

CSTElement Class
----------------

The Constant Strain Triangle (CST) element for standard finite element analysis.

.. autoclass:: elements.CSTElement
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

GraFEAElement Class
-------------------

The edge-based GraFEA element with damage-dependent stiffness.

.. autoclass:: elements.GraFEAElement
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Standard CST Element
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from elements import CSTElement
   from physics import IsotropicMaterial
   import numpy as np

   # Define material
   material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

   # Define element nodes (counter-clockwise)
   nodes = np.array([
       [0.0, 0.0],
       [1.0, 0.0],
       [0.5, 0.866]
   ])

   # Create element
   elem = CSTElement(nodes, material)

   # Get stiffness matrix
   K = elem.K  # 6x6 matrix

   # Compute strain from displacement
   u = np.array([0, 0, 0.001, 0, 0.0005, 0.001])  # 6 DOFs
   strain = elem.compute_strain(u)
   stress = elem.compute_stress(u)

   # Strain energy
   psi = elem.compute_strain_energy_density(u)

GraFEA Element
~~~~~~~~~~~~~~

.. code-block:: python

   from elements import GraFEAElement

   # Create GraFEA element
   elem = GraFEAElement(nodes, material)

   # Access edge properties
   print(f"Edge lengths: {elem.L}")
   print(f"Edge angles: {elem.phi}")

   # Transformation matrix (tensor to edge)
   T = elem.T

   # Edge stiffness matrix
   A = elem.A

   # Compute edge strains
   edge_strains = elem.compute_edge_strains(u)

   # Verify equivalence with tensor formulation
   is_equivalent = elem.verify_energy_equivalence(u)
   print(f"Energy equivalence: {is_equivalent}")

Element with Damage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Edge damage values (one per edge)
   d = np.array([0.0, 0.5, 0.0])  # Middle edge 50% damaged

   # Damaged stiffness matrix
   K_damaged = elem.compute_stiffness_damaged(d)

   # Stiffness with tension-compression split
   K_split = elem.compute_stiffness_with_split(d, u)

   # Internal force vector
   f_int = elem.compute_internal_force_grafea(u, d)

Comparing Formulations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create both element types
   cst = CSTElement(nodes, material)
   grafea = GraFEAElement(nodes, material)

   # Compare undamaged stiffness
   K_cst = cst.K
   K_grafea = grafea.compute_stiffness_undamaged()

   # Should be identical (within numerical tolerance)
   diff = np.linalg.norm(K_cst - K_grafea)
   print(f"Stiffness difference: {diff}")

   # Compare strain energy
   psi_cst = cst.compute_strain_energy_density(u)
   psi_grafea = grafea.strain_energy_density(u)
   print(f"Energy difference: {abs(psi_cst - psi_grafea)}")
