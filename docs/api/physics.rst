Physics Module
==============

The physics module provides material models, damage evolution, tension-compression
splitting, and surface energy computation for phase-field fracture.

.. contents:: Contents
   :local:
   :depth: 2

Material Model
--------------

IsotropicMaterial
~~~~~~~~~~~~~~~~~

.. autoclass:: physics.IsotropicMaterial
   :members:
   :undoc-members:
   :show-inheritance:

Material Factory Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: physics.create_steel_material

.. autofunction:: physics.create_aluminum_material

.. autofunction:: physics.create_glass_material

Damage Model
------------

Degradation Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: physics.degradation_function

.. autofunction:: physics.degradation_derivative

HistoryField Class
~~~~~~~~~~~~~~~~~~

.. autoclass:: physics.HistoryField
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Damage Computation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: physics.compute_driving_force

.. autofunction:: physics.compute_driving_force_full

.. autofunction:: physics.compute_damage_energy_release_rate

.. autofunction:: physics.enforce_damage_bounds

.. autofunction:: physics.identify_fully_damaged_edges

.. autofunction:: physics.compute_crack_length

.. autofunction:: physics.compute_dissipated_energy

Tension-Compression Split
-------------------------

.. autofunction:: physics.spectral_split_2d

.. autofunction:: physics.compute_split_energy_miehe

.. autofunction:: physics.spectral_split

.. autofunction:: physics.simple_edge_split

.. autofunction:: physics.volumetric_deviatoric_split

.. autofunction:: physics.verify_energy_conservation

Surface Energy
--------------

.. autofunction:: physics.compute_edge_volumes

.. autofunction:: physics.compute_surface_energy

.. autofunction:: physics.compute_surface_energy_derivative

.. autofunction:: physics.assemble_damage_system

.. autofunction:: physics.assemble_damage_system_simplified

.. autofunction:: physics.compute_surface_energy_rate

.. autofunction:: physics.estimate_l0_from_mesh

Usage Examples
--------------

Material Definition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from physics import IsotropicMaterial, create_steel_material

   # Define custom material
   material = IsotropicMaterial(
       E=210e9,      # Young's modulus [Pa]
       nu=0.3,       # Poisson's ratio
       Gc=2700,      # Critical energy release rate [J/m^2]
       l0=0.01       # Length scale [m]
   )

   # Access derived properties
   print(f"Shear modulus: {material.shear_modulus:.2e}")
   print(f"Bulk modulus: {material.bulk_modulus:.2e}")
   print(f"Critical stress: {material.critical_stress():.2e}")

   # Get constitutive matrix
   C = material.constitutive_matrix(plane_stress=False)

   # Use factory function
   steel = create_steel_material(Gc=2700, l0=0.01)

Degradation Function
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from physics import degradation_function, degradation_derivative
   import numpy as np

   # Damage values
   d = np.linspace(0, 1, 11)

   # Compute degradation
   g = degradation_function(d, order=2)  # (1-d)^2
   g_prime = degradation_derivative(d, order=2)

   print("d\t g(d)\t g'(d)")
   for i in range(len(d)):
       print(f"{d[i]:.1f}\t {g[i]:.3f}\t {g_prime[i]:.3f}")

History Field
~~~~~~~~~~~~~

.. code-block:: python

   from physics import HistoryField

   # Create history field for n_edges
   history = HistoryField(n_edges=100)

   # Update with current driving force
   psi_plus = compute_tensile_energy(...)  # Your function
   history.update(psi_plus)

   # Get current values
   H = history.get_values()

Tension-Compression Split
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from physics import spectral_split_2d, compute_split_energy_miehe
   import numpy as np

   # Strain tensor (Voigt: [eps_xx, eps_yy, eps_xy])
   strain = np.array([0.001, -0.0005, 0.0002])

   # Material parameters
   lmbda = material.lame_lambda
   mu = material.lame_mu

   # Spectral split
   eps_plus, eps_minus = spectral_split_2d(strain)

   # Compute split energies
   psi_plus, psi_minus = compute_split_energy_miehe(strain, lmbda, mu)

   print(f"Tensile energy: {psi_plus:.2e}")
   print(f"Compressive energy: {psi_minus:.2e}")

   # Verify conservation
   from physics import verify_energy_conservation
   is_conserved = verify_energy_conservation(strain, lmbda, mu)
   print(f"Energy conserved: {is_conserved}")

Surface Energy Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from physics import (compute_edge_volumes, compute_surface_energy,
                        assemble_damage_system)

   # Compute edge volumes
   omega = compute_edge_volumes(mesh, thickness=1.0)

   # Current damage field
   damage = np.zeros(mesh.n_edges)
   damage[50:60] = 0.5  # Partial damage in region

   # Compute surface energy
   E_frac = compute_surface_energy(
       damage, mesh, edge_graph, material.Gc, material.l0
   )
   print(f"Surface energy: {E_frac:.2e} J")

   # Assemble damage system for update
   K_d, f_d = assemble_damage_system(
       mesh, edge_graph, material, omega, driving_force=H
   )
