Postprocessing Module
=====================

The postprocessing module provides visualization functions and energy tracking
utilities for analyzing simulation results.

.. contents:: Contents
   :local:
   :depth: 2

Visualization
-------------

Mesh Plotting
~~~~~~~~~~~~~

.. autofunction:: postprocess.plot_mesh

Field Visualization
~~~~~~~~~~~~~~~~~~~

.. autofunction:: postprocess.plot_damage_field

.. autofunction:: postprocess.plot_displacement_field

.. autofunction:: postprocess.plot_deformed_mesh

Results Plotting
~~~~~~~~~~~~~~~~

.. autofunction:: postprocess.plot_load_displacement

.. autofunction:: postprocess.plot_energy_evolution

Energy Tracking
---------------

EnergyRecord
~~~~~~~~~~~~

.. autoclass:: postprocess.EnergyRecord
   :members:
   :undoc-members:
   :show-inheritance:

EnergyTracker
~~~~~~~~~~~~~

.. autoclass:: postprocess.EnergyTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Plotting the Mesh
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_mesh
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   # Basic mesh plot
   plot_mesh(mesh, ax=axes[0])
   axes[0].set_title('Basic Mesh')

   # With node labels
   plot_mesh(mesh, ax=axes[1], show_nodes=True, node_labels=True)
   axes[1].set_title('With Node Labels')

   # With edge indices
   plot_mesh(mesh, ax=axes[2], show_edges=True)
   axes[2].set_title('With Edges')

   plt.tight_layout()
   plt.show()

Damage Field Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_damage_field
   import matplotlib.pyplot as plt

   # Get final damage from results
   final = results[-1]

   fig, ax = plt.subplots(figsize=(10, 6))
   plot_damage_field(mesh, final.damage, ax=ax, cmap='hot_r')
   ax.set_title('Damage Field')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')

   # Add colorbar
   sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(0, 1))
   plt.colorbar(sm, ax=ax, label='Damage [-]')
   plt.show()

Deformed Mesh
~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_deformed_mesh
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Original mesh
   plot_mesh(mesh, ax=axes[0])
   axes[0].set_title('Undeformed')

   # Deformed mesh (scaled for visibility)
   plot_deformed_mesh(mesh, final.displacement, scale=50, ax=axes[1])
   axes[1].set_title('Deformed (50x scale)')

   plt.tight_layout()
   plt.show()

Displacement Field
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_displacement_field
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))

   # Plot displacement vectors
   plot_displacement_field(mesh, final.displacement, ax=ax,
                          scale=1000, color='blue')
   ax.set_title('Displacement Vectors')
   plt.show()

Energy Evolution
~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_energy_evolution
   import matplotlib.pyplot as plt

   # Plot all energy components
   fig, ax = plt.subplots(figsize=(10, 6))
   plot_energy_evolution(results, ax=ax)
   ax.set_xlabel('Load Factor')
   ax.set_ylabel('Energy [J]')
   ax.legend()
   plt.show()

Load-Displacement Curve
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import plot_load_displacement
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 6))
   plot_load_displacement(results, ax=ax)
   ax.set_xlabel('Displacement [m]')
   ax.set_ylabel('Reaction Force [N]')
   ax.set_title('Load-Displacement Response')
   plt.show()

Energy Tracking
~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocess import EnergyTracker, EnergyRecord

   # Create tracker and populate from results
   tracker = EnergyTracker()
   tracker.from_results(results)

   # Or add records manually
   tracker.add_record(EnergyRecord(
       step=0,
       load_factor=0.0,
       strain_energy=0.0,
       surface_energy=0.0,
       external_work=0.0
   ))

   # Get summary statistics
   summary = tracker.get_energy_summary()
   print(f"Peak strain energy: {summary['max_strain_energy']:.4e}")
   print(f"Final surface energy: {summary['final_surface_energy']:.4e}")
   print(f"Total dissipation: {summary['total_dissipation']:.4e}")

   # Export to CSV
   tracker.export_csv('energy_history.csv')

   # Quick plot
   tracker.plot_energy()
   plt.show()

Creating Publication-Quality Figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from postprocess import plot_damage_field, plot_energy_evolution

   # Set up figure with multiple panels
   fig = plt.figure(figsize=(12, 10))

   # Damage at different load steps
   steps_to_plot = [10, 25, 40, -1]  # Selected steps
   for i, step_idx in enumerate(steps_to_plot):
       ax = fig.add_subplot(2, 2, i + 1)
       result = results[step_idx]
       plot_damage_field(mesh, result.damage, ax=ax, cmap='hot_r')
       ax.set_title(f'Load Factor = {result.load_factor:.2f}')
       ax.set_aspect('equal')

   plt.tight_layout()
   plt.savefig('damage_evolution.png', dpi=300, bbox_inches='tight')
   plt.show()

   # Energy plot with styling
   fig, ax = plt.subplots(figsize=(8, 6))
   plot_energy_evolution(results, ax=ax)

   ax.set_xlabel('Load Factor [-]', fontsize=12)
   ax.set_ylabel('Energy [J]', fontsize=12)
   ax.legend(fontsize=10)
   ax.grid(True, alpha=0.3)

   plt.savefig('energy_evolution.pdf', bbox_inches='tight')
   plt.show()

Animation
~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib.animation import FuncAnimation
   from postprocess import plot_damage_field

   fig, ax = plt.subplots(figsize=(10, 6))

   def update(frame):
       ax.clear()
       result = results[frame]
       plot_damage_field(mesh, result.damage, ax=ax, cmap='hot_r')
       ax.set_title(f'Load Factor: {result.load_factor:.3f}')
       ax.set_aspect('equal')
       return ax.collections

   anim = FuncAnimation(fig, update, frames=len(results), interval=100)
   anim.save('damage_animation.mp4', writer='ffmpeg', fps=10)
   plt.show()
