"""
Postprocessing Module
=====================

Visualization, energy tracking, and crack extraction.
"""

from .visualization import (
    plot_mesh,
    plot_damage_field,
    plot_displacement_field,
    plot_deformed_mesh,
    plot_load_displacement,
    plot_energy_evolution,
)
from .energy_tracking import EnergyTracker

__all__ = [
    "plot_mesh",
    "plot_damage_field",
    "plot_displacement_field",
    "plot_deformed_mesh",
    "plot_load_displacement",
    "plot_energy_evolution",
    "EnergyTracker",
]
