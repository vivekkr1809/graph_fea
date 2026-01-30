"""
Physics Module
==============

Material models, damage evolution, tension-compression split, and surface energy.
"""

from .material import IsotropicMaterial
from .tension_split import spectral_split_2d, spectral_split, simple_edge_split
from .damage import (
    degradation_function,
    degradation_derivative,
    HistoryField,
    compute_driving_force,
)
from .surface_energy import (
    compute_edge_volumes,
    compute_surface_energy,
    assemble_damage_system,
)

__all__ = [
    "IsotropicMaterial",
    "spectral_split_2d",
    "spectral_split",
    "simple_edge_split",
    "degradation_function",
    "degradation_derivative",
    "HistoryField",
    "compute_driving_force",
    "compute_edge_volumes",
    "compute_surface_energy",
    "assemble_damage_system",
]
