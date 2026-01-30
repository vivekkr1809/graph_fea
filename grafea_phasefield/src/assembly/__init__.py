"""
Assembly Module
===============

Global matrix assembly and boundary condition application.
"""

from .global_assembly import (
    assemble_global_stiffness,
    assemble_global_stiffness_undamaged,
    assemble_internal_force,
    compute_all_edge_strains,
    compute_all_tensor_strains,
    compute_strain_energy,
    compute_nodal_damage,
    compute_element_damage,
)
from .boundary_conditions import (
    apply_dirichlet_bc,
    create_bc_from_region,
    create_fixed_bc,
    create_roller_bc,
    merge_bcs,
    BoundaryConditionManager,
)

__all__ = [
    "assemble_global_stiffness",
    "assemble_global_stiffness_undamaged",
    "assemble_internal_force",
    "compute_all_edge_strains",
    "compute_all_tensor_strains",
    "compute_strain_energy",
    "compute_nodal_damage",
    "compute_element_damage",
    "apply_dirichlet_bc",
    "create_bc_from_region",
    "create_fixed_bc",
    "create_roller_bc",
    "merge_bcs",
    "BoundaryConditionManager",
]
