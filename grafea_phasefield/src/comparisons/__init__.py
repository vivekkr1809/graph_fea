"""
Comparison Studies Module
=========================

Systematic comparison of GraFEA-PF against:
1. Standard FEM phase-field (node-based damage)
2. Original GraFEA (binary edge damage, no regularization)

Also includes mesh sensitivity and length scale studies.
"""

from .fem_pf_reference import FEMPhasefieldSolver
from .original_grafea import OriginalGraFEASolver
from .comparison_metrics import (
    compute_path_deviation,
    compute_load_displacement_error,
    compute_path_smoothness,
)
from .compare_grafea_vs_fem import run_full_comparison
from .compare_grafea_versions import compare_grafea_versions
from .mesh_sensitivity import run_mesh_sensitivity_study, analyze_mesh_convergence
from .length_scale_study import run_length_scale_study, analyze_length_scale_effects
from .computational_cost import run_scaling_study

__all__ = [
    "FEMPhasefieldSolver",
    "OriginalGraFEASolver",
    "compute_path_deviation",
    "compute_load_displacement_error",
    "compute_path_smoothness",
    "run_full_comparison",
    "compare_grafea_versions",
    "run_mesh_sensitivity_study",
    "analyze_mesh_convergence",
    "run_length_scale_study",
    "analyze_length_scale_effects",
    "run_scaling_study",
]
