"""
GraFEA Phase-Field Framework
============================

Edge-based phase-field fracture framework using Graph-based Finite Element Analysis.

Modules:
    mesh: Triangle mesh with edge-based data structures
    elements: CST and GraFEA element implementations
    physics: Damage models, tension-compression split, surface energy
    assembly: Global matrix assembly and boundary conditions
    solvers: Staggered solver for coupled problem
    postprocess: Visualization and energy tracking
"""

from . import mesh
from . import elements
from . import physics
from . import assembly
from . import solvers
from . import postprocess

__version__ = "0.1.0"
__all__ = ["mesh", "elements", "physics", "assembly", "solvers", "postprocess"]
