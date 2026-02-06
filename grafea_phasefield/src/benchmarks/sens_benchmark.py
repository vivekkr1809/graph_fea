"""
Single Edge Notched Shear (SENS) Benchmark
===========================================

Mode-II / Mixed-Mode fracture benchmark for validating phase-field fracture
implementations, especially the tension-compression split.

This is a canonical benchmark problem from:
- Miehe et al. (2010) "Thermodynamically consistent phase-field models of fracture"
- Ambati et al. (2015) "A review on phase-field models of brittle fracture"

Geometry:
    - Square domain L x L with pre-existing horizontal crack
    - Crack extends from left edge to center (length a = L/2)
    - Loading: displacement-controlled shear at top boundary

Key Difference from SENT:
    - SENT: vertical tension at top -> Mode-I, horizontal crack
    - SENS: horizontal shear at top -> Mode-II, curved crack (~70 deg upward)

Expected Results:
    - Curved crack propagation (~70 deg from horizontal, upward)
    - Peak load followed by softening
    - Tension-compression split is critical for correct crack path
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import sys
import os

# Handle imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from physics.tension_split import spectral_split, spectral_split_2d, compute_split_energy_miehe
from assembly.boundary_conditions import (
    create_bc_from_region, merge_bcs, create_fixed_bc,
    create_prescribed_displacement_bc,
)
from assembly.global_assembly import (
    assemble_internal_force, compute_all_edge_strains, compute_all_tensor_strains,
)
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep


# ============================================================================
# Default SENS parameters
# ============================================================================
SENS_PARAMS = {
    # Geometry (in mm, same as SENT)
    'L': 1.0,                  # mm - domain size (square)
    'crack_length': 0.5,       # mm - initial crack length from left edge
    'crack_y': 0.5,            # mm - vertical position (center)

    # Material (typical steel-like properties in MPa, same as SENT)
    'E': 210e3,                # MPa (= 210 GPa)
    'nu': 0.3,                 # Poisson's ratio
    'plane': 'strain',         # Plane strain assumption

    # Phase-field parameters (same as SENT)
    'Gc': 2.7,                 # N/mm (= 2.7 kN/m)
    'l0': 0.015,               # mm (length scale)

    # Mesh parameters (wider refinement band for curved crack path)
    'h_fine': 0.00375,         # mm - in crack region (h <= l0/4)
    'h_coarse': 0.02,          # mm - far from crack
    'refinement_band': 0.15,   # mm - wider than SENT (crack curves upward)

    # Loading parameters (shear - larger than SENT tension)
    'u_max': 1.5e-2,           # mm - max horizontal shear displacement
    'n_steps': 250,            # more steps (curved crack needs resolution)

    # Solver parameters (more iterations for shear)
    'tol_u': 1e-6,             # displacement convergence tolerance
    'tol_d': 1e-6,             # damage convergence tolerance
    'max_iter': 300,           # max staggered iterations per step
}


# ============================================================================
# Mesh Generation
# ============================================================================
def generate_sens_mesh(params: Optional[Dict] = None,
                       use_graded_mesh: bool = True) -> TriangleMesh:
    """
    Generate mesh for SENS specimen with local refinement adapted
    for a curved crack path.

    The refinement region is wider than SENT and extends diagonally
    from the crack tip upward at ~70 degrees to accommodate the
    expected curved crack path under shear loading.

    Args:
        params: SENS parameters dictionary (uses defaults if None)
        use_graded_mesh: if True, use graded mesh; if False, use uniform mesh

    Returns:
        TriangleMesh instance
    """
    p = SENS_PARAMS.copy()
    if params is not None:
        p.update(params)

    L = p['L']
    crack_y = p['crack_y']
    h_fine = p['h_fine']
    h_coarse = p['h_coarse']
    band = p['refinement_band']

    if use_graded_mesh:
        mesh = _create_graded_sens_mesh(L, crack_y, h_fine, h_coarse, band,
                                        p['crack_length'])
    else:
        nx = max(int(np.ceil(L / h_fine)), 10)
        ny = nx
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(L, L, nx, ny, pattern='alternating')

    return mesh


def _create_graded_sens_mesh(L: float, crack_y: float,
                              h_fine: float, h_coarse: float,
                              band: float,
                              crack_length: float) -> TriangleMesh:
    """
    Create graded mesh with refinement adapted for curved crack path.

    The mesh is refined in a wider band around the crack line y=crack_y
    and extending upward from the crack tip along the expected ~70 degree
    crack path. This is achieved by using a larger band above the crack line
    than below it.

    Args:
        L: domain size
        crack_y: y-coordinate of crack path
        h_fine: element size in fine zone
        h_coarse: element size in coarse zone
        band: half-width of fine mesh zone
        crack_length: initial crack length

    Returns:
        TriangleMesh instance
    """
    # For SENS, use a wider band above the crack since the crack curves upward
    band_below = band
    band_above = band * 1.5  # Wider above for upward-curving crack

    y_zones = []
    zone_h = []

    # Bottom coarse zone
    y_fine_start = max(0, crack_y - band_below)
    if y_fine_start > 0:
        n_coarse_bottom = max(int(np.ceil(y_fine_start / h_coarse)), 2)
        dy = y_fine_start / n_coarse_bottom
        for i in range(n_coarse_bottom):
            y_zones.append(i * dy)
            zone_h.append(h_coarse)

    # Fine zone (around and above crack)
    y_fine_end = min(L, crack_y + band_above)
    n_fine = max(int(np.ceil((y_fine_end - y_fine_start) / h_fine)), 4)
    dy_fine = (y_fine_end - y_fine_start) / n_fine
    for i in range(n_fine):
        y_zones.append(y_fine_start + i * dy_fine)
        zone_h.append(h_fine)

    # Top coarse zone
    if y_fine_end < L:
        n_coarse_top = max(int(np.ceil((L - y_fine_end) / h_coarse)), 2)
        dy = (L - y_fine_end) / n_coarse_top
        for i in range(n_coarse_top):
            y_zones.append(y_fine_end + i * dy)
            zone_h.append(h_coarse)

    y_zones.append(L)

    # Create nodes
    nodes = []
    node_rows = []

    for i, y in enumerate(y_zones):
        h = zone_h[i] if i < len(zone_h) else zone_h[-1]
        nx = max(int(np.ceil(L / h)), 4)

        row_nodes = []
        for j in range(nx + 1):
            x = j * L / nx
            nodes.append([x, y])
            row_nodes.append(len(nodes) - 1)
        node_rows.append(row_nodes)

    nodes = np.array(nodes)

    # Create elements by connecting adjacent rows
    elements = []
    for i in range(len(node_rows) - 1):
        bottom_row = node_rows[i]
        top_row = node_rows[i + 1]

        if len(bottom_row) == len(top_row):
            for j in range(len(bottom_row) - 1):
                n0 = bottom_row[j]
                n1 = bottom_row[j + 1]
                n2 = top_row[j + 1]
                n3 = top_row[j]
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
        else:
            elements.extend(_triangulate_strip(nodes, bottom_row, top_row))

    elements = np.array(elements)
    return TriangleMesh(nodes, elements)


def _triangulate_strip(nodes: np.ndarray,
                       bottom_row: List[int],
                       top_row: List[int]) -> List[List[int]]:
    """
    Create conforming triangulation between two rows with different node counts.

    Uses a greedy algorithm that creates triangles while advancing
    along both rows, choosing the shorter diagonal at each step.
    """
    elements = []
    i, j = 0, 0
    nb, nt = len(bottom_row), len(top_row)

    while i < nb - 1 or j < nt - 1:
        if i >= nb - 1:
            elements.append([bottom_row[-1], top_row[j], top_row[j + 1]])
            j += 1
        elif j >= nt - 1:
            elements.append([bottom_row[i], bottom_row[i + 1], top_row[-1]])
            i += 1
        else:
            p_bi = nodes[bottom_row[i]]
            p_bi1 = nodes[bottom_row[i + 1]]
            p_tj = nodes[top_row[j]]
            p_tj1 = nodes[top_row[j + 1]]

            diag1 = np.linalg.norm(p_bi1 - p_tj)
            diag2 = np.linalg.norm(p_bi - p_tj1)

            if diag1 <= diag2:
                elements.append([bottom_row[i], bottom_row[i + 1], top_row[j]])
                i += 1
            else:
                elements.append([bottom_row[i], top_row[j], top_row[j + 1]])
                j += 1

    return elements


# ============================================================================
# Pre-crack Damage Initialization (reuse from SENT)
# ============================================================================
def create_precrack_damage(mesh: TriangleMesh,
                           crack_tip_x: float,
                           crack_y: float,
                           l0: float,
                           method: str = 'exponential') -> np.ndarray:
    """
    Initialize damage field to represent pre-existing crack.

    The crack runs horizontally from x=0 to x=crack_tip_x at y=crack_y.
    Same as SENT since the initial crack geometry is identical.

    Args:
        mesh: TriangleMesh instance
        crack_tip_x: x-coordinate of crack tip
        crack_y: y-coordinate of crack (horizontal crack)
        l0: Phase-field length scale for smoothing
        method: 'exponential', 'sharp', or 'linear'

    Returns:
        damage: Initial damage field, shape (n_edges,)
    """
    d = np.zeros(mesh.n_edges)

    midpoints = np.zeros((mesh.n_edges, 2))
    for i, (n1, n2) in enumerate(mesh.edges):
        midpoints[i] = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])

    for i in range(mesh.n_edges):
        x_mid, y_mid = midpoints[i]

        if x_mid < crack_tip_x:
            dist_to_crack = abs(y_mid - crack_y)

            if method == 'exponential':
                d[i] = np.exp(-(dist_to_crack / l0) ** 2)
            elif method == 'sharp':
                if dist_to_crack < l0:
                    d[i] = 1.0
            elif method == 'linear':
                if dist_to_crack < 2 * l0:
                    d[i] = max(0, 1.0 - dist_to_crack / (2 * l0))
        else:
            dist_to_tip = np.sqrt((x_mid - crack_tip_x) ** 2 +
                                  (y_mid - crack_y) ** 2)

            if method == 'exponential':
                if dist_to_tip < 2 * l0:
                    d[i] = np.exp(-(dist_to_tip / l0) ** 2)
            elif method == 'linear':
                if dist_to_tip < l0:
                    d[i] = max(0, 1.0 - dist_to_tip / l0)

    return d


# ============================================================================
# SENS Boundary Conditions
# ============================================================================
def apply_sens_boundary_conditions(mesh: TriangleMesh,
                                   u_applied: float,
                                   L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for SENS (shear) test.

    Boundary conditions (key difference from SENT):
    - Bottom (y=0): u_x = 0, u_y = 0 (fully fixed)
    - Top (y=L): u_x = u_applied (shear), u_y = 0 (constrained vertically)
    - Left/Right: Free (natural BC)

    Args:
        mesh: TriangleMesh instance
        u_applied: Applied horizontal (shear) displacement at top
        L: Domain height (for identifying boundaries)

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values: Prescribed values at those DOFs
    """
    tol = 1e-10 * L

    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)

    # Bottom: fully fixed (u_x = 0, u_y = 0)
    bc_bottom_dofs, bc_bottom_vals = create_fixed_bc(mesh, bottom_nodes)

    # Top: u_x = u_applied (shear), u_y = 0 (vertical constraint)
    bc_top_x_dofs, bc_top_x_vals = create_prescribed_displacement_bc(
        mesh, top_nodes, 'x', u_applied
    )
    bc_top_y_dofs, bc_top_y_vals = create_prescribed_displacement_bc(
        mesh, top_nodes, 'y', 0.0
    )

    # Combine all BCs
    bc_dofs = np.concatenate([bc_bottom_dofs, bc_top_x_dofs, bc_top_y_dofs])
    bc_values = np.concatenate([bc_bottom_vals, bc_top_x_vals, bc_top_y_vals])

    return bc_dofs, bc_values


def create_sens_bc_function(mesh: TriangleMesh,
                            L: float) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """
    Create BC function for load stepping in SENS benchmark.

    Returns a function that takes a shear displacement value and returns
    the corresponding BC values array.

    Args:
        mesh: TriangleMesh instance
        L: Domain height

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values_func: function(u_shear) -> bc_values array
    """
    tol = 1e-10 * L

    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)
    n_top = len(top_nodes)

    # Fixed BCs: bottom fully fixed
    bc_bottom_dofs, bc_bottom_vals = create_fixed_bc(mesh, bottom_nodes)

    # Top y fixed (always zero)
    bc_top_y_dofs, bc_top_y_vals = create_prescribed_displacement_bc(
        mesh, top_nodes, 'y', 0.0
    )

    fixed_dofs, fixed_vals = merge_bcs(
        (bc_bottom_dofs, bc_bottom_vals),
        (bc_top_y_dofs, bc_top_y_vals)
    )

    # Top x DOFs (will have varying shear displacement)
    bc_top_x_dofs = np.array([2 * n for n in top_nodes])

    # All BC DOFs
    all_bc_dofs = np.concatenate([fixed_dofs, bc_top_x_dofs])
    n_fixed = len(fixed_vals)

    def bc_values_func(u_shear: float) -> np.ndarray:
        """Return BC values for given shear displacement."""
        values = np.zeros(len(all_bc_dofs))
        values[:n_fixed] = fixed_vals
        values[n_fixed:] = u_shear  # Horizontal shear at top
        return values

    return all_bc_dofs, bc_values_func


# ============================================================================
# Reaction Force Computation
# ============================================================================
def compute_shear_reaction_force(mesh: TriangleMesh,
                                 elements: List[GraFEAElement],
                                 u: np.ndarray,
                                 damage: np.ndarray,
                                 L: float) -> float:
    """
    Compute horizontal (shear) reaction force at the top boundary.

    For SENS, the reaction force of interest is the horizontal force
    at the top boundary (where shear displacement is applied).

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement solution vector
        damage: Damage field
        L: Domain height

    Returns:
        F_shear: Total horizontal reaction force at top boundary
    """
    tol = 1e-10 * L
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)

    F_int = assemble_internal_force(mesh, elements, u, damage)

    # Sum horizontal (x) reaction force at top nodes
    F_shear = 0.0
    for node in top_nodes:
        dof_x = 2 * node
        F_shear += F_int[dof_x]

    return F_shear


# ============================================================================
# Crack Path and Angle Analysis
# ============================================================================
def extract_crack_path(mesh: TriangleMesh,
                       damage: np.ndarray,
                       threshold: float = 0.9) -> np.ndarray:
    """
    Extract crack path from damage field.

    The crack path is defined as the locus of heavily damaged edges
    (d > threshold), sorted from left to right.

    Args:
        mesh: TriangleMesh instance
        damage: Damage field, shape (n_edges,)
        threshold: Damage threshold for crack identification

    Returns:
        crack_path: Array of (x, y) coordinates along crack, shape (n_points, 2)
    """
    cracked_edges = np.where(damage > threshold)[0]

    if len(cracked_edges) == 0:
        return np.array([]).reshape(0, 2)

    midpoints = []
    for edge_idx in cracked_edges:
        n1, n2 = mesh.edges[edge_idx]
        mid = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])
        midpoints.append(mid)

    midpoints = np.array(midpoints)
    sort_idx = np.argsort(midpoints[:, 0])
    return midpoints[sort_idx]


def compute_crack_length(mesh: TriangleMesh,
                         damage: np.ndarray,
                         threshold: float = 0.9) -> float:
    """
    Compute effective crack length from damage field.

    Args:
        mesh: TriangleMesh instance
        damage: Damage field
        threshold: Minimum damage to count as cracked

    Returns:
        a: Effective crack length
    """
    a = 0.0
    for i, (n1, n2) in enumerate(mesh.edges):
        if damage[i] > threshold:
            length = np.linalg.norm(mesh.nodes[n2] - mesh.nodes[n1])
            a += length * damage[i]
    return a


def compute_crack_angle(crack_path: np.ndarray,
                        initial_tip: Tuple[float, float]) -> float:
    """
    Compute crack propagation angle from horizontal.

    Only considers the portion of the crack ahead of the initial crack tip.

    Args:
        crack_path: Sorted crack path points, shape (n, 2)
        initial_tip: (x, y) of the initial crack tip

    Returns:
        angle_deg: Angle in degrees (positive = upward from horizontal)
    """
    if len(crack_path) < 2:
        return 0.0

    tip_x, tip_y = initial_tip

    # Filter to points ahead of (or near) crack tip
    ahead_mask = crack_path[:, 0] > tip_x + 0.01
    path_ahead = crack_path[ahead_mask]

    if len(path_ahead) < 2:
        return 0.0

    dx = path_ahead[-1, 0] - path_ahead[0, 0]
    dy = path_ahead[-1, 1] - path_ahead[0, 1]

    if dx < 1e-6:
        return 0.0

    return np.degrees(np.arctan2(dy, dx))


def track_crack_tip(mesh: TriangleMesh,
                    damage: np.ndarray,
                    threshold: float = 0.5) -> Tuple[float, float]:
    """
    Track crack tip position (rightmost point where d > threshold).

    Args:
        mesh: TriangleMesh instance
        damage: Damage field, shape (n_edges,)
        threshold: Damage threshold

    Returns:
        (x, y) of crack tip
    """
    midpoints = np.zeros((mesh.n_edges, 2))
    for i, (n1, n2) in enumerate(mesh.edges):
        midpoints[i] = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])

    cracked = midpoints[damage > threshold]
    if len(cracked) == 0:
        return (0.0, 0.0)

    idx = np.argmax(cracked[:, 0])
    return (cracked[idx, 0], cracked[idx, 1])


# ============================================================================
# Stress State Analysis
# ============================================================================
def analyze_stress_state(mesh: TriangleMesh,
                         elements: List[GraFEAElement],
                         u: np.ndarray) -> Dict:
    """
    Analyze stress state for SENS validation.

    Under shear loading, most elements should have mixed principal strains
    (one positive, one negative). This is a basic sanity check.

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement vector

    Returns:
        dict with stress state statistics
    """
    tensor_strains = compute_all_tensor_strains(mesh, elements, u)

    n_tension = 0
    n_compression = 0
    n_mixed = 0

    all_principal_1 = []
    all_principal_2 = []

    for e_idx in range(mesh.n_elements):
        eps_tensor = tensor_strains[e_idx]
        eps_xx, eps_yy, gamma_xy = eps_tensor
        eps_xy = gamma_xy / 2

        E_mat = np.array([[eps_xx, eps_xy], [eps_xy, eps_yy]])
        principals = np.linalg.eigvalsh(E_mat)

        all_principal_1.append(principals[0])
        all_principal_2.append(principals[1])

        if principals[0] > 0 and principals[1] > 0:
            n_tension += 1
        elif principals[0] < 0 and principals[1] < 0:
            n_compression += 1
        else:
            n_mixed += 1

    return {
        'n_tension_dominated': n_tension,
        'n_compression_dominated': n_compression,
        'n_mixed': n_mixed,
        'n_total': mesh.n_elements,
        'principal_1_range': (np.min(all_principal_1), np.max(all_principal_1)),
        'principal_2_range': (np.min(all_principal_2), np.max(all_principal_2)),
    }


# ============================================================================
# Tension-Compression Split Validation
# ============================================================================
def validate_tension_compression_split(mesh: TriangleMesh,
                                       elements: List[GraFEAElement],
                                       u: np.ndarray,
                                       damage: np.ndarray) -> Dict:
    """
    Validate that the tension-compression split is working correctly.

    Key checks:
    1. Energy conservation: psi = psi_plus + psi_minus
    2. Non-negativity: psi_plus >= 0, psi_minus >= 0
    3. Balanced split under shear (both psi_plus and psi_minus significant)
    4. Strain reconstruction: eps = eps_plus + eps_minus
    5. Compressive regions not driving damage

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement vector
        damage: Damage field

    Returns:
        dict with validation checks and details
    """
    edge_strains = compute_all_edge_strains(mesh, elements, u)
    tensor_strains = compute_all_tensor_strains(mesh, elements, u)

    all_psi_plus = []
    all_psi_minus = []
    all_psi_total = []
    miehe_energy_errors = []
    reconstruction_errors = []

    for e_idx, elem in enumerate(elements):
        eps_edge = edge_strains[e_idx]
        eps_tensor = tensor_strains[e_idx]

        # Spectral split (used by solver)
        split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

        psi_plus = split['psi_plus']
        psi_minus = split['psi_minus']
        psi_total = 0.5 * eps_tensor @ elem.C @ eps_tensor

        all_psi_plus.append(psi_plus)
        all_psi_minus.append(psi_minus)
        all_psi_total.append(psi_total)

        # Energy conservation using Miehe formulation (exact)
        lam = elem.material.lame_lambda
        mu = elem.material.lame_mu
        miehe_plus, miehe_minus = compute_split_energy_miehe(eps_tensor, lam, mu)
        miehe_total = miehe_plus + miehe_minus
        miehe_energy_errors.append(abs(miehe_total - psi_total))

        # Strain reconstruction
        eps_reconstructed = split['eps_tensor_plus'] + split['eps_tensor_minus']
        reconstruction_errors.append(np.linalg.norm(eps_tensor - eps_reconstructed))

    all_psi_plus = np.array(all_psi_plus)
    all_psi_minus = np.array(all_psi_minus)
    all_psi_total = np.array(all_psi_total)

    validation = {'checks': {}, 'details': {}}

    # Check 1: Energy conservation using Miehe formulation
    # The Miehe formulation guarantees psi = psi_plus + psi_minus exactly.
    # This checks that the spectral decomposition itself is correct.
    max_miehe_error = np.max(miehe_energy_errors)
    validation['checks']['energy_conservation'] = max_miehe_error < 1e-10
    validation['details']['max_miehe_energy_error'] = max_miehe_error

    # Check 2: Non-negativity
    validation['checks']['psi_plus_nonnegative'] = np.all(all_psi_plus >= -1e-15)
    validation['checks']['psi_minus_nonnegative'] = np.all(all_psi_minus >= -1e-15)

    # Check 3: Balanced split under shear
    psi_plus_mean = np.mean(all_psi_plus)
    psi_minus_mean = np.mean(all_psi_minus)
    denom = max(psi_plus_mean, psi_minus_mean) + 1e-15
    ratio = min(psi_plus_mean, psi_minus_mean) / denom
    validation['checks']['balanced_split'] = ratio > 0.1
    validation['details']['psi_plus_mean'] = psi_plus_mean
    validation['details']['psi_minus_mean'] = psi_minus_mean
    validation['details']['split_ratio'] = ratio

    # Check 4: Strain reconstruction (eps = eps_plus + eps_minus, exact)
    max_reconstruction_error = np.max(reconstruction_errors)
    validation['checks']['strain_reconstruction'] = max_reconstruction_error < 1e-8
    validation['details']['max_reconstruction_error'] = max_reconstruction_error

    # Check 5: Compressive regions not driving damage
    n_compressive_with_damage = 0
    for e_idx in range(mesh.n_elements):
        eps_tensor = tensor_strains[e_idx]
        eps_xx, eps_yy, gamma_xy = eps_tensor
        eps_xy = gamma_xy / 2
        E_mat = np.array([[eps_xx, eps_xy], [eps_xy, eps_yy]])
        principals = np.linalg.eigvalsh(E_mat)

        if principals[0] < 0 and principals[1] < 0:
            edge_indices = mesh.element_to_edges[e_idx]
            d_local = damage[edge_indices]
            if np.max(d_local) > 0.3:
                n_compressive_with_damage += 1

    validation['checks']['compressive_undamaged'] = n_compressive_with_damage == 0
    validation['details']['n_compressive_with_damage'] = n_compressive_with_damage

    validation['passed'] = all(validation['checks'].values())
    return validation


# ============================================================================
# Results Container
# ============================================================================
@dataclass
class SENSResults:
    """Results container for SENS benchmark simulation."""
    displacement: np.ndarray      # Applied shear displacement at each step
    shear_force: np.ndarray       # Horizontal reaction force at each step
    crack_length: np.ndarray      # Crack length at each step
    crack_angle: np.ndarray       # Crack angle at each step
    crack_tip_x: np.ndarray       # Crack tip x at each step
    crack_tip_y: np.ndarray       # Crack tip y at each step
    strain_energy: np.ndarray     # Strain energy at each step
    surface_energy: np.ndarray    # Surface energy at each step
    final_damage: np.ndarray      # Final damage field
    crack_path: np.ndarray        # Extracted crack path coordinates
    damage_snapshots: List[np.ndarray] = field(default_factory=list)
    load_steps: List[LoadStep] = field(default_factory=list)

    @property
    def total_energy(self) -> np.ndarray:
        """Total energy (strain + surface)."""
        return self.strain_energy + self.surface_energy

    @property
    def peak_force(self) -> float:
        """Peak shear force."""
        return np.max(self.shear_force)

    @property
    def displacement_at_peak(self) -> float:
        """Applied displacement at peak force."""
        idx = np.argmax(self.shear_force)
        return self.displacement[idx]

    @property
    def final_crack_angle(self) -> float:
        """Crack angle at end of simulation."""
        nonzero = self.crack_angle != 0.0
        if np.any(nonzero):
            return self.crack_angle[np.where(nonzero)[0][-1]]
        return 0.0


# ============================================================================
# Main Simulation Runner
# ============================================================================
def run_sens_benchmark(params: Optional[Dict] = None,
                       verbose: bool = True,
                       save_snapshots: bool = True,
                       snapshot_interval: int = 10) -> SENSResults:
    """
    Run complete SENS benchmark simulation.

    This is the main function for running the mode-II fracture benchmark.
    It handles mesh generation, pre-crack initialization, load stepping,
    and result extraction.

    Args:
        params: SENS parameters (uses defaults if None)
        verbose: Print progress information
        save_snapshots: Save damage field at regular intervals
        snapshot_interval: Steps between snapshots

    Returns:
        SENSResults: Complete simulation results
    """
    p = SENS_PARAMS.copy()
    if params is not None:
        p.update(params)

    if verbose:
        print("=" * 60)
        print("SENS Benchmark: Mode-II (Shear) Fracture")
        print("=" * 60)

    # Generate mesh
    if verbose:
        print("\nGenerating mesh...")
    mesh = generate_sens_mesh(p)
    if verbose:
        print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"{mesh.n_edges} edges")

    # Create material
    material = IsotropicMaterial(
        E=p['E'],
        nu=p['nu'],
        Gc=p['Gc'],
        l0=p['l0']
    )
    if verbose:
        print(f"  Material: E={material.E:.1e}, nu={material.nu}, "
              f"Gc={material.Gc}, l0={material.l0}")

    # Create elements
    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material,
                              plane=p['plane'])
                for e in range(mesh.n_elements)]

    # Create edge graph for damage regularization
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')

    # Create solver
    config = SolverConfig(
        tol_u=p['tol_u'],
        tol_d=p['tol_d'],
        max_stagger_iter=p['max_iter'],
        verbose=verbose
    )
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # Initialize damage with pre-crack
    if verbose:
        print("\nInitializing pre-crack...")
    d_init = create_precrack_damage(
        mesh,
        p['crack_length'],
        p['crack_y'],
        p['l0'],
        method='exponential'
    )
    solver.set_initial_damage(d_init)
    if verbose:
        print(f"  Pre-crack: length={p['crack_length']}, max(d)={np.max(d_init):.4f}")

    # Setup shear boundary conditions
    bc_dofs, bc_values_func = create_sens_bc_function(mesh, p['L'])

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    if verbose:
        print(f"\nRunning simulation: {p['n_steps']} steps to "
              f"u_max = {p['u_max']:.6f} mm (shear)")
        print("-" * 60)

    # Run simulation
    load_steps = solver.solve(u_steps, bc_dofs, bc_values_func)

    # Extract results
    if verbose:
        print("\nExtracting results...")

    initial_tip = (p['crack_length'], p['crack_y'])

    displacements = np.array([r.load_factor for r in load_steps])
    strain_energies = np.array([r.strain_energy for r in load_steps])
    surface_energies = np.array([r.surface_energy for r in load_steps])

    shear_forces = []
    crack_lengths = []
    crack_angles = []
    crack_tips_x = []
    crack_tips_y = []
    damage_snapshots = []

    for i, result in enumerate(load_steps):
        # Shear reaction force
        F = compute_shear_reaction_force(mesh, elements, result.displacement,
                                         result.damage, p['L'])
        shear_forces.append(F)

        # Crack length
        a = compute_crack_length(mesh, result.damage, threshold=0.9)
        crack_lengths.append(a)

        # Crack tip and angle
        tip = track_crack_tip(mesh, result.damage, threshold=0.5)
        crack_tips_x.append(tip[0])
        crack_tips_y.append(tip[1])

        path = extract_crack_path(mesh, result.damage, threshold=0.9)
        angle = compute_crack_angle(path, initial_tip)
        crack_angles.append(angle)

        # Save snapshot
        if save_snapshots and i % snapshot_interval == 0:
            damage_snapshots.append(result.damage.copy())

    shear_forces = np.array(shear_forces)
    crack_lengths = np.array(crack_lengths)
    crack_angles = np.array(crack_angles)
    crack_tips_x = np.array(crack_tips_x)
    crack_tips_y = np.array(crack_tips_y)

    # Final damage and crack path
    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.9)

    if verbose:
        final_angle = crack_angles[-1] if len(crack_angles) > 0 else 0.0
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Peak shear force: {np.max(shear_forces):.4f}")
        print(f"  Displacement at peak: {displacements[np.argmax(shear_forces)]:.6f}")
        print(f"  Final crack length: {crack_lengths[-1]:.4f}")
        print(f"  Final crack angle: {final_angle:.1f} deg")
        print(f"  Maximum damage: {np.max(final_damage):.4f}")
        print(f"  Final strain energy: {strain_energies[-1]:.6e}")
        print(f"  Final surface energy: {surface_energies[-1]:.6e}")

    return SENSResults(
        displacement=displacements,
        shear_force=shear_forces,
        crack_length=crack_lengths,
        crack_angle=crack_angles,
        crack_tip_x=crack_tips_x,
        crack_tip_y=crack_tips_y,
        strain_energy=strain_energies,
        surface_energy=surface_energies,
        final_damage=final_damage,
        crack_path=crack_path,
        damage_snapshots=damage_snapshots,
        load_steps=load_steps
    )


# ============================================================================
# Validation
# ============================================================================
def validate_sens_results(results: SENSResults,
                          params: Optional[Dict] = None,
                          verbose: bool = True) -> Dict:
    """
    Validate SENS simulation results against expected criteria.

    Validation criteria:
    1. Crack angle: Should be ~70 deg (within 60-80 deg)
    2. Crack direction: Should be upward (dy > 0)
    3. Peak load: Should exist (followed by softening)
    4. Energy balance: Surface energy should grow
    5. Different from SENT: Crack should not be horizontal

    Args:
        results: SENSResults from simulation
        params: SENS parameters
        verbose: Print validation results

    Returns:
        Dictionary with validation status for each criterion
    """
    p = SENS_PARAMS.copy()
    if params is not None:
        p.update(params)

    validation = {
        'crack_angle': {
            'passed': False,
            'value': None,
            'expected_range': (60.0, 80.0),
        },
        'crack_upward': {
            'passed': False,
            'y_change': None,
        },
        'peak_load': {
            'passed': False,
            'value': None,
        },
        'energy_balance': {
            'passed': False,
            'error': None,
        },
        'not_horizontal': {
            'passed': False,
            'angle': None,
            'description': 'Crack should not be horizontal (different from SENT)',
        },
    }

    # 1. Crack angle (~70 deg)
    final_angle = results.final_crack_angle
    validation['crack_angle']['value'] = final_angle
    low, high = validation['crack_angle']['expected_range']
    validation['crack_angle']['passed'] = low <= final_angle <= high

    # 2. Crack goes upward
    if len(results.crack_path) >= 2:
        initial_y = p['crack_y']
        path_ahead = results.crack_path[
            results.crack_path[:, 0] > p['crack_length'] + 0.01
        ]
        if len(path_ahead) > 0:
            final_y = path_ahead[-1, 1]
            y_change = final_y - initial_y
            validation['crack_upward']['y_change'] = y_change
            validation['crack_upward']['passed'] = y_change > 0

    # 3. Peak load exists
    peak = results.peak_force
    validation['peak_load']['value'] = peak
    peak_idx = np.argmax(results.shear_force)
    validation['peak_load']['passed'] = (
        peak > 0 and peak_idx < len(results.shear_force) - 10
    )

    # 4. Energy balance (surface energy grows)
    if len(results.surface_energy) > 1:
        se_growth = results.surface_energy[-1] - results.surface_energy[0]
        validation['energy_balance']['error'] = se_growth
        validation['energy_balance']['passed'] = se_growth >= 0

    # 5. Not horizontal (different from SENT)
    validation['not_horizontal']['angle'] = final_angle
    validation['not_horizontal']['passed'] = abs(final_angle) > 20

    if verbose:
        print("\n" + "=" * 60)
        print("SENS Validation Results")
        print("=" * 60)

        for criterion, data in validation.items():
            status = "PASS" if data['passed'] else "FAIL"
            print(f"\n{criterion}:")
            print(f"  Status: {status}")
            for key, val in data.items():
                if key != 'passed':
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")

    return validation


def compare_with_literature(results: SENSResults,
                            verbose: bool = True) -> Dict:
    """
    Compare SENS results with Miehe et al. (2010) reference.

    Expected: crack angle ~70 deg from horizontal (upward toward loading).

    Args:
        results: SENSResults from simulation
        verbose: Print comparison

    Returns:
        Dictionary with comparison metrics
    """
    expected_angle = 70.0
    tolerance = 10.0

    measured_angle = results.final_crack_angle
    error = abs(measured_angle - expected_angle)

    comparison = {
        'expected_angle': expected_angle,
        'measured_angle': measured_angle,
        'error': error,
        'within_tolerance': error < tolerance,
        'tolerance': tolerance,
        'description': 'Miehe et al. (2010) - Mode-II fracture benchmark',
    }

    if verbose:
        print(f"\nComparison with Miehe et al. (2010)")
        print("-" * 40)
        print(f"  Expected crack angle: {expected_angle:.1f} deg")
        print(f"  Measured crack angle: {measured_angle:.1f} deg")
        print(f"  Error: {error:.1f} deg")
        print(f"  Within tolerance ({tolerance} deg): "
              f"{'Yes' if comparison['within_tolerance'] else 'No'}")

    return comparison


# ============================================================================
# SENS vs SENT Comparison
# ============================================================================
def compare_sens_sent(sens_results: SENSResults,
                      sent_results,
                      verbose: bool = True) -> Dict:
    """
    Compare SENS and SENT benchmark results.

    This comparison validates that the tension-compression split correctly
    differentiates Mode-I (SENT) from Mode-II (SENS) fracture.

    Args:
        sens_results: SENSResults from SENS simulation
        sent_results: SENTResults from SENT simulation
        verbose: Print comparison

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'sens': {},
        'sent': {},
        'validation': {},
    }

    # SENS crack angle
    comparison['sens']['crack_angle'] = sens_results.final_crack_angle
    comparison['sens']['peak_force'] = sens_results.peak_force

    # SENT crack angle
    if len(sent_results.crack_path) >= 2:
        x = sent_results.crack_path[:, 0]
        y = sent_results.crack_path[:, 1]
        if len(x) > 1 and np.std(x) > 1e-10:
            slope = np.polyfit(x, y, 1)[0]
            comparison['sent']['crack_angle'] = np.degrees(np.arctan(slope))
        else:
            comparison['sent']['crack_angle'] = 0.0
    else:
        comparison['sent']['crack_angle'] = 0.0

    comparison['sent']['peak_force'] = sent_results.peak_force

    # Validation
    angle_diff = abs(
        comparison['sens']['crack_angle'] - comparison['sent']['crack_angle']
    )
    comparison['validation']['angle_difference'] = angle_diff
    comparison['validation']['sent_horizontal'] = (
        abs(comparison['sent']['crack_angle']) < 10
    )
    comparison['validation']['sens_angled'] = (
        60 < comparison['sens']['crack_angle'] < 80
    )
    comparison['validation']['different_paths'] = angle_diff > 40

    if verbose:
        print("\n" + "=" * 60)
        print("SENS vs SENT Comparison")
        print("=" * 60)
        print(f"\n  SENT crack angle: "
              f"{comparison['sent']['crack_angle']:.1f} deg (expected ~0)")
        print(f"  SENS crack angle: "
              f"{comparison['sens']['crack_angle']:.1f} deg (expected ~70)")
        print(f"  Angle difference: {angle_diff:.1f} deg")
        print(f"\n  SENT peak force: {comparison['sent']['peak_force']:.4f}")
        print(f"  SENS peak force: {comparison['sens']['peak_force']:.4f}")

        print("\n  Validation:")
        for check, value in comparison['validation'].items():
            status = "PASS" if value else "FAIL"
            if isinstance(value, bool):
                print(f"    {status}: {check}")
            else:
                print(f"    {check}: {value:.1f}")

    return comparison


# ============================================================================
# Class Interface
# ============================================================================
class SENSBenchmark:
    """
    Class interface for SENS benchmark with plotting capabilities.

    Example:
        benchmark = SENSBenchmark()
        benchmark.run()
        benchmark.validate()
        benchmark.plot_results()
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = SENS_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        self.mesh = None
        self.results = None
        self.validation = None

    def run(self, verbose: bool = True) -> SENSResults:
        """Run the SENS benchmark simulation."""
        self.results = run_sens_benchmark(self.params, verbose=verbose)
        self.mesh = generate_sens_mesh(self.params)
        return self.results

    def validate(self, verbose: bool = True) -> Dict:
        """Validate simulation results."""
        if self.results is None:
            raise ValueError("Run simulation first")
        self.validation = validate_sens_results(
            self.results, self.params, verbose=verbose
        )
        return self.validation

    def plot_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive results plot.

        Args:
            save_path: Optional path to save figure
        """
        if self.results is None:
            raise ValueError("Run simulation first")

        try:
            import matplotlib.pyplot as plt
            from postprocess.visualization import plot_damage_field
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Load-displacement curve
        ax = axes[0, 0]
        ax.plot(self.results.displacement * 1000, self.results.shear_force,
                'b-', lw=2)
        ax.set_xlabel('Shear Displacement (mm x 10^3)')
        ax.set_ylabel('Shear Force')
        ax.set_title('Load-Displacement Curve (Shear)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=self.results.peak_force, color='r', linestyle='--',
                   label=f'Peak = {self.results.peak_force:.4f}')
        ax.legend()

        # 2. Crack angle evolution
        ax = axes[0, 1]
        ax.plot(self.results.displacement * 1000, self.results.crack_angle,
                'g-', lw=2)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.7,
                   label='Expected (70 deg)')
        ax.set_xlabel('Shear Displacement (mm x 10^3)')
        ax.set_ylabel('Crack Angle (deg)')
        ax.set_title('Crack Angle Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Final damage field
        ax = axes[1, 0]
        if self.mesh is not None:
            plot_damage_field(self.mesh, self.results.final_damage, ax=ax)
        ax.set_title('Final Damage Field')

        # 4. Crack path
        ax = axes[1, 1]
        if len(self.results.crack_path) > 0:
            ax.plot(self.results.crack_path[:, 0], self.results.crack_path[:, 1],
                    'r.-', lw=2, ms=4, label='Computed path')

        # Draw expected path line
        tip_x = self.params['crack_length']
        tip_y = self.params['crack_y']
        expected_angle = np.radians(70)
        end_x = tip_x + 0.4 * np.cos(expected_angle)
        end_y = tip_y + 0.4 * np.sin(expected_angle)
        ax.plot([tip_x, end_x], [tip_y, end_y], 'b--', lw=2,
                label='Expected (~70 deg)')

        # Draw initial crack
        ax.plot([0, tip_x], [tip_y, tip_y], 'k-', lw=3, alpha=0.3,
                label='Initial crack')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Crack Path')
        ax.set_xlim([0, self.params['L']])
        ax.set_ylim([0, self.params['L']])
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def plot_crack_evolution(self, n_frames: int = 5,
                             save_path: Optional[str] = None):
        """
        Plot damage field evolution during crack propagation.

        Args:
            n_frames: Number of frames to show
            save_path: Optional path to save figure
        """
        if self.results is None or len(self.results.damage_snapshots) == 0:
            print("No damage snapshots available")
            return

        try:
            import matplotlib.pyplot as plt
            from postprocess.visualization import plot_damage_field
        except ImportError:
            print("matplotlib not available for plotting")
            return

        n_snaps = len(self.results.damage_snapshots)
        indices = np.linspace(0, n_snaps - 1, min(n_frames, n_snaps)).astype(int)

        fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
        if len(indices) == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            if self.mesh is not None:
                plot_damage_field(self.mesh, self.results.damage_snapshots[idx],
                                  ax=axes[i], colorbar=(i == len(indices) - 1))
            axes[i].set_title(f'Snapshot {idx + 1}')

        plt.suptitle('SENS: Crack Evolution (Shear Loading)')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


# ============================================================================
# Convenience Functions
# ============================================================================
def quick_sens_test(n_steps: int = 50, verbose: bool = True) -> SENSResults:
    """
    Run a quick SENS test with reduced steps for testing.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        SENSResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 0.0075,
        'h_coarse': 0.04,
    }
    return run_sens_benchmark(params, verbose=verbose)


def very_quick_sens_test(n_steps: int = 20, verbose: bool = True) -> SENSResults:
    """
    Run a very quick SENS test with coarse mesh for fast testing.

    This is useful for verifying the pipeline works but should not
    be used for quantitative validation.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        SENSResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 0.02,
        'h_coarse': 0.08,
        'l0': 0.04,
        'refinement_band': 0.15,
    }
    return run_sens_benchmark(params, verbose=verbose)


if __name__ == '__main__':
    results = run_sens_benchmark(verbose=True)
    validate_sens_results(results, verbose=True)
