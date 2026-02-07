"""
L-Shaped Panel Benchmark
========================

Critical test of crack NUCLEATION capability.

This benchmark is fundamentally different from SENT/SENS/TPB:
    - NO pre-existing crack or notch
    - Crack must nucleate spontaneously at the inner corner stress concentration
    - Original GraFEA CANNOT do this (requires pre-crack)
    - Phase-field enables nucleation via energy minimization

If this benchmark succeeds, it demonstrates a major advantage of the
edge-based phase-field approach over original GraFEA.

Geometry:
    - L-shaped domain: 250 x 250 mm outer with 150 x 150 mm inner cutout
    - Inner corner at (100, 100) mm
    - Two legs: vertical (100 mm wide) and horizontal (100 mm wide)

Loading:
    - Bottom horizontal edge: fully fixed (u_x = u_y = 0)
    - Top of vertical leg: applied vertical displacement

Expected Results:
    - Elastic deformation until critical load
    - Damage nucleation at inner corner
    - Crack propagation toward outer boundary (~45 degrees)

References:
    - Winkler (2001) "Traglastuntersuchungen von unbewehrten und
      bewehrten Betonstrukturen auf der Grundlage eines objektiven
      Werkstoffgesetzes fur Beton"
    - Ambati et al. (2015) Phase-field benchmark
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
from physics.tension_split import spectral_split
from assembly.boundary_conditions import (
    create_bc_from_region, merge_bcs, create_fixed_bc,
    create_prescribed_displacement_bc,
)
from assembly.global_assembly import (
    assemble_internal_force, compute_all_edge_strains,
    compute_all_tensor_strains,
)
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep


# ============================================================================
# Default L-Shaped Panel parameters
# ============================================================================
L_PANEL_PARAMS = {
    # Geometry (in mm)
    'outer_size': 250.0,     # mm - outer dimension (square)
    'inner_size': 150.0,     # mm - inner cutout dimension
    'thickness': 1.0,        # mm - out-of-plane

    # Derived
    'leg_width': 100.0,      # mm - width of each leg (250 - 150)

    # Material (Concrete-like)
    'E': 25.85e3,            # MPa (= 25.85 GPa, concrete)
    'nu': 0.18,
    'plane': 'strain',

    # Phase-field
    'Gc': 0.095,             # N/mm (= 95 J/m^2)
    'l0': 5.0,               # mm - length scale for this problem

    # Mesh
    'h_fine': 1.25,          # mm - near corner (l0/4)
    'h_coarse': 10.0,        # mm - far from corner
    'corner_refine_radius': 50.0,  # mm - refinement around corner

    # Loading
    'u_max': 1.0,            # mm - max vertical displacement
    'n_steps': 200,

    # Solver
    'tol_u': 1e-6,
    'tol_d': 1e-6,
    'max_iter': 300,         # May need more iterations for nucleation
}


# ============================================================================
# Mesh Generation for L-Shaped Domain
# ============================================================================
def generate_l_panel_mesh(params: Optional[Dict] = None,
                           use_graded_mesh: bool = True) -> TriangleMesh:
    """
    Generate L-shaped panel mesh with local refinement at inner corner.

    The L-shape is created by combining two rectangular regions:
    1. Bottom horizontal leg (full width)
    2. Top vertical leg (left portion only)

    Args:
        params: L-panel parameters (uses defaults if None)
        use_graded_mesh: if True, use graded mesh; if False, use uniform mesh

    Returns:
        TriangleMesh instance
    """
    p = L_PANEL_PARAMS.copy()
    if params is not None:
        p.update(params)

    outer = p['outer_size']
    inner = p['inner_size']
    h_fine = p['h_fine']
    h_coarse = p['h_coarse']
    refine_radius = p['corner_refine_radius']

    # Inner corner position
    corner_x = outer - inner  # = 100 mm
    corner_y = outer - inner  # = 100 mm

    if use_graded_mesh:
        mesh = _create_graded_l_panel_mesh(
            outer, inner, corner_x, corner_y,
            h_fine, h_coarse, refine_radius
        )
    else:
        mesh = _create_uniform_l_panel_mesh(outer, inner, h_fine)

    return mesh


def _create_graded_l_panel_mesh(outer: float, inner: float,
                                  corner_x: float, corner_y: float,
                                  h_fine: float, h_coarse: float,
                                  refine_radius: float) -> TriangleMesh:
    """
    Create graded L-shaped panel mesh with refinement at inner corner.

    Strategy: Build the L-shape as two rectangular regions meshed together:
    1. Bottom rectangle: [0, outer] x [0, corner_y]
    2. Left rectangle: [0, corner_x] x [corner_y, outer]

    Refinement is applied near the inner corner (corner_x, corner_y).

    Args:
        outer: outer dimension
        inner: inner cutout dimension
        corner_x: inner corner x-position
        corner_y: inner corner y-position
        h_fine: fine element size
        h_coarse: coarse element size
        refine_radius: refinement radius around corner

    Returns:
        TriangleMesh instance
    """
    all_nodes = []
    all_elements = []

    # ======================================================
    # Part 1: Bottom rectangle [0, outer] x [0, corner_y]
    # ======================================================
    bottom_nodes, bottom_elements, bottom_rows = _create_graded_rect_mesh(
        x_min=0, x_max=outer,
        y_min=0, y_max=corner_y,
        corner_x=corner_x, corner_y=corner_y,
        h_fine=h_fine, h_coarse=h_coarse,
        refine_radius=refine_radius,
        all_nodes=all_nodes,
    )
    all_elements.extend(bottom_elements)

    # ======================================================
    # Part 2: Left rectangle [0, corner_x] x [corner_y, outer]
    # ======================================================
    # We need to share the top row of the bottom rectangle (at y=corner_y)
    # with the bottom row of the left rectangle. Extract nodes at y=corner_y
    # with x <= corner_x.
    top_nodes, top_elements, _ = _create_graded_rect_mesh(
        x_min=0, x_max=corner_x,
        y_min=corner_y, y_max=outer,
        corner_x=corner_x, corner_y=corner_y,
        h_fine=h_fine, h_coarse=h_coarse,
        refine_radius=refine_radius,
        all_nodes=all_nodes,
        shared_bottom_row=bottom_rows[-1],
        shared_x_max=corner_x,
    )
    all_elements.extend(top_elements)

    nodes = np.array(all_nodes)
    elements = np.array(all_elements)

    return TriangleMesh(nodes, elements)


def _create_graded_rect_mesh(x_min: float, x_max: float,
                               y_min: float, y_max: float,
                               corner_x: float, corner_y: float,
                               h_fine: float, h_coarse: float,
                               refine_radius: float,
                               all_nodes: list,
                               shared_bottom_row: Optional[List[int]] = None,
                               shared_x_max: Optional[float] = None,
                               ) -> Tuple[list, list, list]:
    """
    Create graded rectangular mesh region with refinement near the corner.

    Args:
        x_min, x_max, y_min, y_max: rectangle bounds
        corner_x, corner_y: inner corner position for refinement
        h_fine, h_coarse: element sizes
        refine_radius: refinement radius
        all_nodes: shared node list (mutated)
        shared_bottom_row: if provided, reuse these nodes for bottom row
        shared_x_max: max x for shared nodes filtering

    Returns:
        (nodes_added, elements, row_list)
    """
    width = x_max - x_min
    height = y_max - y_min

    # Determine y-zones with refinement
    y_zones = []
    zone_h = []

    # Create vertical zones based on distance to corner
    ny_target = max(int(np.ceil(height / h_coarse)), 2)
    for i in range(ny_target + 1):
        y = y_min + i * height / ny_target
        dist_to_corner = np.sqrt((0.5 * (x_min + x_max) - corner_x) ** 2 +
                                  (y - corner_y) ** 2)
        if dist_to_corner < refine_radius:
            h_local = h_fine + (h_coarse - h_fine) * (dist_to_corner / refine_radius)
        else:
            h_local = h_coarse
        y_zones.append(y)
        if i < ny_target:
            zone_h.append(h_local)

    # Refine y-zones near corner
    refined_y = [y_zones[0]]
    for i in range(1, len(y_zones)):
        y_prev = refined_y[-1]
        y_next = y_zones[i]
        h = zone_h[i - 1] if i - 1 < len(zone_h) else h_coarse

        dy = y_next - y_prev
        if dy > h * 1.5:
            n_sub = max(int(np.ceil(dy / h)), 2)
            for j in range(1, n_sub):
                refined_y.append(y_prev + j * dy / n_sub)
        refined_y.append(y_next)

    # Remove duplicates and sort
    refined_y = sorted(set(round(y, 10) for y in refined_y))

    # Create nodes
    node_rows = []
    elements = []

    for row_idx, y in enumerate(refined_y):
        # Determine horizontal element size at this y
        dist_to_corner_y = abs(y - corner_y)
        if dist_to_corner_y < refine_radius:
            h_x = h_fine + (h_coarse - h_fine) * (dist_to_corner_y / refine_radius)
        else:
            h_x = h_coarse

        nx = max(int(np.ceil(width / h_x)), 2)

        # Check if we should use shared bottom row
        if row_idx == 0 and shared_bottom_row is not None:
            # Filter shared nodes to x <= shared_x_max
            shared_row = []
            for node_idx in shared_bottom_row:
                nx_val = all_nodes[node_idx][0]
                if shared_x_max is None or nx_val <= shared_x_max + 1e-10:
                    shared_row.append(node_idx)
            node_rows.append(shared_row)
            continue

        row_nodes = []
        for j in range(nx + 1):
            x = x_min + j * width / nx
            # Check if this node already exists in all_nodes (for sharing)
            found = False
            if row_idx == 0 and shared_bottom_row is not None:
                # Already handled above
                pass
            else:
                # Check for existing node at this position
                for existing_idx in range(len(all_nodes)):
                    ex, ey = all_nodes[existing_idx]
                    if abs(ex - x) < 1e-10 and abs(ey - y) < 1e-10:
                        row_nodes.append(existing_idx)
                        found = True
                        break
                if not found:
                    row_nodes.append(len(all_nodes))
                    all_nodes.append([x, y])

        node_rows.append(row_nodes)

    # Create elements from rows
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
            elements.extend(_triangulate_strip_l(all_nodes, bottom_row, top_row))

    return all_nodes, elements, node_rows


def _create_uniform_l_panel_mesh(outer: float, inner: float,
                                   h: float) -> TriangleMesh:
    """
    Create uniform L-shaped panel mesh (simple approach).

    Creates a structured mesh for each rectangular part and merges them.

    Args:
        outer: outer dimension
        inner: inner cutout dimension
        h: target element size

    Returns:
        TriangleMesh instance
    """
    corner_x = outer - inner
    corner_y = outer - inner

    all_nodes = []
    all_elements = []

    # Bottom rectangle: [0, outer] x [0, corner_y]
    nx_bottom = max(int(np.ceil(outer / h)), 4)
    ny_bottom = max(int(np.ceil(corner_y / h)), 4)

    node_map_bottom = {}
    for j in range(ny_bottom + 1):
        for i in range(nx_bottom + 1):
            x = i * outer / nx_bottom
            y = j * corner_y / ny_bottom
            idx = len(all_nodes)
            all_nodes.append([x, y])
            node_map_bottom[(i, j)] = idx

    for j in range(ny_bottom):
        for i in range(nx_bottom):
            n0 = node_map_bottom[(i, j)]
            n1 = node_map_bottom[(i + 1, j)]
            n2 = node_map_bottom[(i + 1, j + 1)]
            n3 = node_map_bottom[(i, j + 1)]
            all_elements.append([n0, n1, n2])
            all_elements.append([n0, n2, n3])

    # Top-left rectangle: [0, corner_x] x [corner_y, outer]
    nx_top = max(int(np.ceil(corner_x / h)), 4)
    ny_top = max(int(np.ceil((outer - corner_y) / h)), 4)

    node_map_top = {}
    for j in range(ny_top + 1):
        for i in range(nx_top + 1):
            x = i * corner_x / nx_top
            y = corner_y + j * (outer - corner_y) / ny_top

            # Check if this node already exists (shared boundary at y=corner_y)
            if j == 0:
                # Find matching node in bottom rectangle
                found = False
                for idx_b in range(len(all_nodes)):
                    ex, ey = all_nodes[idx_b]
                    if abs(ex - x) < 1e-10 and abs(ey - y) < 1e-10:
                        node_map_top[(i, j)] = idx_b
                        found = True
                        break
                if not found:
                    idx = len(all_nodes)
                    all_nodes.append([x, y])
                    node_map_top[(i, j)] = idx
            else:
                idx = len(all_nodes)
                all_nodes.append([x, y])
                node_map_top[(i, j)] = idx

    for j in range(ny_top):
        for i in range(nx_top):
            n0 = node_map_top[(i, j)]
            n1 = node_map_top[(i + 1, j)]
            n2 = node_map_top[(i + 1, j + 1)]
            n3 = node_map_top[(i, j + 1)]
            all_elements.append([n0, n1, n2])
            all_elements.append([n0, n2, n3])

    nodes = np.array(all_nodes)
    elements = np.array(all_elements)

    return TriangleMesh(nodes, elements)


def _triangulate_strip_l(nodes: list,
                          bottom_row: List[int],
                          top_row: List[int]) -> List[List[int]]:
    """
    Create conforming triangulation between two rows with different node counts.
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

            diag1 = np.linalg.norm(
                np.array(p_bi1) - np.array(p_tj))
            diag2 = np.linalg.norm(
                np.array(p_bi) - np.array(p_tj1))

            if diag1 <= diag2:
                elements.append([bottom_row[i], bottom_row[i + 1], top_row[j]])
                i += 1
            else:
                elements.append([bottom_row[i], top_row[j], top_row[j + 1]])
                j += 1

    return elements


# ============================================================================
# Boundary Identification
# ============================================================================
def identify_l_panel_boundaries(mesh: TriangleMesh,
                                 params: Dict) -> Dict:
    """
    Identify boundary nodes for L-shaped panel.

    Args:
        mesh: TriangleMesh instance
        params: L-panel parameters

    Returns:
        Dictionary with boundary node arrays
    """
    outer = params['outer_size']
    inner = params['inner_size']
    corner_x = outer - inner
    corner_y = outer - inner

    tol = 1e-6

    boundaries = {}

    # Bottom edge (y = 0, 0 <= x <= outer)
    boundaries['bottom'] = mesh.get_nodes_in_region(
        lambda x, y: y < tol
    )

    # Left edge (x = 0, 0 <= y <= outer)
    boundaries['left'] = mesh.get_nodes_in_region(
        lambda x, y: x < tol
    )

    # Top of vertical leg (y = outer, 0 <= x <= corner_x)
    boundaries['top_leg'] = mesh.get_nodes_in_region(
        lambda x, y: y > outer - tol and x <= corner_x + tol
    )

    # Inner corner (closest node)
    dists = np.sqrt((mesh.nodes[:, 0] - corner_x) ** 2 +
                    (mesh.nodes[:, 1] - corner_y) ** 2)
    boundaries['inner_corner'] = np.array([np.argmin(dists)])

    return boundaries


# ============================================================================
# L-Panel Boundary Conditions
# ============================================================================
def apply_l_panel_boundary_conditions(mesh: TriangleMesh,
                                       u_applied: float,
                                       params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for L-shaped panel.

    Configuration:
    - Bottom horizontal edge: fully fixed (u_x = u_y = 0)
    - Top of vertical leg: u_y = u_applied (vertical displacement)

    Args:
        mesh: TriangleMesh instance
        u_applied: Applied vertical displacement at top of vertical leg
        params: L-panel parameters

    Returns:
        bc_dofs, bc_values
    """
    outer = params['outer_size']
    inner = params['inner_size']
    corner_x = outer - inner

    tol = 1e-6

    # Bottom: fully fixed
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    bc_bottom_dofs, bc_bottom_vals = create_fixed_bc(mesh, bottom_nodes)

    # Top of vertical leg: prescribed vertical displacement
    top_leg_nodes = mesh.get_nodes_in_region(
        lambda x, y: y > outer - tol and x <= corner_x + tol
    )
    bc_top_dofs, bc_top_vals = create_prescribed_displacement_bc(
        mesh, top_leg_nodes, 'y', u_applied
    )

    # Combine
    bc_dofs, bc_values = merge_bcs(
        (bc_bottom_dofs, bc_bottom_vals),
        (bc_top_dofs, bc_top_vals),
    )

    return bc_dofs, bc_values


def create_l_panel_bc_function(mesh: TriangleMesh,
                                params: Dict) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """
    Create BC function for load stepping in L-panel benchmark.

    Args:
        mesh: TriangleMesh instance
        params: L-panel parameters

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values_func: function(u_applied) -> bc_values array
    """
    outer = params['outer_size']
    inner = params['inner_size']
    corner_x = outer - inner

    tol = 1e-6

    # Bottom: fully fixed (always zero)
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    bc_bottom_dofs, bc_bottom_vals = create_fixed_bc(mesh, bottom_nodes)

    # Top of vertical leg: will have varying displacement
    top_leg_nodes = mesh.get_nodes_in_region(
        lambda x, y: y > outer - tol and x <= corner_x + tol
    )
    bc_top_dofs = np.array([2 * n + 1 for n in top_leg_nodes])  # y-DOFs

    # All BC DOFs
    all_bc_dofs = np.concatenate([bc_bottom_dofs, bc_top_dofs])
    n_fixed = len(bc_bottom_vals)

    def bc_values_func(u_applied: float) -> np.ndarray:
        """Return BC values for given applied displacement."""
        values = np.zeros(len(all_bc_dofs))
        values[:n_fixed] = bc_bottom_vals
        values[n_fixed:] = u_applied  # Vertical displacement at top
        return values

    return all_bc_dofs, bc_values_func


# ============================================================================
# Reaction Force Computation
# ============================================================================
def compute_l_panel_reaction_force(mesh: TriangleMesh,
                                    elements: List[GraFEAElement],
                                    u: np.ndarray,
                                    damage: np.ndarray,
                                    params: Dict) -> float:
    """
    Compute vertical reaction force at top of vertical leg.

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement solution vector
        damage: Damage field
        params: L-panel parameters

    Returns:
        F_y: Total vertical reaction force
    """
    outer = params['outer_size']
    inner = params['inner_size']
    corner_x = outer - inner

    tol = 1e-6
    top_leg_nodes = mesh.get_nodes_in_region(
        lambda x, y: y > outer - tol and x <= corner_x + tol
    )

    F_int = assemble_internal_force(mesh, elements, u, damage)

    F_y = 0.0
    for node in top_leg_nodes:
        dof_y = 2 * node + 1
        F_y += F_int[dof_y]

    return abs(F_y)


# ============================================================================
# Crack Path and Nucleation Analysis
# ============================================================================
def extract_crack_path(mesh: TriangleMesh,
                       damage: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
    """
    Extract crack path from damage field.

    Args:
        mesh: TriangleMesh instance
        damage: Damage field
        threshold: Damage threshold

    Returns:
        crack_path: Array of (x, y) coordinates, shape (n_points, 2)
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
    # Sort by distance from inner corner
    corner_x = mesh.nodes[:, 0].max() - 150  # approximate
    corner_y = corner_x
    dists = np.sqrt((midpoints[:, 0] - corner_x) ** 2 +
                    (midpoints[:, 1] - corner_y) ** 2)
    sort_idx = np.argsort(dists)
    return midpoints[sort_idx]


def check_nucleation(damage: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if crack nucleation has occurred.

    Nucleation is detected when damage exceeds threshold at any location.

    Args:
        damage: Damage field
        threshold: Nucleation threshold

    Returns:
        True if nucleation detected
    """
    return np.max(damage) > threshold


def find_nucleation_location(mesh: TriangleMesh,
                              damage: np.ndarray,
                              threshold: float = 0.1) -> Optional[Tuple[float, float]]:
    """
    Find location where damage first exceeds threshold.

    Args:
        mesh: TriangleMesh instance
        damage: Damage field
        threshold: Nucleation threshold

    Returns:
        (x, y) of nucleation location, or None
    """
    if np.max(damage) < threshold:
        return None

    max_idx = np.argmax(damage)
    midpoints = mesh.compute_edge_midpoints()
    return tuple(midpoints[max_idx])


# ============================================================================
# Results Container
# ============================================================================
@dataclass
class LPanelResults:
    """Results container for L-shaped panel benchmark."""
    displacement: np.ndarray      # Applied displacement at each step
    reaction_force: np.ndarray    # Vertical reaction force at each step
    max_damage: np.ndarray        # Maximum damage at each step
    strain_energy: np.ndarray     # Strain energy at each step
    surface_energy: np.ndarray    # Surface energy at each step
    final_damage: np.ndarray      # Final damage field
    crack_path: np.ndarray        # Extracted crack path coordinates
    nucleation_detected: bool     # Whether nucleation occurred
    nucleation_step: Optional[int]     # Step at which nucleation was detected
    nucleation_location: Optional[Tuple[float, float]]  # Nucleation (x, y)
    damage_snapshots: List[np.ndarray] = field(default_factory=list)
    load_steps: List[LoadStep] = field(default_factory=list)

    @property
    def total_energy(self) -> np.ndarray:
        """Total energy (strain + surface)."""
        return self.strain_energy + self.surface_energy

    @property
    def peak_force(self) -> float:
        """Peak reaction force."""
        return np.max(self.reaction_force)

    @property
    def displacement_at_peak(self) -> float:
        """Applied displacement at peak force."""
        idx = np.argmax(self.reaction_force)
        return self.displacement[idx]


# ============================================================================
# Main Simulation Runner
# ============================================================================
def run_l_panel_benchmark(params: Optional[Dict] = None,
                           verbose: bool = True,
                           save_snapshots: bool = True,
                           snapshot_interval: int = 10) -> LPanelResults:
    """
    Run L-shaped panel simulation with nucleation tracking.

    This is THE critical benchmark: starts with ZERO damage and tests
    whether the phase-field formulation can nucleate a crack at the
    inner corner stress concentration.

    Args:
        params: L-panel parameters (uses defaults if None)
        verbose: Print progress information
        save_snapshots: Save damage snapshots
        snapshot_interval: Steps between snapshots

    Returns:
        LPanelResults: Complete simulation results
    """
    p = L_PANEL_PARAMS.copy()
    if params is not None:
        p.update(params)

    if verbose:
        print("=" * 60)
        print("L-Shaped Panel Benchmark: Crack NUCLEATION Test")
        print("=" * 60)
        print("  NO pre-existing crack or damage!")
        print("  Crack must nucleate spontaneously at inner corner.")

    # Generate mesh
    if verbose:
        print("\nGenerating L-shaped panel mesh...")
    mesh = generate_l_panel_mesh(p)
    if verbose:
        print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"{mesh.n_edges} edges")

    # Create material (concrete)
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

    # Create edge graph
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')

    # Create solver
    config = SolverConfig(
        tol_u=p['tol_u'],
        tol_d=p['tol_d'],
        max_stagger_iter=p['max_iter'],
        verbose=verbose
    )
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # CRITICAL: Start with ZERO damage (no initial crack!)
    # The solver initializes with zero damage by default.
    if verbose:
        print("\n  Initial damage: ZERO (nucleation test)")

    # Setup boundary conditions
    bc_dofs, bc_values_func = create_l_panel_bc_function(mesh, p)

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    if verbose:
        print(f"\nRunning simulation: {p['n_steps']} steps to "
              f"u_max = {p['u_max']:.4f} mm")
        print("Watching for crack nucleation at inner corner...")
        print("-" * 60)

    # Run simulation
    load_steps = solver.solve(u_steps, bc_dofs, bc_values_func)

    # Extract results
    if verbose:
        print("\nExtracting results...")

    displacements = np.array([r.load_factor for r in load_steps])
    strain_energies = np.array([r.strain_energy for r in load_steps])
    surface_energies = np.array([r.surface_energy for r in load_steps])

    forces = []
    max_damages = []
    damage_snapshots = []
    nucleation_detected = False
    nucleation_step = None
    nucleation_location = None

    for i, result in enumerate(load_steps):
        # Reaction force
        F = compute_l_panel_reaction_force(mesh, elements, result.displacement,
                                            result.damage, p)
        forces.append(F)

        # Max damage
        d_max = np.max(result.damage)
        max_damages.append(d_max)

        # Check nucleation
        if not nucleation_detected and check_nucleation(result.damage, threshold=0.1):
            nucleation_detected = True
            nucleation_step = i
            nucleation_location = find_nucleation_location(mesh, result.damage)
            if verbose:
                print(f"\n  *** CRACK NUCLEATION DETECTED ***")
                print(f"  Step: {i}")
                print(f"  Displacement: {displacements[i]:.4f} mm")
                print(f"  Location: {nucleation_location}")
                print()

        # Save snapshot
        if save_snapshots and i % snapshot_interval == 0:
            damage_snapshots.append(result.damage.copy())

    forces = np.array(forces)
    max_damages = np.array(max_damages)

    # Final damage and crack path
    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.5)

    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Peak force: {np.max(forces):.4f}")
        print(f"  Maximum damage: {np.max(final_damage):.4f}")
        print(f"  Nucleation detected: {nucleation_detected}")
        if nucleation_detected:
            print(f"  Nucleation step: {nucleation_step}")
            print(f"  Nucleation location: {nucleation_location}")
        print(f"  Final strain energy: {strain_energies[-1]:.6e}")
        print(f"  Final surface energy: {surface_energies[-1]:.6e}")

    return LPanelResults(
        displacement=displacements,
        reaction_force=forces,
        max_damage=max_damages,
        strain_energy=strain_energies,
        surface_energy=surface_energies,
        final_damage=final_damage,
        crack_path=crack_path,
        nucleation_detected=nucleation_detected,
        nucleation_step=nucleation_step,
        nucleation_location=nucleation_location,
        damage_snapshots=damage_snapshots,
        load_steps=load_steps,
    )


# ============================================================================
# Validation
# ============================================================================
def validate_l_panel_results(results: LPanelResults,
                              params: Optional[Dict] = None,
                              verbose: bool = True) -> Dict:
    """
    Validate L-shaped panel nucleation results.

    Critical checks:
    1. Nucleation occurred (no pre-crack!)
    2. Nucleation at inner corner
    3. Crack propagates after nucleation
    4. Crack direction is physically reasonable (~45 degrees)
    5. Peak load exists
    6. Started with zero damage

    Args:
        results: LPanelResults from simulation
        params: L-panel parameters
        verbose: Print validation results

    Returns:
        Dictionary with validation status
    """
    p = L_PANEL_PARAMS.copy()
    if params is not None:
        p.update(params)

    corner_x = p['outer_size'] - p['inner_size']
    corner_y = corner_x
    l0 = p['l0']

    validation = {
        'nucleation_occurred': {
            'passed': False,
            'nucleation_step': None,
        },
        'nucleation_at_corner': {
            'passed': False,
            'distance_to_corner': None,
            'location': None,
        },
        'crack_propagated': {
            'passed': False,
            'damaged_edges_count': None,
        },
        'correct_direction': {
            'passed': False,
            'crack_dx': None,
            'crack_dy': None,
        },
        'reasonable_angle': {
            'passed': False,
            'crack_angle': None,
        },
        'has_peak': {
            'passed': False,
            'peak_force': None,
        },
        'started_undamaged': {
            'passed': False,
        },
    }

    # CHECK 1: Nucleation occurred
    validation['nucleation_occurred']['passed'] = results.nucleation_detected
    validation['nucleation_occurred']['nucleation_step'] = results.nucleation_step

    # CHECK 2: Nucleation at inner corner
    if results.nucleation_location is not None:
        nuc_x, nuc_y = results.nucleation_location
        dist = np.sqrt((nuc_x - corner_x) ** 2 + (nuc_y - corner_y) ** 2)
        validation['nucleation_at_corner']['distance_to_corner'] = dist
        validation['nucleation_at_corner']['location'] = results.nucleation_location
        validation['nucleation_at_corner']['passed'] = dist < l0 * 5

    # CHECK 3: Crack propagated
    d_final = results.final_damage
    damaged_count = np.sum(d_final > 0.5)
    validation['crack_propagated']['damaged_edges_count'] = int(damaged_count)
    validation['crack_propagated']['passed'] = damaged_count > 5

    # CHECK 4 & 5: Crack direction and angle
    crack_path = results.crack_path
    if len(crack_path) > 1:
        dx = crack_path[-1, 0] - crack_path[0, 0]
        dy = crack_path[-1, 1] - crack_path[0, 1]

        validation['correct_direction']['crack_dx'] = dx
        validation['correct_direction']['crack_dy'] = dy
        # Expected: crack goes toward upper-right (dx > 0, dy > 0)
        validation['correct_direction']['passed'] = dx > 0 and dy > 0

        if dx > 0:
            angle = np.degrees(np.arctan2(dy, dx))
            validation['reasonable_angle']['crack_angle'] = angle
            # Expected roughly 45 degrees (within 15-75)
            validation['reasonable_angle']['passed'] = 15 < angle < 75
        else:
            validation['reasonable_angle']['crack_angle'] = None

    # CHECK 6: Peak load
    peak = results.peak_force
    peak_idx = np.argmax(results.reaction_force)
    validation['has_peak']['peak_force'] = peak
    validation['has_peak']['passed'] = (
        peak > 0 and peak_idx < len(results.reaction_force) - 10
    )

    # CHECK 7: Started undamaged
    if len(results.load_steps) > 0:
        initial_damage = results.load_steps[0].damage
        validation['started_undamaged']['passed'] = np.max(initial_damage) < 0.01

    if verbose:
        print("\n" + "=" * 60)
        print("L-Panel Validation Results")
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


def compare_with_original_grafea(results: LPanelResults) -> Dict:
    """
    Document comparison with original GraFEA capability.

    Original GraFEA CANNOT nucleate cracks - it requires pre-existing
    damage or geometric defects. This is a key advantage of the
    phase-field approach.

    Args:
        results: LPanelResults from simulation

    Returns:
        Comparison dictionary
    """
    return {
        'original_grafea': {
            'can_nucleate': False,
            'requires_precrack': True,
            'would_fail_l_panel': True,
        },
        'edge_based_phasefield': {
            'can_nucleate': results.nucleation_detected,
            'requires_precrack': False,
            'nucleation_location': results.nucleation_location,
        },
        'advantage_demonstrated': results.nucleation_detected,
    }


# ============================================================================
# Class Interface
# ============================================================================
class LPanelBenchmark:
    """
    Class interface for L-shaped panel benchmark with plotting.

    Example:
        benchmark = LPanelBenchmark()
        benchmark.run()
        benchmark.validate()
        benchmark.plot_results()
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = L_PANEL_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        self.mesh = None
        self.results = None
        self.validation = None

    def run(self, verbose: bool = True) -> LPanelResults:
        """Run the L-panel benchmark simulation."""
        self.results = run_l_panel_benchmark(self.params, verbose=verbose)
        self.mesh = generate_l_panel_mesh(self.params)
        return self.results

    def validate(self, verbose: bool = True) -> Dict:
        """Validate simulation results."""
        if self.results is None:
            raise ValueError("Run simulation first")
        self.validation = validate_l_panel_results(
            self.results, self.params, verbose=verbose
        )
        return self.validation

    def plot_results(self, save_path: Optional[str] = None):
        """Create comprehensive results plot."""
        if self.results is None:
            raise ValueError("Run simulation first")

        try:
            import matplotlib.pyplot as plt
            from postprocess.visualization import plot_damage_field
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Load-Displacement with nucleation marker
        ax = axes[0, 0]
        ax.plot(self.results.displacement, self.results.reaction_force,
                'b-', lw=2)
        if self.results.nucleation_step is not None:
            nuc_u = self.results.displacement[self.results.nucleation_step]
            nuc_F = self.results.reaction_force[self.results.nucleation_step]
            ax.axvline(x=nuc_u, color='r', linestyle='--', label='Nucleation')
            ax.plot(nuc_u, nuc_F, 'ro', ms=10)
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Reaction Force [N]')
        ax.set_title('Load-Displacement with Nucleation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Max Damage Evolution
        ax = axes[0, 1]
        ax.plot(self.results.displacement, self.results.max_damage,
                'r-', lw=2)
        ax.axhline(y=0.1, color='g', linestyle='--', label='Nucleation threshold')
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Maximum Damage')
        ax.set_title('Damage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Energy Evolution
        ax = axes[0, 2]
        ax.plot(self.results.displacement, self.results.strain_energy,
                'b-', label='Strain', lw=2)
        ax.plot(self.results.displacement, self.results.surface_energy,
                'r-', label='Surface', lw=2)
        ax.plot(self.results.displacement, self.results.total_energy,
                'k--', label='Total', lw=2)
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Energy [N*mm]')
        ax.set_title('Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4-6. Damage snapshots at different stages
        if len(self.results.damage_snapshots) >= 3:
            snap_indices = [0, len(self.results.damage_snapshots) // 2,
                           len(self.results.damage_snapshots) - 1]
            titles = ['Early', 'Mid', 'Final']
        else:
            snap_indices = list(range(min(3, len(self.results.damage_snapshots))))
            titles = [f'Snapshot {i}' for i in snap_indices]

        for col, (idx, title) in enumerate(zip(snap_indices, titles)):
            ax = axes[1, col]
            if idx < len(self.results.damage_snapshots) and self.mesh is not None:
                plot_damage_field(self.mesh, self.results.damage_snapshots[idx],
                                  ax=ax, colorbar=(col == 2))
                # Mark inner corner
                corner_x = self.params['outer_size'] - self.params['inner_size']
                corner_y = corner_x
                ax.plot(corner_x, corner_y, 'go', ms=15,
                        markerfacecolor='none', linewidth=3)
            ax.set_title(f'Damage: {title}')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()


# ============================================================================
# Convenience Functions
# ============================================================================
def quick_l_panel_test(n_steps: int = 50, verbose: bool = True) -> LPanelResults:
    """
    Run a quick L-panel test with coarser mesh.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        LPanelResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 5.0,
        'h_coarse': 20.0,
        'l0': 10.0,
    }
    return run_l_panel_benchmark(params, verbose=verbose)


def very_quick_l_panel_test(n_steps: int = 20,
                             verbose: bool = True) -> LPanelResults:
    """
    Run a very quick L-panel test for pipeline testing.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        LPanelResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 10.0,
        'h_coarse': 40.0,
        'l0': 20.0,
        'corner_refine_radius': 60.0,
    }
    return run_l_panel_benchmark(params, verbose=verbose)


if __name__ == '__main__':
    results = run_l_panel_benchmark(verbose=True)
    validate_l_panel_results(results, verbose=True)
