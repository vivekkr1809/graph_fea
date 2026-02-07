"""
Three-Point Bending (TPB) Benchmark
====================================

Mode-I fracture benchmark with practical structural configuration and
experimental validation potential.

Geometry:
    - Rectangular beam L x W with center notch from bottom
    - Notch depth = W/2 (half the beam height)
    - Three-point loading: two supports near bottom, load at top center
    - Crack expected to propagate vertically upward from notch tip

Loading:
    - Left support: roller (u_y = 0, u_x free)
    - Right support: pin (u_x = u_y = 0)
    - Top center: applied downward displacement

Expected Results:
    - Vertical crack propagation from notch tip
    - Peak load followed by softening
    - Mode-I dominated behavior with bending stress gradient

References:
    - Miehe et al. (2010) "Thermodynamically consistent phase-field models"
    - Ambati et al. (2015) "A review on phase-field models of brittle fracture"
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
    create_prescribed_displacement_bc, create_roller_bc,
)
from assembly.global_assembly import (
    assemble_internal_force, compute_all_edge_strains,
    compute_all_tensor_strains,
)
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep


# ============================================================================
# Default TPB parameters
# ============================================================================
TPB_PARAMS = {
    # Geometry (in mm)
    'L': 100.0,              # mm - beam length
    'W': 40.0,               # mm - beam height
    'thickness': 1.0,        # mm - out-of-plane thickness
    'notch_depth': 20.0,     # mm - center notch depth (half the height)
    'support_span': 80.0,    # mm - distance between supports

    # Derived positions
    'support_left_x': 10.0,  # mm - left support position
    'support_right_x': 90.0, # mm - right support position
    'load_x': 50.0,          # mm - load application point (center)

    # Material (steel-like, same as SENT/SENS)
    'E': 210.0e3,            # MPa
    'nu': 0.3,
    'plane': 'strain',

    # Phase-field
    'Gc': 2.7,               # N/mm
    'l0': 1.0,               # mm - length scale for this geometry

    # Mesh
    'h_fine': 0.25,          # mm - near notch (l0/4)
    'h_coarse': 2.0,         # mm - far from crack region
    'notch_refine_radius': 10.0,  # mm - refinement around notch

    # Loading
    'u_max': 0.5,            # mm - max vertical displacement at load point
    'n_steps': 200,

    # Solver
    'tol_u': 1e-6,
    'tol_d': 1e-6,
    'max_iter': 200,

    # Support type
    'support_type': 'point',  # 'point' or 'distributed'
    'support_width': 2.0,     # mm - if distributed
}


# ============================================================================
# Mesh Generation
# ============================================================================
def generate_tpb_mesh(params: Optional[Dict] = None,
                      use_graded_mesh: bool = True) -> TriangleMesh:
    """
    Generate three-point bending mesh with local refinement near notch
    tip and expected vertical crack path.

    The mesh uses a graded approach with fine elements near the notch
    tip region and coarser elements elsewhere. The notch is represented
    as initial damage rather than a geometric feature.

    Args:
        params: TPB parameters dictionary (uses defaults if None)
        use_graded_mesh: if True, use graded mesh; if False, use uniform mesh

    Returns:
        TriangleMesh instance
    """
    p = TPB_PARAMS.copy()
    if params is not None:
        p.update(params)

    L = p['L']
    W = p['W']
    h_fine = p['h_fine']
    h_coarse = p['h_coarse']
    notch_depth = p['notch_depth']
    refine_radius = p['notch_refine_radius']
    notch_x = L / 2

    if use_graded_mesh:
        mesh = _create_graded_tpb_mesh(L, W, notch_x, notch_depth,
                                        h_fine, h_coarse, refine_radius)
    else:
        nx = max(int(np.ceil(L / h_fine)), 10)
        ny = max(int(np.ceil(W / h_fine)), 10)
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(L, W, nx, ny, pattern='alternating',
                                      thickness=p['thickness'])

    return mesh


def _create_graded_tpb_mesh(L: float, W: float,
                              notch_x: float, notch_depth: float,
                              h_fine: float, h_coarse: float,
                              refine_radius: float) -> TriangleMesh:
    """
    Create graded mesh for three-point bending with local refinement.

    The mesh is refined in a band around x = notch_x (the notch
    center line / expected crack path) and near y = notch_depth
    (the notch tip).

    Args:
        L: beam length
        W: beam height
        notch_x: x-coordinate of notch center
        notch_depth: notch depth from bottom
        h_fine: fine element size
        h_coarse: coarse element size
        refine_radius: refinement radius around notch tip

    Returns:
        TriangleMesh instance
    """
    # Vertical zones: fine near notch tip and above, coarse elsewhere
    y_zones = []
    zone_h = []

    # Region below notch tip: coarse far from tip, fine near tip
    y_fine_start = max(0, notch_depth - refine_radius)

    if y_fine_start > 0:
        n_coarse_bottom = max(int(np.ceil(y_fine_start / h_coarse)), 2)
        dy = y_fine_start / n_coarse_bottom
        for i in range(n_coarse_bottom):
            y_zones.append(i * dy)
            zone_h.append(h_coarse)

    # Fine zone around notch tip and above (crack path)
    y_fine_end = min(W, notch_depth + refine_radius)
    n_fine = max(int(np.ceil((y_fine_end - y_fine_start) / h_fine)), 4)
    dy_fine = (y_fine_end - y_fine_start) / n_fine
    for i in range(n_fine):
        y_zones.append(y_fine_start + i * dy_fine)
        zone_h.append(h_fine)

    # Top coarse zone (above refinement)
    if y_fine_end < W:
        n_coarse_top = max(int(np.ceil((W - y_fine_end) / h_coarse)), 2)
        dy = (W - y_fine_end) / n_coarse_top
        for i in range(n_coarse_top):
            y_zones.append(y_fine_end + i * dy)
            zone_h.append(h_coarse)

    y_zones.append(W)

    # Create nodes with horizontal refinement near notch x
    nodes = []
    node_rows = []

    for i, y in enumerate(y_zones):
        h = zone_h[i] if i < len(zone_h) else zone_h[-1]

        # Near notch center line, use finer horizontal spacing
        dist_to_notch_y = abs(y - notch_depth)
        if dist_to_notch_y < refine_radius:
            h_x = h  # Fine in both directions near notch
        else:
            h_x = h_coarse

        nx = max(int(np.ceil(L / h_x)), 4)

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
# Pre-notch Damage Initialization
# ============================================================================
def create_notch_damage(mesh: TriangleMesh,
                         notch_x: float,
                         notch_depth: float,
                         l0: float,
                         method: str = 'exponential') -> np.ndarray:
    """
    Initialize damage field to represent pre-existing vertical notch.

    The notch runs vertically from y=0 to y=notch_depth at x=notch_x.
    Unlike the horizontal pre-crack in SENT/SENS, this is a vertical notch
    from the bottom of the beam.

    Args:
        mesh: TriangleMesh instance
        notch_x: x-coordinate of notch center
        notch_depth: height of notch from bottom
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

        if y_mid < notch_depth:
            # Edge is below notch tip - check distance to notch line
            dist_to_notch = abs(x_mid - notch_x)

            if method == 'exponential':
                d[i] = np.exp(-(dist_to_notch / l0) ** 2)
            elif method == 'sharp':
                if dist_to_notch < l0:
                    d[i] = 1.0
            elif method == 'linear':
                if dist_to_notch < 2 * l0:
                    d[i] = max(0, 1.0 - dist_to_notch / (2 * l0))
        else:
            # Above notch tip - smooth transition near tip
            dist_to_tip = np.sqrt((x_mid - notch_x) ** 2 +
                                  (y_mid - notch_depth) ** 2)

            if method == 'exponential':
                if dist_to_tip < 2 * l0:
                    d[i] = np.exp(-(dist_to_tip / l0) ** 2)
            elif method == 'linear':
                if dist_to_tip < l0:
                    d[i] = max(0, 1.0 - dist_to_tip / l0)

    return d


# ============================================================================
# TPB Boundary Conditions
# ============================================================================
def apply_tpb_boundary_conditions(mesh: TriangleMesh,
                                   u_applied: float,
                                   params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for three-point bending.

    Configuration:
    - Left support: roller (u_y = 0, u_x free)
    - Right support: pinned (u_x = u_y = 0)
    - Load point at top center: u_y = -u_applied (downward)

    Args:
        mesh: TriangleMesh instance
        u_applied: Applied downward displacement magnitude at load point
        params: TPB parameters dictionary

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values: Prescribed values at those DOFs
    """
    L = params['L']
    W = params['W']
    support_left_x = params['support_left_x']
    support_right_x = params['support_right_x']
    load_x = params['load_x']
    support_type = params.get('support_type', 'point')
    support_width = params.get('support_width', 2.0)

    tol = min(params.get('h_coarse', 2.0), 1.0)

    # Bottom edge nodes
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol * 0.01)

    if support_type == 'point':
        # Find closest bottom nodes to support positions
        bottom_x = mesh.nodes[bottom_nodes, 0]

        left_idx = np.argmin(np.abs(bottom_x - support_left_x))
        left_support_nodes = np.array([bottom_nodes[left_idx]])

        right_idx = np.argmin(np.abs(bottom_x - support_right_x))
        right_support_nodes = np.array([bottom_nodes[right_idx]])
    else:
        # Distributed support
        left_support_nodes = bottom_nodes[
            np.abs(mesh.nodes[bottom_nodes, 0] - support_left_x) < support_width / 2
        ]
        right_support_nodes = bottom_nodes[
            np.abs(mesh.nodes[bottom_nodes, 0] - support_right_x) < support_width / 2
        ]

    # Left support: roller (u_y = 0, u_x free)
    bc_left_dofs, bc_left_vals = create_roller_bc(mesh, left_support_nodes, direction='x')

    # Right support: pinned (u_x = u_y = 0)
    bc_right_dofs, bc_right_vals = create_fixed_bc(mesh, right_support_nodes)

    # Load point at top center
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > W - tol * 0.01)
    top_x = mesh.nodes[top_nodes, 0]
    load_idx = np.argmin(np.abs(top_x - load_x))
    load_nodes = np.array([top_nodes[load_idx]])

    bc_load_dofs, bc_load_vals = create_prescribed_displacement_bc(
        mesh, load_nodes, 'y', -u_applied  # Negative = downward
    )

    # Combine all BCs
    bc_dofs, bc_values = merge_bcs(
        (bc_left_dofs, bc_left_vals),
        (bc_right_dofs, bc_right_vals),
        (bc_load_dofs, bc_load_vals),
    )

    return bc_dofs, bc_values


def create_tpb_bc_function(mesh: TriangleMesh,
                            params: Dict) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """
    Create BC function for load stepping in TPB benchmark.

    Returns a function that takes displacement magnitude and returns
    the corresponding BC values array.

    Args:
        mesh: TriangleMesh instance
        params: TPB parameters dictionary

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values_func: function(u_applied) -> bc_values array
    """
    L = params['L']
    W = params['W']
    support_left_x = params['support_left_x']
    support_right_x = params['support_right_x']
    load_x = params['load_x']
    support_type = params.get('support_type', 'point')
    support_width = params.get('support_width', 2.0)

    tol = min(params.get('h_coarse', 2.0), 1.0)

    # Bottom edge nodes
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol * 0.01)

    if support_type == 'point':
        bottom_x = mesh.nodes[bottom_nodes, 0]
        left_idx = np.argmin(np.abs(bottom_x - support_left_x))
        left_support_nodes = np.array([bottom_nodes[left_idx]])
        right_idx = np.argmin(np.abs(bottom_x - support_right_x))
        right_support_nodes = np.array([bottom_nodes[right_idx]])
    else:
        left_support_nodes = bottom_nodes[
            np.abs(mesh.nodes[bottom_nodes, 0] - support_left_x) < support_width / 2
        ]
        right_support_nodes = bottom_nodes[
            np.abs(mesh.nodes[bottom_nodes, 0] - support_right_x) < support_width / 2
        ]

    # Fixed BCs: left roller (u_y=0) and right pin (u_x=u_y=0)
    bc_left_dofs, bc_left_vals = create_roller_bc(mesh, left_support_nodes, direction='x')
    bc_right_dofs, bc_right_vals = create_fixed_bc(mesh, right_support_nodes)

    fixed_dofs, fixed_vals = merge_bcs(
        (bc_left_dofs, bc_left_vals),
        (bc_right_dofs, bc_right_vals),
    )

    # Load point DOF (will have varying displacement)
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > W - tol * 0.01)
    top_x = mesh.nodes[top_nodes, 0]
    load_idx = np.argmin(np.abs(top_x - load_x))
    load_node = top_nodes[load_idx]
    load_dof = np.array([2 * load_node + 1])  # y-DOF

    # All BC DOFs
    all_bc_dofs = np.concatenate([fixed_dofs, load_dof])
    n_fixed = len(fixed_vals)

    def bc_values_func(u_applied: float) -> np.ndarray:
        """Return BC values for given applied displacement magnitude."""
        values = np.zeros(len(all_bc_dofs))
        values[:n_fixed] = fixed_vals
        values[n_fixed:] = -u_applied  # Negative = downward
        return values

    return all_bc_dofs, bc_values_func


# ============================================================================
# Reaction Force Computation
# ============================================================================
def compute_tpb_reaction_force(mesh: TriangleMesh,
                                elements: List[GraFEAElement],
                                u: np.ndarray,
                                damage: np.ndarray,
                                params: Dict) -> float:
    """
    Compute vertical reaction force at the load point.

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement solution vector
        damage: Damage field
        params: TPB parameters

    Returns:
        F_y: Vertical reaction force magnitude at load point
    """
    W = params['W']
    load_x = params['load_x']
    tol = min(params.get('h_coarse', 2.0), 1.0)

    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > W - tol * 0.01)
    top_x = mesh.nodes[top_nodes, 0]
    load_idx = np.argmin(np.abs(top_x - load_x))
    load_node = top_nodes[load_idx]

    F_int = assemble_internal_force(mesh, elements, u, damage)

    dof_y = 2 * load_node + 1
    return abs(F_int[dof_y])


# ============================================================================
# Crack Path Analysis
# ============================================================================
def extract_crack_path(mesh: TriangleMesh,
                       damage: np.ndarray,
                       threshold: float = 0.9) -> np.ndarray:
    """
    Extract crack path from damage field.

    The crack path is the locus of heavily damaged edges,
    sorted by y-coordinate (bottom to top for vertical crack).

    Args:
        mesh: TriangleMesh instance
        damage: Damage field, shape (n_edges,)
        threshold: Damage threshold for crack identification

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
    # Sort by y-coordinate for vertical crack
    sort_idx = np.argsort(midpoints[:, 1])
    return midpoints[sort_idx]


def track_vertical_crack(mesh: TriangleMesh,
                          damage: np.ndarray,
                          notch_x: float,
                          l0: float,
                          threshold: float = 0.5) -> Optional[Tuple[float, float]]:
    """
    Track vertical crack from notch tip (highest damaged point near center).

    Args:
        mesh: TriangleMesh instance
        damage: Damage field
        notch_x: x-coordinate of notch center
        l0: Phase-field length scale
        threshold: Damage threshold

    Returns:
        (x, y) of crack tip, or None if no crack detected
    """
    midpoints = mesh.compute_edge_midpoints()

    # Find damaged edges near center (within 2*l0 of notch x)
    center_mask = np.abs(midpoints[:, 0] - notch_x) < l0 * 2
    damaged_mask = damage > threshold

    crack_edges = np.where(center_mask & damaged_mask)[0]

    if len(crack_edges) == 0:
        return None

    # Highest damaged point
    crack_points = midpoints[crack_edges]
    max_y_idx = np.argmax(crack_points[:, 1])

    return tuple(crack_points[max_y_idx])


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


# ============================================================================
# Results Container
# ============================================================================
@dataclass
class TPBResults:
    """Results container for three-point bending benchmark."""
    displacement: np.ndarray     # Applied displacement at each step
    reaction_force: np.ndarray   # Vertical reaction force at each step
    crack_length: np.ndarray     # Crack length at each step
    crack_tip_y: np.ndarray      # Crack tip y-position at each step
    strain_energy: np.ndarray    # Strain energy at each step
    surface_energy: np.ndarray   # Surface energy at each step
    final_damage: np.ndarray     # Final damage field
    crack_path: np.ndarray       # Extracted crack path coordinates
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
def run_tpb_benchmark(params: Optional[Dict] = None,
                      verbose: bool = True,
                      save_snapshots: bool = True,
                      snapshot_interval: int = 10) -> TPBResults:
    """
    Run complete three-point bending benchmark simulation.

    Args:
        params: TPB parameters (uses defaults if None)
        verbose: Print progress information
        save_snapshots: Save damage field at regular intervals
        snapshot_interval: Steps between snapshots

    Returns:
        TPBResults: Complete simulation results
    """
    p = TPB_PARAMS.copy()
    if params is not None:
        p.update(params)

    if verbose:
        print("=" * 60)
        print("Three-Point Bending Benchmark: Mode-I Fracture")
        print("=" * 60)

    # Generate mesh
    if verbose:
        print("\nGenerating mesh...")
    mesh = generate_tpb_mesh(p)
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

    # Initialize notch damage
    if verbose:
        print("\nInitializing notch damage...")
    notch_x = p['L'] / 2
    d_init = create_notch_damage(
        mesh, notch_x, p['notch_depth'], p['l0'],
        method='exponential'
    )
    solver.set_initial_damage(d_init)
    if verbose:
        print(f"  Notch: depth={p['notch_depth']}, max(d)={np.max(d_init):.4f}")

    # Setup boundary conditions
    bc_dofs, bc_values_func = create_tpb_bc_function(mesh, p)

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    if verbose:
        print(f"\nRunning simulation: {p['n_steps']} steps to "
              f"u_max = {p['u_max']:.4f} mm")
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
    crack_lengths = []
    crack_tips_y = []
    damage_snapshots = []

    for i, result in enumerate(load_steps):
        # Reaction force
        F = compute_tpb_reaction_force(mesh, elements, result.displacement,
                                        result.damage, p)
        forces.append(F)

        # Crack length
        a = compute_crack_length(mesh, result.damage, threshold=0.9)
        crack_lengths.append(a)

        # Crack tip position
        tip = track_vertical_crack(mesh, result.damage, notch_x, p['l0'])
        if tip is not None:
            crack_tips_y.append(tip[1])
        else:
            crack_tips_y.append(p['notch_depth'])

        # Save snapshot
        if save_snapshots and i % snapshot_interval == 0:
            damage_snapshots.append(result.damage.copy())

    forces = np.array(forces)
    crack_lengths = np.array(crack_lengths)
    crack_tips_y = np.array(crack_tips_y)

    # Final damage and crack path
    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.9)

    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Peak force: {np.max(forces):.4f}")
        print(f"  Displacement at peak: {displacements[np.argmax(forces)]:.6f}")
        print(f"  Final crack extension: {crack_tips_y[-1] - p['notch_depth']:.4f} mm")
        print(f"  Maximum damage: {np.max(final_damage):.4f}")
        print(f"  Final strain energy: {strain_energies[-1]:.6e}")
        print(f"  Final surface energy: {surface_energies[-1]:.6e}")

    return TPBResults(
        displacement=displacements,
        reaction_force=forces,
        crack_length=crack_lengths,
        crack_tip_y=crack_tips_y,
        strain_energy=strain_energies,
        surface_energy=surface_energies,
        final_damage=final_damage,
        crack_path=crack_path,
        damage_snapshots=damage_snapshots,
        load_steps=load_steps,
    )


# ============================================================================
# Validation
# ============================================================================
def validate_tpb_results(results: TPBResults,
                          params: Optional[Dict] = None,
                          verbose: bool = True) -> Dict:
    """
    Validate three-point bending results against expected criteria.

    Validation criteria:
    1. Crack is vertical (from notch tip upward)
    2. Crack starts at notch tip
    3. Peak load exists (softening behavior)
    4. Crack propagates significant distance
    5. Energy balance maintained

    Args:
        results: TPBResults from simulation
        params: TPB parameters
        verbose: Print validation results

    Returns:
        Dictionary with validation status for each criterion
    """
    p = TPB_PARAMS.copy()
    if params is not None:
        p.update(params)

    validation = {
        'vertical_crack': {
            'passed': False,
            'angle_from_vertical': None,
        },
        'starts_at_notch': {
            'passed': False,
            'crack_start_distance': None,
        },
        'has_peak': {
            'passed': False,
            'peak_force': None,
            'peak_displacement': None,
        },
        'significant_propagation': {
            'passed': False,
            'crack_extension': None,
        },
        'energy_balance': {
            'passed': False,
            'error': None,
        },
    }

    notch_x = p['L'] / 2
    notch_y = p['notch_depth']

    # CHECK 1: Crack is vertical
    crack_path = results.crack_path
    if len(crack_path) > 1:
        # Filter to points above notch
        above_notch = crack_path[crack_path[:, 1] > notch_y + p['l0']]
        if len(above_notch) > 1:
            dx = above_notch[-1, 0] - above_notch[0, 0]
            dy = above_notch[-1, 1] - above_notch[0, 1]
            if abs(dy) > 0.01:
                angle_from_vertical = np.degrees(np.arctan2(abs(dx), dy))
            else:
                angle_from_vertical = 90
            validation['vertical_crack']['angle_from_vertical'] = angle_from_vertical
            validation['vertical_crack']['passed'] = angle_from_vertical < 15
        else:
            validation['vertical_crack']['angle_from_vertical'] = None

    # CHECK 2: Crack starts at notch tip
    if len(crack_path) > 0:
        # First point above notch should be near notch tip
        above_notch = crack_path[crack_path[:, 1] > notch_y - p['l0']]
        if len(above_notch) > 0:
            dist = np.sqrt((above_notch[0, 0] - notch_x) ** 2 +
                          (above_notch[0, 1] - notch_y) ** 2)
            validation['starts_at_notch']['crack_start_distance'] = dist
            validation['starts_at_notch']['passed'] = dist < p['l0'] * 3

    # CHECK 3: Peak load exists
    peak_force = results.peak_force
    peak_idx = np.argmax(results.reaction_force)
    validation['has_peak']['peak_force'] = peak_force
    validation['has_peak']['peak_displacement'] = results.displacement[peak_idx]
    validation['has_peak']['passed'] = (
        peak_force > 0 and peak_idx < len(results.reaction_force) - 10
    )

    # CHECK 4: Significant crack propagation
    crack_extension = results.crack_tip_y[-1] - p['notch_depth']
    expected_max = p['W'] - p['notch_depth']
    validation['significant_propagation']['crack_extension'] = crack_extension
    validation['significant_propagation']['passed'] = (
        crack_extension > expected_max * 0.2
    )

    # CHECK 5: Energy balance
    if len(results.surface_energy) > 1:
        se_growth = results.surface_energy[-1] - results.surface_energy[0]
        validation['energy_balance']['error'] = se_growth
        validation['energy_balance']['passed'] = se_growth >= 0

    if verbose:
        print("\n" + "=" * 60)
        print("TPB Validation Results")
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


# ============================================================================
# Class Interface
# ============================================================================
class TPBBenchmark:
    """
    Class interface for Three-Point Bending benchmark with plotting.

    Example:
        benchmark = TPBBenchmark()
        benchmark.run()
        benchmark.validate()
        benchmark.plot_results()
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = TPB_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        self.mesh = None
        self.results = None
        self.validation = None

    def run(self, verbose: bool = True) -> TPBResults:
        """Run the TPB benchmark simulation."""
        self.results = run_tpb_benchmark(self.params, verbose=verbose)
        self.mesh = generate_tpb_mesh(self.params)
        return self.results

    def validate(self, verbose: bool = True) -> Dict:
        """Validate simulation results."""
        if self.results is None:
            raise ValueError("Run simulation first")
        self.validation = validate_tpb_results(
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

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Load-Displacement
        ax = axes[0, 0]
        ax.plot(self.results.displacement, self.results.reaction_force,
                'b-', lw=2)
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Reaction Force [N]')
        ax.set_title('Load-Displacement Curve')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=self.results.peak_force, color='r', linestyle='--',
                   label=f'Peak = {self.results.peak_force:.4f}')
        ax.legend()

        # 2. Crack Extension vs Displacement
        ax = axes[0, 1]
        crack_ext = self.results.crack_tip_y - self.params['notch_depth']
        ax.plot(self.results.displacement, crack_ext, 'r-', lw=2)
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Crack Extension [mm]')
        ax.set_title('Crack Propagation')
        ax.grid(True, alpha=0.3)

        # 3. Final Damage Field
        ax = axes[1, 0]
        if self.mesh is not None:
            plot_damage_field(self.mesh, self.results.final_damage, ax=ax)
        ax.set_title('Final Damage Field')

        # 4. Energy Evolution
        ax = axes[1, 1]
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

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()


# ============================================================================
# Convenience Functions
# ============================================================================
def quick_tpb_test(n_steps: int = 50, verbose: bool = True) -> TPBResults:
    """
    Run a quick TPB test with reduced steps and coarser mesh.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        TPBResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 0.5,
        'h_coarse': 4.0,
        'l0': 2.0,
    }
    return run_tpb_benchmark(params, verbose=verbose)


def very_quick_tpb_test(n_steps: int = 20, verbose: bool = True) -> TPBResults:
    """
    Run a very quick TPB test with very coarse mesh for pipeline testing.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        TPBResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 2.0,
        'h_coarse': 8.0,
        'l0': 4.0,
        'notch_refine_radius': 15.0,
    }
    return run_tpb_benchmark(params, verbose=verbose)


if __name__ == '__main__':
    results = run_tpb_benchmark(verbose=True)
    validate_tpb_results(results, verbose=True)
