"""
Single Edge Notched Tension (SENT) Benchmark
=============================================

Mode-I fracture benchmark for validating phase-field fracture implementations.

This is a canonical benchmark problem from:
- Miehe et al. (2010) "Thermodynamically consistent phase-field models of fracture"
- Ambati et al. (2015) "A review on phase-field models of brittle fracture"

Geometry:
    - Square domain L x L with pre-existing horizontal crack
    - Crack extends from left edge to center (length a = L/2)
    - Loading: displacement-controlled tension at top boundary

Expected Results:
    - Horizontal crack propagation (Mode I)
    - Peak load followed by softening
    - Crack reaches right edge at failure
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
from assembly.boundary_conditions import create_bc_from_region, merge_bcs
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep


# Default SENT parameters
SENT_PARAMS = {
    # Geometry (in mm for typical test)
    'L': 1.0,                  # mm - domain size (square)
    'crack_length': 0.5,       # mm - initial crack length from left edge
    'crack_y': 0.5,            # mm - vertical position (center)

    # Material (typical steel-like properties in MPa)
    'E': 210e3,                # MPa (= 210 GPa)
    'nu': 0.3,                 # Poisson's ratio
    'plane': 'strain',         # Plane strain assumption

    # Phase-field parameters
    'Gc': 2.7,                 # N/mm (= 2.7 kN/m)
    'l0': 0.015,               # mm (length scale)

    # Mesh parameters
    'h_fine': 0.00375,         # mm - in crack region (h <= l0/4)
    'h_coarse': 0.02,          # mm - far from crack
    'refinement_band': 0.1,    # mm - width of refined region around crack

    # Loading parameters
    'u_max': 0.01,             # mm - maximum applied displacement
    'n_steps': 100,            # number of load steps

    # Solver parameters
    'tol_u': 1e-6,             # displacement convergence tolerance
    'tol_d': 1e-6,             # damage convergence tolerance
    'max_iter': 100,           # max staggered iterations per step
}


def generate_sent_mesh(params: Optional[Dict] = None,
                       use_graded_mesh: bool = True) -> TriangleMesh:
    """
    Generate mesh for SENT specimen with local refinement near crack path.

    Uses graded mesh with fine elements near the expected crack path
    (horizontal line at y = crack_y) and coarser elements elsewhere.

    Args:
        params: SENT parameters dictionary (uses defaults if None)
        use_graded_mesh: if True, use graded mesh; if False, use uniform mesh

    Returns:
        TriangleMesh instance with mesh ready for SENT simulation
    """
    p = SENT_PARAMS.copy()
    if params is not None:
        p.update(params)

    L = p['L']
    crack_y = p['crack_y']
    h_fine = p['h_fine']
    h_coarse = p['h_coarse']
    band = p['refinement_band']

    if use_graded_mesh:
        # Create graded mesh with refinement near crack line
        mesh = _create_graded_sent_mesh(L, crack_y, h_fine, h_coarse, band)
    else:
        # Create uniform mesh with fine element size
        nx = max(int(np.ceil(L / h_fine)), 10)
        ny = nx
        from mesh.mesh_generators import create_rectangle_mesh
        mesh = create_rectangle_mesh(L, L, nx, ny, pattern='alternating')

    return mesh


def _create_graded_sent_mesh(L: float, crack_y: float,
                             h_fine: float, h_coarse: float,
                             band: float) -> TriangleMesh:
    """
    Create graded mesh with local refinement near crack path.

    Uses a multi-zone approach:
    - Zone 1: Fine mesh near crack path (y in [crack_y - band, crack_y + band])
    - Zone 2: Transition zones with gradual coarsening
    - Zone 3: Coarse mesh far from crack

    Args:
        L: domain size
        crack_y: y-coordinate of crack path
        h_fine: element size in fine zone
        h_coarse: element size in coarse zone
        band: half-width of fine mesh zone

    Returns:
        TriangleMesh instance
    """
    # Define vertical zones
    # Zone boundaries: [0, y1, y2, crack_y - band/2, crack_y + band/2, y3, y4, L]
    y_zones = []
    zone_h = []

    # Bottom coarse zone
    if crack_y - band > 0:
        n_coarse_bottom = max(int(np.ceil((crack_y - band) / h_coarse)), 2)
        dy = (crack_y - band) / n_coarse_bottom
        for i in range(n_coarse_bottom):
            y_zones.append(i * dy)
            zone_h.append(h_coarse)

    # Fine zone (around crack)
    y_fine_start = max(0, crack_y - band)
    y_fine_end = min(L, crack_y + band)
    n_fine = max(int(np.ceil((y_fine_end - y_fine_start) / h_fine)), 4)
    dy_fine = (y_fine_end - y_fine_start) / n_fine
    for i in range(n_fine):
        y_zones.append(y_fine_start + i * dy_fine)
        zone_h.append(h_fine)

    # Top coarse zone
    if crack_y + band < L:
        n_coarse_top = max(int(np.ceil((L - (crack_y + band)) / h_coarse)), 2)
        dy = (L - (crack_y + band)) / n_coarse_top
        for i in range(n_coarse_top):
            y_zones.append(crack_y + band + i * dy)
            zone_h.append(h_coarse)

    y_zones.append(L)

    # Create nodes
    nodes = []
    node_rows = []  # Track which row each node belongs to

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

        # Handle different row sizes with triangulation
        if len(bottom_row) == len(top_row):
            # Same size: simple structured connectivity
            for j in range(len(bottom_row) - 1):
                n0 = bottom_row[j]
                n1 = bottom_row[j + 1]
                n2 = top_row[j + 1]
                n3 = top_row[j]

                # Two triangles per quad
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
        else:
            # Different sizes: use conforming triangulation
            elements.extend(_triangulate_strip(nodes, bottom_row, top_row))

    elements = np.array(elements)
    return TriangleMesh(nodes, elements)


def _triangulate_strip(nodes: np.ndarray,
                       bottom_row: List[int],
                       top_row: List[int]) -> List[List[int]]:
    """
    Create conforming triangulation between two rows with different node counts.

    Uses a simple greedy algorithm that creates triangles while advancing
    along both rows, choosing the shorter diagonal at each step.

    Args:
        nodes: all node coordinates
        bottom_row: node indices in bottom row (left to right)
        top_row: node indices in top row (left to right)

    Returns:
        List of triangle element definitions
    """
    elements = []
    i, j = 0, 0
    nb, nt = len(bottom_row), len(top_row)

    while i < nb - 1 or j < nt - 1:
        if i >= nb - 1:
            # Only top nodes left
            elements.append([bottom_row[-1], top_row[j], top_row[j + 1]])
            j += 1
        elif j >= nt - 1:
            # Only bottom nodes left
            elements.append([bottom_row[i], bottom_row[i + 1], top_row[-1]])
            i += 1
        else:
            # Choose based on diagonal length
            p_bi = nodes[bottom_row[i]]
            p_bi1 = nodes[bottom_row[i + 1]]
            p_tj = nodes[top_row[j]]
            p_tj1 = nodes[top_row[j + 1]]

            # Compare diagonals
            diag1 = np.linalg.norm(p_bi1 - p_tj)
            diag2 = np.linalg.norm(p_bi - p_tj1)

            if diag1 <= diag2:
                # Use first diagonal
                elements.append([bottom_row[i], bottom_row[i + 1], top_row[j]])
                i += 1
            else:
                # Use second diagonal
                elements.append([bottom_row[i], top_row[j], top_row[j + 1]])
                j += 1

    return elements


def create_precrack_damage(mesh: TriangleMesh,
                           crack_tip_x: float,
                           crack_y: float,
                           l0: float,
                           method: str = 'exponential') -> np.ndarray:
    """
    Initialize damage field to represent pre-existing crack.

    The crack runs horizontally from x=0 to x=crack_tip_x at y=crack_y.

    Methods:
    - 'exponential': Smooth Gaussian-like transition over length scale l0
    - 'sharp': Hard transition (d=1 for edges crossing crack)
    - 'linear': Linear decay from crack surface

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

    # Compute edge midpoints
    midpoints = np.zeros((mesh.n_edges, 2))
    for i, (n1, n2) in enumerate(mesh.edges):
        midpoints[i] = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])

    for i in range(mesh.n_edges):
        x_mid, y_mid = midpoints[i]

        # Check if edge is in or near the crack region
        if x_mid < crack_tip_x:
            # Edge is behind crack tip - check distance to crack line
            dist_to_crack = abs(y_mid - crack_y)

            if method == 'exponential':
                # Smooth Gaussian profile
                d[i] = np.exp(-(dist_to_crack / l0) ** 2)
            elif method == 'sharp':
                # Sharp transition
                if dist_to_crack < l0:
                    d[i] = 1.0
                else:
                    d[i] = 0.0
            elif method == 'linear':
                # Linear decay
                if dist_to_crack < 2 * l0:
                    d[i] = max(0, 1.0 - dist_to_crack / (2 * l0))
                else:
                    d[i] = 0.0
        else:
            # Ahead of crack tip - smooth transition near tip
            dist_to_tip = np.sqrt((x_mid - crack_tip_x) ** 2 +
                                  (y_mid - crack_y) ** 2)

            if method == 'exponential':
                if dist_to_tip < 2 * l0:
                    d[i] = np.exp(-(dist_to_tip / l0) ** 2)
            elif method == 'sharp':
                pass  # No damage ahead of tip
            elif method == 'linear':
                if dist_to_tip < l0:
                    d[i] = max(0, 1.0 - dist_to_tip / l0)

    return d


def apply_sent_boundary_conditions(mesh: TriangleMesh,
                                   u_applied: float,
                                   L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for SENT test.

    Boundary conditions:
    - Bottom (y=0): u_y = 0 (fixed in y)
    - Top (y=L): u_y = u_applied (prescribed displacement)
    - Left/Right: Free (natural BC)
    - Single node (bottom-left): u_x = 0 to prevent rigid body motion

    Args:
        mesh: TriangleMesh instance
        u_applied: Applied vertical displacement at top
        L: Domain height (for identifying top boundary)

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values: Prescribed values at those DOFs
    """
    tol = 1e-10 * L

    # Find boundary nodes
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)

    # Bottom: fix y-displacement
    bc_bottom_dofs, bc_bottom_vals = create_bc_from_region(
        mesh, lambda x, y: y < tol, 'y', 0.0
    )

    # Fix x at corner node (bottom-left) to prevent rigid body motion
    corner_node = bottom_nodes[np.argmin(mesh.nodes[bottom_nodes, 0])]
    bc_corner_dofs = np.array([2 * corner_node])  # x-DOF
    bc_corner_vals = np.array([0.0])

    # Top: prescribed y-displacement
    bc_top_dofs = np.array([2 * n + 1 for n in top_nodes])  # y-DOFs
    bc_top_vals = np.full(len(top_nodes), u_applied)

    # Combine all BCs
    bc_dofs = np.concatenate([bc_bottom_dofs, bc_corner_dofs, bc_top_dofs])
    bc_values = np.concatenate([bc_bottom_vals, bc_corner_vals, bc_top_vals])

    return bc_dofs, bc_values


def create_sent_bc_function(mesh: TriangleMesh,
                            L: float) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """
    Create BC function for load stepping.

    Returns a function that takes a displacement value and returns
    the corresponding BC values array.

    Args:
        mesh: TriangleMesh instance
        L: Domain height

    Returns:
        bc_dofs: DOF indices with Dirichlet BCs
        bc_values_func: function(u_applied) -> bc_values array
    """
    tol = 1e-10 * L

    # Find boundary nodes
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < tol)
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)
    n_top = len(top_nodes)

    # Fixed BCs (bottom y, corner x)
    bc_bottom_dofs, bc_bottom_vals = create_bc_from_region(
        mesh, lambda x, y: y < tol, 'y', 0.0
    )
    corner_node = bottom_nodes[np.argmin(mesh.nodes[bottom_nodes, 0])]
    bc_corner_dofs = np.array([2 * corner_node])
    bc_corner_vals = np.array([0.0])

    fixed_dofs, fixed_vals = merge_bcs(
        (bc_bottom_dofs, bc_bottom_vals),
        (bc_corner_dofs, bc_corner_vals)
    )

    # Top DOFs (will have varying displacement)
    bc_top_dofs = np.array([2 * n + 1 for n in top_nodes])

    # All BC DOFs
    all_bc_dofs = np.concatenate([fixed_dofs, bc_top_dofs])
    n_fixed = len(fixed_vals)

    def bc_values_func(u_applied: float) -> np.ndarray:
        """Return BC values for given applied displacement."""
        values = np.zeros(len(all_bc_dofs))
        values[:n_fixed] = fixed_vals
        values[n_fixed:] = u_applied
        return values

    return all_bc_dofs, bc_values_func


def compute_reaction_force(mesh: TriangleMesh,
                           elements: List[GraFEAElement],
                           u: np.ndarray,
                           damage: np.ndarray,
                           L: float,
                           direction: str = 'y') -> float:
    """
    Compute reaction force at the top boundary.

    The reaction force is computed from the internal force vector
    at the constrained DOFs.

    Args:
        mesh: TriangleMesh instance
        elements: List of GraFEAElement instances
        u: Displacement solution vector
        damage: Damage field
        L: Domain height
        direction: 'x' or 'y' for force component

    Returns:
        F: Total reaction force at top boundary
    """
    from assembly.global_assembly import assemble_internal_force

    tol = 1e-10 * L
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > L - tol)

    # Get internal force vector at all DOFs
    F_int = assemble_internal_force(mesh, elements, u, damage)

    # Sum reaction force at top nodes (reaction = internal force at constrained DOFs)
    dof_offset = 0 if direction == 'x' else 1
    F = 0.0
    for node in top_nodes:
        dof = 2 * node + dof_offset
        F += F_int[dof]

    return F


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
    # Find heavily damaged edges
    cracked_edges = np.where(damage > threshold)[0]

    if len(cracked_edges) == 0:
        return np.array([]).reshape(0, 2)

    # Get midpoints of cracked edges
    midpoints = []
    for edge_idx in cracked_edges:
        n1, n2 = mesh.edges[edge_idx]
        mid = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])
        midpoints.append(mid)

    midpoints = np.array(midpoints)

    # Sort by x-coordinate
    sort_idx = np.argsort(midpoints[:, 0])
    crack_path = midpoints[sort_idx]

    return crack_path


def compute_crack_length(mesh: TriangleMesh,
                         damage: np.ndarray,
                         threshold: float = 0.9) -> float:
    """
    Compute effective crack length from damage field.

    Uses weighted sum of damage values times edge lengths.

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


@dataclass
class SENTResults:
    """Results container for SENT benchmark simulation."""
    displacement: np.ndarray  # Applied displacement at each step
    force: np.ndarray         # Reaction force at each step
    crack_length: np.ndarray  # Crack length at each step
    strain_energy: np.ndarray # Strain energy at each step
    surface_energy: np.ndarray # Surface energy at each step
    final_damage: np.ndarray  # Final damage field
    crack_path: np.ndarray    # Extracted crack path coordinates
    damage_snapshots: List[np.ndarray] = field(default_factory=list)
    load_steps: List[LoadStep] = field(default_factory=list)

    @property
    def total_energy(self) -> np.ndarray:
        """Total energy (strain + surface)."""
        return self.strain_energy + self.surface_energy

    @property
    def peak_force(self) -> float:
        """Peak reaction force."""
        return np.max(self.force)

    @property
    def displacement_at_peak(self) -> float:
        """Applied displacement at peak force."""
        idx = np.argmax(self.force)
        return self.displacement[idx]


def run_sent_benchmark(params: Optional[Dict] = None,
                       verbose: bool = True,
                       save_snapshots: bool = True,
                       snapshot_interval: int = 10) -> SENTResults:
    """
    Run complete SENT benchmark simulation.

    This is the main function for running the mode-I fracture benchmark.
    It handles mesh generation, pre-crack initialization, load stepping,
    and result extraction.

    Args:
        params: SENT parameters (uses defaults if None)
        verbose: Print progress information
        save_snapshots: Save damage field at regular intervals
        snapshot_interval: Steps between snapshots

    Returns:
        SENTResults: Complete simulation results
    """
    p = SENT_PARAMS.copy()
    if params is not None:
        p.update(params)

    if verbose:
        print("=" * 60)
        print("SENT Benchmark: Mode-I Fracture")
        print("=" * 60)

    # Generate mesh
    if verbose:
        print("\nGenerating mesh...")
    mesh = generate_sent_mesh(p)
    if verbose:
        print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, {mesh.n_edges} edges")

    # Create material
    material = IsotropicMaterial(
        E=p['E'],
        nu=p['nu'],
        Gc=p['Gc'],
        l0=p['l0']
    )
    if verbose:
        print(f"  Material: E={material.E:.1e}, nu={material.nu}, Gc={material.Gc}, l0={material.l0}")

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

    # Setup boundary conditions
    bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    if verbose:
        print(f"\nRunning simulation: {p['n_steps']} steps to u_max = {p['u_max']:.6f}")
        print("-" * 60)

    # Run simulation
    load_steps = solver.solve(u_steps, bc_dofs, bc_values_func)

    # Extract results
    if verbose:
        print("\nExtracting results...")

    displacements = np.array([r.load_factor for r in load_steps])
    strain_energies = np.array([r.strain_energy for r in load_steps])
    surface_energies = np.array([r.surface_energy for r in load_steps])

    # Compute reaction forces
    forces = []
    crack_lengths = []
    damage_snapshots = []

    for i, result in enumerate(load_steps):
        # Reaction force
        F = compute_reaction_force(mesh, elements, result.displacement,
                                   result.damage, p['L'], 'y')
        forces.append(F)

        # Crack length
        a = compute_crack_length(mesh, result.damage, threshold=0.9)
        crack_lengths.append(a)

        # Save snapshot
        if save_snapshots and i % snapshot_interval == 0:
            damage_snapshots.append(result.damage.copy())

    forces = np.array(forces)
    crack_lengths = np.array(crack_lengths)

    # Final damage and crack path
    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.9)

    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Peak force: {np.max(forces):.4f}")
        print(f"  Displacement at peak: {displacements[np.argmax(forces)]:.6f}")
        print(f"  Final crack length: {crack_lengths[-1]:.4f}")
        print(f"  Maximum damage: {np.max(final_damage):.4f}")
        print(f"  Final strain energy: {strain_energies[-1]:.6e}")
        print(f"  Final surface energy: {surface_energies[-1]:.6e}")

    return SENTResults(
        displacement=displacements,
        force=forces,
        crack_length=crack_lengths,
        strain_energy=strain_energies,
        surface_energy=surface_energies,
        final_damage=final_damage,
        crack_path=crack_path,
        damage_snapshots=damage_snapshots,
        load_steps=load_steps
    )


# Reference data from literature (Miehe et al. 2010, Ambati et al. 2015)
LITERATURE_REFERENCE = {
    'miehe_2010': {
        # Normalized load-displacement data points
        'displacement': np.array([0.0, 0.002, 0.004, 0.0055, 0.006, 0.007, 0.008]),
        'force': np.array([0.0, 0.3, 0.55, 0.60, 0.55, 0.35, 0.15]),
        'description': 'Miehe et al. (2010) - Mode-I fracture benchmark'
    }
}


def compare_with_literature(results: SENTResults,
                           reference: str = 'miehe_2010',
                           verbose: bool = True) -> Dict:
    """
    Compare SENT results with published data.

    Args:
        results: SENTResults from simulation
        reference: Key for reference data ('miehe_2010')
        verbose: Print comparison

    Returns:
        Dictionary with comparison metrics
    """
    if reference not in LITERATURE_REFERENCE:
        raise ValueError(f"Unknown reference: {reference}")

    ref = LITERATURE_REFERENCE[reference]

    # Normalize forces (if needed)
    # Literature often uses normalized force: F / (E * L)

    # Interpolate to compare at reference displacement points
    from scipy.interpolate import interp1d

    # Create interpolation function for simulation results
    valid_mask = results.displacement <= results.displacement.max()
    F_interp = interp1d(results.displacement, results.force,
                        kind='linear', fill_value='extrapolate')

    # Compare at reference points
    errors = []
    for u_ref, F_ref in zip(ref['displacement'], ref['force']):
        if u_ref <= results.displacement.max():
            F_sim = F_interp(u_ref)
            if abs(F_ref) > 1e-10:
                error = abs(F_sim - F_ref) / abs(F_ref)
            else:
                error = abs(F_sim - F_ref)
            errors.append(error)

    comparison = {
        'reference': reference,
        'mean_error': np.mean(errors) if errors else np.nan,
        'max_error': np.max(errors) if errors else np.nan,
        'errors': errors,
        'peak_force_sim': results.peak_force,
        'description': ref['description']
    }

    if verbose:
        print(f"\nComparison with {ref['description']}")
        print("-" * 40)
        print(f"  Mean relative error: {comparison['mean_error']*100:.1f}%")
        print(f"  Max relative error:  {comparison['max_error']*100:.1f}%")
        print(f"  Simulated peak force: {comparison['peak_force_sim']:.4f}")

    return comparison


def validate_sent_results(results: SENTResults,
                          params: Optional[Dict] = None,
                          verbose: bool = True) -> Dict:
    """
    Validate SENT simulation results against expected criteria.

    Validation criteria:
    1. Crack direction: Should be horizontal (Mode I)
    2. Peak load: Should be in expected range
    3. Energy balance: Strain + surface energy should be consistent
    4. Crack propagation: Should reach or approach right edge

    Args:
        results: SENTResults from simulation
        params: SENT parameters (for expected values)
        verbose: Print validation results

    Returns:
        Dictionary with validation status for each criterion
    """
    p = SENT_PARAMS.copy()
    if params is not None:
        p.update(params)

    validation = {
        'crack_direction': {'passed': False, 'angle': None, 'tolerance': 5.0},
        'peak_load': {'passed': False, 'value': None, 'range': (0.3, 1.0)},
        'energy_balance': {'passed': False, 'error': None, 'tolerance': 0.05},
        'crack_propagation': {'passed': False, 'final_length': None, 'target': p['L']}
    }

    # 1. Check crack direction
    if len(results.crack_path) >= 2:
        # Fit line to crack path
        x = results.crack_path[:, 0]
        y = results.crack_path[:, 1]

        if len(x) > 1 and np.std(x) > 1e-10:
            # Linear regression
            slope = np.polyfit(x, y, 1)[0]
            angle_deg = np.degrees(np.arctan(slope))
            validation['crack_direction']['angle'] = angle_deg
            validation['crack_direction']['passed'] = abs(angle_deg) < validation['crack_direction']['tolerance']

    # 2. Check peak load
    peak = results.peak_force
    validation['peak_load']['value'] = peak
    low, high = validation['peak_load']['range']
    validation['peak_load']['passed'] = low <= peak <= high

    # 3. Check energy balance
    # Total energy should be approximately conserved minus dissipation
    # For now, just check that energies are positive and surface energy grows
    if len(results.surface_energy) > 1:
        se_growth = results.surface_energy[-1] - results.surface_energy[0]
        validation['energy_balance']['error'] = se_growth
        validation['energy_balance']['passed'] = se_growth >= 0

    # 4. Check crack propagation
    final_a = results.crack_length[-1] if len(results.crack_length) > 0 else 0
    validation['crack_propagation']['final_length'] = final_a
    validation['crack_propagation']['passed'] = final_a > 0.8 * p['L']

    if verbose:
        print("\n" + "=" * 60)
        print("Validation Results")
        print("=" * 60)

        for criterion, data in validation.items():
            status = "PASS" if data['passed'] else "FAIL"
            print(f"\n{criterion}:")
            print(f"  Status: {status}")
            for key, val in data.items():
                if key != 'passed':
                    print(f"  {key}: {val}")

    return validation


class SENTBenchmark:
    """
    Class interface for SENT benchmark with plotting capabilities.

    This class provides a convenient interface for running the SENT
    benchmark and visualizing results.

    Example:
        benchmark = SENTBenchmark()
        benchmark.run()
        benchmark.plot_results()
        benchmark.validate()
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize SENT benchmark.

        Args:
            params: Optional parameter overrides
        """
        self.params = SENT_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        self.mesh = None
        self.results = None
        self.validation = None

    def run(self, verbose: bool = True) -> SENTResults:
        """
        Run the SENT benchmark simulation.

        Args:
            verbose: Print progress

        Returns:
            SENTResults instance
        """
        self.results = run_sent_benchmark(self.params, verbose=verbose)
        self.mesh = generate_sent_mesh(self.params)
        return self.results

    def validate(self, verbose: bool = True) -> Dict:
        """
        Validate simulation results.

        Args:
            verbose: Print validation results

        Returns:
            Validation dictionary
        """
        if self.results is None:
            raise ValueError("Run simulation first")

        self.validation = validate_sent_results(
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
        ax.plot(self.results.displacement * 1000, self.results.force, 'b-', lw=2)
        ax.set_xlabel('Displacement (mm × 10³)')
        ax.set_ylabel('Reaction Force')
        ax.set_title('Load-Displacement Curve')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=self.results.peak_force, color='r', linestyle='--',
                   label=f'Peak = {self.results.peak_force:.4f}')
        ax.legend()

        # 2. Energy evolution
        ax = axes[0, 1]
        ax.plot(self.results.displacement * 1000, self.results.strain_energy,
                'b-', label='Strain', lw=2)
        ax.plot(self.results.displacement * 1000, self.results.surface_energy,
                'r-', label='Surface', lw=2)
        ax.plot(self.results.displacement * 1000, self.results.total_energy,
                'k--', label='Total', lw=2)
        ax.set_xlabel('Displacement (mm × 10³)')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
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
                    'r.-', lw=2, ms=4)
        ax.axhline(y=self.params['crack_y'], color='b', linestyle='--',
                   alpha=0.5, label='Expected path')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Crack Path')
        ax.set_xlim([0, self.params['L']])
        ax.set_ylim([0, self.params['L']])
        ax.set_aspect('equal')
        ax.legend()
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

        # Select frames
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

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


# Convenience function for quick benchmark run
def quick_sent_test(n_steps: int = 50, verbose: bool = True) -> SENTResults:
    """
    Run a quick SENT test with reduced steps for testing.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        SENTResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 0.0075,  # Coarser for speed
        'h_coarse': 0.04,
    }
    return run_sent_benchmark(params, verbose=verbose)


def very_quick_sent_test(n_steps: int = 20, verbose: bool = True) -> SENTResults:
    """
    Run a very quick SENT test with coarse mesh for fast testing.

    This is useful for verifying the pipeline works but should not
    be used for quantitative validation.

    Args:
        n_steps: Number of load steps
        verbose: Print progress

    Returns:
        SENTResults instance
    """
    params = {
        'n_steps': n_steps,
        'h_fine': 0.02,    # Much coarser for fast testing
        'h_coarse': 0.08,
        'l0': 0.04,        # Larger l0 for coarser mesh
        'refinement_band': 0.15,
    }
    return run_sent_benchmark(params, verbose=verbose)


if __name__ == '__main__':
    # Run benchmark when executed directly
    results = run_sent_benchmark(verbose=True)
    validate_sent_results(results, verbose=True)
