"""
Length Scale Study
==================

Parametric study of the phase-field length scale l0 on fracture behavior.

Characterizes:
- Damage band width vs l0
- Peak load vs l0 (brittleness transition)
- Dissipated energy vs l0
"""

import numpy as np
import json
import sys
import os
from typing import Dict, Optional, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep
from benchmarks.sent_benchmark import (
    SENT_PARAMS, generate_sent_mesh, create_precrack_damage,
    create_sent_bc_function, compute_reaction_force, extract_crack_path,
)

LENGTH_SCALE_STUDY_PARAMS = {
    'benchmark': 'SENT',
    'h_fine': 0.00375,       # Fixed fine mesh
    'l0_values': [
        0.00375,   # l0 = h  (minimum recommended)
        0.0075,    # l0 = 2h
        0.015,     # l0 = 4h (standard)
        0.030,     # l0 = 8h (wide band)
    ],
    # SENT base parameters
    'E': 210e3,
    'nu': 0.3,
    'Gc': 2.7,
    'L': 1.0,
    'crack_length': 0.5,
    'crack_y': 0.5,
    'h_coarse': 0.02,
    'refinement_band': 0.1,
    'plane': 'strain',
    'u_max': 0.01,
    'n_steps': 100,
    'tol_u': 1e-6,
    'tol_d': 1e-6,
    'max_iter': 100,
}


def measure_damage_band_width(damage, mesh, threshold=0.5, sample_x=None):
    """
    Measure the width of the damage band perpendicular to crack direction.

    Parameters
    ----------
    damage : np.ndarray
        Edge damage values, shape (n_edges,).
    mesh : TriangleMesh
        Mesh with edge midpoint coordinates.
    threshold : float
        Damage level defining the band boundary.
    sample_x : float, optional
        x-coordinate to sample. Default: 0.75 (behind crack tip for SENT).

    Returns
    -------
    float
        Width of the damage band at the sample location.
    """
    midpoints = mesh.edge_midpoints

    if sample_x is None:
        sample_x = 0.75

    # Find edges near the sampling line
    tol = np.max(mesh.edge_lengths) * 2
    near_mask = np.abs(midpoints[:, 0] - sample_x) < tol
    near_edges = np.where(near_mask)[0]

    if len(near_edges) == 0:
        return 0.0

    # Among those, find edges with d > threshold
    damaged_mask = damage[near_edges] > threshold
    damaged = near_edges[damaged_mask]

    if len(damaged) == 0:
        return 0.0

    y_values = midpoints[damaged, 1]
    width = np.max(y_values) - np.min(y_values)

    return width


def run_single_l0(params, l0, verbose=False):
    """
    Run SENT benchmark with a specific length scale l0.

    Parameters
    ----------
    params : dict
        Base parameters.
    l0 : float
        Phase-field length scale.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results including displacement, force, final_damage, crack_path,
        strain_energy, surface_energy.
    """
    p = params.copy()
    p['l0'] = l0

    # Adjust refinement band to accommodate larger l0
    p['refinement_band'] = max(p.get('refinement_band', 0.1), 4 * l0)

    if verbose:
        print(f"\n--- Running SENT with l0 = {l0:.6f} (h/l0 = {p['h_fine']/l0:.2f}) ---")

    # Generate mesh
    mesh = generate_sent_mesh(p)
    if verbose:
        print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"{mesh.n_edges} edges")

    # Material
    material = IsotropicMaterial(E=p['E'], nu=p['nu'], Gc=p['Gc'], l0=l0)

    # Elements
    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material,
                              plane=p['plane'])
                for e in range(mesh.n_elements)]

    # Edge graph
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')

    # Solver
    config = SolverConfig(
        tol_u=p['tol_u'],
        tol_d=p['tol_d'],
        max_stagger_iter=p['max_iter'],
        verbose=verbose,
    )
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # Pre-crack
    d_init = create_precrack_damage(
        mesh, p['crack_length'], p['crack_y'], l0, method='exponential'
    )
    solver.set_initial_damage(d_init)

    # Boundary conditions
    bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    # Solve
    load_steps = solver.solve(u_steps, bc_dofs, bc_values_func)

    # Extract results
    displacements = np.array([r.load_factor for r in load_steps])
    strain_energies = np.array([r.strain_energy for r in load_steps])
    surface_energies = np.array([r.surface_energy for r in load_steps])

    forces = []
    for result in load_steps:
        F = compute_reaction_force(mesh, elements, result.displacement,
                                   result.damage, p['L'], 'y')
        forces.append(F)
    forces = np.array(forces)

    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.9)

    return {
        'l0': l0,
        'mesh': mesh,
        'displacement': displacements,
        'force': forces,
        'final_damage': final_damage,
        'crack_path': crack_path,
        'strain_energy': strain_energies,
        'surface_energy': surface_energies,
        'peak_load': float(np.max(forces)),
        'n_nodes': mesh.n_nodes,
        'n_elements': mesh.n_elements,
        'n_edges': mesh.n_edges,
    }


def run_length_scale_study(params=None, verbose=True):
    """
    Run SENT benchmark with multiple values of l0.

    Parameters
    ----------
    params : dict, optional
        Study parameters (uses defaults if None).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results keyed by l0 value.
    """
    p = LENGTH_SCALE_STUDY_PARAMS.copy()
    if params is not None:
        p.update(params)

    l0_values = p['l0_values']
    results = {}

    if verbose:
        print("=" * 60)
        print("Length Scale Parametric Study")
        print("=" * 60)
        print(f"  h_fine = {p['h_fine']}")
        print(f"  l0 values: {l0_values}")

    for l0 in l0_values:
        result = run_single_l0(p, l0, verbose=verbose)
        results[l0] = result

    if verbose:
        print("\n" + "=" * 60)
        print("Study Complete")
        print("=" * 60)

    return results


def analyze_length_scale_effects(results):
    """
    Analyze how length scale affects fracture behavior.

    Parameters
    ----------
    results : dict
        Results from run_length_scale_study, keyed by l0.

    Returns
    -------
    dict
        Analysis results for each l0 value, plus trends.
    """
    analysis = {}

    for l0, result in results.items():
        mesh = result['mesh']
        final_damage = result['final_damage']

        # Measure damage band width at multiple x-locations
        sample_xs = [0.6, 0.7, 0.75, 0.8]
        band_widths = []
        for sx in sample_xs:
            w = measure_damage_band_width(final_damage, mesh, threshold=0.5,
                                          sample_x=sx)
            if w > 0:
                band_widths.append(w)

        mean_band_width = np.mean(band_widths) if band_widths else 0.0

        # Peak load
        peak_load = result['peak_load']

        # Dissipated energy (area under F-u curve)
        u = result['displacement']
        F = result['force']
        dissipated = float(np.trapezoid(np.maximum(F, 0), u))

        # Brittleness index
        peak_idx = np.argmax(F)
        if peak_idx > 0 and dissipated > 1e-20:
            elastic_at_peak = 0.5 * F[peak_idx] * u[peak_idx]
            brittleness = elastic_at_peak / dissipated
        else:
            brittleness = 1.0

        # Max damage achieved
        d_max = float(np.max(final_damage))

        analysis[l0] = {
            'damage_band_width': mean_band_width,
            'band_width_ratio': mean_band_width / l0 if l0 > 0 else 0.0,
            'peak_load': peak_load,
            'dissipated_energy': dissipated,
            'brittleness_index': brittleness,
            'd_max': d_max,
            'final_surface_energy': float(result['surface_energy'][-1]),
            'final_strain_energy': float(result['strain_energy'][-1]),
        }

    # Compute trends
    l0_arr = np.array(sorted(analysis.keys()))
    if len(l0_arr) >= 2:
        peaks = np.array([analysis[l0]['peak_load'] for l0 in l0_arr])
        bands = np.array([analysis[l0]['damage_band_width'] for l0 in l0_arr])
        diss = np.array([analysis[l0]['dissipated_energy'] for l0 in l0_arr])

        analysis['trends'] = {
            'l0_values': l0_arr.tolist(),
            'peak_loads': peaks.tolist(),
            'band_widths': bands.tolist(),
            'dissipated_energies': diss.tolist(),
            'peak_load_decreases_with_l0': bool(
                np.all(np.diff(peaks[np.argsort(l0_arr)]) <= 0) or
                peaks[-1] < peaks[0]
            ),
            'band_width_increases_with_l0': bool(
                np.all(np.diff(bands[np.argsort(l0_arr)]) >= 0) or
                bands[-1] > bands[0]
            ),
        }

    return analysis


def save_length_scale_results(results, analysis, filepath):
    """
    Save length scale study results to JSON.

    Parameters
    ----------
    results : dict
        Raw results (mesh objects are not serialized).
    analysis : dict
        Analysis results.
    filepath : str
        Output file path.
    """
    # Serialize results (skip non-serializable objects)
    serializable = {}
    for l0, result in results.items():
        serializable[str(l0)] = {
            'l0': l0,
            'peak_load': result['peak_load'],
            'displacement': result['displacement'].tolist(),
            'force': result['force'].tolist(),
            'strain_energy': result['strain_energy'].tolist(),
            'surface_energy': result['surface_energy'].tolist(),
            'n_nodes': result['n_nodes'],
            'n_elements': result['n_elements'],
            'n_edges': result['n_edges'],
            'crack_path': result['crack_path'].tolist()
                         if len(result['crack_path']) > 0 else [],
        }

    # Serialize analysis
    analysis_ser = {}
    for key, val in analysis.items():
        if key == 'trends':
            analysis_ser['trends'] = val
        else:
            analysis_ser[str(key)] = val

    output = {
        'results': serializable,
        'analysis': analysis_ser,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def print_length_scale_summary(analysis):
    """
    Print formatted summary of length scale study.

    Parameters
    ----------
    analysis : dict
        Analysis results from analyze_length_scale_effects.
    """
    print("\n" + "=" * 70)
    print("Length Scale Study Summary")
    print("=" * 70)
    print(f"{'l0':>10s} {'Band Width':>12s} {'W/l0':>8s} {'Peak Load':>12s} "
          f"{'Dissipated':>12s} {'Brittleness':>12s}")
    print("-" * 70)

    for key in sorted(k for k in analysis.keys() if k != 'trends'):
        l0 = key
        data = analysis[key]
        print(f"{l0:10.6f} {data['damage_band_width']:12.6f} "
              f"{data['band_width_ratio']:8.2f} {data['peak_load']:12.6f} "
              f"{data['dissipated_energy']:12.6e} {data['brittleness_index']:12.4f}")

    if 'trends' in analysis:
        t = analysis['trends']
        print("\nTrends:")
        print(f"  Peak load decreases with l0: {t.get('peak_load_decreases_with_l0', 'N/A')}")
        print(f"  Band width increases with l0: {t.get('band_width_increases_with_l0', 'N/A')}")

    print("\nExpected: Band width ~ 2-4 * l0")
    print("Expected: Larger l0 -> lower peak load, higher dissipation (more ductile)")
