"""
Computational Cost Analysis
============================

Profile computational cost as a function of problem size.
Compare GraFEA-PF vs FEM-PF scaling behavior.
"""

import numpy as np
import json
import time
import tracemalloc
import sys
import os
from typing import Dict, Optional, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from physics.surface_energy import assemble_damage_system, compute_edge_volumes
from physics.damage import HistoryField, compute_driving_force
from physics.tension_split import spectral_split
from assembly.global_assembly import (
    assemble_global_stiffness, compute_all_edge_strains,
)
from assembly.boundary_conditions import apply_dirichlet_bc, create_bc_from_region, merge_bcs
from solvers.staggered_solver import SolverConfig
from benchmarks.sent_benchmark import (
    SENT_PARAMS, generate_sent_mesh, create_precrack_damage,
    create_sent_bc_function,
)

from .fem_pf_reference import FEMPhasefieldSolver


SCALING_STUDY_PARAMS = {
    'benchmark': 'SENT',
    'mesh_node_targets': [500, 1000, 2000, 5000, 10000],
    'n_profile_steps': 10,  # Profile 10 representative steps
    # Base SENT parameters
    'E': 210e3,
    'nu': 0.3,
    'Gc': 2.7,
    'l0': 0.015,
    'L': 1.0,
    'crack_length': 0.5,
    'crack_y': 0.5,
    'plane': 'strain',
    'refinement_band': 0.1,
}


def estimate_h_for_node_count(n_target, L=1.0):
    """
    Estimate element size h to achieve approximately n_target nodes.

    For a uniform triangular mesh on [0,L]^2:
    n_nodes ~ (L/h + 1)^2, so h ~ L / sqrt(n_target)

    Parameters
    ----------
    n_target : int
        Target number of nodes.
    L : float
        Domain size.

    Returns
    -------
    float
        Estimated element size.
    """
    h = L / np.sqrt(n_target)
    return max(h, 1e-5)


def profile_grafea_pf_step(mesh, elements, material, edge_graph,
                           damage, displacement, history_H,
                           bc_dofs, bc_values, F_ext):
    """
    Profile a single GraFEA-PF load step in detail.

    Parameters
    ----------
    mesh : TriangleMesh
    elements : list of GraFEAElement
    material : IsotropicMaterial
    edge_graph : EdgeGraph
    damage : np.ndarray, shape (n_edges,)
    displacement : np.ndarray, shape (2*n_nodes,)
    history_H : np.ndarray, shape (n_edges,)
    bc_dofs : np.ndarray
    bc_values : np.ndarray
    F_ext : np.ndarray

    Returns
    -------
    dict
        Timing breakdown for each operation.
    """
    from scipy.sparse.linalg import spsolve

    timings = {}

    # 1. Stiffness assembly
    t0 = time.perf_counter()
    K = assemble_global_stiffness(mesh, elements, damage)
    timings['assembly_K'] = time.perf_counter() - t0

    # 2. Displacement solve
    t0 = time.perf_counter()
    K_bc, F_bc = apply_dirichlet_bc(K, F_ext.copy(), bc_dofs, bc_values)
    u = spsolve(K_bc, F_bc)
    timings['solve_u'] = time.perf_counter() - t0

    # 3. Edge strain computation + driving force
    t0 = time.perf_counter()
    edge_strains = compute_all_edge_strains(mesh, elements, u)
    driving_force = compute_driving_force(mesh, elements, edge_strains, material)
    H_new = np.maximum(history_H, driving_force)
    timings['update_H'] = time.perf_counter() - t0

    # 4. Damage system assembly
    t0 = time.perf_counter()
    A_d, b_d = assemble_damage_system(
        mesh, edge_graph, H_new, material.Gc, material.l0
    )
    timings['assembly_damage'] = time.perf_counter() - t0

    # 5. Damage solve
    t0 = time.perf_counter()
    d_new = spsolve(A_d, b_d)
    d_new = np.clip(d_new, 0.0, 1.0)
    d_new = np.maximum(d_new, damage)
    timings['solve_d'] = time.perf_counter() - t0

    timings['total'] = sum(timings.values())

    return timings


def profile_fem_pf_step(fem_solver, bc_dofs, bc_values, F_ext):
    """
    Profile a single FEM-PF load step in detail.

    Parameters
    ----------
    fem_solver : FEMPhasefieldSolver
    bc_dofs : np.ndarray
    bc_values : np.ndarray
    F_ext : np.ndarray

    Returns
    -------
    dict
        Timing breakdown.
    """
    from scipy.sparse.linalg import spsolve

    timings = {}

    # 1. Stiffness assembly
    t0 = time.perf_counter()
    K = fem_solver._assemble_stiffness()
    timings['assembly_K'] = time.perf_counter() - t0

    # 2. Displacement solve
    t0 = time.perf_counter()
    K_bc, F_bc = apply_dirichlet_bc(K, F_ext.copy(), bc_dofs, bc_values)
    u = spsolve(K_bc, F_bc)
    fem_solver.displacement = u
    timings['solve_u'] = time.perf_counter() - t0

    # 3. History update
    t0 = time.perf_counter()
    fem_solver._update_history()
    timings['update_H'] = time.perf_counter() - t0

    # 4. Damage system assembly
    t0 = time.perf_counter()
    A_d, b_d = fem_solver._assemble_damage_system()
    timings['assembly_damage'] = time.perf_counter() - t0

    # 5. Damage solve
    t0 = time.perf_counter()
    d_new = spsolve(A_d, b_d)
    d_new = np.clip(d_new, 0.0, 1.0)
    d_new = np.maximum(d_new, fem_solver.damage)
    fem_solver.damage = d_new
    timings['solve_d'] = time.perf_counter() - t0

    timings['total'] = sum(timings.values())

    return timings


def run_scaling_study(params=None, verbose=True):
    """
    Run scaling study across multiple mesh sizes.

    For each mesh size:
    1. Generate mesh
    2. Set up both GraFEA-PF and FEM-PF solvers
    3. Run n_profile_steps representative steps
    4. Record average timing and memory

    Parameters
    ----------
    params : dict, optional
        Study parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict
        One entry per mesh size with timing breakdown for both methods.
    """
    p = SCALING_STUDY_PARAMS.copy()
    if params is not None:
        p.update(params)

    node_targets = p['mesh_node_targets']
    n_steps = p['n_profile_steps']
    results = []

    if verbose:
        print("=" * 70)
        print("Computational Cost Scaling Study")
        print("=" * 70)

    for n_target in node_targets:
        if verbose:
            print(f"\n--- Target: {n_target} nodes ---")

        # Determine h_fine for target node count
        h_fine = estimate_h_for_node_count(n_target, p['L'])
        h_coarse = min(h_fine * 4, p.get('h_coarse', 0.02))

        mesh_params = {
            **p,
            'h_fine': h_fine,
            'h_coarse': h_coarse,
        }

        # Generate mesh
        mesh = generate_sent_mesh(mesh_params)

        if verbose:
            print(f"  h_fine={h_fine:.5f}, actual: {mesh.n_nodes} nodes, "
                  f"{mesh.n_elements} elements, {mesh.n_edges} edges")

        entry = {
            'target_nodes': n_target,
            'actual_nodes': mesh.n_nodes,
            'n_elements': mesh.n_elements,
            'n_edges': mesh.n_edges,
            'h_fine': h_fine,
        }

        # Material
        material = IsotropicMaterial(
            E=p['E'], nu=p['nu'], Gc=p['Gc'], l0=p['l0']
        )

        # Elements
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material,
                                  plane=p['plane'])
                    for e in range(mesh.n_elements)]

        # Edge graph
        edge_graph = EdgeGraph(mesh, weight_scheme='distance')

        # Pre-crack
        d_init = create_precrack_damage(
            mesh, p['crack_length'], p['crack_y'], p['l0'],
            method='exponential'
        )

        # BCs
        bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
        F_ext = np.zeros(2 * mesh.n_nodes)

        # --- Profile GraFEA-PF ---
        if verbose:
            print("  Profiling GraFEA-PF...")

        damage_gpf = d_init.copy()
        disp_gpf = np.zeros(2 * mesh.n_nodes)
        history_gpf = material.critical_strain_energy_density() * d_init.copy()

        tracemalloc.start()
        grafea_timings = []
        u_steps = np.linspace(0, p.get('u_max', 0.01), n_steps + 1)[1:]

        for step_idx, u_val in enumerate(u_steps):
            bc_vals = bc_values_func(u_val)
            t = profile_grafea_pf_step(
                mesh, elements, material, edge_graph,
                damage_gpf, disp_gpf, history_gpf,
                bc_dofs, bc_vals, F_ext,
            )
            grafea_timings.append(t)

        _, grafea_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        entry['grafea_pf_memory_MB'] = grafea_peak_mem / 1e6
        entry['grafea_pf_timing'] = {
            k: float(np.mean([t[k] for t in grafea_timings]))
            for k in grafea_timings[0].keys()
        }
        entry['grafea_pf_dof_u'] = mesh.n_nodes * 2
        entry['grafea_pf_dof_d'] = mesh.n_edges

        # --- Profile FEM-PF ---
        if verbose:
            print("  Profiling FEM-PF...")

        config = SolverConfig(verbose=False)
        fem_solver = FEMPhasefieldSolver(mesh, material, config)
        fem_solver.set_initial_damage_from_edges(d_init, mesh)

        tracemalloc.start()
        fem_timings = []

        for step_idx, u_val in enumerate(u_steps):
            bc_vals = bc_values_func(u_val)
            t = profile_fem_pf_step(fem_solver, bc_dofs, bc_vals, F_ext)
            fem_timings.append(t)

        _, fem_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        entry['fem_pf_memory_MB'] = fem_peak_mem / 1e6
        entry['fem_pf_timing'] = {
            k: float(np.mean([t[k] for t in fem_timings]))
            for k in fem_timings[0].keys()
        }
        entry['fem_pf_dof_u'] = mesh.n_nodes * 2
        entry['fem_pf_dof_d'] = mesh.n_nodes

        results.append(entry)

        if verbose:
            gt = entry['grafea_pf_timing']
            ft = entry['fem_pf_timing']
            print(f"    GraFEA-PF: total={gt['total']:.4f}s/step, "
                  f"DOF_d={entry['grafea_pf_dof_d']}, "
                  f"mem={entry['grafea_pf_memory_MB']:.1f}MB")
            print(f"    FEM-PF:    total={ft['total']:.4f}s/step, "
                  f"DOF_d={entry['fem_pf_dof_d']}, "
                  f"mem={entry['fem_pf_memory_MB']:.1f}MB")

    return results


def fit_scaling_exponent(node_counts, timings):
    """
    Fit time = C * N^alpha to determine scaling exponent.

    Parameters
    ----------
    node_counts : array-like
        Number of nodes for each data point.
    timings : array-like
        Wall time for each data point.

    Returns
    -------
    dict
        'alpha': scaling exponent, 'C': coefficient,
        'r_squared': goodness of fit.
    """
    node_counts = np.asarray(node_counts, dtype=float)
    timings = np.asarray(timings, dtype=float)

    # Filter positive values
    valid = (node_counts > 0) & (timings > 0)
    if np.sum(valid) < 2:
        return {'alpha': float('nan'), 'C': float('nan'), 'r_squared': 0.0}

    log_n = np.log(node_counts[valid])
    log_t = np.log(timings[valid])

    coeffs = np.polyfit(log_n, log_t, 1)
    alpha = coeffs[0]
    C = np.exp(coeffs[1])

    # R-squared
    log_t_pred = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_t - log_t_pred) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-20)

    return {'alpha': float(alpha), 'C': float(C), 'r_squared': float(r_squared)}


def analyze_scaling(results):
    """
    Analyze scaling behavior from profiling results.

    Parameters
    ----------
    results : list of dict
        Results from run_scaling_study.

    Returns
    -------
    dict
        Scaling analysis with exponents for each operation.
    """
    if len(results) < 2:
        return {'error': 'Not enough data points for scaling analysis'}

    nodes = np.array([r['actual_nodes'] for r in results])

    analysis = {
        'node_counts': nodes.tolist(),
        'grafea_pf': {},
        'fem_pf': {},
    }

    # Fit scaling for each operation
    operations = ['assembly_K', 'solve_u', 'assembly_damage', 'solve_d', 'total']

    for op in operations:
        grafea_times = np.array([r['grafea_pf_timing'].get(op, 0) for r in results])
        fem_times = np.array([r['fem_pf_timing'].get(op, 0) for r in results])

        analysis['grafea_pf'][op] = {
            'times': grafea_times.tolist(),
            'scaling': fit_scaling_exponent(nodes, grafea_times),
        }
        analysis['fem_pf'][op] = {
            'times': fem_times.tolist(),
            'scaling': fit_scaling_exponent(nodes, fem_times),
        }

    # Memory scaling
    grafea_mem = np.array([r['grafea_pf_memory_MB'] for r in results])
    fem_mem = np.array([r['fem_pf_memory_MB'] for r in results])

    analysis['grafea_pf']['memory'] = {
        'values_MB': grafea_mem.tolist(),
        'scaling': fit_scaling_exponent(nodes, grafea_mem),
    }
    analysis['fem_pf']['memory'] = {
        'values_MB': fem_mem.tolist(),
        'scaling': fit_scaling_exponent(nodes, fem_mem),
    }

    # DOF ratios
    analysis['dof_ratios'] = {
        'damage_grafea': [r['grafea_pf_dof_d'] for r in results],
        'damage_fem': [r['fem_pf_dof_d'] for r in results],
        'ratio': [r['grafea_pf_dof_d'] / max(r['fem_pf_dof_d'], 1)
                  for r in results],
    }

    # Speedup / slowdown
    analysis['speedup'] = {}
    for op in operations:
        grafea_times = [r['grafea_pf_timing'].get(op, 0) for r in results]
        fem_times = [r['fem_pf_timing'].get(op, 0) for r in results]
        analysis['speedup'][op] = [
            ft / max(gt, 1e-20) for gt, ft in zip(grafea_times, fem_times)
        ]

    return analysis


def save_scaling_results(results, analysis, filepath):
    """
    Save scaling study results to JSON.

    Parameters
    ----------
    results : list of dict
        Raw profiling results.
    analysis : dict
        Scaling analysis.
    filepath : str
        Output path.
    """
    output = {
        'profiling': results,
        'analysis': analysis,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def print_scaling_summary(results, analysis):
    """
    Print formatted scaling study summary.

    Parameters
    ----------
    results : list of dict
        Profiling results.
    analysis : dict
        Scaling analysis.
    """
    print("\n" + "=" * 90)
    print("Computational Cost Scaling Summary")
    print("=" * 90)

    # Timing table
    print(f"\n{'Nodes':>8s} | {'Method':>10s} | {'DOF(u)':>8s} | {'DOF(d)':>8s} | "
          f"{'K asm':>8s} | {'u solve':>8s} | {'d asm':>8s} | {'d solve':>8s} | "
          f"{'Total':>8s} | {'Mem MB':>8s}")
    print("-" * 90)

    for r in results:
        for method in ['grafea_pf', 'fem_pf']:
            t = r[f'{method}_timing']
            label = 'GraFEA' if method == 'grafea_pf' else 'FEM-PF'
            print(f"{r['actual_nodes']:8d} | {label:>10s} | "
                  f"{r[f'{method}_dof_u']:8d} | {r[f'{method}_dof_d']:8d} | "
                  f"{t.get('assembly_K', 0):8.4f} | {t.get('solve_u', 0):8.4f} | "
                  f"{t.get('assembly_damage', 0):8.4f} | {t.get('solve_d', 0):8.4f} | "
                  f"{t.get('total', 0):8.4f} | {r[f'{method}_memory_MB']:8.1f}")

    # Scaling exponents
    print("\nScaling Exponents (time ~ N^alpha):")
    print(f"{'Operation':>20s} | {'GraFEA-PF alpha':>16s} | {'FEM-PF alpha':>14s}")
    print("-" * 55)
    for op in ['assembly_K', 'solve_u', 'assembly_damage', 'solve_d', 'total']:
        ga = analysis.get('grafea_pf', {}).get(op, {}).get('scaling', {})
        fa = analysis.get('fem_pf', {}).get(op, {}).get('scaling', {})
        g_alpha = ga.get('alpha', float('nan'))
        f_alpha = fa.get('alpha', float('nan'))
        print(f"{op:>20s} | {g_alpha:16.2f} | {f_alpha:14.2f}")

    # DOF ratios
    if 'dof_ratios' in analysis:
        ratios = analysis['dof_ratios']['ratio']
        print(f"\nDamage DOF ratio (GraFEA/FEM): "
              f"mean = {np.mean(ratios):.2f}, "
              f"range = [{min(ratios):.2f}, {max(ratios):.2f}]")
        print("(Expected: ~1.5 for triangular meshes since n_edges ~ 1.5 * n_nodes)")


def generate_cost_table_csv(results, filepath):
    """
    Generate CSV table of computational costs.

    Parameters
    ----------
    results : list of dict
        Profiling results.
    filepath : str
        Output CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    header = ("Nodes,Method,DOF_u,DOF_d,Assembly_K_s,Solve_u_s,"
              "Assembly_d_s,Solve_d_s,Total_s,Memory_MB\n")

    with open(filepath, 'w') as f:
        f.write(header)
        for r in results:
            for method in ['grafea_pf', 'fem_pf']:
                t = r[f'{method}_timing']
                label = 'GraFEA-PF' if method == 'grafea_pf' else 'FEM-PF'
                f.write(f"{r['actual_nodes']},{label},"
                        f"{r[f'{method}_dof_u']},{r[f'{method}_dof_d']},"
                        f"{t.get('assembly_K', 0):.6f},"
                        f"{t.get('solve_u', 0):.6f},"
                        f"{t.get('assembly_damage', 0):.6f},"
                        f"{t.get('solve_d', 0):.6f},"
                        f"{t.get('total', 0):.6f},"
                        f"{r[f'{method}_memory_MB']:.2f}\n")
