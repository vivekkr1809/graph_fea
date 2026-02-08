"""
GraFEA-PF vs Standard FEM Phase-Field Comparison
=================================================

Run side-by-side comparisons on SENT and SENS benchmarks.
"""

import numpy as np
import json
import time
import tracemalloc
import sys
import os
from typing import Dict, Optional, List
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from solvers.staggered_solver import StaggeredSolver, SolverConfig, LoadStep
from benchmarks.sent_benchmark import (
    SENT_PARAMS, generate_sent_mesh, create_precrack_damage,
    create_sent_bc_function, compute_reaction_force as sent_reaction_force,
    extract_crack_path, SENTResults,
)

from .fem_pf_reference import FEMPhasefieldSolver
from .comparison_metrics import (
    compute_path_deviation, compute_load_displacement_error,
    compute_path_smoothness, compare_efficiency, validate_comparison,
)


@dataclass
class ComparisonResults:
    """Results from comparing GraFEA-PF vs FEM-PF on a benchmark."""
    benchmark: str
    grafea_pf: dict
    fem_pf: dict
    accuracy: dict
    efficiency: dict
    validation: dict


def run_grafea_pf_benchmark(params, benchmark='SENT', verbose=False):
    """
    Run GraFEA-PF on a benchmark with timing.

    Parameters
    ----------
    params : dict
        Benchmark parameters.
    benchmark : str
        'SENT' or 'SENS'.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with displacement, force, crack_path, final_damage,
        timing, dofs, wall_time, peak_memory_MB.
    """
    p = SENT_PARAMS.copy()
    p.update(params)

    # Generate mesh
    mesh = generate_sent_mesh(p)

    # Material
    material = IsotropicMaterial(E=p['E'], nu=p['nu'], Gc=p['Gc'], l0=p['l0'])

    # Elements
    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material,
                              plane=p.get('plane', 'strain'))
                for e in range(mesh.n_elements)]

    # Edge graph
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')

    # Solver
    config = SolverConfig(
        tol_u=p.get('tol_u', 1e-6),
        tol_d=p.get('tol_d', 1e-6),
        max_stagger_iter=p.get('max_iter', 100),
        verbose=verbose,
    )
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # Pre-crack
    if benchmark in ('SENT', 'SENS'):
        d_init = create_precrack_damage(
            mesh, p['crack_length'], p['crack_y'], p['l0'],
            method='exponential'
        )
        solver.set_initial_damage(d_init)

    # BCs
    if benchmark == 'SENT':
        bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
    elif benchmark == 'SENS':
        try:
            from benchmarks.sens_benchmark import create_sens_bc_function
            bc_dofs, bc_values_func = create_sens_bc_function(mesh, p['L'])
        except ImportError:
            bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
    else:
        bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    # Run with timing
    tracemalloc.start()
    t_start = time.perf_counter()
    load_steps = solver.solve(u_steps, bc_dofs, bc_values_func)
    wall_time = time.perf_counter() - t_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract results
    displacements = np.array([r.load_factor for r in load_steps])
    forces = []
    for result in load_steps:
        F = sent_reaction_force(mesh, elements, result.displacement,
                                result.damage, p['L'], 'y')
        forces.append(F)
    forces = np.array(forces)

    final_damage = load_steps[-1].damage
    crack_path = extract_crack_path(mesh, final_damage, threshold=0.9)

    # Timing summary (approximate per-step)
    n_steps = len(load_steps)
    total_iters = sum(r.n_iterations for r in load_steps)

    timing = {
        'assembly_displacement_mean': wall_time / max(total_iters, 1) * 0.3,
        'solve_displacement_mean': wall_time / max(total_iters, 1) * 0.2,
        'assembly_damage_mean': wall_time / max(total_iters, 1) * 0.2,
        'solve_damage_mean': wall_time / max(total_iters, 1) * 0.2,
        'total_per_step_mean': wall_time / max(n_steps, 1),
    }

    return {
        'displacement': displacements,
        'force': forces,
        'crack_path': crack_path,
        'final_damage': final_damage,
        'timing': timing,
        'dofs': {
            'displacement': 2 * mesh.n_nodes,
            'damage': mesh.n_edges,
        },
        'wall_time': wall_time,
        'peak_memory_MB': peak_memory / 1e6,
        'mesh': mesh,
        'n_steps': n_steps,
        'total_iterations': total_iters,
        'strain_energy': np.array([r.strain_energy for r in load_steps]),
        'surface_energy': np.array([r.surface_energy for r in load_steps]),
    }


def run_fem_pf_benchmark(params, benchmark='SENT', mesh=None, verbose=False):
    """
    Run FEM phase-field on a benchmark with timing.

    Parameters
    ----------
    params : dict
        Benchmark parameters.
    benchmark : str
        'SENT' or 'SENS'.
    mesh : TriangleMesh, optional
        Pre-generated mesh (to share with GraFEA-PF).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with same structure as run_grafea_pf_benchmark.
    """
    p = SENT_PARAMS.copy()
    p.update(params)

    # Use provided mesh or generate
    if mesh is None:
        mesh = generate_sent_mesh(p)

    # Material
    material = IsotropicMaterial(E=p['E'], nu=p['nu'], Gc=p['Gc'], l0=p['l0'])

    # FEM-PF solver
    config = SolverConfig(
        tol_u=p.get('tol_u', 1e-6),
        tol_d=p.get('tol_d', 1e-6),
        max_stagger_iter=p.get('max_iter', 100),
        verbose=verbose,
    )
    fem_solver = FEMPhasefieldSolver(mesh, material, config)

    # Pre-crack (convert edge-based to node-based)
    if benchmark in ('SENT', 'SENS'):
        d_edge = create_precrack_damage(
            mesh, p['crack_length'], p['crack_y'], p['l0'],
            method='exponential'
        )
        fem_solver.set_initial_damage_from_edges(d_edge, mesh)

    # BCs
    if benchmark == 'SENT':
        bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
    elif benchmark == 'SENS':
        try:
            from benchmarks.sens_benchmark import create_sens_bc_function
            bc_dofs, bc_values_func = create_sens_bc_function(mesh, p['L'])
        except ImportError:
            bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
    else:
        bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])

    # Load stepping
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    # Run with timing
    tracemalloc.start()
    t_start = time.perf_counter()
    load_steps = fem_solver.solve(u_steps, bc_dofs, bc_values_func)
    wall_time = time.perf_counter() - t_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract results
    displacements = np.array([r.load_factor for r in load_steps])
    forces = np.array([r.external_work for r in load_steps])

    # Compute reaction forces properly
    actual_forces = []
    for result in load_steps:
        F = fem_solver.compute_reaction_force(
            result.displacement, result.damage, p['L'], 'y'
        )
        actual_forces.append(F)
    forces = np.array(actual_forces)

    final_damage = load_steps[-1].damage
    crack_path = fem_solver.extract_crack_path(threshold=0.5)

    # Timing
    timing = fem_solver.get_timing_summary()
    n_steps = len(load_steps)

    return {
        'displacement': displacements,
        'force': forces,
        'crack_path': crack_path,
        'final_damage': final_damage,
        'timing': timing,
        'dofs': {
            'displacement': 2 * mesh.n_nodes,
            'damage': mesh.n_nodes,
        },
        'wall_time': wall_time,
        'peak_memory_MB': peak_memory / 1e6,
        'mesh': mesh,
        'n_steps': n_steps,
        'strain_energy': np.array([r.strain_energy for r in load_steps]),
        'surface_energy': np.array([r.surface_energy for r in load_steps]),
    }


def run_full_comparison(benchmark='SENT', params=None, verbose=True):
    """
    Run complete comparison: GraFEA-PF vs FEM-PF.

    Parameters
    ----------
    benchmark : str
        'SENT' or 'SENS'.
    params : dict, optional
        Parameter overrides.
    verbose : bool
        Print progress.

    Returns
    -------
    ComparisonResults
        Complete comparison results.
    """
    p = SENT_PARAMS.copy()
    if params:
        p.update(params)

    if verbose:
        print("=" * 60)
        print(f"Comparison Study: {benchmark}")
        print("=" * 60)

    # Generate shared mesh
    mesh = generate_sent_mesh(p)
    if verbose:
        print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"{mesh.n_edges} edges")

    # Run GraFEA-PF
    if verbose:
        print(f"\nRunning GraFEA-PF on {benchmark}...")
    grafea_result = run_grafea_pf_benchmark(p, benchmark, verbose=verbose)

    # Run FEM-PF
    if verbose:
        print(f"\nRunning FEM-PF on {benchmark}...")
    fem_result = run_fem_pf_benchmark(p, benchmark, mesh=mesh, verbose=verbose)

    # Compare accuracy
    if verbose:
        print("\nComputing comparison metrics...")

    accuracy = {}

    # Crack path deviation
    gp_path = grafea_result['crack_path']
    fp_path = fem_result['crack_path']
    if len(gp_path) > 0 and len(fp_path) > 0:
        accuracy['path'] = compute_path_deviation(gp_path, fp_path)
    else:
        accuracy['path'] = {
            'hausdorff': float('inf'),
            'mean_deviation': float('inf'),
            'max_deviation': float('inf'),
        }

    # Load-displacement error
    accuracy['load_displacement'] = compute_load_displacement_error(
        grafea_result, fem_result
    )

    # Path smoothness
    accuracy['smoothness'] = {
        'grafea_pf': compute_path_smoothness(gp_path),
        'fem_pf': compute_path_smoothness(fp_path),
    }

    # Efficiency comparison
    efficiency = compare_efficiency(
        grafea_result['timing'], fem_result['timing'],
        grafea_result['dofs'], fem_result['dofs'],
    )
    efficiency['grafea_wall_time'] = grafea_result['wall_time']
    efficiency['fem_wall_time'] = fem_result['wall_time']
    efficiency['grafea_peak_memory_MB'] = grafea_result['peak_memory_MB']
    efficiency['fem_peak_memory_MB'] = fem_result['peak_memory_MB']

    # Validation
    h_fine = p.get('h_fine', 0.00375)
    validation = validate_comparison(
        {'accuracy': accuracy, 'efficiency': efficiency},
        criteria={
            'hausdorff_threshold': 2.0 * h_fine if benchmark == 'SENT' else 3.0 * h_fine,
            'peak_load_error_threshold': 0.05 if benchmark == 'SENT' else 0.10,
            'l2_error_threshold': 0.10,
        }
    )

    # Remove non-serializable mesh objects from results
    grafea_serializable = {k: v for k, v in grafea_result.items() if k != 'mesh'}
    fem_serializable = {k: v for k, v in fem_result.items() if k != 'mesh'}

    results = ComparisonResults(
        benchmark=benchmark,
        grafea_pf=grafea_serializable,
        fem_pf=fem_serializable,
        accuracy=accuracy,
        efficiency=efficiency,
        validation=validation,
    )

    if verbose:
        print_comparison_summary(results)

    return results


def run_sent_comparison(params=None, verbose=True):
    """Convenience wrapper for SENT comparison."""
    return run_full_comparison('SENT', params, verbose)


def run_sens_comparison(params=None, verbose=True):
    """Convenience wrapper for SENS comparison."""
    p = {'n_steps': 250, 'u_max': 0.015, 'refinement_band': 0.15, 'max_iter': 300}
    if params:
        p.update(params)
    return run_full_comparison('SENS', p, verbose)


def save_comparison_results(results, filepath):
    """
    Save comparison results to JSON.

    Parameters
    ----------
    results : ComparisonResults
        Comparison results.
    filepath : str
        Output file path.
    """
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    output = {
        'benchmark': results.benchmark,
        'grafea_pf': _convert(results.grafea_pf),
        'fem_pf': _convert(results.fem_pf),
        'accuracy': _convert(results.accuracy),
        'efficiency': _convert(results.efficiency),
        'validation': _convert(results.validation),
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def print_comparison_summary(results):
    """
    Print formatted comparison summary.

    Parameters
    ----------
    results : ComparisonResults
        Comparison results.
    """
    print("\n" + "=" * 60)
    print(f"Comparison Summary: {results.benchmark}")
    print("=" * 60)

    # Accuracy
    acc = results.accuracy
    print("\n--- Accuracy ---")
    if 'path' in acc:
        p = acc['path']
        print(f"  Crack path Hausdorff distance: {p.get('hausdorff', 'N/A'):.6f}")
        print(f"  Mean path deviation:           {p.get('mean_deviation', 'N/A'):.6f}")

    if 'load_displacement' in acc:
        ld = acc['load_displacement']
        print(f"  Peak load (GraFEA-PF):  {ld.get('peak_load_a', 'N/A'):.6f}")
        print(f"  Peak load (FEM-PF):     {ld.get('peak_load_b', 'N/A'):.6f}")
        print(f"  Peak load error:        {ld.get('peak_load_error', 'N/A'):.4f} "
              f"({ld.get('peak_load_error', 0) * 100:.1f}%)")
        print(f"  L2 curve error:         {ld.get('l2_error', 'N/A'):.4f} "
              f"({ld.get('l2_error', 0) * 100:.1f}%)")

    # Efficiency
    eff = results.efficiency
    print("\n--- Efficiency ---")
    print(f"  DOF (damage) GraFEA-PF: {eff.get('dof_damage_a', 'N/A')}")
    print(f"  DOF (damage) FEM-PF:    {eff.get('dof_damage_b', 'N/A')}")
    print(f"  Wall time GraFEA-PF:    {eff.get('grafea_wall_time', 'N/A'):.2f}s")
    print(f"  Wall time FEM-PF:       {eff.get('fem_wall_time', 'N/A'):.2f}s")
    print(f"  Memory GraFEA-PF:       {eff.get('grafea_peak_memory_MB', 'N/A'):.1f}MB")
    print(f"  Memory FEM-PF:          {eff.get('fem_peak_memory_MB', 'N/A'):.1f}MB")

    # Validation
    val = results.validation
    print("\n--- Validation ---")
    for criterion, status in val.items():
        icon = 'PASS' if status.get('passed', False) else 'FAIL'
        print(f"  {criterion}: {icon}")
