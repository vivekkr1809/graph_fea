"""
Mesh Sensitivity Study
=======================

Demonstrate mesh-independent crack paths for GraFEA-PF
vs mesh-dependent paths for original GraFEA.
"""

import numpy as np
import json
import sys
import os
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from mesh.edge_graph import EdgeGraph
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from solvers.staggered_solver import StaggeredSolver, SolverConfig
from benchmarks.sent_benchmark import (
    SENT_PARAMS, generate_sent_mesh, create_precrack_damage,
    create_sent_bc_function, compute_reaction_force, extract_crack_path,
)

from .original_grafea import OriginalGraFEASolver
from .comparison_metrics import compute_path_deviation, compute_load_displacement_error


MESH_SENSITIVITY_PARAMS = {
    'benchmark': 'SENT',
    'l0': 0.015,
    'mesh_configs': [
        {'name': 'coarse',    'h_fine': 0.030,   'h_l0_ratio': 2.0},
        {'name': 'medium',    'h_fine': 0.015,   'h_l0_ratio': 1.0},
        {'name': 'fine',      'h_fine': 0.0075,  'h_l0_ratio': 0.5},
        {'name': 'very_fine', 'h_fine': 0.00375, 'h_l0_ratio': 0.25},
    ],
    'mesh_types': ['structured', 'unstructured'],
    # SENT base params
    'E': 210e3,
    'nu': 0.3,
    'Gc': 2.7,
    'L': 1.0,
    'crack_length': 0.5,
    'crack_y': 0.5,
    'u_max': 0.01,
    'n_steps': 100,
    'h_coarse': 0.02,
    'refinement_band': 0.1,
    'plane': 'strain',
    'tol_u': 1e-6,
    'tol_d': 1e-6,
    'max_iter': 100,
}


def _perturb_mesh(mesh, amplitude, crack_y, l0):
    """
    Perturb interior nodes to create unstructured-like mesh.

    Does NOT perturb boundary nodes or nodes near the crack path.

    Parameters
    ----------
    mesh : TriangleMesh
        Original structured mesh.
    amplitude : float
        Maximum perturbation amplitude.
    crack_y : float
        y-coordinate of crack (avoid perturbation near here).
    l0 : float
        Phase-field length scale.

    Returns
    -------
    TriangleMesh
        Perturbed mesh.
    """
    rng = np.random.RandomState(42)  # Reproducible
    nodes = mesh.nodes.copy()
    boundary_set = set(mesh.boundary_nodes.tolist())

    for i in range(len(nodes)):
        if i in boundary_set:
            continue
        x, y = nodes[i]
        # Don't perturb near crack
        if abs(y - crack_y) < 3 * l0 and x < 0.6:
            continue
        nodes[i, 0] += amplitude * (rng.rand() - 0.5) * 2
        nodes[i, 1] += amplitude * (rng.rand() - 0.5) * 2

    return TriangleMesh(nodes, mesh.elements.copy(), mesh.thickness)


def generate_mesh_for_study(params, h_fine, mesh_type='structured'):
    """
    Generate a mesh with specified element size and type.

    Parameters
    ----------
    params : dict
        Study parameters.
    h_fine : float
        Fine element size.
    mesh_type : str
        'structured' or 'unstructured'.

    Returns
    -------
    TriangleMesh
    """
    p = params.copy()
    p['h_fine'] = h_fine
    p['h_coarse'] = max(h_fine * 3, params.get('h_coarse', 0.02))
    p['refinement_band'] = max(params.get('refinement_band', 0.1), 4 * h_fine)

    mesh = generate_sent_mesh(p)

    if mesh_type == 'unstructured':
        mesh = _perturb_mesh(
            mesh,
            amplitude=0.15 * h_fine,
            crack_y=params.get('crack_y', 0.5),
            l0=params.get('l0', 0.015),
        )

    return mesh


def run_single_mesh_config(params, h_fine, mesh_type,
                           run_original=True, verbose=False):
    """
    Run both GraFEA-PF and Original GraFEA on a single mesh config.

    Parameters
    ----------
    params : dict
        Study parameters.
    h_fine : float
        Fine element size.
    mesh_type : str
        'structured' or 'unstructured'.
    run_original : bool
        Also run original GraFEA.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results for both methods on this mesh.
    """
    p = params.copy()
    l0 = p.get('l0', 0.015)

    mesh = generate_mesh_for_study(p, h_fine, mesh_type)

    if verbose:
        print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"{mesh.n_edges} edges")

    material = IsotropicMaterial(E=p['E'], nu=p['nu'], Gc=p['Gc'], l0=l0)

    result = {
        'mesh_size': h_fine,
        'h_l0_ratio': h_fine / l0,
        'mesh_type': mesh_type,
        'n_nodes': mesh.n_nodes,
        'n_elements': mesh.n_elements,
        'n_edges': mesh.n_edges,
    }

    # --- GraFEA-PF ---
    if verbose:
        print("  Running GraFEA-PF...")

    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material,
                              plane=p.get('plane', 'strain'))
                for e in range(mesh.n_elements)]
    edge_graph = EdgeGraph(mesh, weight_scheme='distance')
    config = SolverConfig(
        tol_u=p.get('tol_u', 1e-6), tol_d=p.get('tol_d', 1e-6),
        max_stagger_iter=p.get('max_iter', 100), verbose=False,
    )
    solver_pf = StaggeredSolver(mesh, elements, material, edge_graph, config)

    d_init = create_precrack_damage(
        mesh, p['crack_length'], p['crack_y'], l0, method='exponential'
    )
    solver_pf.set_initial_damage(d_init)

    bc_dofs, bc_values_func = create_sent_bc_function(mesh, p['L'])
    u_steps = np.linspace(0, p['u_max'], p['n_steps'])

    load_steps_pf = solver_pf.solve(u_steps, bc_dofs, bc_values_func)

    disp_pf = np.array([r.load_factor for r in load_steps_pf])
    forces_pf = []
    for r in load_steps_pf:
        F = compute_reaction_force(mesh, elements, r.displacement,
                                   r.damage, p['L'], 'y')
        forces_pf.append(F)
    forces_pf = np.array(forces_pf)

    final_damage_pf = load_steps_pf[-1].damage
    crack_path_pf = extract_crack_path(mesh, final_damage_pf, threshold=0.9)

    result['grafea_pf'] = {
        'crack_path': crack_path_pf,
        'force': forces_pf,
        'displacement': disp_pf,
        'peak_load': float(np.max(forces_pf)),
        'final_damage': final_damage_pf,
    }

    # --- Original GraFEA ---
    if run_original:
        if verbose:
            print("  Running Original GraFEA...")

        solver_orig = OriginalGraFEASolver(mesh, material, config)
        d_binary = (d_init > 0.5).astype(float)
        solver_orig.set_initial_damage(d_binary)

        load_steps_orig = solver_orig.solve(u_steps, bc_dofs, bc_values_func)

        disp_orig = np.array([r.load_factor for r in load_steps_orig])
        forces_orig = []
        elements_orig = solver_orig.elements
        for r in load_steps_orig:
            F = compute_reaction_force(mesh, elements_orig, r.displacement,
                                       r.damage, p['L'], 'y')
            forces_orig.append(F)
        forces_orig = np.array(forces_orig)

        final_damage_orig = load_steps_orig[-1].damage
        crack_path_orig = extract_crack_path(mesh, final_damage_orig, threshold=0.5)

        result['grafea_original'] = {
            'crack_path': crack_path_orig,
            'force': forces_orig,
            'displacement': disp_orig,
            'peak_load': float(np.max(forces_orig)),
            'final_damage': final_damage_orig,
        }

    return result


def run_mesh_sensitivity_study(params=None, verbose=True):
    """
    Run SENT on all mesh configurations.

    Parameters
    ----------
    params : dict, optional
        Study parameters.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results keyed by "{name}_{type}".
    """
    p = MESH_SENSITIVITY_PARAMS.copy()
    if params:
        p.update(params)

    results = {}

    if verbose:
        print("=" * 60)
        print("Mesh Sensitivity Study")
        print("=" * 60)

    for config in p['mesh_configs']:
        for mesh_type in p['mesh_types']:
            key = f"{config['name']}_{mesh_type}"
            if verbose:
                print(f"\n--- {key}: h={config['h_fine']}, "
                      f"h/l0={config['h_l0_ratio']}, type={mesh_type} ---")

            result = run_single_mesh_config(
                p, config['h_fine'], mesh_type,
                run_original=True, verbose=verbose,
            )
            results[key] = result

    if verbose:
        print("\n" + "=" * 60)
        print("Mesh Sensitivity Study Complete")
        print("=" * 60)

    return results


def analyze_mesh_convergence(results):
    """
    Analyze convergence of crack paths and peak loads.

    Uses the finest structured mesh as reference.

    Parameters
    ----------
    results : dict
        Results from run_mesh_sensitivity_study.

    Returns
    -------
    dict
        Convergence analysis for both methods.
    """
    analysis = {'grafea_pf': {}, 'grafea_original': {}}

    # Find reference (finest structured mesh)
    ref_key = 'very_fine_structured'
    if ref_key not in results:
        # Fallback: finest available
        structured_keys = [k for k in results.keys() if 'structured' in k]
        if structured_keys:
            ref_key = min(structured_keys, key=lambda k: results[k]['mesh_size'])
        else:
            ref_key = min(results.keys(), key=lambda k: results[k]['mesh_size'])

    ref = results[ref_key]

    for method in ['grafea_pf', 'grafea_original']:
        if method not in ref:
            continue

        ref_path = np.asarray(ref[method]['crack_path'])
        ref_peak = ref[method]['peak_load']

        for key, data in results.items():
            if key == ref_key or method not in data:
                continue

            path = np.asarray(data[method]['crack_path'])

            # Path deviation
            if len(path) > 0 and len(ref_path) > 0:
                path_dev = compute_path_deviation(path, ref_path)
            else:
                path_dev = {'hausdorff': float('inf'), 'mean_deviation': float('inf')}

            # Peak load error
            peak = data[method]['peak_load']
            if abs(ref_peak) > 1e-20:
                peak_err = abs(peak - ref_peak) / abs(ref_peak)
            else:
                peak_err = abs(peak - ref_peak)

            analysis[method][key] = {
                'h': data['mesh_size'],
                'h_l0': data.get('h_l0_ratio', 0),
                'mesh_type': data['mesh_type'],
                'path_hausdorff': path_dev['hausdorff'],
                'path_mean_dev': path_dev['mean_deviation'],
                'peak_load_error': peak_err,
                'peak_load': peak,
            }

        # Convergence rate from structured meshes
        struct_data = {
            k: v for k, v in analysis[method].items()
            if isinstance(v, dict) and v.get('mesh_type') == 'structured'
        }
        if len(struct_data) >= 2:
            h_vals = np.array([v['h'] for v in struct_data.values()])
            dev_vals = np.array([v['path_hausdorff'] for v in struct_data.values()])

            valid = (dev_vals > 0) & np.isfinite(dev_vals)
            if np.sum(valid) >= 2:
                log_h = np.log(h_vals[valid])
                log_dev = np.log(dev_vals[valid])
                p_coeff = np.polyfit(log_h, log_dev, 1)[0]
                analysis[method]['convergence_rate'] = float(p_coeff)
            else:
                analysis[method]['convergence_rate'] = float('inf')

    # Structured vs unstructured comparison
    analysis['structured_vs_unstructured'] = {}
    for config_name in ['coarse', 'medium', 'fine', 'very_fine']:
        s_key = f"{config_name}_structured"
        u_key = f"{config_name}_unstructured"
        if s_key not in results or u_key not in results:
            continue

        entry = {}
        for method in ['grafea_pf', 'grafea_original']:
            if method not in results[s_key] or method not in results[u_key]:
                continue
            s_path = np.asarray(results[s_key][method]['crack_path'])
            u_path = np.asarray(results[u_key][method]['crack_path'])
            if len(s_path) > 0 and len(u_path) > 0:
                dev = compute_path_deviation(s_path, u_path)
                entry[f'{method}_hausdorff'] = dev['hausdorff']
                entry[f'{method}_mean_dev'] = dev['mean_deviation']
            else:
                entry[f'{method}_hausdorff'] = float('inf')

        h = results[s_key]['mesh_size']
        entry['h'] = h
        analysis['structured_vs_unstructured'][config_name] = entry

    return analysis


def save_mesh_sensitivity_results(results, analysis, dirpath):
    """
    Save mesh sensitivity results.

    Parameters
    ----------
    results : dict
        Raw results.
    analysis : dict
        Convergence analysis.
    dirpath : str
        Output directory.
    """
    os.makedirs(dirpath, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    # Save summary (without large arrays)
    summary = {}
    for key, data in results.items():
        entry = {
            'mesh_size': data['mesh_size'],
            'h_l0_ratio': data.get('h_l0_ratio', 0),
            'mesh_type': data['mesh_type'],
            'n_nodes': data['n_nodes'],
            'n_elements': data['n_elements'],
            'n_edges': data['n_edges'],
        }
        for method in ['grafea_pf', 'grafea_original']:
            if method in data:
                entry[method] = {
                    'peak_load': data[method]['peak_load'],
                    'crack_path': data[method]['crack_path'].tolist()
                                 if isinstance(data[method]['crack_path'], np.ndarray)
                                 else data[method]['crack_path'],
                }
        summary[key] = entry

    with open(os.path.join(dirpath, 'mesh_sensitivity_summary.json'), 'w') as f:
        json.dump(_convert(summary), f, indent=2, default=str)

    with open(os.path.join(dirpath, 'mesh_convergence_analysis.json'), 'w') as f:
        json.dump(_convert(analysis), f, indent=2, default=str)


def print_mesh_sensitivity_summary(analysis):
    """Print formatted convergence summary."""
    print("\n" + "=" * 70)
    print("Mesh Convergence Summary")
    print("=" * 70)

    for method in ['grafea_pf', 'grafea_original']:
        method_label = 'GraFEA-PF' if method == 'grafea_pf' else 'Original GraFEA'
        print(f"\n--- {method_label} ---")
        print(f"{'Mesh':>25s} {'h':>10s} {'h/l0':>8s} "
              f"{'Hausdorff':>12s} {'Peak Err':>10s}")
        print("-" * 70)

        data = analysis.get(method, {})
        for key in sorted(k for k in data.keys()
                          if isinstance(data[k], dict) and 'h' in data[k]):
            d = data[key]
            print(f"{key:>25s} {d['h']:10.6f} {d.get('h_l0', 0):8.2f} "
                  f"{d['path_hausdorff']:12.6f} {d['peak_load_error']:10.4f}")

        rate = data.get('convergence_rate', 'N/A')
        print(f"\n  Convergence rate: {rate}")

    # Structured vs unstructured
    sv = analysis.get('structured_vs_unstructured', {})
    if sv:
        print("\n--- Structured vs Unstructured ---")
        print(f"{'Config':>15s} {'h':>10s} {'PF Hausdorff':>14s} {'Orig Hausdorff':>16s}")
        print("-" * 60)
        for config_name, data in sv.items():
            pf_h = data.get('grafea_pf_hausdorff', float('inf'))
            orig_h = data.get('grafea_original_hausdorff', float('inf'))
            print(f"{config_name:>15s} {data.get('h', 0):10.6f} "
                  f"{pf_h:14.6f} {orig_h:16.6f}")
