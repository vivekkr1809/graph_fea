"""
Publication-Quality Figure Generation
======================================

Generate all comparison study figures for the paper.
Uses matplotlib with LaTeX-compatible formatting.
"""

import numpy as np
import os
import sys
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Figure style parameters
FIGURE_PARAMS = {
    'figsize': (6, 4.5),
    'figsize_wide': (10, 4.5),
    'dpi': 300,
    'font_size': 11,
    'font_family': 'serif',
    'line_width': 1.5,
    'marker_size': 4,
    'colors': {
        'grafea_pf': '#2196F3',
        'fem_pf': '#FF9800',
        'grafea_original': '#4CAF50',
        'reference': '#9E9E9E',
    },
    'linestyles': {
        'grafea_pf': '-',
        'fem_pf': '--',
        'grafea_original': '-.',
        'reference': ':',
    },
    'markers': {
        'grafea_pf': 'o',
        'fem_pf': 's',
        'grafea_original': '^',
    },
    'labels': {
        'grafea_pf': 'GraFEA-PF',
        'fem_pf': 'FEM-PF',
        'grafea_original': 'Original GraFEA',
    },
    'format': 'pdf',
}


def _setup_style():
    """Apply publication-quality plot style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        'font.size': FIGURE_PARAMS['font_size'],
        'font.family': FIGURE_PARAMS['font_family'],
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': FIGURE_PARAMS['line_width'],
        'lines.markersize': FIGURE_PARAMS['marker_size'],
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'figure.dpi': FIGURE_PARAMS['dpi'],
    })


def _save_figure(fig, filepath, tight=True):
    """Save figure to file with proper formatting."""
    if tight:
        fig.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Fig. 6: SENT Crack Path Comparison (3 methods)
# ============================================================

def plot_sent_crack_paths(grafea_pf_path, fem_pf_path, original_path=None,
                          L=1.0, save_path=None):
    """
    Plot SENT crack path comparison (all three methods).

    Parameters
    ----------
    grafea_pf_path : np.ndarray, shape (N, 2)
        GraFEA-PF crack path.
    fem_pf_path : np.ndarray, shape (N, 2)
        FEM-PF crack path.
    original_path : np.ndarray, optional
        Original GraFEA crack path.
    L : float
        Domain size.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_PARAMS['figsize'])

    c = FIGURE_PARAMS['colors']
    ls = FIGURE_PARAMS['linestyles']
    lb = FIGURE_PARAMS['labels']

    # Reference line (expected horizontal)
    ax.axhline(y=0.5 * L, color=c['reference'], linestyle=':',
               linewidth=1, label='Expected (horizontal)', zorder=1)

    # Plot crack paths
    if len(grafea_pf_path) > 0:
        ax.plot(grafea_pf_path[:, 0], grafea_pf_path[:, 1],
                color=c['grafea_pf'], linestyle=ls['grafea_pf'],
                linewidth=2, label=lb['grafea_pf'], zorder=3)

    if len(fem_pf_path) > 0:
        ax.plot(fem_pf_path[:, 0], fem_pf_path[:, 1],
                color=c['fem_pf'], linestyle=ls['fem_pf'],
                linewidth=2, label=lb['fem_pf'], zorder=2)

    if original_path is not None and len(original_path) > 0:
        ax.plot(original_path[:, 0], original_path[:, 1],
                color=c['grafea_original'], linestyle=ls['grafea_original'],
                linewidth=1.5, label=lb['grafea_original'], zorder=2)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('SENT: Crack Path Comparison')
    ax.set_xlim([0, L])
    ax.set_ylim([0.3 * L, 0.7 * L])
    ax.set_aspect('equal')
    ax.legend(loc='best')

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 7: SENT Load-Displacement Curves
# ============================================================

def plot_sent_load_displacement(results_dict, save_path=None):
    """
    Plot SENT load-displacement curves for multiple methods.

    Parameters
    ----------
    results_dict : dict
        Keys are method names, values are dicts with 'displacement' and 'force'.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_PARAMS['figsize'])

    c = FIGURE_PARAMS['colors']
    ls = FIGURE_PARAMS['linestyles']
    lb = FIGURE_PARAMS['labels']

    for method_key, result in results_dict.items():
        u = result['displacement']
        F = result['force']
        color = c.get(method_key, '#000000')
        style = ls.get(method_key, '-')
        label = lb.get(method_key, method_key)
        ax.plot(u * 1000, F, color=color, linestyle=style,
                linewidth=FIGURE_PARAMS['line_width'], label=label)

    ax.set_xlabel('Displacement (mm $\\times$ $10^3$)')
    ax.set_ylabel('Reaction Force (N)')
    ax.set_title('SENT: Load-Displacement Response')
    ax.legend(loc='best')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 9: SENS Crack Path Comparison
# ============================================================

def plot_sens_crack_paths(grafea_pf_path, fem_pf_path, original_path=None,
                          L=1.0, expected_angle=70.0, save_path=None):
    """
    Plot SENS crack path comparison.

    Parameters
    ----------
    grafea_pf_path, fem_pf_path : np.ndarray
        Crack paths.
    original_path : np.ndarray, optional
        Original GraFEA crack path.
    L : float
        Domain size.
    expected_angle : float
        Expected crack angle in degrees from horizontal.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_PARAMS['figsize'])

    c = FIGURE_PARAMS['colors']
    ls = FIGURE_PARAMS['linestyles']
    lb = FIGURE_PARAMS['labels']

    # Expected direction line
    angle_rad = np.radians(expected_angle)
    x_start = 0.5 * L
    y_start = 0.5 * L
    dx = 0.4 * L * np.cos(angle_rad)
    dy = 0.4 * L * np.sin(angle_rad)
    ax.plot([x_start, x_start + dx], [y_start, y_start + dy],
            color=c['reference'], linestyle=':', linewidth=1,
            label=f'Expected (~{expected_angle:.0f}$^\\circ$)', zorder=1)

    if len(grafea_pf_path) > 0:
        ax.plot(grafea_pf_path[:, 0], grafea_pf_path[:, 1],
                color=c['grafea_pf'], linestyle=ls['grafea_pf'],
                linewidth=2, label=lb['grafea_pf'], zorder=3)

    if len(fem_pf_path) > 0:
        ax.plot(fem_pf_path[:, 0], fem_pf_path[:, 1],
                color=c['fem_pf'], linestyle=ls['fem_pf'],
                linewidth=2, label=lb['fem_pf'], zorder=2)

    if original_path is not None and len(original_path) > 0:
        ax.plot(original_path[:, 0], original_path[:, 1],
                color=c['grafea_original'], linestyle=ls['grafea_original'],
                linewidth=1.5, label=lb['grafea_original'], zorder=2)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('SENS: Crack Path Comparison')
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.set_aspect('equal')
    ax.legend(loc='best')

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 11: L-Shaped Panel Nucleation Comparison
# ============================================================

def plot_l_shaped_nucleation(grafea_pf_damage, original_damage,
                             mesh, save_path=None):
    """
    Plot L-shaped panel nucleation comparison.

    Parameters
    ----------
    grafea_pf_damage : np.ndarray
        GraFEA-PF final damage (edge-based).
    original_damage : np.ndarray
        Original GraFEA final damage (edge-based).
    mesh : TriangleMesh
        Mesh for both.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_PARAMS['figsize_wide'])

    # Plot damage as colored edges
    for ax, damage, title in [
        (ax1, grafea_pf_damage, 'GraFEA-PF (nucleates)'),
        (ax2, original_damage, 'Original GraFEA (no nucleation)')
    ]:
        _plot_edge_damage(ax, mesh, damage)
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')

    if save_path:
        _save_figure(fig, save_path)
    return fig


def _plot_edge_damage(ax, mesh, damage, cmap='hot_r'):
    """Helper to plot edge damage on mesh."""
    if not HAS_MATPLOTLIB:
        return

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    for i in range(mesh.n_edges):
        n1, n2 = mesh.edges[i]
        x = [mesh.nodes[n1, 0], mesh.nodes[n2, 0]]
        y = [mesh.nodes[n1, 1], mesh.nodes[n2, 1]]
        color = sm.to_rgba(damage[i])
        lw = 0.3 if damage[i] < 0.1 else (1.5 if damage[i] > 0.8 else 0.8)
        ax.plot(x, y, color=color, linewidth=lw)

    plt.colorbar(sm, ax=ax, label='Damage')


# ============================================================
# Fig. 12: Mesh Convergence Crack Paths
# ============================================================

def plot_mesh_convergence_paths(results, method='grafea_pf',
                                mesh_type='structured', save_path=None):
    """
    Plot crack path overlay for multiple mesh sizes.

    Parameters
    ----------
    results : dict
        Mesh sensitivity results keyed by "{name}_{type}".
    method : str
        'grafea_pf' or 'grafea_original'.
    mesh_type : str
        'structured' or 'unstructured'.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_PARAMS['figsize'])

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
    mesh_labels = ['coarse', 'medium', 'fine', 'very_fine']

    for idx, name in enumerate(mesh_labels):
        key = f"{name}_{mesh_type}"
        if key not in results:
            continue
        data = results[key]
        if method in data and 'crack_path' in data[method]:
            path = np.asarray(data[method]['crack_path'])
            if len(path) > 0:
                h = data.get('mesh_size', data.get('h_fine', '?'))
                label = f'h = {h}' if isinstance(h, str) else f'h = {h:.5f}'
                ax.plot(path[:, 0], path[:, 1],
                        color=colors[idx], linewidth=1.5 + 0.3 * idx,
                        label=label)

    title_method = 'GraFEA-PF' if method == 'grafea_pf' else 'Original GraFEA'
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Mesh Convergence: {title_method} ({mesh_type})')
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 13: Mesh Convergence Load-Displacement
# ============================================================

def plot_mesh_convergence_ld(results, method='grafea_pf',
                              mesh_type='structured', save_path=None):
    """
    Plot load-displacement curves for multiple mesh sizes.

    Parameters
    ----------
    results : dict
        Mesh sensitivity results.
    method : str
        'grafea_pf' or 'grafea_original'.
    mesh_type : str
        'structured' or 'unstructured'.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGURE_PARAMS['figsize'])

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
    mesh_labels = ['coarse', 'medium', 'fine', 'very_fine']

    for idx, name in enumerate(mesh_labels):
        key = f"{name}_{mesh_type}"
        if key not in results:
            continue
        data = results[key]
        if method in data:
            u = np.asarray(data[method].get('displacement', []))
            F = np.asarray(data[method].get('force', []))
            if len(u) > 0 and len(F) > 0:
                h = data.get('mesh_size', '?')
                label = f'h = {h}' if isinstance(h, str) else f'h = {h:.5f}'
                ax.plot(u * 1000, F, color=colors[idx],
                        linewidth=1.5, label=label)

    title_method = 'GraFEA-PF' if method == 'grafea_pf' else 'Original GraFEA'
    ax.set_xlabel('Displacement (mm $\\times$ $10^3$)')
    ax.set_ylabel('Reaction Force (N)')
    ax.set_title(f'Mesh Convergence: {title_method} ({mesh_type})')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 14: Length Scale Study
# ============================================================

def plot_length_scale_study(analysis, save_path=None):
    """
    Plot length scale study results (band width and peak load vs l0).

    Parameters
    ----------
    analysis : dict
        Analysis from analyze_length_scale_effects.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    # Collect data
    l0_vals = sorted(k for k in analysis.keys() if k != 'trends')
    band_widths = [analysis[l0]['damage_band_width'] for l0 in l0_vals]
    peak_loads = [analysis[l0]['peak_load'] for l0 in l0_vals]
    dissipated = [analysis[l0]['dissipated_energy'] for l0 in l0_vals]
    l0_arr = np.array(l0_vals)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # Band width vs l0
    ax1.plot(l0_arr * 1000, np.array(band_widths) * 1000,
             'o-', color=FIGURE_PARAMS['colors']['grafea_pf'], linewidth=2)
    # Reference: width = 2*l0 and 4*l0
    ax1.plot(l0_arr * 1000, 2 * l0_arr * 1000, '--', color='gray',
             linewidth=1, label='$2\\ell_0$')
    ax1.plot(l0_arr * 1000, 4 * l0_arr * 1000, ':', color='gray',
             linewidth=1, label='$4\\ell_0$')
    ax1.set_xlabel('$\\ell_0$ (mm $\\times$ $10^3$)')
    ax1.set_ylabel('Band Width (mm $\\times$ $10^3$)')
    ax1.set_title('Damage Band Width')
    ax1.legend()

    # Peak load vs l0
    ax2.plot(l0_arr * 1000, peak_loads, 's-',
             color=FIGURE_PARAMS['colors']['grafea_pf'], linewidth=2)
    ax2.set_xlabel('$\\ell_0$ (mm $\\times$ $10^3$)')
    ax2.set_ylabel('Peak Load (N)')
    ax2.set_title('Peak Load')

    # Dissipated energy vs l0
    ax3.plot(l0_arr * 1000, dissipated, '^-',
             color=FIGURE_PARAMS['colors']['grafea_pf'], linewidth=2)
    ax3.set_xlabel('$\\ell_0$ (mm $\\times$ $10^3$)')
    ax3.set_ylabel('Dissipated Energy (N$\\cdot$mm)')
    ax3.set_title('Dissipated Energy')

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Fig. 15: Computational Cost Scaling
# ============================================================

def plot_computational_cost(results, analysis=None, save_path=None):
    """
    Plot computational cost comparison.

    Parameters
    ----------
    results : list of dict
        Profiling results from run_scaling_study.
    analysis : dict, optional
        Scaling analysis.
    save_path : str, optional
        Path to save figure.
    """
    if not HAS_MATPLOTLIB:
        return
    _setup_style()

    nodes = np.array([r['actual_nodes'] for r in results])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    c = FIGURE_PARAMS['colors']
    lb = FIGURE_PARAMS['labels']

    # Total time per step vs nodes (log-log)
    grafea_total = [r['grafea_pf_timing']['total'] for r in results]
    fem_total = [r['fem_pf_timing']['total'] for r in results]

    ax1.loglog(nodes, grafea_total, 'o-', color=c['grafea_pf'],
               label=lb['grafea_pf'], linewidth=2)
    ax1.loglog(nodes, fem_total, 's--', color=c['fem_pf'],
               label=lb['fem_pf'], linewidth=2)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Time per Step (s)')
    ax1.set_title('Total Time Scaling')
    ax1.legend()

    # Timing breakdown (stacked bar for largest mesh)
    r = results[-1]  # Largest mesh
    ops = ['assembly_K', 'solve_u', 'assembly_damage', 'solve_d']
    op_labels = ['K assembly', 'u solve', 'd assembly', 'd solve']
    x = np.arange(2)
    width = 0.35

    grafea_vals = [r['grafea_pf_timing'].get(op, 0) for op in ops]
    fem_vals = [r['fem_pf_timing'].get(op, 0) for op in ops]

    bottom_g = np.zeros(1)
    bottom_f = np.zeros(1)
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (gv, fv, label, bc) in enumerate(
            zip(grafea_vals, fem_vals, op_labels, bar_colors)):
        ax2.bar(0, gv, width, bottom=bottom_g[0], label=label if True else '',
                color=bc, alpha=0.85)
        ax2.bar(1, fv, width, bottom=bottom_f[0], color=bc, alpha=0.85)
        bottom_g[0] += gv
        bottom_f[0] += fv

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([lb['grafea_pf'], lb['fem_pf']])
    ax2.set_ylabel('Time (s)')
    ax2.set_title(f'Timing Breakdown ({r["actual_nodes"]} nodes)')
    ax2.legend(fontsize=8)

    # Memory vs nodes
    grafea_mem = [r['grafea_pf_memory_MB'] for r in results]
    fem_mem = [r['fem_pf_memory_MB'] for r in results]

    ax3.plot(nodes, grafea_mem, 'o-', color=c['grafea_pf'],
             label=lb['grafea_pf'], linewidth=2)
    ax3.plot(nodes, fem_mem, 's--', color=c['fem_pf'],
             label=lb['fem_pf'], linewidth=2)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Peak Memory (MB)')
    ax3.set_title('Memory Usage')
    ax3.legend()

    # DOF comparison
    grafea_dof_d = [r['grafea_pf_dof_d'] for r in results]
    fem_dof_d = [r['fem_pf_dof_d'] for r in results]

    ax4.plot(nodes, grafea_dof_d, 'o-', color=c['grafea_pf'],
             label=f'{lb["grafea_pf"]} (edges)', linewidth=2)
    ax4.plot(nodes, fem_dof_d, 's--', color=c['fem_pf'],
             label=f'{lb["fem_pf"]} (nodes)', linewidth=2)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Damage DOFs')
    ax4.set_title('Damage DOF Count')
    ax4.legend()

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ============================================================
# Summary: Generate All Figures
# ============================================================

def generate_all_figures(comparison_results, mesh_results=None,
                         length_scale_analysis=None, scaling_results=None,
                         output_dir='docs/figures'):
    """
    Generate all publication-quality figures.

    Parameters
    ----------
    comparison_results : dict
        Results from compare_grafea_vs_fem for SENT and SENS.
    mesh_results : dict, optional
        Results from mesh sensitivity study.
    length_scale_analysis : dict, optional
        Analysis from length scale study.
    scaling_results : list, optional
        Results from scaling study.
    output_dir : str
        Output directory for figures.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; cannot generate figures.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fmt = FIGURE_PARAMS['format']

    print("Generating publication-quality figures...")

    # Fig 6: SENT crack paths
    if 'SENT' in comparison_results:
        sent = comparison_results['SENT']
        gp_path = np.asarray(sent.get('grafea_pf', {}).get('crack_path', []))
        fp_path = np.asarray(sent.get('fem_pf', {}).get('crack_path', []))
        og_path = np.asarray(sent.get('grafea_original', {}).get('crack_path', []))
        if len(gp_path) > 0 or len(fp_path) > 0:
            plot_sent_crack_paths(
                gp_path, fp_path, og_path if len(og_path) > 0 else None,
                save_path=os.path.join(output_dir, f'fig06_sent_crack_paths.{fmt}')
            )
            print(f"  Fig. 6: SENT crack paths -> saved")

    # Fig 7: SENT load-displacement
    if 'SENT' in comparison_results:
        sent = comparison_results['SENT']
        results_dict = {}
        for key in ['grafea_pf', 'fem_pf', 'grafea_original']:
            if key in sent:
                data = sent[key]
                if 'displacement' in data and 'force' in data:
                    results_dict[key] = data
        if results_dict:
            plot_sent_load_displacement(
                results_dict,
                save_path=os.path.join(output_dir, f'fig07_sent_load_displacement.{fmt}')
            )
            print(f"  Fig. 7: SENT load-displacement -> saved")

    # Fig 9: SENS crack paths
    if 'SENS' in comparison_results:
        sens = comparison_results['SENS']
        gp_path = np.asarray(sens.get('grafea_pf', {}).get('crack_path', []))
        fp_path = np.asarray(sens.get('fem_pf', {}).get('crack_path', []))
        og_path = np.asarray(sens.get('grafea_original', {}).get('crack_path', []))
        if len(gp_path) > 0 or len(fp_path) > 0:
            plot_sens_crack_paths(
                gp_path, fp_path, og_path if len(og_path) > 0 else None,
                save_path=os.path.join(output_dir, f'fig09_sens_crack_paths.{fmt}')
            )
            print(f"  Fig. 9: SENS crack paths -> saved")

    # Fig 12: Mesh convergence paths
    if mesh_results:
        for method in ['grafea_pf', 'grafea_original']:
            method_label = 'pf' if method == 'grafea_pf' else 'orig'
            plot_mesh_convergence_paths(
                mesh_results, method=method, mesh_type='structured',
                save_path=os.path.join(
                    output_dir,
                    f'fig12_mesh_convergence_paths_{method_label}.{fmt}'
                )
            )
        print(f"  Fig. 12: Mesh convergence paths -> saved")

    # Fig 13: Mesh convergence L-D
    if mesh_results:
        plot_mesh_convergence_ld(
            mesh_results, method='grafea_pf', mesh_type='structured',
            save_path=os.path.join(output_dir, f'fig13_mesh_convergence_ld.{fmt}')
        )
        print(f"  Fig. 13: Mesh convergence L-D -> saved")

    # Fig 14: Length scale study
    if length_scale_analysis:
        plot_length_scale_study(
            length_scale_analysis,
            save_path=os.path.join(output_dir, f'fig14_length_scale_study.{fmt}')
        )
        print(f"  Fig. 14: Length scale study -> saved")

    # Fig 15: Computational cost
    if scaling_results:
        plot_computational_cost(
            scaling_results,
            save_path=os.path.join(output_dir, f'fig15_computational_cost.{fmt}')
        )
        print(f"  Fig. 15: Computational cost -> saved")

    print("All figures generated.")
