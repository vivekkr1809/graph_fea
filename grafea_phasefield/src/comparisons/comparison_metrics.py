"""
Comparison Metrics
==================

Quantitative comparison metrics for comparing crack paths, load-displacement
curves, and computational efficiency between different methods (GraFEA-PF,
standard FEM phase-field, original GraFEA).

Functions:
    compute_path_deviation       - Distance metrics between two crack paths
    compute_load_displacement_error - Compare load-displacement curves
    compute_path_smoothness      - Curvature-based path smoothness analysis
    compute_crack_angle          - Crack propagation angle relative to reference
    compute_energy_metrics       - Compare energy quantities between methods
    compare_efficiency           - Compare computational efficiency
    generate_comparison_table    - Format comparison results as a string table
    validate_comparison          - Check results against validation criteria
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional


# ============================================================================
# Default validation criteria
# ============================================================================
DEFAULT_CRITERIA = {
    'sent_hausdorff_h_ratio': 2.0,      # Hausdorff distance < 2 * h_fine
    'sent_peak_load_error': 0.05,        # Peak load relative error < 5%
    'sent_l2_error': 0.10,               # L2 curve error < 10%
    'sens_hausdorff_h_ratio': 3.0,       # Hausdorff distance < 3 * h_fine
    'sens_peak_load_error': 0.10,        # Peak load relative error < 10%
    'sens_crack_angle_tolerance': 5.0,   # Crack angle within +/-5 deg of ~70 deg
}


# ============================================================================
# 1. Path Deviation Metrics
# ============================================================================
def compute_path_deviation(path_a: np.ndarray,
                           path_b: np.ndarray) -> Dict[str, float]:
    """
    Compute multiple distance metrics between two crack paths.

    Uses a combination of Hausdorff distance and mean/max point-to-curve
    distances to quantify how closely two crack paths agree. Both symmetric
    and directed variants are returned.

    Parameters
    ----------
    path_a : np.ndarray, shape (N, 2)
        Crack path coordinates (x, y) for method A.
    path_b : np.ndarray, shape (M, 2)
        Crack path coordinates (x, y) for method B.

    Returns
    -------
    dict
        'hausdorff'    : float — symmetric Hausdorff distance (max of directed)
        'mean_deviation': float — mean of bidirectional point-to-curve distances
        'max_deviation' : float — maximum point-to-curve distance (either direction)
        'directed_ab'  : float — directed Hausdorff distance from A to B
        'directed_ba'  : float — directed Hausdorff distance from B to A

    Notes
    -----
    If either path is empty (zero points), all metrics are set to np.inf
    except 'mean_deviation' which is set to np.nan (undefined average).
    """
    # Handle empty paths
    if path_a.ndim != 2:
        path_a = path_a.reshape(-1, 2)
    if path_b.ndim != 2:
        path_b = path_b.reshape(-1, 2)

    if len(path_a) == 0 or len(path_b) == 0:
        return {
            'hausdorff': np.inf,
            'mean_deviation': np.nan,
            'max_deviation': np.inf,
            'directed_ab': np.inf,
            'directed_ba': np.inf,
        }

    # Directed Hausdorff distances using scipy
    d_ab, _, _ = directed_hausdorff(path_a, path_b)
    d_ba, _, _ = directed_hausdorff(path_b, path_a)
    hausdorff = max(d_ab, d_ba)

    # Point-to-curve distances using cKDTree for efficiency
    tree_b = cKDTree(path_b)
    tree_a = cKDTree(path_a)

    # Distances from every point in A to nearest point in B
    dists_a_to_b, _ = tree_b.query(path_a)
    # Distances from every point in B to nearest point in A
    dists_b_to_a, _ = tree_a.query(path_b)

    # Bidirectional mean deviation: average of both sets of nearest-neighbour distances
    mean_deviation = 0.5 * (np.mean(dists_a_to_b) + np.mean(dists_b_to_a))

    # Maximum deviation across both directions
    max_deviation = max(np.max(dists_a_to_b), np.max(dists_b_to_a))

    return {
        'hausdorff': float(hausdorff),
        'mean_deviation': float(mean_deviation),
        'max_deviation': float(max_deviation),
        'directed_ab': float(d_ab),
        'directed_ba': float(d_ba),
    }


# ============================================================================
# 2. Load-Displacement Curve Error
# ============================================================================
def compute_load_displacement_error(results_a, results_b) -> Dict[str, float]:
    """
    Compare load-displacement curves from two methods.

    Both curves are interpolated to a common set of displacement points
    (the union of both displacement ranges, clipped to their intersection).
    The comparison is performed on the interpolated curves.

    Parameters
    ----------
    results_a, results_b : dict-like or SENTResults/SENSResults
        Must provide 'displacement' and 'force' as arrays, either via
        dictionary keys or as object attributes.  Method B is treated as the
        reference for relative error calculations.

    Returns
    -------
    dict
        'peak_load_a', 'peak_load_b'             : peak loads
        'peak_load_error'                         : |F_a - F_b| / |F_b|
        'peak_displacement_a', 'peak_displacement_b' : displacement at peak
        'peak_displacement_error'                 : relative error in peak disp.
        'l2_error'                                : L2 norm of difference / L2 of reference
        'area_under_curve_a', 'area_under_curve_b': integral F du (energy proxy)
        'area_error'                              : relative error in area
    """
    # Extract arrays from dict-like or attribute-based objects
    u_a, f_a = _extract_load_disp(results_a)
    u_b, f_b = _extract_load_disp(results_b)

    # --- Peak load metrics ---
    peak_load_a = float(np.max(f_a))
    peak_load_b = float(np.max(f_b))
    idx_peak_a = int(np.argmax(f_a))
    idx_peak_b = int(np.argmax(f_b))
    peak_disp_a = float(u_a[idx_peak_a])
    peak_disp_b = float(u_b[idx_peak_b])

    peak_load_error = _safe_relative_error(peak_load_a, peak_load_b)
    peak_disp_error = _safe_relative_error(peak_disp_a, peak_disp_b)

    # --- Interpolate to common displacement grid ---
    u_min = max(np.min(u_a), np.min(u_b))
    u_max = min(np.max(u_a), np.max(u_b))

    if u_max <= u_min:
        # No overlapping range
        return {
            'peak_load_a': peak_load_a,
            'peak_load_b': peak_load_b,
            'peak_load_error': peak_load_error,
            'peak_displacement_a': peak_disp_a,
            'peak_displacement_b': peak_disp_b,
            'peak_displacement_error': peak_disp_error,
            'l2_error': np.nan,
            'area_under_curve_a': float(np.trapezoid(f_a, u_a)),
            'area_under_curve_b': float(np.trapezoid(f_b, u_b)),
            'area_error': np.nan,
        }

    n_common = max(len(u_a), len(u_b), 200)
    u_common = np.linspace(u_min, u_max, n_common)

    interp_a = interp1d(u_a, f_a, kind='linear', bounds_error=False,
                         fill_value=0.0)
    interp_b = interp1d(u_b, f_b, kind='linear', bounds_error=False,
                         fill_value=0.0)

    f_a_interp = interp_a(u_common)
    f_b_interp = interp_b(u_common)

    # --- L2 error (normalized by reference) ---
    diff_norm = np.sqrt(np.trapezoid((f_a_interp - f_b_interp) ** 2, u_common))
    ref_norm = np.sqrt(np.trapezoid(f_b_interp ** 2, u_common))
    l2_error = diff_norm / ref_norm if ref_norm > 1e-15 else np.inf

    # --- Area under curve (dissipated energy proxy) ---
    area_a = float(np.trapezoid(f_a, u_a))
    area_b = float(np.trapezoid(f_b, u_b))
    area_error = _safe_relative_error(area_a, area_b)

    return {
        'peak_load_a': peak_load_a,
        'peak_load_b': peak_load_b,
        'peak_load_error': float(peak_load_error),
        'peak_displacement_a': peak_disp_a,
        'peak_displacement_b': peak_disp_b,
        'peak_displacement_error': float(peak_disp_error),
        'l2_error': float(l2_error),
        'area_under_curve_a': area_a,
        'area_under_curve_b': area_b,
        'area_error': float(area_error),
    }


# ============================================================================
# 3. Path Smoothness
# ============================================================================
def compute_path_smoothness(path: np.ndarray) -> Dict[str, float]:
    """
    Quantify crack path smoothness using discrete curvature analysis.

    The discrete curvature at each interior point is computed via the
    circumscribed-circle formula (Menger curvature). Direction changes
    are counted when the local turning angle exceeds 15 degrees.

    Parameters
    ----------
    path : np.ndarray, shape (N, 2)
        Ordered crack path coordinates (x, y).

    Returns
    -------
    dict
        'mean_curvature'    : average discrete curvature along path
        'max_curvature'     : maximum curvature
        'direction_changes' : count of points with angle deviation > 15 deg
        'total_path_length' : arc length of path
        'straightness_ratio': end-to-end distance / total path length
                              (1.0 = perfectly straight)
    """
    if path.ndim != 2:
        path = path.reshape(-1, 2)

    n = len(path)

    if n < 3:
        # Not enough points for curvature analysis
        if n < 2:
            return {
                'mean_curvature': 0.0,
                'max_curvature': 0.0,
                'direction_changes': 0,
                'total_path_length': 0.0,
                'straightness_ratio': 1.0,
            }

        # Exactly 2 points: can compute length and straightness
        seg_length = np.linalg.norm(path[1] - path[0])
        return {
            'mean_curvature': 0.0,
            'max_curvature': 0.0,
            'direction_changes': 0,
            'total_path_length': float(seg_length),
            'straightness_ratio': 1.0,
        }

    # Segment vectors and lengths
    segments = np.diff(path, axis=0)  # shape (n-1, 2)
    seg_lengths = np.linalg.norm(segments, axis=1)  # shape (n-1,)
    total_path_length = float(np.sum(seg_lengths))

    # Straight-line (end-to-end) distance
    end_to_end = float(np.linalg.norm(path[-1] - path[0]))
    straightness_ratio = end_to_end / total_path_length if total_path_length > 1e-15 else 1.0

    # Discrete curvature at interior points using Menger curvature:
    #   kappa = 2 * |cross(P1-P0, P2-P0)| / (|P1-P0| * |P2-P1| * |P2-P0|)
    curvatures = np.zeros(n - 2)
    for i in range(n - 2):
        p0 = path[i]
        p1 = path[i + 1]
        p2 = path[i + 2]

        d01 = np.linalg.norm(p1 - p0)
        d12 = np.linalg.norm(p2 - p1)
        d02 = np.linalg.norm(p2 - p0)

        denom = d01 * d12 * d02
        if denom < 1e-30:
            curvatures[i] = 0.0
        else:
            # 2D cross product magnitude
            cross_mag = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                            (p1[1] - p0[1]) * (p2[0] - p0[0]))
            curvatures[i] = 2.0 * cross_mag / denom

    mean_curvature = float(np.mean(curvatures))
    max_curvature = float(np.max(curvatures))

    # Count direction changes: angle between consecutive segments > 15 deg
    direction_changes = 0
    angle_threshold_rad = np.radians(15.0)

    for i in range(len(segments) - 1):
        v1 = segments[i]
        v2 = segments[i + 1]
        len1 = seg_lengths[i]
        len2 = seg_lengths[i + 1]

        if len1 < 1e-15 or len2 < 1e-15:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle > angle_threshold_rad:
            direction_changes += 1

    return {
        'mean_curvature': mean_curvature,
        'max_curvature': max_curvature,
        'direction_changes': int(direction_changes),
        'total_path_length': total_path_length,
        'straightness_ratio': float(straightness_ratio),
    }


# ============================================================================
# 4. Crack Propagation Angle
# ============================================================================
def compute_crack_angle(crack_path: np.ndarray,
                        reference_direction: Tuple[float, float] = (1, 0)
                        ) -> Dict[str, float]:
    """
    Compute the angle of crack propagation relative to a reference direction.

    Local propagation angles are computed from consecutive point pairs along
    the crack path, then summarized via mean, initial, final, and standard
    deviation statistics.

    Parameters
    ----------
    crack_path : np.ndarray, shape (N, 2)
        Ordered crack path coordinates (x, y).
    reference_direction : tuple (dx, dy), default (1, 0)
        Reference direction for angle measurement. Default is horizontal.

    Returns
    -------
    dict
        'mean_angle_deg'   : mean propagation angle in degrees
        'initial_angle_deg': angle from the first few points
        'final_angle_deg'  : angle from the last few points
        'angle_std_deg'    : standard deviation of local angles
    """
    if crack_path.ndim != 2:
        crack_path = crack_path.reshape(-1, 2)

    n = len(crack_path)

    if n < 2:
        return {
            'mean_angle_deg': 0.0,
            'initial_angle_deg': 0.0,
            'final_angle_deg': 0.0,
            'angle_std_deg': 0.0,
        }

    # Reference direction unit vector
    ref = np.array(reference_direction, dtype=float)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-15:
        ref = np.array([1.0, 0.0])
    else:
        ref = ref / ref_norm

    # Compute local propagation angles at each segment
    segments = np.diff(crack_path, axis=0)  # shape (n-1, 2)
    seg_lengths = np.linalg.norm(segments, axis=1)

    local_angles = []
    for i in range(len(segments)):
        if seg_lengths[i] < 1e-15:
            continue
        seg_unit = segments[i] / seg_lengths[i]
        # Signed angle from reference direction using atan2
        angle = np.degrees(np.arctan2(
            ref[0] * seg_unit[1] - ref[1] * seg_unit[0],
            ref[0] * seg_unit[0] + ref[1] * seg_unit[1]
        ))
        local_angles.append(angle)

    if len(local_angles) == 0:
        return {
            'mean_angle_deg': 0.0,
            'initial_angle_deg': 0.0,
            'final_angle_deg': 0.0,
            'angle_std_deg': 0.0,
        }

    local_angles = np.array(local_angles)
    mean_angle = float(np.mean(local_angles))
    angle_std = float(np.std(local_angles))

    # Initial angle: from first few segments (up to 5 or 10% of path)
    n_initial = max(1, min(5, len(local_angles) // 10 + 1))
    initial_angle = float(np.mean(local_angles[:n_initial]))

    # Final angle: from last few segments
    n_final = max(1, min(5, len(local_angles) // 10 + 1))
    final_angle = float(np.mean(local_angles[-n_final:]))

    return {
        'mean_angle_deg': mean_angle,
        'initial_angle_deg': initial_angle,
        'final_angle_deg': final_angle,
        'angle_std_deg': angle_std,
    }


# ============================================================================
# 5. Energy Metrics
# ============================================================================
def compute_energy_metrics(results_a, results_b) -> Dict[str, float]:
    """
    Compare energy quantities between two methods.

    Computes relative errors in final strain energy, final surface energy,
    and total energy dissipation. Method B is treated as the reference.

    Parameters
    ----------
    results_a, results_b : object or dict
        Must provide 'strain_energy' and 'surface_energy' arrays, either
        as dictionary keys or as object attributes (e.g., SENTResults).

    Returns
    -------
    dict
        'final_strain_energy_a', 'final_strain_energy_b' : final strain energies
        'strain_energy_error'   : relative error at final step
        'final_surface_energy_a', 'final_surface_energy_b' : final surface energies
        'surface_energy_error'  : relative error at final step
        'total_dissipation_a', 'total_dissipation_b' : total energy dissipated
    """
    se_a = _get_array(results_a, 'strain_energy')
    se_b = _get_array(results_b, 'strain_energy')
    surf_a = _get_array(results_a, 'surface_energy')
    surf_b = _get_array(results_b, 'surface_energy')

    # Final values
    final_se_a = float(se_a[-1]) if len(se_a) > 0 else 0.0
    final_se_b = float(se_b[-1]) if len(se_b) > 0 else 0.0
    final_surf_a = float(surf_a[-1]) if len(surf_a) > 0 else 0.0
    final_surf_b = float(surf_b[-1]) if len(surf_b) > 0 else 0.0

    # Relative errors (B is reference)
    strain_energy_error = _safe_relative_error(final_se_a, final_se_b)
    surface_energy_error = _safe_relative_error(final_surf_a, final_surf_b)

    # Total dissipation: sum of surface energy increments (monotonically
    # increasing by irreversibility, so final - initial is the total)
    total_dissipation_a = float(surf_a[-1] - surf_a[0]) if len(surf_a) > 1 else 0.0
    total_dissipation_b = float(surf_b[-1] - surf_b[0]) if len(surf_b) > 1 else 0.0

    return {
        'final_strain_energy_a': final_se_a,
        'final_strain_energy_b': final_se_b,
        'strain_energy_error': float(strain_energy_error),
        'final_surface_energy_a': final_surf_a,
        'final_surface_energy_b': final_surf_b,
        'surface_energy_error': float(surface_energy_error),
        'total_dissipation_a': total_dissipation_a,
        'total_dissipation_b': total_dissipation_b,
    }


# ============================================================================
# 6. Computational Efficiency Comparison
# ============================================================================
def compare_efficiency(timing_a: Dict[str, float],
                       timing_b: Dict[str, float],
                       dofs_a: Dict[str, int],
                       dofs_b: Dict[str, int]) -> Dict[str, float]:
    """
    Compare computational efficiency between two methods.

    Timing ratios less than 1.0 indicate that method A is faster than
    method B.

    Parameters
    ----------
    timing_a, timing_b : dict
        Mean timing values with keys:
        - 'assembly_displacement_mean' : mean time for displacement assembly (s)
        - 'solve_displacement_mean'    : mean time for displacement solve (s)
        - 'assembly_damage_mean'       : mean time for damage assembly (s)
        - 'solve_damage_mean'          : mean time for damage solve (s)
        - 'total_per_step_mean'        : mean total time per staggered step (s)

    dofs_a, dofs_b : dict
        DOF counts with keys:
        - 'displacement' : number of displacement DOFs
        - 'damage'       : number of damage DOFs

    Returns
    -------
    dict
        DOF counts, DOF ratios, absolute timings, timing ratios, and
        speedup factor.
    """
    result = {}

    # --- DOF counts ---
    disp_dofs_a = dofs_a.get('displacement', 0)
    disp_dofs_b = dofs_b.get('displacement', 0)
    dmg_dofs_a = dofs_a.get('damage', 0)
    dmg_dofs_b = dofs_b.get('damage', 0)

    result['displacement_dofs_a'] = disp_dofs_a
    result['displacement_dofs_b'] = disp_dofs_b
    result['damage_dofs_a'] = dmg_dofs_a
    result['damage_dofs_b'] = dmg_dofs_b
    result['total_dofs_a'] = disp_dofs_a + dmg_dofs_a
    result['total_dofs_b'] = disp_dofs_b + dmg_dofs_b

    # DOF ratios
    result['displacement_dof_ratio'] = (
        disp_dofs_a / disp_dofs_b if disp_dofs_b > 0 else np.inf
    )
    result['damage_dof_ratio'] = (
        dmg_dofs_a / dmg_dofs_b if dmg_dofs_b > 0 else np.inf
    )
    result['total_dof_ratio'] = (
        (disp_dofs_a + dmg_dofs_a) / (disp_dofs_b + dmg_dofs_b)
        if (disp_dofs_b + dmg_dofs_b) > 0 else np.inf
    )

    # --- Absolute timings ---
    timing_keys = [
        'assembly_displacement_mean',
        'solve_displacement_mean',
        'assembly_damage_mean',
        'solve_damage_mean',
        'total_per_step_mean',
    ]

    for key in timing_keys:
        result[f'{key}_a'] = timing_a.get(key, 0.0)
        result[f'{key}_b'] = timing_b.get(key, 0.0)

    # --- Timing ratios (a / b, < 1 means A is faster) ---
    for key in timing_keys:
        val_a = timing_a.get(key, 0.0)
        val_b = timing_b.get(key, 0.0)
        ratio_key = f'{key}_ratio'
        if val_b > 1e-15:
            result[ratio_key] = val_a / val_b
        else:
            result[ratio_key] = np.inf if val_a > 1e-15 else 1.0

    # --- Overall speedup factor (B_total / A_total, >1 means A is faster) ---
    total_a = timing_a.get('total_per_step_mean', 0.0)
    total_b = timing_b.get('total_per_step_mean', 0.0)
    if total_a > 1e-15:
        result['speedup_factor'] = total_b / total_a
    else:
        result['speedup_factor'] = np.inf if total_b > 1e-15 else 1.0

    return result


# ============================================================================
# 7. Comparison Table
# ============================================================================
def generate_comparison_table(comparison_results: Dict,
                              method_names: Tuple[str, str] = ('GraFEA-PF', 'FEM-PF')
                              ) -> str:
    """
    Generate a formatted comparison table as a string.

    Produces a human-readable table summarising path deviation, load-displacement
    error, energy metrics, and efficiency ratios from a full comparison dict.

    Parameters
    ----------
    comparison_results : dict
        Dictionary of comparison metrics, typically from a full comparison
        run. Expected top-level keys (all optional):
        - 'path_deviation'  : output of compute_path_deviation
        - 'load_displacement': output of compute_load_displacement_error
        - 'energy'          : output of compute_energy_metrics
        - 'efficiency'      : output of compare_efficiency
        - 'smoothness_a', 'smoothness_b' : output of compute_path_smoothness
        - 'crack_angle_a', 'crack_angle_b' : output of compute_crack_angle

    method_names : tuple of str
        Display names for the two methods.

    Returns
    -------
    str
        Formatted comparison table.
    """
    name_a, name_b = method_names
    lines = []
    sep = '=' * 72

    lines.append(sep)
    lines.append(f'  Comparison: {name_a}  vs  {name_b}')
    lines.append(sep)

    # --- Path Deviation ---
    pd = comparison_results.get('path_deviation')
    if pd is not None:
        lines.append('')
        lines.append('  Crack Path Deviation')
        lines.append('  ' + '-' * 40)
        lines.append(f'    Hausdorff distance        : {pd["hausdorff"]:.6e}')
        lines.append(f'    Mean deviation             : {pd["mean_deviation"]:.6e}')
        lines.append(f'    Max deviation              : {pd["max_deviation"]:.6e}')
        lines.append(f'    Directed ({name_a} -> {name_b}) : {pd["directed_ab"]:.6e}')
        lines.append(f'    Directed ({name_b} -> {name_a}) : {pd["directed_ba"]:.6e}')

    # --- Load-Displacement ---
    ld = comparison_results.get('load_displacement')
    if ld is not None:
        lines.append('')
        lines.append('  Load-Displacement Comparison')
        lines.append('  ' + '-' * 40)
        lines.append(f'    Peak load ({name_a})          : {ld["peak_load_a"]:.6e}')
        lines.append(f'    Peak load ({name_b})          : {ld["peak_load_b"]:.6e}')
        lines.append(f'    Peak load error (relative) : {ld["peak_load_error"]:.4f}'
                      f'  ({ld["peak_load_error"]*100:.2f}%)')
        lines.append(f'    Peak disp ({name_a})          : {ld["peak_displacement_a"]:.6e}')
        lines.append(f'    Peak disp ({name_b})          : {ld["peak_displacement_b"]:.6e}')
        lines.append(f'    Peak disp error (relative) : {ld["peak_displacement_error"]:.4f}'
                      f'  ({ld["peak_displacement_error"]*100:.2f}%)')
        lines.append(f'    L2 error (normalized)      : {ld["l2_error"]:.4f}'
                      f'  ({ld["l2_error"]*100:.2f}%)')
        lines.append(f'    Area under curve ({name_a})   : {ld["area_under_curve_a"]:.6e}')
        lines.append(f'    Area under curve ({name_b})   : {ld["area_under_curve_b"]:.6e}')
        lines.append(f'    Area error (relative)      : {ld["area_error"]:.4f}'
                      f'  ({ld["area_error"]*100:.2f}%)')

    # --- Energy ---
    en = comparison_results.get('energy')
    if en is not None:
        lines.append('')
        lines.append('  Energy Comparison')
        lines.append('  ' + '-' * 40)
        lines.append(f'    Final strain energy ({name_a})  : {en["final_strain_energy_a"]:.6e}')
        lines.append(f'    Final strain energy ({name_b})  : {en["final_strain_energy_b"]:.6e}')
        lines.append(f'    Strain energy error          : {en["strain_energy_error"]:.4f}'
                      f'  ({en["strain_energy_error"]*100:.2f}%)')
        lines.append(f'    Final surface energy ({name_a}) : {en["final_surface_energy_a"]:.6e}')
        lines.append(f'    Final surface energy ({name_b}) : {en["final_surface_energy_b"]:.6e}')
        lines.append(f'    Surface energy error         : {en["surface_energy_error"]:.4f}'
                      f'  ({en["surface_energy_error"]*100:.2f}%)')
        lines.append(f'    Total dissipation ({name_a})    : {en["total_dissipation_a"]:.6e}')
        lines.append(f'    Total dissipation ({name_b})    : {en["total_dissipation_b"]:.6e}')

    # --- Smoothness ---
    sm_a = comparison_results.get('smoothness_a')
    sm_b = comparison_results.get('smoothness_b')
    if sm_a is not None and sm_b is not None:
        lines.append('')
        lines.append('  Path Smoothness')
        lines.append('  ' + '-' * 40)
        lines.append(f'    {"Metric":<25s}  {name_a:>12s}  {name_b:>12s}')
        lines.append(f'    {"Mean curvature":<25s}  {sm_a["mean_curvature"]:>12.4e}'
                      f'  {sm_b["mean_curvature"]:>12.4e}')
        lines.append(f'    {"Max curvature":<25s}  {sm_a["max_curvature"]:>12.4e}'
                      f'  {sm_b["max_curvature"]:>12.4e}')
        lines.append(f'    {"Direction changes":<25s}  {sm_a["direction_changes"]:>12d}'
                      f'  {sm_b["direction_changes"]:>12d}')
        lines.append(f'    {"Total path length":<25s}  {sm_a["total_path_length"]:>12.6f}'
                      f'  {sm_b["total_path_length"]:>12.6f}')
        lines.append(f'    {"Straightness ratio":<25s}  {sm_a["straightness_ratio"]:>12.6f}'
                      f'  {sm_b["straightness_ratio"]:>12.6f}')

    # --- Crack Angle ---
    ca_a = comparison_results.get('crack_angle_a')
    ca_b = comparison_results.get('crack_angle_b')
    if ca_a is not None and ca_b is not None:
        lines.append('')
        lines.append('  Crack Angle')
        lines.append('  ' + '-' * 40)
        lines.append(f'    {"Metric":<25s}  {name_a:>12s}  {name_b:>12s}')
        lines.append(f'    {"Mean angle (deg)":<25s}  {ca_a["mean_angle_deg"]:>12.2f}'
                      f'  {ca_b["mean_angle_deg"]:>12.2f}')
        lines.append(f'    {"Initial angle (deg)":<25s}  {ca_a["initial_angle_deg"]:>12.2f}'
                      f'  {ca_b["initial_angle_deg"]:>12.2f}')
        lines.append(f'    {"Final angle (deg)":<25s}  {ca_a["final_angle_deg"]:>12.2f}'
                      f'  {ca_b["final_angle_deg"]:>12.2f}')
        lines.append(f'    {"Angle std (deg)":<25s}  {ca_a["angle_std_deg"]:>12.2f}'
                      f'  {ca_b["angle_std_deg"]:>12.2f}')

    # --- Efficiency ---
    eff = comparison_results.get('efficiency')
    if eff is not None:
        lines.append('')
        lines.append('  Computational Efficiency')
        lines.append('  ' + '-' * 40)
        lines.append(f'    Total DOFs ({name_a})        : {eff.get("total_dofs_a", "N/A")}')
        lines.append(f'    Total DOFs ({name_b})        : {eff.get("total_dofs_b", "N/A")}')
        lines.append(f'    DOF ratio (A/B)           : {eff.get("total_dof_ratio", 0.0):.3f}')
        t_a = eff.get('total_per_step_mean_a', 0.0)
        t_b = eff.get('total_per_step_mean_b', 0.0)
        lines.append(f'    Time per step ({name_a})     : {t_a:.4f} s')
        lines.append(f'    Time per step ({name_b})     : {t_b:.4f} s')
        lines.append(f'    Timing ratio (A/B)        : '
                      f'{eff.get("total_per_step_mean_ratio", 0.0):.3f}')
        lines.append(f'    Speedup factor (B/A)      : '
                      f'{eff.get("speedup_factor", 0.0):.3f}')

    lines.append('')
    lines.append(sep)

    return '\n'.join(lines)


# ============================================================================
# 8. Validation Against Criteria
# ============================================================================
def validate_comparison(comparison_results: Dict,
                        criteria: Optional[Dict[str, float]] = None) -> Dict:
    """
    Check comparison results against validation criteria.

    Each criterion produces a pass/fail status along with the measured value
    and threshold.

    Parameters
    ----------
    comparison_results : dict
        Full comparison results dictionary. Expected keys depend on which
        criteria are being checked:
        - 'path_deviation' with 'hausdorff'
        - 'load_displacement' with 'peak_load_error', 'l2_error'
        - 'crack_angle_a' with 'mean_angle_deg'
        - 'h_fine' : mesh element size for Hausdorff normalization

    criteria : dict, optional
        Override default criteria thresholds. Keys:
        - 'sent_hausdorff_h_ratio' : Hausdorff / h_fine must be below this
        - 'sent_peak_load_error'   : peak load relative error threshold
        - 'sent_l2_error'          : L2 curve error threshold
        - 'sens_hausdorff_h_ratio' : Hausdorff / h_fine (for SENS)
        - 'sens_peak_load_error'   : peak load relative error threshold (SENS)
        - 'sens_crack_angle_tolerance' : degrees tolerance from ~70 deg

    Returns
    -------
    dict
        'criteria' : dict of individual criterion results, each containing:
            'passed' : bool
            'measured': measured value
            'threshold': threshold value
        'all_passed' : bool — True only if every evaluated criterion passes
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA.copy()

    validation = {'criteria': {}, 'all_passed': True}

    h_fine = comparison_results.get('h_fine', None)
    pd = comparison_results.get('path_deviation')
    ld = comparison_results.get('load_displacement')
    ca_a = comparison_results.get('crack_angle_a')

    # Helper to evaluate and store a criterion
    def _check(name, measured, threshold, comparator='less'):
        """Record a criterion check."""
        if comparator == 'less':
            passed = measured < threshold
        elif comparator == 'abs_less':
            passed = abs(measured) < threshold
        else:
            passed = measured < threshold

        validation['criteria'][name] = {
            'passed': bool(passed),
            'measured': float(measured),
            'threshold': float(threshold),
        }
        if not passed:
            validation['all_passed'] = False

    # --- SENT Hausdorff criterion ---
    if pd is not None and h_fine is not None:
        hausdorff_ratio = pd['hausdorff'] / h_fine if h_fine > 1e-15 else np.inf
        _check('sent_hausdorff_h_ratio',
               hausdorff_ratio,
               criteria.get('sent_hausdorff_h_ratio',
                            DEFAULT_CRITERIA['sent_hausdorff_h_ratio']))

    # --- SENT peak load error ---
    if ld is not None and 'sent_peak_load_error' in criteria:
        _check('sent_peak_load_error',
               ld.get('peak_load_error', np.inf),
               criteria['sent_peak_load_error'])

    # --- SENT L2 error ---
    if ld is not None and 'sent_l2_error' in criteria:
        _check('sent_l2_error',
               ld.get('l2_error', np.inf),
               criteria['sent_l2_error'])

    # --- SENS Hausdorff criterion ---
    sens_pd = comparison_results.get('sens_path_deviation')
    if sens_pd is not None and h_fine is not None:
        hausdorff_ratio = sens_pd['hausdorff'] / h_fine if h_fine > 1e-15 else np.inf
        _check('sens_hausdorff_h_ratio',
               hausdorff_ratio,
               criteria.get('sens_hausdorff_h_ratio',
                            DEFAULT_CRITERIA['sens_hausdorff_h_ratio']))

    # --- SENS peak load error ---
    sens_ld = comparison_results.get('sens_load_displacement')
    if sens_ld is not None and 'sens_peak_load_error' in criteria:
        _check('sens_peak_load_error',
               sens_ld.get('peak_load_error', np.inf),
               criteria['sens_peak_load_error'])

    # --- SENS crack angle tolerance ---
    if ca_a is not None and 'sens_crack_angle_tolerance' in criteria:
        expected_angle = 70.0
        measured_angle = ca_a.get('mean_angle_deg', 0.0)
        angle_deviation = abs(measured_angle - expected_angle)
        _check('sens_crack_angle_tolerance',
               angle_deviation,
               criteria['sens_crack_angle_tolerance'],
               comparator='less')

    return validation


# ============================================================================
# Private Helpers
# ============================================================================
def _extract_load_disp(results) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract displacement and force arrays from a results object or dict.

    Supports dict-like access (results['displacement']) and attribute access
    (results.displacement). For SENS results, falls back to 'shear_force'
    if 'force' is not available.

    Parameters
    ----------
    results : dict or object
        Must contain 'displacement' and 'force' (or 'shear_force').

    Returns
    -------
    (displacement, force) : tuple of np.ndarray
    """
    if isinstance(results, dict):
        u = np.asarray(results['displacement'])
        f = np.asarray(results.get('force', results.get('shear_force')))
    else:
        u = np.asarray(results.displacement)
        # Try 'force' first, fall back to 'shear_force' for SENSResults
        if hasattr(results, 'force'):
            f = np.asarray(results.force)
        elif hasattr(results, 'shear_force'):
            f = np.asarray(results.shear_force)
        else:
            raise AttributeError(
                "Results object must have 'force' or 'shear_force' attribute."
            )
    return u, f


def _get_array(obj, key: str) -> np.ndarray:
    """
    Get a numpy array from a dict or object attribute.

    Parameters
    ----------
    obj : dict or object
    key : str

    Returns
    -------
    np.ndarray
    """
    if isinstance(obj, dict):
        return np.asarray(obj[key])
    return np.asarray(getattr(obj, key))


def _safe_relative_error(value: float, reference: float) -> float:
    """
    Compute |value - reference| / |reference| with safe denominator.

    Returns np.inf if the reference is effectively zero.

    Parameters
    ----------
    value : float
    reference : float

    Returns
    -------
    float
    """
    if abs(reference) < 1e-15:
        return np.inf if abs(value) > 1e-15 else 0.0
    return abs(value - reference) / abs(reference)
