"""
Visualization
=============

Plotting functions for mesh, damage, displacement, and energy.
"""

import numpy as np
from typing import Optional, List, Dict, TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh
    from solvers.staggered_solver import LoadStep


def plot_mesh(mesh: 'TriangleMesh',
              ax: Optional['plt.Axes'] = None,
              show_nodes: bool = False,
              show_edges: bool = True,
              node_labels: bool = False,
              **kwargs) -> 'plt.Axes':
    """
    Plot mesh triangulation.

    Args:
        mesh: TriangleMesh instance
        ax: matplotlib axes (created if None)
        show_nodes: whether to show node points
        show_edges: whether to show edge lines
        node_labels: whether to label nodes with indices
        **kwargs: passed to triplot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if show_edges:
        tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
        ax.triplot(tri, 'k-', lw=0.5, **kwargs)

    if show_nodes:
        ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 'ko', ms=3)

    if node_labels:
        for i, (x, y) in enumerate(mesh.nodes):
            ax.annotate(str(i), (x, y), fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


def plot_damage_field(mesh: 'TriangleMesh',
                      damage: np.ndarray,
                      ax: Optional['plt.Axes'] = None,
                      cmap: str = 'hot_r',
                      vmin: float = 0.0,
                      vmax: float = 1.0,
                      linewidth: float = 2.0,
                      colorbar: bool = True,
                      **kwargs) -> 'plt.Axes':
    """
    Plot damage field on edges as colored lines.

    Args:
        mesh: TriangleMesh instance
        damage: edge damage values, shape (n_edges,)
        ax: matplotlib axes
        cmap: colormap name
        vmin, vmax: color scale limits
        linewidth: line width for edges
        colorbar: whether to show colorbar
        **kwargs: passed to LineCollection

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Create line segments for each edge
    segments = []
    for n1, n2 in mesh.edges:
        p1, p2 = mesh.nodes[n1], mesh.nodes[n2]
        segments.append([p1, p2])

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, **kwargs)
    lc.set_array(damage)
    lc.set_linewidth(linewidth)
    lc.set_clim(vmin, vmax)

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')

    if colorbar:
        plt.colorbar(lc, ax=ax, label='Damage')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


def plot_displacement_field(mesh: 'TriangleMesh',
                            u: np.ndarray,
                            component: str = 'magnitude',
                            ax: Optional['plt.Axes'] = None,
                            cmap: str = 'viridis',
                            colorbar: bool = True,
                            **kwargs) -> 'plt.Axes':
    """
    Plot displacement field on mesh.

    Args:
        mesh: TriangleMesh instance
        u: displacement vector, shape (2*n_nodes,) or (n_nodes, 2)
        component: 'x', 'y', or 'magnitude'
        ax: matplotlib axes
        cmap: colormap name
        colorbar: whether to show colorbar
        **kwargs: passed to tripcolor

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    u = np.asarray(u)
    if u.ndim == 1:
        u_reshaped = u.reshape(-1, 2)
    else:
        u_reshaped = u

    if component == 'x':
        values = u_reshaped[:, 0]
        label = 'Displacement x'
    elif component == 'y':
        values = u_reshaped[:, 1]
        label = 'Displacement y'
    elif component == 'magnitude':
        values = np.linalg.norm(u_reshaped, axis=1)
        label = 'Displacement magnitude'
    else:
        raise ValueError(f"Unknown component: {component}")

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    tcf = ax.tripcolor(tri, values, shading='gouraud', cmap=cmap, **kwargs)
    ax.set_aspect('equal')

    if colorbar:
        plt.colorbar(tcf, ax=ax, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


def plot_deformed_mesh(mesh: 'TriangleMesh',
                       u: np.ndarray,
                       scale: float = 1.0,
                       ax: Optional['plt.Axes'] = None,
                       show_original: bool = True,
                       **kwargs) -> 'plt.Axes':
    """
    Plot deformed mesh.

    Args:
        mesh: TriangleMesh instance
        u: displacement vector
        scale: displacement magnification factor
        ax: matplotlib axes
        show_original: whether to show original mesh
        **kwargs: passed to triplot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    u = np.asarray(u)
    if u.ndim == 1:
        u_reshaped = u.reshape(-1, 2)
    else:
        u_reshaped = u

    deformed_nodes = mesh.nodes + scale * u_reshaped

    if show_original:
        tri_orig = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
        ax.triplot(tri_orig, 'b--', lw=0.5, alpha=0.3, label='Original')

    tri_def = Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1], mesh.elements)
    ax.triplot(tri_def, 'k-', lw=0.5, label='Deformed', **kwargs)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


def plot_load_displacement(results: List['LoadStep'],
                           mesh: 'TriangleMesh',
                           node_idx: int,
                           direction: str = 'y',
                           ax: Optional['plt.Axes'] = None,
                           **kwargs) -> 'plt.Axes':
    """
    Plot load-displacement curve.

    Args:
        results: list of LoadStep instances
        mesh: TriangleMesh instance (for computing reactions)
        node_idx: node index for displacement measurement
        direction: 'x' or 'y'
        ax: matplotlib axes
        **kwargs: passed to plot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    displacements = []
    loads = []

    dof = 2 * node_idx + (0 if direction == 'x' else 1)

    for result in results:
        displacements.append(result.displacement[dof])
        loads.append(result.load_factor)

    ax.plot(displacements, loads, 'b-o', ms=3, **kwargs)
    ax.set_xlabel(f'Displacement {direction}')
    ax.set_ylabel('Load factor')
    ax.grid(True, alpha=0.3)

    return ax


def plot_force_displacement(results: List['LoadStep'],
                            mesh: 'TriangleMesh',
                            node_idx: int,
                            direction: str = 'y',
                            ax: Optional['plt.Axes'] = None,
                            **kwargs) -> 'plt.Axes':
    """
    Plot force-displacement curve.

    Estimates reaction force from energy derivative.

    Args:
        results: list of LoadStep instances
        mesh: TriangleMesh instance
        node_idx: node for displacement measurement
        direction: 'x' or 'y'
        ax: matplotlib axes
        **kwargs: passed to plot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    displacements = []
    forces = []

    dof = 2 * node_idx + (0 if direction == 'x' else 1)

    for i, result in enumerate(results):
        displacements.append(result.displacement[dof])

        # Estimate force from energy gradient
        if i == 0:
            forces.append(0.0)
        else:
            du = result.displacement[dof] - results[i-1].displacement[dof]
            dE = result.strain_energy - results[i-1].strain_energy
            if abs(du) > 1e-15:
                forces.append(dE / du)
            else:
                forces.append(forces[-1] if forces else 0.0)

    ax.plot(displacements, forces, 'b-o', ms=3, **kwargs)
    ax.set_xlabel(f'Displacement {direction}')
    ax.set_ylabel('Force (estimated)')
    ax.grid(True, alpha=0.3)

    return ax


def plot_energy_evolution(results: List['LoadStep'],
                          ax: Optional['plt.Axes'] = None,
                          normalize: bool = False,
                          **kwargs) -> 'plt.Axes':
    """
    Plot energy components vs load step.

    Args:
        results: list of LoadStep instances
        ax: matplotlib axes
        normalize: whether to normalize by maximum total energy
        **kwargs: passed to plot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    steps = [r.step for r in results]
    strain = np.array([r.strain_energy for r in results])
    surface = np.array([r.surface_energy for r in results])
    total = strain + surface

    if normalize and np.max(total) > 0:
        scale = np.max(total)
        strain /= scale
        surface /= scale
        total /= scale

    ax.plot(steps, strain, 'b-', label='Strain energy', **kwargs)
    ax.plot(steps, surface, 'r-', label='Surface energy', **kwargs)
    ax.plot(steps, total, 'k--', label='Total', **kwargs)

    ax.set_xlabel('Load step')
    ylabel = 'Energy (normalized)' if normalize else 'Energy'
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_damage_evolution(results: List['LoadStep'],
                          mesh: 'TriangleMesh',
                          ax: Optional['plt.Axes'] = None,
                          **kwargs) -> 'plt.Axes':
    """
    Plot damage statistics over load steps.

    Args:
        results: list of LoadStep instances
        mesh: TriangleMesh instance
        ax: matplotlib axes
        **kwargs: passed to plot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    steps = [r.step for r in results]
    max_damage = [np.max(r.damage) for r in results]
    mean_damage = [np.mean(r.damage) for r in results]

    # Count damaged edges (d > 0.5)
    n_damaged = [np.sum(r.damage > 0.5) for r in results]
    frac_damaged = [n / mesh.n_edges for n in n_damaged]

    ax.plot(steps, max_damage, 'r-', label='Max damage', **kwargs)
    ax.plot(steps, mean_damage, 'b-', label='Mean damage', **kwargs)
    ax.plot(steps, frac_damaged, 'g--', label='Fraction damaged (d>0.5)', **kwargs)

    ax.set_xlabel('Load step')
    ax.set_ylabel('Damage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_convergence(results: List['LoadStep'],
                     ax: Optional['plt.Axes'] = None,
                     **kwargs) -> 'plt.Axes':
    """
    Plot convergence metrics over load steps.

    Args:
        results: list of LoadStep instances
        ax: matplotlib axes
        **kwargs: passed to plot

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    steps = [r.step for r in results]
    n_iters = [r.n_iterations for r in results]
    res_u = [r.residual_u for r in results]
    res_d = [r.residual_d for r in results]

    ax2 = ax.twinx()

    l1, = ax.plot(steps, n_iters, 'b-o', ms=3, label='Iterations')
    l2, = ax2.semilogy(steps, res_u, 'r-', label='Residual u', alpha=0.7)
    l3, = ax2.semilogy(steps, res_d, 'g-', label='Residual d', alpha=0.7)

    ax.set_xlabel('Load step')
    ax.set_ylabel('Number of iterations', color='b')
    ax2.set_ylabel('Residual', color='r')

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    ax.grid(True, alpha=0.3)

    return ax


def create_animation_frames(results: List['LoadStep'],
                            mesh: 'TriangleMesh',
                            output_dir: str,
                            field: str = 'damage',
                            prefix: str = 'frame') -> List[str]:
    """
    Create animation frames from simulation results.

    Args:
        results: list of LoadStep instances
        mesh: TriangleMesh instance
        output_dir: directory for output files
        field: 'damage' or 'displacement'
        prefix: filename prefix

    Returns:
        List of output filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    filenames = []
    for i, result in enumerate(results):
        fig, ax = plt.subplots(figsize=(10, 8))

        if field == 'damage':
            plot_damage_field(mesh, result.damage, ax=ax)
            ax.set_title(f'Step {result.step}, λ={result.load_factor:.4f}')
        elif field == 'displacement':
            plot_displacement_field(mesh, result.displacement, ax=ax)
            ax.set_title(f'Step {result.step}, λ={result.load_factor:.4f}')

        filename = os.path.join(output_dir, f'{prefix}_{i:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        filenames.append(filename)

    return filenames


def plot_stress_field(mesh: 'TriangleMesh',
                      stresses: np.ndarray,
                      component: str = 'von_mises',
                      ax: Optional['plt.Axes'] = None,
                      cmap: str = 'jet',
                      colorbar: bool = True,
                      **kwargs) -> 'plt.Axes':
    """
    Plot stress field on elements.

    Args:
        mesh: TriangleMesh instance
        stresses: shape (n_elements, 3) or (n_elements,) for scalar
        component: 'xx', 'yy', 'xy', or 'von_mises'
        ax: matplotlib axes
        cmap: colormap name
        colorbar: whether to show colorbar
        **kwargs: passed to tripcolor

    Returns:
        ax: matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if stresses.ndim == 2:
        if component == 'xx':
            values = stresses[:, 0]
            label = 'σ_xx'
        elif component == 'yy':
            values = stresses[:, 1]
            label = 'σ_yy'
        elif component == 'xy':
            values = stresses[:, 2]
            label = 'τ_xy'
        elif component == 'von_mises':
            sig_xx, sig_yy, tau_xy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
            values = np.sqrt(sig_xx**2 + sig_yy**2 - sig_xx*sig_yy + 3*tau_xy**2)
            label = 'σ_vm'
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        values = stresses
        label = 'Stress'

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    tcf = ax.tripcolor(tri, values, shading='flat', cmap=cmap, **kwargs)
    ax.set_aspect('equal')

    if colorbar:
        plt.colorbar(tcf, ax=ax, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax
