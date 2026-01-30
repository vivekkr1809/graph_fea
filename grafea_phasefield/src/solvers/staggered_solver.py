"""
Staggered Solver
================

Alternating minimization solver for coupled displacement-damage problem.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import List, Callable, Optional, TYPE_CHECKING
import sys
import os

# Handle imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if TYPE_CHECKING:
    from mesh.triangle_mesh import TriangleMesh
    from mesh.edge_graph import EdgeGraph
    from elements.grafea_element import GraFEAElement
    from physics.material import IsotropicMaterial


@dataclass
class SolverConfig:
    """Configuration for staggered solver."""
    tol_u: float = 1e-6          # Displacement convergence tolerance
    tol_d: float = 1e-6          # Damage convergence tolerance
    max_stagger_iter: int = 100  # Maximum staggered iterations per step
    min_stagger_iter: int = 2    # Minimum iterations (ensure coupling)
    verbose: bool = True         # Print convergence info
    damage_bounds: tuple = (0.0, 1.0)  # Min and max damage values


@dataclass
class LoadStep:
    """Result for a single load step."""
    step: int
    load_factor: float
    displacement: np.ndarray
    damage: np.ndarray
    strain_energy: float
    surface_energy: float
    external_work: float
    converged: bool
    n_iterations: int
    residual_u: float = 0.0
    residual_d: float = 0.0


class StaggeredSolver:
    """
    Staggered solution algorithm for GraFEA phase-field.

    The coupled problem is solved by alternating between:
    1. Mechanical equilibrium (fixed damage)
    2. Damage evolution (fixed displacement)

    Algorithm:
        For each load step:
            While not converged:
                1. Solve mechanical equilibrium with current damage
                2. Update history field from strains
                3. Solve damage evolution
                4. Check convergence

    Attributes:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        material: IsotropicMaterial instance
        edge_graph: EdgeGraph for damage regularization
        config: SolverConfig instance
        damage: current damage field
        displacement: current displacement field
        history: HistoryField instance
        results: list of LoadStep results
    """

    def __init__(self, mesh: 'TriangleMesh',
                 elements: List['GraFEAElement'],
                 material: 'IsotropicMaterial',
                 edge_graph: 'EdgeGraph',
                 config: Optional[SolverConfig] = None):
        """
        Initialize staggered solver.

        Args:
            mesh: TriangleMesh instance
            elements: list of GraFEAElement instances
            material: IsotropicMaterial instance
            edge_graph: EdgeGraph for damage regularization
            config: SolverConfig (optional)
        """
        self.mesh = mesh
        self.elements = elements
        self.material = material
        self.edge_graph = edge_graph
        self.config = config or SolverConfig()

        # State variables
        self.n_dof = 2 * mesh.n_nodes
        self.n_edges = mesh.n_edges

        # Initialize fields
        self.damage = np.zeros(self.n_edges)
        self.displacement = np.zeros(self.n_dof)

        # History field for damage irreversibility
        from physics.damage import HistoryField
        self.history = HistoryField(self.n_edges)

        # Results storage
        self.results: List[LoadStep] = []

    def solve(self, load_factors: np.ndarray,
              bc_dofs: np.ndarray,
              bc_values_func: Callable[[float], np.ndarray],
              external_force_func: Optional[Callable[[float], np.ndarray]] = None
              ) -> List[LoadStep]:
        """
        Solve the coupled problem over load steps.

        Args:
            load_factors: array of load multipliers
            bc_dofs: DOF indices with Dirichlet BCs
            bc_values_func: function(load_factor) -> bc_values array
            external_force_func: function(load_factor) -> force vector (optional)

        Returns:
            List of LoadStep results
        """
        self.results = []

        for step, load in enumerate(load_factors):
            if self.config.verbose:
                print(f"\n=== Load Step {step + 1}/{len(load_factors)}: λ = {load:.6f} ===")

            # Get boundary conditions for this step
            bc_values = bc_values_func(load)

            # External force
            if external_force_func is not None:
                F_ext = external_force_func(load)
            else:
                F_ext = np.zeros(self.n_dof)

            # Staggered iteration
            result = self._staggered_iteration(step, load, bc_dofs, bc_values, F_ext)
            self.results.append(result)

            if not result.converged:
                if self.config.verbose:
                    print(f"WARNING: Step {step + 1} did not converge!")

        return self.results

    def _staggered_iteration(self, step: int, load_factor: float,
                              bc_dofs: np.ndarray, bc_values: np.ndarray,
                              F_ext: np.ndarray) -> LoadStep:
        """
        Perform staggered iterations for a single load step.

        Args:
            step: step index
            load_factor: current load factor
            bc_dofs: constrained DOF indices
            bc_values: prescribed values
            F_ext: external force vector

        Returns:
            LoadStep result
        """
        u_old = self.displacement.copy()
        d_old = self.damage.copy()

        converged = False
        residual_u = 0.0
        residual_d = 0.0

        for iteration in range(self.config.max_stagger_iter):
            # Step 1: Mechanical solve
            u_new = self._solve_mechanics(bc_dofs, bc_values, F_ext)

            # Step 2: Compute edge strains and update history
            edge_strains = self._compute_edge_strains(u_new)
            driving_force = self._compute_driving_force(edge_strains)
            self.history.update(driving_force)

            # Step 3: Damage solve
            d_new = self._solve_damage()

            # Step 4: Check convergence
            u_norm = np.linalg.norm(u_new)
            d_norm = np.linalg.norm(d_new)

            if u_norm > 1e-10:
                residual_u = np.linalg.norm(u_new - u_old) / u_norm
            else:
                residual_u = np.linalg.norm(u_new - u_old)

            if d_norm > 1e-10:
                residual_d = np.linalg.norm(d_new - d_old) / d_norm
            else:
                residual_d = np.linalg.norm(d_new - d_old)

            if self.config.verbose:
                max_d = np.max(d_new)
                print(f"  Iter {iteration + 1}: Δu = {residual_u:.2e}, "
                      f"Δd = {residual_d:.2e}, max(d) = {max_d:.4f}")

            # Check convergence (after minimum iterations)
            if iteration >= self.config.min_stagger_iter - 1:
                if residual_u < self.config.tol_u and residual_d < self.config.tol_d:
                    converged = True
                    if self.config.verbose:
                        print(f"  Converged in {iteration + 1} iterations")
                    break

            # Update for next iteration
            u_old = u_new.copy()
            d_old = d_new.copy()
            self.displacement = u_new
            self.damage = d_new

        # Final update
        self.displacement = u_new
        self.damage = d_new

        # Compute energies
        edge_strains = self._compute_edge_strains(self.displacement)
        strain_energy = self._compute_strain_energy(edge_strains, self.damage)

        from physics.surface_energy import compute_surface_energy
        surface_energy = compute_surface_energy(
            self.mesh, self.edge_graph, self.damage,
            self.material.Gc, self.material.l0
        )

        external_work = np.dot(F_ext, self.displacement)

        return LoadStep(
            step=step,
            load_factor=load_factor,
            displacement=self.displacement.copy(),
            damage=self.damage.copy(),
            strain_energy=strain_energy,
            surface_energy=surface_energy,
            external_work=external_work,
            converged=converged,
            n_iterations=iteration + 1,
            residual_u=residual_u,
            residual_d=residual_d
        )

    def _solve_mechanics(self, bc_dofs: np.ndarray,
                         bc_values: np.ndarray,
                         F_ext: np.ndarray) -> np.ndarray:
        """
        Solve mechanical equilibrium with current damage.

        K(d) u = F_ext

        Args:
            bc_dofs: constrained DOF indices
            bc_values: prescribed values
            F_ext: external force vector

        Returns:
            u: displacement solution
        """
        from assembly.global_assembly import assemble_global_stiffness
        from assembly.boundary_conditions import apply_dirichlet_bc

        K = assemble_global_stiffness(self.mesh, self.elements, self.damage)
        K_bc, F_bc = apply_dirichlet_bc(K, F_ext, bc_dofs, bc_values)

        return spsolve(K_bc, F_bc)

    def _compute_edge_strains(self, u: np.ndarray) -> np.ndarray:
        """
        Compute edge strains for all elements.

        Args:
            u: displacement vector

        Returns:
            edge_strains: shape (n_elements, 3)
        """
        from assembly.global_assembly import compute_all_edge_strains
        return compute_all_edge_strains(self.mesh, self.elements, u)

    def _compute_driving_force(self, edge_strains: np.ndarray) -> np.ndarray:
        """
        Compute damage driving force from edge strains.

        Args:
            edge_strains: shape (n_elements, 3)

        Returns:
            driving_force: shape (n_edges,)
        """
        from physics.damage import compute_driving_force
        return compute_driving_force(
            self.mesh, self.elements, edge_strains, self.material
        )

    def _solve_damage(self) -> np.ndarray:
        """
        Solve damage evolution equation.

        Returns:
            d: new damage field
        """
        from physics.surface_energy import assemble_damage_system

        A_d, b_d = assemble_damage_system(
            self.mesh, self.edge_graph, self.history.H,
            self.material.Gc, self.material.l0
        )
        d = spsolve(A_d, b_d)

        # Enforce bounds
        d = np.clip(d, self.config.damage_bounds[0], self.config.damage_bounds[1])

        # Enforce irreversibility
        d = np.maximum(d, self.damage)

        return d

    def _compute_strain_energy(self, edge_strains: np.ndarray,
                                damage: np.ndarray) -> float:
        """
        Compute total strain energy.

        Args:
            edge_strains: shape (n_elements, 3)
            damage: edge damage values

        Returns:
            E_strain: total strain energy
        """
        from physics.tension_split import spectral_split

        total = 0.0
        for e_idx, elem in enumerate(self.elements):
            d_local = damage[self.mesh.element_to_edges[e_idx]]
            eps_edge = edge_strains[e_idx]

            # Get split
            eps_tensor = elem.T_inv @ eps_edge
            split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

            psi = elem.strain_energy_density_degraded(eps_edge, d_local, split)
            total += psi * elem.area * elem.thickness

        return total

    def reset(self) -> None:
        """Reset solver state to initial conditions."""
        self.damage = np.zeros(self.n_edges)
        self.displacement = np.zeros(self.n_dof)
        self.history.reset()
        self.results = []

    def set_initial_damage(self, damage: np.ndarray) -> None:
        """
        Set initial damage (e.g., for pre-existing crack).

        Args:
            damage: initial damage field, shape (n_edges,)
        """
        if len(damage) != self.n_edges:
            raise ValueError(f"damage has wrong size: {len(damage)} != {self.n_edges}")
        self.damage = damage.copy()
        # Also update history to prevent damage from healing
        self.history.set_values(self.material.critical_strain_energy_density() * damage)

    def set_initial_crack(self, edge_indices: np.ndarray,
                          damage_value: float = 0.99) -> None:
        """
        Set initial crack by damaging specific edges.

        Args:
            edge_indices: indices of edges forming the crack
            damage_value: damage value for crack edges
        """
        self.damage[edge_indices] = damage_value
        # Set history to prevent healing
        psi_c = self.material.critical_strain_energy_density()
        self.history.H[edge_indices] = psi_c * damage_value

    def get_results_summary(self) -> dict:
        """
        Get summary statistics from results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}

        return {
            'n_steps': len(self.results),
            'converged_steps': sum(r.converged for r in self.results),
            'max_damage': max(np.max(r.damage) for r in self.results),
            'final_strain_energy': self.results[-1].strain_energy,
            'final_surface_energy': self.results[-1].surface_energy,
            'total_iterations': sum(r.n_iterations for r in self.results),
        }


def create_load_stepping(max_load: float, n_steps: int,
                         stepping: str = 'linear') -> np.ndarray:
    """
    Create load factor array for simulation.

    Args:
        max_load: maximum load factor
        n_steps: number of load steps
        stepping: 'linear', 'quadratic', or 'adaptive'

    Returns:
        load_factors: array of load factors
    """
    if stepping == 'linear':
        return np.linspace(0, max_load, n_steps)
    elif stepping == 'quadratic':
        # Denser at beginning, sparser at end
        t = np.linspace(0, 1, n_steps)
        return max_load * t ** 2
    elif stepping == 'sqrt':
        # Sparser at beginning, denser at end
        t = np.linspace(0, 1, n_steps)
        return max_load * np.sqrt(t)
    else:
        raise ValueError(f"Unknown stepping: {stepping}")


def run_tension_test(mesh: 'TriangleMesh',
                     material: 'IsotropicMaterial',
                     max_displacement: float,
                     n_steps: int = 50,
                     verbose: bool = True) -> List[LoadStep]:
    """
    Convenience function to run a simple uniaxial tension test.

    Boundary conditions:
    - Bottom: fixed in y
    - One node fixed in x (prevent rigid body motion)
    - Top: prescribed vertical displacement

    Args:
        mesh: TriangleMesh instance
        material: IsotropicMaterial instance
        max_displacement: maximum applied displacement
        n_steps: number of load steps
        verbose: print progress

    Returns:
        List of LoadStep results
    """
    from mesh.edge_graph import EdgeGraph
    from elements.grafea_element import GraFEAElement
    from assembly.boundary_conditions import create_bc_from_region, merge_bcs

    # Create elements
    elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                for e in range(mesh.n_elements)]

    # Create edge graph
    edge_graph = EdgeGraph(mesh)

    # Get domain bounds
    x_min, y_min = mesh.nodes.min(axis=0)
    x_max, y_max = mesh.nodes.max(axis=0)
    tol = 1e-6 * max(y_max - y_min, x_max - x_min)

    # Bottom: fixed in y
    bc_bottom_dofs, bc_bottom_vals = create_bc_from_region(
        mesh, lambda x, y: y < y_min + tol, 'y', 0.0
    )

    # Fix one node in x to prevent rigid body motion
    bottom_nodes = mesh.get_nodes_in_region(lambda x, y: y < y_min + tol)
    bc_fix_x = (np.array([2 * bottom_nodes[0]]), np.array([0.0]))

    # Top: will have prescribed displacement
    top_nodes = mesh.get_nodes_in_region(lambda x, y: y > y_max - tol)
    bc_top_dofs = np.array([2 * n + 1 for n in top_nodes])

    # Merge fixed BCs
    fixed_dofs, fixed_vals = merge_bcs(
        (bc_bottom_dofs, bc_bottom_vals),
        bc_fix_x
    )

    # All BC DOFs
    all_bc_dofs = np.concatenate([fixed_dofs, bc_top_dofs])

    def bc_values_func(load_factor):
        values = np.zeros(len(all_bc_dofs))
        values[:len(fixed_vals)] = fixed_vals
        values[len(fixed_vals):] = load_factor  # Top displacement
        return values

    # Create solver
    config = SolverConfig(verbose=verbose)
    solver = StaggeredSolver(mesh, elements, material, edge_graph, config)

    # Load stepping
    load_factors = create_load_stepping(max_displacement, n_steps, 'linear')

    # Run simulation
    return solver.solve(load_factors, all_bc_dofs, bc_values_func)
