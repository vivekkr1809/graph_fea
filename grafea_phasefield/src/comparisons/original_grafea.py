"""
Original GraFEA Solver (Binary Edge Damage)
============================================

Implements the original GraFEA framework with threshold-based binary edge damage.
Unlike the GraFEA phase-field solver, there is NO regularization (no graph Laplacian,
no length scale parameter in the damage evolution). Each edge is either fully intact
(phi=1) or fully broken (phi=0), determined by comparing the edge strain against a
critical value derived from the fracture energy criterion.

This serves as the baseline comparison to demonstrate the improvements gained by
adding phase-field regularization to the edge-based GraFEA framework.

Key differences from GraFEA-PF:
    - Binary damage: d in {0, 1} instead of d in [0, 1]
    - No regularization: damage is purely local (no graph Laplacian diffusion)
    - Threshold criterion: edge breaks when tensile strain exceeds eps_crit
    - Mesh-dependent crack path: no length scale to regularize the damage band
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from typing import List, Callable, Optional, Dict, Tuple
import sys
import os

# Handle imports (same pattern as the rest of the codebase)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.triangle_mesh import TriangleMesh
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from assembly.global_assembly import (
    assemble_global_stiffness,
    assemble_internal_force,
    compute_all_edge_strains,
)
from assembly.boundary_conditions import apply_dirichlet_bc
from solvers.staggered_solver import SolverConfig, LoadStep


class OriginalGraFEASolver:
    """
    Original GraFEA with binary (threshold-based) edge damage.
    No phase-field regularization -- damage follows mesh edges.

    In this formulation, each edge k has a binary damage state:
        d_k in {0, 1}
    where d_k = 0 means intact and d_k = 1 means broken.

    The damage criterion is based on the tensile edge strain:
        if max_tensile_strain_k > eps_crit_k  =>  d_k = 1

    The critical strain for edge k is derived from an energy balance:
        (1/2) * A_kk * eps^2 * omega_k = Gc * L_k * t
    leading to:
        eps_crit_k = sqrt(2 * Gc * L_k / (A_kk * omega_k))

    where:
        A_kk   = average diagonal stiffness entry for edge k
        omega_k = edge volume = sum of (A_e * t / 3) for elements sharing edge k
        L_k    = edge length
        Gc     = critical energy release rate
        t      = thickness

    Once an edge breaks, it stays broken (irreversibility).

    Attributes:
        mesh: TriangleMesh instance
        elements: list of GraFEAElement instances
        material: IsotropicMaterial instance
        config: SolverConfig instance
        damage: edge damage field, shape (n_edges,), values in {0, 1}
        displacement: current displacement vector, shape (2*n_nodes,)
        eps_crit: critical strain for each edge, shape (n_edges,)
        results: list of LoadStep results
    """

    def __init__(self, mesh: TriangleMesh, material: IsotropicMaterial,
                 config: Optional[SolverConfig] = None):
        """
        Initialize original GraFEA solver.

        Args:
            mesh: TriangleMesh instance
            material: IsotropicMaterial instance
            config: SolverConfig (optional, uses defaults if None)
        """
        self.mesh = mesh
        self.material = material
        self.config = config or SolverConfig()

        # State dimensions
        self.n_dof = 2 * mesh.n_nodes
        self.n_edges = mesh.n_edges

        # Create GraFEA elements (same element type as the phase-field solver)
        self.elements = [
            GraFEAElement(mesh.nodes[mesh.elements[e]], material)
            for e in range(mesh.n_elements)
        ]

        # Binary damage: 1.0 = broken, 0.0 = intact
        # Convention matches GraFEA-PF: d=1 means fully damaged
        self.damage = np.zeros(self.n_edges)

        # Displacement field
        self.displacement = np.zeros(self.n_dof)

        # Compute critical strains for each edge
        self.eps_crit = self._compute_critical_strains()

        # Results storage
        self.results: List[LoadStep] = []

    def _compute_critical_strains(self) -> np.ndarray:
        """
        Compute critical strain for each edge based on the energy criterion.

        For edge k, we equate the edge strain energy to the fracture energy:
            (1/2) * A_kk_avg * eps^2 * omega_k = Gc * L_k * t

        Solving for eps_crit:
            eps_crit_k = sqrt(2 * Gc * L_k * t / (A_kk_avg * omega_k))

        where:
            omega_k = sum over elements containing edge k of (A_e * t / 3)
            A_kk_avg = average of A[local_k, local_k] over elements sharing edge k
            L_k = edge length
            t = thickness

        Returns:
            eps_crit: critical strain for each edge, shape (n_edges,)
        """
        n_edges = self.n_edges
        Gc = self.material.Gc

        # Compute edge volumes: omega_k = sum_{e containing k} A_e * t / 3
        omega = np.zeros(n_edges)
        # Compute average diagonal stiffness A_kk for each edge
        A_kk_sum = np.zeros(n_edges)
        A_kk_count = np.zeros(n_edges)

        for e_idx, elem in enumerate(self.elements):
            edge_indices = self.mesh.element_to_edges[e_idx]
            elem_volume_per_edge = elem.area * elem.thickness / 3.0

            for local_k in range(3):
                global_k = edge_indices[local_k]
                omega[global_k] += elem_volume_per_edge
                A_kk_sum[global_k] += elem.A[local_k, local_k]
                A_kk_count[global_k] += 1

        # Average A_kk over elements sharing each edge
        A_kk_avg = np.ones(n_edges)
        nonzero = A_kk_count > 0
        A_kk_avg[nonzero] = A_kk_sum[nonzero] / A_kk_count[nonzero]

        # Edge lengths
        L = self.mesh.edge_lengths

        # Critical strains
        # eps_crit = sqrt(2 * Gc / (A_kk_avg * omega / L))
        # Equivalently: eps_crit = sqrt(2 * Gc * L / (A_kk_avg * omega))
        eps_crit = np.zeros(n_edges)
        valid = (A_kk_avg > 0) & (omega > 0)
        eps_crit[valid] = np.sqrt(
            2.0 * Gc * L[valid] / (A_kk_avg[valid] * omega[valid])
        )

        # For edges with no volume contribution (should not happen in a valid mesh),
        # set a very large critical strain so they never break
        eps_crit[~valid] = 1e20

        return eps_crit

    def _update_damage(self, u: np.ndarray) -> int:
        """
        Update binary damage based on edge strain criterion.

        For each edge, compute the maximum tensile strain across all elements
        sharing it. If this exceeds the critical strain and the edge is still
        intact, break it (set d=1).

        Only tensile (positive) edge strains contribute to the damage criterion.
        Damage is irreversible: once broken, always broken.

        Args:
            u: displacement vector, shape (2*n_nodes,)

        Returns:
            n_newly_broken: number of edges that broke in this update
        """
        # Compute edge strains for all elements
        edge_strains = compute_all_edge_strains(self.mesh, self.elements, u)

        # For each global edge, find the maximum tensile strain across elements
        max_tensile_strain = np.zeros(self.n_edges)

        for e_idx in range(self.mesh.n_elements):
            edge_indices = self.mesh.element_to_edges[e_idx]
            for local_k in range(3):
                global_k = edge_indices[local_k]
                strain_k = edge_strains[e_idx, local_k]
                # Only consider tensile (positive) strains
                if strain_k > max_tensile_strain[global_k]:
                    max_tensile_strain[global_k] = strain_k

        # Identify edges that should break:
        # - Currently intact (d == 0)
        # - Tensile strain exceeds critical value
        intact_mask = self.damage < 0.5  # d=0 means intact
        exceeds_crit = max_tensile_strain > self.eps_crit
        newly_broken = intact_mask & exceeds_crit

        n_newly_broken = np.sum(newly_broken)

        # Break the edges (irreversible)
        self.damage[newly_broken] = 1.0

        return int(n_newly_broken)

    def _compute_strain_energy(self, u: np.ndarray, damage: np.ndarray) -> float:
        """
        Compute total strain energy with current damage.

        Uses the same degraded energy formulation as GraFEA-PF for consistency.

        Args:
            u: displacement vector, shape (2*n_nodes,)
            damage: edge damage field, shape (n_edges,)

        Returns:
            E_strain: total strain energy
        """
        from physics.tension_split import spectral_split

        u = np.asarray(u).flatten()
        total_energy = 0.0

        for e_idx, elem in enumerate(self.elements):
            node_indices = self.mesh.elements[e_idx]
            dof_indices = []
            for n in node_indices:
                dof_indices.extend([2 * n, 2 * n + 1])

            u_e = u[dof_indices]
            eps_edge = elem.compute_edge_strains(u_e)
            eps_tensor = elem.T_inv @ eps_edge

            # Get local damage
            edge_indices = self.mesh.element_to_edges[e_idx]
            d_local = damage[edge_indices]

            # Get tension-compression split
            split = spectral_split(eps_tensor, eps_edge, elem.C, elem.T)

            # Degraded energy
            psi = elem.strain_energy_density_degraded(eps_edge, d_local, split)
            total_energy += psi * elem.area * elem.thickness

        return total_energy

    def solve(self, load_factors: np.ndarray,
              bc_dofs: np.ndarray,
              bc_values_func: Callable[[float], np.ndarray],
              external_force_func: Optional[Callable[[float], np.ndarray]] = None
              ) -> List[LoadStep]:
        """
        Solve the quasi-static fracture problem over load steps.

        For each load step, the algorithm iterates between:
            1. Assemble stiffness with current damage
            2. Apply boundary conditions and solve for displacement
            3. Compute edge strains and update binary damage
            4. If any new edges broke, re-solve with updated damage
            5. Repeat until no new edges break or max iterations reached

        This is the original GraFEA approach: no phase-field equation is solved.
        Instead, edges are simply removed (broken) when their strain exceeds
        the critical value.

        Args:
            load_factors: array of load multipliers for each step
            bc_dofs: DOF indices with Dirichlet boundary conditions
            bc_values_func: function(load_factor) -> bc_values array
            external_force_func: function(load_factor) -> force vector (optional)

        Returns:
            List of LoadStep results (same format as StaggeredSolver)
        """
        self.results = []

        for step, load in enumerate(load_factors):
            if self.config.verbose:
                print(f"\n=== Load Step {step + 1}/{len(load_factors)}: "
                      f"lambda = {load:.6f} ===")

            # Get boundary conditions for this step
            bc_values = bc_values_func(load)

            # External force
            if external_force_func is not None:
                F_ext = external_force_func(load)
            else:
                F_ext = np.zeros(self.n_dof)

            # Iterative solve with damage update
            result = self._solve_step(step, load, bc_dofs, bc_values, F_ext)
            self.results.append(result)

            if not result.converged:
                if self.config.verbose:
                    print(f"WARNING: Step {step + 1} did not converge "
                          f"(new edges still breaking after {result.n_iterations} iters)")

        return self.results

    def _solve_step(self, step: int, load_factor: float,
                    bc_dofs: np.ndarray, bc_values: np.ndarray,
                    F_ext: np.ndarray) -> LoadStep:
        """
        Solve a single load step with iterative damage update.

        The iteration continues until no new edges break or the maximum
        number of staggered iterations is reached.

        Args:
            step: step index
            load_factor: current load factor
            bc_dofs: constrained DOF indices
            bc_values: prescribed displacement values
            F_ext: external force vector

        Returns:
            LoadStep result
        """
        converged = False
        n_iterations = 0
        total_newly_broken = 0

        for iteration in range(self.config.max_stagger_iter):
            n_iterations = iteration + 1

            # Step 1: Assemble stiffness with current damage
            K = assemble_global_stiffness(self.mesh, self.elements, self.damage)

            # Step 2: Apply BCs and solve for displacement
            K_bc, F_bc = apply_dirichlet_bc(K, F_ext, bc_dofs, bc_values)
            u_new = spsolve(K_bc, F_bc)

            # Step 3: Update binary damage
            n_broken = self._update_damage(u_new)
            total_newly_broken += n_broken

            if self.config.verbose:
                n_total_broken = int(np.sum(self.damage > 0.5))
                print(f"  Iter {iteration + 1}: {n_broken} new edges broken, "
                      f"{n_total_broken}/{self.n_edges} total broken")

            # Step 4: Check convergence -- no new edges broke
            if n_broken == 0:
                converged = True
                self.displacement = u_new
                if self.config.verbose:
                    print(f"  Converged in {n_iterations} iterations "
                          f"(no new edges broken)")
                break

            # Update displacement for next iteration
            self.displacement = u_new

        # If we exited the loop without converging, use the last displacement
        if not converged:
            self.displacement = u_new

        # Compute energies
        strain_energy = self._compute_strain_energy(self.displacement, self.damage)

        # No phase-field regularization => surface energy is zero
        surface_energy = 0.0

        # External work
        external_work = float(np.dot(F_ext, self.displacement))

        # Residual: fraction of edges that broke in this step
        residual_d = total_newly_broken / max(self.n_edges, 1)

        return LoadStep(
            step=step,
            load_factor=load_factor,
            displacement=self.displacement.copy(),
            damage=self.damage.copy(),
            strain_energy=strain_energy,
            surface_energy=surface_energy,
            external_work=external_work,
            converged=converged,
            n_iterations=n_iterations,
            residual_u=0.0,
            residual_d=residual_d
        )

    def set_initial_damage(self, damage: np.ndarray) -> None:
        """
        Set initial damage field (e.g., for pre-existing crack).

        The provided damage array is binarized: values >= 0.5 are set to 1 (broken),
        values < 0.5 are set to 0 (intact), to maintain the binary damage convention.

        Args:
            damage: initial damage field, shape (n_edges,)
        """
        if len(damage) != self.n_edges:
            raise ValueError(
                f"damage has wrong size: {len(damage)} != {self.n_edges}"
            )
        # Binarize: enforce {0, 1} values
        self.damage = np.where(damage >= 0.5, 1.0, 0.0)

    def set_initial_crack(self, edge_indices: np.ndarray,
                          damage_value: float = 0.99) -> None:
        """
        Set pre-crack by damaging specific edges.

        Since this solver uses binary damage, any damage_value >= 0.5
        results in the edge being fully broken (d=1).

        Args:
            edge_indices: indices of edges forming the initial crack
            damage_value: damage value (>= 0.5 means broken)
        """
        edge_indices = np.asarray(edge_indices, dtype=np.int64)
        if damage_value >= 0.5:
            self.damage[edge_indices] = 1.0
        else:
            self.damage[edge_indices] = 0.0

    def compute_reaction_force(self, u: np.ndarray, damage: np.ndarray,
                               L: float, direction: str = 'y') -> float:
        """
        Compute reaction force at the top boundary.

        The reaction force is obtained from the internal force vector
        evaluated at the constrained DOFs on the top boundary.

        Args:
            u: displacement vector, shape (2*n_nodes,)
            damage: damage field, shape (n_edges,)
            L: domain height (used to identify top boundary)
            direction: 'x' or 'y' for the force component

        Returns:
            F_total: total reaction force at the top boundary
        """
        tol = 1e-10 * L
        top_nodes = self.mesh.get_nodes_in_region(lambda x, y: y > L - tol)

        # Get internal force vector
        F_int = assemble_internal_force(self.mesh, self.elements, u, damage)

        # Sum reaction force at top nodes
        dof_offset = 0 if direction == 'x' else 1
        F_total = 0.0
        for node in top_nodes:
            dof = 2 * node + dof_offset
            F_total += F_int[dof]

        return F_total

    def extract_crack_path(self, threshold: float = 0.5) -> np.ndarray:
        """
        Extract crack path from the binary damage field.

        For binary damage, any threshold < 1 will capture all broken edges.
        The crack path is returned as midpoints of broken edges, sorted by
        x-coordinate.

        Args:
            threshold: damage threshold for crack identification
                       (default 0.5 captures all broken edges since d in {0,1})

        Returns:
            crack_path: array of (x, y) coordinates along the crack,
                        shape (n_points, 2), sorted by x
        """
        # Find broken edges
        cracked_edges = np.where(self.damage > threshold)[0]

        if len(cracked_edges) == 0:
            return np.array([]).reshape(0, 2)

        # Get midpoints of cracked edges
        midpoints = []
        for edge_idx in cracked_edges:
            n1, n2 = self.mesh.edges[edge_idx]
            mid = 0.5 * (self.mesh.nodes[n1] + self.mesh.nodes[n2])
            midpoints.append(mid)

        midpoints = np.array(midpoints)

        # Sort by x-coordinate
        sort_idx = np.argsort(midpoints[:, 0])
        return midpoints[sort_idx]

    def reset(self) -> None:
        """Reset solver state to initial conditions."""
        self.damage = np.zeros(self.n_edges)
        self.displacement = np.zeros(self.n_dof)
        self.eps_crit = self._compute_critical_strains()
        self.results = []

    def get_results_summary(self) -> dict:
        """
        Get summary statistics from results.

        Returns:
            Dictionary with summary statistics, compatible with
            the comparison framework.
        """
        if not self.results:
            return {}

        return {
            'n_steps': len(self.results),
            'converged_steps': sum(r.converged for r in self.results),
            'max_damage': max(np.max(r.damage) for r in self.results),
            'n_broken_edges': int(np.sum(self.damage > 0.5)),
            'n_total_edges': self.n_edges,
            'fraction_broken': float(np.sum(self.damage > 0.5)) / self.n_edges,
            'final_strain_energy': self.results[-1].strain_energy,
            'final_surface_energy': self.results[-1].surface_energy,
            'total_iterations': sum(r.n_iterations for r in self.results),
        }
