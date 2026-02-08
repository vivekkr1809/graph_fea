"""
FEM Phase-Field Reference Solver (Node-Based Damage)
=====================================================

Standard FEM phase-field solver using CST elements with node-based damage
for comparison against GraFEA edge-based formulation. Uses the AT2 model
with spectral tension-compression split (Miehe et al. 2010).

This solver does NOT depend on FEniCS -- it uses only numpy, scipy.sparse,
and the existing codebase infrastructure.

Key differences from the GraFEA solver
---------------------------------------
- Damage is a scalar field defined at **nodes** (n_nodes DOFs) rather than
  on edges (n_edges DOFs).
- Regularization uses the standard FEM gradient operator instead of the
  graph Laplacian.
- The driving history variable H is element-level (one value per element)
  rather than edge-level.
- Stiffness degradation uses the element-average nodal damage.

References
----------
Miehe, C., Welschinger, F., & Hofacker, M. (2010). Thermodynamically
consistent phase-field models of fracture: Variational principles and
multi-field FE implementations.  IJNME, 83(10), 1273-1311.

Bourdin, B., Francfort, G. A., & Marigo, J.-J. (2000). Numerical
experiments in revisited brittle fracture.  JMPS, 48(4), 797-826.
"""

import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Tuple

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.material import IsotropicMaterial
from physics.tension_split import spectral_split_2d, compute_split_energy_miehe
from solvers.staggered_solver import SolverConfig, LoadStep
from mesh.triangle_mesh import TriangleMesh
from assembly.boundary_conditions import apply_dirichlet_bc


# ---------------------------------------------------------------------------
# Small residual stiffness to avoid singular matrices when damage -> 1.
# ---------------------------------------------------------------------------
_ETA = 1.0e-8


class FEMPhasefieldSolver:
    """Standard FEM phase-field solver with node-based damage (AT2 model).

    The coupled displacement-damage problem is solved via staggered
    (alternating minimisation) iterations at each load step:

        1. Fix damage, solve for displacement (mechanical equilibrium).
        2. Compute element strains, update the history field.
        3. Fix displacement, solve for damage (phase-field evolution).
        4. Enforce damage bounds [0, 1] and irreversibility d >= d_old.
        5. Check convergence.

    Parameters
    ----------
    mesh : TriangleMesh
        Triangulated domain (from ``mesh.triangle_mesh``).
    material : IsotropicMaterial
        Elastic + phase-field material parameters (from ``physics.material``).
    config : SolverConfig, optional
        Iteration control.  Defaults are taken from ``SolverConfig``.
    plane : str, optional
        ``'strain'`` (default) or ``'stress'``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        mesh: TriangleMesh,
        material: IsotropicMaterial,
        config: Optional[SolverConfig] = None,
        plane: str = "strain",
    ):
        self.mesh = mesh
        self.material = material
        self.config = config or SolverConfig()
        self.plane = plane

        self.n_nodes: int = mesh.n_nodes
        self.n_elements: int = mesh.n_elements
        self.n_dof: int = 2 * mesh.n_nodes

        # Constitutive matrix (undamaged)
        self.C: np.ndarray = material.constitutive_matrix(plane)

        # ----------------------------------------------------------
        # Pre-compute per-element geometric data
        # ----------------------------------------------------------
        self._b = np.zeros((self.n_elements, 3))
        self._c = np.zeros((self.n_elements, 3))
        self._area = np.zeros(self.n_elements)
        self._B = [None] * self.n_elements  # (3, 6) per element
        self._B_scalar = [None] * self.n_elements  # (2, 3) per element

        for e in range(self.n_elements):
            X = mesh.nodes[mesh.elements[e]]
            # b_i = y_j - y_k, c_i = x_k - x_j  with cyclic (i,j,k)
            b = np.array([
                X[1, 1] - X[2, 1],
                X[2, 1] - X[0, 1],
                X[0, 1] - X[1, 1],
            ])
            c = np.array([
                X[2, 0] - X[1, 0],
                X[0, 0] - X[2, 0],
                X[1, 0] - X[0, 0],
            ])
            A = 0.5 * abs(
                (X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1])
                - (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1])
            )
            self._b[e] = b
            self._c[e] = c
            self._area[e] = A

            # Strain-displacement matrix (3, 6) for vector field
            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2 * i] = b[i]
                B[1, 2 * i + 1] = c[i]
                B[2, 2 * i] = c[i]
                B[2, 2 * i + 1] = b[i]
            B /= 2.0 * A
            self._B[e] = B

            # Scalar-field gradient matrix (2, 3)
            B_scalar = np.zeros((2, 3))
            B_scalar[0, :] = b / (2.0 * A)
            B_scalar[1, :] = c / (2.0 * A)
            self._B_scalar[e] = B_scalar

        # ----------------------------------------------------------
        # State variables
        # ----------------------------------------------------------
        self.displacement: np.ndarray = np.zeros(self.n_dof)
        self.damage: np.ndarray = np.zeros(self.n_nodes)
        self.damage_old: np.ndarray = np.zeros(self.n_nodes)
        self.history: np.ndarray = np.zeros(self.n_elements)

        # ----------------------------------------------------------
        # Timing instrumentation
        # ----------------------------------------------------------
        self.timing: Dict[str, List[float]] = {
            "assembly_displacement": [],
            "solve_displacement": [],
            "assembly_damage": [],
            "solve_damage": [],
            "total_per_step": [],
        }

        # Results storage
        self.results: List[LoadStep] = []

    # ==================================================================
    #  Element-level computations
    # ==================================================================

    def _element_dof_indices(self, e_idx: int) -> np.ndarray:
        """Return the 6 global displacement DOF indices for element *e_idx*."""
        nodes = self.mesh.elements[e_idx]
        dofs = np.empty(6, dtype=np.int64)
        dofs[0] = 2 * nodes[0]
        dofs[1] = 2 * nodes[0] + 1
        dofs[2] = 2 * nodes[1]
        dofs[3] = 2 * nodes[1] + 1
        dofs[4] = 2 * nodes[2]
        dofs[5] = 2 * nodes[2] + 1
        return dofs

    # ------------------------------------------------------------------
    # 1. Element stiffness with degradation
    # ------------------------------------------------------------------
    def _compute_element_stiffness(
        self, e_idx: int, d_elem: float
    ) -> np.ndarray:
        """Compute CST element stiffness degraded by element-average damage.

        Parameters
        ----------
        e_idx : int
            Element index.
        d_elem : float
            Element-average damage (mean of the three nodal values).

        Returns
        -------
        K_e : np.ndarray, shape (6, 6)
            Degraded element stiffness matrix.

        Notes
        -----
        For the spectral-split formulation, degrading the *full* constitutive
        matrix by the average element damage is the standard simplification
        used in the staggered scheme (the split only enters the history
        driving force, not the stiffness assembly).
        """
        g = (1.0 - d_elem) ** 2 + _ETA
        B = self._B[e_idx]
        A = self._area[e_idx]
        t = self.mesh.thickness
        K_e = g * (B.T @ self.C @ B) * A * t
        return K_e

    # ------------------------------------------------------------------
    # 2. Global stiffness assembly
    # ------------------------------------------------------------------
    def _assemble_stiffness(self) -> csr_matrix:
        """Assemble the global stiffness matrix with current nodal damage.

        Returns
        -------
        K : csr_matrix, shape (2*n_nodes, 2*n_nodes)
        """
        n = self.n_dof
        K = lil_matrix((n, n))
        elements = self.mesh.elements

        for e in range(self.n_elements):
            # Element-average damage from the three nodal values
            node_ids = elements[e]
            d_elem = (
                self.damage[node_ids[0]]
                + self.damage[node_ids[1]]
                + self.damage[node_ids[2]]
            ) / 3.0

            K_e = self._compute_element_stiffness(e, d_elem)
            dofs = self._element_dof_indices(e)

            for i_loc in range(6):
                ig = dofs[i_loc]
                for j_loc in range(6):
                    jg = dofs[j_loc]
                    K[ig, jg] += K_e[i_loc, j_loc]

        return K.tocsr()

    # ------------------------------------------------------------------
    # 3. Element history (tensile strain energy)
    # ------------------------------------------------------------------
    def _compute_element_history(
        self, e_idx: int, u: np.ndarray
    ) -> float:
        """Compute tensile strain energy for *e_idx* and return updated H.

        The history variable enforces irreversibility:

            H_e = max(H_e_old, psi_plus)

        Parameters
        ----------
        e_idx : int
            Element index.
        u : np.ndarray, shape (n_dof,)
            Global displacement vector.

        Returns
        -------
        H_new : float
            Updated (max of old and current) tensile energy density.
        """
        dofs = self._element_dof_indices(e_idx)
        u_e = u[dofs]
        B = self._B[e_idx]

        eps = B @ u_e  # [eps_xx, eps_yy, gamma_xy]

        psi_plus, _ = compute_split_energy_miehe(
            eps, self.material.lame_lambda, self.material.lame_mu
        )

        return max(self.history[e_idx], psi_plus)

    # ------------------------------------------------------------------
    # 4. Damage system assembly (node-based)
    # ------------------------------------------------------------------
    def _assemble_damage_system(self) -> Tuple[csr_matrix, np.ndarray]:
        """Assemble the linear system for the AT2 damage sub-problem.

        Weak form (for each test function q):

            integral (Gc/l0 + 2*H) d q dx
            + integral Gc*l0 grad(d) . grad(q) dx
            = integral 2*H q dx

        Discretised on CST elements this gives per element:

        - Consistent scalar mass matrix:

            M_e = (A*t / 12) * [[2, 1, 1],
                                 [1, 2, 1],
                                 [1, 1, 2]]

        - Scalar gradient stiffness:

            K_grad_e = B_scalar^T B_scalar * A * t

          where B_scalar is the (2, 3) matrix of shape function partial
          derivatives.

        - Element contribution:

            A_d_local = (Gc/l0 + 2*H_e) * M_e  +  Gc*l0 * K_grad_e
            b_d_local = 2*H_e * M_e @ [1, 1, 1]^T

        Returns
        -------
        A_d : csr_matrix, shape (n_nodes, n_nodes)
            System matrix (symmetric positive-definite).
        b_d : np.ndarray, shape (n_nodes,)
            Right-hand side vector.
        """
        n = self.n_nodes
        A_d = lil_matrix((n, n))
        b_d = np.zeros(n)

        Gc = self.material.Gc
        l0 = self.material.l0
        t = self.mesh.thickness
        ones3 = np.ones(3)

        # Consistent scalar mass template (will be scaled)
        M_template = np.array(
            [[2.0, 1.0, 1.0],
             [1.0, 2.0, 1.0],
             [1.0, 1.0, 2.0]]
        )

        elements = self.mesh.elements

        for e in range(self.n_elements):
            A_e = self._area[e]
            H_e = self.history[e]
            node_ids = elements[e]

            # Consistent mass matrix
            M_e = (A_e * t / 12.0) * M_template

            # Gradient stiffness
            Bs = self._B_scalar[e]  # (2, 3)
            K_grad_e = (Bs.T @ Bs) * A_e * t

            # Local system
            A_local = (Gc / l0 + 2.0 * H_e) * M_e + Gc * l0 * K_grad_e
            b_local = 2.0 * H_e * (M_e @ ones3)

            # Scatter into global system
            for i_loc in range(3):
                ig = node_ids[i_loc]
                b_d[ig] += b_local[i_loc]
                for j_loc in range(3):
                    jg = node_ids[j_loc]
                    A_d[ig, jg] += A_local[i_loc, j_loc]

        return A_d.tocsr(), b_d

    # ==================================================================
    #  Main solver
    # ==================================================================

    def solve(
        self,
        load_factors: np.ndarray,
        bc_dofs: np.ndarray,
        bc_values_func: Callable[[float], np.ndarray],
        external_force_func: Optional[Callable[[float], np.ndarray]] = None,
    ) -> List[LoadStep]:
        """Solve the coupled problem over a series of load steps.

        Parameters
        ----------
        load_factors : np.ndarray
            Array of load multipliers (one per step).
        bc_dofs : np.ndarray
            Indices of constrained displacement DOFs.
        bc_values_func : callable
            ``bc_values_func(load_factor) -> np.ndarray`` returning the
            prescribed values that correspond to *bc_dofs* for the given
            load factor.
        external_force_func : callable, optional
            ``external_force_func(load_factor) -> np.ndarray`` returning
            the external force vector of length ``n_dof``.

        Returns
        -------
        results : List[LoadStep]
            One ``LoadStep`` per load factor.
        """
        self.results = []

        for step_idx, lf in enumerate(load_factors):
            t_step_start = time.perf_counter()

            if self.config.verbose:
                print(
                    f"\n=== Load Step {step_idx + 1}/{len(load_factors)}: "
                    f"lambda = {lf:.6f} ==="
                )

            bc_values = bc_values_func(lf)

            if external_force_func is not None:
                F_ext = external_force_func(lf)
            else:
                F_ext = np.zeros(self.n_dof)

            result = self._staggered_iteration(
                step_idx, lf, bc_dofs, bc_values, F_ext
            )

            self.timing["total_per_step"].append(
                time.perf_counter() - t_step_start
            )

            self.results.append(result)

            if not result.converged and self.config.verbose:
                print(f"WARNING: Step {step_idx + 1} did not converge!")

        return self.results

    # ------------------------------------------------------------------
    # Staggered iteration for a single load step
    # ------------------------------------------------------------------
    def _staggered_iteration(
        self,
        step: int,
        load_factor: float,
        bc_dofs: np.ndarray,
        bc_values: np.ndarray,
        F_ext: np.ndarray,
    ) -> LoadStep:
        """Perform staggered (alternating minimisation) iterations."""
        u_old = self.displacement.copy()
        d_old_iter = self.damage.copy()

        # Save damage at start of step for irreversibility
        self.damage_old = self.damage.copy()

        converged = False
        residual_u = 0.0
        residual_d = 0.0
        n_iter = 0

        for iteration in range(self.config.max_stagger_iter):
            n_iter = iteration + 1

            # ---- Step 1: Solve displacement (mechanical equilibrium) ----
            t0 = time.perf_counter()
            K = self._assemble_stiffness()
            t_asm_u = time.perf_counter() - t0
            self.timing["assembly_displacement"].append(t_asm_u)

            t0 = time.perf_counter()
            K_bc, F_bc = apply_dirichlet_bc(K, F_ext.copy(), bc_dofs, bc_values)
            u_new = spsolve(K_bc, F_bc)
            t_sol_u = time.perf_counter() - t0
            self.timing["solve_displacement"].append(t_sol_u)

            # ---- Step 2: Compute element strains and update history -----
            for e in range(self.n_elements):
                self.history[e] = self._compute_element_history(e, u_new)

            # ---- Step 3: Solve damage (phase-field evolution) -----------
            t0 = time.perf_counter()
            A_d, b_d = self._assemble_damage_system()
            t_asm_d = time.perf_counter() - t0
            self.timing["assembly_damage"].append(t_asm_d)

            t0 = time.perf_counter()
            d_new = spsolve(A_d, b_d)
            t_sol_d = time.perf_counter() - t0
            self.timing["solve_damage"].append(t_sol_d)

            # ---- Step 4: Enforce bounds and irreversibility -------------
            d_new = np.clip(d_new, 0.0, 1.0)
            d_new = np.maximum(d_new, self.damage_old)

            # ---- Step 5: Check convergence ------------------------------
            u_norm = np.linalg.norm(u_new)
            d_norm = np.linalg.norm(d_new)

            if u_norm > 1.0e-10:
                residual_u = np.linalg.norm(u_new - u_old) / u_norm
            else:
                residual_u = np.linalg.norm(u_new - u_old)

            if d_norm > 1.0e-10:
                residual_d = np.linalg.norm(d_new - d_old_iter) / d_norm
            else:
                residual_d = np.linalg.norm(d_new - d_old_iter)

            if self.config.verbose:
                max_d = np.max(d_new) if len(d_new) > 0 else 0.0
                print(
                    f"  Iter {n_iter}: "
                    f"Du = {residual_u:.2e}, "
                    f"Dd = {residual_d:.2e}, "
                    f"max(d) = {max_d:.4f}"
                )

            if iteration >= self.config.min_stagger_iter - 1:
                if (
                    residual_u < self.config.tol_u
                    and residual_d < self.config.tol_d
                ):
                    converged = True
                    if self.config.verbose:
                        print(f"  Converged in {n_iter} iterations")
                    # Commit solution before breaking
                    self.displacement = u_new
                    self.damage = d_new
                    break

            # Prepare next iteration
            u_old = u_new.copy()
            d_old_iter = d_new.copy()
            self.displacement = u_new
            self.damage = d_new

        else:
            # Loop exhausted without break -- commit last solution
            self.displacement = u_new
            self.damage = d_new

        # ----- Compute energies for this load step -----
        strain_energy = self._compute_strain_energy(self.displacement, self.damage)
        surface_energy = self._compute_surface_energy(self.damage)
        external_work = float(np.dot(F_ext, self.displacement))

        return LoadStep(
            step=step,
            load_factor=load_factor,
            displacement=self.displacement.copy(),
            damage=self.damage.copy(),
            strain_energy=strain_energy,
            surface_energy=surface_energy,
            external_work=external_work,
            converged=converged,
            n_iterations=n_iter,
            residual_u=residual_u,
            residual_d=residual_d,
        )

    # ==================================================================
    #  Energy computations
    # ==================================================================

    def _compute_strain_energy(
        self, u: np.ndarray, damage: np.ndarray
    ) -> float:
        """Compute total (degraded) strain energy.

        E_mech = sum_e [ g(d_e) psi_plus_e + psi_minus_e ] * A_e * t

        where ``d_e`` is the element-average damage.
        """
        total = 0.0
        elements = self.mesh.elements
        t = self.mesh.thickness

        for e in range(self.n_elements):
            dofs = self._element_dof_indices(e)
            u_e = u[dofs]
            B = self._B[e]
            eps = B @ u_e

            psi_plus, psi_minus = compute_split_energy_miehe(
                eps, self.material.lame_lambda, self.material.lame_mu
            )

            node_ids = elements[e]
            d_elem = (
                damage[node_ids[0]]
                + damage[node_ids[1]]
                + damage[node_ids[2]]
            ) / 3.0
            g = (1.0 - d_elem) ** 2 + _ETA

            psi = g * psi_plus + psi_minus
            total += psi * self._area[e] * t

        return total

    def _compute_surface_energy(self, damage: np.ndarray) -> float:
        """Compute fracture surface energy using AT2 functional.

        E_frac = d^T (Gc/(2 l0) M) d  +  d^T (Gc l0/2 K_grad) d

        where M and K_grad are the assembled consistent-mass and
        gradient-stiffness matrices for the scalar damage field.
        """
        Gc = self.material.Gc
        l0 = self.material.l0
        t = self.mesh.thickness
        elements = self.mesh.elements

        M_template = np.array(
            [[2.0, 1.0, 1.0],
             [1.0, 2.0, 1.0],
             [1.0, 1.0, 2.0]]
        )

        E_local = 0.0
        E_grad = 0.0

        for e in range(self.n_elements):
            A_e = self._area[e]
            node_ids = elements[e]
            d_e = damage[node_ids]  # (3,)

            M_e = (A_e * t / 12.0) * M_template
            Bs = self._B_scalar[e]
            K_grad_e = (Bs.T @ Bs) * A_e * t

            E_local += float(d_e @ M_e @ d_e)
            E_grad += float(d_e @ K_grad_e @ d_e)

        E_frac = Gc / (2.0 * l0) * E_local + Gc * l0 / 2.0 * E_grad
        return E_frac

    # ==================================================================
    #  Post-processing utilities
    # ==================================================================

    def compute_reaction_force(
        self,
        u: np.ndarray,
        damage: np.ndarray,
        L: float,
        direction: str = "y",
        tol_frac: float = 1.0e-6,
    ) -> float:
        """Compute net reaction force at one boundary.

        The reaction force is obtained from the global internal-force
        vector:  F_int = K(d) u, summed over the relevant DOFs at the
        boundary.

        Parameters
        ----------
        u : np.ndarray
            Displacement vector (2*n_nodes,).
        damage : np.ndarray
            Nodal damage vector (n_nodes,).
        L : float
            Domain extent in the direction perpendicular to the
            boundary.  For a uniaxial tension test with loading in *y*,
            *L* is the height (y_max).
        direction : str
            ``'x'`` or ``'y'``.
        tol_frac : float
            Fractional tolerance for identifying boundary nodes.

        Returns
        -------
        R : float
            Total reaction force at the identified boundary.
        """
        nodes = self.mesh.nodes
        if direction == "y":
            coord = nodes[:, 1]
        else:
            coord = nodes[:, 0]

        tol = tol_frac * (coord.max() - coord.min())
        top_val = coord.max()
        top_nodes = np.where(np.abs(coord - top_val) < tol)[0]

        # Assemble internal force from element contributions
        F_int = np.zeros(self.n_dof)
        elements = self.mesh.elements
        t_thick = self.mesh.thickness

        for e in range(self.n_elements):
            B = self._B[e]
            dofs = self._element_dof_indices(e)
            u_e = u[dofs]
            eps = B @ u_e

            node_ids = elements[e]
            d_elem = (
                damage[node_ids[0]]
                + damage[node_ids[1]]
                + damage[node_ids[2]]
            ) / 3.0

            # Degraded stress (full constitutive matrix degraded uniformly)
            g = (1.0 - d_elem) ** 2 + _ETA
            sigma = g * (self.C @ eps)

            f_e = B.T @ sigma * self._area[e] * t_thick
            for i_loc in range(6):
                F_int[dofs[i_loc]] += f_e[i_loc]

        # Sum reaction in the requested direction at the top boundary
        if direction == "y":
            dof_indices = 2 * top_nodes + 1
        else:
            dof_indices = 2 * top_nodes

        return float(np.sum(F_int[dof_indices]))

    def extract_crack_path(
        self, threshold: float = 0.5
    ) -> np.ndarray:
        """Extract approximate crack path from the nodal damage field.

        Elements whose average nodal damage exceeds *threshold* are
        identified as "cracked".  Their centroids are returned sorted
        by ascending x-coordinate.

        Parameters
        ----------
        threshold : float
            Minimum element-average damage to be considered cracked.

        Returns
        -------
        path : np.ndarray, shape (n_cracked, 2)
            Centroids of cracked elements sorted by x.
        """
        elements = self.mesh.elements
        nodes = self.mesh.nodes

        centroids = []
        for e in range(self.n_elements):
            node_ids = elements[e]
            d_avg = (
                self.damage[node_ids[0]]
                + self.damage[node_ids[1]]
                + self.damage[node_ids[2]]
            ) / 3.0
            if d_avg > threshold:
                cx = np.mean(nodes[node_ids, 0])
                cy = np.mean(nodes[node_ids, 1])
                centroids.append([cx, cy])

        if len(centroids) == 0:
            return np.empty((0, 2))

        centroids = np.array(centroids)
        order = np.argsort(centroids[:, 0])
        return centroids[order]

    # ==================================================================
    #  Initial damage helpers
    # ==================================================================

    def set_initial_damage_from_edges(
        self,
        edge_damage: np.ndarray,
        mesh: Optional[TriangleMesh] = None,
    ) -> None:
        """Convert edge-based damage to node-based by averaging.

        Each node receives the mean damage of its connected edges.

        Parameters
        ----------
        edge_damage : np.ndarray, shape (n_edges,)
            Edge-based damage values.
        mesh : TriangleMesh, optional
            Mesh to use; defaults to ``self.mesh``.
        """
        if mesh is None:
            mesh = self.mesh

        if len(edge_damage) != mesh.n_edges:
            raise ValueError(
                f"edge_damage length {len(edge_damage)} != "
                f"n_edges {mesh.n_edges}"
            )

        nodal_damage = np.zeros(mesh.n_nodes)
        counts = np.zeros(mesh.n_nodes)

        for edge_idx, (n1, n2) in enumerate(mesh.edges):
            d_val = edge_damage[edge_idx]
            nodal_damage[n1] += d_val
            nodal_damage[n2] += d_val
            counts[n1] += 1.0
            counts[n2] += 1.0

        nonzero = counts > 0
        nodal_damage[nonzero] /= counts[nonzero]

        self.damage = np.clip(nodal_damage, 0.0, 1.0)
        self.damage_old = self.damage.copy()

        # Set history so that the initial crack does not heal
        psi_c = self.material.critical_strain_energy_density()
        for e in range(self.n_elements):
            node_ids = self.mesh.elements[e]
            d_elem = (
                self.damage[node_ids[0]]
                + self.damage[node_ids[1]]
                + self.damage[node_ids[2]]
            ) / 3.0
            self.history[e] = max(self.history[e], psi_c * d_elem)

    def set_initial_damage(self, damage: np.ndarray) -> None:
        """Set initial nodal damage directly.

        Parameters
        ----------
        damage : np.ndarray, shape (n_nodes,)
            Nodal damage values in [0, 1].
        """
        if len(damage) != self.n_nodes:
            raise ValueError(
                f"damage length {len(damage)} != n_nodes {self.n_nodes}"
            )
        self.damage = np.clip(np.array(damage, dtype=np.float64), 0.0, 1.0)
        self.damage_old = self.damage.copy()

        psi_c = self.material.critical_strain_energy_density()
        for e in range(self.n_elements):
            node_ids = self.mesh.elements[e]
            d_elem = (
                self.damage[node_ids[0]]
                + self.damage[node_ids[1]]
                + self.damage[node_ids[2]]
            ) / 3.0
            self.history[e] = max(self.history[e], psi_c * d_elem)

    def set_initial_crack_nodes(
        self, node_indices: np.ndarray, damage_value: float = 0.99
    ) -> None:
        """Set initial crack by prescribing damage at specific nodes.

        Parameters
        ----------
        node_indices : array-like
            Indices of nodes forming the initial crack.
        damage_value : float
            Damage to assign (default 0.99).
        """
        node_indices = np.asarray(node_indices, dtype=np.int64)
        self.damage[node_indices] = damage_value
        self.damage_old = self.damage.copy()

        psi_c = self.material.critical_strain_energy_density()
        for e in range(self.n_elements):
            node_ids = self.mesh.elements[e]
            d_elem = (
                self.damage[node_ids[0]]
                + self.damage[node_ids[1]]
                + self.damage[node_ids[2]]
            ) / 3.0
            self.history[e] = max(self.history[e], psi_c * d_elem)

    # ==================================================================
    #  Reset / timing
    # ==================================================================

    def reset(self) -> None:
        """Reset solver to the initial (undamaged, unloaded) state."""
        self.displacement = np.zeros(self.n_dof)
        self.damage = np.zeros(self.n_nodes)
        self.damage_old = np.zeros(self.n_nodes)
        self.history = np.zeros(self.n_elements)
        self.results = []
        for key in self.timing:
            self.timing[key] = []

    def get_timing_summary(self) -> Dict[str, float]:
        """Return mean timings (in seconds) for each instrumented phase.

        Returns
        -------
        summary : dict
            Keys match ``self.timing``; values are the arithmetic mean
            of all recorded samples for that phase.  An additional key
            ``'total_mean'`` gives the mean total wall-clock time per
            load step.
        """
        summary: Dict[str, float] = {}
        for key, vals in self.timing.items():
            if len(vals) > 0:
                summary[key] = float(np.mean(vals))
            else:
                summary[key] = 0.0
        # Alias for convenience
        summary["total_mean"] = summary.get("total_per_step", 0.0)
        return summary

    def get_results_summary(self) -> dict:
        """Return summary statistics from completed load steps.

        Returns
        -------
        info : dict
            Keys: ``n_steps``, ``converged_steps``, ``max_damage``,
            ``final_strain_energy``, ``final_surface_energy``,
            ``total_iterations``.
        """
        if not self.results:
            return {}

        return {
            "n_steps": len(self.results),
            "converged_steps": sum(r.converged for r in self.results),
            "max_damage": float(max(np.max(r.damage) for r in self.results)),
            "final_strain_energy": float(self.results[-1].strain_energy),
            "final_surface_energy": float(self.results[-1].surface_energy),
            "total_iterations": sum(r.n_iterations for r in self.results),
        }
