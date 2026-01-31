"""
Tests for Assembly Module
=========================
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh.triangle_mesh import TriangleMesh
from mesh.mesh_generators import create_rectangle_mesh, create_single_element
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial
from assembly.global_assembly import (
    assemble_global_stiffness, assemble_internal_force,
    compute_all_edge_strains, compute_strain_energy
)
from assembly.boundary_conditions import (
    apply_dirichlet_bc, create_bc_from_region, merge_bcs,
    create_fixed_bc, BoundaryConditionManager
)


class TestGlobalAssembly:
    """Tests for global assembly functions."""

    @pytest.fixture
    def setup_simple_mesh(self):
        """Create simple mesh with elements."""
        mesh = create_rectangle_mesh(1, 1, 3, 3)
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.02)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]
        return mesh, elements, material

    def test_stiffness_matrix_shape(self, setup_simple_mesh):
        """Test stiffness matrix shape."""
        mesh, elements, _ = setup_simple_mesh
        damage = np.zeros(mesh.n_edges)

        K = assemble_global_stiffness(mesh, elements, damage)

        expected_size = 2 * mesh.n_nodes
        assert K.shape == (expected_size, expected_size)

    def test_stiffness_matrix_symmetry(self, setup_simple_mesh):
        """Global stiffness should be symmetric."""
        mesh, elements, _ = setup_simple_mesh
        damage = np.zeros(mesh.n_edges)

        K = assemble_global_stiffness(mesh, elements, damage)
        K_dense = K.toarray()

        # Use relative tolerance based on matrix magnitude
        # Small numerical asymmetries can arise from floating-point operations
        max_val = np.max(np.abs(K_dense))
        assert np.allclose(K_dense, K_dense.T, atol=max_val * 1e-14)

    def test_stiffness_matrix_sparsity(self, setup_simple_mesh):
        """Stiffness matrix should be sparse."""
        mesh, elements, _ = setup_simple_mesh
        damage = np.zeros(mesh.n_edges)

        K = assemble_global_stiffness(mesh, elements, damage)

        # Count non-zeros
        nnz = K.nnz
        total = K.shape[0] ** 2

        # For FEM stiffness matrices, sparsity depends on mesh connectivity
        # For a 3x3 mesh (small), the density can be higher
        # A 6-node CST element contributes 36 non-zeros per element
        # For larger meshes, sparsity improves significantly
        # Accept density < 0.5 for small meshes
        assert nnz / total < 0.5, \
            f"Matrix too dense: {nnz}/{total} = {nnz/total:.2%}"

    def test_internal_force_equilibrium(self, setup_simple_mesh):
        """Internal forces should sum to zero (equilibrium)."""
        mesh, elements, _ = setup_simple_mesh

        # Apply some displacement
        u = np.zeros(2 * mesh.n_nodes)
        u[::2] = 0.001 * mesh.nodes[:, 0]  # Linear x-displacement

        damage = np.zeros(mesh.n_edges)
        F = assemble_internal_force(mesh, elements, u, damage)

        # Sum of all forces should be zero
        Fx = np.sum(F[::2])
        Fy = np.sum(F[1::2])

        assert np.isclose(Fx, 0, atol=1e-6)
        assert np.isclose(Fy, 0, atol=1e-6)

    def test_strain_energy_positive(self, setup_simple_mesh):
        """Strain energy should be non-negative."""
        mesh, elements, _ = setup_simple_mesh

        u = np.zeros(2 * mesh.n_nodes)
        u[::2] = 0.001 * mesh.nodes[:, 0]

        damage = np.zeros(mesh.n_edges)
        E = compute_strain_energy(mesh, elements, u, damage)

        assert E >= 0


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    @pytest.fixture
    def setup_mesh(self):
        """Create test mesh."""
        return create_rectangle_mesh(1, 1, 5, 5)

    def test_bc_from_region_bottom(self, setup_mesh):
        """Test BCs on bottom edge."""
        mesh = setup_mesh

        bc_dofs, bc_vals = create_bc_from_region(
            mesh, lambda x, y: y < 0.01, 'y', 0.0
        )

        # Should have (nx+1) nodes on bottom
        assert len(bc_dofs) == 6  # nx+1 = 6

    def test_bc_from_region_both_components(self, setup_mesh):
        """Test BCs on both components."""
        mesh = setup_mesh

        bc_dofs, bc_vals = create_bc_from_region(
            mesh, lambda x, y: y < 0.01, 'both', 0.0
        )

        # Should have 2*(nx+1) DOFs
        assert len(bc_dofs) == 12

    def test_merge_bcs(self, setup_mesh):
        """Test merging multiple BC sets."""
        mesh = setup_mesh

        bc1_dofs, bc1_vals = create_bc_from_region(
            mesh, lambda x, y: y < 0.01, 'y', 0.0
        )
        bc2_dofs, bc2_vals = create_bc_from_region(
            mesh, lambda x, y: x < 0.01, 'x', 0.0
        )

        merged_dofs, merged_vals = merge_bcs(
            (bc1_dofs, bc1_vals),
            (bc2_dofs, bc2_vals)
        )

        # Check no duplicates
        assert len(merged_dofs) == len(set(merged_dofs))

    def test_bc_application_elimination(self, setup_mesh):
        """Test BC application using elimination method."""
        mesh = setup_mesh
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.02)
        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2 * mesh.n_nodes)

        # Fix bottom, prescribe top
        bc_dofs = np.array([0, 1, 2, 3])  # Just a few DOFs
        bc_vals = np.array([0, 0, 0.001, 0])

        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_vals, method='elimination')

        # Rows/cols for constrained DOFs should be identity
        K_bc_dense = K_bc.toarray()
        for i, dof in enumerate(bc_dofs):
            assert np.isclose(K_bc_dense[dof, dof], 1.0)
            assert np.isclose(F_bc[dof], bc_vals[i])


class TestPatchTest:
    """Patch test verification."""

    def test_patch_test_constant_strain(self):
        """
        CRITICAL: Constant strain must be reproduced exactly.
        """
        # Create mesh
        mesh = create_rectangle_mesh(1, 1, 5, 5, pattern='right')
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.02)

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        # Apply linear displacement: u_x = 0.001*x, u_y = 0
        # Expected strain: ε_xx = 0.001, ε_yy = 0, γ_xy = 0
        expected_strain = np.array([0.001, 0, 0])

        # Set BCs: prescribed displacement on all boundary nodes
        bc_dofs = []
        bc_values = []
        for i in mesh.boundary_nodes:
            x, y = mesh.nodes[i]
            bc_dofs.extend([2*i, 2*i+1])
            bc_values.extend([0.001*x, 0])

        bc_dofs = np.array(bc_dofs)
        bc_values = np.array(bc_values)

        # Solve
        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2*mesh.n_nodes)

        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        # Check all elements have same strain
        edge_strains = compute_all_edge_strains(mesh, elements, u)

        for e_idx, elem in enumerate(elements):
            eps_tensor = elem.T_inv @ edge_strains[e_idx]
            assert np.allclose(eps_tensor, expected_strain, rtol=1e-4), \
                f"Element {e_idx}: expected {expected_strain}, got {eps_tensor}"

    def test_patch_test_biaxial(self):
        """Patch test with biaxial strain."""
        mesh = create_rectangle_mesh(1, 1, 4, 4)
        material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.02)

        elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
                    for e in range(mesh.n_elements)]

        # Apply biaxial: u_x = 0.001*x, u_y = 0.0005*y
        # Expected strain: ε_xx = 0.001, ε_yy = 0.0005, γ_xy = 0
        expected_strain = np.array([0.001, 0.0005, 0])

        bc_dofs = []
        bc_values = []
        for i in mesh.boundary_nodes:
            x, y = mesh.nodes[i]
            bc_dofs.extend([2*i, 2*i+1])
            bc_values.extend([0.001*x, 0.0005*y])

        bc_dofs = np.array(bc_dofs)
        bc_values = np.array(bc_values)

        damage = np.zeros(mesh.n_edges)
        K = assemble_global_stiffness(mesh, elements, damage)
        F = np.zeros(2*mesh.n_nodes)

        K_bc, F_bc = apply_dirichlet_bc(K, F, bc_dofs, bc_values)
        u = spsolve(K_bc, F_bc)

        edge_strains = compute_all_edge_strains(mesh, elements, u)

        for e_idx, elem in enumerate(elements):
            eps_tensor = elem.T_inv @ edge_strains[e_idx]
            assert np.allclose(eps_tensor, expected_strain, rtol=1e-4)


class TestBCManager:
    """Tests for BoundaryConditionManager."""

    def test_bc_manager_basic(self):
        """Test basic BC manager functionality."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        bc_mgr = BoundaryConditionManager(mesh)

        bc_mgr.fix_region(lambda x, y: y < 0.01, 'both', name='bottom')

        assert len(bc_mgr.bc_dofs) > 0
        assert len(bc_mgr.bc_dofs) == len(bc_mgr.bc_values)

    def test_bc_manager_multiple_regions(self):
        """Test BC manager with multiple regions."""
        mesh = create_rectangle_mesh(1, 1, 5, 5)
        bc_mgr = BoundaryConditionManager(mesh)

        bc_mgr.fix_region(lambda x, y: y < 0.01, 'y', name='bottom')
        bc_mgr.prescribe_displacement(lambda x, y: y > 0.99, 'y', 0.001, name='top')

        assert len(bc_mgr.bc_dofs) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
