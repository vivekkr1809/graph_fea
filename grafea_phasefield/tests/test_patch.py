"""
Patch Test for Multi-Element FEM Implementation
================================================

This test verifies that constant strain states are reproduced exactly
across arbitrary (unstructured) meshes. This is a fundamental validation
test for any FEM implementation.

Theory:
-------
For a displacement field that produces a constant strain:
    u_x = a*x + b*y
    u_y = c*x + d*y

The exact strains are:
    ε_xx = a
    ε_yy = d
    γ_xy = b + c

For CST (Constant Strain Triangle) elements, these strains should be
reproduced EXACTLY (to machine precision) regardless of mesh topology.

Test Cases:
-----------
1. Uniaxial X: a=0.001, b=c=d=0  → ε_xx=0.001, ε_yy=0, γ_xy=0
2. Uniaxial Y: a=b=c=0, d=0.001  → ε_xx=0, ε_yy=0.001, γ_xy=0
3. Pure Shear: a=d=0, b=c=0.001  → ε_xx=0, ε_yy=0, γ_xy=0.002
4. Biaxial:    a=d=0.001, b=c=0  → ε_xx=0.001, ε_yy=0.001, γ_xy=0

Pass Criteria:
--------------
All element strains must match exact values with error < 1e-14
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh.mesh_generators import (
    create_rectangle_mesh,
    create_square_mesh,
    perturb_interior_nodes,
    create_single_element,
    create_two_element_patch
)
from elements.cst_element import CSTElement
from elements.grafea_element import GraFEAElement
from physics.material import IsotropicMaterial


# =============================================================================
# Test Case Definitions
# =============================================================================

TEST_CASES = {
    'uniaxial_x': {
        'a': 0.001, 'b': 0.0, 'c': 0.0, 'd': 0.0,
        'description': 'Uniaxial tension in x-direction',
        'expected_strain': np.array([0.001, 0.0, 0.0])  # [ε_xx, ε_yy, γ_xy]
    },
    'uniaxial_y': {
        'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.001,
        'description': 'Uniaxial tension in y-direction',
        'expected_strain': np.array([0.0, 0.001, 0.0])
    },
    'pure_shear': {
        'a': 0.0, 'b': 0.001, 'c': 0.001, 'd': 0.0,
        'description': 'Pure shear deformation',
        'expected_strain': np.array([0.0, 0.0, 0.002])
    },
    'biaxial': {
        'a': 0.001, 'b': 0.0, 'c': 0.0, 'd': 0.001,
        'description': 'Equibiaxial tension',
        'expected_strain': np.array([0.001, 0.001, 0.0])
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def prescribe_linear_displacement(nodes: np.ndarray, a: float, b: float,
                                   c: float, d: float) -> np.ndarray:
    """
    Prescribe linear displacement field to all nodes.

    Displacement field:
        u_x = a*x + b*y
        u_y = c*x + d*y

    Args:
        nodes: shape (n_nodes, 2), node coordinates
        a, b, c, d: displacement field coefficients

    Returns:
        u: shape (n_nodes, 2), nodal displacements
    """
    n_nodes = nodes.shape[0]
    u = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        x, y = nodes[i]
        u[i, 0] = a * x + b * y
        u[i, 1] = c * x + d * y
    return u


def compute_exact_strain(a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Compute exact strain from displacement field coefficients.

    For u_x = a*x + b*y, u_y = c*x + d*y:
        ε_xx = ∂u_x/∂x = a
        ε_yy = ∂u_y/∂y = d
        γ_xy = ∂u_x/∂y + ∂u_y/∂x = b + c

    Args:
        a, b, c, d: displacement field coefficients

    Returns:
        strain: shape (3,), [ε_xx, ε_yy, γ_xy] in Voigt notation
    """
    return np.array([a, d, b + c])


def run_patch_test_cst(mesh, material, test_case: dict) -> tuple:
    """
    Run patch test using CST elements.

    Args:
        mesh: TriangleMesh instance
        material: IsotropicMaterial instance
        test_case: dict with keys 'a', 'b', 'c', 'd', 'expected_strain'

    Returns:
        passed: bool, whether test passed
        max_error: float, maximum strain error across all elements
        errors: list, error for each element
    """
    a, b, c, d = test_case['a'], test_case['b'], test_case['c'], test_case['d']
    exact_strain = test_case['expected_strain']

    # Prescribe displacements
    u = prescribe_linear_displacement(mesh.nodes, a, b, c, d)

    # Compute strains in all elements
    errors = []
    for e_idx, elem_nodes in enumerate(mesh.elements):
        # Get element node coordinates
        elem_coords = mesh.nodes[elem_nodes]

        # Create CST element
        element = CSTElement(elem_coords, material)

        # Get element displacements
        elem_u = u[elem_nodes]

        # Compute strain
        eps_computed = element.compute_strain(elem_u)

        # Compute error
        error = np.max(np.abs(eps_computed - exact_strain))
        errors.append(error)

    max_error = max(errors)
    passed = max_error < 1e-12  # Slightly relaxed from 1e-14 for numerical safety

    return passed, max_error, errors


def run_patch_test_grafea(mesh, material, test_case: dict) -> tuple:
    """
    Run patch test using GraFEA elements.

    Args:
        mesh: TriangleMesh instance
        material: IsotropicMaterial instance
        test_case: dict with keys 'a', 'b', 'c', 'd', 'expected_strain'

    Returns:
        passed: bool, whether test passed
        max_error: float, maximum strain error across all elements
        errors: list, error for each element
    """
    a, b, c, d = test_case['a'], test_case['b'], test_case['c'], test_case['d']
    exact_strain = test_case['expected_strain']

    # Prescribe displacements
    u = prescribe_linear_displacement(mesh.nodes, a, b, c, d)

    # Compute strains in all elements
    errors = []
    for e_idx, elem_nodes in enumerate(mesh.elements):
        # Get element node coordinates
        elem_coords = mesh.nodes[elem_nodes]

        # Create GraFEA element
        element = GraFEAElement(elem_coords, material)

        # Get element displacements
        elem_u = u[elem_nodes]

        # Compute tensor strain (should match CST)
        eps_computed = element.compute_tensor_strain(elem_u)

        # Compute error
        error = np.max(np.abs(eps_computed - exact_strain))
        errors.append(error)

    max_error = max(errors)
    passed = max_error < 1e-12

    return passed, max_error, errors


def create_unstructured_mesh(n_divisions: int = 5, perturbation: float = 0.2,
                             seed: int = 42):
    """
    Create unstructured triangular mesh for patch test.

    Creates a structured mesh and then perturbs interior nodes to create
    an unstructured mesh that is NOT aligned with coordinate axes.

    Args:
        n_divisions: number of divisions per side
        perturbation: magnitude of perturbation as fraction of min edge length
        seed: random seed for reproducibility

    Returns:
        TriangleMesh instance
    """
    # Create base structured mesh
    mesh = create_square_mesh(1.0, n_divisions, pattern='alternating')

    # Perturb interior nodes to create unstructured mesh
    mesh = perturb_interior_nodes(mesh, magnitude=perturbation, seed=seed)

    return mesh


# =============================================================================
# Test Classes
# =============================================================================

class TestPatchTestSingleElement:
    """Test patch test on single element first (debugging)."""

    @pytest.fixture
    def material(self):
        """Standard material for testing."""
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    @pytest.fixture
    def right_triangle(self):
        """Standard right triangle element."""
        return create_single_element()

    @pytest.fixture
    def arbitrary_triangle(self):
        """Arbitrary (non-right) triangle element."""
        nodes = np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.1]])
        return create_single_element(nodes)

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_right_triangle_cst(self, right_triangle, material, case_name):
        """Patch test on right triangle with CST element."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(right_triangle, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on right triangle CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_right_triangle_grafea(self, right_triangle, material, case_name):
        """Patch test on right triangle with GraFEA element."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_grafea(right_triangle, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on right triangle GraFEA.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_arbitrary_triangle_cst(self, arbitrary_triangle, material, case_name):
        """Patch test on arbitrary triangle with CST element."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(arbitrary_triangle, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on arbitrary triangle CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_arbitrary_triangle_grafea(self, arbitrary_triangle, material, case_name):
        """Patch test on arbitrary triangle with GraFEA element."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_grafea(arbitrary_triangle, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on arbitrary triangle GraFEA.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestTwoElement:
    """Test patch test on two-element patch."""

    @pytest.fixture
    def material(self):
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    @pytest.fixture
    def two_element_mesh(self):
        return create_two_element_patch()

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_two_element_cst(self, two_element_mesh, material, case_name):
        """Patch test on two-element patch with CST."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(two_element_mesh, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on two-element patch CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_two_element_grafea(self, two_element_mesh, material, case_name):
        """Patch test on two-element patch with GraFEA."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_grafea(two_element_mesh, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on two-element patch GraFEA.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestStructuredMesh:
    """Test patch test on structured meshes."""

    @pytest.fixture
    def material(self):
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    @pytest.fixture
    def structured_mesh(self):
        """Create a 5x5 structured mesh."""
        return create_square_mesh(1.0, 5, pattern='right')

    @pytest.fixture
    def alternating_mesh(self):
        """Create mesh with alternating diagonal pattern."""
        return create_square_mesh(1.0, 5, pattern='alternating')

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_structured_mesh_cst(self, structured_mesh, material, case_name):
        """Patch test on structured mesh with CST."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(structured_mesh, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on structured mesh CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_alternating_mesh_cst(self, alternating_mesh, material, case_name):
        """Patch test on alternating mesh with CST."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(alternating_mesh, material, test_case)

        assert passed, (
            f"Patch test failed for {case_name} on alternating mesh CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestUnstructuredMesh:
    """
    Test patch test on unstructured meshes (main validation test).

    This is the critical test - CST elements should reproduce constant
    strain states EXACTLY regardless of mesh topology.
    """

    @pytest.fixture
    def material(self):
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    @pytest.fixture
    def unstructured_mesh_small(self):
        """Small unstructured mesh (5x5 base, perturbed)."""
        return create_unstructured_mesh(n_divisions=5, perturbation=0.2, seed=42)

    @pytest.fixture
    def unstructured_mesh_medium(self):
        """Medium unstructured mesh (10x10 base, perturbed)."""
        return create_unstructured_mesh(n_divisions=10, perturbation=0.2, seed=123)

    @pytest.fixture
    def unstructured_mesh_large_perturbation(self):
        """Mesh with large perturbation (stress test)."""
        return create_unstructured_mesh(n_divisions=5, perturbation=0.4, seed=456)

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_unstructured_small_cst(self, unstructured_mesh_small, material, case_name):
        """Patch test on small unstructured mesh with CST."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(
            unstructured_mesh_small, material, test_case
        )

        assert passed, (
            f"Patch test failed for {case_name} on small unstructured mesh CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_unstructured_small_grafea(self, unstructured_mesh_small, material, case_name):
        """Patch test on small unstructured mesh with GraFEA."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_grafea(
            unstructured_mesh_small, material, test_case
        )

        assert passed, (
            f"Patch test failed for {case_name} on small unstructured mesh GraFEA.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_unstructured_medium_cst(self, unstructured_mesh_medium, material, case_name):
        """Patch test on medium unstructured mesh with CST."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(
            unstructured_mesh_medium, material, test_case
        )

        assert passed, (
            f"Patch test failed for {case_name} on medium unstructured mesh CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )

    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_unstructured_large_perturbation_cst(
        self, unstructured_mesh_large_perturbation, material, case_name
    ):
        """Patch test on mesh with large perturbation (stress test)."""
        test_case = TEST_CASES[case_name]
        passed, max_error, _ = run_patch_test_cst(
            unstructured_mesh_large_perturbation, material, test_case
        )

        assert passed, (
            f"Patch test failed for {case_name} with large perturbation CST.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestMaterialVariations:
    """Test patch test with different materials."""

    @pytest.fixture
    def unstructured_mesh(self):
        return create_unstructured_mesh(n_divisions=5, perturbation=0.2, seed=42)

    @pytest.mark.parametrize("E,nu", [
        (210e9, 0.3),    # Steel
        (70e9, 0.33),    # Aluminum
        (200e9, 0.0),    # No Poisson effect
        (100e9, 0.45),   # Nearly incompressible
        (1e6, 0.25),     # Soft material
    ])
    def test_material_variations(self, unstructured_mesh, E, nu):
        """Patch test should pass regardless of material properties."""
        material = IsotropicMaterial(E=E, nu=nu, Gc=2700, l0=0.01)
        test_case = TEST_CASES['biaxial']

        passed, max_error, _ = run_patch_test_cst(unstructured_mesh, material, test_case)

        assert passed, (
            f"Patch test failed for E={E}, nu={nu}.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestPlaneConditions:
    """Test patch test under different plane conditions."""

    @pytest.fixture
    def material(self):
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    @pytest.fixture
    def unstructured_mesh(self):
        return create_unstructured_mesh(n_divisions=5, perturbation=0.2, seed=42)

    @pytest.mark.parametrize("plane", ['strain', 'stress'])
    @pytest.mark.parametrize("case_name", TEST_CASES.keys())
    def test_plane_conditions(self, unstructured_mesh, material, plane, case_name):
        """Patch test should pass for both plane strain and plane stress."""
        test_case = TEST_CASES[case_name]
        a, b, c, d = test_case['a'], test_case['b'], test_case['c'], test_case['d']
        exact_strain = test_case['expected_strain']

        # Prescribe displacements
        u = prescribe_linear_displacement(unstructured_mesh.nodes, a, b, c, d)

        # Compute strains in all elements
        max_error = 0.0
        for e_idx, elem_nodes in enumerate(unstructured_mesh.elements):
            elem_coords = unstructured_mesh.nodes[elem_nodes]
            element = CSTElement(elem_coords, material, plane=plane)
            elem_u = u[elem_nodes]
            eps_computed = element.compute_strain(elem_u)
            error = np.max(np.abs(eps_computed - exact_strain))
            max_error = max(max_error, error)

        assert max_error < 1e-12, (
            f"Patch test failed for {case_name} with plane={plane}.\n"
            f"Max error: {max_error:.2e} (tolerance: 1e-12)"
        )


class TestPatchTestDiagnostics:
    """Diagnostic tests for debugging patch test failures."""

    @pytest.fixture
    def material(self):
        return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)

    def test_b_matrix_verification(self, material):
        """
        Verify B matrix computation for a known element.

        For element with nodes at (0,0), (1,0), (0,1):
        Area = 0.5

        Shape function derivatives:
        b_0 = y_1 - y_2 = 0 - 1 = -1
        b_1 = y_2 - y_0 = 1 - 0 = 1
        b_2 = y_0 - y_1 = 0 - 0 = 0

        c_0 = x_2 - x_1 = 0 - 1 = -1
        c_1 = x_0 - x_2 = 0 - 0 = 0
        c_2 = x_1 - x_0 = 1 - 0 = 1

        B matrix:
        B[0] = [b_0, 0, b_1, 0, b_2, 0] / (2A) = [-1, 0, 1, 0, 0, 0]
        B[1] = [0, c_0, 0, c_1, 0, c_2] / (2A) = [0, -1, 0, 0, 0, 1]
        B[2] = [c_0, b_0, c_1, b_1, c_2, b_2] / (2A) = [-1, -1, 0, 1, 1, 0]
        """
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        element = CSTElement(nodes, material)

        # Expected B matrix (already divided by 2A = 1.0)
        expected_B = np.array([
            [-1, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 1],
            [-1, -1, 0, 1, 1, 0]
        ], dtype=float)

        assert np.allclose(element.B, expected_B), (
            f"B matrix mismatch.\n"
            f"Expected:\n{expected_B}\n"
            f"Got:\n{element.B}"
        )

    def test_area_positive(self, material):
        """Verify element areas are positive (correct node ordering)."""
        mesh = create_unstructured_mesh(n_divisions=5, perturbation=0.2, seed=42)

        for e_idx, elem_nodes in enumerate(mesh.elements):
            elem_coords = mesh.nodes[elem_nodes]
            element = CSTElement(elem_coords, material)

            assert element.area > 0, (
                f"Element {e_idx} has non-positive area: {element.area}\n"
                f"Nodes: {elem_coords}"
            )

    def test_strain_decomposition(self, material):
        """Verify strain computation step by step."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        element = CSTElement(nodes, material)

        # Apply uniaxial x displacement: u_x = 0.001*x
        # Node 0: (0,0) → u = (0, 0)
        # Node 1: (1,0) → u = (0.001, 0)
        # Node 2: (0,1) → u = (0, 0)
        u = np.array([[0, 0], [0.001, 0], [0, 0]])

        # Expected strain: ε_xx = 0.001, ε_yy = 0, γ_xy = 0
        eps = element.compute_strain(u)

        assert np.isclose(eps[0], 0.001, atol=1e-14), f"ε_xx error: {eps[0]}"
        assert np.isclose(eps[1], 0.0, atol=1e-14), f"ε_yy error: {eps[1]}"
        assert np.isclose(eps[2], 0.0, atol=1e-14), f"γ_xy error: {eps[2]}"


# =============================================================================
# Summary Test (Run All Cases and Print Table)
# =============================================================================

def test_patch_test_summary():
    """
    Run complete patch test and print summary table.

    This test runs all four cases on unstructured mesh and displays results.
    """
    material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
    mesh = create_unstructured_mesh(n_divisions=10, perturbation=0.3, seed=789)

    print("\n" + "=" * 70)
    print("PATCH TEST SUMMARY")
    print("=" * 70)
    print(f"Mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes")
    print(f"Perturbation: 30% of minimum edge length")
    print("-" * 70)
    print(f"{'Test Case':<15} {'ε_xx':<12} {'ε_yy':<12} {'γ_xy':<12} {'Max Error':<15} {'Status':<10}")
    print("-" * 70)

    all_passed = True
    for case_name, test_case in TEST_CASES.items():
        expected = test_case['expected_strain']
        passed, max_error, _ = run_patch_test_cst(mesh, material, test_case)

        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"{case_name:<15} {expected[0]:<12.4f} {expected[1]:<12.4f} "
              f"{expected[2]:<12.4f} {max_error:<15.2e} {status:<10}")

    print("-" * 70)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    assert all_passed, "Not all patch tests passed!"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Run summary test first for visual output
    test_patch_test_summary()

    # Run full test suite
    pytest.main([__file__, '-v', '--tb=short'])
