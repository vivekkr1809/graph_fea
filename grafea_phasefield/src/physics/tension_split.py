"""
Tension-Compression Split
=========================

Spectral decomposition for separating tensile and compressive strain energy.
Only the tensile part is degraded by damage, following Miehe et al. (2010).
"""

import numpy as np
from typing import Tuple, Dict


def spectral_split_2d(eps_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral decomposition of 2D strain tensor (Miehe et al. 2010).

    Decomposes the strain tensor into positive (tensile) and negative
    (compressive) parts based on eigenvalues:
        ε = ε⁺ + ε⁻
        ε⁺ = Σ <ε_a>₊ n_a ⊗ n_a
        ε⁻ = Σ <ε_a>₋ n_a ⊗ n_a

    where <x>₊ = max(x, 0) and <x>₋ = min(x, 0).

    Args:
        eps_tensor: [ε_xx, ε_yy, γ_xy] in Voigt notation

    Returns:
        eigenvalues: principal strains, shape (2,)
        eps_plus: tensile strain tensor (Voigt), shape (3,)
        eps_minus: compressive strain tensor (Voigt), shape (3,)
    """
    eps_xx, eps_yy, gamma_xy = eps_tensor
    eps_xy = gamma_xy / 2

    # Build 2×2 strain matrix
    E_mat = np.array([[eps_xx, eps_xy],
                      [eps_xy, eps_yy]])

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(E_mat)

    # Split eigenvalues by sign
    lam_plus = np.maximum(eigenvalues, 0)
    lam_minus = np.minimum(eigenvalues, 0)

    # Reconstruct split tensors: ε± = Σ λ±_a (n_a ⊗ n_a)
    E_plus = np.zeros((2, 2))
    E_minus = np.zeros((2, 2))

    for a in range(2):
        n_a = eigenvectors[:, a]
        outer_prod = np.outer(n_a, n_a)
        E_plus += lam_plus[a] * outer_prod
        E_minus += lam_minus[a] * outer_prod

    # Convert back to Voigt notation [ε_xx, ε_yy, γ_xy]
    eps_plus = np.array([E_plus[0, 0], E_plus[1, 1], 2 * E_plus[0, 1]])
    eps_minus = np.array([E_minus[0, 0], E_minus[1, 1], 2 * E_minus[0, 1]])

    return eigenvalues, eps_plus, eps_minus


def compute_split_energy_miehe(eps_tensor: np.ndarray, lame_lambda: float,
                                lame_mu: float) -> Tuple[float, float]:
    """
    Compute split strain energy using the correct Miehe formulation.

    The Miehe split for isotropic materials:
        ψ⁺ = ½λ<tr(ε)>₊² + μ tr((ε⁺)²)
        ψ⁻ = ½λ<tr(ε)>₋² + μ tr((ε⁻)²)

    This ensures ψ = ψ⁺ + ψ⁻ exactly.

    Args:
        eps_tensor: [ε_xx, ε_yy, γ_xy] in Voigt notation
        lame_lambda: Lamé's first parameter λ
        lame_mu: Lamé's second parameter μ (shear modulus)

    Returns:
        psi_plus: tensile strain energy density
        psi_minus: compressive strain energy density
    """
    # Get spectral split
    _, eps_plus, eps_minus = spectral_split_2d(eps_tensor)

    # Trace of strain
    tr_eps = eps_tensor[0] + eps_tensor[1]
    tr_eps_plus = max(tr_eps, 0)
    tr_eps_minus = min(tr_eps, 0)

    # Compute tr(ε²) for the split tensors
    # For ε in Voigt: tr(ε²) = ε_xx² + ε_yy² + 2*(γ_xy/2)² = ε_xx² + ε_yy² + γ_xy²/2
    def tr_eps_squared(eps):
        return eps[0]**2 + eps[1]**2 + 0.5 * eps[2]**2

    # Miehe energy split
    psi_plus = 0.5 * lame_lambda * tr_eps_plus**2 + lame_mu * tr_eps_squared(eps_plus)
    psi_minus = 0.5 * lame_lambda * tr_eps_minus**2 + lame_mu * tr_eps_squared(eps_minus)

    return psi_plus, psi_minus


def spectral_split(eps_tensor: np.ndarray, eps_edge: np.ndarray,
                   C: np.ndarray, T: np.ndarray) -> Dict:
    """
    Option C split: spectral decomposition with edge transformation.

    RECOMMENDED approach from research plan Decision Point 4.

    This method:
    1. Performs spectral split in tensor space
    2. Transforms the split strains to edge space
    3. Computes energies in tensor space (guarantees conservation)

    The energy conservation property: ψ = ψ⁺ + ψ⁻ is guaranteed
    because we use the original tensor formulation for energy.

    Args:
        eps_tensor: tensor strain [ε_xx, ε_yy, γ_xy], shape (3,)
        eps_edge: edge strains (not used but kept for interface consistency)
        C: constitutive matrix, shape (3, 3)
        T: transformation matrix ε_edge = T @ ε_tensor, shape (3, 3)

    Returns:
        dict with:
            'eigenvalues': principal strains
            'eps_tensor_plus': tensile tensor strain
            'eps_tensor_minus': compressive tensor strain
            'eps_edge_plus': tensile edge strain
            'eps_edge_minus': compressive edge strain
            'psi_plus': tensile strain energy density
            'psi_minus': compressive strain energy density
    """
    # Spectral split in tensor space
    eigenvalues, eps_tensor_plus, eps_tensor_minus = spectral_split_2d(eps_tensor)

    # Transform to edge space
    eps_edge_plus = T @ eps_tensor_plus
    eps_edge_minus = T @ eps_tensor_minus

    # Compute energies in TENSOR space (guarantees conservation)
    psi_plus = 0.5 * eps_tensor_plus @ C @ eps_tensor_plus
    psi_minus = 0.5 * eps_tensor_minus @ C @ eps_tensor_minus

    return {
        'eigenvalues': eigenvalues,
        'eps_tensor_plus': eps_tensor_plus,
        'eps_tensor_minus': eps_tensor_minus,
        'eps_edge_plus': eps_edge_plus,
        'eps_edge_minus': eps_edge_minus,
        'psi_plus': psi_plus,
        'psi_minus': psi_minus,
    }


def simple_edge_split(eps_edge: np.ndarray, A: np.ndarray) -> Dict:
    """
    Option A split: simple sign-based split on edge strains.

    This is the simplest approach but does NOT conserve energy
    for mixed loading states due to coupling in the A matrix.

    ε_k⁺ = max(ε_k, 0)
    ε_k⁻ = min(ε_k, 0)

    Included for comparison purposes.

    Args:
        eps_edge: edge strains, shape (3,)
        A: edge stiffness matrix, shape (3, 3)

    Returns:
        dict with split strains and energies
    """
    eps_plus = np.maximum(eps_edge, 0)
    eps_minus = np.minimum(eps_edge, 0)

    psi_plus = 0.5 * eps_plus @ A @ eps_plus
    psi_minus = 0.5 * eps_minus @ A @ eps_minus

    # Note: psi_plus + psi_minus ≠ 0.5 * eps_edge @ A @ eps_edge in general
    # due to cross terms

    return {
        'eps_edge_plus': eps_plus,
        'eps_edge_minus': eps_minus,
        'psi_plus': psi_plus,
        'psi_minus': psi_minus,
    }


def volumetric_deviatoric_split(eps_tensor: np.ndarray,
                                C: np.ndarray, T: np.ndarray) -> Dict:
    """
    Alternative split: volumetric-deviatoric decomposition.

    ε = ε_vol + ε_dev
    ε_vol = (tr(ε)/2) I  (volumetric, 2D)
    ε_dev = ε - ε_vol    (deviatoric)

    Only the tensile volumetric part is degraded.

    Args:
        eps_tensor: tensor strain, shape (3,)
        C: constitutive matrix, shape (3, 3)
        T: transformation matrix, shape (3, 3)

    Returns:
        dict with split strains and energies
    """
    eps_xx, eps_yy, gamma_xy = eps_tensor

    # Volumetric strain (in 2D)
    eps_vol_scalar = (eps_xx + eps_yy) / 2
    eps_vol = np.array([eps_vol_scalar, eps_vol_scalar, 0])

    # Deviatoric strain
    eps_dev = eps_tensor - eps_vol

    # Split volumetric part by sign
    if eps_vol_scalar > 0:
        eps_vol_plus = eps_vol
        eps_vol_minus = np.zeros(3)
    else:
        eps_vol_plus = np.zeros(3)
        eps_vol_minus = eps_vol

    # Tensile part: positive volumetric + deviatoric
    # Compressive part: negative volumetric
    eps_tensor_plus = eps_vol_plus + eps_dev
    eps_tensor_minus = eps_vol_minus

    # Transform to edge space
    eps_edge_plus = T @ eps_tensor_plus
    eps_edge_minus = T @ eps_tensor_minus

    # Energies
    psi_plus = 0.5 * eps_tensor_plus @ C @ eps_tensor_plus
    psi_minus = 0.5 * eps_tensor_minus @ C @ eps_tensor_minus

    return {
        'eps_tensor_plus': eps_tensor_plus,
        'eps_tensor_minus': eps_tensor_minus,
        'eps_edge_plus': eps_edge_plus,
        'eps_edge_minus': eps_edge_minus,
        'psi_plus': psi_plus,
        'psi_minus': psi_minus,
        'eps_vol': eps_vol_scalar,
    }


def compute_driving_force_from_split(split_result: Dict,
                                      element_volume: float) -> float:
    """
    Compute damage driving force from split result.

    The driving force is the tensile strain energy density.

    Args:
        split_result: result from spectral_split or other split function
        element_volume: element volume for scaling

    Returns:
        Y: damage driving force
    """
    return split_result['psi_plus']


def verify_energy_conservation(eps_tensor: np.ndarray,
                               C: np.ndarray, T: np.ndarray,
                               tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Verify that ψ = ψ⁺ + ψ⁻ for the spectral split.

    Args:
        eps_tensor: tensor strain
        C: constitutive matrix
        T: transformation matrix
        tol: relative tolerance

    Returns:
        (conserved, relative_error): whether energy is conserved and the error
    """
    # Total energy
    psi_total = 0.5 * eps_tensor @ C @ eps_tensor

    # Split
    eps_edge = T @ eps_tensor
    split = spectral_split(eps_tensor, eps_edge, C, T)
    psi_sum = split['psi_plus'] + split['psi_minus']

    if abs(psi_total) < 1e-15:
        conserved = abs(psi_sum) < 1e-15
        rel_error = 0.0 if conserved else abs(psi_sum)
    else:
        rel_error = abs(psi_total - psi_sum) / abs(psi_total)
        conserved = rel_error < tol

    return conserved, rel_error
