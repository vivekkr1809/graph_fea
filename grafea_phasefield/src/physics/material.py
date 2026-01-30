"""
Material Models
===============

Material property classes for linear elastic materials.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IsotropicMaterial:
    """
    Isotropic linear elastic material with phase-field parameters.

    Attributes:
        E: Young's modulus [Pa]
        nu: Poisson's ratio [-]
        Gc: Critical energy release rate [J/m²]
        l0: Phase-field length scale [m]
    """
    E: float
    nu: float
    Gc: float = 2700.0  # Default for steel
    l0: float = 0.01    # Must be set based on mesh

    def __post_init__(self):
        """Validate material parameters."""
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if not -1 < self.nu < 0.5:
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {self.nu}")
        if self.Gc <= 0:
            raise ValueError(f"Gc must be positive, got {self.Gc}")
        if self.l0 <= 0:
            raise ValueError(f"l0 must be positive, got {self.l0}")

    @property
    def lame_lambda(self) -> float:
        """
        First Lamé parameter λ.

        λ = E·ν / ((1+ν)(1-2ν))
        """
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def lame_mu(self) -> float:
        """
        Second Lamé parameter μ (shear modulus).

        μ = E / (2(1+ν))
        """
        return self.E / (2 * (1 + self.nu))

    @property
    def bulk_modulus(self) -> float:
        """
        Bulk modulus K.

        K = E / (3(1-2ν))
        """
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def shear_modulus(self) -> float:
        """Alias for lame_mu."""
        return self.lame_mu

    def constitutive_matrix(self, plane: str = 'strain') -> np.ndarray:
        """
        Return 3×3 constitutive matrix C in Voigt notation.

        Relates stress to strain: σ = C ε
        where σ = [σ_xx, σ_yy, τ_xy]^T and ε = [ε_xx, ε_yy, γ_xy]^T

        Args:
            plane: 'strain' for plane strain, 'stress' for plane stress

        Returns:
            C: shape (3, 3), constitutive matrix
        """
        E, nu = self.E, self.nu

        if plane == 'strain':
            # Plane strain: ε_zz = 0
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2]
            ])

        elif plane == 'stress':
            # Plane stress: σ_zz = 0
            factor = E / (1 - nu ** 2)
            C = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])

        else:
            raise ValueError(f"Unknown plane condition: {plane}")

        return C

    def compliance_matrix(self, plane: str = 'strain') -> np.ndarray:
        """
        Return 3×3 compliance matrix S = C^{-1}.

        Relates strain to stress: ε = S σ

        Args:
            plane: 'strain' or 'stress'

        Returns:
            S: shape (3, 3), compliance matrix
        """
        return np.linalg.inv(self.constitutive_matrix(plane))

    def wave_speeds(self) -> tuple:
        """
        Compute elastic wave speeds (for reference density ρ=1).

        Returns:
            (c_p, c_s): P-wave and S-wave speeds
        """
        # For density ρ = 1
        lam, mu = self.lame_lambda, self.lame_mu
        c_p = np.sqrt(lam + 2 * mu)  # P-wave (compression)
        c_s = np.sqrt(mu)            # S-wave (shear)
        return c_p, c_s

    def critical_stress(self) -> float:
        """
        Theoretical critical stress for fracture initiation.

        σ_c = √(E·Gc / l0)  (1D approximation)

        Returns:
            σ_c: critical stress
        """
        return np.sqrt(self.E * self.Gc / self.l0)

    def critical_strain_energy_density(self) -> float:
        """
        Critical strain energy density for damage initiation.

        ψ_c ≈ Gc / (2·l0)

        Returns:
            ψ_c: critical strain energy density
        """
        return self.Gc / (2 * self.l0)


def create_steel_material(l0: float = 0.01) -> IsotropicMaterial:
    """
    Create material with typical steel properties.

    Args:
        l0: phase-field length scale

    Returns:
        IsotropicMaterial instance
    """
    return IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=l0)


def create_aluminum_material(l0: float = 0.01) -> IsotropicMaterial:
    """
    Create material with typical aluminum properties.

    Args:
        l0: phase-field length scale

    Returns:
        IsotropicMaterial instance
    """
    return IsotropicMaterial(E=70e9, nu=0.33, Gc=10000, l0=l0)


def create_glass_material(l0: float = 0.001) -> IsotropicMaterial:
    """
    Create material with typical glass properties.

    Args:
        l0: phase-field length scale

    Returns:
        IsotropicMaterial instance
    """
    return IsotropicMaterial(E=70e9, nu=0.22, Gc=8, l0=l0)
