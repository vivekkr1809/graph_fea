"""
Energy Tracking
===============

Track and analyze energy quantities during simulation.
"""

import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from solvers.staggered_solver import LoadStep


@dataclass
class EnergyRecord:
    """Record of energy quantities at a single time step."""
    step: int
    load_factor: float
    strain_energy: float
    surface_energy: float
    external_work: float
    kinetic_energy: float = 0.0  # For dynamic problems

    @property
    def total_energy(self) -> float:
        """Total potential energy (strain + surface)."""
        return self.strain_energy + self.surface_energy

    @property
    def dissipation(self) -> float:
        """Energy balance: external work - total energy."""
        return self.external_work - self.total_energy


class EnergyTracker:
    """
    Track energy quantities over the simulation.

    Provides tools for:
    - Recording energy at each step
    - Computing energy rates and derivatives
    - Checking energy balance
    - Exporting data
    """

    def __init__(self):
        """Initialize energy tracker."""
        self.records: List[EnergyRecord] = []

    def add_record(self, step: int, load_factor: float,
                   strain_energy: float, surface_energy: float,
                   external_work: float, kinetic_energy: float = 0.0) -> None:
        """
        Add a new energy record.

        Args:
            step: load step index
            load_factor: current load factor
            strain_energy: elastic strain energy
            surface_energy: fracture surface energy
            external_work: work done by external forces
            kinetic_energy: kinetic energy (for dynamic)
        """
        record = EnergyRecord(
            step=step,
            load_factor=load_factor,
            strain_energy=strain_energy,
            surface_energy=surface_energy,
            external_work=external_work,
            kinetic_energy=kinetic_energy
        )
        self.records.append(record)

    def from_results(self, results: List['LoadStep']) -> None:
        """
        Populate from solver results.

        Args:
            results: list of LoadStep instances
        """
        self.records = []
        for result in results:
            self.add_record(
                step=result.step,
                load_factor=result.load_factor,
                strain_energy=result.strain_energy,
                surface_energy=result.surface_energy,
                external_work=result.external_work
            )

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get energy quantities as numpy arrays.

        Returns:
            Dictionary with keys: 'step', 'load', 'strain', 'surface',
            'total', 'work', 'dissipation'
        """
        n = len(self.records)
        if n == 0:
            return {key: np.array([]) for key in
                    ['step', 'load', 'strain', 'surface', 'total', 'work', 'dissipation']}

        return {
            'step': np.array([r.step for r in self.records]),
            'load': np.array([r.load_factor for r in self.records]),
            'strain': np.array([r.strain_energy for r in self.records]),
            'surface': np.array([r.surface_energy for r in self.records]),
            'total': np.array([r.total_energy for r in self.records]),
            'work': np.array([r.external_work for r in self.records]),
            'dissipation': np.array([r.dissipation for r in self.records]),
        }

    def compute_rates(self) -> Dict[str, np.ndarray]:
        """
        Compute rates of change of energy quantities.

        Uses finite differences with respect to load factor.

        Returns:
            Dictionary with rate arrays
        """
        arrays = self.get_arrays()
        n = len(self.records)

        if n < 2:
            return {f'd{key}': np.array([]) for key in
                    ['strain', 'surface', 'total', 'work']}

        d_load = np.diff(arrays['load'])
        d_load[d_load == 0] = 1e-15  # Avoid division by zero

        return {
            'd_strain': np.diff(arrays['strain']) / d_load,
            'd_surface': np.diff(arrays['surface']) / d_load,
            'd_total': np.diff(arrays['total']) / d_load,
            'd_work': np.diff(arrays['work']) / d_load,
        }

    def check_energy_balance(self, tol: float = 0.1) -> Dict:
        """
        Check energy balance.

        For quasi-static fracture:
        External work = Strain energy + Surface energy (fracture dissipation)

        Args:
            tol: tolerance for balance check (relative)

        Returns:
            Dictionary with balance information
        """
        arrays = self.get_arrays()

        if len(self.records) == 0:
            return {'balanced': True, 'max_error': 0.0}

        # Compute imbalance
        imbalance = arrays['work'] - arrays['total']

        # Relative error
        scale = np.maximum(np.abs(arrays['work']), np.abs(arrays['total']))
        scale[scale < 1e-10] = 1.0
        rel_error = np.abs(imbalance) / scale

        balanced = np.all(rel_error < tol)

        return {
            'balanced': balanced,
            'max_error': np.max(rel_error),
            'mean_error': np.mean(rel_error),
            'imbalance': imbalance,
            'rel_error': rel_error,
        }

    def find_peak_load(self) -> Optional[Dict]:
        """
        Find the peak load (maximum external work rate).

        This typically corresponds to crack initiation.

        Returns:
            Dictionary with peak info or None if not found
        """
        if len(self.records) < 2:
            return None

        rates = self.compute_rates()
        d_work = rates['d_work']

        # Peak is where derivative changes from positive to negative
        # or maximum value
        peak_idx = np.argmax(d_work)

        arrays = self.get_arrays()
        return {
            'step': int(arrays['step'][peak_idx]),
            'load_factor': arrays['load'][peak_idx],
            'strain_energy': arrays['strain'][peak_idx],
            'surface_energy': arrays['surface'][peak_idx],
        }

    def find_fracture_initiation(self, threshold: float = 0.01) -> Optional[int]:
        """
        Find the step where fracture initiates.

        Defined as when surface energy starts to increase significantly.

        Args:
            threshold: relative increase threshold

        Returns:
            Step index or None
        """
        arrays = self.get_arrays()
        surface = arrays['surface']

        if len(surface) < 2:
            return None

        # Find first step where surface energy increases above threshold
        for i in range(1, len(surface)):
            if surface[i] > threshold * np.max(surface[1:]):
                return i

        return None

    def get_summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary values
        """
        if len(self.records) == 0:
            return {}

        arrays = self.get_arrays()

        return {
            'n_steps': len(self.records),
            'final_strain_energy': arrays['strain'][-1],
            'final_surface_energy': arrays['surface'][-1],
            'max_strain_energy': np.max(arrays['strain']),
            'max_surface_energy': np.max(arrays['surface']),
            'total_work': arrays['work'][-1],
        }

    def to_csv(self, filename: str) -> None:
        """
        Export energy data to CSV file.

        Args:
            filename: output filename
        """
        arrays = self.get_arrays()
        n = len(self.records)

        with open(filename, 'w') as f:
            f.write('step,load_factor,strain_energy,surface_energy,total_energy,external_work,dissipation\n')
            for i in range(n):
                f.write(f"{arrays['step'][i]},{arrays['load'][i]:.10e},"
                        f"{arrays['strain'][i]:.10e},{arrays['surface'][i]:.10e},"
                        f"{arrays['total'][i]:.10e},{arrays['work'][i]:.10e},"
                        f"{arrays['dissipation'][i]:.10e}\n")

    def plot(self, ax=None, normalize: bool = False):
        """
        Plot energy evolution.

        Args:
            ax: matplotlib axes
            normalize: whether to normalize energies
        """
        from .visualization import plot_energy_evolution
        # Convert to LoadStep-like objects for compatibility
        class FakeLoadStep:
            def __init__(self, record):
                self.step = record.step
                self.strain_energy = record.strain_energy
                self.surface_energy = record.surface_energy

        fake_results = [FakeLoadStep(r) for r in self.records]
        return plot_energy_evolution(fake_results, ax=ax, normalize=normalize)


def compute_J_integral_approximation(strain_energy_rate: float,
                                      crack_length_rate: float) -> float:
    """
    Approximate J-integral from energy release rate.

    J = -dΠ/da = -d(U - W)/da

    where U is strain energy, W is external work, a is crack length.

    Args:
        strain_energy_rate: dU/dt (rate of strain energy change)
        crack_length_rate: da/dt (crack growth rate)

    Returns:
        J: approximate J-integral value
    """
    if abs(crack_length_rate) < 1e-15:
        return 0.0

    return -strain_energy_rate / crack_length_rate


def estimate_crack_length(damage: np.ndarray,
                          edge_lengths: np.ndarray,
                          threshold: float = 0.5) -> float:
    """
    Estimate crack length from damage field.

    Args:
        damage: edge damage values
        edge_lengths: lengths of all edges
        threshold: damage threshold for "cracked"

    Returns:
        Estimated crack length
    """
    cracked = damage > threshold
    return np.sum(edge_lengths[cracked] * damage[cracked])
