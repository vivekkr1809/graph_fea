"""
Benchmarks
==========

Standard benchmark problems for validating phase-field fracture implementations.
"""

from .sent_benchmark import (
    SENT_PARAMS,
    generate_sent_mesh,
    create_precrack_damage,
    apply_sent_boundary_conditions,
    run_sent_benchmark,
    extract_crack_path,
    compare_with_literature,
    validate_sent_results,
    SENTBenchmark,
    quick_sent_test,
    very_quick_sent_test,
)

from .sens_benchmark import (
    SENS_PARAMS,
    generate_sens_mesh,
    apply_sens_boundary_conditions,
    create_sens_bc_function,
    run_sens_benchmark,
    validate_sens_results,
    compare_sens_sent,
    SENSBenchmark,
    SENSResults,
    quick_sens_test,
    very_quick_sens_test,
    compute_crack_angle,
    analyze_stress_state,
    validate_tension_compression_split,
)

__all__ = [
    # SENT exports
    'SENT_PARAMS',
    'generate_sent_mesh',
    'create_precrack_damage',
    'apply_sent_boundary_conditions',
    'run_sent_benchmark',
    'extract_crack_path',
    'compare_with_literature',
    'validate_sent_results',
    'SENTBenchmark',
    'quick_sent_test',
    'very_quick_sent_test',
    # SENS exports
    'SENS_PARAMS',
    'generate_sens_mesh',
    'apply_sens_boundary_conditions',
    'create_sens_bc_function',
    'run_sens_benchmark',
    'validate_sens_results',
    'compare_sens_sent',
    'SENSBenchmark',
    'SENSResults',
    'quick_sens_test',
    'very_quick_sens_test',
    'compute_crack_angle',
    'analyze_stress_state',
    'validate_tension_compression_split',
]
