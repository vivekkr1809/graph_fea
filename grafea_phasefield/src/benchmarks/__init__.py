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

__all__ = [
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
]
