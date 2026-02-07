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

from .tpb_benchmark import (
    TPB_PARAMS,
    generate_tpb_mesh,
    create_notch_damage,
    apply_tpb_boundary_conditions,
    create_tpb_bc_function,
    run_tpb_benchmark,
    validate_tpb_results,
    TPBBenchmark,
    TPBResults,
    quick_tpb_test,
    very_quick_tpb_test,
    compute_tpb_reaction_force,
    track_vertical_crack,
)

from .l_shaped_panel_benchmark import (
    L_PANEL_PARAMS,
    generate_l_panel_mesh,
    identify_l_panel_boundaries,
    apply_l_panel_boundary_conditions,
    create_l_panel_bc_function,
    run_l_panel_benchmark,
    validate_l_panel_results,
    compare_with_original_grafea,
    LPanelBenchmark,
    LPanelResults,
    quick_l_panel_test,
    very_quick_l_panel_test,
    check_nucleation,
    find_nucleation_location,
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
    # TPB exports
    'TPB_PARAMS',
    'generate_tpb_mesh',
    'create_notch_damage',
    'apply_tpb_boundary_conditions',
    'create_tpb_bc_function',
    'run_tpb_benchmark',
    'validate_tpb_results',
    'TPBBenchmark',
    'TPBResults',
    'quick_tpb_test',
    'very_quick_tpb_test',
    'compute_tpb_reaction_force',
    'track_vertical_crack',
    # L-Panel exports
    'L_PANEL_PARAMS',
    'generate_l_panel_mesh',
    'identify_l_panel_boundaries',
    'apply_l_panel_boundary_conditions',
    'create_l_panel_bc_function',
    'run_l_panel_benchmark',
    'validate_l_panel_results',
    'compare_with_original_grafea',
    'LPanelBenchmark',
    'LPanelResults',
    'quick_l_panel_test',
    'very_quick_l_panel_test',
    'check_nucleation',
    'find_nucleation_location',
]
