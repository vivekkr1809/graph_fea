Changelog
=========

All notable changes to the GraFEA Phase-Field Framework are documented here.

Version 0.1.0 (2024)
--------------------

Initial release of the GraFEA Phase-Field Framework.

Features
~~~~~~~~

- **Mesh Module**

  - ``TriangleMesh`` class for 2D triangular meshes
  - ``EdgeGraph`` class for damage regularization
  - Mesh generators: rectangle, square, notched specimens
  - Mesh refinement utilities

- **Elements Module**

  - ``CSTElement``: Standard Constant Strain Triangle
  - ``GraFEAElement``: Novel edge-based element formulation

- **Physics Module**

  - ``IsotropicMaterial`` dataclass with derived properties
  - Spectral tension-compression split
  - History field for damage irreversibility
  - Graph Laplacian surface energy

- **Assembly Module**

  - Global stiffness matrix assembly (damaged/undamaged)
  - Boundary condition utilities
  - Strain and energy computation

- **Solvers Module**

  - ``StaggeredSolver``: Alternating minimization algorithm
  - Configurable tolerances and iteration limits
  - ``LoadStep`` result dataclass

- **Postprocessing Module**

  - Mesh and field visualization
  - Energy evolution plotting
  - ``EnergyTracker`` for analysis

Documentation
~~~~~~~~~~~~~

- Sphinx-based documentation
- Theoretical background
- User guide with examples
- Full API reference
- Tutorials for common use cases

Testing
~~~~~~~

- Comprehensive test suite
- Unit tests for all modules
- Integration tests
- Patch test verification


Future Plans
------------

Planned features for future releases:

- 3D extension with tetrahedral elements
- Dynamic/quasi-static solver options
- Adaptive mesh refinement
- GPU acceleration
- Additional material models
