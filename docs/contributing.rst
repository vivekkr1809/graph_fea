Contributing Guide
==================

We welcome contributions to the GraFEA Phase-Field Framework! This guide will help
you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally::

      git clone https://github.com/YOUR_USERNAME/graph_fea.git
      cd graph_fea/grafea_phasefield

3. Create a virtual environment and install in development mode::

      python -m venv venv
      source venv/bin/activate
      pip install -e ".[dev]"

4. Create a branch for your changes::

      git checkout -b feature/your-feature-name

Development Workflow
--------------------

Code Style
~~~~~~~~~~

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Use type hints where appropriate

Example docstring format::

   def compute_damage(strain, material):
       """Compute damage from strain state.

       Args:
           strain: Strain tensor in Voigt notation (3,)
           material: IsotropicMaterial instance

       Returns:
           float: Damage value in [0, 1]

       Raises:
           ValueError: If strain has wrong shape

       Example:
           >>> strain = np.array([0.001, 0.0, 0.0])
           >>> mat = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.01)
           >>> d = compute_damage(strain, mat)
       """

Running Tests
~~~~~~~~~~~~~

Always run tests before submitting::

   pytest tests/ -v

For coverage report::

   pytest tests/ --cov=src --cov-report=html

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Test documentation builds locally::

   cd docs
   pip install -r requirements.txt
   make html

View at ``_build/html/index.html``.

Submitting Changes
------------------

1. Commit your changes with clear messages::

      git add .
      git commit -m "Add feature: description of feature"

2. Push to your fork::

      git push origin feature/your-feature-name

3. Open a Pull Request on GitHub with:

   - Clear description of changes
   - Reference to any related issues
   - Test results

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow existing code style

Areas for Contribution
----------------------

We especially welcome contributions in:

- **New mesh generators**: Support for different geometries
- **Alternative solvers**: Monolithic solver, Newton-Raphson schemes
- **Performance**: Parallelization, optimized assembly
- **Visualization**: Enhanced plotting, VTK export
- **Documentation**: Tutorials, examples, API improvements
- **Testing**: Additional test cases, edge cases

Reporting Issues
----------------

When reporting bugs, please include:

1. Description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. System information (OS, Python version)
5. Minimal code example if possible

Questions
---------

For questions about contributing, please open an issue on GitHub or contact the
maintainers.

Thank you for contributing!
