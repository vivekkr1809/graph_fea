Installation
============

This guide covers the installation of the GraFEA Phase-Field Framework.

Requirements
------------

- Python 3.8 or higher
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4 (for visualization)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- pytest >= 6.0 (for running tests)
- meshio >= 5.0 (for mesh I/O)
- Sphinx (for building documentation)

Installation from Source
------------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/vivekkr1809/graph_fea.git
      cd graph_fea/grafea_phasefield

2. Create and activate a virtual environment (recommended):

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      # or: venv\Scripts\activate  # Windows

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

   This installs the package in editable mode along with development dependencies.

Verifying Installation
----------------------

After installation, verify everything is working by running the test suite:

.. code-block:: bash

   pytest tests/ -v

All tests should pass. You can also run a quick check by importing the main modules:

.. code-block:: python

   from mesh import create_rectangle_mesh, EdgeGraph
   from elements import GraFEAElement, CSTElement
   from physics import IsotropicMaterial
   from solvers import StaggeredSolver

   print("GraFEA Phase-Field Framework installed successfully!")

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

The documentation will be available at ``docs/_build/html/index.html``.

Troubleshooting
---------------

**ImportError: No module named 'mesh'**
   Make sure you installed the package with ``pip install -e .`` from the
   ``grafea_phasefield`` directory.

**NumPy/SciPy version conflicts**
   Try creating a fresh virtual environment and reinstalling.

**Tests failing**
   Ensure you have the latest version of the repository and all dependencies
   are correctly installed with ``pip install -e ".[dev]"``.
