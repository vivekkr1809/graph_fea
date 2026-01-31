# GraFEA Phase-Field Framework

Edge-based phase-field fracture framework using Graph-based Finite Element Analysis (GraFEA).

## Overview

This framework implements a novel approach to phase-field fracture mechanics where:
- Damage is defined on mesh edges rather than at integration points
- Strain energy is computed using edge-based formulations
- The graph Laplacian provides regularization for damage evolution

## Key Features

- **Edge-based damage**: Damage variables are associated with mesh edges
- **Spectral tension-compression split**: Only tensile strain energy drives damage
- **Graph Laplacian regularization**: Smooth damage fields via edge graph
- **Staggered solver**: Robust alternating minimization algorithm

## Installation

```bash
# Clone the repository
git clone https://github.com/vivekkr1809/graph_fea.git
cd graph_fea/grafea_phasefield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from mesh import create_rectangle_mesh, EdgeGraph
from elements import GraFEAElement
from physics import IsotropicMaterial
from solvers import StaggeredSolver, SolverConfig

# Create mesh
mesh = create_rectangle_mesh(1.0, 1.0, 10, 10)

# Material properties
material = IsotropicMaterial(E=210e9, nu=0.3, Gc=2700, l0=0.05)

# Create elements
elements = [GraFEAElement(mesh.nodes[mesh.elements[e]], material)
            for e in range(mesh.n_elements)]

# Create edge graph for damage regularization
edge_graph = EdgeGraph(mesh)

# Setup solver
solver = StaggeredSolver(mesh, elements, material, edge_graph)

# Define boundary conditions and run
# ... (see examples for complete usage)
```

## Documentation

Full documentation is available in the `../docs/` directory and can be built using Sphinx.

### Building Documentation

```bash
cd ../docs
pip install -r requirements.txt
make html
```

Then open `_build/html/index.html` in your browser.

### Documentation Contents

- **Getting Started**: Installation guide and quick start tutorial
- **User Guide**: Detailed usage instructions and best practices
- **Theory**: Mathematical background of the edge-based phase-field formulation
- **Tutorials**: Step-by-step examples including simple tension and advanced usage
- **API Reference**: Complete documentation for all modules

## Project Structure

```
grafea_phasefield/
├── src/
│   ├── mesh/          # Mesh and edge graph data structures
│   ├── elements/      # CST and GraFEA element implementations
│   ├── physics/       # Material, damage, tension split, surface energy
│   ├── assembly/      # Global matrix assembly and boundary conditions
│   ├── solvers/       # Staggered solver
│   └── postprocess/   # Visualization and energy tracking
├── tests/             # Test suite
├── examples/          # Example scripts
└── docs/              # Documentation (in parent directory)
```

## Running Tests

```bash
pytest tests/ -v
```

## Theory

The framework is based on the following key ideas:

1. **Edge-based strain energy**: For a triangle with edge strains ε_edge:
   ```
   ψ = ½ ε_edge^T A ε_edge
   ```
   where A = T^(-T) C T^(-1) transforms the constitutive matrix to edge space.

2. **Degraded energy with damage**:
   ```
   ψ_d = ½ Σ_ij A_ij (1-d_i)(1-d_j) ε_i^+ ε_j^+ + ½ Σ_ij A_ij ε_i^- ε_j^-
   ```

3. **Surface energy via graph Laplacian**:
   ```
   E_frac = Σ_i (Gc/2l0) d_i² ω_i + (Gc l0/4) Σ_ij w_ij (d_j - d_i)²
   ```

## License

MIT License

## References

- Reddy & Srinivasa: Graph-based Finite Element Analysis
- Miehe et al. (2010): Thermodynamically consistent phase-field models
