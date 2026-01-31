Theoretical Background
======================

This section describes the mathematical foundations of the GraFEA Phase-Field Framework.

Overview
--------

The framework combines three key concepts:

1. **Graph-based Finite Element Analysis (GraFEA)**: Strain energy computed using edge-based formulations
2. **Phase-field fracture mechanics**: Diffuse crack representation with damage variable
3. **Tension-compression split**: Only tensile energy drives damage evolution

Standard Phase-Field Model
--------------------------

In standard phase-field fracture, the total energy functional is:

.. math::

   \Pi(u, d) = \int_\Omega \psi(u, d) \, d\Omega + \int_\Omega \gamma(d, \nabla d) \, d\Omega - W_{\text{ext}}

where:

- :math:`u` is the displacement field
- :math:`d \in [0, 1]` is the damage field (0 = intact, 1 = fully broken)
- :math:`\psi(u, d)` is the degraded strain energy density
- :math:`\gamma(d, \nabla d)` is the crack surface energy density
- :math:`W_{\text{ext}}` is the work done by external forces

Degradation Function
~~~~~~~~~~~~~~~~~~~~

The strain energy is degraded by a degradation function :math:`g(d)`:

.. math::

   \psi(u, d) = g(d) \psi_0(u)

Common choices for the degradation function:

- Quadratic: :math:`g(d) = (1-d)^2`
- Cubic: :math:`g(d) = (1-d)^3`

Both satisfy :math:`g(0) = 1` (intact), :math:`g(1) = 0` (fully broken), and :math:`g'(d) < 0`.

Crack Surface Energy
~~~~~~~~~~~~~~~~~~~~

The AT2 (Ambrosio-Tortorelli) crack surface energy is:

.. math::

   \gamma(d, \nabla d) = \frac{G_c}{2l_0} d^2 + \frac{G_c l_0}{2} |\nabla d|^2

where:

- :math:`G_c` is the critical energy release rate [J/m²]
- :math:`l_0` is the regularization length scale [m]

Edge-Based Formulation
----------------------

In GraFEA, the strain energy is reformulated in terms of edge quantities.

Tensor to Edge Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a triangular element with edges :math:`i = 1, 2, 3`, the strain along each edge is:

.. math::

   \varepsilon_i = \hat{e}_i^T \varepsilon \hat{e}_i

where :math:`\hat{e}_i` is the unit vector along edge :math:`i` and :math:`\varepsilon` is the strain tensor.

This can be written as a linear transformation:

.. math::

   \boldsymbol{\varepsilon}_{\text{edge}} = T \boldsymbol{\varepsilon}_{\text{Voigt}}

where :math:`T` is the :math:`3 \times 3` transformation matrix built from edge orientations.

Edge-Based Strain Energy
~~~~~~~~~~~~~~~~~~~~~~~~

The strain energy density in terms of edge strains is:

.. math::

   \psi_0 = \frac{1}{2} \boldsymbol{\varepsilon}_{\text{edge}}^T A \boldsymbol{\varepsilon}_{\text{edge}}

where the edge stiffness matrix is:

.. math::

   A = T^{-T} C T^{-1}

with :math:`C` being the constitutive matrix in Voigt notation.

This formulation is mathematically equivalent to the standard tensor formulation but
enables natural definition of edge-based damage.

Edge-Based Damage
~~~~~~~~~~~~~~~~~

In GraFEA, damage is defined on mesh edges rather than at integration points or nodes.
The degraded strain energy becomes:

.. math::

   \psi_d = \frac{1}{2} \sum_{i,j} A_{ij} (1-d_i)(1-d_j) \varepsilon_i^+ \varepsilon_j^+
          + \frac{1}{2} \sum_{i,j} A_{ij} \varepsilon_i^- \varepsilon_j^-

where:

- :math:`d_i` is the damage on edge :math:`i`
- :math:`\varepsilon_i^+` and :math:`\varepsilon_i^-` are tensile and compressive components

This geometric degradation naturally tracks crack surfaces through the mesh edges.

Tension-Compression Split
-------------------------

To prevent crack interpenetration under compression, we split the strain energy:

.. math::

   \psi_0 = \psi^+ + \psi^-

Only the tensile part :math:`\psi^+` is degraded by damage.

Spectral Decomposition
~~~~~~~~~~~~~~~~~~~~~~

The strain tensor is decomposed into positive and negative parts using eigenvalue decomposition:

.. math::

   \boldsymbol{\varepsilon} = \sum_{a=1}^{n} \varepsilon_a \mathbf{n}_a \otimes \mathbf{n}_a

where :math:`\varepsilon_a` are eigenvalues and :math:`\mathbf{n}_a` are eigenvectors.

The positive and negative parts are:

.. math::

   \boldsymbol{\varepsilon}^+ = \sum_a \langle \varepsilon_a \rangle_+ \mathbf{n}_a \otimes \mathbf{n}_a

   \boldsymbol{\varepsilon}^- = \sum_a \langle \varepsilon_a \rangle_- \mathbf{n}_a \otimes \mathbf{n}_a

where :math:`\langle x \rangle_\pm = (x \pm |x|)/2` are the positive/negative parts.

Miehe Formulation
~~~~~~~~~~~~~~~~~

Following Miehe et al. (2010), the split energy densities are:

.. math::

   \psi^+ = \frac{\lambda}{2} \langle \text{tr}(\boldsymbol{\varepsilon}) \rangle_+^2
          + \mu \, \text{tr}((\boldsymbol{\varepsilon}^+)^2)

   \psi^- = \frac{\lambda}{2} \langle \text{tr}(\boldsymbol{\varepsilon}) \rangle_-^2
          + \mu \, \text{tr}((\boldsymbol{\varepsilon}^-)^2)

This ensures energy conservation: :math:`\psi_0 = \psi^+ + \psi^-`.

Graph Laplacian Regularization
------------------------------

For edge-based damage, the gradient term in the surface energy is replaced by a
graph Laplacian term computed on the edge graph.

Edge Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~

The edge graph is constructed as follows:

- **Nodes**: Mesh edges become graph nodes
- **Edges**: Two graph nodes are connected if their corresponding mesh edges share an element
- **Weights**: Based on distance between edge midpoints or mesh geometry

Surface Energy on Edge Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The discrete surface energy becomes:

.. math::

   E_{\text{frac}} = \sum_i \frac{G_c}{2l_0} d_i^2 \omega_i
                   + \frac{G_c l_0}{4} \sum_{i,j} w_{ij} (d_j - d_i)^2

where:

- :math:`\omega_i` is the volume (area) associated with edge :math:`i`
- :math:`w_{ij}` are the weights of the graph Laplacian

Edge Volume Computation
~~~~~~~~~~~~~~~~~~~~~~~

The volume associated with edge :math:`i` is:

.. math::

   \omega_i = h \sum_{e \ni i} \frac{A^e}{3}

where :math:`h` is the thickness and the sum is over elements containing edge :math:`i`.

Staggered Solution Algorithm
----------------------------

The coupled displacement-damage problem is solved using a staggered (alternating minimization)
scheme:

1. **Mechanical step**: Fix :math:`d`, solve for :math:`u`

   .. math::

      \frac{\partial \Pi}{\partial u} = 0 \quad \Rightarrow \quad K(d) u = f

2. **Update history**: Compute driving force from current strains

   .. math::

      H^{n+1} = \max(H^n, \psi^+)

3. **Damage step**: Fix :math:`u`, solve for :math:`d`

   .. math::

      \frac{\partial \Pi}{\partial d} = 0

4. **Check convergence**: Repeat until changes in :math:`u` and :math:`d` are below tolerance

Damage Irreversibility
~~~~~~~~~~~~~~~~~~~~~~

Damage is constrained to be monotonically increasing through the history field:

.. math::

   \dot{d} \geq 0

This is enforced by using :math:`H = \max(\psi^+)` as the driving force rather than
the current :math:`\psi^+`.

Damage Evolution Equation
~~~~~~~~~~~~~~~~~~~~~~~~~

The damage field satisfies:

.. math::

   2(1-d)H - \frac{G_c}{l_0} d + G_c l_0 \nabla^2 d = 0

In discrete form with edge-based damage, this becomes a sparse linear system.

References
----------

1. Miehe, C., Welschinger, F., & Hofacker, M. (2010). Thermodynamically consistent
   phase-field models of fracture. *International Journal for Numerical Methods in Engineering*.

2. Reddy, J.N., & Srinivasa, A.R. (2015). Graph-based Finite Element Analysis.

3. Bourdin, B., Francfort, G.A., & Marigo, J.J. (2000). Numerical experiments in
   revisited brittle fracture. *Journal of the Mechanics and Physics of Solids*.

4. Ambrosio, L., & Tortorelli, V.M. (1990). Approximation of functionals depending
   on jumps by elliptic functionals via :math:`\Gamma`-convergence.
