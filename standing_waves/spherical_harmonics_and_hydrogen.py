import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Spherical Harmonics and Hydrogen Orbitals")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from hydrogen import (
        HARTREE_EV,
        degeneracy,
        hydrogen_energy,
        hydrogen_energy_ev,
        node_counts,
        node_geometry,
        numerical_levels,
        quantum_numbers,
    )
    from coordinates_plots import (
        cartesian_cell_figure,
        length_factors_figure,
        polar_grid_figure,
        polar_point_figure,
        spherical_coordinates_figure,
        spherical_surfaces_figure,
    )
    from drum_fourier import TARGETS
    from complex_waves import complex_vs_real_figure
    from drum_fourier_plots import drum_fourier_figure, drum_fourier_modes_figure
    from hydrogen_plots import (
        central_levels_figure,
        coordinates_figure,
        drum_figure,
        drum_gallery_figure,
        drum_gif,
        fourier_circle_figure,
        fourier_sphere_figure,
        harmonic_figure,
        harmonic_gallery_figure,
        level_check_figure,
        node_anatomy_figure,
        node_placement_figure,
        orbital_figure,
        orbital_gallery_figure,
        pole_figure,
        radial_state_figure,
        string_figure,
        string_gif,
    )

    return (
        HARTREE_EV,
        TARGETS,
        complex_vs_real_figure,
        cartesian_cell_figure,
        central_levels_figure,
        length_factors_figure,
        drum_fourier_figure,
        drum_fourier_modes_figure,
        polar_grid_figure,
        polar_point_figure,
        spherical_coordinates_figure,
        spherical_surfaces_figure,
        coordinates_figure,
        degeneracy,
        drum_figure,
        drum_gallery_figure,
        drum_gif,
        fourier_circle_figure,
        fourier_sphere_figure,
        harmonic_figure,
        harmonic_gallery_figure,
        hydrogen_energy,
        hydrogen_energy_ev,
        level_check_figure,
        mo,
        node_anatomy_figure,
        node_counts,
        node_geometry,
        node_placement_figure,
        np,
        numerical_levels,
        orbital_figure,
        orbital_gallery_figure,
        plt,
        pole_figure,
        quantum_numbers,
        radial_state_figure,
        string_figure,
        string_gif,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Spherical Harmonics and Hydrogen Orbitals

    The previous notebook (`standing_waves_in_boxes.py`) showed that a wave confined to a box can only take certain shapes, and
    each shape has its own energy. An atom is also a confined wave, but the confinement is a pull toward the centre,
    and the natural shape is a sphere. This notebook finds the allowed waves of the hydrogen atom: first the waves on
    a sphere (the **spherical harmonics**), then the waves along the distance from the nucleus, and then it puts them
    together into the orbitals that chemists draw.

    One idea runs through the whole notebook: **the shape of a standing wave is set by where its nodes are and which
    way they point.** A string has nodes that are points. A drumhead has nodes that are lines, of two kinds: circles
    around the centre and diameters through it. An atom has nodes that are surfaces, of three kinds: spheres ($n_r$ of
    them), cones ($n_\vartheta$) and planes ($n_\varphi$). By section 6 the shape of every orbital can be read off by counting
    nodes of each kind.

    **Read first:** `standing_waves_in_boxes.py` (this folder), and `../quantization`, section 5, which states the hydrogen
    quantum numbers without deriving them. This notebook supplies the reasons, and the later notebooks (`from_atoms_to_molecules.py`,
    `basis_sets_from_base_states.py`, `hartree_fock_mean_field.py`, `../dft/`) use $n$, $\ell$, $m_\ell$ and the orbital shapes unchanged.

    **Contents**

    1. From a box to an atom: new coordinates
    2. Polar coordinates in 2D and 3D
    3. Nodes and their direction: string, drumhead, atom
    4. Standing waves on a sphere: the spherical harmonics
    5. The radial part: effective potential and energy levels
    6. Hydrogen orbitals, and reading an orbital from its nodes
    7. A short bridge to molecules

    **How much is derived.** The full separation of the Schrödinger equation is not repeated here. The equations
    that come out of it are stated, and each statement is checked numerically in `test_hydrogen.py` (the list is at
    the end of section 7). The box notebook measured energy in units of $E_1$; from here on units are atomic units, here and
    in every later notebook: lengths in bohr ($a_0 = 0.0529$ nm, so 1 bohr $=0.529$ Å), energies in hartree (1 hartree $=27.2$ eV), $\hbar=m_e=1$. Equations stay in these units; the figures show lengths in ångström.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Spherical coordinates**: the distance $r$ from the centre, the polar angle $\vartheta$ measured from the
          $z$ axis ($0$ at the north pole, $\pi$ at the south pole) and the azimuth $\varphi$ around the axis.
        - **Spherical harmonic** $Y_{\ell m_\ell}(\vartheta,\varphi)$: a standing wave on the surface of a sphere.
        - **Orbital**: the wave function of one electron, $\psi_{n\ell m_\ell}$ (also written $\psi_{n_r\ell m_\ell}$, since
          $n=n_r+\ell+1$). Its square $\lvert\psi\rvert^2$ is the
          probability density.
        - **Node**: a point, line or surface where the wave is zero. The wave changes sign across it.
        - **Radial node**: a sphere around the nucleus on which $\psi$ is zero; there are $n_r$ of them.
        - **Polar node**: a cone around the $z$ axis (or the flat plane $z=0$) on which $\psi$ is zero; there are
          $n_\vartheta$ of them.
        - **Azimuthal node**: a plane through the $z$ axis on which $\psi$ is zero; there are $n_\varphi=\lvert m_\ell\rvert$ of
          them. The polar and azimuthal nodes together are the **angular nodes**, $\ell=n_\vartheta+n_\varphi$ in number, and
          all the nodes are $N_{\text{nodes}}=n-1=n_r+n_\vartheta+n_\varphi$.
        - **Drumhead mode**: a standing wave of a circular membrane held fixed at its rim.
        - Hydrogen's lowest level is $-\tfrac12$ hartree.
        - We write $m_\ell$ for the whole number that counts turns around the axis, because the plain letter $m$ is
          used for mass.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 From a box to an atom: new coordinates

    In a box the three natural directions are $x$, $y$ and $z$, and the waves are products of three one-dimensional
    waves, one for each direction. The pull of the nucleus depends only on the distance $r$ from it, so the problem looks
    the same in every direction. The natural coordinates are then the distance $r$, the polar angle $\vartheta$ and
    the azimuth $\varphi$, shown in the figure.

    What carries over from the box is the method: write the wave as a product with one factor per coordinate, and
    demand that each factor fits its surroundings. For the atom the product has two parts,

    $$
    \psi(r,\vartheta,\varphi) = R(r)\;Y(\vartheta,\varphi),
    $$

    a **radial** wave $R$ that depends only on the distance, and an **angular** wave $Y$ that depends only on the direction,
    that is, on the position on a sphere around the nucleus. The two parts are linked by a single number. The angular
    wave contributes an energy that depends on how much it curves over the sphere; we will see that this number is
    $\ell(\ell+1)$, and it enters the radial equation as a repulsive term. Sections 2 to 6 follow the order of the contents above:
    coordinates, nodes, the waves on a sphere and their number $\ell(\ell+1)$, the radial equation with that number in it, and the
    product of the two parts.

    The kinetic energy in these coordinates, the **Laplacian** $\nabla^2$, is derived at the end of section 2. How the
    equation then splits into an angular and a radial part, what the effective potential is, and how the hydrogen
    energies $-1/(2n^2)$ come out, is derived in section 5.
    """)
    return


@app.cell
def _(coordinates_figure, plt):
    _fig = coordinates_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 Polar coordinates in 2D and 3D

    Section 1 named the coordinates $r$, $\vartheta$ and $\varphi$ without saying what they mean. This section does,
    first on a flat sheet (two numbers per point) and then in space (three numbers per point). The figures are written as
    slide-ready image files by `export_figures.py`.

    **Two dimensions: $(r,\varphi)$.** A point on a sheet can be described in two ways. The **Cartesian** description
    gives two distances, $x$ along a horizontal axis and $y$ along a vertical axis. The **polar** description gives
    the distance $r$ from the centre (the origin), and the angle $\varphi$ between the horizontal axis and the line from
    the centre to the point, measured counterclockwise. The two descriptions are linked by

    $$
    x = r\cos\varphi,\qquad y = r\sin\varphi,
    \qquad\text{and back:}\qquad
    r=\sqrt{x^2+y^2},\qquad \varphi=\operatorname{atan2}(y,x).
    $$

    The function $\operatorname{atan2}(y,x)$ is the angle of the point $(x,y)$, placed in the correct quarter of the
    plane; the plain ratio $y/x$ would not tell $(1,1)$ from $(-1,-1)$. The ranges are $r\ge 0$ and
    $0\le\varphi<2\pi$: the angle **wraps**, so $\varphi=2\pi$ is the same direction as $\varphi=0$, and at
    $r=0$ the angle means nothing. Move the sliders: the point is the same point in both descriptions.
    """)
    return


@app.cell
def _(mo):
    r_pt = mo.ui.slider(0.0, 2.0, step=0.1, value=1.6, label=r"$r$", show_value=True)
    phi_pt = mo.ui.slider(0, 355, step=5, value=40, label=r"$\varphi$ (degrees)", show_value=True)
    return phi_pt, r_pt


@app.cell
def _(mo, phi_pt, plt, polar_point_figure, r_pt):
    _fig = polar_point_figure(r_pt.value, phi_pt.value, slide=False)
    _out = mo.vstack([mo.hstack([r_pt, phi_pt], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plt, polar_grid_figure):
    _fig = polar_grid_figure(slide=False)
    _out = mo.vstack([
        mo.md(r"""
        **The coordinate grid.** Fix one coordinate and let the other vary, and a line appears. A line of constant $r$ is
        a **circle** around the centre; a line of constant $\varphi$ is a **ray** from the centre. Every point of the
        sheet is where one circle and one ray cross.
        """),
        _fig,
        mo.md(r"""
        This is why the drum of section 3 has the nodes it has. A node at constant $r$ is a circle, and a node at
        constant $\varphi$ is a diameter (two rays on opposite sides of the centre).
        """),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Three dimensions: $(r,\vartheta,\varphi)$.** In space a point needs three numbers. The Cartesian description
    uses $(x,y,z)$. The **spherical** description uses the distance $r\ge0$ from the centre, the **polar angle**
    $\vartheta$ between the $z$ axis and the line from the centre to the point ($0\le\vartheta\le\pi$, from the north
    pole to the south pole), and the **azimuth** $\varphi$ ($0\le\varphi<2\pi$), the angle around the $z$ axis of the
    shadow that the point casts on the $xy$ plane, measured from the $x$ axis. The conversions are

    $$
    x=r\sin\vartheta\cos\varphi,\qquad y=r\sin\vartheta\sin\varphi,\qquad z=r\cos\vartheta .
    $$

    The shadow on the $xy$ plane has length $r\sin\vartheta$, which is why $\sin\vartheta$ appears in $x$ and $y$; the
    height of the point is $z=r\cos\vartheta$. This notebook uses the **physics convention** for the names, matching the
    rest of the notebook: $\vartheta$ is the polar angle and $\varphi$ is the azimuth. Some books swap the two
    names, so check which one is meant before comparing formulas.
    """)
    return


@app.cell
def _(mo, plt, spherical_coordinates_figure):
    _fig = spherical_coordinates_figure(slide=False)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plt, spherical_surfaces_figure):
    _fig = spherical_surfaces_figure(slide=False)
    _out = mo.vstack([
        mo.md(r"""
        **The coordinate surfaces.** Fix one of the three coordinates and a surface appears, as the lines did in two
        dimensions:

        - constant $r$: a **sphere** around the centre;
        - constant $\vartheta$: a **cone** around the $z$ axis (for $\vartheta=\pi/2$ the cone is the flat $xy$ plane);
        - constant $\varphi$: a **half-plane** that starts at the $z$ axis (because $\varphi$ runs over $0\le\varphi<2\pi$ and
          $r\ge0$, a given $\varphi$ is one half-plane, and $\varphi+\pi$ is the half-plane on the opposite side).

        A **node** of $\cos(m_\ell\varphi)$ is more than one half-plane: it is the **full plane** through the $z$ axis, the
        two half-planes at $\varphi_0$ and $\varphi_0+\pi$ together (the lighter half in the figure). The reason: for a whole number $m_\ell$, $\cos(m_\ell(\varphi+\pi))=(-1)^{m_\ell}\cos(m_\ell\varphi)$, so a zero at
        $\varphi_0$ is also a zero at $\varphi_0+\pi$. For $m_\ell=1$, $\cos\varphi$ is proportional to $x$, and its node is the plane
        $x=0$, with $y>0$ on one side of the axis and $y<0$ on the other. In two dimensions the
        same node is a diameter, two opposite rays. (The figure shows the sphere with a quarter cut away, so that the axes
        stay visible, and the part of the plane below the floor lighter.)
        """),
        _fig,
        mo.md(r"""
        The surface of a sphere of fixed radius is labelled by the two angles $(\vartheta,\varphi)$ alone: a point on
        it is a direction. On that sphere, a ring of latitude is a line of constant $\vartheta$ (the cone meets the sphere
        in a circle), and a meridian is a line of constant $\varphi$. These two families of lines are the ones that
        section 4 uses to place the nodes of a standing wave on a sphere.
        """),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why use them.** Two reasons. First, a function of $r$ alone, such as the pull of a nucleus, looks the same in every
    direction: in polar coordinates it does not depend on the angles at all, while in $x,y,z$ it depends on all three.
    Second, nodes at fixed coordinates are simple: a circle, a sphere, a cone and a half-plane each have one coordinate
    fixed. Sections 3 to 6 use both facts.

    **Small areas and volumes.** To add up a probability one needs the size of a small patch. Take a small step $dr$ along a
    ray and a small step $d\varphi$ around the centre: the patch is nearly a rectangle with sides $dr$ and $r\,d\varphi$
    (an arc is the radius times the angle), so its area is
    $dA=r\,dr\,d\varphi$. In space, add a step $d\vartheta$, whose side is $r\,d\vartheta$; the azimuth step moves along a
    circle of latitude of radius $r\sin\vartheta$, so its side is $r\sin\vartheta\,d\varphi$, and the volume is

    $$
    dV=r^2\sin\vartheta\;dr\,d\vartheta\,d\varphi .
    $$

    Both factors are used, not derived any further, when the notebook says that an orbital is normalised.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Laplacian in spherical coordinates

    The kinetic energy of a wave is $-\frac{\hbar^2}{2m}\nabla^2\psi$, where $\hbar$ is the reduced Planck constant, $m$ the
    mass and $\nabla^2\psi$ (the **Laplacian**) measures how much the wave curves: in $x,y,z$ it is the sum of the second
    derivatives $\partial_x^2\psi+\partial_y^2\psi+\partial_z^2\psi$, with $\partial_x=\partial/\partial x$. In $r,\vartheta,\varphi$ the formula
    changes, because a small step in each coordinate has a **different length**. A step $dr$ is a length $dr$. A step
    $d\vartheta$ moves along an arc of radius $r$, so its length is $r\,d\vartheta$. A step $d\varphi$ moves along a circle of
    latitude of radius $r\sin\vartheta$, so its length is $r\sin\vartheta\,d\varphi$. Writing the length of a step as
    $h\,d(\text{coordinate})$, the factors $h$ are

    $$
    h_r=1,\qquad h_\vartheta=r,\qquad h_\varphi=r\sin\vartheta .
    $$

    The three directions are perpendicular, so the small cell in the figure is nearly a box, and its volume is
    $dV$ above.
    """)
    return


@app.cell
def _(cartesian_cell_figure, length_factors_figure, mo, plt):
    _fig1 = cartesian_cell_figure(slide=False)
    _fig2 = length_factors_figure(slide=False)
    _out = mo.vstack([
        mo.md("In $x,y,z$ the small cell is a plain box: its edges are $dx$, $dy$ and $dz$, so every $h=1$."),
        _fig1,
        mo.md("In $r,\\vartheta,\\varphi$ the cell is curved and its edges are $dr$, $r\\,d\\vartheta$ and $r\\sin\\vartheta\\,d\\varphi$."),
        _fig2,
    ])
    plt.close(_fig1)
    plt.close(_fig2)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The Laplacian is the **divergence of the gradient**. The gradient of $\psi$ points the way the wave rises fastest, and
    its component along direction $i$ is the change of $\psi$ per unit *length*, $\frac{1}{h_i}\partial_i\psi$. The divergence of
    a field $F$ is the net flow out of the cell divided by its volume. The flow out through the two faces
    perpendicular to direction $i$ is the change of (area of the face) $\times F_i$ across the cell, and the area of such a
    face is the product of the other two lengths, $h_jh_k\,dx_j\,dx_k$. Dividing by the volume gives the Laplacian
    (this step is the standard divergence formula, stated here and not proved):

    $$
    \nabla^2\psi=\frac{1}{h_rh_\vartheta h_\varphi}\Bigl[\partial_r\Bigl(\frac{h_rh_\vartheta h_\varphi}{h_r^{2}}\,\partial_r\psi\Bigr)
    \;+\;\partial_\vartheta\Bigl(\frac{h_rh_\vartheta h_\varphi}{h_\vartheta^{2}}\,\partial_\vartheta\psi\Bigr)
    \;+\;\partial_\varphi\Bigl(\frac{h_rh_\vartheta h_\varphi}{h_\varphi^{2}}\,\partial_\varphi\psi\Bigr)\Bigr].
    $$

    Put in $h_rh_\vartheta h_\varphi=r^2\sin\vartheta$. The three factors in front of the derivatives are $r^2\sin\vartheta$,
    $\sin\vartheta$ and $1/\sin\vartheta$, and the result is

    $$
    \nabla^2\psi=\frac{1}{r^2}\,\partial_r\bigl(r^2\,\partial_r\psi\bigr)
    \;+\;\frac{1}{r^2\sin\vartheta}\,\partial_\vartheta\bigl(\sin\vartheta\,\partial_\vartheta\psi\bigr)
    \;+\;\frac{1}{r^2\sin^2\vartheta}\,\partial_\varphi^{2}\psi .
    $$

    Expand the first term, $\frac{1}{r^2}\partial_r(r^2\partial_r\psi)=\partial_r^2\psi+\frac2r\partial_r\psi$, and collect the angular
    terms in one operator, $\Lambda$ ("lambda", the angular part of the Laplacian):

    $$
    \nabla^2=\partial_r^2+\frac{2}{r}\,\partial_r+\frac{1}{r^2}\,\Lambda,
    \qquad
    \Lambda=\frac{1}{\sin\vartheta}\,\partial_\vartheta\bigl(\sin\vartheta\,\partial_\vartheta\bigr)+\frac{1}{\sin^2\vartheta}\,\partial_\varphi^{2}.
    $$

    What each term means:

    - $\partial_r^2+\frac2r\partial_r$ is the curvature along $r$. The extra $\frac2r\partial_r$ comes from the area of a shell
      growing as $r^2$: a wave that moves outward spreads over larger and larger shells.
    - $\frac{1}{r^2}\Lambda$ is the curvature across the angles. The factor $1/r^2$ is there because a fixed angle covers a
      distance proportional to $r$, so the same pattern of angles is less curved at a larger radius: $\Lambda/r^2$ is the
      Laplacian on a sphere of radius $r$. The operator $\Lambda$ is the one of section 4, $-\Lambda Y_{\ell m_\ell}=\ell(\ell+1)Y_{\ell m_\ell}$.
    - The angular momentum squared is $\hat L^2=-\hbar^2\Lambda$. So the kinetic energy is a radial part plus
      $\hat L^2/(2mr^2)$, the cost of going round the centre.

    **The drum.** On the flat drum only two coordinates are needed, $h_r=1$ and $h_\varphi=r$, and the same steps give
    $\nabla^2=\partial_r^2+\frac1r\partial_r+\frac{1}{r^2}\partial_\varphi^2$. For a mode $R(r)\cos(m_\ell\varphi)$ the equation
    $-\nabla^2\psi=k^2\psi$ becomes $R''+\frac1rR'+\big(k^2-\frac{m_\ell^2}{r^2}\big)R=0$, which is the **Bessel equation**: its
    solutions $J_{\lvert m_\ell\rvert}(kr)$ are the waves in $r$ of section 3. **Checked in the tests:** both formulas
    against the Cartesian second derivatives of test functions such as $x^2+yz$ and $r^2\cos\vartheta$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 Nodes and their direction: string, drumhead, atom

    A **node** is a place where the wave is zero, and the wave changes sign when you cross it. A standing wave is
    pictured most simply by its nodes: they cut the region into pieces, and the wave is positive in one piece, negative
    in the next. Two facts about a node are worth keeping track of: **how many** there are, and **which direction
    the node cuts across**. We follow both from a string to a drumhead to an atom.

    **One dimension: a string.** A string has one direction, along its length $x$. A node is a point on the string, and
    it cuts across $x$. A wave with $n$ half-wavelengths has $N_\text{nodes}=n-1$ nodes inside the string. More nodes, more energy.
    """)
    return


@app.cell
def _(mo):
    n_str = mo.ui.slider(1, 8, step=1, value=4, label=r"$n$", show_value=True)
    return (n_str,)


@app.cell
def _(mo, n_str, plt, string_figure, string_gif):
    _fig = string_figure(n_str.value)
    _out = mo.vstack([n_str, _fig, mo.md("Moving, for one full period (the nodes stay fixed):"),
                      mo.image(src=string_gif(n_str.value))])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Two dimensions: a drumhead.** The previous notebook used a rectangular box for two dimensions. A drum is a
    better model for what follows, because a round membrane has the same coordinates as an atom seen from above: a
    distance $r$ from the centre and an angle $\varphi$ around it. The membrane is held fixed at its rim, and its
    vibration modes are products of a wave in $r$ and a wave in $\varphi$,

    $$
    \psi_{n\ell m_\ell}(r,\varphi) = R_{n\ell}(r)\times
    \begin{cases}\cos(m_\ell\varphi), & m_\ell\ge 0,\\[2pt] \sin(\lvert m_\ell\rvert\varphi), & m_\ell<0,\end{cases}
    \qquad R_{n\ell}(r)=J_\ell\!\big(j_{\ell,\,n_r+1}\,r\big),\qquad m_\ell=0,\pm1,\pm2,\dots
    $$

    where $J_\ell$ is a Bessel function (a wave in $r$ whose swings shrink toward the rim, like ripples spreading on
    water), and $j_{\ell,\,n_r+1}$ is its $(n_r+1)$-th zero, chosen so that the rim, $r=1$, is a node that is held
    fixed. We do not need the formula for $J$; the figure shows what it does. The names of the whole numbers are the
    ones that the atom will use:

    - $\ell=\lvert m_\ell\rvert$ is the number of **angular nodes**, the lines through the centre;
    - $n_r$ is the number of **radial nodes**, the circles around the centre (the fixed rim is not counted);
    - $n=n_r+\ell+1$, so that the total number of nodes is $N_\text{nodes}=n-1=\ell+n_r$.

    The angular factor is the ring wave of
    `../quantization`: it returns to itself after one turn only if $m_\ell$ is a whole number. The two signs of
    $m_\ell$ are the two real waves of the ring: $m_\ell>0$ is the cosine, and $m_\ell<0$ is the sine, the same shape
    turned by a quarter of the period $2\pi/\lvert m_\ell\rvert$. They have the same frequency, and
    $m_\ell=0$ is the same in every direction (a rotation-symmetric drum).

    The two factors give two **different kinds of node line**, and each cuts across a different direction:

    - $n_r$ **circles** around the centre. They cut across $r$, and they are the string's nodes, bent into circles.
    - $\ell=\lvert m_\ell\rvert$ **diameters** through the centre. They cut across $\varphi$: they come from the zeros of
      $\cos(m_\ell\varphi)$ or $\sin(\lvert m_\ell\rvert\varphi)$, which have $2\ell$ zeros around a
      full turn, and two opposite zeros make one diameter.

    In three dimensions the $\ell$ angular nodes split into $n_\varphi=\lvert m_\ell\rvert$ planes that contain the axis and
    $n_\vartheta=\ell-\lvert m_\ell\rvert$ cones. The drum is flat and has no cones, $n_\vartheta=0$, so all of its $\ell$ angular
    nodes contain the centre and $\ell=n_\varphi=\lvert m_\ell\rvert$.
    """)
    return


@app.cell
def _(mo):
    m_drum = mo.ui.slider(-6, 6, step=1, value=2, label=r"$m_\ell$ ($\ell=\lvert m_\ell\rvert$ diameters)", show_value=True)
    nr_drum = mo.ui.slider(0, 3, step=1, value=2, label=r"$n_r$ (circles)", show_value=True)
    return m_drum, nr_drum


@app.cell
def _(drum_figure, drum_gif, m_drum, mo, nr_drum, plt):
    _fig = drum_figure(m_drum.value, nr_drum.value)
    _out = mo.vstack([mo.hstack([m_drum, nr_drum], justify="start"), _fig,
                      mo.md("The same mode vibrating (colour is the sign of the mode, so the nodal circles and "
                            "diameters show as white seams that never move):"),
                      mo.image(src=drum_gif(m_drum.value, nr_drum.value))])
    plt.close(_fig)
    _out
    return


@app.cell
def _(drum_gallery_figure, mo, plt):
    _fig = drum_gallery_figure(3, 2)
    _out = mo.vstack([
        mo.md("All the drum modes with up to $n_r=2$ circles and $\\ell=3$ diameters. Each column has one $\\ell$, "
              "each row one $n_r$; the shape is the product of the two."),
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Any pattern on the drum as a sum of modes

    The box notebook made a promise about the string: the sine waves are enough: any shape of the string that
    is zero at the ends is a sum of the standing waves $\sin(n\pi x/L)$. The drum keeps the promise in two
    dimensions. Take the modes of a drum in the order of their frequencies $j_{\ell,\,n_r+1}$ (which is the physical
    order, the order in which they would be heard; $n$ is not the order). A mode is labelled by its number of circles
    $n_r$, its number of angular nodes $\ell$ and the sign of $m_\ell=\pm\ell$ (cosine for $m_\ell>0$, sine for $m_\ell<0$),
    and has $N_\text{nodes}=n_r+\ell$ nodes. Call the modes in this order $\psi_1,\psi_2,\psi_3,\dots$, scaled so that

    $$
    \int_0^{2\pi}\!\!\int_0^1 \psi_k\,\psi_{k'}\;r\,dr\,d\varphi=\begin{cases}1,&k=k',\\0,&k\ne k'.\end{cases}
    $$

    The zero for two different modes is called the **orthogonality** of the modes. It is the same fact that made the coefficients of
    the string's sines easy. (The weight $r\,dr\,d\varphi$ is the area element of section 2.) A pattern $f(r,\varphi)$
    with $f=0$ at the rim is approximated by the first $M$ modes,

    $$
    f(r,\varphi)\approx\sum_{k=1}^{M} c_k\,\psi_k(r,\varphi),
    \qquad
    c_k=\int_0^{2\pi}\!\!\int_0^1 f\,\psi_k\;r\,dr\,d\varphi ,
    $$

    where each coefficient $c_k$ is found by multiplying $f$ by that one mode and adding up: orthogonality makes
    all the other modes drop out. Choose a pattern and move the slider. The three pictures use one colour scale (blue
    positive, red negative); the graph shows how much of the pattern is left after $M$ modes, as a fraction of its own size.
    """)
    return


@app.cell
def _(TARGETS, mo):
    target_dd = mo.ui.dropdown(options={v: k for k, v in TARGETS.items()}, value=TARGETS["poke"], label="pattern")
    m_modes = mo.ui.slider(1, 200, step=1, value=10, label=r"$M$ (number of modes)", show_value=True)
    return m_modes, target_dd


@app.cell
def _(drum_fourier_figure, drum_fourier_modes_figure, m_modes, mo, plt, target_dd):
    _fig = drum_fourier_figure(target_dd.value, m_modes.value)
    _fig2 = drum_fourier_modes_figure(m_modes.value)
    _out = mo.vstack([
        mo.hstack([target_dd, m_modes], justify="start"),
        _fig,
        mo.md(r"""
        The modes in the sum, by number of angular nodes $\ell$ and number of circles $n_r$ (a cell holds the cosine mode
        on its left and the sine mode on its right; the red one is the last one added):
        """),
        _fig2,
        mo.md(r"""
        This works because the modes are **complete**: every smooth pattern that is zero at the rim is the limit of such
        sums, so the error can be made as small as one likes by adding modes. The speed is another matter. A smooth, wide
        pattern needs few modes; a sharp feature, such as the narrow pluck or the thin gaps of the rose, needs many, because small details need modes
        with many nodes. This is the 2D version of the ringing of a Fourier series for a square pulse in section 4.
        """),
    ])
    plt.close(_fig)
    plt.close(_fig2)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two differences from the string are worth noting. The drum's frequencies are proportional to the zeros
    $j_{\ell,\,n_r+1}$, which are not equally spaced, so they are not whole-number multiples of the lowest one; and the rim
    is a fixed boundary, whereas the atom's "rim" is infinity, where the wave simply dies away. What stays the same is
    the counting: **each coordinate has its own family of nodes and its own whole number.**

    **What $n$ does and does not order.** $n=n_r+\ell+1$ counts all the nodes, but it does not order the energies of
    the drum. The frequencies go as $j_{0,1}=2.405$, $j_{1,1}=3.832$, $j_{2,1}=5.136$ and $j_{0,2}=5.520$, so the mode with
    two nodes ($\ell=2$, $n_r=0$) lies below the mode with one node ($\ell=0$, $n_r=1$). Hydrogen is different: there the levels
    with the same $n$ do have the same energy (section 5, "What it says").

    **Three dimensions: an atom.** The atom has one more direction, the polar angle $\vartheta$, and so one more
    family of nodes. In three dimensions a node is a surface, and the three kinds are the three-dimensional versions
    of what we have seen:

    | Direction the node cuts across | String | Drumhead | Atom | Whole number |
    |---|---|---|---|---|
    | distance ($x$ or $r$) | $n-1$ points | $n_r$ circles | $n_r=n-\ell-1$ **spheres** | $n_r$ (radial) |
    | polar angle $\vartheta$ | none | none | $n_\vartheta=\ell-\lvert m_\ell\rvert$ **cones** | $n_\vartheta$ |
    | azimuth $\varphi$ | none | $n_\varphi=\lvert m_\ell\rvert=\ell$ diameters | $n_\varphi=\lvert m_\ell\rvert$ **planes** through the $z$ axis | $n_\varphi$ ($m_\ell=\pm n_\varphi$) |
    | angular total $\ell$ | none | $n_\varphi=\ell$ | $n_\vartheta+n_\varphi=\ell$ | $\ell$ |
    | **all nodes** $N_{\text{nodes}}$ | $n-1$ | $n_r+n_\varphi=n-1$ | $n_r+n_\vartheta+n_\varphi=n-1$ | |

    A polar node at $\vartheta=\pi/2$ is the flat equatorial ($xy$) plane and not a cone: $p_z$, for example, has
    $n_\vartheta=1$ and a planar node. The drum has no cones.

    $$
    N_{\text{nodes}}\;=\;\textcolor{#FF84FA}{n_r}\;+\;\underbrace{\textcolor{#FF8021}{n_\vartheta}\;+\;\textcolor{#6B913C}{n_\varphi}}_{\textcolor{#7030A0}{\ell}}
    \;=\;n-1 .
    $$

    The azimuth row is the same on the drum and in the atom: the diameters of the drum are the traces, in a
    horizontal slice, of the atom's planes through the axis (a plane through the axis is two half-planes, at $\varphi_0$ and
    $\varphi_0+\pi$, and a diameter is two rays). The distance row is the same for the string, the drum and
    the atom. The polar-angle row is new, and the next section is about it: the nodes that cut across the polar angle
    and the azimuth together number $\ell=n_\vartheta+n_\varphi$, and the part of them that cuts across the azimuth is
    $n_\varphi=\lvert m_\ell\rvert$.

    **The rest of the notebook applies this table.** Given $(n,\ell,m_\ell)$, draw $n_r$ spheres, $n_\vartheta$ cones and
    $n_\varphi$ planes, and the wave is positive and negative in the pieces between them. That is the shape of the orbital.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Standing waves on a sphere: the spherical harmonics

    The surface of a sphere has no walls, so the condition that fixes the allowed waves is different from the box.
    A wave on the sphere must have one definite value at every point and stay finite everywhere. Two places need
    attention: going once around the axis, and the two poles where the azimuth $\varphi$ loses its meaning.

    **Around the axis: $m_\ell$ is a whole number.** Going once around the axis, $\varphi\to\varphi+2\pi$, brings us
    back to the same point, so the wave must return to the same value. This is the ring of `../quantization`,
    section 2: the factor $e^{i m_\ell\varphi}$ is single-valued only when $m_\ell$ is a whole number,

    $$
    Y(\vartheta,\varphi)\;\propto\; e^{i m_\ell \varphi},
    \qquad m_\ell = 0,\ \pm1,\ \pm2,\ \dots
    $$

    The real and imaginary parts of this factor are $\cos(m_\ell\varphi)$ and $\sin(m_\ell\varphi)$: they have
    $\lvert m_\ell\rvert$ full periods around the axis, so the wave has $2\lvert m_\ell\rvert$ sign changes along a
    circle of constant latitude. Each sign change lies on a nodal line that passes through both poles.

    **From pole to pole: $\ell$ is a whole number.** What is left is a wave in the polar angle $\vartheta$. Its equation
    is a standing-wave equation like the others, with a number $\ell(\ell+1)$ playing the part of $k^2$. For the simplest
    case $m_\ell=0$ it reads

    $$
    \frac{d^2y}{d\vartheta^2}+\cot\vartheta\,\frac{dy}{d\vartheta}+\ell(\ell+1)\,y=0 .
    $$

    Near a pole the term $\cot\vartheta$ becomes large. Every solution then behaves like $a + c\,\ln\vartheta$: a constant
    plus a part that grows without limit as the pole is approached. A wave with $c\neq0$ would be infinite on the polar
    axis, which cannot describe a particle. We need $c=0$ at both poles. Solving the equation numerically from the
    equator toward the pole, starting even or odd about the equator, shows when that happens. This is the same
    method as the shooting in the oscillator notebook: the wave must behave at the end.
    """)
    return


@app.cell
def _(mo):
    ell_pole = mo.ui.slider(0.0, 5.0, step=0.1, value=1.5, label=r"$\ell$ (any real number)", show_value=True)
    return (ell_pole,)


@app.cell
def _(ell_pole, mo, plt, pole_figure):
    _fig = pole_figure(ell_pole.value)
    _out = mo.vstack([ell_pole, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The right-hand panel is the result. For a general $\ell$ both starts grow near the pole. Only when $\ell$ is a whole number
    does one of the two starts (even for even $\ell$, odd for odd $\ell$) have $c=0$. That solution is a polynomial in
    $\cos\vartheta$, a **Legendre polynomial**; it is finite at both poles and is the one we keep. So:

    $$
    \ell = 0,\ 1,\ 2,\ \dots,\qquad \lvert m_\ell\rvert\le\ell .
    $$

    The second condition is stated, not derived here: for a wave with $m_\ell\neq0$ the same argument gives a polynomial
    of degree $\ell$ only if $\lvert m_\ell\rvert$ does not exceed $\ell$. The solutions
    for a given pair are the **spherical harmonics** $Y_{\ell m_\ell}(\vartheta,\varphi)$.

    **What the number $\ell$ means.** Applying the angular part of the kinetic-energy operator to a harmonic gives back
    the same harmonic times a number,

    $$
    -\left[\frac{1}{\sin\vartheta}\frac{\partial}{\partial\vartheta}\Big(\sin\vartheta\,\frac{\partial}{\partial\vartheta}\Big)
    +\frac{1}{\sin^2\vartheta}\frac{\partial^2}{\partial\varphi^2}\right]Y_{\ell m_\ell}
    = \ell(\ell+1)\,Y_{\ell m_\ell}.
    $$

    The left side is the curvature of the wave over the sphere. The bracket is the sphere's counterpart of the second
    derivative $d^2/dx^2$ of a string: the more the wave wiggles, the larger the number. The number does not depend on
    $m_\ell$, because turning the axis does not change how much the wave curves. So the $2\ell+1$ harmonics with
    $m_\ell=-\ell,\dots,\ell$ all have the same angular "energy": this is the degeneracy $2\ell+1$ of the angular
    part, and it is a consequence of the symmetry of the sphere, as in the square box of the previous notebook.

    **Angular nodes.** A standing wave is described by where it vanishes. The harmonic $Y_{\ell m_\ell}$ has exactly
    $\ell$ angular nodes in total, $\ell=n_\vartheta+n_\varphi$. Each is a surface through the centre (a plane or a cone)
    that meets the sphere in a line. They are of two kinds, and each cuts across a different direction, as in the table
    of section 3:

    | Kind of angular node | Meets the sphere in | Cuts across | How many | Where it comes from |
    |---|---|---|---|---|
    | plane through the $z$ axis | a great circle through both poles (two opposite meridians) | azimuth $\varphi$ | $n_\varphi=\lvert m_\ell\rvert$ | $\cos(m_\ell\varphi)$ or $\sin(m_\ell\varphi)$ |
    | cone around the $z$ axis | a circle of constant latitude | polar angle $\vartheta$ | $n_\vartheta=\ell-\lvert m_\ell\rvert$ | zeros of the Legendre function in $\cos\vartheta$ |
    | **total** | | | $\ell$ | |

    (The nodal *lines* seen on the sphere are the meridians and circles of latitude where these planes and cones meet it.
    A cone at $\vartheta=\pi/2$ is the flat equatorial plane $z=0$: $p_z$ has $n_\vartheta=1$ and a planar node.)

    A harmonic with $m_\ell=\pm\ell$ has no cones, $n_\vartheta=0$ (all its nodes are planes, a "sectoral" shape),
    and a harmonic with $m_\ell=0$ has no planes, $n_\varphi=0$ (only cones, a "zonal" shape). More angular nodes
    means a shorter wavelength on the sphere and a larger $\ell(\ell+1)$, as for the box where more nodes meant a larger
    $E$.[^real]

    [^real]: The harmonics with $m_\ell\neq0$ are complex waves, $e^{im_\ell\varphi}$ times a function of $\vartheta$. The
        figures show the **real** harmonics: the real combinations $\cos(\lvert m_\ell\rvert\varphi)$ for $m_\ell>0$ and
        $\sin(\lvert m_\ell\rvert\varphi)$ for $m_\ell<0$. They have the same $\ell$, the same energy and the same angular
        nodes as the complex ones, but fixed directions, which is how orbitals are drawn in chemistry.
    """)
    return


@app.cell
def _(mo):
    l_harm = mo.ui.slider(0, 4, step=1, value=3, label=r"$\ell$", show_value=True)
    m_harm = mo.ui.slider(-4, 4, step=1, value=2, label=r"$m_\ell$", show_value=True)
    return l_harm, m_harm


@app.cell
def _(harmonic_figure, l_harm, m_harm, mo, plt):
    _l = l_harm.value
    _m = max(-_l, min(_l, m_harm.value))
    _note = (
        mo.md(rf"The value of $m_\ell$ is limited to $\lvert m_\ell\rvert\le\ell$, so $m_\ell={_m}$ is shown.")
        if _m != m_harm.value
        else mo.md("")
    )
    _fig = harmonic_figure(_l, _m)
    _out = mo.vstack([mo.hstack([l_harm, m_harm], justify="start"), _note, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    show_harm_nodes = mo.ui.switch(value=True, label="show nodes")
    return (show_harm_nodes,)


@app.cell
def _(l_harm, m_harm, mo, plt, show_harm_nodes):
    from harmonic_nodes_plots import harmonic_nodes_figure as _nodes_figure

    _l = l_harm.value
    _m = max(-_l, min(_l, m_harm.value))
    _fig = _nodes_figure(_l, _m, show_nodes=show_harm_nodes.value)
    _out = mo.vstack([
        mo.md(
            r"The same harmonic (set $\ell$ and $m_\ell$ with the sliders above), with its nodes drawn: "
            r"each cone has a constant polar angle $\vartheta$, each plane contains the $z$ axis."
        ),
        show_harm_nodes,
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The left picture is the usual way to draw a harmonic: in each direction, the distance of the surface from the centre is
    the size $\lvert Y\rvert$ of the wave in that direction, and the colour tells the sign (blue positive, red negative).
    The right picture shows the same wave unrolled like a world map; the black curves are the nodal lines. Count them
    against the title. The next figure shows all the real harmonics for $\ell=0,1,2,3$. Chemists call these the
    s, p, d and f shapes, and the letters are the standard names for $\ell=0,1,2,3$.
    """)
    return


@app.cell
def _(harmonic_gallery_figure, plt):
    _fig = harmonic_gallery_figure(3)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    show_gallery_nodes = mo.ui.switch(value=True, label="show nodes")
    return (show_gallery_nodes,)


@app.cell
def _(mo, plt, show_gallery_nodes):
    from harmonic_nodes_plots import harmonic_nodes_gallery_figure as _gallery_figure

    _fig = _gallery_figure(3, show_nodes=show_gallery_nodes.value)
    _out = mo.vstack([
        mo.md(r"The same shapes with all their nodes: cones (constant $\vartheta$) and planes through the $z$ axis."),
        show_gallery_nodes,
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    _rows = [
        "| $\\ell$ | name | $2\\ell+1$ harmonics | $\\ell(\\ell+1)$ | angular nodes (total $\\ell$) |",
        "|---|---|---|---|---|",
    ]
    for _l, _name in enumerate("spdf"):
        _rows.append(f"| {_l} | {_name} | {2 * _l + 1} | {_l * (_l + 1)} | {_l} |")
    mo.vstack([
        mo.md("The shapes in the figure are scaled to fill their panels, so a larger $\\ell$ does not mean a larger shape."),
        mo.md("\n".join(_rows)),
        mo.md(r"""
        **Checked in the tests:** the harmonics are orthonormal (each has total probability 1 over the sphere, and any
        two different ones are orthogonal); applying the operator above to each harmonic by finite differences gives
        $\ell(\ell+1)$; the number of angular nodes is $\ell$, split into $n_\varphi=\lvert m_\ell\rvert$ planes (meridians on the
        sphere) and $n_\vartheta=\ell-\lvert m_\ell\rvert$ cones (circles of latitude); and the numerical solution of the polar equation has $c=0$
        exactly at whole $\ell$. **Stated, not shown:** the restriction $\lvert m_\ell\rvert\le\ell$ is taken from the
        form of the harmonics, not derived.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where the angular nodes sit

    The count is $\ell$ in total. The next question is where they go. There is one rule for each kind of node.

    **Planes (cut across the azimuth; meridians on the sphere).** The azimuth dependence of a real harmonic is $\cos(m_\ell\varphi)$ or
    $\sin(m_\ell\varphi)$. Each vanishes at $2\lvert m_\ell\rvert$ values of $\varphi$ in one turn, which are the
    $\lvert m_\ell\rvert$ planes through the $z$ axis (each plane holds two opposite meridians). The planes are equally
    spaced, $\pi/\lvert m_\ell\rvert$ apart in $\varphi$:

    $$
    \cos(m_\ell\varphi)=0\ \text{ at }\ \varphi=\frac{\pi}{2\lvert m_\ell\rvert}+k\,\frac{\pi}{\lvert m_\ell\rvert},\qquad
    \sin(\lvert m_\ell\rvert\varphi)=0\ \text{ at }\ \varphi=k\,\frac{\pi}{\lvert m_\ell\rvert},
    $$

    with $k=0,1,\dots,\lvert m_\ell\rvert-1$ (the same planes again for $k$ beyond that).

    **Cones (cut across the polar angle; circles of latitude on the sphere).** The polar dependence is the Legendre function
    $P_\ell^{\lvert m_\ell\rvert}(\cos\vartheta)$. It has exactly $\ell-\lvert m_\ell\rvert$ zeros strictly between the
    poles. They are **not** equally spaced in $\cos\vartheta$, and they are mirror-symmetric about the equator
    (so one zero sits on the equator exactly when $\ell-\lvert m_\ell\rvert$ is odd). For large $\ell$ they come close
    to equal spacing in the angle $\vartheta$ itself, a wavelength of about $2\pi/\ell$ on the sphere.

    Try it. The dashed lines in the middle map are the predicted positions, computed from the rules (the
    azimuths from the formula above, the latitudes as the roots of a polynomial). The white curves are found from the wave
    itself. Where they agree you see a dashed line sitting on a white one.
    """)
    return


@app.cell
def _(mo):
    l_pl = mo.ui.slider(0, 8, step=1, value=4, label=r"$\ell$", show_value=True)
    m_pl = mo.ui.slider(0, 8, step=1, value=1, label=r"$\lvert m_\ell\rvert$", show_value=True)
    kind_pl = mo.ui.radio(
        options={"cos (real, mℓ > 0)": "cos", "sin (real, mℓ < 0)": "sin"},
        value="cos (real, mℓ > 0)",
        label="real form",
        inline=True,
    )
    return kind_pl, l_pl, m_pl


@app.cell
def _(kind_pl, l_pl, m_pl, mo, node_placement_figure, plt):
    _l = l_pl.value
    _m = min(m_pl.value, _l)
    _note = (
        mo.md(rf"The value of $\lvert m_\ell\rvert$ is limited to $\le\ell$, so $\lvert m_\ell\rvert={_m}$ is shown.")
        if _m != m_pl.value
        else mo.md("")
    )
    _fig = node_placement_figure(_l, _m, kind_pl.value)
    _out = mo.vstack([mo.hstack([l_pl, m_pl, kind_pl], justify="start"), _note, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Orange marks the $n_\vartheta$ cones, which meet the sphere in circles of latitude (cut across $\vartheta$), and green the
    $n_\varphi$ planes through the axis (cut across $\varphi$), as in the table of section 3. The right panel is the polar
    part alone: each time the curve crosses zero is one cone. The switch only turns the planes by half their spacing, $\pi/(2\lvert m_\ell\rvert)$,
    about the axis ($\cos$ to $\sin$); the nodes keep the same number and the same spacing.

    **Worked example, $\ell=4$, $\lvert m_\ell\rvert=1$.** One plane ($n_\varphi=1$) and three cones ($n_\vartheta=3$), $\ell=4$ angular nodes in total. The
    polar function is proportional to $\sin\vartheta\,(7\cos^3\vartheta-3\cos\vartheta)=\sin\vartheta\,\cos\vartheta\,
    (7\cos^2\vartheta-3)$. Its zeros between the poles are

    $$
    \cos\vartheta=0,\ \ \cos\vartheta=\pm\sqrt{3/7}\quad\Longrightarrow\quad
    \vartheta=90^\circ,\ \ 49.1^\circ,\ \ 130.9^\circ .
    $$

    (The factor $\sin\vartheta$ vanishes only at the poles, which are not nodes.) The figure shows these three
    angles on the right, and the tests check them.

    **Three more things to notice.**

    - *Complex harmonics have fewer visible nodes.* The complex $Y_{\ell m_\ell}$ never vanishes on the $n_\varphi$ planes; they appear in
      the real $\cos$ or $\sin$ form used in the figures (see "Complex waves and real waves" below).
    - *The $z$ axis is a choice.* The $2\ell+1$ harmonics of one $\ell$ have the same energy, so any combination of them
      is an equally good standing wave, and so is the same pattern turned to point along another axis. What is fixed is
      the total: always $\ell$ angular nodes, split differently. At one extreme ($m_\ell=0$) all of them are cones
      ($n_\vartheta=\ell$); at the other ($\lvert m_\ell\rvert=\ell$) all are planes through the axis ($n_\varphi=\ell$), equally spaced.
    - *Why the count is $\ell$.* The usual explanation is the same as for the box: the harmonics are the standing waves of
      the sphere with the value $\ell(\ell+1)$ for the curvature, and a wave with more nodes wiggles more, so it has a
      larger value. Whole numbers of nodes fit in the way whole numbers of half wavelengths fit on a string. What the
      numerics confirm is the count and the positions (below), not a proof that the number of zeros of $P_\ell^{\lvert m_\ell\rvert}$ is
      $\ell-\lvert m_\ell\rvert$: that is a known property of the Legendre functions, here verified for all
      $\ell\le8$.

    **Checked in the tests:** for every $\ell\le8$ and $0\le\lvert m_\ell\rvert\le\ell$, the polar function changes sign
    $\ell-\lvert m_\ell\rvert$ times; its zeros agree with the roots of the polynomial prediction, and the real harmonic
    vanishes there at every azimuth; the zeros are mirror-symmetric about the equator, with one on it exactly when
    $\ell-\lvert m_\ell\rvert$ is odd; the azimuth planes are at the stated angles, $\pi/\lvert m_\ell\rvert$ apart, with
    $2\lvert m_\ell\rvert$ sign changes round the axis; the total is $\ell$ for both real forms; the example above gives
    $49.1^\circ$, $90^\circ$, $130.9^\circ$; and the complex harmonic has the same size at every azimuth with its phase winding
    $m_\ell$ times.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Complex waves and real waves: why orbitals are drawn real

    The harmonics with $m_\ell\neq0$ were introduced as $e^{im_\ell\varphi}$, a complex wave, yet the pictures show
    $\cos(m_\ell\varphi)$ and $\sin(m_\ell\varphi)$. The two descriptions differ in what the wave does, not in its energy.

    **The complex wave $e^{im_\ell\varphi}$ travels round the ring.** At each azimuth it is a point on a circle in the
    complex plane, and going once round the axis turns that point $m_\ell$ times. It has no nodes: its size is $1$
    everywhere, so $\lvert\psi\rvert^2$, the probability density, is the same at every azimuth. It has a definite angular
    momentum about the $z$ axis, $L_z=m_\ell\hbar$, and the sign of $m_\ell$ is the direction of travel.

    **The real waves $\cos(m_\ell\varphi)$ and $\sin(m_\ell\varphi)$ stand still.** They have $\lvert m_\ell\rvert$ nodal
    planes through the axis and a probability density that is large in some directions and zero in others. Neither
    has a single value of $L_z$: each is an equal mix of $+m_\ell$ and $-m_\ell$.

    **Why both are allowed.** The waves $+m_\ell$ and $-m_\ell$ have the same energy, so any combination of them is an
    equally good stationary state. The sum and difference give the two real waves,

    $$
    \cos(m_\ell\varphi)=\tfrac12\bigl(e^{im_\ell\varphi}+e^{-im_\ell\varphi}\bigr),\qquad
    \sin(m_\ell\varphi)=\tfrac1{2i}\bigl(e^{im_\ell\varphi}-e^{-im_\ell\varphi}\bigr),
    $$

    and for the harmonics the real combinations $Y_{\ell m_\ell}\pm Y_{\ell,-m_\ell}$ (divided by $\sqrt2$, or by $\sqrt2\,i$ for the
    difference) have the same $\ell$-dependence in $\vartheta$. Which pair to use is a choice of description. The
    complex pair is the choice when the question is about $L_z$. The real pair is the choice when the question is about
    shape, and it is the one chemists draw: its nodal planes have fixed directions, so each orbital points somewhere
    in space, and a bond can be made along that direction. The Hamiltonian of the atom is real, so the real part of a
    stationary state is again a stationary state with the same energy, which is why nothing is lost by drawing only that part.

    **The time dependence changes only the overall phase.** A stationary state of energy $E$ evolves as
    $\psi(\varphi,t)=\psi(\varphi)\,e^{-iEt/\hbar}$. This multiplies the wave by a point on the unit circle that turns
    with time, so it changes the phase of the whole wave at once and leaves $\lvert\psi\rvert^2$ unchanged: nothing
    that is measured about a stationary state depends on $t$. Move the slider, which is $\theta=Et/\hbar$. In the left
    column the real part and the imaginary part of the complex wave slide round the ring, trading places, while
    $\lvert\psi\rvert^2$ stays flat. In the right column the nodes of the real wave stay where they are and only its height
    breathes between the real and the imaginary part.
    """)
    return


@app.cell
def _(mo):
    m_cx = mo.ui.slider(0, 4, step=1, value=2, label=r"$m_\ell$", show_value=True)
    theta_cx = mo.ui.slider(0, 360, step=5, value=0, label=r"overall phase $\theta=Et/\hbar$ (degrees)", show_value=True)
    return m_cx, theta_cx


@app.cell
def _(complex_vs_real_figure, m_cx, mo, plt, theta_cx):
    _fig = complex_vs_real_figure(m_cx.value, theta_cx.value)
    _out = mo.vstack([mo.hstack([m_cx, theta_cx], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Checked in the tests** (`test_complex_waves.py`): $\lvert e^{im_\ell\varphi}\rvert^2=1$ for every overall phase; the phase
    of $e^{im_\ell\varphi}$ winds $m_\ell$ times round the axis; $\cos(m_\ell\varphi)$ has $2\lvert m_\ell\rvert$ equally spaced zeros
    that do not move with $\theta$, while the zeros of the real part of the complex wave do; and the sum of the $+m_\ell$ and $-m_\ell$
    waves is exactly the cosine.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The same idea as a Fourier series

    The harmonics have a second job besides describing electrons: they are the natural building blocks for **any**
    shape on a circle or a sphere, in the same way that sines are the building blocks for any shape along a line.

    - **On a line** (the string, `standing_waves_in_boxes.py` section 1) and **in a drum** (section 3 above), the standing waves are the building blocks.
    - **Around a circle**, any shape is a sum of the ring waves $\cos(m_\ell\varphi)$ and $\sin(m_\ell\varphi)$, the Fourier series
      on a circle and the angular factor of the drum.
    - **On a sphere**, any shape is a sum of spherical harmonics, $f(\vartheta,\varphi)=\sum a_{\ell m_\ell}Y_{\ell m_\ell}$.
      It is the Fourier series for a sphere.

    In each case the number of nodes plays the part of frequency: $m_\ell$ counts the periods round the circle, and
    $\ell$ counts the angular nodes of a wave on the sphere. A smooth shape needs only a few low harmonics; a sharp feature needs many
    high ones. As for the drum in section 3, the weight of each harmonic is found by **projection**: multiply the shape by that
    harmonic and add up the result over the whole circle or sphere,

    $$
    a_{\ell m_\ell}=\int f(\vartheta,\varphi)\,Y_{\ell m_\ell}(\vartheta,\varphi)\,d\Omega ,
    $$

    which works because different harmonics are orthogonal (the orthonormality listed above). Cutting the sum off at some
    $\ell$ keeps all the detail down to a size of about $180^\circ/\ell$ and smooths out everything finer, which is why the
    same harmonics appear in weather models, in maps of the sky and in the angular part of an atom. The sliders show it.
    First a square pulse on a circle, where the overshoot at the jumps is the familiar ringing of a Fourier series, and
    then a map of bumps on a sphere.
    """)
    return


@app.cell
def _(mo):
    m_four = mo.ui.slider(1, 20, step=1, value=3, label=r"harmonics kept, up to $m_\ell$", show_value=True)
    return (m_four,)


@app.cell
def _(fourier_circle_figure, m_four, mo, plt):
    _fig = fourier_circle_figure(m_four.value)
    _out = mo.vstack([m_four, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    l_four = mo.ui.slider(0, 16, step=1, value=3, label=r"harmonics kept, up to $\ell$", show_value=True)
    return (l_four,)


@app.cell
def _(fourier_sphere_figure, l_four, mo, plt):
    _fig = fourier_sphere_figure(l_four.value)
    _out = mo.vstack([l_four, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Checked in the tests:** the Fourier coefficients of $\cos 3\varphi$ and $\sin 5\varphi$ are non-zero only at
    $3$ and $5$; the circle and sphere series converge as more harmonics are kept, and the total strength of the
    coefficients equals the strength of the shape (Parseval's relation); a shape built from two harmonics is
    projected back onto exactly those two.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 The radial part: effective potential and energy levels

    **The short version.** For an electron in a potential that depends only on the distance $r$ (a **central potential**
    $V(r)$), the Schrödinger equation separates, and what remains is **two eigenvalue problems**, one for the angles and
    one for the distance:

    $$
    \begin{array}{lll}
    \text{angular:} & \quad{-\Lambda}\,Y_{\ell m_\ell}=\textcolor{#7030A0}{\ell(\ell+1)}\,Y_{\ell m_\ell}
    & \quad\text{eigenvalue }\textcolor{#7030A0}{\ell(\ell+1)}\\[6pt]
    \text{radial:} & \quad\Bigl[-\dfrac{\hbar^2}{2m}\dfrac{d^2}{dr^2}+V_{\text{eff}}(r)\Bigr]u_{n_r\ell}=\textcolor{#FF8021}{E}\,u_{n_r\ell}
    & \quad\text{eigenvalue }\textcolor{#FF8021}{E},
    \end{array}
    \qquad
    V_{\text{eff}}=V+\frac{\hbar^2\,\textcolor{#7030A0}{\ell(\ell+1)}}{2m\,r^2},
    $$

    where $u=rR$ is the radial wave in a convenient form, $\hbar$ is the reduced Planck constant and $m$ the mass of the electron.

    - **Angular.** The operator $\Lambda$ acts on the sphere. Its eigenfunctions are the spherical harmonics
      $Y_{\ell m_\ell}$ of section 4, and its eigenvalues are the numbers $\ell(\ell+1)$.
    - **Radial.** The operator acts along $r$ only, so this is a **1D Schrödinger equation**, with the effective potential
      $V_{\text{eff}}$. Its eigenfunctions $u_{n_r\ell}$ are the radial waves and its eigenvalues are the energies $E$. As on the
      string, the wave must vanish at $r=0$ and die away far out, and only some energies fit; the eigenfunctions are labelled by
      the number of **radial nodes** $n_r$ (spheres around the nucleus), the string's $n-1$ again.
    - **How they are linked.** The eigenvalue $\ell(\ell+1)$ of the angular problem enters the radial problem as a **fixed
      number**, inside $V_{\text{eff}}$. So each $\ell$ gives its own 1D problem, with its own ladder of energies $E$, and the
      three-dimensional problem is a family of 1D problems, one for each $\ell$.
    - **The centrifugal barrier.** The extra term in $V_{\text{eff}}$ is the cost of going round the centre: it grows as $r$
      shrinks, so it acts as a wall that keeps states with $\ell>0$ away from $r=0$.
    - **Hydrogen.** For the pull of the nucleus, $V=-1/r$ (atomic units), the energies come out as
      $E=-13.6\ \text{eV}/n^2$ with $n=n_r+\ell+1$ (stated here, derived in the Details below). They depend on $n$ only, not on
      how the nodes split between $n_r$ and $\ell$. The figure shows it: choose $\ell$; the horizontal lines are the allowed
      energies, and the blue curves are the radial waves drawn at their energies.

    The derivations are below, folded away (**Details**). Only the points above are needed for the next section.
    """)
    return


@app.cell
def _(mo):
    l_ctr = mo.ui.slider(0, 4, step=1, value=1, label=r"$\ell$", show_value=True)
    return (l_ctr,)


@app.cell
def _(central_levels_figure, l_ctr, mo, plt):
    _fig = central_levels_figure(l_ctr.value)
    _out = mo.vstack([l_ctr, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.accordion({
        "Details (optional): from the Schrödinger equation to the radial equation": mo.md(r"""
    **The angular part: a particle on a sphere**
    
        Take a particle that is free to move over the surface of a sphere of radius $a$ and nowhere else. Its wave depends on
        the two angles only, and the Laplacian on the sphere is $\Lambda/a^2$, so the Schrödinger equation is
        $-\frac{\hbar^2}{2ma^2}\Lambda Y=EY$. By section 4, $-\Lambda Y_{\ell m_\ell}=\ell(\ell+1)Y_{\ell m_\ell}$, so the allowed energies are
    
        $$
        E_\ell=\frac{\hbar^2\,\ell(\ell+1)}{2ma^2},\qquad \ell=0,1,2,\dots,
        $$
    
        with $2\ell+1$ states (the values of $m_\ell$) in each level. The $Y_{\ell m_\ell}$ are the eigenfunctions of two
        operators: of $\hat L^2=-\hbar^2\Lambda$ (the angular momentum squared) with eigenvalue $\hbar^2\ell(\ell+1)$, and of
        $\hat L_z=-i\hbar\,\partial_\varphi$ with eigenvalue $\hbar m_\ell$, because $\partial_\varphi e^{im_\ell\varphi}=im_\ell e^{im_\ell\varphi}$.
    
        **Separating the equation for a central potential**
    
        For an electron in $V(r)$ the Schrödinger equation is $-\frac{\hbar^2}{2m}\nabla^2\psi+V\psi=E\psi$, and with the
        Laplacian of section 2,
    
        $$
        {}-\frac{\hbar^2}{2m}\Bigl[\frac{1}{r^2}\,\partial_r\bigl(r^2\,\partial_r\psi\bigr)+\frac{1}{r^2}\,\Lambda\psi\Bigr]+V(r)\,\psi=E\,\psi .
        $$
    
        Try a product, $\psi=R(r)\,Y(\vartheta,\varphi)$. The radial derivatives act only on $R$ and $\Lambda$ acts only on $Y$.
        Divide everything by $RY$ and multiply by $-2mr^2/\hbar^2$:
    
        $$
        \underbrace{\frac{1}{R}\,\frac{d}{dr}\Bigl(r^2\,\frac{dR}{dr}\Bigr)-\frac{2mr^2}{\hbar^2}\bigl[V(r)-E\bigr]}_{\text{depends on }r\text{ only}}
        \;=\;\underbrace{-\frac{1}{Y}\,\Lambda Y}_{\text{depends on the angles only}} .
        $$
    
        A function of $r$ alone can equal a function of the angles alone only if both are the same constant. That constant
        is $\ell(\ell+1)$: the right side is exactly the equation of section 4, where finiteness at the poles forced the
        constant to be $\ell(\ell+1)$ with $\ell$ a whole number. So the angular part is solved already, by $Y_{\ell m_\ell}$, and the
        radial part is
    
        $$
        \frac{1}{r^2}\,\frac{d}{dr}\Bigl(r^2\,\frac{dR}{dr}\Bigr)+\Bigl[\frac{2m}{\hbar^2}\bigl(E-V(r)\bigr)-\frac{\ell(\ell+1)}{r^2}\Bigr]R=0 .
        $$
    
        **The radial equation and the effective potential**
    
        Write $u(r)=rR(r)$. Then $\frac{d}{dr}\bigl(r^2R'\bigr)=r\,u''$, and the radial equation becomes one for a wave along a line,
    
        $$
        {}-\frac{\hbar^2}{2m}\,\frac{d^2u}{dr^2}+V_{\text{eff}}(r)\,u=E\,u,
        \qquad
        V_{\text{eff}}(r)=V(r)+\frac{\hbar^2\,\ell(\ell+1)}{2m\,r^2}.
        $$
    
        The extra term in $V_{\text{eff}}$ is the $\hat L^2/(2mr^2)$ of section 2: a wave that curves over the sphere costs
        kinetic energy, and the cost grows as $r$ gets smaller, because the same angle spans a smaller circle. It acts as a wall
        (the "centrifugal barrier") that pushes an electron with $\ell>0$ away from the nucleus. Two conditions go with the
        equation: $u(0)=0$, because $R$ must stay finite at the nucleus, and the normalisation of the whole wave,
        $\int\lvert\psi\rvert^2dV=\int_0^\infty R^2r^2dr\int\lvert Y\rvert^2d\Omega=\int_0^\infty u^2\,dr=1$.
    
        For hydrogen, in atomic units ($\hbar=m=1$, lengths in bohr, energies in hartree), $V=-1/r$ with $r$ in bohr and the equation is
    
        $$
        {}-\frac12\,\frac{d^2u}{dr^2}+\Big[\underbrace{-\frac{1}{r}}_{\text{pull of the nucleus}}
        +\underbrace{\frac{\ell(\ell+1)}{2r^2}}_{\text{angular curvature}}\Big]u
        = \textcolor{#FF8021}{E}\,u .
        $$
    
        The figure above shows the three terms for the $\ell$ chosen with the slider, the allowed levels (horizontal lines) and the
        radial waves $u(r)$ drawn at their energies.
    """)
    })
    return


@app.cell
def _(mo):
    mo.accordion({
        "Details (optional): solving the radial equation for hydrogen exactly": mo.md(r"""
        Bound states have $E<0$. Write $E=-\kappa^2/2$ with $\kappa>0$ (the Greek letter kappa, a decay rate in units of inverse
        bohr, i.e. $1/0.529$ Å$^{-1}$). The equation $-\frac12u''+V_{\text{eff}}u=Eu$ is
    
        $$
        u''=\Bigl[\frac{\ell(\ell+1)}{r^2}-\frac{2}{r}+\kappa^2\Bigr]u .
        $$
    
        Look at the two ends first. **Near the nucleus** the $1/r^2$ term wins, $u''\approx\ell(\ell+1)u/r^2$, and the solution
        that vanishes at $r=0$ is $u\sim r^{\ell+1}$ (the other one, $r^{-\ell}$, is infinite). **Far away** the constant wins,
        $u''\approx\kappa^2u$, and the solution that dies away is $u\sim e^{-\kappa r}$ (the other one grows without limit). So
        write
    
        $$
        u(r)=r^{\ell+1}\,e^{-\kappa r}\,f(r),
        $$
    
        where $f$ carries what is left. Putting this into the equation and collecting terms (several lines of algebra, not shown)
        gives an equation for $f$,
    
        $$
        r\,f''+2\bigl(\ell+1-\kappa r\bigr)f'+2\bigl[1-\kappa(\ell+1)\bigr]f=0 .
        $$
    
        Try a power series $f=\sum_k c_kr^k$. Each power of $r$ gives a relation between neighbouring coefficients (again
        not shown),
    
        $$
        c_{k+1}=\frac{2\bigl[\kappa(k+\ell+1)-1\bigr]}{(k+1)(k+2\ell+2)}\;c_k .
        $$
    
        If the series never stops, $c_{k+1}/c_k\to2\kappa/k$ for large $k$, which is the ratio of the series of $e^{2\kappa r}$, so
        $f\sim e^{2\kappa r}$ and $u\sim e^{+\kappa r}$: the wave grows far away and cannot be normalised. (This is the usual
        argument, stated and not proved.) The series must **stop**: some coefficient $c_{k+1}$ must be zero, which happens when
        $\kappa(k+\ell+1)=1$ for one whole number $k=n_r$. Then $f$ is a polynomial of degree $n_r$, so it has $n_r$ zeros: these
        are the radial nodes. The condition is
    
        $$
        \frac{1}{\kappa}=n_r+\ell+1=n,
        \qquad
        E_n=-\frac{\kappa^2}{2}=-\frac{1}{2n^2}\ \text{hartree}=-\frac{13.6\ \text{eV}}{n^2}.
        $$
    
        The polynomial $f$ is a generalised Laguerre polynomial, so that the radial function is
    
        $$
        R_{n\ell}(r)\;\propto\;r^{\ell}\,e^{-r/n}\,L^{\,2\ell+1}_{\,n-\ell-1}\!\Bigl(\frac{2r}{n}\Bigr).
        $$
    
        That the series gives the Laguerre polynomials is **stated, not derived** here; the notebook checks it instead: the
        exact functions solve the radial equation, are normalised and have $n_r=n-\ell-1$ nodes (see the tests below).
    """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    The same equation can also be solved on a computer, which needs none of the steps above and is the check that they
    are right. Put it on a grid of distances $r_i = i\,h$ and replace the second derivative by its finite-difference
    form. The conditions are that $u$ vanishes at the nucleus, $u(0)=0$, and dies away far from it, as in the oscillator. The
    equation becomes a matrix eigenvalue problem, and its eigenvalues are the allowed energies, its
    eigenvectors the radial waves. The result for each $\ell$ is a ladder of energies. Counting the sign changes
    of the $k$-th wave gives $k$ **radial nodes**, spheres around the nucleus, and their number is called $n_r$. The waves are
    labelled by

    $$
    n = n_r + \ell + 1,\qquad n=\ell+1,\ \ell+2,\ \dots,
    $$

    which is the labelling of section 3: more nodes along $r$ means more energy, as for the box.

    The energies that come out are

    $$
    \textcolor{#FF8021}{E}_n = -\frac{1}{2n^2}\ \text{hartree} = -\frac{13.6\ \text{eV}}{n^2},
    $$

    and they depend on $n$ only, not on $\ell$. The figure compares the numerical wave with the known exact radial
    wave for the chosen $n$ and $\ell$; the horizontal line is the energy, the black dots are the radial nodes.
    """)
    return


@app.cell
def _(mo):
    n_rad = mo.ui.slider(1, 4, step=1, value=3, label=r"$n$", show_value=True)
    l_rad = mo.ui.slider(0, 3, step=1, value=1, label=r"$\ell$", show_value=True)
    return l_rad, n_rad


@app.cell
def _(l_rad, mo, n_rad, plt, radial_state_figure):
    _n = n_rad.value
    _l = min(l_rad.value, _n - 1)
    _note = (
        mo.md(rf"For $n={_n}$ the value of $\ell$ is limited to $\ell\le n-1$, so $\ell={_l}$ is shown.")
        if _l != l_rad.value
        else mo.md("")
    )
    _fig = radial_state_figure(_n, _l)
    _out = mo.vstack([mo.hstack([n_rad, l_rad], justify="start"), _note, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(hydrogen_energy, hydrogen_energy_ev, mo, numerical_levels):
    _levels = numerical_levels(3, 4)
    _rows = [
        "| $n$ | $\\ell$ | radial nodes $n_r=n-\\ell-1$ | numerical $E$ (hartree) | exact $-1/(2n^2)$ | difference |",
        "|---|---|---|---|---|---|",
    ]
    for (_n, _l), _e in sorted(_levels.items()):
        _ex = float(hydrogen_energy(_n))
        _rows.append(f"| {_n} | {_l} | {_n - _l - 1} | {_e:.5f} | {_ex:.5f} | {_e - _ex:+.1e} |")
    mo.vstack([
        mo.md("**Numerical and exact energies.** The matrix gives the same energy for every $\\ell$ of one $n$ "
              "(to the accuracy of the grid):"),
        mo.md("\n".join(_rows)),
        mo.md(
            f"The lowest level is $-0.5$ hartree $={float(hydrogen_energy_ev(1)):.2f}$ eV, the energy needed to remove "
            f"the electron from hydrogen; the next is $-1/8$ hartree $={float(hydrogen_energy_ev(2)):.2f}$ eV."
        ),
    ])
    return


@app.cell
def _(level_check_figure, plt):
    _fig = level_check_figure(4)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(degeneracy, mo, quantum_numbers):
    _rows = ["| $n$ | allowed $\\ell$ | $\\sum_\\ell(2\\ell+1)$ | orbitals in the level, $n^2$ |", "|---|---|---|---|"]
    for _n in range(1, 6):
        _ls = ", ".join(str(_l) for _l in range(_n))
        _terms = " + ".join(str(2 * _l + 1) for _l in range(_n))
        assert degeneracy(_n) == len(quantum_numbers(_n)) == _n * _n
        _rows.append(f"| {_n} | {_ls} | {_terms} | {_n * _n} |")
    mo.vstack([
        mo.md(r"""
        **Why $n^2$ states in a level.** The energy depends on $n$ alone, while a given $n$ allows
        $\ell=0,1,\dots,n-1$ (the radial node count $n_r=n-\ell-1$ must be zero or more), and each $\ell$ has $2\ell+1$
        values of $m_\ell$. So the level $n$ contains

        $$
        \sum_{\ell=0}^{n-1}(2\ell+1) = n^2
        $$

        orbitals. The degeneracy in $m_\ell$ comes from the symmetry of the sphere (every direction is equivalent); the
        degeneracy in $\ell$ is not, as "What it says" below explains.
        Counting each electron's spin doubles the numbers to $2n^2$.
        """),
        mo.md("\n".join(_rows)),
        mo.md(r"""
        **Checked in the tests:** the exact radial functions solve the radial equation, are normalised and mutually
        orthogonal, have $n_r=n-\ell-1$ nodes, and give $\langle r\rangle=\tfrac12\big(3n^2-\ell(\ell+1)\big)$ bohr ($\times\,0.529$ Å). The
        finite-difference matrix reproduces $-1/(2n^2)$ to better than $0.15\%$ for $n\le4$ and every allowed $\ell$,
        its eigenvectors have the right numbers of nodes and agree with the exact radial functions, and the number of
        numerical states (counting $2\ell+1$ for each $\ell$) at each of the first four levels is $n^2$. **Stated, not
        shown:** that no other energies exist below zero, and the degeneracy in $\ell$ as a property of $-1/r$ (the matrix
        shows it for the values computed).
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What it says

    - **Hydrogen's energy counts nodes.** Since $n_r+\ell=n-1=N_{\text{nodes}}$, the energy $E_n=-1/(2n^2)$ depends only on the
      total number of nodes, and not on how they divide between $n_r$ radial nodes and $\ell$ angular ones. That is the
      degeneracy in $\ell$ found above ($n^2$ states in a level). It is not explained by the symmetry of the sphere, it is special to the
      $-1/r$ attraction, and it is removed as soon as the potential differs from it, as in every atom with more than one electron. Other central
      potentials do not have it: the drum has $E\propto j_{\ell,\,n_r+1}^2$, where a radial node and an angular node cost
      different energy, and the three-dimensional harmonic oscillator has $E=\hbar\omega\bigl(2n_r+\ell+\tfrac32\bigr)$ (stated,
      not derived here), where a radial node costs twice as much as an angular one.
    - **At the nucleus.** Near it $u\sim r^{\ell+1}$, so $R\sim r^{\ell}$ (the centrifugal barrier above at work): only $\ell=0$ has
      $\psi\neq0$ at the nucleus; all other orbitals are zero there.
    - **Only $R$ changes.** For any central potential the angular part is the same $Y_{\ell m_\ell}$, with the same
      angular nodes; only the radial function $R(r)$, and with it the energies, depends on $V(r)$. The shapes of
      sections 4 and 6 are therefore not special to hydrogen.

    **Checked in the tests:** the Laplacian formulas of section 2 against Cartesian second derivatives of test functions;
    the numerical radial equation for $\ell=0,1,2$ against $-1/(2n^2)$ with $n_r$ nodes for each level; the normalisation of
    $R_{n\ell}$; and that $R_{n\ell}(0)\neq0$ only for $\ell=0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Hydrogen orbitals

    Multiplying the two parts gives the orbital, $\psi_{n_r\ell m_\ell}(r,\vartheta,\varphi)=R_{n_r\ell}(r)\,Y_{\ell m_\ell}(\vartheta,\varphi)$,
    which section 3 also labelled $\psi_{n\ell m_\ell}$ with $n=n_r+\ell+1$. Its
    square gives the probability density. Three kinds of node combine, as in the table of section 3: $n_r=n-\ell-1$
    spheres from the radial part, and from the angular part $n_\vartheta=\ell-\lvert m_\ell\rvert$ cones and
    $n_\varphi=\lvert m_\ell\rvert$ planes through the axis (each angular node of the sphere extends outward as a surface
    through the nucleus). The angular nodes number $\ell=n_\vartheta+n_\varphi$ and the total is
    $N_{\text{nodes}}=n-1=n_r+n_\vartheta+n_\varphi$. The names are
    1s, 2s, 2p, 3s, 3p, 3d: the number is $n$ and the letter is $\ell$.

    The first figure shows the six lowest distinct orbitals in a plane through the axis (top) and as the probability of
    finding the electron at a given distance (bottom). Notice that a larger $n_r$ means more peaks in the probability
    distribution, and that the orbitals of one $n$ extend over a similar range of distances, in line with their common energy.
    """)
    return


@app.cell
def _(orbital_gallery_figure, plt):
    _fig = orbital_gallery_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The sliders choose any orbital up to $n=4$ and $\ell=3$. Four views are shown: the signed wave in a plane through the
    axis, the probability density in the same plane, a cloud of points drawn at random with probability $\lvert\psi\rvert^2$
    (each point is where the electron might be found, seen from the side), and the probability per unit distance.[^2p]

    [^2p]: For $\ell=1$ the three real orbitals are the familiar $p_z$ ($m_\ell=0$), $p_x$ ($m_\ell=1$) and $p_y$
        ($m_\ell=-1$): the harmonics are proportional to $z/r$, $x/r$ and $y/r$. The plane of the figure is chosen so that the
        orbital does not vanish in it: the $xz$ plane for $m_\ell\ge0$, and the plane containing the $y$ axis for $m_\ell<0$.
    """)
    return


@app.cell
def _(mo):
    n_orb = mo.ui.slider(1, 4, step=1, value=3, label=r"$n$", show_value=True)
    l_orb = mo.ui.slider(0, 3, step=1, value=2, label=r"$\ell$", show_value=True)
    m_orb = mo.ui.slider(-3, 3, step=1, value=0, label=r"$m_\ell$", show_value=True)
    return l_orb, m_orb, n_orb


@app.cell
def _(l_orb, m_orb, mo, n_orb, orbital_figure, plt):
    _n = n_orb.value
    _l = min(l_orb.value, _n - 1)
    _m = max(-_l, min(_l, m_orb.value))
    _note = (
        mo.md(rf"The sliders are limited to $\ell\le n-1$ and $\lvert m_\ell\rvert\le\ell$, so $\ell={_l}$, $m_\ell={_m}$ is shown.")
        if (_l, _m) != (l_orb.value, m_orb.value)
        else mo.md("")
    )
    _fig = orbital_figure(_n, _l, _m)
    _out = mo.vstack([mo.hstack([n_orb, l_orb, m_orb], justify="start"), _note, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Checked in the tests:** the orbitals are orthonormal in three dimensions (integrated over all space); the
    $2p$ orbitals are proportional to $z$, $x$ and $y$ times a function of $r$; points drawn by the sampling code have the
    mean distance $\langle r\rangle=5$ bohr $=2.65$ Å and the mean $\langle r^2\rangle=30$ bohr$^2$ $=8.4$ Å$^2$ of the $2p$ orbital.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading an orbital from its nodes

    The sliders below draw the nodes of any orbital, by kind: spheres (pink, cut across $r$), cones (orange, cut across
    $\vartheta$) and planes through the axis (green, cut across $\varphi$), with the counts $n_r$, $n_\vartheta$ and $n_\varphi$. The left panel is a plane through the $z$
    axis. A sphere appears as a circle, a cone as two straight lines through the centre, and a plane through the axis
    as the axis itself. The right panel is a horizontal slice seen from above, like the drum: spheres appear as circles
    and planes through the axis as diameters. If the slice at $z=0$ would itself be a node (the flat plane $\vartheta=\pi/2$ is a polar node, not a cone), the slice is taken a little above it, where the cones show up as circles.

    The wave changes sign across every node and nowhere else, so the signs of the pieces follow from the nodes alone.
    """)
    return


@app.cell
def _(mo):
    n_nod = mo.ui.slider(1, 4, step=1, value=3, label=r"$n$", show_value=True)
    l_nod = mo.ui.slider(0, 3, step=1, value=2, label=r"$\ell$", show_value=True)
    m_nod = mo.ui.slider(-3, 3, step=1, value=-2, label=r"$m_\ell$", show_value=True)
    return l_nod, m_nod, n_nod


@app.cell
def _(l_nod, m_nod, mo, n_nod, node_anatomy_figure, plt):
    _n = n_nod.value
    _l = min(l_nod.value, _n - 1)
    _m = max(-_l, min(_l, m_nod.value))
    _note = (
        mo.md(rf"The sliders are limited to $\ell\le n-1$ and $\lvert m_\ell\rvert\le\ell$, so $\ell={_l}$, $m_\ell={_m}$ is shown.")
        if (_l, _m) != (l_nod.value, m_nod.value)
        else mo.md("")
    )
    _fig = node_anatomy_figure(_n, _l, _m)
    _out = mo.vstack([mo.hstack([n_nod, l_nod, m_nod], justify="start"), _note, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, node_counts, node_geometry):
    _cases = [
        (1, 0, 0, "1s", "no node: one blob"),
        (2, 0, 0, "2s", "$n_r=1$: a sphere, a core and a shell of opposite sign"),
        (2, 1, 0, "2p$_z$", "$n_\\vartheta=1$: one flat plane across $z$, two lobes, up and down"),
        (2, 1, 1, "2p$_x$", "$n_\\varphi=1$: one plane through the axis, two lobes, left and right"),
        (3, 0, 0, "3s", "$n_r=2$: two spheres, a core and two shells"),
        (3, 1, 0, "3p$_z$", "$n_r=1$ sphere and $n_\\vartheta=1$ flat plane"),
        (3, 2, 0, "3d$_{z^2}$", "$n_\\vartheta=2$: two cones, two lobes along $z$ and a ring round the waist"),
        (3, 2, -2, "3d$_{xy}$", "$n_\\varphi=2$: two planes through the axis, four lobes in a cloverleaf"),
        (3, 2, 1, "3d$_{xz}$", "$n_\\vartheta=1$ flat plane and $n_\\varphi=1$ plane through the axis: four lobes"),
        (4, 3, 0, "4f$_{z^3}$", "$n_\\vartheta=3$: three polar nodes (two cones and the flat plane), lobes stacked along $z$"),
    ]
    _rows = [
        "| Orbital | $n$ | $\\ell$ | $m_\\ell$ | $n_r$ (spheres) | $n_\\vartheta$ (cones) | $n_\\varphi$ (planes) | $N_{\\text{nodes}}=n-1$ | what the nodes make |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _n, _l, _m, _name, _what in _cases:
        _g = node_geometry(_n, _l, _m)
        _a, _b, _c = len(_g["spheres"]), len(_g["cones"]), len(_g["planes"])
        assert (_a, _b, _c) == node_counts(_n, _l, _m) and _a + _b + _c == _n - 1
        _rows.append(f"| {_name} | {_n} | {_l} | {_m} | {_a} | {_b} | {_c} | {_a + _b + _c} | {_what} |")
    mo.vstack([
        mo.md("**Reading the table.** The three numbers of nodes are $n_r=n-\\ell-1$, $n_\\vartheta=\\ell-\\lvert m_\\ell\\rvert$ and "
              "$n_\\varphi=\\lvert m_\\ell\\rvert$, computed here from the actual zeros of the wave function:"),
        mo.md("\n".join(_rows)),
        mo.md(r"""
        So the shape of an orbital is a statement about nodes. $\ell=n_\vartheta+n_\varphi$ says how many of the nodes cut across
        an angle (cones and planes together), $\lvert m_\ell\rvert=n_\varphi$ says how many of those are planes through the axis, and
        $n_r=n-\ell-1$ says how many more there are, as spheres. The cones and planes decide the direction in which the lobes point; the spheres decide
        how many shells there are. The real orbitals $p_x$, $p_y$, $p_z$ differ only in the direction of their one node.

        **Checked in the tests:** for fourteen orbitals, the numbers of spheres, cones and planes are
        $n_r=n-\ell-1$, $n_\vartheta=\ell-\lvert m_\ell\rvert$ and $n_\varphi=\lvert m_\ell\rvert$ (total $N_{\text{nodes}}=n-1$), and the wave function is zero (to
        one part in $10^5$ of its largest value) at random points on each of the stated surfaces. The drumhead modes are
        tested in the same way: the circles and diameters are where the wave vanishes, $-\nabla^2\psi=j^2\psi$ holds, and the
        zeros of the Bessel functions agree with the standard values.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 A short bridge to molecules

    Hydrogen is one of the very few problems in chemistry that can be solved exactly, because there is one electron and one
    centre of force. Anything larger loses the separation into a radial and an angular part. The next notebooks take the path
    from here to the labels of modern calculations: `from_atoms_to_molecules.py` (fixed nuclei,
    hybrids, methane), then `basis_sets_from_base_states.py` (which functions to put on the atoms), then `hartree_fock_mean_field.py` (one orbital per
    electron in the average field of the others, and why $2s$ lies below $2p$), then the DFT notebooks
    starting with `../dft/reading_a_dft_label.py`.

    **Takeaways for a slide**

    1. Each direction has its own family of nodes: spheres across $r$ ($n_r=n-\ell-1$), cones across $\vartheta$
       ($n_\vartheta=\ell-\lvert m_\ell\rvert$), planes through the axis across $\varphi$ ($n_\varphi=\lvert m_\ell\rvert$); the
       angular ones number $\ell=n_\vartheta+n_\varphi$ and all of them $N_{\text{nodes}}=n-1$. The shape of an orbital is the
       pattern of these nodes.
    2. A wave on a sphere has to be single-valued and finite, which makes $m_\ell$ and $\ell$ whole numbers; the $2\ell+1$
       harmonics of one $\ell$ share the angular energy $\ell(\ell+1)$.
    3. The radial wave sees the pull $-1/r$ plus a push away from the centre; the energy $-13.6\ \text{eV}/n^2$ depends
       on $n$ only, so each level holds $n^2$ orbitals ($2n^2$ with spin).
    4. The separation into $R\,Y$ needs one centre and a potential that depends on $r$ only. Atoms and molecules reuse the shapes but not the
       exact solution; `from_atoms_to_molecules.py` is next.

    **What was checked, and what was not.** All the numerical statements marked above are tested in `test_hydrogen.py`.
    The restriction $\lvert m_\ell\rvert\le\ell$, the full separation of the Schrödinger equation and the whole of
    section 7 are stated without derivation or calculation here.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    Free sources, checked on 2026-11-13 (the full table, with what each covers, is in `SOURCES.md`).

    - [Central forces (Physics LibreTexts, UC Davis Physics 9HC, 7.3)](https://phys.libretexts.org/Courses/University_of_California_Davis/UCD:_Physics_9HC__Introduction_to_Waves_Physical_Optics_and_Quantum_Theory/7:_Quantum_Theory_in_Three_Dimensions/7.3:_Central_Forces): the Laplacian in spherical coordinates and the separation into radial and angular parts (derivation).
    - [Spherical harmonics (LibreTexts, Pacific Union College, 6.2)](https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/06:_The_Hydrogen_Atom/6.02:_The_Wavefunctions_of_a_Rigid_Rotator_are_Called_Spherical_Harmonics): the angular solutions (intro).
    - [Hydrogen orbitals and the three quantum numbers (LibreTexts, Pacific Union College, 6.4)](https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/06%3A_The_Hydrogen_Atom/6.04%3A_Hydrogen_Atomic_Orbitals_Depend_upon_Three_Quantum_Numbers): $n$, $\ell$, $m_\ell$ (intro).
    - [The hydrogen atom (Physics LibreTexts, Fowler, 4.6)](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Quantum_Mechanics_%28Fowler%29/04:_Angular_Momentum_Spin_and_the_Hydrogen_Atom/4.06:_The_Hydrogen_Atom): the radial equation and its Laguerre-polynomial solutions (derivation).
    - [Orbital angular momentum and the p-orbitals (LibreTexts, Grinnell College, 6.6)](https://chem.libretexts.org/Courses/Grinnell_College/CHM_364:_Physical_Chemistry_2_%28Grinnell_College%29/06:_The_Hydrogen_Atom/6.06:_Orbital_Angular_Momentum_and_the_p-Orbitals): $p_x$ and $p_y$ as combinations of the $m_\ell=\pm1$ functions, real versus complex (intro).
    """)
    return


if __name__ == "__main__":
    app.run()
