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
        node_geometry,
        numerical_levels,
        quantum_numbers,
    )
    from hydrogen_plots import (
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
        veff_figure,
    )

    return (
        HARTREE_EV,
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
        veff_figure,
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
    around the centre and diameters through it. An atom has nodes that are surfaces, of three kinds: spheres, cones and
    planes. By section 5 the shape of every orbital can be read off by counting nodes of each kind.

    **Read first:** `standing_waves_in_boxes.py` (this folder), and `../quantization`, section 5, which states the hydrogen
    quantum numbers without deriving them. This notebook supplies the reasons.

    **Contents**

    1. From a box to an atom: new coordinates
    2. Nodes and their direction: string, drumhead, atom
    3. Standing waves on a sphere: the spherical harmonics
    4. The radial part: effective potential and energy levels
    5. Hydrogen orbitals, and reading an orbital from its nodes
    6. A short bridge to molecules

    **How much is derived.** The full separation of the Schrödinger equation is not repeated here. The equations
    that come out of it are stated, and each statement is checked numerically in `test_hydrogen.py` (the list is at
    the end of section 6). Units are atomic units: lengths in bohr ($a_0 = 0.0529$ nm), energies in hartree
    (1 hartree $=27.2$ eV), $\hbar=m_e=1$.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Spherical coordinates**: the distance $r$ from the centre, the polar angle $\vartheta$ measured from the
          $z$ axis ($0$ at the north pole, $\pi$ at the south pole) and the azimuth $\varphi$ around the axis.
        - **Spherical harmonic** $Y_{\ell m_\ell}(\vartheta,\varphi)$: a standing wave on the surface of a sphere.
        - **Orbital**: the wave function of one electron, $\psi_{n\ell m_\ell}$. Its square $\lvert\psi\rvert^2$ is the
          probability density.
        - **Node**: a point, line or surface where the wave is zero. The wave changes sign across it.
        - **Nodal line**: a line on the sphere where the harmonic is zero.
        - **Radial node**: a sphere around the nucleus on which $\psi$ is zero.
        - **Drumhead mode**: a standing wave of a circular membrane held fixed at its rim.
        - **Hartree**: the atomic unit of energy, $27.211$ eV; hydrogen's lowest level is $-\tfrac12$ hartree.
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
    $\ell(\ell+1)$, and it enters the radial equation as a repulsive term. So the plan is:

    1. Look at the nodes of a string and of a drumhead, to learn what a node is and which way it can point (section 2).
    2. Find all the waves on a sphere and the number $\ell(\ell+1)$ that goes with each (section 3).
    3. Put that number into the radial equation and find its allowed energies (section 4).
    4. Multiply the two parts and read the shapes (section 5).
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
    ## 2 Nodes and their direction: string, drumhead, atom

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

    On the sphere of section 3, the $\ell$ angular nodes split into $\lvert m_\ell\rvert$ that contain the axis and
    $\ell-\lvert m_\ell\rvert$ that are circles of latitude. The drum has no latitude circles, because it is flat, so
    all of its $\ell$ angular nodes contain the centre and $\ell=\lvert m_\ell\rvert$.
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
    Two differences from the string are worth noting. The drum's frequencies are proportional to the zeros
    $j_{\ell,\,n_r+1}$, which are not equally spaced, so they are not whole-number multiples of the lowest one; and the rim
    is a fixed boundary, whereas the atom's "rim" is infinity, where the wave simply dies away. What stays the same is
    the counting: **each coordinate has its own family of nodes and its own whole number.**

    **What $n$ does and does not order.** $n=n_r+\ell+1$ counts all the nodes, but it does not order the energies of
    the drum. The frequencies go as $j_{0,1}=2.405$, $j_{1,1}=3.832$, $j_{2,1}=5.136$ and $j_{0,2}=5.520$, so the mode with
    two nodes ($\ell=2$, $n_r=0$) lies below the mode with one node ($\ell=0$, $n_r=1$). In hydrogen, the levels
    with the same $n$ and different $\ell$ have the same energy, but this accidental degeneracy is a special property of
    the Coulomb pull, and not a general property of standing waves.

    **Three dimensions: an atom.** The atom has one more direction, the polar angle $\vartheta$, and so one more
    family of nodes. In three dimensions a node is a surface, and the three kinds are the three-dimensional versions
    of what we have seen:

    | Direction the node cuts across | String | Drumhead | Atom | Whole number |
    |---|---|---|---|---|
    | distance ($x$ or $r$) | $n-1$ points | $n_r=n-\ell-1$ circles | $n_r=n-\ell-1$ **spheres** | $n_r$ (radial) |
    | polar angle $\vartheta$ | none | none | $\ell-\lvert m_\ell\rvert$ **cones** (a flat plane through the centre counts as a cone) | $\ell$ |
    | azimuth $\varphi$ | none | $\lvert m_\ell\rvert=\ell$ diameters | $\lvert m_\ell\rvert$ **planes** through the $z$ axis | $m_\ell$ |
    | **total** $N_\text{nodes}$ | $n-1$ | $n_r+\ell=n-1$ | $n-1$ | |

    The azimuth row is the same on the drum and in the atom: the diameters of the drum are the traces, in a
    horizontal slice, of the atom's planes through the axis. The distance row is the same for the string, the drum and
    the atom. The polar-angle row is new, and the next section is about it: the nodes that cut across the polar angle
    and the azimuth together number $\ell$, and the part of them that cuts across the azimuth is $\lvert m_\ell\rvert$.

    **The rest of the notebook applies this table.** Given $(n,\ell,m_\ell)$, draw that many spheres, cones and
    planes, and the wave is positive and negative in the pieces between them. That is the shape of the orbital.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 Standing waves on a sphere: the spherical harmonics

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

    **Nodal lines.** A standing wave is described by where it vanishes. The harmonic $Y_{\ell m_\ell}$ has exactly
    $\ell$ nodal lines on the sphere in total. They are of two kinds, and each cuts across a different direction, as in
    the table of section 2:

    | Kind of nodal line | Cuts across | How many | Where it comes from |
    |---|---|---|---|
    | circles through both poles (meridians) | azimuth $\varphi$ | $\lvert m_\ell\rvert$ | $\cos(m_\ell\varphi)$ or $\sin(m_\ell\varphi)$ |
    | circles of constant latitude | polar angle $\vartheta$ | $\ell-\lvert m_\ell\rvert$ | zeros of the Legendre function in $\cos\vartheta$ |
    | **total** | | $\ell$ | |

    Reaching out from the centre, a meridian becomes a plane through the axis and a circle of latitude becomes a cone
    (the equator becomes the flat plane $z=0$).

    A harmonic with $m_\ell=\pm\ell$ has no circles of latitude (all nodal lines are meridians, a "sectoral" shape),
    and a harmonic with $m_\ell=0$ has no meridians (only circles of latitude, a "zonal" shape). More nodal lines
    means a shorter wavelength on the sphere and a larger $\ell(\ell+1)$, as for the box where more nodes meant a larger
    $E$.[^real]

    [^real]: The harmonics with $m_\ell\neq0$ are complex waves, $e^{im_\ell\varphi}$ times a function of $\vartheta$. The
        figures show the **real** harmonics: the real combinations $\cos(\lvert m_\ell\rvert\varphi)$ for $m_\ell>0$ and
        $\sin(\lvert m_\ell\rvert\varphi)$ for $m_\ell<0$. They have the same $\ell$, the same energy and the same nodal
        lines as the complex ones, but fixed directions, which is how orbitals are drawn in chemistry.
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
    _rows = [
        "| $\\ell$ | name | $2\\ell+1$ harmonics | $\\ell(\\ell+1)$ | nodal lines (total $\\ell$) |",
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
        $\ell(\ell+1)$; the number of nodal lines is $\ell$, split into $\lvert m_\ell\rvert$ meridians and
        $\ell-\lvert m_\ell\rvert$ circles of latitude; and the numerical solution of the polar equation has $c=0$
        exactly at whole $\ell$. **Stated, not shown:** the restriction $\lvert m_\ell\rvert\le\ell$ is taken from the
        form of the harmonics, not derived.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where the nodal lines sit

    The count is $\ell$ in total. The next question is where they go. There is one rule for each kind of line.

    **Meridians (cut across the azimuth).** The azimuth dependence of a real harmonic is $\cos(m_\ell\varphi)$ or
    $\sin(m_\ell\varphi)$. Each vanishes at $2\lvert m_\ell\rvert$ values of $\varphi$ in one turn, which are the
    $\lvert m_\ell\rvert$ planes through the $z$ axis (each plane holds two opposite meridians). The planes are equally
    spaced, $\pi/\lvert m_\ell\rvert$ apart in $\varphi$:

    $$
    \cos(m_\ell\varphi)=0\ \text{ at }\ \varphi=\frac{\pi}{2\lvert m_\ell\rvert}+k\,\frac{\pi}{\lvert m_\ell\rvert},\qquad
    \sin(\lvert m_\ell\rvert\varphi)=0\ \text{ at }\ \varphi=k\,\frac{\pi}{\lvert m_\ell\rvert},
    $$

    with $k=0,1,\dots,\lvert m_\ell\rvert-1$ (the same planes again for $k$ beyond that).

    **Circles of latitude (cut across the polar angle).** The polar dependence is the Legendre function
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
    Orange marks the circles of latitude (cut across $\vartheta$), green the planes through the axis (cut across
    $\varphi$), as in the table of section 2. The right panel is the polar part alone: each time the curve crosses zero is
    one circle of latitude. The switch only turns the planes by half their spacing, $\pi/(2\lvert m_\ell\rvert)$,
    about the axis ($\cos$ to $\sin$); the nodes keep the same number and the same spacing.

    **Worked example, $\ell=4$, $\lvert m_\ell\rvert=1$.** One plane and three circles of latitude, four in total. The
    polar function is proportional to $\sin\vartheta\,(7\cos^3\vartheta-3\cos\vartheta)=\sin\vartheta\,\cos\vartheta\,
    (7\cos^2\vartheta-3)$. Its zeros between the poles are

    $$
    \cos\vartheta=0,\ \ \cos\vartheta=\pm\sqrt{3/7}\quad\Longrightarrow\quad
    \vartheta=90^\circ,\ \ 49.1^\circ,\ \ 130.9^\circ .
    $$

    (The factor $\sin\vartheta$ vanishes only at the poles, which are not nodal lines.) The figure shows these three
    angles on the right, and the tests check them.

    **Three more things to notice.**

    - *Complex harmonics have fewer visible nodes.* The complex $Y_{\ell m_\ell}$ contains $e^{im_\ell\varphi}$, whose size is
      $1$ everywhere: it never vanishes, it only winds its phase $m_\ell$ times round the axis. So the complex harmonic
      vanishes only on the $\ell-\lvert m_\ell\rvert$ circles of latitude. The $\lvert m_\ell\rvert$ planes appear when
      two of them ($\pm m_\ell$) are combined into the real $\cos$ or $\sin$ form, which is the form used in the figures.
    - *The $z$ axis is a choice.* The $2\ell+1$ harmonics of one $\ell$ have the same energy, so any combination of them
      is an equally good standing wave, and so is the same pattern turned to point along another axis. What is fixed is
      the total: always $\ell$ nodal surfaces, split differently. At one extreme ($m_\ell=0$) all of them are circles of
      latitude; at the other ($\lvert m_\ell\rvert=\ell$) all are planes through the axis, equally spaced.
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
    ### The same idea as a Fourier series

    The harmonics have a second job besides describing electrons: they are the natural building blocks for **any**
    shape on a circle or a sphere, in the same way that sines are the building blocks for any shape along a line.

    - **On a line** (the string), any shape that is zero at the ends is a sum of the standing waves
      $\sin(n\pi x/L)$. That is a Fourier series.
    - **Around a circle**, any shape is a sum of the ring waves $\cos(m_\ell\varphi)$ and $\sin(m_\ell\varphi)$. That is
      the Fourier series on a circle, and it is the angular factor of the drum.
    - **On a sphere**, any shape is a sum of spherical harmonics, $f(\vartheta,\varphi)=\sum a_{\ell m_\ell}Y_{\ell m_\ell}$.
      It is the Fourier series for a sphere.

    In each case the number of nodes plays the part of frequency: $m_\ell$ counts the periods round the circle, and
    $\ell$ counts the nodal lines on the sphere. A smooth shape needs only a few low harmonics; a sharp feature needs many
    high ones. The weight of each harmonic is found by **projection**: multiply the shape by that harmonic and add up the
    result over the whole circle or sphere,

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
    ## 4 The radial part: effective potential and energy levels

    The angular number $\ell(\ell+1)$ goes into the equation for the distance. Writing $u(r)=rR(r)$ (a convenient
    form, because then the equation looks like one for a wave along a line), the radial equation in atomic units is

    $$
    -\frac12\,\frac{d^2u}{dr^2}+\Big[\underbrace{-\frac{1}{r}}_{\text{pull of the nucleus}}
    +\underbrace{\frac{\ell(\ell+1)}{2r^2}}_{\text{angular curvature}}\Big]u
    = \textcolor{#FF8021}{E}\,u .
    $$

    This is a wave along a line with a potential. The first term in the bracket is the attraction of the nucleus.
    The second term comes from the angular part: a wave that curves over the sphere costs kinetic energy, and the cost is
    larger the closer the electron is to the centre, because the same angle spans a smaller circle. It acts like a
    wall that pushes an electron with $\ell>0$ away from the nucleus. The sum is the **effective potential**, drawn
    for $\ell=0,1,2$ together with the energy levels that we will find. Nothing is added by hand: the level lines are
    the exact values $-1/(2n^2)$ from the next paragraphs.
    """)
    return


@app.cell
def _(plt, veff_figure):
    _fig = veff_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now apply the usual condition for a bound wave: $u$ must vanish at the nucleus, $u(0)=0$ (because $R$ is finite
    there), and must die away far from it, as in the oscillator. Only some energies allow that. We find them by
    putting the equation on a grid of distances $r_i = i\,h$ and replacing the second derivative by its finite-difference
    form. The equation becomes a matrix eigenvalue problem, and its eigenvalues are the allowed energies, its
    eigenvectors the radial waves. The result for each $\ell$ is a ladder of energies. Counting the sign changes
    of the $k$-th wave gives $k$ **radial nodes**, so the waves are labelled by

    $$
    n = \ell + 1 + (\text{number of radial nodes}),\qquad n=\ell+1,\ \ell+2,\ \dots,
    $$

    which is the third condition of `../quantization`: more nodes along $r$ means more energy, as for the box.

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
        "| $n$ | $\\ell$ | radial nodes $n-\\ell-1$ | numerical $E$ (hartree) | exact $-1/(2n^2)$ | difference |",
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
        $\ell=0,1,\dots,n-1$ (the radial node count $n-\ell-1$ must be zero or more), and each $\ell$ has $2\ell+1$
        values of $m_\ell$. So the level $n$ contains

        $$
        \sum_{\ell=0}^{n-1}(2\ell+1) = n^2
        $$

        orbitals. The degeneracy in $m_\ell$ comes from the symmetry of the sphere (every direction is equivalent). The
        degeneracy in $\ell$ is **not** explained by that symmetry: it is special to the $-1/r$ attraction, and it is
        removed as soon as the potential differs from $-1/r$, as it does in every atom with more than one electron.
        Counting each electron's spin doubles the numbers to $2n^2$.
        """),
        mo.md("\n".join(_rows)),
        mo.md(r"""
        **Checked in the tests:** the exact radial functions solve the radial equation, are normalised and mutually
        orthogonal, have $n-\ell-1$ nodes, and give $\langle r\rangle=\tfrac12\big(3n^2-\ell(\ell+1)\big)$ bohr. The
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
    ## 5 Hydrogen orbitals

    Multiplying the two parts gives the orbital, $\psi_{n\ell m_\ell}=R_{n\ell}(r)\,Y_{\ell m_\ell}(\vartheta,\varphi)$. Its
    square gives the probability density. Three kinds of node combine, as in the table of section 2: $n-\ell-1$ spheres from the radial part, and
    from the angular part $\ell-\lvert m_\ell\rvert$ cones and $\lvert m_\ell\rvert$ planes through the axis (each
    nodal line of the sphere extends outward as a surface through the nucleus). The total is $n-1$ nodal surfaces. The names are
    1s, 2s, 2p, 3s, 3p, 3d: the number is $n$ and the letter is $\ell$.

    The first figure shows the six lowest distinct orbitals in a plane through the axis (top) and as the probability of
    finding the electron at a given distance (bottom). Notice that more radial nodes means more peaks in the probability
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
    mean distance $\langle r\rangle=5$ bohr and the mean $\langle r^2\rangle=30$ bohr$^2$ of the $2p$ orbital.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading an orbital from its nodes

    The sliders below draw the nodes of any orbital, by kind: spheres (pink, cut across $r$), cones (orange, cut across
    $\vartheta$) and planes through the axis (green, cut across $\varphi$). The left panel is a plane through the $z$
    axis. A sphere appears as a circle, a cone as two straight lines through the centre, and a plane through the axis
    as the axis itself. The right panel is a horizontal slice seen from above, like the drum: spheres appear as circles
    and planes through the axis as diameters. If the slice at $z=0$ would itself be a node (a flat plane is one of the
    cones), the slice is taken a little above it, where the cones show up as circles.

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
def _(mo, node_geometry):
    _cases = [
        (1, 0, 0, "1s", "no node: one blob"),
        (2, 0, 0, "2s", "a sphere: a core and a shell of opposite sign"),
        (2, 1, 0, "2p$_z$", "one flat plane across $z$: two lobes, up and down"),
        (2, 1, 1, "2p$_x$", "one plane through the axis: two lobes, left and right"),
        (3, 0, 0, "3s", "two spheres: a core and two shells"),
        (3, 1, 0, "3p$_z$", "a sphere and a flat plane"),
        (3, 2, 0, "3d$_{z^2}$", "two cones: two lobes along $z$ and a ring round the waist"),
        (3, 2, -2, "3d$_{xy}$", "two planes through the axis: four lobes in a cloverleaf"),
        (3, 2, 1, "3d$_{xz}$", "one plane through the axis and one flat plane: four lobes"),
        (4, 3, 0, "4f$_{z^3}$", "three cones (one flat): lobes stacked along $z$"),
    ]
    _rows = [
        "| Orbital | $n$ | $\\ell$ | $m_\\ell$ | spheres | cones | planes | total $n-1$ | what the nodes make |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _n, _l, _m, _name, _what in _cases:
        _g = node_geometry(_n, _l, _m)
        _a, _b, _c = len(_g["spheres"]), len(_g["cones"]), len(_g["planes"])
        assert _a + _b + _c == _n - 1
        _rows.append(f"| {_name} | {_n} | {_l} | {_m} | {_a} | {_b} | {_c} | {_a + _b + _c} | {_what} |")
    mo.vstack([
        mo.md("**Reading the table.** The three numbers of nodes are $n-\\ell-1$, $\\ell-\\lvert m_\\ell\\rvert$ and "
              "$\\lvert m_\\ell\\rvert$, computed here from the actual zeros of the wave function:"),
        mo.md("\n".join(_rows)),
        mo.md(r"""
        So the shape of an orbital is a statement about nodes. $\ell$ says how many of the nodes cut across
        an angle (cones and planes together), $m_\ell$ says how many of those are planes through the axis, and $n$ says how many
        more there are, as spheres. The cones and planes decide the direction in which the lobes point; the spheres decide
        how many shells there are. The real orbitals $p_x$, $p_y$, $p_z$ differ only in the direction of their one node.

        **Checked in the tests:** for fourteen orbitals, the numbers of spheres, cones and planes are
        $n-\ell-1$, $\ell-\lvert m_\ell\rvert$ and $\lvert m_\ell\rvert$ (total $n-1$), and the wave function is zero (to
        one part in $10^5$ of its largest value) at random points on each of the stated surfaces. The drumhead modes are
        tested in the same way: the circles and diameters are where the wave vanishes, $-\nabla^2\psi=j^2\psi$ holds, and the
        zeros of the Bessel functions agree with the standard values.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 A short bridge to molecules

    Hydrogen is one of the very few problems in chemistry that can be solved exactly. What goes wrong with anything
    larger, and what is done about it?

    - **Many electrons.** With two or more electrons, each electron is pushed by the others as well as pulled by the
      nucleus, and the wave function depends on all the positions at once. It no longer separates into a radial and an
      angular part. The standard step is the **orbital (mean-field) idea**: give each electron its own orbital, and let it
      feel the nucleus plus the average push of the others. The orbitals keep the angular shapes of this notebook
      (s, p, d, ...), but the energies now depend on $\ell$ as well as on $n$, because the electrons shield the nucleus
      differently. Density functional theory is a more refined version of the same idea (see the
      decoder notebook).
    - **Moving nuclei.** The nuclei are thousands of times heavier than the electrons, so they move slowly. The
      **Born-Oppenheimer approximation** treats them as fixed points while the electrons find their lowest
      state, and then repeats this for each arrangement of the nuclei. In this notebook the nucleus was a fixed point
      from the start.
    - **More than one centre.** In a molecule the potential no longer has spherical symmetry, so the orbitals cannot be
      found by separating variables. The practical method, **LCAO** (linear combination of atomic-like orbitals), builds each
      molecular orbital as a sum of functions centred on the atoms. Each of these basis functions has the form
      (radial function) $\times$ (real spherical harmonic), as in section 5. The radial function is replaced by a sum of
      Gaussians, which makes the integrals fast, and the sum is made large enough to be accurate. Mixing s and p functions on
      carbon gives the $sp^3$ hybrids that point to the corners of a tetrahedron in methane.

    Worked calculations that use these ideas are in `../../qm-calculations`: `carbon` (the carbon atom, its density
    and a p orbital), `methane` (its molecular orbitals and a C-H bond orbital) and `basis-functions` (the shapes of the
    basis functions on hydrogen and oxygen). How the labels of a modern calculation, such as the functional and the basis set, are read is explained in
    `../dft/reading_a_dft_label.py`.

    **Takeaways for a slide**

    1. Each direction has its own family of nodes: spheres across $r$ ($n-\ell-1$), cones across $\vartheta$
       ($\ell-\lvert m_\ell\rvert$), planes through the axis across $\varphi$ ($\lvert m_\ell\rvert$); in total $n-1$. The shape of an orbital is the
       pattern of these nodes.
    2. A wave on a sphere has to be single-valued and finite, which makes $m_\ell$ and $\ell$ whole numbers; the $2\ell+1$
       harmonics of one $\ell$ share the angular energy $\ell(\ell+1)$.
    3. The radial wave sees the pull $-1/r$ plus a push away from the centre; the energy $-13.6\ \text{eV}/n^2$ depends
       on $n$ only, so each level holds $n^2$ orbitals ($2n^2$ with spin).
    4. Atoms and molecules reuse these shapes: many electrons need orbitals in an average field, fixed nuclei, and
       atom-centred basis functions.

    **What was checked, and what was not.** All the numerical statements marked above are tested in `test_hydrogen.py`.
    The restriction $\lvert m_\ell\rvert\le\ell$, the full separation of the Schrödinger equation and the whole of
    section 6 are stated without derivation or calculation here.
    """)
    return


if __name__ == "__main__":
    app.run()
