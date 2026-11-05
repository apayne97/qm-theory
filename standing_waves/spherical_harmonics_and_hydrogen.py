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
        numerical_levels,
        quantum_numbers,
    )
    from hydrogen_plots import (
        coordinates_figure,
        harmonic_figure,
        harmonic_gallery_figure,
        level_check_figure,
        orbital_figure,
        orbital_gallery_figure,
        pole_figure,
        radial_state_figure,
        veff_figure,
    )

    return (
        HARTREE_EV,
        coordinates_figure,
        degeneracy,
        harmonic_figure,
        harmonic_gallery_figure,
        hydrogen_energy,
        hydrogen_energy_ev,
        level_check_figure,
        mo,
        np,
        numerical_levels,
        orbital_figure,
        orbital_gallery_figure,
        plt,
        pole_figure,
        quantum_numbers,
        radial_state_figure,
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

    **Read first:** `standing_waves_in_boxes.py` (this folder), and `../quantization`, section 5, which states the hydrogen
    quantum numbers without deriving them. This notebook supplies the reasons.

    **Contents**

    1. From a box to an atom: new coordinates
    2. Standing waves on a sphere: the spherical harmonics
    3. The radial part: effective potential and energy levels
    4. Hydrogen orbitals
    5. A short bridge to molecules

    **How much is derived.** The full separation of the Schrödinger equation is not repeated here. The equations
    that come out of it are stated, and each statement is checked numerically in `test_hydrogen.py` (the list is at
    the end of section 5). Units are atomic units: lengths in bohr ($a_0 = 0.0529$ nm), energies in hartree
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
        - **Nodal line**: a line on the sphere where the harmonic is zero.
        - **Radial node**: a sphere around the nucleus on which $\psi$ is zero.
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

    1. Find all the waves on a sphere and the number $\ell(\ell+1)$ that goes with each (section 2).
    2. Put that number into the radial equation and find its allowed energies (section 3).
    3. Multiply the two parts (section 4).
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
    ## 2 Standing waves on a sphere: the spherical harmonics

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
    $\ell$ nodal lines on the sphere in total. They are of two kinds:

    | Kind of nodal line | How many | Where it comes from |
    |---|---|---|
    | circles through both poles (meridians) | $\lvert m_\ell\rvert$ | $\cos(m_\ell\varphi)$ or $\sin(m_\ell\varphi)$ |
    | circles of constant latitude | $\ell-\lvert m_\ell\rvert$ | zeros of the Legendre function in $\cos\vartheta$ |
    | **total** | $\ell$ | |

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
    ## 3 The radial part: effective potential and energy levels

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
    ## 4 Hydrogen orbitals

    Multiplying the two parts gives the orbital, $\psi_{n\ell m_\ell}=R_{n\ell}(r)\,Y_{\ell m_\ell}(\vartheta,\varphi)$. Its
    square gives the probability density. Two kinds of node combine: $n-\ell-1$ spherical shells from the radial part, and $\ell$
    nodal surfaces from the angular part (each nodal line of the sphere extends outward as a surface through the
    nucleus: a plane for a meridian, a cone for a circle of latitude). The total is $n-1$ nodal surfaces. The names are
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
    ## 5 A short bridge to molecules

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
      (radial function) $\times$ (real spherical harmonic), as in section 4. The radial function is replaced by a sum of
      Gaussians, which makes the integrals fast, and the sum is made large enough to be accurate. Mixing s and p functions on
      carbon gives the $sp^3$ hybrids that point to the corners of a tetrahedron in methane.

    Worked calculations that use these ideas are in `../../qm-calculations`: `carbon` (the carbon atom, its density
    and a p orbital), `methane` (its molecular orbitals and a C-H bond orbital) and `basis-functions` (the shapes of the
    basis functions on hydrogen and oxygen). How the labels of a modern calculation, such as the functional and the basis set, are read is explained in
    `../dft/reading_a_dft_label.py`.

    **Takeaways for a slide**

    1. A wave on a sphere has to be single-valued and finite, which gives whole numbers $m_\ell$ and $\ell$; the harmonic
       $Y_{\ell m_\ell}$ has $\ell$ nodal lines, and $2\ell+1$ of them share the same angular energy $\ell(\ell+1)$.
    2. The radial wave sees the pull $-1/r$ plus a push $\ell(\ell+1)/2r^2$ away from the centre, and has $n-\ell-1$ nodes.
    3. The energy $-13.6\ \text{eV}/n^2$ depends on $n$ only, so each level holds $n^2$ orbitals ($2n^2$ with spin).
    4. Atoms and molecules reuse these shapes: many electrons need orbitals in an average field, fixed nuclei, and
       atom-centred basis functions.

    **What was checked, and what was not.** All the numerical statements marked above are tested in `test_hydrogen.py`.
    The restriction $\lvert m_\ell\rvert\le\ell$, the full separation of the Schrödinger equation and the whole of
    section 5 are stated without derivation or calculation here.
    """)
    return


if __name__ == "__main__":
    app.run()
