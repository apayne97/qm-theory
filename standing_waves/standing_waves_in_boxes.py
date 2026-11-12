import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Standing Waves in 1D, 2D and 3D")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from standing_waves import box_energy_ev, degeneracies, energy_units, levels
    from standing_waves_plots import (
        aspect_fan_figure,
        box2d_figure,
        counting_figure,
        kspace_figure,
        level_diagram_figure,
        line_figure,
        slices_figure,
    )

    return (
        aspect_fan_figure,
        box2d_figure,
        box_energy_ev,
        counting_figure,
        degeneracies,
        kspace_figure,
        level_diagram_figure,
        levels,
        line_figure,
        mo,
        plt,
        slices_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Standing Waves in One, Two and Three Dimensions

    A wave on a string fixed at both ends can only vibrate in certain patterns, and each pattern has its own
    frequency. The same is true of the wave that describes an electron confined to a region. This notebook follows
    that idea from a line, to a square, to a cube, and shows what changes at each step. The last section prepares the
    step from a cube to a sphere, which is the shape of an atom (the next notebook, `spherical_harmonics_and_hydrogen.py`).

    **Read first:** `../quantization`, section 3 (the box in one dimension). This notebook repeats its main result
    briefly and then goes further.

    **Contents**

    1. One dimension: whole numbers of half-wavelengths
    2. Two dimensions: a wave for each direction
    3. The allowed waves form a lattice, and energy is a distance
    4. Degeneracy: equal energies, and what breaks them
    5. Three dimensions
    6. Counting states: why more dimensions means more levels
    7. From a box to an atom

    Energies are measured in units of $E_1$, the lowest energy in a one-dimensional box of length $L_x$, which is
    $E_1 = \pi^2\hbar^2/(2 m_e L_x^2)$. For an electron in a box of side $L_x = 10$ Å (1 nm), $E_1 = 0.376$ eV.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Standing wave**: a wave pattern that does not travel; it only oscillates in place, with fixed points
          (nodes) where it is always zero.
        - **Node**: a point, line or surface where the wave function is zero.
        - **Boundary condition**: a requirement on the wave at the edge of the region. Here the wave must be zero
          at the walls of the box.
        - **Level**: one allowed energy. **State**: one allowed wave. Several states can share one level.
        - **Degeneracy**: the number of different states that have the same energy.
        - **Wave vector** $\mathbf k$: for a wave $\sin(kx)$, the number $k$ is $2\pi$ divided by the wavelength.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 One dimension: whole numbers of half-wavelengths

    An electron in a box of length $L$ has a wave function $\psi(x)$ that must be zero at both walls, $\psi(0)=\psi(L)=0$.
    A wave $\sin(kx)$ is zero at $x=0$ automatically. It is also zero at $x=L$ only if $kL$ is a whole multiple of $\pi$:

    $$
    k_n = \frac{n\pi}{L},\qquad n = 1, 2, 3, \dots
    $$

    That is the same as saying that $n$ half-wavelengths fit exactly between the walls. The Schrödinger equation of a
    free particle, $E = \hbar^2 k^2/2m_e$, then gives the energies

    $$
    E_n = \frac{\hbar^2 k_n^2}{2 m_e} = n^2\,\frac{\pi^2\hbar^2}{2 m_e L^2} = n^2 E_1 .
    $$

    Three things to take from this. The levels are labelled by a whole number because the wave has to fit. The levels
    are not evenly spaced: $E_n$ grows as $n^2$. And the lowest energy $E_1$ is not zero: squeezing the wave into a box
    forces it to curve, and the curvature costs energy. It also grows as the box gets smaller ($E_1\propto 1/L^2$).

    One more fact is used later. These waves are enough to build any shape of the string that is zero at both ends: every such
    shape is a sum $\sum_n c_n\sin(n\pi x/L)$ (a Fourier series). The drumhead and sphere sections of the next notebook
    extend this to round membranes and to the sphere.
    """)
    return


@app.cell
def _(mo):
    n_line = mo.ui.slider(1, 8, step=1, value=3, label=r"number of half-wavelengths $n$", show_value=True)
    return (n_line,)


@app.cell
def _(line_figure, mo, n_line, plt):
    _fig = line_figure(n_line.value)
    _out = mo.vstack([n_line, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 Two dimensions: a wave for each direction

    In a rectangular box with sides $L_x$ and $L_y$ the wave must be zero on all four walls. The simplest waves that do
    this are products, with a whole number for each direction:

    $$
    \psi_{n_x n_y}(x,y) = \sin\!\Big(\frac{n_x\pi x}{L_x}\Big)\,\sin\!\Big(\frac{n_y\pi y}{L_y}\Big),
    \qquad n_x, n_y = 1, 2, 3, \dots
    $$

    The energy is the sum of the two one-dimensional energies, because the kinetic energy of motion along $x$ and along
    $y$ simply add:

    $$
    E_{n_x n_y} = \frac{\pi^2\hbar^2}{2m_e}\left(\frac{n_x^2}{L_x^2}+\frac{n_y^2}{L_y^2}\right).
    $$

    The picture is the product of two strings. The wave changes sign across each **node line**: there are $n_x - 1$ node
    lines that cross the $x$ direction and $n_y - 1$ that cross the $y$ direction. The wave is blue where $\psi>0$ and
    red where $\psi<0$, and the probability of finding the electron is $|\psi|^2$.
    """)
    return


@app.cell
def _(mo):
    nx2 = mo.ui.slider(1, 6, step=1, value=2, label=r"$n_x$", show_value=True)
    ny2 = mo.ui.slider(1, 6, step=1, value=3, label=r"$n_y$", show_value=True)
    ratio2 = mo.ui.slider(0.75, 1.5, step=0.05, value=1.0, label=r"side ratio $L_y/L_x$", show_value=True)
    return nx2, ny2, ratio2


@app.cell
def _(box2d_figure, mo, nx2, ny2, plt, ratio2):
    _fig = box2d_figure(nx2.value, ny2.value, ratio2.value)
    _out = mo.vstack([mo.hstack([nx2, ny2, ratio2], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 The allowed waves form a lattice, and energy is a distance

    Define the wave vector $\mathbf k = (k_x, k_y)$ with $k_x = n_x\pi/L_x$ and $k_y = n_y\pi/L_y$. The energy is then

    $$
    E = \frac{\hbar^2}{2m_e}\,\big(k_x^2 + k_y^2\big) = \frac{\hbar^2}{2m_e}\,\lvert\mathbf k\rvert^2 .
    $$

    So the allowed waves are the points of a rectangular grid in the $(k_x, k_y)$ plane, and **the energy of a state is
    the squared distance of its grid point from the corner**. All points at the same distance have the same energy, and
    they lie on a circle. This is the picture to keep: the walls decide which grid points exist, and the distance
    from the origin decides the energy.
    """)
    return


@app.cell
def _(kspace_figure, mo, nx2, ny2, plt, ratio2):
    _fig = kspace_figure(ratio2.value, nx2.value, ny2.value)
    _out = mo.vstack([mo.md("The sliders of section 2 select the state (circled) and the side ratio."), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Degeneracy: equal energies, and what breaks them

    In a **square** box ($L_x = L_y$) the states $(n_x, n_y)=(1,2)$ and $(2,1)$ are different waves (one has a node line
    across $x$, the other across $y$) with the same energy $5E_1$. They sit on the same circle in the $k$-plane. This is
    **degeneracy**, and here it has a simple cause: the square looks the same after swapping $x$ and $y$, so a
    swapped wave has the same energy.

    Some degeneracies have no such reason. In the square, $(1,7)$, $(5,5)$ and $(7,1)$ all have the energy $50E_1$,
    because $1+49 = 25+25$. That is a coincidence of whole numbers ("accidental" degeneracy).

    Make one side longer and the symmetry is gone. The figure shows how the lowest states move apart as the ratio of the
    sides changes.
    """)
    return


@app.cell
def _(aspect_fan_figure, plt):
    _fig = aspect_fan_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The rule behind it: **degeneracy is a sign of symmetry**. The more symmetric the container, the more states share
    a level. A sphere is the most symmetric container of all, and the degeneracy of the hydrogen levels, $n^2$ states in
    the level $n$, is the result.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Three dimensions

    A box with three sides needs three whole numbers, and the energy is again a sum:

    $$
    \psi_{n_x n_y n_z} = \sin\!\Big(\frac{n_x\pi x}{L_x}\Big)\sin\!\Big(\frac{n_y\pi y}{L_y}\Big)\sin\!\Big(\frac{n_z\pi z}{L_z}\Big),
    \qquad
    E = \frac{\pi^2\hbar^2}{2m_e}\left(\frac{n_x^2}{L_x^2}+\frac{n_y^2}{L_y^2}+\frac{n_z^2}{L_z^2}\right).
    $$

    The node lines of the square become node **planes**: $n_x - 1$ planes perpendicular to $x$, and so on. The figure
    cuts the wave with three planes.
    """)
    return


@app.cell
def _(mo):
    nx3 = mo.ui.slider(1, 4, step=1, value=2, label=r"$n_x$", show_value=True)
    ny3 = mo.ui.slider(1, 4, step=1, value=1, label=r"$n_y$", show_value=True)
    nz3 = mo.ui.slider(1, 4, step=1, value=3, label=r"$n_z$", show_value=True)
    return nx3, ny3, nz3


@app.cell
def _(mo, nx3, ny3, nz3, plt, slices_figure):
    _fig = slices_figure(nx3.value, ny3.value, nz3.value)
    _out = mo.vstack([mo.hstack([nx3, ny3, nz3], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(degeneracies, level_diagram_figure, levels, mo, plt):
    _rows = ["| Level | Energy ($E_1$) | States | Which $(n_x,n_y,n_z)$ |", "|---|---|---|---|"]
    for _i, (_e, _s) in enumerate(levels((1.0, 1.0, 1.0), 15), start=1):
        _which = ", ".join("(" + ",".join(map(str, st)) + ")" for st in _s)
        _rows.append(f"| {_i} | {_e:.0f} | {len(_s)} | {_which} |")
    _figs = [level_diagram_figure(d, 15) for d in (1, 2, 3)]
    _out = mo.vstack([
        mo.md("In a **cube** the lowest levels and their degeneracies:"),
        mo.md("\n".join(_rows)),
        mo.hstack([mo.as_html(f) for f in _figs], justify="center"),
    ])
    for _f in _figs:
        plt.close(_f)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    In the cube the degeneracies are 1, 3, 3, 3, 1, 6. Most of them come from permuting the three axes: all the ways to
    order $(2,1,1)$ give $3$ states, all the ways to order $(3,2,1)$ give $6$. The same box with three different sides
    would have no degeneracy at all.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Counting states: why more dimensions means more levels

    How many states have an energy below $E$? In the $k$-plane this is the number of grid points inside a quarter
    circle, in the cube the number inside one eighth of a sphere. For large $E$ the count is close to the area or volume:

    $$
    N_{1D}(E)\approx\sqrt{E},\qquad N_{2D}(E)\approx\frac{\pi}{4}E,\qquad N_{3D}(E)\approx\frac{\pi}{6}E^{3/2},
    \qquad (E \text{ in units of } E_1).
    $$

    More dimensions means many more states at the same energy, and they pile up faster as $E$ increases. For a large
    box or a many-electron system the levels merge into an almost continuous band.
    """)
    return


@app.cell
def _(counting_figure, plt):
    _fig = counting_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(box_energy_ev, mo):
    _rows = ["| Box side $L$ (Å) | $E_1$ (eV) | Lowest 3D level $3E_1$ (eV) |", "|---|---|---|"]
    for _L in (0.1, 0.3, 1.0, 3.0, 10.0):  # in nm; shown in ångström
        _e1 = float(box_energy_ev(1, _L))
        _rows.append(f"| {_L * 10:g} | {_e1:.4g} | {3 * _e1:.4g} |")
    mo.vstack([
        mo.md("**Sizes.** An electron in a cubic box: confinement energy scales as $1/L^2$, and at atomic sizes "
              "(1 Å, the size of a chemical bond) it is tens of electronvolts."),
        mo.md("\n".join(_rows)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 From a box to an atom

    An atom is not a box: the electron is held by the pull of the nucleus, which gets weaker with distance, and there is
    no hard wall. Three things carry over unchanged, and one changes.

    - The wave must fit its surroundings, so only some waves are allowed and each has its own energy.
    - Each independent direction of the problem brings one whole number.
    - Symmetry produces degeneracy.
    - What changes is the shape. The natural directions are no longer $x$, $y$, $z$, but distance $r$, polar angle
      $\vartheta$ and azimuth $\varphi$ (see `../quantization`, section 5). The waves around the nucleus separate into
      a radial part and an angular part. The angular parts are the **spherical harmonics**, the standing waves on the
      surface of a sphere, and they are the subject of the next notebook.

    **Takeaways for a slide**

    1. A confined wave must fit; the fitting condition gives whole numbers and energy levels.
    2. In $d$ dimensions there are $d$ whole numbers, and the energy is a sum of squares.
    3. Symmetry shows up as several states with one energy (degeneracy), and breaking it splits them.
    4. Confinement costs energy, growing as $1/L^2$.
    """)
    return


if __name__ == "__main__":
    app.run()
