import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from quantization_plots import (
        box_figure,
        box_levels_figure,
        comparison_figure,
        free_wave_figure,
        hydrogen_levels_figure,
        ring_figure,
    )

    return (
        box_figure,
        box_levels_figure,
        comparison_figure,
        free_wave_figure,
        hydrogen_levels_figure,
        mo,
        np,
        plt,
        ring_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Where Whole Numbers Come From

    A wave with a definite energy and momentum, $\psi = e^{i(kx-\omega t)}$, has no reason to prefer one value of
    $k$ or $\omega$ over another. Replacing $\omega$ by $E/\hbar$ and $k$ by $p/\hbar$ does not change this: the
    constant $\hbar$ only converts between the language of waves and the language of particles. Yet the electron in
    an atom has only particular energies, labelled by whole numbers. This notebook shows where the whole numbers
    come from.

    The answer is that the whole numbers do not belong to the wave. They belong to **a condition that the wave must
    satisfy** where its region ends or closes on itself. A free wave satisfies no such condition, and every $k$ is
    allowed. A wave on a ring, in a box, or around a nucleus must fit its surroundings, and only particular values of
    $k$ fit.

    **Read first:** the plane-wave notebook (`../plane_waves`), and for section 4 the oscillator notebook
    (`../oscillator`).

    **Colours.** Figures and displayed equations give each quantity one colour throughout this series: blue for
    the height $H$ and $\operatorname{Re}\psi$, lime for $\operatorname{Im}\psi$, light blue for the complex wave
    $\psi$, orange for $\omega$ and $E$, green for $k$, magenta for $T$ and $\lambda$, purple for the phase
    $\theta$. Running text is never coloured.

    **Contents**

    1. A free wave: every value allowed, and the phase is arbitrary.
    2. A wave on a ring: the two ends must agree.
    3. A wave in a box: the wave must be zero at the walls.
    4. The same pattern in the oscillator.
    5. Hydrogen: three conditions, three whole numbers.

    Sections 1 and 2 use units in which $\hbar = m_e = 1$, where $m_e$ is the mass of the electron.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Phase** $\theta$: the angle of the arrow that represents a complex number. A wave $e^{i\theta}$ has real
          part $\cos\theta$ and imaginary part $\sin\theta$.
        - **Single-valued**: having exactly one value at each point of space.
        - **Boundary condition**: a requirement on the wave function where the region of motion ends, for example at
          a wall.
        - **Node**: a point where a real function crosses zero.
        - **Allowed energy**: an energy $E$ for which the Schrödinger equation has a solution that satisfies every
          condition of the problem.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 A free wave: every value allowed

    For a particle that feels no force, a state of definite energy is the plane wave

    $$
    \textcolor{#5ECCF3}{\psi}(x,t) = e^{i(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega} t)} = e^{i(px-\textcolor{#FF8021}{E}t)/\hbar},
    \qquad \textcolor{#FF8021}{E}=\hbar\textcolor{#FF8021}{\omega}=\frac{\hbar^2\textcolor{#6B913C}{k}^2}{2m_e}.
    $$

    The relation between $E$ and $k$ is the Schrödinger equation of a free particle. It holds for **every** $k$, so
    the energy can be any positive number. The right panel of the figure is a continuous curve: nothing selects some
    points on it.[^free]

    Any starting phase $\theta_0$ is also allowed: multiplying the whole wave by $e^{i\theta_0}$ gives another
    solution. This is the sense in which the phase can be anything. Such a change has no observable effect, because
    the probability density $|\psi|^2$ is the squared length of the arrow and a turned arrow has the same length (the
    shaded band in the left panel does not move when $\theta_0$ is changed). The phase of one wave relative to
    another does matter, as in the interference of the plane-wave notebook. The overall phase does not.

    [^free]: A plane wave has the same size everywhere, so it cannot be a particle that is somewhere. A real free
        electron is a wave packet, a sum of plane waves with a spread of $k$. This does not alter the point made
        here: the allowed values of $k$ form a continuum.
    """)
    return


@app.cell
def _(mo):
    free_k = mo.ui.slider(0.3, 3.5, step=0.05, value=1.7, label=r"wave number $k$", show_value=True)
    free_theta = mo.ui.slider(0, 6.28, step=0.05, value=0.8, label=r"starting phase $\theta_0$", show_value=True)
    return free_k, free_theta


@app.cell
def _(free_k, free_theta, free_wave_figure, mo, plt):
    _fig = free_wave_figure(free_k.value, free_theta.value)
    _out = mo.vstack([mo.hstack([free_k, free_theta], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 A wave on a ring: the two ends must agree

    Suppose the particle can move only around a circle of radius $R$. A position on the ring is given by the angle
    $\varphi$, measured in radians from a chosen starting point, so that the distance along the ring is $R\varphi$.
    The angle $\varphi=2\pi$ is the same place as $\varphi=0$.

    A wave of definite energy has the same form as before, $e^{ikR\varphi}$. It is convenient to write
    $q = kR$. The number $q$ is the number of wavelengths that fit around the circumference $2\pi R$, because the
    wavelength is $2\pi/k$:

    $$
    \textcolor{#5ECCF3}{\psi}(\varphi) = e^{i\textcolor{#6B913C}{q}\varphi},
    \qquad
    \textcolor{#6B913C}{q} = \frac{2\pi R}{\textcolor{#DF2CB2}{\lambda}} .
    $$

    Going once around the ring takes $\varphi$ from $0$ to $2\pi$, and the arrow turns through the angle $2\pi q$.
    The wave function must have **one** value at each place. At the seam, $\varphi=0$ and $\varphi=2\pi$ are the
    same place, so the arrow must return to where it started:

    $$
    e^{i\,2\pi \textcolor{#6B913C}{q}} = 1
    \quad\Longleftrightarrow\quad
    \textcolor{#6B913C}{q} = 0, \pm1, \pm2, \dots
    $$

    The arrow must complete a whole number of turns. If it did not, the wave function would jump at the seam, and
    a jump has an infinite second derivative, so the Schrödinger equation could not hold there. The size of the
    jump is $|e^{i2\pi q}-1| = 2|\sin\pi q|$, which is zero only for whole $q$.

    The figure shows the real part of the wave drawn on the ring (left, where the real part pushes the ring in or
    out), the same wave with the ring unrolled (middle), and the energy $E=\hbar^2q^2/2m_eR^2$ (right). Move the
    slider through a whole number and the ends meet.[^winding]

    [^winding]: The whole number $q$ is the number of times the arrow turns around the ring. A whole number of turns
        cannot change smoothly into another whole number, which is why the allowed values are separated.
    """)
    return


@app.cell
def _(mo):
    ring_q = mo.ui.slider(0, 5, step=0.05, value=2.5, label=r"wavelengths around the ring $q$", show_value=True, full_width=True)
    return (ring_q,)


@app.cell
def _(mo, plt, ring_figure, ring_q):
    _fig = ring_figure(ring_q.value)
    _out = mo.vstack([ring_q, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The momentum along the ring is $p=\hbar k=\hbar q/R$, so the angular momentum, momentum times radius, is

    $$
    L = R\,p = \textcolor{#6B913C}{q}\,\hbar .
    $$

    Angular momentum around the ring comes only in whole multiples of $\hbar$. Negative $q$ means travelling the
    other way round, with the same energy, so every level except $q=0$ is shared by two waves. The energy
    $E=\hbar^2q^2/2m_eR^2$ takes the values $0,\ \tfrac12,\ 2,\ \tfrac92,\dots$ (in units of $\hbar^2/m_eR^2$): the
    whole number $q$ is squared. Nothing was assumed about $\omega$ being a whole number. Only the count of turns
    around the ring is.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 A wave in a box: the wave must be zero at the walls

    Now let the particle move along a line between two walls at $x=0$ and $x=L$ that it cannot enter. Outside
    the box the wave function is zero. The wave function has no jump, so it must also be zero at the walls
    themselves:

    $$
    \textcolor{#5ECCF3}{\psi}(0)=0,\qquad \textcolor{#5ECCF3}{\psi}(L)=0 .
    $$

    Inside the box the particle is free, so a state of definite energy is a combination of a wave going right and
    a wave going left,

    $$
    \textcolor{#5ECCF3}{\psi}(x) = a\,e^{i\textcolor{#6B913C}{k}x} + b\,e^{-i\textcolor{#6B913C}{k}x}.
    $$

    The first condition, $\psi(0)=a+b=0$, gives $b=-a$ and so $\psi(x)=2ia\sin kx$: a standing wave. The second
    condition requires $\sin kL=0$, that is

    $$
    \textcolor{#6B913C}{k}L = n\pi,
    \qquad
    \textcolor{#FF8021}{E}_n = \frac{\hbar^2\textcolor{#6B913C}{k}^2}{2m_e} = \frac{n^2\pi^2\hbar^2}{2m_eL^2},
    \qquad n=1,2,3,\dots
    $$

    The figure shows the wave $\sin kx$ for a trial $k$. The value at the far wall (red dot) is generally not zero.
    The right panel plots that value against $kL/\pi$. It is zero at the whole numbers and only there. This is the
    same test as the sign change in the shooting figure of the oscillator notebook: start at one end, follow the
    solution, and ask whether it satisfies the condition at the other end.
    """)
    return


@app.cell
def _(mo):
    box_k = mo.ui.slider(0.5, 5.5, step=0.05, value=2.6, label=r"trial wave number, $kL/\pi$", show_value=True, full_width=True)
    return (box_k,)


@app.cell
def _(box_figure, box_k, mo, plt):
    _fig = box_figure(box_k.value)
    _out = mo.vstack([box_k, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The state with $kL=n\pi$ has $n-1$ nodes inside the box, and the wavelength is $2L/n$: a whole number of half
    wavelengths fits between the walls. A shorter wavelength means a larger $k$ and a larger energy, so the
    energy grows with the number of nodes.

    The energies of an electron depend on the length of the box as $1/L^2$. For $L=1$ nm the lowest level is
    $E_1=0.38$ eV. The figure shows the first eight levels as the box is made longer.[^spacing]

    [^spacing]: The spacing between neighbouring levels also falls as $1/L^2$. For a box of a few centimetres, or
        for any heavy particle, the levels are so close together that they cannot be told apart, and the energy
        looks continuous. This is why the free-particle picture of section 1 is a good description of an electron
        in a large region, and why whole numbers do not show up in everyday objects.
    """)
    return


@app.cell
def _(mo):
    box_length = mo.ui.slider(0.5, 6, step=0.1, value=1.5, label="box length $L$ (nm)", show_value=True, full_width=True)
    return (box_length,)


@app.cell
def _(box_length, box_levels_figure, mo, plt):
    _fig = box_levels_figure(box_length.value)
    _out = mo.vstack([box_length, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 The same pattern in the oscillator

    The three examples differ in the condition and share a structure:

    | | region | condition | what is allowed |
    |---|---|---|---|
    | ring | closes on itself | one value at the seam | $q=0,\pm1,\pm2,\dots$ |
    | box | ends at hard walls | $\psi=0$ at both walls | $kL=n\pi$ |
    | oscillator | ends smoothly | $\psi\to0$ far away on both sides | $E=(n+\tfrac12)\hbar\omega$ |

    In each case the wave inside the region is locally an ordinary wave. Its wavelength is set by the energy (and,
    where there is a force, by the potential). The condition at the edge or the seam then removes all energies
    except a discrete set. The whole number counts the nodes, or the turns of the arrow, that fit.

    The oscillator has no hard wall, so the condition is not that the wave vanishes at a point. It is that the wave
    dies away on both sides, as in the shooting figure of the oscillator notebook. The figure below shows the lowest
    three states of each system, drawn on their own energy levels: more energy, shorter wavelength, more nodes.
    """)
    return


@app.cell
def _(comparison_figure, mo, plt):
    _fig = comparison_figure()
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Hydrogen: three conditions, three whole numbers

    The electron in hydrogen feels the attraction of the nucleus, $V(r)\propto-1/r$, which depends only on the
    distance $r$ from the nucleus. A position in space is described by three coordinates: the distance $r$, the
    polar angle $\vartheta$ (measured from a chosen axis, from $0$ to $\pi$), and the azimuth $\varphi$ (the angle
    around that axis, from $0$ to $2\pi$). The wave function has to satisfy one condition in each coordinate, and
    each condition gives one whole number.

    | coordinate | condition | whole number |
    |---|---|---|
    | azimuth $\varphi$ | one value at the seam $\varphi=0\equiv2\pi$: exactly the ring of section 2 | $m_\ell=0,\pm1,\pm2,\dots$, with $L_z=m_\ell\hbar$ |
    | polar angle $\vartheta$ | the wave function stays finite on the axis, at $\vartheta=0$ and $\vartheta=\pi$ | $\ell=0,1,2,\dots$, with $\lvert m_\ell\rvert\le\ell$ |
    | distance $r$ | finite at the nucleus and dies away far from it, as in the oscillator | $n=\ell+1,\ell+2,\dots$ |

    The first row is the ring again. The second and third rows are the box and oscillator conditions, one for each
    remaining coordinate. The derivation of the second row and the form of the radial wave functions are the
    subject of the next part. The result for the energy is

    $$
    \textcolor{#FF8021}{E}_n = -\frac{13.6\ \text{eV}}{n^2},
    \qquad n=1,2,3,\dots
    $$

    The energy depends on $n$ only, so all the states with the same $n$ share a level. For a given $n$ the number of
    pairs $(\ell,m_\ell)$ is $\sum_{\ell=0}^{n-1}(2\ell+1)=n^2$. The figure shows the levels, separated by $\ell$, with
    the number of values of $m_\ell$ beside each.[^spin]

    The energies are negative because the electron is bound. At energies above zero the electron is free to leave,
    no condition at infinity can be imposed, and every energy is allowed: this is the continuum of section 1 again.

    [^spin]: Spin doubles each of these counts to $2n^2$. Spin is the subject of the spinor notebook.
    """)
    return


@app.cell
def _(mo):
    h_levels = mo.ui.slider(2, 5, step=1, value=4, label=r"levels shown, up to $n$", show_value=True)
    return (h_levels,)


@app.cell
def _(h_levels, hydrogen_levels_figure, mo, plt):
    _fig = hydrogen_levels_figure(h_levels.value)
    _out = mo.vstack([h_levels, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A free wave $e^{i(kx-\omega t)}$ allows every $k$ and every energy. The substitutions $E=\hbar\omega$ and
      $p=\hbar k$ do not change that, and the overall phase is unobservable.
    - Whole numbers appear when the wave must satisfy a condition: one value after a full circuit of a ring
      ($q$ whole turns of the arrow), zero at the walls of a box ($kL=n\pi$), or dying away at both ends of an
      oscillator.
    - The whole number counts nodes or turns that fit. The energy is a function of that number, such as $q^2$,
      $n^2$ or $n+\tfrac12$, so $\omega$ is not itself a whole number.
    - In hydrogen the azimuth, the polar angle and the distance each impose one condition, giving $m_\ell$, $\ell$ and
      $n$ and the energies $-13.6\ \text{eV}/n^2$.
    - These whole numbers explain the **levels** of a bound electron. They do not explain why there is exactly one
      electron, or why all electrons are identical. That question concerns the electron as a quantum of a field,
      which is a separate idea.

    The numerical functions are in `quantization.py`, with tests in `test_quantization.py`, and the figures are in
    `quantization_plots.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Sources.** The ring, the box and the hydrogen levels are standard results found in any quantum mechanics text; the free plane wave and the phase follow Chapters 7 and 16 of R. P. Feynman, R. B. Leighton and M. Sands, *The Feynman Lectures on Physics*, Vol. III: Quantum Mechanics (Caltech online edition, edited by M. A. Gottlieb and R. Pfeiffer, 2013), <https://www.feynmanlectures.caltech.edu/III_toc.html>. The side-by-side comparison of the three conditions was written for this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
