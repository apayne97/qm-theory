import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from oscillator import energy
    from oscillator_plots import (
        levels_figure,
        shooting_figure,
        state_grid_figure,
        superposition_figure,
    )

    return (
        energy,
        levels_figure,
        mo,
        np,
        plt,
        shooting_figure,
        state_grid_figure,
        superposition_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Bound States: the Quantum Harmonic Oscillator

    A particle that cannot escape, such as an electron held by a nucleus, has a quantum state that dies away far
    from the centre. This single requirement allows only particular values of the energy. The simplest example in
    which the whole story can be seen is the **harmonic oscillator**: a particle pulled back towards $x=0$ by a
    force proportional to its distance from $x=0$, which corresponds to the potential energy

    $$
    V(x) = \tfrac12\, m\,\textcolor{#FF8021}{\omega}^2 x^2 .
    $$

    Here $m$ is the mass and $\omega$ is the angular frequency with which a classical particle would swing back and
    forth. This notebook shows what the allowed states look like, how they change in time, and why the energies
    are restricted.

    **Read first:** the plane-wave notebook (`../plane_waves`): $\omega$, $k$, $E=\hbar\omega$ and complex
    waves.

    **Colours.** Figures and displayed equations give each quantity one colour throughout this series: blue for
    the height $H$ and $\operatorname{Re}\psi$, lime for $\operatorname{Im}\psi$, light blue for the complex wave
    $\psi$, orange for $\omega$ and $E$, green for $k$, magenta for $T$ and $\lambda$, purple for the phase
    $\theta$. Running text is never coloured.

    **Contents**

    1. The allowed energies and the shapes of the states.
    2. The real and imaginary parts of a state, and how they turn in time.
    3. A superposition of two states: the probability swings back and forth.
    4. Why only these energies: the condition that the state dies away at both ends.

    All figures use units in which $\hbar = m = \omega = 1$, so that lengths and energies are pure numbers. Energies
    are then measured in units of $\hbar\omega$.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Wave function** $\psi(x)$: the complex number (amplitude) that describes the state at the position $x$.
          Its real part is the horizontal part of an arrow in the complex plane and its imaginary part is the
          vertical part.
        - **Probability density** $|\psi(x)|^2$: the squared length of the amplitude. The probability of finding
          the particle between $x$ and $x+dx$ is $|\psi(x)|^2\,dx$, so the total area under $|\psi|^2$ is 1.
        - **Energy eigenstate**: a state of definite energy $E$. It obeys the Schrödinger equation below.
        - **Node**: a point where a real function crosses zero.
        - **Bound**: unable to escape, so that $\psi(x)\to 0$ far from the centre.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 The allowed energies and the shapes of the states

    A state of definite energy $E$ is described by a wave function $\psi(x)$ that obeys the Schrödinger equation.
    For the oscillator, in the units above,

    $$
    -\tfrac12\,\textcolor{#5ECCF3}{\psi}''(x) + \tfrac12\,x^2\,\textcolor{#5ECCF3}{\psi}(x) = \textcolor{#FF8021}{E}\,\textcolor{#5ECCF3}{\psi}(x),
    $$

    where $\psi''$ is the second derivative of $\psi$ with respect to $x$. The first term is the kinetic energy and
    the second is the potential energy $V(x)$. A solution of this equation exists for every value of $E$, but for
    almost every $E$ it grows without limit far from the centre, and such a $\psi$ cannot describe a particle that is
    somewhere. Section 4 shows this. The values of $E$ for which the solution does die away are

    $$
    \textcolor{#FF8021}{E}_n = \left(n+\tfrac12\right)\hbar\textcolor{#FF8021}{\omega}, \qquad n = 0, 1, 2, \dots
    $$

    and the corresponding wave function is a real function $\psi_n(x)$ with **$n$ nodes**.[^zeropoint] The figure
    shows the potential (the parabola), each allowed energy as a dashed line, and each $\psi_n$ drawn on its own
    level. Between the points where a level meets the parabola, the particle would be allowed to be found by
    classical mechanics. Outside that region the wave function does not vanish at once but decays smoothly.

    [^zeropoint]: The lowest energy is $\tfrac12\hbar\omega$ and not 0. A state squeezed into a small region has a
        wide spread of momentum, and that spread carries energy. The balance between the squeezing and the pull of
        the potential fixes the lowest state.
    """)
    return


@app.cell
def _(mo):
    n_levels = mo.ui.slider(1, 8, step=1, value=5, label="highest level shown", show_value=True)
    return (n_levels,)


@app.cell
def _(levels_figure, mo, n_levels, plt):
    _fig = levels_figure(n_levels.value)
    _out = mo.vstack([n_levels, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Three features of the states are visible. The number of nodes of $\psi_n$ equals $n$. The states of even $n$
    are mirror-symmetric about $x=0$ and those of odd $n$ change sign under that mirror image. And as $n$ grows
    the state spreads over a wider region, because a particle of higher energy swings further before it turns.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 The real and imaginary parts of a state

    A state of definite energy $E_n$ changes in time only through a phase. Its full amplitude is

    $$
    \textcolor{#5ECCF3}{\psi}_n(x)\,e^{-i\textcolor{#FF8021}{E}_nt/\hbar},
    $$

    a real function $\psi_n(x)$ multiplied by an arrow of length 1 that turns at the rate $\omega_n = E_n/\hbar$.[^turning]
    At $t=0$ the arrow points along the real axis, so the whole state is real. As time passes the arrow turns, and
    the state acquires an imaginary part: the real part is $\psi_n(x)\cos(\omega_n t)$ and the imaginary part is
    $-\psi_n(x)\sin(\omega_n t)$. The figure shows both parts for the first six states. Blue is the real part and
    green is the imaginary part. The slider sets the time $t$.

    Every state keeps the same shape. Only its mixture of real and imaginary parts changes, and a state of higher
    $n$ turns faster. The probability density $|\psi_n|^2=\psi_n^2$ does not change at all, because the arrow
    always has length 1. This is why a state of definite energy is called **stationary**: nothing that can be
    measured about the position changes with time.

    [^turning]: This is the relation $E=\hbar\omega$ of the plane-wave notebook: a state of definite energy is an
        arrow that turns at the rate $E/\hbar$. Here the arrow multiplies a whole function of $x$ instead of a
        single plane wave.
    """)
    return


@app.cell
def _(mo):
    grid_time = mo.ui.slider(0, 2 * 3.14159265, step=0.05, value=1.0, label="time $t$ (units of $1/\\omega$)", show_value=True, full_width=True)
    return (grid_time,)


@app.cell
def _(grid_time, mo, plt, state_grid_figure):
    _fig = state_grid_figure(t=grid_time.value)
    _out = mo.vstack([grid_time, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 A superposition of two states

    A particle need not be in a state of definite energy. Adding two energy eigenstates with weights $c_a$ and $c_b$
    gives another allowed state,

    $$
    \textcolor{#5ECCF3}{\psi}(x,t) = c_a\,\textcolor{#5ECCF3}{\psi}_a(x)\,e^{-i\textcolor{#FF8021}{E}_at/\hbar} + c_b\,\textcolor{#5ECCF3}{\psi}_b(x)\,e^{-i\textcolor{#FF8021}{E}_bt/\hbar},
    $$

    with the weights chosen so that the total probability is 1. The two arrows now turn at different rates, so the
    two terms keep changing their relative phase, and the probability density, which depends on that relative
    phase, changes with time. The pattern repeats after the time $2\pi\hbar/|E_b-E_a|$.

    For neighbouring levels $E_b-E_a=\hbar\omega$, so the probability moves back and forth once in each classical
    period $2\pi/\omega$: the lump of probability swings like the classical particle. The red dashed line marks the
    average position. Choose two levels, the weight of the second one, and the time (in classical periods).
    Two levels that are two steps apart give a motion that repeats twice as fast, and the average position then
    stays at 0, because both states have the same mirror symmetry (the same is true for any two levels of the same
    parity).
    """)
    return


@app.cell
def _(mo):
    sup_a = mo.ui.dropdown(options=[0, 1, 2, 3, 4, 5], value=0, label="first level $n_a$")
    sup_b = mo.ui.dropdown(options=[0, 1, 2, 3, 4, 5], value=1, label="second level $n_b$")
    sup_weight = mo.ui.slider(0, 1, step=0.05, value=1.0, label="weight of the second level $c_b/c_a$", show_value=True)
    sup_time = mo.ui.slider(0, 2, step=0.02, value=0.0, label="time (classical periods)", show_value=True, full_width=True)
    return sup_a, sup_b, sup_time, sup_weight


@app.cell
def _(mo, np, plt, sup_a, sup_b, sup_time, sup_weight, superposition_figure):
    _na, _nb = sup_a.value, sup_b.value
    if _na == _nb:
        _coefficients = {_na: 1.0}
    else:
        _coefficients = {_na: 1.0, _nb: sup_weight.value}
    _fig = superposition_figure(_coefficients, 2 * np.pi * sup_time.value)
    _out = mo.vstack([mo.hstack([sup_a, sup_b, sup_weight], justify="start"), sup_time, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Why only these energies

    The Schrödinger equation of section 1 can be solved for any trial energy $E$. Start far to the left, where
    the state must be almost zero, and follow the solution to the right. The only condition imposed at the left end
    is that the state dies away there. The slider sets the trial energy.

    For a typical $E$ the solution looks reasonable near the centre and then bends away and grows without limit on the
    right (the figure cuts it off at the edges). Such a function cannot be a particle that is somewhere. If $E$ is
    raised a little, the solution bends the other way and grows downwards. Between the two there is a particular $E$
    for which the solution dies away on the right as well: that is an allowed energy $E_n$. The right panel shows the
    value of the solution at the far right end as the trial energy changes. It changes sign each time $E$ passes an
    allowed energy, so it is zero at those energies and only there.

    This is the same logic as for a string with fixed ends. A string can vibrate only in shapes that fit the two
    ends, and the condition that the state dies away at both ends of the oscillator plays the role of the fixed
    ends. Requiring the state to vanish at both ends, and not only at one, is what selects the energies.
    """)
    return


@app.cell
def _(mo):
    trial_energy = mo.ui.slider(0.05, 5.8, step=0.01, value=1.0, label=r"trial energy $E$", show_value=True, full_width=True)
    return (trial_energy,)


@app.cell
def _(mo, plt, shooting_figure, trial_energy):
    _fig = shooting_figure(trial_energy.value)
    _out = mo.vstack([trial_energy, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(energy, mo, np):
    _rows = ["| level $n$ | energy $E_n = (n+\\tfrac12)\\hbar\\omega$ | nodes of $\\psi_n$ |", "|---|---|---|"]
    for _n in range(6):
        _rows.append(f"| {_n} | {float(energy(_n)):g} | {_n} |")
    mo.md("Set the slider to one of these energies, for example 0.5 or 2.5, to see a solution that dies away on the right.\n\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A bound particle has a wave function that dies away at both ends. For the harmonic oscillator this allows
      only the energies $E_n=(n+\tfrac12)\hbar\omega$, evenly spaced by $\hbar\omega$.
    - The state of energy $E_n$ is a real function $\psi_n(x)$ with $n$ nodes, multiplied by an arrow that turns at
      the rate $E_n/\hbar$. Its real and imaginary parts change in time, but its probability density does not.
    - A superposition of two levels has a probability density that moves with the frequency
      $|E_b-E_a|/\hbar$. For neighbouring levels this is the classical frequency $\omega$.
    - For a trial energy that is not allowed, the solution grows without limit at one end. The allowed energies are
      the ones for which both ends are satisfied, like the fixed ends of a vibrating string.

    The numerical functions are in `oscillator.py`, with tests in `test_oscillator.py`. The bare real-part curves
    used as icons are produced by `make_icons.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Sources.** The energy levels and the wave functions of the harmonic oscillator are standard results found in any quantum mechanics text. The shooting demonstration of section 4 and the presentation of the real and imaginary parts turning in time were written for this notebook. The meaning of $\omega$ and $E=\hbar\omega$ follows the plane-wave notebook, which cites Feynman, Leighton and Sands, *The Feynman Lectures on Physics*, Vol. III (Caltech online edition, 2013).
    """)
    return


if __name__ == "__main__":
    app.run()
