import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from field_quanta_plots import (
        chain_figure,
        counting_figure,
        double_slit_figure,
        electron_field_figure,
        ladder_figure,
        packet_figure,
        stiffness_figure,
    )

    return (
        chain_figure,
        counting_figure,
        double_slit_figure,
        electron_field_figure,
        ladder_figure,
        mo,
        packet_figure,
        plt,
        stiffness_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # What Makes Every Excitation the Same?

    Fire electrons one at a time at two narrow slits. Each electron arrives at a single point of a screen, and
    after many arrivals the points build up a pattern of fringes. The accelerating voltage $V$ of the electron gun
    fixes the energy of **every** electron that arrives: each has gained the energy $eV$. Turning the intensity of
    the gun up or down changes how many electrons arrive per second, and never the energy of each one. The same is
    true of light: a dimmer lamp of the same colour sends fewer photons, and each photon has the same energy.

    This notebook asks why. Why is a count allowed to change while the unit stays fixed, and why are all
    electrons the same unit? The answer has two parts, and the first step is to separate two different kinds of
    whole number that both go by the name "quantization".

    **Read first:** the plane-wave, oscillator and quantization notebooks (`../plane_waves`, `../oscillator`,
    `../quantization`).

    **Colours.** Figures and displayed equations give each quantity one colour throughout this series: blue for
    the height $H$ and $\operatorname{Re}\psi$, lime for $\operatorname{Im}\psi$, light blue for the complex wave
    $\psi$, orange for $\omega$ and $E$, green for $k$, magenta for $T$ and $\lambda$, purple for the phase
    $\theta$. Running text is never coloured.

    **Contents**

    1. The double slit: the voltage sets the energy, the intensity sets the number.
    2. Two kinds of whole number.
    3. One electron in vacuum: no energy levels, and a located electron is a packet.
    4. A toy field: a ring of masses on springs.
    5. Each mode counts quanta: a ladder with equal steps.
    6. The properties belong to the field.
    7. Indistinguishable quanta, and the rule for electrons.
    8. Both together, and the limits of the picture.

    Sections 4 and 5 use units in which $\hbar=1$, where $\hbar$ is the constant that converts a frequency into an
    energy. Sections 1, 3 and 6 use eV and nm.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Excitation** or **quantum**: one unit of energy and momentum carried by a field, such as one electron
          or one photon.
        - **Field**: a quantity defined at every point of space (for a toy version, at every mass of a chain)
          that can be disturbed; its disturbances travel as waves.
        - **Mode**: one wave of the field with a definite $k$ and a definite frequency $\omega$. Every disturbance
          is a sum of modes.
        - **Dispersion relation**: the rule $\omega(k)$ giving the frequency of the mode with wave number $k$.
        - **Occupation**: the number of quanta present in a mode.
        - **Packet**: a sum of waves with a spread of $k$, which is localised near one place.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 The double slit: the voltage sets the energy, the intensity sets the number

    An electron that has gained the energy $eV$ moving through the voltage $V$ has the wave number $k$ given by

    $$
    \frac{\hbar^2\textcolor{#6B913C}{k}^2}{2m_e} = eV ,
    $$

    from the free-particle energy of the quantization notebook. Its wavelength is $\lambda=2\pi/k$. For two slits a
    distance $d$ apart and a screen at distance $L$, the bright fringes are a distance $\lambda L/d$ apart, under
    the broad envelope of a single slit. The figure uses $d=500$ nm, slit width $100$ nm and $L=0.5$ m.

    The left panel shows where a chosen number of electrons land, as a histogram, with the predicted pattern as the
    red curve. Each electron contributes one count. The right panel shows the wavelength for every voltage.

    - Move the **voltage**: the energy of every electron changes, the wavelength shrinks, and the fringes move
      closer together.
    - Move the **number of electrons**: the pattern fills in. No electron changes its energy.

    The gun does not make electrons. It frees electrons that are already present in a heated metal and then
    accelerates them through the voltage $V$.[^gun]

    [^gun]: In a metal, the electrons are quanta of the electron field that occupy states of the metal. Heating
        the metal (or applying a strong electric field) gives some of them enough energy to leave its surface.
        The electron count is conserved in such processes. A new electron can appear only together with a
        positron (an electron-like quantum of opposite charge), which needs at least $2m_ec^2\approx1.02$ MeV.
    """)
    return


@app.cell
def _(mo):
    slit_volts = mo.ui.slider(10, 400, step=5, value=100, label="voltage $V$ (V)", show_value=True)
    slit_count = mo.ui.slider(20, 3000, step=20, value=300, label="electrons detected", show_value=True)
    return slit_count, slit_volts


@app.cell
def _(double_slit_figure, mo, plt, slit_count, slit_volts):
    _fig = double_slit_figure(slit_volts.value, slit_count.value)
    _out = mo.vstack([mo.hstack([slit_volts, slit_count], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 Two kinds of whole number

    The word "quantization" is used for two different things.

    | | Energy levels of **one particle** | Quanta of **a field** |
    |---|---|---|
    | What is a whole number | the label of an allowed state ($n$ in hydrogen, the box or the ring) | the count of identical quanta present |
    | Where it comes from | a condition that the wave must satisfy where its region closes or confines it | the equal spacing of a mode's energy ladder (for light), or the rule of at most one per mode (for electrons) |
    | Present for a free particle in vacuum? | no: every energy is allowed | yes: quanta are counted anywhere |
    | What it explains | why hydrogen has particular energies | why all electrons (or all photons of one mode) are the same, and why intensity changes only the number |

    The quantization notebook was about the left column. A free electron has every energy, and whole numbers
    appear only when a condition is imposed. This notebook is about the right column. The two are independent.
    An electron in an atom is one quantum of the electron field (right column) that occupies a bound state of the
    nucleus's attraction (left column); section 8 returns to this.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 One electron in vacuum: no levels, and a located electron is a packet

    A free electron has the wave $e^{i(kx-\omega t)}$ for every $k$, and so every energy
    $\hbar^2k^2/2m_e$ above zero is allowed (section 1 of the quantization notebook). In vacuum nothing selects
    particular values, and there is no level label $n$.

    A wave with exactly one $k$ is spread evenly over all space, so it describes no electron *in a place*. To
    describe an electron near a point, add waves of different $k$ into a **packet**. The narrower the packet in
    position, the wider it must be in $k$. For the packet below, $\sigma_x$ is its width in position and
    $\sigma_k$ its spread in $k$ (each is the standard deviation of the corresponding probability), and

    $$
    \sigma_x\,\sigma_{\textcolor{#6B913C}{k}} = \tfrac12 .
    $$

    Two consequences follow.

    - A **definite energy** needs a definite $k$, and so a packet that is long.
    - A **located electron** has a spread of $k$, and so a spread of energy of about $\hbar^2\sigma_k^2/2m_e$ if it
      is not moving overall. For $\sigma_x=1$ nm this is about $0.01$ eV. This spread belongs to the packet, which
      is a superposition. It does not mean that a particular energy is "selected".

    The figure centres the packet at $k=4$ per nm so that the oscillation can be seen. For an electron that is
    not moving overall, the centre is $k=0$ and the real part has no oscillation. In the gun of section 1, the
    packet is long, so $\sigma_k$ is small and the energy is nearly definite.
    """)
    return


@app.cell
def _(mo):
    packet_width = mo.ui.slider(0.3, 3.0, step=0.1, value=1.0, label=r"width in position $\sigma_x$ (nm)", show_value=True)
    return (packet_width,)


@app.cell
def _(mo, packet_figure, packet_width, plt):
    _fig = packet_figure(packet_width.value)
    _out = mo.vstack([packet_width, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 A toy field: a ring of masses on springs

    To see what a field and its quanta are, take a system small enough to describe completely. Let $N$ equal
    masses $m$ lie along a line with spacing $a$, each joined to its neighbours by identical springs of stiffness
    $K$. Each mass moves up and down. The last mass is joined to the first, so that the chain has no ends.[^ends]
    This chain is the toy version of a field: a quantity (the displacement) defined at every position that can be
    disturbed.

    Newton's law for mass number $i$ is
    $m\,\ddot x_i = K\,(x_{i+1} - 2x_i + x_{i-1})$. A **mode** is a wave of definite $k$ in which every mass moves
    as $x_i = \cos(k\,a i-\omega t)$, with $k=2\pi j/(Na)$ for $j=0,1,\dots,N-1$ so that the wave closes on itself.
    Substituting into Newton's law, using $\cos(\theta\pm ka)$ for the neighbours, gives

    $$
    -m\textcolor{#FF8021}{\omega}^2 = -4K\sin^2\!\left(\frac{\textcolor{#6B913C}{k}a}{2}\right)
    \quad\Longrightarrow\quad
    \textcolor{#FF8021}{\omega}(\textcolor{#6B913C}{k}) = 2\sqrt{\frac{K}{m}}\,\left|\sin\frac{\textcolor{#6B913C}{k}a}{2}\right| .
    $$

    This rule, frequency as a function of wave number, is the **dispersion relation** of the chain. It is
    fixed by $m$ and $K$ alone. Every disturbance of the chain is a sum of these modes, and each mode moves
    independently of the others. For long waves ($ka\ll1$) the relation becomes $\omega=v|k|$ with $v=a\sqrt{K/m}$,
    the speed of sound along the chain.

    The figure shows one mode as a running wave, and the dispersion relation with that mode marked. Move the
    slider through the modes and the time.

    [^ends]: A ring of masses (equivalently, a chain whose last mass is joined to its first) is bookkeeping that
        lets the modes be counted: $k$ takes the discrete values $2\pi j/(Na)$. A field filling infinite space
        has no such closing condition, and every $k$ is allowed. Letting $N$ grow makes the values of $k$ crowd
        together into a continuum, and nothing else in this notebook changes. The whole numbers here label
        *which mode*, not how many quanta (section 5).
    """)
    return


@app.cell
def _(mo):
    chain_j = mo.ui.slider(0, 12, step=1, value=3, label="mode number $j$", show_value=True)
    chain_t = mo.ui.slider(0, 12, step=0.1, value=0.0, label="time $t$", show_value=True)
    return chain_j, chain_t


@app.cell
def _(chain_figure, chain_j, chain_t, mo, plt):
    _fig = chain_figure(24, chain_j.value, chain_t.value)
    _out = mo.vstack([mo.hstack([chain_j, chain_t], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Each mode counts quanta: a ladder with equal steps

    In the chain, all the masses taking part in one mode move together, so the mode is described by a single
    number: its **size** $u$ at a given moment (how far the whole pattern is stretched). Newton's law gives
    $\ddot u=-\omega^2u$: the size obeys the equation of a mass on a spring, with the frequency $\omega$ of the
    mode. Each mode is therefore a harmonic oscillator of frequency $\omega$.

    In quantum mechanics, such an oscillator has a wave function $\psi(u)$, the amplitude for the mode to have
    size $u$. Its potential energy $\tfrac12\omega^2u^2$ grows without limit, so $\psi(u)$ must die away at very
    large $|u|$, as in the oscillator notebook. That requirement allows only the energies

    $$
    \textcolor{#FF8021}{E}_n = \left(n+\tfrac12\right)\hbar\textcolor{#FF8021}{\omega},
    \qquad n=0,1,2,\dots
    $$

    The right panel of the figure shows $\psi_n(u)$. The axis $u$ is the size of the mode. It is **not** a place
    in space: the conditions at large $|u|$ are limits of the size of the disturbance, not edges of the region
    the field fills. This is why the whole number $n$ appears even when the field fills all of space.

    **Why $n$ counts quanta.** The levels are separated by exactly $\hbar\omega$, whatever $n$ is. Raising $n$
    by one always costs the same energy, and for a running wave it always adds the same momentum $\hbar k$. So
    the state with level $n$ is "$n$ identical units of energy $\hbar\omega$ and momentum $\hbar k$ are present". A
    state of the whole field is given by listing the occupation $n_k$ of every mode, and

    $$
    \textcolor{#FF8021}{E} = \text{(constant)} + \sum_{\textcolor{#6B913C}{k}} n_{\textcolor{#6B913C}{k}}\,\hbar\textcolor{#FF8021}{\omega}_{\textcolor{#6B913C}{k}} ,
    \qquad
    P = \sum_{\textcolor{#6B913C}{k}} n_{\textcolor{#6B913C}{k}}\,\hbar \textcolor{#6B913C}{k} .
    $$

    The constant is the sum of the $\tfrac12\hbar\omega$ terms, which does not change as quanta are added.

    The middle panel shows why equal spacing matters. For comparison, the levels of a particle in a box have steps
    that grow with $n$: adding one more unit would cost a different amount each time. Nothing about a box makes
    its levels "copies" of each other. For a mode of a field, every added quantum costs the same as the one
    before, so the quanta of a mode are the same.[^bosons]

    [^bosons]: This argument is complete for a field whose modes are ordinary oscillators, such as light. The
        electron field differs in how many quanta a mode may hold (section 7).
    """)
    return


@app.cell
def _(mo):
    ladder_n = mo.ui.slider(0, 6, step=1, value=3, label="quanta in the mode, $n$", show_value=True)
    ladder_omega = mo.ui.slider(0.5, 2.5, step=0.1, value=1.0, label=r"frequency $\omega$", show_value=True)
    return ladder_n, ladder_omega


@app.cell
def _(ladder_figure, ladder_n, ladder_omega, mo, plt):
    _fig = ladder_figure(ladder_omega.value, ladder_n.value)
    _out = mo.vstack([mo.hstack([ladder_n, ladder_omega], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 The properties belong to the field

    The size of the step, $\hbar\omega_k$, is set by the dispersion relation, which is set by $m$ and $K$. It does
    not belong to any one quantum. Stiffen every spring and the energy of every quantum of every mode rises
    together. The figure shows the chain's dispersion relation for springs $K$ (faint) and for a stiffer set.
    All quanta of the chain are alike, because they are all excitations of the same chain with the same $m$ and $K$.

    **The electron field.** The same reasoning applies with the electron field in place of the chain. The
    dispersion relation of the electron field is

    $$
    \textcolor{#FF8021}{E}_{\textcolor{#6B913C}{k}} = \sqrt{(\hbar \textcolor{#6B913C}{k} c)^2 + (m_ec^2)^2}
    \;\approx\; m_ec^2 + \frac{\hbar^2\textcolor{#6B913C}{k}^2}{2m_e}
    \quad(\text{slow electrons}),
    $$

    where $c$ is the speed of light and $m_e$ is a property of the field. The second form is the free-electron energy
    of section 1 with a constant added.

    - The **mass** fixes the dispersion relation, and so the energy and momentum of every electron. In
      particular, it fixes a floor: creating one electron, however gently, costs at least $m_ec^2$ (right panel
      below), about $0.511$ MeV. This is the cost of the first quantum, and it is a different thing from the
      $\tfrac12\hbar\omega$ of an empty mode in section 5.
    - The **charge** does not enter these energies. It sets how strongly the field responds to electric and
      magnetic fields, that is, its coupling to light.
    - The **voltage of the gun** in section 1 selects which $k$ the electrons occupy, through
      $\hbar^2k^2/2m_e=eV$. The energy of every detected electron is then fixed. The intensity is only the number of
      quanta.

    Every electron has the same mass and charge because every electron is a quantum of one field that has this
    mass and charge. These values belong to the field.
    """)
    return


@app.cell
def _(mo):
    stiff = mo.ui.slider(1.0, 4.0, step=0.1, value=2.5, label=r"stiffness of the springs, in units of $K$", show_value=True)
    return (stiff,)


@app.cell
def _(mo, plt, stiff, stiffness_figure):
    _fig = stiffness_figure(stiff.value)
    _out = mo.vstack([stiff, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    elec_volts = mo.ui.slider(10, 400, step=5, value=100, label="voltage $V$ (V)", show_value=True)
    return (elec_volts,)


@app.cell
def _(elec_volts, electron_field_figure, mo, plt):
    _fig = electron_field_figure(elec_volts.value)
    _out = mo.vstack([elec_volts, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 Indistinguishable quanta, and the rule for electrons

    Suppose $n$ quanta are in one mode. Is there a question of which quantum is which? In the description by
    occupation, a state is fully specified by the list $(n_1,n_2,\dots)$, the number in each mode. Nothing else is
    left to specify: there is no name attached to a quantum, and so nothing to swap. This is what "identical"
    means here. It is a statement about how states are counted, and the counts differ from those of named objects.

    The figure shows the number of different states of $N$ quanta spread over $M$ modes for three cases.

    - **Named:** each quantum has its own label, so which one is in which mode matters: $M^N$ states.
    - **Identical, any number per mode** (light): only the occupations matter: $\binom{M+N-1}{N}$ states.
    - **Identical, at most one per mode** (electrons): $\binom{M}{N}$ states, and none when $N>M$.

    **The rule for electrons.** The energy ladder of section 5 has infinitely many equal rungs, which suits
    light. A mode of the electron field holds **0 or 1** electron, so its ladder stops after one rung. Two
    electrons cannot be in the same mode with the same spin, which is the exclusion principle, and it is why
    atoms have layered shells of electrons. The identity of electrons still follows the argument of sections 4 to
    6, because the mass and charge are the field's. What changes is the occupation rule.

    This notebook **states** the rule for electrons and does not derive it. It is tied to the electron having
    spin $\tfrac12$, which is the subject of the exchange part of the spinor notebook, not yet written. In
    particular, the decay condition used in section 5 to find the ladder does not produce the 0-or-1 rule: that
    comes from how states of several electrons change under an exchange.
    """)
    return


@app.cell
def _(mo):
    count_modes = mo.ui.slider(2, 10, step=1, value=6, label="number of modes $M$", show_value=True)
    count_quanta = mo.ui.slider(2, 8, step=1, value=5, label="most quanta shown", show_value=True)
    return count_modes, count_quanta


@app.cell
def _(count_modes, count_quanta, counting_figure, mo, plt):
    _fig = counting_figure(count_modes.value, count_quanta.value)
    _out = mo.vstack([mo.hstack([count_modes, count_quanta], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 Both together, and the limits of the picture

    An electron in a hydrogen atom is a quantum of the electron field. It is a quantum in the sense of the right
    column of section 2: it has the field's mass and charge, and it is counted, one electron at a time. It also
    occupies a bound state, which the nucleus's attraction confines, so its energy is one of the discrete levels
    $-13.6\ \text{eV}/n^2$ of the left column. The first fact explains why every electron is the same. The second
    explains which energies a particular atom has.

    **What this explanation does not do.**

    - It does not say why the electron field has the mass and charge that it has. These are inputs, as the spring
      constant and the mass were inputs for the chain.
    - It does not derive the 0-or-1 rule from the oscillator. That rule is stated, with the reason deferred (section 7).
    - It describes quanta as clean, independent units, which holds exactly when the field's modes do not interact.
      In real theories the electron field interacts with the electromagnetic field and with itself, and
      "one electron" becomes an approximate description: a quantum is a good unit when the interaction is weak or
      the particles are far apart.
    - The chain is a toy. It shows the mechanism of equal steps and field-wide properties, and the real electron
      field is not a chain of masses on springs.

    ## Summary

    - The voltage of an emitter fixes the energy of every electron it sends, and the intensity fixes only how many.
    - There are two different whole numbers. The labels of the **levels of one particle** come from conditions on
      the wave (bound states). The **count of quanta** comes from the field's modes.
    - A free electron in vacuum has every energy. A located electron is a packet, with a spread of $k$ and of
      energy that comes from the packet.
    - Each mode of a field is an oscillator, whose energies are spaced by exactly $\hbar\omega$. The level $n$
      counts quanta, and each added quantum adds the same energy $\hbar\omega$ and momentum $\hbar k$, so the quanta of
      a mode are the same.
    - The step $\hbar\omega(k)$ is fixed by the dispersion relation, which belongs to the field. The electron
      field's mass and charge are properties of the field, so every electron has them.
    - Quanta are identical in the sense that a state lists occupations and carries no label for "which one". For
      electrons the occupation of a mode is 0 or 1, which is stated here and left to the spinor notebook.

    The numerical functions are in `field_quanta.py`, with tests in `test_field_quanta.py`, and the figures are in
    `field_quanta_plots.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Sources.** The double slit follows Chapter 1 of R. P. Feynman, R. B. Leighton and M. Sands, *The Feynman Lectures on Physics*, Vol. III: Quantum Mechanics (Caltech online edition, edited by M. A. Gottlieb and R. Pfeiffer, 2013), <https://www.feynmanlectures.caltech.edu/III_toc.html>. The ring of masses on springs, its modes and the ladder of equally spaced levels are standard results for coupled oscillators; the toy field and the argument of sections 4 to 8 were written for this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
