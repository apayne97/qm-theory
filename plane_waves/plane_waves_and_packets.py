import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from waves import (
        group_velocity,
        hole_amplitude,
        intensity,
        packet,
        spreading_time,
        two_hole_distances,
        wavelengths_in_lump,
        widths_travelled_before_spreading,
        width,
    )
    from wave_plots import (
        arrow_figure,
        band_shapes_figure,
        entropy_family_figure,
        flat_vs_lump_figure,
        interference_figure,
        interference_schematic,
        lump_buildup_figure,
        min_uncertainty_figure,
        moving_wave_figure,
        oscillator_energy_figure,
        packet_times_figure,
        phase_figure,
        probability_figure,
        rest_energy_phase_figure,
        space_cosine_figure,
        space_twist_figure,
        spacetime_figure,
        space_features_figure,
        spreading_figure,
        time_cosine_figure,
        time_features_figure,
        time_rotation_figure,
        travelling_features_figure,
        wave_surface_figure,
        width_tradeoff_figure,
    )

    return (
        arrow_figure,
        band_shapes_figure,
        entropy_family_figure,
        flat_vs_lump_figure,
        group_velocity,
        hole_amplitude,
        intensity,
        interference_figure,
        interference_schematic,
        lump_buildup_figure,
        min_uncertainty_figure,
        mo,
        moving_wave_figure,
        np,
        oscillator_energy_figure,
        packet,
        packet_times_figure,
        phase_figure,
        plt,
        probability_figure,
        rest_energy_phase_figure,
        space_cosine_figure,
        space_features_figure,
        space_twist_figure,
        spacetime_figure,
        spreading_figure,
        spreading_time,
        time_cosine_figure,
        time_features_figure,
        time_rotation_figure,
        travelling_features_figure,
        two_hole_distances,
        wave_surface_figure,
        width_tradeoff_figure,
        wavelengths_in_lump,
        widths_travelled_before_spreading,
        width,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Waves of Probability Amplitude: Energy, Momentum and the Plane Wave

    In quantum mechanics the state of a particle is described by an **amplitude**, a complex number that
    depends on position and time. The probability of finding the particle is the square of the amplitude's
    length. A particle of definite energy and definite momentum has an amplitude that is a wave:

    $$
    e^{\,i(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega} t)}
    $$

    This notebook explains what the two numbers $\omega$ and $k$ mean, how they are related to the energy
    and momentum of the particle, and why a superposition of waves with nearby values of $k$ behaves like a
    classical particle: a localized lump that moves at the classical velocity. The only mathematics assumed is
    trigonometry. Complex numbers are introduced in section 2.

    **Read first:** nothing; the only mathematics assumed is trigonometry.

    **Contents**

    1. A wave in trigonometric form: the phase, size, period, frequency, wavelength and wave number.
    2. The same waves with complex numbers, and why the two descriptions are equivalent.
    3. Adding waves: interference, with arrows.
    4. Why the waves must be complex for quantum amplitudes.
    5. Energy and momentum as the rates $\omega$ and $k$.
    6. Wave packets: when a lump moves like a particle.

    All figures use units in which lengths and times are pure numbers ($\hbar = m = 1$, where $m$ is the mass of
    the particle).
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Wave**: a quantity (such as the height of a water surface) that depends on position and time and is
          unchanged in shape when it is moved along at a steady speed.
        - **Phase** $\theta$: the angle inside a cosine, $\cos\theta$. The cosine repeats when the phase grows by $2\pi$.
        - **Size** $A$: the largest height of a real wave. (The word **amplitude** is kept for the complex number
          that describes a quantum state; section 4.)
        - **Period** $T$ and **angular frequency** $\omega=2\pi/T$: the time for one cycle, and the phase gained per unit time.
        - **Wavelength** $\lambda$ and **wave number** $k=2\pi/\lambda$: the length of one cycle, and the phase gained per unit length.
        - **Complex number**: a number $a+ib$, drawn as an arrow in a plane, with a length and an angle (its phase).
        - **Intensity** $I$: the squared length of the complex wave, $|\psi|^2$, proportional to the energy of the wave.
        - **Wave packet**: a sum of waves with nearby $k$, which forms a lump.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 A wave in trigonometric form

    The simplest wave is a quantity, such as the height of the surface of water, that varies as a cosine. Every
    such wave is built from one function, so we start with that function alone. It is then described in three
    steps: how it varies in time at one position, how it varies in space at one moment, and then both together.

    **Colour code.** As in the study-group slides, each quantity has one colour in the displayed equations and the figures (never in running text): height $H$ (blue), size $A$ (red), phase $\theta$ and phase difference $\delta$ (purple), angular frequency $\omega$ (orange), wave number $k$ (green), period $T$ and wavelength $\lambda$ (magenta), the complex wave $\psi$ (light blue), and the distance $r$ from a source (pink).

    **The phase.** Consider

    $$
    \textcolor{#4E67C8}{H}(\textcolor{#7030A0}{\theta}) = \textcolor{#F14124}{A}\cos\textcolor{#7030A0}{\theta} ,
    $$

    where $\theta$ is an angle in radians. The angle $\theta$ is called the **phase**: it says how far through its
    cycle the cosine has got. At $\theta=0$ the cosine is at a crest ($H=A$), at $\theta=\pi/2$ it crosses
    zero on the way down, at $\theta=\pi$ it is at a trough ($H=-A$), and at $\theta=2\pi$ it is back at the
    crest. The cosine therefore repeats whenever the phase grows by $2\pi$. The size $A$ is the largest
    height, and the phase decides where in the cycle the height is. The right panel shows the same value as an
    arrow of length $A$ turning on a circle: the phase is the angle of the arrow, and $H$ is its horizontal part.
    This picture returns in section 2.

    Everything below is this one function with a particular phase: $\theta=\omega t$ at a fixed position
    (section 1.1), $\theta=kx$ at a fixed moment (section 1.2), and $\theta=kx-\omega t$ in general
    (section 1.3). The slider sets the phase.
    """)
    return


@app.cell
def _(mo):
    phase_theta = mo.ui.slider(0, 4, step=0.02, value=0.7, label=r"phase $\theta / \pi$", show_value=True)
    return (phase_theta,)


@app.cell
def _(mo, np, phase_figure, phase_theta, plt):
    _fig = phase_figure(phase_theta.value * np.pi)
    _out = mo.vstack([phase_theta, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Oscillation in time

    At one fixed position the height oscillates as

    $$
    \textcolor{#4E67C8}{H}(t) = \textcolor{#F14124}{A}\cos(\textcolor{#FF8021}{\omega} t).
    $$

    - The **size** $A$ is the largest height reached.
    - The argument of the cosine is the phase $\theta=\omega t$. It grows by $\omega$ in every unit of time, so $\omega$ is the
      number of radians covered per unit of time. It is called the **angular frequency**.
    - The cosine repeats when its argument has grown by $2\pi$. This takes the **period**
      $$
      \textcolor{#DF2CB2}{T} = \frac{2\pi}{\textcolor{#FF8021}{\omega}},
      $$
      and the number of oscillations per unit of time is the **frequency** $f = 1/T = \omega/2\pi$.

    A large $\omega$ means a rapid oscillation.
    """)
    return


@app.cell
def _(mo):
    time_amp = mo.ui.slider(0.5, 2.5, step=0.1, value=1.5, label=r"size $A$", show_value=True)
    time_omega = mo.ui.slider(0.5, 6, step=0.1, value=2.0, label=r"angular frequency $\omega$", show_value=True)
    return time_amp, time_omega


@app.cell
def _(mo, plt, time_amp, time_cosine_figure, time_omega):
    _fig = time_cosine_figure(time_amp.value, time_omega.value)
    _out = mo.vstack([mo.hstack([time_amp, time_omega], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plt, time_features_figure):
    _fig = time_features_figure()
    _out = mo.vstack([
        mo.md(
            r"""
            **Reading the graph.** Each quantity of this section appears as a feature of the curve.

            - **$t$ and $H(t)$** are the horizontal and vertical positions of a point on the curve: at the time $t$ the height is $H(t)$ (dashed lines from the black point).
            - **$A$**, the size, is a *height*: the distance from the axis to a crest (red arrow).
            - **$T$**, the period, is a *time*: the horizontal distance from one crest to the next.
            - **$\omega$** is not a length on this graph. It is the rate at which the phase $\omega t$ grows, so it shows up on the phase scale along the top: the phase reaches $2\pi$ exactly when $t=T$, which is why $\omega=2\pi/T$. A larger $\omega$ packs more oscillations into the same time (lower panel).
            """
        ),
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Variation in space

    At one fixed moment the height varies along the line of positions as

    $$
    \textcolor{#4E67C8}{H}(x) = \textcolor{#F14124}{A}\cos(\textcolor{#6B913C}{k}x).
    $$

    Now the phase $\theta=kx$ grows by $k$ in every unit of distance, so **$k$ is the number of radians covered per
    unit of length**. It is called the **wave number**, and it is the counterpart in space of $\omega$. The
    cosine repeats after the distance

    $$
    \textcolor{#DF2CB2}{\lambda} = \frac{2\pi}{\textcolor{#6B913C}{k}},
    $$

    which is the **wavelength**, the counterpart in space of the period $T$. The number of wavelengths per unit
    of length is $1/\lambda = k/2\pi$.
    """)
    return


@app.cell
def _(mo):
    space_k = mo.ui.slider(0.5, 6, step=0.1, value=2.0, label=r"wave number $k$", show_value=True)
    return (space_k,)


@app.cell
def _(mo, plt, space_cosine_figure, space_k):
    _fig = space_cosine_figure(space_k.value)
    _out = mo.vstack([space_k, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plt, space_features_figure):
    _fig = space_features_figure()
    _out = mo.vstack([
        mo.md(
            r"""
            **Reading the graph.** The same graph with position in place of time.

            - **$x$ and $H(x)$** are the horizontal and vertical positions of a point on the curve.
            - **$A$** is again the height of a crest.
            - **$\lambda$**, the wavelength, is a *length*: the horizontal distance from one crest to the next.
            - **$k$** is the rate at which the phase $kx$ grows along the line. The phase reaches $2\pi$ when $x=\lambda$ (top scale), so $k=2\pi/\lambda$. A larger $k$ means a shorter wavelength (lower panel).
            """
        ),
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.3 A travelling wave

    A wave that varies in both time and space has the height

    $$
    \textcolor{#4E67C8}{H}(x,t) = \textcolor{#F14124}{A}\cos(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega} t).
    $$

    At a fixed position this is the oscillation of section 1.1, and at a fixed time it is the pattern of section
    1.2. The argument $kx-\omega t$ is again the **phase**, now depending on both position and time. A crest is a point of fixed phase, for example
    $kx-\omega t = 0$. Then $x = (\omega/k)\,t$, so the whole pattern moves along the line at the speed

    $$
    v_{\text{phase}} = \frac{\textcolor{#FF8021}{\omega}}{\textcolor{#6B913C}{k}} = \frac{\textcolor{#DF2CB2}{\lambda}}{\textcolor{#DF2CB2}{T}}.
    $$

    The form $\cos(kx-\omega t)$ follows from the motion of a source. Suppose a source at $x=0$ moves as
    $A\cos(\omega t)$ and the disturbance travels outward at the speed $v$. The motion at the distance $r$ is the
    motion of the source a time $r/v$ earlier, which is
    $A\cos\!\big(\omega(t-r/v)\big) = A\cos(\omega t - kr)$ with $k=\omega/v$. Since the cosine is unchanged by a
    change of sign of its argument, this is $A\cos(kr-\omega t)$. **The distance from the source enters only
    through the phase $kr$**, which increases by $2\pi$ for each further wavelength. Here $r$ is the distance from
    the source, which plays the role of the position $x$ along the direction in which the wave travels: for a wave
    moving along a line, $r=x$ and this is $A\cos(kx-\omega t)$. In section 3, $r_1$ and $r_2$ are the distances
    from two sources (two holes) to a detector, and $x$ will be the position of the detector along a line behind
    the wall. This is used in section 3.

    A phase that has a constant added, $A\cos(kx-\omega t+\theta_0)$, is the same wave shifted along the line.

    The speed of the pattern is not yet the speed of a particle. That is the subject of section 6.
    """)
    return


@app.cell
def _(mo):
    wave_k = mo.ui.slider(0.5, 6, step=0.1, value=2.0, label=r"$k$", show_value=True)
    wave_omega = mo.ui.slider(0.5, 6, step=0.1, value=3.0, label=r"$\omega$", show_value=True)
    wave_t = mo.ui.slider(0, 6, step=0.1, value=1.2, label=r"time $t$", show_value=True)
    return wave_k, wave_omega, wave_t


@app.cell
def _(mo, moving_wave_figure, plt, wave_k, wave_omega, wave_t):
    _fig = moving_wave_figure(wave_k.value, wave_omega.value, wave_t.value)
    _out = mo.vstack([mo.hstack([wave_k, wave_omega, wave_t], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, plt, travelling_features_figure):
    _fig = travelling_features_figure()
    _out = mo.vstack([
        mo.md(
            r"""
            **Both variables on one graph.** Left: the height against position at two times. A crest, a point of fixed phase $kx-\omega t=0$, has moved to the right by $(\omega/k)\,t$, so the whole pattern slides along. Right: the height against time at two fixed positions. The crest reaches the farther position later, after the delay $(k/\omega)\,x$. The space graph and the time graph are two slices of the same wave.
            """
        ),
        _fig,
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The whole wave as a surface.** The height $H(x,t)=A\cos(kx-\omega t)$ depends on two variables, so its
    graph is a surface. The green curve is the slice $t=0$, the space graph of section 1.2 (set by $k$), and the orange curve
    is the slice $x=0$, the time graph of section 1.1 (set by $\omega$). The ridges are the crests, and they run diagonally
    because a crest moves along $x$ as $t$ advances, at the speed $\omega/k$. Change $A$, $k$ and $\omega$ and
    watch what each one does: $A$ sets the height of the ridges, $k$ sets how close together they are along $x$,
    and $\omega$ sets how close together they are along $t$.
    """)
    return


@app.cell
def _(mo):
    surf_A = mo.ui.slider(0.5, 2.5, step=0.1, value=1.5, label=r"size $A$", show_value=True)
    surf_k = mo.ui.slider(0.5, 4, step=0.1, value=1.5, label=r"wave number $k$", show_value=True)
    surf_omega = mo.ui.slider(0.5, 4, step=0.1, value=2.0, label=r"angular frequency $\omega$", show_value=True)
    return surf_A, surf_k, surf_omega


@app.cell
def _(mo, plt, surf_A, surf_k, surf_omega, wave_surface_figure):
    _fig = wave_surface_figure(surf_A.value, surf_k.value, surf_omega.value)
    _out = mo.vstack([mo.hstack([surf_A, surf_k, surf_omega], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 The same waves with complex numbers

    For classical waves such as water, writing the wave with complex numbers is a neat mathematical trick. For quantum
    amplitudes it becomes necessary (section 4). This section shows that the two descriptions are the same wave.

    ### 2.1 An arrow in the plane

    A complex number $z = a + ib$ can be drawn as an arrow in a plane: $a$ is the horizontal component (the
    *real part*) and $b$ is the vertical component (the *imaginary part*). The arrow has a length $|z|$ and
    points at some angle from the horizontal axis.

    The arrow of length 1 pointing at the angle $\theta$ (in radians) is written

    $$
    e^{i\textcolor{#7030A0}{\theta}} = \cos\textcolor{#7030A0}{\theta} + i\sin\textcolor{#7030A0}{\theta} ,
    $$

    with horizontal component $\cos\theta$ and vertical component $\sin\theta$. An arrow of length $A$ at the
    same angle is $A$ times this,

    $$
    \textcolor{#5ECCF3}{\psi}(\textcolor{#7030A0}{\theta}) = \textcolor{#F14124}{A}\,e^{i\textcolor{#7030A0}{\theta}} = \textcolor{#F14124}{A}\cos\textcolor{#7030A0}{\theta} + i\textcolor{#F14124}{A}\sin\textcolor{#7030A0}{\theta} .
    $$

    This is the arrow in the right panel of the figure in section 1, and the angle $\theta$ is the **phase** of
    section 1. Its horizontal component is the cosine of section 1,

    $$
    \textcolor{#4E67C8}{H}(\textcolor{#7030A0}{\theta}) = \textcolor{#F14124}{A}\cos\textcolor{#7030A0}{\theta} = \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}(\textcolor{#7030A0}{\theta}),
    $$

    where $\operatorname{Re}$ means the real part. A notation is fixed here and used from now on: the capital
    $H$ is the **real height** of the wave, and $\psi$ is the **complex wave**, the arrow, whose real
    part is $H$. The later notebooks use the same letter $\psi$ for the wave function.

    **The length of an arrow, and why $|\psi|^2$ is not $\psi^2$.** The **absolute value** $|\psi|$ of a complex
    number is the length of its arrow, the distance from the origin to its tip. The arrow $\psi = a + ib$ has
    horizontal component $a$ and vertical component $b$, where $b$ is the real number multiplying $i$, and
    these are the two short sides of a right triangle whose long side is the arrow. By Pythagoras

    $$
    |\textcolor{#5ECCF3}{\psi}| = \sqrt{a^2 + b^2}, \qquad |\textcolor{#5ECCF3}{\psi}|^2 = a^2 + b^2 .
    $$

    For example $3+4i$ has length $\sqrt{9+16}=5$. The squared length $|\psi|^2$ is the quantity called the
    intensity above. For $\psi = A\,e^{i\theta}$ it is $A^2\cos^2\theta + A^2\sin^2\theta = A^2$, so the
    length is $A$, as it should be.

    Simply squaring $\psi$ does not give this. Since $i^2=-1$,

    $$
    \textcolor{#5ECCF3}{\psi}^2 = (a+ib)^2 = a^2 - b^2 + 2iab ,
    $$

    which is a complex number in general, can be negative, and is not a length. For the arrow pointing straight
    up, $\psi = i$, it gives $\psi^2=-1$, although the arrow has length 1. The squared length is instead
    $\psi$ times its **complex conjugate** $\psi^{*}$, which is $\psi$ with the sign of $i$ reversed (the arrow
    reflected in the horizontal axis):

    $$
    \textcolor{#5ECCF3}{\psi}^{*} = a - ib, \qquad
    \textcolor{#5ECCF3}{\psi}^{*}\textcolor{#5ECCF3}{\psi} = (a-ib)(a+ib) = a^2 + iab - iab - i^2b^2 = a^2 + b^2 = |\textcolor{#5ECCF3}{\psi}|^2 .
    $$

    The two terms with $i$ cancel and $-i^2=+1$, so the result is a real, non-negative number, as a length
    squared and a probability must be. For $\psi = i$ it gives $(-i)(i) = 1$.
    """)
    return


@app.cell
def _(mo):
    theta_slider = mo.ui.slider(0, 2 * 3.14159265, step=0.05, value=1.0, label=r"phase $\theta$", show_value=True)
    return (theta_slider,)


@app.cell
def _(arrow_figure, mo, plt, theta_slider):
    _fig = arrow_figure(theta_slider.value)
    _out = mo.vstack([theta_slider, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 The real wave is the horizontal component of a turning arrow

    Now replace the phase $\theta$ by the phase of section 1.3, $\theta = kx-\omega t$. The **complex wave** is

    $$
    \textcolor{#5ECCF3}{\psi}(x,t) = \textcolor{#F14124}{A}\,e^{\,i(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega} t)},
    $$

    and its real part is the wave of section 1.3,

    $$
    \textcolor{#4E67C8}{H}(x,t) = \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}(x,t) = \textcolor{#F14124}{A}\cos(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega} t).
    $$

    These are equal for every $x$ and $t$: $H$ is the real wave, and $\psi$ is the arrow of length $A$ at the angle
    $kx-\omega t$ of which $H$ is the horizontal component. The vertical component is
    $A\sin(kx-\omega t)$, which is the same wave a quarter of a period later. A wave with a starting phase,
    $A\cos(kx-\omega t+\theta_0)$, is the real part of $\psi = a\,e^{i(kx-\omega t)}$ with the **complex
    amplitude** $a = A\,e^{i\theta_0}$, so that a single complex number holds both the size and the starting
    phase.

    The quantities of section 1 now have a geometric meaning.

    **$\omega$ turns the arrow in time.** At a fixed position the arrow is $\psi(t) = \big(A\,e^{ikx}\big)\,e^{-i\omega t}$.
    The bracket does not change with time, and the phase $-\omega t$ of the second factor grows in proportion to
    the time, so the arrow turns at a steady rate, and **$\omega$ is the number of
    radians through which it turns per unit of time**.[^timesign] One complete turn takes the period
    $T=2\pi/\omega$ of section 1.1. The curve in the right panel of the figure, the horizontal component of the
    arrow, is the oscillation $\cos\omega t$ of section 1.1, and the other curve is its vertical component.

    [^timesign]: The minus sign means that the arrow turns clockwise. Some texts write $e^{+i\omega t}$ in
        place of $e^{-i\omega t}$, and then the arrow turns the other way. That reverses the sign of every
        phase and changes no intensity.
    """)
    return


@app.cell
def _(mo):
    arrow_omega = mo.ui.slider(0.5, 6, step=0.1, value=2.0, label=r"rotation rate $\omega$", show_value=True)
    arrow_t = mo.ui.slider(0, 10, step=0.1, value=1.3, label=r"time $t$", show_value=True)
    return arrow_omega, arrow_t


@app.cell
def _(arrow_omega, arrow_t, mo, plt, time_rotation_figure):
    _fig = time_rotation_figure(arrow_omega.value, arrow_t.value)
    _out = mo.vstack([mo.hstack([arrow_omega, arrow_t], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **$k$ twists the arrow in space.** At a fixed time the arrow at the position $x$ is $\psi(x) = \big(A\,e^{-i\omega t}\big)\,e^{ikx}$. The
    bracket is fixed, and the phase $kx$ of the second factor grows in proportion to the distance, so neighbouring arrows are turned relative to each other, and
    **$k$ is the number of radians through which the arrow turns per unit of length**. One complete turn takes
    the wavelength $\lambda=2\pi/k$ of section 1.2. The upper strip in the figure shows the arrow at evenly
    spaced positions, and the lower plot shows its horizontal component, which is the pattern $\cos kx$ of
    section 1.2, and its vertical component.

    Putting the two together, the arrow $\psi(x,t)$ at each position turns in time at the rate
    $\omega$ while neighbouring positions are turned relative to each other at the rate $k$, and the crests of
    its horizontal component move at the speed $\omega/k$ found in section 1.3. **Nothing has been added to the
    description: the real wave and the turning arrow are the same wave written in two ways.**
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(0.5, 6, step=0.1, value=2.0, label=r"twist rate $k$", show_value=True)
    return (k_slider,)


@app.cell
def _(k_slider, mo, plt, space_twist_figure):
    _fig = space_twist_figure(k_slider.value)
    _out = mo.vstack([k_slider, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 Adding waves: interference

    Waves add when the same medium is disturbed by two sources. Consider water waves in a shallow trough that
    pass through two small holes in a wall. Each hole acts as a source of circular waves of wavelength
    $\lambda = 2\pi/k$ and angular frequency $\omega$. A detector on the far side measures the intensity of the
    wave motion, which is proportional to the average of the square of the height of the water. The detector is
    at the distance $r_1$ from hole 1 and $r_2$ from hole 2. (These are distances from each hole to the detector, as
    in section 1.3, not the position $x$ of the detector along the line behind the wall, which appears in the
    figure.)
    """)
    return


@app.cell
def _(mo):
    sch_x = mo.ui.slider(-6, 6, step=0.1, value=2.0, label="detector position $x$", show_value=True)
    sch_wavelength = mo.ui.slider(0.6, 1.6, step=0.05, value=1.0, label=r"wavelength $\lambda$", show_value=True)
    sch_d = mo.ui.slider(2, 6, step=0.25, value=4.0, label="hole separation $d$", show_value=True)
    return sch_d, sch_wavelength, sch_x


@app.cell
def _(interference_schematic, mo, plt, sch_d, sch_wavelength, sch_x):
    _fig = interference_schematic(sch_x.value, sch_wavelength.value, sch_d.value)
    _out = mo.vstack([mo.hstack([sch_x, sch_wavelength, sch_d], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The calculation goes in steps, from the complex waves to the real height.

    **Step 1: one complex wave for each hole.** By section 1.3 the real height at the detector from hole 1 is
    $H_1(t) = A_1\cos(kr_1-\omega t)$, where $A_1$ is the size of the wave at the detector, treated as constant
    over the range of positions considered. Its complex wave, in the notation of section 2.2, is

    $$
    \textcolor{#5ECCF3}{\psi}_1(t) = \textcolor{#F14124}{A}_1\,e^{\,i(\textcolor{#6B913C}{k}\textcolor{#FF84FA}{r}_1-\textcolor{#FF8021}{\omega} t)},\qquad \textcolor{#4E67C8}{H}_1(t) = \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}_1(t).
    $$

    The direction of this arrow is the phase $kr_1-\omega t$. The part $kr_1$ is set by the distance from the hole
    (one more wavelength turns it by $2\pi$), and the part $-\omega t$ turns the arrow in time. Hole 2 gives
    $\psi_2(t) = A_2\,e^{\,i(kr_2-\omega t)}$ and $H_2(t) = \operatorname{Re}\,\psi_2(t)$.

    **Step 2: add the complex waves.** The sum of the two arrows is

    $$
    \textcolor{#5ECCF3}{\psi}_{12}(t) = \textcolor{#5ECCF3}{\psi}_1(t) + \textcolor{#5ECCF3}{\psi}_2(t).
    $$

    **Step 3: the real part is the real height.** The real part of a sum is the sum of the real parts, so

    $$
    \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}_{12}(t) = \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}_1(t) + \operatorname{Re}\,\textcolor{#5ECCF3}{\psi}_2(t) = \textcolor{#4E67C8}{H}_1(t)+\textcolor{#4E67C8}{H}_2(t) = \textcolor{#4E67C8}{H}_{12}(t).
    $$

    This is the height of the water with both holes open, since the heights add. Adding the arrows and then taking the real
    part gives the same wave as adding the real waves.

    **Step 4: separate the part that turns.** Both arrows contain the factor $e^{-i\omega t}$, so

    $$
    \textcolor{#5ECCF3}{\psi}_{12}(t) = \big(\textcolor{#F14124}{A}_1\,e^{\,i\textcolor{#6B913C}{k}\textcolor{#FF84FA}{r}_1}+\textcolor{#F14124}{A}_2\,e^{\,i\textcolor{#6B913C}{k}\textcolor{#FF84FA}{r}_2}\big)\,e^{-i\textcolor{#FF8021}{\omega} t}.
    $$

    The bracket is the sum of the two arrows at $t=0$ (the top left of the figure below), placed tip to tail. The
    factor $e^{-i\omega t}$ turns the whole sum, and the two arrows keep the same angle $\delta$ between them,
    where $\delta = k(r_2-r_1)$, so the *length* of the sum does not change in time.

    **Step 5: the intensity.** What a detector measures is the energy of the wave motion that arrives, and the
    energy of a wave is proportional to the square of its height (section 3.1 derives this from the energy of a
    harmonic oscillator). So the intensity is
    proportional to the **average of $H^2$ over a period**: the *real* height, squared. The complex wave gives this
    average with no extra work. Write the sum of Step 4 as $\psi_{12}(t) = c\,e^{-i\omega t}$, where
    $c = A_1e^{ikr_1}+A_2e^{ikr_2}$ is the bracket, and write $c=|c|e^{i\alpha}$. Then

    $$
    \textcolor{#4E67C8}{H}_{12}(t) = \operatorname{Re}\big[\,|c|\,e^{i(\alpha-\textcolor{#FF8021}{\omega} t)}\big] = |c|\cos(\textcolor{#FF8021}{\omega} t-\alpha),
    \qquad
    \langle \textcolor{#4E67C8}{H}_{12}^2\rangle = |c|^2\,\langle\cos^2(\textcolor{#FF8021}{\omega} t-\alpha)\rangle = \tfrac12|c|^2 ,
    $$

    because the average of $\cos^2$ over a period is $\tfrac12$ whatever the starting phase $\alpha$. Since
    $|e^{-i\omega t}|=1$, the length of the arrow is $|\psi_{12}(t)|=|c|$ at every time, so

    $$
    \langle \textcolor{#4E67C8}{H}_{12}^2\rangle = \tfrac12\,|\textcolor{#5ECCF3}{\psi}_{12}|^2 .
    $$

    The real height $H$ is the horizontal part of the arrow and swings between $\pm|\psi|$, while the length $|\psi|$
    of the arrow does not change. The factor $\tfrac12$ is the same for every wave, and only ratios of intensities
    matter, so it is dropped and the intensity is written $I = |\psi|^2$. The intensity
    of one hole is
    $I_1 = |\psi_1|^2 = A_1^2$, and with both holes open

    $$
    I_{12} = |\textcolor{#5ECCF3}{\psi}_{12}|^2 = \big|\textcolor{#F14124}{A}_1\,e^{\,i\textcolor{#6B913C}{k}\textcolor{#FF84FA}{r}_1}+\textcolor{#F14124}{A}_2\,e^{\,i\textcolor{#6B913C}{k}\textcolor{#FF84FA}{r}_2}\big|^2 = \textcolor{#F14124}{A}_1^2 + \textcolor{#F14124}{A}_2^2 + 2\textcolor{#F14124}{A}_1\textcolor{#F14124}{A}_2\cos\textcolor{#7030A0}{\delta} .
    $$

    The last step uses $|u+v|^2 = |u|^2+|v|^2+2\operatorname{Re}(u^*v)$ with $u^*v = A_1A_2\,e^{i\delta}$. In
    terms of the single-hole intensities,

    $$
    I_{12} = I_1 + I_2 + 2\sqrt{I_1I_2}\,\cos\textcolor{#7030A0}{\delta} .
    $$

    The last term is the interference term, and $\delta$ is the angle between the two arrows. When the arrows
    are parallel their lengths add, and when they are opposite their lengths subtract. For two waves of equal
    size, $I_{12}$ is largest, four times the intensity of one wave, where $\delta$ is a multiple of $2\pi$ (the
    distances differ by whole wavelengths), and zero where $\delta$ is an odd multiple of $\pi$ (the distances
    differ by an odd number of half-wavelengths). The result is not $I_1+I_2$: the waves **interfere**. For
    waves of unequal size, or for any number of waves, the arrows are simply placed tip to tail, with no extra
    work.[^trigexample]

    [^trigexample]: The same result in trigonometric form, for two waves of equal size $A$. The heights are
        $H_1=A\cos(kr_1-\omega t)$ and $H_2=A\cos(kr_2-\omega t)$, and the identity
        $\cos a+\cos b = 2\cos\tfrac{a-b}{2}\cos\tfrac{a+b}{2}$ gives
        $H_{12}(t) = 2A\cos\tfrac{\delta}{2}\,\cos\!\big(\tfrac{k(r_1+r_2)}{2}-\omega t\big)$.
        This is a wave of the same frequency whose size is $2A|\cos(\delta/2)|$, so
        $I_{12}=4A^2\cos^2(\delta/2)$. The arrows give the same: with $A_1=A_2=A$,
        $I_{12}=2A^2+2A^2\cos\delta=4A^2\cos^2(\delta/2)$. For unequal sizes, or for three or more waves, the
        identity no longer applies directly and the algebra grows quickly, whereas arrows keep adding.

    The figure shows the arrows $\psi_1$, $\psi_2$ and their sum at $t=0$ (top left), the real heights at the detector
    over two periods (top right), and the intensity along the line of detector positions (bottom). The holes are
    a distance $d$ apart, and the line of detector positions is $20$ length units behind the wall, in units where
    the wavelength is of order one. The dashed line is the sum of the intensities of the two waves taken
    separately.
    """)
    return


@app.cell
def _(mo):
    slit_x = mo.ui.slider(-15, 15, step=0.1, value=0.0, label="detector position $x$", show_value=True)
    slit_wavelength = mo.ui.slider(0.6, 1.6, step=0.05, value=1.0, label=r"wavelength $\lambda$", show_value=True)
    slit_d = mo.ui.slider(2, 6, step=0.25, value=4.0, label="hole separation $d$", show_value=True)
    return slit_d, slit_wavelength, slit_x


@app.cell
def _(interference_figure, mo, plt, slit_d, slit_wavelength, slit_x):
    _fig = interference_figure(slit_x.value, slit_wavelength.value, slit_d.value)
    _out = mo.vstack([mo.hstack([slit_x, slit_wavelength, slit_d], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(hole_amplitude, intensity, mo, np, slit_d, slit_wavelength, slit_x, two_hole_distances):
    _k = 2 * np.pi / slit_wavelength.value
    _r1, _r2 = two_hole_distances(slit_x.value, slit_d.value, 20.0)
    _h1, _h2 = hole_amplitude(_r1, _k), hole_amplitude(_r2, _k)
    _delta = _k * (_r2 - _r1)
    mo.md(
        rf"""
        At $x = {slit_x.value:g}$: $r_2 - r_1 = {_r2 - _r1:.2f}$, which is {(_r2 - _r1) / slit_wavelength.value:.2f}
        wavelengths, so $\delta = {_delta / (2 * np.pi):.2f}\times 2\pi$.
        $I_1 + I_2 = {intensity(_h1) + intensity(_h2):.2f}$ and $I_{{12}} = |\psi_1+\psi_2|^2 = {intensity(_h1 + _h2):.2f}$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.1 Why the energy goes as the height squared

    The intensity of Step 5 rests on one fact: the energy of a wave is proportional to the square of its height. It
    follows from the energy of a harmonic oscillator. A small piece of water at the detector, of mass $m$, moves up and
    down under a restoring force $-\kappa H$ from the water around it. Its height obeys $m{\ddot H} = -\kappa H$, with the
    solution

    $$
    {\textcolor{#4E67C8}{H}}(t) = {\textcolor{#F14124}{A}}\cos({\textcolor{#FF8021}{\omega}} t), \qquad {\textcolor{#FF8021}{\omega}}^2 = \kappa/m .
    $$

    Its potential and kinetic energies are

    $$
    E_{\text{pot}} = \tfrac12\kappa {\textcolor{#4E67C8}{H}}^2 = \tfrac12 m{\textcolor{#FF8021}{\omega}}^2{\textcolor{#F14124}{A}}^2\cos^2({\textcolor{#FF8021}{\omega}} t),
    \qquad
    E_{\text{kin}} = \tfrac12 m{\textcolor{#4E67C8}{\dot H}}^2 = \tfrac12 m{\textcolor{#FF8021}{\omega}}^2{\textcolor{#F14124}{A}}^2\sin^2({\textcolor{#FF8021}{\omega}} t),
    $$

    and their sum is

    $$
    \textcolor{#FF8021}{E} = E_{\text{pot}} + E_{\text{kin}} = \tfrac12 m{\textcolor{#FF8021}{\omega}}^2{\textcolor{#F14124}{A}}^2 .
    $$

    The total does not depend on time: the energy passes back and forth between the two forms while the sum stays
    fixed (figure). Two things follow.

    - For a given $m$ and $\omega$, the energy is proportional to ${A}^2$: twice the height, four times the energy.
    - The average of ${H}^2$ over a period is $\tfrac12{A}^2$, so the energy can be written $E = m{\omega}^2\,\langle {H}^2\rangle$.
      The energy is proportional to the **average of the real height squared**.

    A wave is a set of such oscillators, so the energy it delivers to the detector in each second, its intensity, is
    proportional to $\langle {H}^2\rangle$ for waves of the same frequency in the same medium. This is the relation used
    in Step 5. For the two holes the water at the detector oscillates with size $|c|$, so its energy is proportional
    to $|c|^2=|{\psi}_{12}|^2$.
    """)
    return


@app.cell
def _(mo):
    osc_A = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label=r"size $A$", show_value=True)
    osc_omega = mo.ui.slider(1, 3, step=0.1, value=2.0, label=r"angular frequency $\omega$", show_value=True)
    return osc_A, osc_omega


@app.cell
def _(mo, osc_A, osc_omega, oscillator_energy_figure, plt):
    _fig = oscillator_energy_figure(osc_A.value, osc_omega.value)
    _out = mo.vstack([mo.hstack([osc_A, osc_omega], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Why the waves must be complex for quantum amplitudes

    For classical waves such as water, the height is a real quantity and the complex number is only a neat way
    to add waves: a real cosine would do, with more algebra. For a particle the situation is different, and the
    complex form is necessary. The amplitude $e^{i(kx-\omega t)}$ has length 1 at every position and
    time, so the probability of finding the particle is the same everywhere and does not change. This is the
    correct description of a state of definite energy and definite momentum: the momentum is known exactly, so
    by the uncertainty principle the position is completely undetermined.

    A real wave such as $\cos(kx-\omega t)$ would not do. Its square has nodes, positions where the
    probability is zero, and it would oscillate in time. The complex form avoids both, because its length does
    not change as the arrow turns.
    """)
    return


@app.cell
def _(mo, plt, probability_figure, wave_k, wave_omega, wave_t):
    _fig = probability_figure(wave_k.value, wave_omega.value, wave_t.value)
    _out = mo.vstack([mo.md("The same $k$, $\\omega$ and $t$ as in section 1.3:"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The rule of section 3 holds for these amplitudes as well. If $a_1$ and $a_2$ are the amplitudes for a
    particle to arrive at a detector through hole 1 and through hole 2, the probability with both holes open is
    $|a_1+a_2|^2$ and not $|a_1|^2+|a_2|^2$. Here the complex number is the amplitude itself, and its
    real part alone would not give the right interference.

    **Why the squared length?** For water, the squared length $|\psi|^2$ was justified by the energy of the wave,
    which grows as the square of the real height (Step 5 of section 3). For a particle there is no height to
    square. The rule that the probability is the squared length of the amplitude, $P=|a|^2$, is a postulate of
    quantum mechanics (the Born rule) that experiment confirms. The arithmetic is the same as for water, but the
    squared length now has a different physical meaning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Energy and momentum are these rates

    The physical content is in two relations that connect the rotation rates to the quantities a physicist
    measures. Here $\hbar$ is Planck's constant divided by $2\pi$.

    $$
    \textcolor{#FF8021}{E} = \hbar\,\textcolor{#FF8021}{\omega}, \qquad\qquad p = \hbar\,\textcolor{#6B913C}{k} .
    $$

    - **Energy is the rotation rate in time.** A state of definite energy $E$ has an arrow that turns at the
      rate $\omega = E/\hbar$. A high-energy state turns quickly. Adding the same constant to every energy
      multiplies all amplitudes by the same turning arrow, which changes no probability, so only differences
      of energy are observable.
    - **Momentum is the rotation rate in space.** A state of definite momentum $p$ has an arrow that twists at
      the rate $k = p/\hbar$ along the line, so its wavelength is
      $$
      \textcolor{#DF2CB2}{\lambda} = \frac{2\pi}{\textcolor{#6B913C}{k}} = \frac{\textcolor{#5ECCF3}{\psi}}{p}, \qquad \textcolor{#5ECCF3}{\psi} = 2\pi\hbar .
      $$
      The larger the momentum, the shorter the wavelength.

    **Where the two relations come from.** Both have the same origin in symmetry, which is also the origin of
    the conservation of angular momentum.[^symmetry] Suppose the physical situation is unchanged by a shift in
    position, as it is in empty space. A plane wave $e^{ikx}$ is then a state that a shift only multiplies by a
    phase, proportional to the size of the shift, whose rate is $k$. Such a phase factor cannot change as
    time passes, so the rate $k$ is conserved. The same holds for a shift in time, which multiplies a state of
    definite energy by a phase whose rate is $\omega$. These conserved rates are the quantum forms of the
    conservation of momentum and of energy. The factor $\hbar$ converts them to the momentum and energy of
    classical mechanics, to which they must agree for large systems, and its value is found by experiment.

    **Why the same constant appears in both.** Relativity requires it. Energy and momentum are the time part
    and the space part of a single quantity, and so are frequency and wave number, so one constant $\hbar$
    converts both. A particle at rest has a pure rotation in time, $e^{-iE_0t/\hbar}$. An observer who sees the
    particle moving sees the same rotation re-expressed in terms of the new time and position, and part of it
    becomes a twist in space whose rate is the particle's momentum divided by $\hbar$.[^lorentz]

    For a slowly moving particle of mass $m$ ($v \ll c$) the energy follows from the relativistic expression
    in the footnote, $E = E_0/\sqrt{1-v^2/c^2}$ with rest energy $E_0 = mc^2$. For small $u = v^2/c^2$ the
    square root factor is $(1-u)^{-1/2} \approx 1 + u/2$, and $p \approx mv$, so

    $$
    E \approx E_0\left(1 + \frac{v^2}{2c^2}\right) = E_0 + \tfrac12 mv^2 = E_0 + \frac{p^2}{2m} .
    $$

    The energy is the rest energy $E_0$ plus the kinetic energy $p^2/2m$. Dividing by $\hbar$ gives the rotation
    rate of the amplitude in time, which is the rest-energy rate plus a rate that depends on $k$:

    $$
    \textcolor{#FF8021}{\omega}(\textcolor{#6B913C}{k}) = \textcolor{#FF8021}{\omega}_{\mathrm{rest}} + \frac{\hbar \textcolor{#6B913C}{k}^2}{2m},
    \qquad \textcolor{#FF8021}{\omega}_{\mathrm{rest}} = \frac{E_0}{\hbar} = \frac{mc^2}{\hbar} .
    $$

    **Why the rest-energy term is left out.** The rate $\omega_{\mathrm{rest}}$ is the same for every $k$. A wave
    function built from plane waves with amplitudes $a(k)$ (the sum of section 6) is, at time $t$,

    $$
    \textcolor{#5ECCF3}{\psi}(x,t) = \sum_{\textcolor{#6B913C}{k}} a(\textcolor{#6B913C}{k})\,e^{\,i\left(\textcolor{#6B913C}{k}x-\left(\textcolor{#FF8021}{\omega}_{\mathrm{rest}}+\hbar \textcolor{#6B913C}{k}^2/2m\right)t\right)}
    = e^{-i\textcolor{#FF8021}{\omega}_{\mathrm{rest}}t}\;\sum_{\textcolor{#6B913C}{k}} a(\textcolor{#6B913C}{k})\,e^{\,i\left(\textcolor{#6B913C}{k}x-\hbar \textcolor{#6B913C}{k}^2t/2m\right)} .
    $$

    The factor $e^{-i\omega_{\mathrm{rest}}t}$ does not depend on $k$, so it comes outside the sum and multiplies
    the whole wave function. It is an arrow of length 1, so it turns $\psi$ without changing its length, and
    the probability density is the same with or without it:

    $$
    |\textcolor{#5ECCF3}{\psi}|^2 = \big|e^{-i\textcolor{#FF8021}{\omega}_{\mathrm{rest}}t}\big|^2\,\Big|\sum_{\textcolor{#6B913C}{k}} a(\textcolor{#6B913C}{k})\,e^{\,i\left(\textcolor{#6B913C}{k}x-\hbar \textcolor{#6B913C}{k}^2t/2m\right)}\Big|^2
    = \Big|\sum_{\textcolor{#6B913C}{k}} a(\textcolor{#6B913C}{k})\,e^{\,i\left(\textcolor{#6B913C}{k}x-\hbar \textcolor{#6B913C}{k}^2t/2m\right)}\Big|^2 .
    $$

    Only the differences between the rotation rates of different components change how they add. Two
    components $k_1$ and $k_2$ turn relative to each other at the rate
    $\omega(k_1)-\omega(k_2) = \hbar(k_1^2-k_2^2)/2m$, in which $\omega_{\mathrm{rest}}$ cancels. The speed of
    the lump, $d\omega/dk = \hbar k/m$, also has no $\omega_{\mathrm{rest}}$, because the slope of a constant is
    zero. So adding $E_0$ to the energy is a choice of where to start measuring energy, as with the height of
    the floor in a gravity problem, and it changes no probability. It is dropped, and from here on

    $$
    \textcolor{#FF8021}{\omega}(\textcolor{#6B913C}{k}) = \frac{\hbar \textcolor{#6B913C}{k}^2}{2m}.
    $$

    The figure shows this for the packet of section 6. Slide the phase $\omega_{\mathrm{rest}}t$ that the rest
    energy has turned the wave function through. The real part of $\psi$ changes, because the whole arrow is
    turned, while the probability density does not move. (The term could not be dropped if the rest energy
    itself changed, as in a decay, because then the difference of $E_0$ before and after is a physical energy.)

    [^symmetry]: The companion notebook `../spinors/spinor_rotation.py` shows that a state which a rotation
        only multiplies by a phase $e^{-im\theta}$ keeps its value of $m$, the component of angular momentum,
        for all time. Shifts of position and of time give momentum and energy in the same way. Conservation of
        momentum corresponds to the sameness of space from place to place, conservation of energy to the
        sameness of time from moment to moment, and conservation of angular momentum to the sameness of space
        in all directions.

    [^lorentz]: Let $E_0$ be the energy of the particle at rest, so its amplitude is $e^{-iE_0 t/\hbar}$ in
        its own rest frame, with $t$ the time there. An observer who sees the particle move at speed $v$
        along $+x'$ uses coordinates $(x', t')$ related to the rest-frame time by the Lorentz transformation
        $t = \dfrac{t' - x'v/c^2}{\sqrt{1 - v^2/c^2}}$, where $c$ is the speed of light. Substituting,
        $E_0 t = E' t' - p' x'$ with $E' = \dfrac{E_0}{\sqrt{1 - v^2/c^2}}$ and $p' = \dfrac{E' v}{c^2}$.
        These are the relativistic energy and momentum of a particle of rest energy $E_0$ moving at speed
        $v$. The amplitude the moving observer finds is therefore $e^{-i(E't' - p'x')/\hbar}$: it turns in
        time at the rate $E'/\hbar$ and twists in space at the rate $p'/\hbar$, which are $\omega = E'/\hbar$
        and $k = p'/\hbar$.
    """)
    return


@app.cell
def _(mo):
    rest_phase = mo.ui.slider(0.0, 6.28, step=0.1, value=1.5, label=r"phase $\omega_{\mathrm{rest}}\,t$ (radians)", show_value=True)
    return (rest_phase,)


@app.cell
def _(mo, plt, rest_energy_phase_figure, rest_phase):
    _fig = rest_energy_phase_figure(rest_phase.value)
    _out = mo.vstack([rest_phase, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Wave packets: when a lump moves like a particle

    **From intensity to probability density.** In section 3 the real height $H$ of the water gave an
    intensity, the energy arriving, proportional to the average of $H^2$, which the complex
    wave writes as $I = |\psi|^2$. The same quantity now gives the probability of finding a particle.
    The squared length $|\psi(x,t)|^2$ is the **probability density** $\rho(x,t)$, the probability per unit
    length, and the probability of finding the particle between $a$ and $b$ is its integral over that interval,
    just as the energy of the water between $a$ and $b$ is the integral of its energy per unit length:

    $$
    P(a<x<b,\,t) = \int_a^b \rho(x,t)\,dx = \int_a^b |\textcolor{#5ECCF3}{\psi}(x,t)|^2\,dx .
    $$

    The particle is somewhere, so the probability of finding it somewhere on the whole line must be 1:

    $$
    \int_{-\infty}^{\infty} |\textcolor{#5ECCF3}{\psi}(x,t)|^2\,dx = 1 .
    $$

    This **normalization** fixes the overall size of $\psi$, which for water could be anything. It is a real
    requirement on a wave function: the integral must be finite, and then $\psi$ is divided by the square root
    of its value so that the integral is 1.

    **A single plane wave fails the requirement.** A plane wave $\psi = A\,e^{i(kx-\omega t)}$ has the same
    squared length at every position and time, as found in section 2.1: $|\psi|^2 = A^2$, a constant. The
    probability density is the same everywhere, so the particle is equally likely to be found anywhere. The
    probability of an interval is the area under this flat line, a rectangle of height $A^2$, so

    $$
    P(a<x<b) = \int_a^b A^2\,dx = A^2\,(b-a) ,
    \qquad
    \int_{-X}^{X} |\textcolor{#5ECCF3}{\psi}|^2\,dx = \int_{-X}^{X} A^2\,dx = 2X\,A^2 .
    $$

    The width $2X$ grows without limit as the region is widened, so over the whole line the area is infinite
    for every $A\neq 0$ (and 0 for $A=0$, which is no wave at all). It can never be made equal to 1, so a
    single plane wave cannot describe a particle that is somewhere. A wave function that can must have a
    density that is not constant: large in one region and small elsewhere, with a finite total. Adding
    plane waves with different $k$ can do this, and the sum is a **wave packet**. The figure compares the two:
    on the left the densities, and on the right the total probability found between $-X$ and $X$ as $X$ grows,
    which levels off at 1 for the packet and keeps growing for the plane wave. (The plane wave is drawn
    with $A=1$ and the packet with total probability 1, so compare the shapes and the trend on the right, not
    the heights on the left.)
    """)
    return


@app.cell
def _(flat_vs_lump_figure, mo, plt):
    _fig = flat_vs_lump_figure(5.0, 1.5)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the size $A$ means for a particle.** For water, the size $A$ was the largest height of the surface.
    A wave function $\psi = A\,e^{i(kx-\omega t)}$ has no height to measure, and $A$ only sets the scale of the
    density, $|\psi|^2 = |A|^2$. Predictions use the density through ratios, such as the probability of one
    interval compared with another, and through the total being 1, so multiplying the whole of $\psi$ by a
    constant changes no prediction once the total is set back to 1. Where $A$ is fixed, normalization fixes it:

    - **A packet** has $A$ built in. The bell curve of section 6 has the prefactor $(2\pi\sigma_x^2)^{-1/4}$,
      chosen so that $\int|\psi|^2dx = 1$.
    - **A plane wave** cannot be normalized, as shown above, so no value of $A$ is the right one, and
      $A=1$ is a convention. Two common choices keep the wave and make its density mean something. Confine
      the particle to a stretch of length $L$ and take $A = 1/\sqrt{L}$, so that
      $\int_{-L/2}^{L/2}|A|^2dx = |A|^2L = 1$, which is the flat density of a particle equally likely to be
      anywhere in the stretch. Or read $|A|^2$ as the number of particles per unit length in a steady beam.

    Sizes do matter when waves are added. In the two-hole interference of section 3, the ratio $A_1/A_2$ sets
    how deep the dark fringes are, through $A_1^2+A_2^2+2A_1A_2\cos\delta$, and both are fixed by the overall
    normalization. A common complex factor $e^{i\alpha}$ in front of $\psi$ changes no density at all, but a phase
    difference between two added waves shifts the fringes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A localized particle is described by adding plane waves whose wave numbers $k$ are close to one
    value $k_0$. Where the waves are in step their amplitudes add, and elsewhere they cancel, which leaves a
    lump of probability. The amplitude is

    $$
    \textcolor{#5ECCF3}{\psi}(x,t) = \sum_{\textcolor{#6B913C}{k}} a(\textcolor{#6B913C}{k})\, e^{\,i(\textcolor{#6B913C}{k}x-\textcolor{#FF8021}{\omega}(\textcolor{#6B913C}{k})\,t)},
    $$

    where $a(k)$ is the amplitude of each wave. The probability of a given $k$ is $|a(k)|^2$.

    **Choosing $a(k)$.** The sum can be made with any choice of $a(k)$, so the choice needs a reason. Two
    things are wanted. The first is a lump with one peak and nothing else, since a stray second bump would be
    another place where the particle might be found. The second is a sum that can be done exactly, so that
    the rest of this section can be worked out in full. The simplest choice is to include every wave in a band
    of $k$ equally and none outside it, a flat top. The figure compares it with a bell curve, the
    **Gaussian** $a(k)\propto e^{-(k-k_0)^2/4\sigma_k^2}$, chosen to have the same root-mean-square width
    $\sigma_k$ (the typical distance from $k_0$). The left panel is the probability of each $k$ and the right
    panel is the probability density of position that the sum gives at $t=0$, drawn on a logarithmic scale
    so that small values show.
    """)
    return


@app.cell
def _(band_shapes_figure, mo, plt):
    _fig = band_shapes_figure()
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both give a main lump of about the same width. The flat band also gives ripples on either side, the
    first about 5% as high as the peak, and they die away slowly (as $1/x^2$). They come from the sharp edges
    of the band: waves that stop abruptly at the edge of the band do not cancel one another completely away
    from the peak. The ripples are so spread out that the root-mean-square width in position of the flat band is
    not even finite, so it has no useful spread to compare with $\sigma_k$. The Gaussian has no edges, and its
    lump has no ripples at all. Three reasons make it the standard choice:

    - **One clean lump.** A smooth bell curve in $k$ gives complete cancellation away from the peak, as the
      figure shows.
    - **The same shape in both variables.** Adding Gaussian-weighted plane waves gives a bell curve in $x$ again.
      One shape then carries through the whole calculation, and the sum can be done exactly, which gives the
      closed-form results for the motion and spreading of the lump in subsections 6.3 and 6.4. For most other
      choices of $a(k)$ there is no such formula.
    - **The tightest possible lump.** Of all shapes, the bell curve makes the product of the two widths,
      $\sigma_x\sigma_k$, as small as it can be, so it is the most localized lump that a given spread of
      wave numbers can make.

    It is a convenient choice and not a law of nature. Other smooth, narrow shapes behave in the same way in
    outline, a lump that moves and spreads, and differ in details such as the ripples. So take $a(k)$ to be
    a bell curve centred on $k_0$ with root-mean-square width $\sigma_{k}$, and take every component to have
    the same phase at $t = 0$, so that they are all in step at
    $x = 0$. Then at $t = 0$ the probability of finding the particle is also a bell curve, centred on $x = 0$
    with root-mean-square width $\sigma_x$, and the two widths are linked by (shown in subsection 6.1)

    $$
    \sigma_x\,\sigma_{\textcolor{#6B913C}{k}} = \tfrac12 , \qquad\text{that is}\qquad \Delta x\,\Delta p = \frac{\hbar}{2},
    $$

    where $\Delta x = \sigma_x$ and $\Delta p = \hbar\sigma_{k}$ are the spreads in position and in momentum.
    This is the uncertainty principle, and $\tfrac12$ is the smallest value the product can take.
    It is a property of the state itself, not an error of measurement. A narrow lump in position
    requires a broad range of $k$, and the reverse.

    In the limit of very closely spaced values of $k$ the sum becomes an integral and can be evaluated
    exactly. The figures below use that exact result.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 From a band of waves to a lump: the uncertainty principle

    The previous pages chose a bell curve for $a(k)$ and stated that the lump in position is then also a bell
    curve, with $\sigma_x\sigma_k = \tfrac12$. This subsection shows how the sum over waves builds the lump
    and where that relation comes from.

    **The lump as a sum of cosines.** Take the band of waves to be centred on $k=0$. (A different centre $k_0$
    only multiplies $\psi$ by $e^{ik_0x}$, which turns the arrow without changing its length, so $|\psi|^2$ is
    the same.) The waves with $k$ and $-k$ have the same weight, and $e^{ikx}+e^{-ikx} = 2\cos kx$, so at
    $t=0$ the lump is a sum of cosines, each with its weight $a(k)$:

    $$
    \textcolor{#5ECCF3}{\psi}(x) = \int_{-\infty}^{\infty} a(\textcolor{#6B913C}{k})\,\cos(\textcolor{#6B913C}{k}x)\,d\textcolor{#6B913C}{k} .
    $$

    Adding up waves of every wave number with weights $a(k)$ is called the **Fourier transform** of $a(k)$. It
    turns a function of $k$ into a function of $x$. The figure builds the lump from a bell curve $a(k)$ with
    $\sigma_k=1$, using waves whose wave numbers are $0.5$ apart. The first slider sets how many are included.
    The left panel shows the weights, and the upper right panel shows each wave (grey) and their sum (blue).
    The second slider sets the time, described below. The lower right panel is the probability density of the
    sum.
    """)
    return


@app.cell
def _(mo):
    lump_waves = mo.ui.slider(0, 12, step=1, value=3, label="waves on each side of $k=0$", show_value=True)
    lump_time = mo.ui.slider(0.0, 4.0, step=0.1, value=0.0, label="time $t$", show_value=True)
    return lump_time, lump_waves


@app.cell
def _(lump_buildup_figure, lump_time, lump_waves, mo, plt):
    _fig = lump_buildup_figure(lump_waves.value, lump_time.value)
    _out = mo.vstack([mo.hstack([lump_waves, lump_time], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    With no waves on each side there is only $k=0$, which is flat and stays flat as time passes, because
    the rest-energy rate of section 5 is left out. With it included the arrow of this wave would turn at the
    rate $mc^2/\hbar$ and its real part would oscillate, while $|\psi|^2$ would still be constant: a wave of
    definite energy changes only by an overall phase. Each wave added makes the sum more
    concentrated. All the waves are in step at $x=0$, where their heights add, and away from $x=0$ they are out
    of step and cancel. Two waves whose wave numbers differ by $\Delta k$ drift out of step after a distance of
    order $1/\Delta k$, so the larger the spread of wave numbers, the sooner the cancelling sets in and the
    narrower the lump. The dashed curve is the limit of very many closely spaced waves (the repeated copies of
    the lump, at $\pm 2\pi/0.5 \approx \pm 12.6$, come from the spacing and move away as the spacing shrinks).

    **Time.** At $t>0$ each wave turns at its own rate, $\omega(k) = \hbar k^2/2m$ from section 5 (with
    $\hbar = m = 1$ here), so the real part of each wave is $\cos(kx)\cos(\omega t)$ and the waves fall out of
    step with one another also at $x=0$. Slide the time forward with all 25 waves included. The peak falls and
    the lump widens, because the waves with larger $k$ turn faster and the cancelling that confined the lump
    is no longer complete. The dashed curve is still the limit of very many closely spaced waves, and the
    sum follows it. This is the spreading of subsection 6.4. With few waves included the sum no longer
    follows it for long, because the small number of $k$ values makes the pattern repeat.

    **The bell curve keeps its shape.** The dashed curve is again a bell curve. This is a well-known property of
    the Fourier transform: the transform of a bell curve is a bell curve, with the width inverted. If $a(k)$ is
    a bell curve with root-mean-square width $\sigma_k$, then

    $$
    a(k) = e^{-k^2/4\sigma_k^2} \quad\longrightarrow\quad \textcolor{#5ECCF3}{\psi}(x) \propto e^{-\sigma_k^2x^2} .
    $$

    We use this result without proving it. It is a special property. Most shapes change character when
    transformed: the flat band of the earlier figure gave ripples. For the width $\sigma_k = 1/\sqrt{2}$ the
    two curves are exactly the same function (of $k$ and of $x$), so that bell curve is its own transform. For
    other widths the result is a bell curve that is narrower or wider.

    **Reading off the widths.** The probability density is $|\psi|^2 \propto e^{-2\sigma_k^2x^2}$. A bell curve
    of root-mean-square width $\sigma_x$ is $e^{-x^2/2\sigma_x^2}$, so matching the exponents gives
    $1/2\sigma_x^2 = 2\sigma_k^2$, that is

    $$
    \sigma_x = \frac{1}{2\sigma_k}, \qquad\text{so that}\qquad \sigma_x\,\sigma_{\textcolor{#6B913C}{k}} = \frac12 .
    $$

    Since $\Delta x = \sigma_x$ and $\Delta p = \hbar\sigma_k$ (section 5, $p=\hbar k$), this is
    $\Delta x\,\Delta p = \hbar/2$, the uncertainty principle. The width of the lump in position is fixed by the
    width of the band of waves that made it: squeeze one and the other widens. The figure shows both densities
    with their widths, each drawn with total probability 1 on a fixed scale, so the height changes as the width
    does. The right panel is the curve $\sigma_x = 1/2\sigma_k$ with a point for the current slider value.
    """)
    return


@app.cell
def _(mo):
    gauss_sigma_w = mo.ui.slider(0.3, 2.0, step=0.05, value=0.8, label=r"$\sigma_k$", show_value=True)
    return (gauss_sigma_w,)


@app.cell
def _(gauss_sigma_w, mo, plt, width_tradeoff_figure):
    _fig = width_tradeoff_figure(gauss_sigma_w.value)
    _out = mo.vstack([gauss_sigma_w, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why this is the smallest product.** The bell curve is not the only choice, and the same calculation
    (build $\psi$ from $a(k)$, then measure both widths) can be done for other shapes. The figure does it
    numerically for the family $a(k) = e^{-|k|^p/p}$. The exponent $p=2$ is the bell curve, $p=1.5$ has fatter
    tails, and large $p$ is a flat top with soft edges. The product $\sigma_x\sigma_k$ is lowest, exactly
    $\tfrac12$, at $p=2$, and larger for every other shape in the family. This illustrates a general fact: for
    every shape $\sigma_x\sigma_k \geq \tfrac12$, with equality only for the bell curve. (The general proof is
    not given here.) That is why the bell curve was called the tightest possible lump.

    **What it says about particles.** A particle described as a lump of waves cannot have both a sharp position
    and a sharp wave number. The lump is narrow in $x$ only if many wave numbers are added, and the width in
    $k$ is then not small. The relation $\sigma_x\sigma_k = \tfrac12$ is not an extra rule laid on top of the
    wave picture. It follows from building a lump from waves, through the way the Fourier transform trades a
    wide band for a narrow lump.
    """)
    return


@app.cell
def _(min_uncertainty_figure, mo, plt):
    _fig = min_uncertainty_figure()
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 Entropy: another way to see the uncertainty principle

    The root-mean-square width $\sigma$ is one way to say how spread out a distribution is. Another is its
    **entropy**, which measures how much is left unknown about where the particle will be found. For a
    probability density $\rho$ the entropy is

    $$
    h = -\int \rho \,\ln \rho \; dx ,
    $$

    the average value of $-\ln\rho$. This is small where the density is large (an expected outcome) and large
    where the density is small (a surprising outcome), so a density concentrated in one place has a low entropy
    and a spread-out density has a high one. (For a density of a continuous variable the entropy can be
    negative, and only comparisons between values matter.) Write $h_x$ for the entropy of the position density
    $|\psi(x)|^2$ and $h_k$ for that of the density $|a(k)|^2$ of wave number. For a bell curve of width
    $\sigma$ the integral gives $h = \tfrac12\ln(2\pi e\,\sigma^2)$, which grows as the curve widens.

    Two facts about the bell curve connect entropy to the uncertainty principle. Both are stated here
    without proof.

    - **At a given width, the bell curve has the largest entropy.** No other shape of width $\sigma$ has more:
      $h \le \tfrac12\ln(2\pi e\,\sigma^2)$. It is the most spread-out, least committal shape that the width
      allows.
    - **The two entropies cannot both be small.** For every lump, $h_x + h_k \ge \ln(\pi e)$, with equality only
      for a bell curve (in position and in wave number). This is the **entropic uncertainty relation**. Here
      the bell curve has the *smallest* total entropy.

    The direction is therefore different in the two facts. For one variable at a fixed width the bell curve
    maximizes the entropy, and for the pair of variables it minimizes the total. Putting the two together gives
    the uncertainty relation of 6.1 again:

    $$
    \ln(\pi e) \;\le\; h_x + h_k \;\le\; \tfrac12\ln(2\pi e\,\sigma_x^2) + \tfrac12\ln(2\pi e\,\sigma_k^2)
    = \ln(2\pi e\,\sigma_x\sigma_k) \quad\Longrightarrow\quad \sigma_x\sigma_k \ge \tfrac12 .
    $$

    The first inequality is the second fact, and the second inequality is the first fact applied to $x$ and to
    $k$. For a bell curve both are equalities, which is why its product is exactly $\tfrac12$. The entropy
    version is also the stronger statement. The flat band of the earlier figure has ripples that die away so
    slowly that its width $\sigma_x$ is infinite, so the width relation says nothing about it, but its entropy
    is finite and the entropic relation still applies.

    The figure uses the family $a(k) = e^{-|k|^p/p}$ of the previous page, where $p=2$ is the bell curve. The
    left and middle panels show $|a|^2$ and $|\psi|^2$ for the chosen $p$ with their entropies. The right panel
    shows the total $h_x+h_k$ against $p$ (dark), the floor $\ln(\pi e)$ (dashed) and the upper bound
    $\ln(2\pi e\,\sigma_x\sigma_k)$ (light). The total touches the floor only at $p=2$ and lies above it for
    every other shape. At $p=2$ the two densities are even the same function, because the bell curve with
    $\sigma_k=1/\sqrt{2}$ is its own transform (6.1).
    """)
    return


@app.cell
def _(mo):
    entropy_p = mo.ui.slider(1.5, 6.0, step=0.25, value=3.0, label=r"shape exponent $p$", show_value=True)
    return (entropy_p,)


@app.cell
def _(entropy_family_figure, entropy_p, mo, plt):
    _fig = entropy_family_figure(entropy_p.value)
    _out = mo.vstack([entropy_p, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.3 The lump moves at the classical velocity

    Each component has its own rotation rate $\omega(k) = \hbar k^2/2m$. Write this exactly around the central
    wave number $k_0$:

    $$
    \textcolor{#FF8021}{\omega}(\textcolor{#6B913C}{k}) = \textcolor{#FF8021}{\omega}_0 + v\,(\textcolor{#6B913C}{k}-\textcolor{#6B913C}{k}_0) + \frac{\hbar}{2m}(\textcolor{#6B913C}{k}-\textcolor{#6B913C}{k}_0)^2, \qquad v = \frac{d\textcolor{#FF8021}{\omega}}{d\textcolor{#6B913C}{k}}\Big|_{\textcolor{#6B913C}{k}_0} = \frac{\hbar \textcolor{#6B913C}{k}_0}{m} = \frac{p_0}{m}.
    $$

    The middle term, linear in $k-k_0$, changes the phase of the component by $-v\,(k-k_0)\,t$. Together with
    $e^{ikx}$ this turns $x$ into $x - vt$ in the sum, so if this were the only term the whole pattern
    would slide along without changing shape at the speed $v = p_0/m$. That is the velocity of a classical
    particle with momentum $p_0$ and mass $m$. The speed of the lump is called the **group velocity** and is
    $d\omega/dk$, which differs from the speed $\omega/k$ of the individual crests of section 1.3.

    The figure shows a packet with $k_0 = 5$ and $\sigma_x = 1.5$. The grey area is the probability, the blue
    curve is the real part of the amplitude (the individual waves inside the lump), and the dashed line is the
    classical position $x = vt$.
    """)
    return


@app.cell
def _(mo, packet_times_figure, plt):
    _fig = packet_times_figure(5.0, 1.5, [0, 3, 9])
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.4 The lump spreads

    The last term in the expansion, quadratic in $k-k_0$, is the one that does not simply shift the pattern.
    The components fall out of step with one another as time passes, and the lump widens. The exact result is
    that the probability stays a bell curve, centred at $x = vt$, whose width grows as

    $$
    \sigma(t) = \sigma_x\sqrt{1 + \left(\frac{t}{t_s}\right)^{2}}, \qquad t_s = \frac{2m\sigma_x^2}{\hbar}.
    $$

    At the time $t_s$ the width has grown by a factor $\sqrt{2}$. The time-and-space map shows the same
    packet as a whole: it moves along the classical path (dashed) and gets wider and fainter.
    """)
    return


@app.cell
def _(mo, plt, spacetime_figure):
    _fig = spacetime_figure(5.0, 1.5, 9)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.5 When is a particle classical?

    Two facts have been found: the lump travels at the classical velocity, and it spreads in the time $t_s$. A
    particle behaves classically if it travels a long way, compared with its own size, before the spreading
    matters. The distance travelled in the time $t_s$ is

    $$
    v\,t_s = \frac{\hbar \textcolor{#6B913C}{k}_0}{m}\cdot\frac{2m\sigma_x^2}{\hbar} = 2\textcolor{#6B913C}{k}_0\,\sigma_x^2 ,
    \qquad\text{that is}\qquad
    \frac{v\,t_s}{\sigma_x} = 2\,\textcolor{#6B913C}{k}_0\,\sigma_x .
    $$

    So the single number that decides the matter is

    $$
    \textcolor{#6B913C}{k}_0\,\sigma_x = \frac{p_0\,\sigma_x}{\hbar} = \frac{p_0}{2\,\Delta p},
    $$

    which compares the size of the lump with the wavelength ($\sigma_x/\lambda = k_0\sigma_x/2\pi$). The last
    form uses $\Delta p = \hbar/2\sigma_x$ and says that **$k_0\sigma_x$ is large exactly when the momentum is
    known to a small fraction of itself**. When $k_0\sigma_x \gg 1$:

    - the lump contains many wavelengths, so the waves that make it up all have nearly the same $k$ and
      therefore nearly the same velocity;
    - the lump travels many times its own width before it spreads appreciably, so for all practical purposes
      it follows the path $x = vt$.

    When $k_0\sigma_x$ is of order one, the position and the momentum cannot both be sharp enough to speak
    of a path, and the quantum character is plain. Two examples, at the values of the physical constants:

    | Particle | Localized to | $k_0\sigma_x$ | Time to spread, $t_s$ |
    |---|---|---|---|
    | Electron, kinetic energy 1 eV | 1 nm | about 5 | about 17 femtoseconds |
    | Dust grain, 1 mg moving at 1 mm/s | 1 µm | about $10^{19}$ | about $6\times10^{8}$ years |

    The electron's lump travels only about ten of its own widths before it has spread, whereas the grain's
    lump spreads so slowly that no experiment could ever detect it. The figure below uses $k_0 = 5$ and lets
    $\sigma_x$ and the time be varied. Narrow lumps (small $k_0\sigma_x$) spread almost at once, and wide
    lumps (large $k_0\sigma_x$) travel far with almost no change of shape.
    """)
    return


@app.cell
def _(mo):
    sigma_slider = mo.ui.slider(0.3, 4.0, step=0.1, value=1.0, label=r"initial width $\sigma_x$", show_value=True)
    spread_time_slider = mo.ui.slider(0, 6, step=0.1, value=2.0, label=r"time $t$", show_value=True)
    return sigma_slider, spread_time_slider


@app.cell
def _(
    group_velocity,
    mo,
    plt,
    sigma_slider,
    spread_time_slider,
    spreading_figure,
    spreading_time,
    wavelengths_in_lump,
    widths_travelled_before_spreading,
    width,
):
    _k0 = 5.0
    _s = sigma_slider.value
    _t = spread_time_slider.value
    _fig = spreading_figure(_k0, _s, _t)
    _readout = mo.md(
        rf"""
        With $k_0 = {_k0:g}$ and $\sigma_x = {_s:g}$:
        $k_0\sigma_x = {wavelengths_in_lump(_k0, _s):.2f}$,
        $t_s = {spreading_time(_s):.2f}$,
        and the lump travels $2k_0\sigma_x = {widths_travelled_before_spreading(_k0, _s):.1f}$ of its own widths
        in the time $t_s$. At $t = {_t:g}$ its width is $\sigma(t) = {width(_t, _s):.2f}$, which is
        {width(_t, _s) / _s:.2f} times the initial width, and it has moved to $x = vt = {group_velocity(_k0) * _t:.1f}$.
        """
    )
    _out = mo.vstack([mo.hstack([sigma_slider, spread_time_slider], justify="start"), _fig, _readout])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 The total probability does not change

    Normalization would be useless if it held at one moment and failed at the next. It holds for all time,
    because of the equation that governs $\psi$. Each plane wave of section 5 has $\omega = \hbar k^2/2m$.
    Substituting $\psi = e^{i(kx-\omega t)}$ shows that this is the same as the equation

    $$
    i\hbar\,\frac{\partial \textcolor{#5ECCF3}{\psi}}{\partial t}
    = -\frac{\hbar^2}{2m}\,\frac{\partial^2 \textcolor{#5ECCF3}{\psi}}{\partial x^2} + V(x)\,\textcolor{#5ECCF3}{\psi},
    $$

    the **Schrödinger equation**. (For $V=0$ the left side gives $\hbar\omega\,\psi$ and the right side
    $\hbar^2k^2/2m\;\psi$. The equation is linear, so it holds for any sum of plane waves, which is every
    packet. The term $V(x)$ is a potential energy and is a real number at each $x$.) Its complex conjugate is

    $$
    {-}i\hbar\,\frac{\partial \textcolor{#5ECCF3}{\psi}^{*}}{\partial t}
    = -\frac{\hbar^2}{2m}\,\frac{\partial^2 \textcolor{#5ECCF3}{\psi}^{*}}{\partial x^2} + V(x)\,\textcolor{#5ECCF3}{\psi}^{*}.
    $$

    **How the density changes in time.** The product rule gives
    $\partial_t\rho = \psi^{*}\,\partial_t\psi + \psi\,\partial_t\psi^{*}$. Solve the two equations above for
    $\partial_t\psi$ and $\partial_t\psi^{*}$ and substitute:

    $$
    \frac{\partial \rho}{\partial t}
    = \frac{i\hbar}{2m}\left(\textcolor{#5ECCF3}{\psi}^{*}\frac{\partial^2 \textcolor{#5ECCF3}{\psi}}{\partial x^2} -
    \textcolor{#5ECCF3}{\psi}\frac{\partial^2 \textcolor{#5ECCF3}{\psi}^{*}}{\partial x^2}\right) -
    \frac{i}{\hbar}V\,|\textcolor{#5ECCF3}{\psi}|^2 + \frac{i}{\hbar}V\,|\textcolor{#5ECCF3}{\psi}|^2 .
    $$

    The two terms with $V$ cancel, because $V$ is real and so is the same in both equations. What remains is
    a derivative of a single quantity, since
    $\psi^{*}\psi'' - \psi\,\psi^{*\prime\prime} = \dfrac{\partial}{\partial x}\big(\psi^{*}\psi' - \psi\,\psi^{*\prime}\big)$:

    $$
    \frac{\partial \rho}{\partial t} + \frac{\partial j}{\partial x} = 0,
    \qquad
    j(x,t) = \frac{\hbar}{2mi}\left(\textcolor{#5ECCF3}{\psi}^{*}\frac{\partial \textcolor{#5ECCF3}{\psi}}{\partial x} -
    \textcolor{#5ECCF3}{\psi}\frac{\partial \textcolor{#5ECCF3}{\psi}^{*}}{\partial x}\right)
    = \frac{\hbar}{m}\,\mathrm{Im}\!\left(\textcolor{#5ECCF3}{\psi}^{*}\frac{\partial \textcolor{#5ECCF3}{\psi}}{\partial x}\right).
    $$

    This is the **continuity equation**, the same as for the mass of a fluid. $j$ is the **probability current**,
    the probability crossing the point $x$ per unit time, and the equation says that the probability in a small
    region changes only by flowing across its ends. For the plane wave, $\psi^{*}\psi' = ik$, so
    $j = \hbar k/m = p/m = v$ times $\rho = 1$: the flat density flows at the classical velocity.

    **The total.** Integrate the continuity equation over all $x$:

    $$
    \frac{d}{dt}\int_{-\infty}^{\infty} \rho\,dx = -\big[\,j\,\big]_{-\infty}^{\infty} = 0 ,
    $$

    because a normalizable $\psi$ goes to zero far away, and with it $j$. If the total is 1 at one time it is 1 at
    every time. The two properties used were that the equation is linear with the conjugate equation obtained by
    $i\to -i$, and that $V$ is real. A complex $V$ would leave a term $\propto \mathrm{Im}\,V\,|\psi|^2$ that
    makes probability appear or disappear, which is how absorption is modelled.
    """)
    return


@app.cell
def _(mo, np, packet):
    _rows = []
    for _t in (0, 3, 9, 30):
        _x = np.linspace(-80, 80, 20001) + 5.0 * _t  # grid follows the packet, centred on x = vt
        _rho = np.abs(packet(_x, _t, 5.0, 1.5)) ** 2
        _rows.append(f"| {_t} | {np.trapezoid(_rho, _x):.6f} | {_rho.max():.3f} |")
    mo.md(
        "The packet of section 6 ($k_0=5$, $\\sigma_x=1.5$), integrated numerically on a grid. The peak falls as "
        "the lump spreads, but the total does not:\n\n"
        "| time $t$ | $\\int\\rho\\,dx$ | peak of $\\rho$ |\n|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A wave is $H = A\cos(kx-\omega t)$. The size $A$ is the largest height, $\omega$ is the angular frequency
      (period $T=2\pi/\omega$), $k$ is the wave number (wavelength $\lambda=2\pi/k$), and the crests move at
      $\omega/k$.
    - The same wave is the horizontal component of an arrow $A\,e^{i(kx-\omega t)}$ that turns at the rate
      $\omega$ in time and $k$ in space. The two descriptions are identical.
    - The complex form makes the addition of waves easy: arrows add tip to tail, and the intensity is the
      squared length of the sum, $|\psi_1+\psi_2|^2$.
    - For a particle the complex form is required, because a state of definite energy and momentum has uniform
      probability, which a real cosine does not.
    - $E = \hbar\omega$ and $p = \hbar k$ identify these rates with energy and momentum, which are conserved
      because the physics is unchanged by shifts in time and in space. Relativity requires the same constant
      $\hbar$ in both.
    - A superposition of waves with nearby $k$ is a lump that moves at $d\omega/dk = p_0/m$, the classical
      velocity, and spreads in the time $t_s = 2m\sigma_x^2/\hbar$.
    - The squared length $|\psi|^2$, the intensity of the water waves, is the probability density $\rho$ for a
      particle. The probability of an interval is $\int_a^b\rho\,dx$ and the total must be 1. A plane wave has
      $\rho=1$ everywhere and total $2X\to\infty$, so it cannot be normalized.
    - The Schrödinger equation gives $\partial_t\rho + \partial_x j = 0$, so the total probability stays 1
      for all time (for a real potential).
    - The lump behaves like a classical particle when $k_0\sigma_x = p_0/2\Delta p \gg 1$: its momentum is
      known to a small fraction of itself, and it travels $2k_0\sigma_x$ of its own widths before it spreads.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Sources.** The interference of two waves (section 3), the plane wave, $E=\hbar\omega$ and
    $p=\hbar k$ (including the relativistic argument), the Gaussian packet and the uncertainty relation
    $\sigma_x\sigma_{k}=\tfrac12$, and the symmetry argument for energy and momentum follow Chapters 1, 7, 16 and
    17 of R. P. Feynman, R. B. Leighton and M. Sands, *The Feynman Lectures on Physics*, Vol. III: Quantum
    Mechanics (Caltech online edition, edited by M. A. Gottlieb and R. Pfeiffer, 2013),
    <https://www.feynmanlectures.caltech.edu/III_toc.html>. The trigonometric introduction in section 1 is
    standard, and the analysis of the spreading of the packet and of the condition $k_0\sigma_x\gg1$ for
    classical behaviour in section 6 is not taken from those chapters.
    """)
    return


if __name__ == "__main__":
    app.run()
