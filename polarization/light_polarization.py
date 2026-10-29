import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from light import (
        E_MINUS,
        E_PLUS,
        E_X,
        ellipse_axes,
        helicity_amplitudes,
        jones,
        linear,
        mean_m,
        plate_torque,
        retarder,
        ring_radius,
        total_m,
    )
    from light_plots import (
        ellipse_figure,
        longitudinal_figure,
        phase_clock_figure,
        plate_figure,
        rotated_state_figure,
        two_circles_figure,
        vortex_figure,
    )

    return (
        E_MINUS,
        E_PLUS,
        E_X,
        ellipse_axes,
        ellipse_figure,
        helicity_amplitudes,
        jones,
        linear,
        longitudinal_figure,
        mean_m,
        mo,
        np,
        phase_clock_figure,
        plate_figure,
        plate_torque,
        plt,
        retarder,
        ring_radius,
        rotated_state_figure,
        total_m,
        two_circles_figure,
        vortex_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Light Polarization and Angular Momentum

    A light wave has a direction of travel and, across it, a direction in which the electric field points. That
    second direction is the **polarization**. This notebook connects it to angular momentum:

    - the two **circular** polarizations are the states of definite spin, $m_s=+1$ and $m_s=-1$;
    - **linear** polarization is a combination of the two, and the direction of the line is stored in their relative phase;
    - light has no state with $m_s=0$, and no state with $|m_s|>1$, although a beam as a whole can carry any whole number of units of angular momentum;
    - a beam carries angular momentum that it can hand to matter, which is the same rotation argument as for the electron, applied to a particle that really is a vector.

    **Read first:** section 3 of the spinor notebook (the spin of ordinary vectors) helps, but the argument is
    repeated here, so nothing else is needed.

    **Contents**

    1. What polarization is: the electric field as a turning or oscillating arrow.
    2. Turning the beam about its axis: the states of definite angular momentum.
    3. Linear light is made of two circular lights.
    4. Why only $m_s=\pm1$.
    5. Light carries angular momentum: absorbers and wave plates.
    6. Beyond spin: twisted beams.
    7. Compared with the electron.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Electric field** $\mathbf E$: an arrow attached to each point of space. A light wave is the electric
          field (together with the magnetic field) oscillating in time.
        - **Polarization**: the direction in which the electric field arrow points, and how it moves.
        - **Complex number**: $a+ib$, drawn as an arrow in a plane. Its length is its size and its angle is its
          **phase**. The number $e^{i\theta}$ has size 1 and phase $\theta$.
        - **Amplitude**: a complex number that describes part of a state. The probability of finding the system in
          that part is the square of the amplitude's size.
        - **Angular momentum** $m_s$ (units of $\hbar$) **along the beam**: how much turning about the direction of
          travel the light carries. A photon is one quantum of light, with energy $\hbar\omega$.
        - **Eigenvector** of a matrix: a vector that the matrix only multiplies by a number.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. What polarization is

    Take a light wave travelling along $z$. At one point in space the electric field is an arrow in the $xy$-plane
    that changes with time at the angular frequency $\omega$. Write each component as the real part of a complex
    amplitude turning at that frequency:

    $$E_x=\operatorname{Re}\!\left[a_x\,e^{-i\omega t}\right],\qquad E_y=\operatorname{Re}\!\left[a_y\,e^{-i\omega t}\right].$$

    The pair $(a_x,a_y)$ is the **state** of the polarization (it is called the Jones vector). Its overall size sets
    the brightness and is fixed here, $|a_x|^2+|a_y|^2=1$, so two numbers remain: how the light divides between
    $x$ and $y$, and the phase difference $\delta$ between the two parts. Write

    $$(a_x,\,a_y)=(\cos\beta,\ e^{i\delta}\sin\beta),\qquad\text{so}\qquad E_x=\cos\beta\,\cos\omega t,\quad E_y=\sin\beta\,\cos(\omega t-\delta).$$

    The $y$ part lags the $x$ part by the angle $\delta$. The tip of the arrow traces out a curve:

    - $\delta=0^\circ$ or $180^\circ$: the two parts rise and fall together, and the tip moves back and forth along a
      straight line. This is **linear** polarization.
    - $\beta=45^\circ$ and $\delta=\pm90^\circ$: equal parts a quarter of a period apart. The tip moves round a circle
      at a steady rate, once per period. This is **circular** polarization. For $\delta=+90^\circ$ it turns
      counterclockwise, seen with the wave coming towards you, and for $\delta=-90^\circ$ clockwise.
    - Anything else: an **ellipse**.

    Move the sliders and watch the curve and the two components.
    """)
    return


@app.cell
def _(mo):
    pol_beta = mo.ui.slider(0, 90, step=5, value=30, label="split between x and y, β (degrees)", full_width=True)
    pol_delta = mo.ui.slider(0, 360, step=5, value=60, label="phase difference, δ (degrees)", full_width=True)
    pol_wt = mo.ui.slider(0, 360, step=5, value=70, label="time, ωt (degrees)", full_width=True)
    mo.vstack([pol_beta, pol_delta, pol_wt])
    return pol_beta, pol_delta, pol_wt


@app.cell
def _(ellipse_axes, ellipse_figure, jones, mean_m, mo, np, plt, pol_beta, pol_delta, pol_wt):
    _j = jones(np.radians(pol_beta.value), np.radians(pol_delta.value))
    _major, _minor, _angle = ellipse_axes(_j)
    _ratio = _minor / _major
    _sense = "counterclockwise" if mean_m(_j) > 0 else "clockwise"
    if _ratio < 0.01:
        _kind = f"**Linear** polarization, along the line at {np.degrees(_angle) % 180:.0f}° to $x$."
    elif _ratio > 0.99:
        _kind = f"**Circular** polarization, turning {_sense}."
    else:
        _kind = (
            f"**Elliptical** polarization, turning {_sense}; semi-axes {_major:.2f} and {_minor:.2f}, "
            f"the long one at {np.degrees(_angle) % 180:.0f}° to $x$."
        )
    _fig = ellipse_figure(pol_beta.value, pol_delta.value, pol_wt.value)
    _out = mo.vstack([_fig, mo.md(_kind)])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Turning the beam about its axis

    Rotate the whole arrangement through an angle $\alpha$ about the direction of travel $z$. The state
    $(a_x,a_y)$ becomes

    $$\begin{pmatrix}a_x'\\ a_y'\end{pmatrix}=R(\alpha)\begin{pmatrix}a_x\\ a_y\end{pmatrix},\qquad
    R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\ \sin\alpha&\cos\alpha\end{pmatrix}.$$

    As in section 2 of the spinor notebook, the angular momentum is read off from this matrix as
    $J=i\,\dfrac{dR}{d\alpha}\Big|_{\alpha=0}$, which here is

    $$J=i\begin{pmatrix}0&-1\\ 1&0\end{pmatrix}=\begin{pmatrix}0&-i\\ i&0\end{pmatrix},\qquad R(\alpha)=e^{-i\alpha J}.$$

    The states of definite angular momentum $m_s$ are the vectors that a rotation only multiplies by a number
    (eigenvectors of $J$): $J\mathbf e=m_s\mathbf e$. For $\mathbf e=(x,y)$ this reads $-iy=m_s x$ and $ix=m_s y$. The
    first gives $y=i\,m_s\,x$. Put this in the second: $ix=i\,m_s^2\,x$, so $m_s^2=1$. There are two values,
    $m_s=+1$ and $m_s=-1$, with the states

    $$\mathbf e_+=\frac1{\sqrt2}\begin{pmatrix}1\\ i\end{pmatrix}\ (m_s=+1),\qquad
    \mathbf e_-=\frac1{\sqrt2}\begin{pmatrix}1\\ -i\end{pmatrix}\ (m_s=-1),\qquad
    R(\alpha)\,\mathbf e_\pm=e^{\mp i\alpha}\,\mathbf e_\pm .$$

    What do they look like? For $\mathbf e_+$ the field is $E_x=\tfrac1{\sqrt2}\cos\omega t$ and
    $E_y=\tfrac1{\sqrt2}\sin\omega t$, the counterclockwise circle of section 1 ($\beta=45^\circ$, $\delta=+90^\circ$).
    For $\mathbf e_-$ it is the clockwise circle. **The two circular polarizations are the states of definite angular
    momentum $m_s=+1$ and $m_s=-1$ along the beam.**[^sign] Turning a circular beam only shifts the clock: the
    turning arrow looks the same, and the arrow that was at one point of the circle at $t=0$ is now at another.

    Any state can be written as a combination $\mathbf a=a_+\mathbf e_++a_-\mathbf e_-$ of the two circular ones. The
    probabilities of finding $m_s=+1$ and $m_s=-1$ are $|a_+|^2$ and $|a_-|^2$, as for the spin states of the spinor
    notebook. A rotation multiplies the amplitudes by $e^{-i\alpha}$ and $e^{+i\alpha}$: the right-hand panel below
    shows the two arrows turning in opposite directions.

    [^sign]: The sign convention is that $m_s=+1$ is the state that turns counterclockwise seen with the wave coming
        towards the viewer, which is positive angular momentum along $+z$. Books on optics call the two circular
        states left and right, and they do not all agree on which is which, because the name depends on the viewpoint
        and on the sign convention for time.
    """)
    return


@app.cell
def _(mo):
    state_choice = mo.ui.dropdown(
        options=[
            "linear, along x",
            "linear, at 45° to x",
            "circular, counterclockwise (m_s = +1)",
            "circular, clockwise (m_s = -1)",
            "elliptical",
        ],
        value="circular, counterclockwise (m_s = +1)",
        label="state before the turn",
    )
    rot_alpha = mo.ui.slider(0, 360, step=5, value=60, label="rotation about the beam, α (degrees)", full_width=True)
    mo.vstack([state_choice, rot_alpha])
    return rot_alpha, state_choice


@app.cell
def _(E_MINUS, E_PLUS, E_X, helicity_amplitudes, jones, linear, mo, np, plt, rot_alpha, rotated_state_figure, state_choice):
    _states = {
        "linear, along x": E_X,
        "linear, at 45° to x": linear(np.pi / 4),
        "circular, counterclockwise (m_s = +1)": E_PLUS,
        "circular, clockwise (m_s = -1)": E_MINUS,
        "elliptical": jones(np.radians(25), np.radians(70)),
    }
    _short = {k: k.split(" (")[0] for k in _states}
    _j0 = _states[state_choice.value]
    _fig = rotated_state_figure(_j0, rot_alpha.value, _short[state_choice.value])
    _alpha = int(rot_alpha.value)

    def _phase(a):
        return f"{np.degrees(np.angle(a)):.0f}°" if abs(a) > 1e-9 else "none (amplitude zero)"

    _a_plus, _a_minus = helicity_amplitudes(_j0)
    _out = mo.vstack([
        _fig,
        mo.md(
            f"Before the turn the amplitudes have phases {_phase(_a_plus)} ($m_s=+1$) and {_phase(_a_minus)} ($m_s=-1$). "
            + f"The turn through {_alpha}° changes them by {-_alpha}° and {_alpha:+d}°, and leaves the probabilities alone."
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Linear light is made of two circular lights

    A linear state at the angle $\alpha$ from $x$ is, by the same reasoning, an equal combination of the two circular
    states:

    $$\begin{pmatrix}\cos\alpha\\ \sin\alpha\end{pmatrix}=\frac1{\sqrt2}\left(e^{-i\alpha}\,\mathbf e_++e^{i\alpha}\,\mathbf e_-\right).$$

    To check, multiply out: the first component is $\tfrac12\left(e^{-i\alpha}+e^{i\alpha}\right)=\cos\alpha$ and the
    second is $\tfrac{i}{2}\left(e^{-i\alpha}-e^{i\alpha}\right)=\sin\alpha$.

    Two things follow.

    - A linear state has $m_s=+1$ and $m_s=-1$ with probability $\tfrac12$ each, **whatever the angle**. A
      measurement of $m_s$ does not tell linear states at different angles apart.
    - The angle is stored in the **relative phase** of the two amplitudes, $\delta=2\alpha$. This is the relative
      phase of the two-amplitude model of section 5 of the spinor notebook. There the relative phase advanced by
      $\alpha$ for the two values $m_s=\pm\tfrac12$; here the two values differ by 2, so it advances by $2\alpha$.[^relphase]

    In terms of the field, each amplitude is an arrow of length $|a_\pm|/\sqrt2$ that turns at the rate $\omega$,
    $a_+$ counterclockwise and $a_-$ clockwise. The field is their sum. With equal lengths the two arrows line up
    twice per period on the same line (at the angle $\alpha$) and cancel across it, so the tip moves along a line.
    With unequal lengths they add to a long axis (the sum of the lengths) and subtract to a short axis (the
    difference), and the tip moves round an ellipse whose long axis is at the angle $\tfrac12\delta$. One amplitude
    zero gives a circle.

    [^relphase]: A turn through $\alpha=\pi$ multiplies both amplitudes by $e^{\mp i\pi}=-1$. The field is then the
        negative of what it was, which is the same state with the clock set half a period later, so a line comes
        back after $\pi$ although the amplitudes themselves come back after $2\pi$.
    """)
    return


@app.cell
def _(mo):
    tc_alpha = mo.ui.slider(0, 360, step=5, value=30, label="angle of the long axis, α (degrees)", full_width=True)
    tc_chi = mo.ui.slider(0, 90, step=5, value=45, label="share of m_s = +1: |a₊| = cos χ, |a₋| = sin χ; χ (degrees)", full_width=True)
    tc_wt = mo.ui.slider(0, 360, step=5, value=70, label="time, ωt (degrees)", full_width=True)
    mo.vstack([tc_alpha, tc_chi, tc_wt])
    return tc_alpha, tc_chi, tc_wt


@app.cell
def _(mo, np, plt, tc_alpha, tc_chi, tc_wt, two_circles_figure):
    _fig = two_circles_figure(tc_alpha.value, tc_chi.value, tc_wt.value)
    _p = np.cos(np.radians(tc_chi.value)) ** 2
    if abs(tc_chi.value - 45) < 1e-9:
        _kind = "equal shares: **linear** polarization."
    elif tc_chi.value in (0, 90):
        _kind = "one amplitude is zero: **circular** polarization."
    else:
        _kind = "unequal shares: **elliptical** polarization."
    _out = mo.vstack([
        _fig,
        mo.md(
            f"Probability of $m_s=+1$: **{_p:.2f}**, of $m_s=-1$: **{1 - _p:.2f}**; relative phase "
            + f"$\\delta=2\\alpha=${(2 * tc_alpha.value) % 360}°; " + _kind
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Why only $m_s=\pm1$

    **Why not $m_s=0$, or $m_s=2$?** Rotation symmetry alone (section 3 of the spinor notebook) allows any integer
    for a boson, and says nothing about which values a particular kind of particle has. For one particle the values of
    $m_s$ run in steps of one from $-s$ to $+s$, and the spin $s$ is fixed by the kind of field that describes it. A
    field with no direction attached (a scalar) has $s=0$ and the single value $m_s=0$. A field whose value at each
    point is an arrow (a vector) has $s=1$, the case of section 3 of the spinor notebook. The ripples of gravity would
    be described by a field with two indices (a tensor) and have $s=2$. All of these are bosons: "boson" says only that
    $s$ is a whole number, not which one. The electric field is an arrow, so light has $s=1$ and at most the three
    values $m_s=-1,0,+1$.

    **Why not even $m_s=0$?** In the language of section 2, the three states of an arrow are $\mathbf e_+$, $\mathbf e_-$
    and the arrow along $z$, which has $m_s=0$ (it is not changed by a turn about $z$). For light this last state is the
    electric field pointing along the direction of travel. Electric field lines start and end on electric charge, which
    in formulas is $\nabla\cdot\mathbf E=\rho/\varepsilon_0$: the divergence of the field (how much the arrows spread out
    from a point) equals the charge density $\rho$ divided by a constant. In empty space $\rho=0$. A field along the beam
    would be $E_z=a_z\cos(kz-\omega t)$, with divergence $\partial E_z/\partial z=-k\,a_z\sin(kz-\omega t)$, which is
    zero everywhere only if $a_z=0$. Arrows pointing along the beam would pile up in some places and thin out in others,
    which needs charge; arrows across the beam do not. A light wave in empty space is therefore **transverse**, only
    $\mathbf e_+$ and $\mathbf e_-$ remain, and light has two polarizations, not three.[^massless]

    [^massless]: This used Maxwell's equations. There is also an argument that does not depend on the equations. The
        three states of a spin-1 particle are counted in the frame where the particle is at rest. Light travels at the
        fastest possible speed and has no such frame, and for a particle of that kind with spin $s$ only the two values
        $m_s=\pm s$ along the direction of travel remain. A spin-1 particle with mass, such as the W boson, has all three.

    The larger whole numbers are not lost, only moved: a beam can carry orbital angular momentum of any whole number
    of units (section 6).
    """)
    return


@app.cell
def _(mo):
    long_wt = mo.ui.slider(0, 360, step=5, value=40, label="time, ωt (degrees)", full_width=True)
    long_wt
    return (long_wt,)


@app.cell
def _(long_wt, longitudinal_figure, mo, plt):
    _fig = longitudinal_figure(long_wt.value)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Light carries angular momentum

    The spin of light is not only a label: a circularly polarized beam carries angular momentum that it can hand to
    matter. A simple argument shows how much. Let the beam fall on an absorber whose charges are dragged round by the
    turning field at the same rate $\omega$. A body that is turned at the angular velocity $\omega$ by a torque $\tau$
    takes in energy at the rate $P=\tau\omega$ (power is torque times angular velocity). The energy absorbed per
    second is the beam's, and the torque is the angular momentum absorbed per second, so

    $$\frac{\text{angular momentum}}{\text{energy}}=\frac{\tau}{P}=\frac1\omega .$$

    One photon has energy $\hbar\omega$ (section 5 of the plane-wave notebook), so it carries angular momentum
    $\hbar$ along the beam: $m_s=+1$ for the counterclockwise state, with the opposite sign for the other. This agrees
    with the rotation argument of section 2.

    **A wave plate.** A slice of crystal in which the $y$ part of the field travels more slowly than the $x$ part
    delays $y$ by an angle $\Gamma$ (the retardance). It multiplies the amplitudes by

    $$\begin{pmatrix}a_x\\ a_y\end{pmatrix}\ \longrightarrow\ \begin{pmatrix}1&0\\ 0&e^{i\Gamma}\end{pmatrix}\begin{pmatrix}a_x\\ a_y\end{pmatrix}.$$

    For incoming $m_s=+1$ light, $\mathbf e_+=(1,i)/\sqrt2$, the outgoing amplitudes on the two circular states are

    $$a_+=\tfrac12\left(1+e^{i\Gamma}\right),\qquad a_-=\tfrac12\left(1-e^{i\Gamma}\right),\qquad
    |a_+|^2=\cos^2\tfrac\Gamma2,\quad |a_-|^2=\sin^2\tfrac\Gamma2 .$$

    The average value of $m_s$ after the plate is $\cos\Gamma$. Angular momentum about the beam is conserved, so a photon
    that enters with $m_s=+1$ and leaves with an average $\cos\Gamma$ leaves $1-\cos\Gamma$ units of $\hbar$ behind in the
    plate. Two cases:

    - **Half-wave plate** ($\Gamma=180^\circ$): the light leaves with $m_s=-1$, the opposite circular state. Each photon
      changes its angular momentum by $2\hbar$, and the plate is pushed round by that amount.
    - **Quarter-wave plate** ($\Gamma=90^\circ$): circular light becomes linear, with average $m_s=0$, and the plate takes
      $\hbar$ per photon.

    The plate can take angular momentum because its fast axis picks a direction in the plane, and so breaks the
    rotation symmetry that conserved $m_s$ for the beam on its own (section 3 of the spinor notebook). Try a state and
    a retardance below.
    """)
    return


@app.cell
def _(mo):
    plate_in = mo.ui.dropdown(
        options=[
            "circular, counterclockwise (m_s = +1)",
            "circular, clockwise (m_s = -1)",
            "linear, at 45° to the fast axis",
            "linear, along the fast axis",
        ],
        value="circular, counterclockwise (m_s = +1)",
        label="light entering the plate",
    )
    plate_gamma = mo.ui.slider(0, 360, step=5, value=90, label="retardance, Γ (degrees)", full_width=True)
    mo.vstack([plate_in, plate_gamma])
    return plate_gamma, plate_in


@app.cell
def _(E_MINUS, E_PLUS, E_X, linear, mean_m, mo, np, plate_figure, plate_gamma, plate_in, plate_torque, plt, retarder):
    _states = {
        "circular, counterclockwise (m_s = +1)": E_PLUS,
        "circular, clockwise (m_s = -1)": E_MINUS,
        "linear, at 45° to the fast axis": linear(np.pi / 4),
        "linear, along the fast axis": E_X,
    }
    _j_in = _states[plate_in.value]
    _plate = retarder(np.radians(plate_gamma.value))
    _fig = plate_figure(_j_in, plate_gamma.value, plate_in.value.split(" (")[0])
    _out = mo.vstack([
        _fig,
        mo.md(
            f"Average $m_s$ before: **{mean_m(_j_in):+.2f}**, after: **{mean_m(_plate @ _j_in):+.2f}**. "
            + f"Angular momentum left in the plate: **{plate_torque(_j_in, _plate):+.2f}** $\\hbar$ per photon."
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Beyond spin: twisted beams

    Spin is not the only angular momentum a beam can have. The wave can also wind round the axis like a spiral
    staircase, with a phase that advances by $2\pi m_\ell$ in one trip round. Writing $\varphi$ for the angle round the
    axis, such a wave contains the factor $e^{i m_\ell\varphi}$. After one full trip the wave must return to its own
    value, so $m_\ell$ is a **whole number, positive, negative or zero**, and nothing limits its size.

    Turn the beam through $\alpha$ about the axis. The factor becomes $e^{im_\ell(\varphi-\alpha)}=e^{-im_\ell\alpha}\,e^{im_\ell\varphi}$.
    This is the same rule as in section 2: the turn only multiplies the wave by a phase, $e^{-im_\ell\alpha}$, so the
    beam has definite angular momentum $m_\ell$ along the axis. It is orbital angular momentum, of the kind that gives the
    $m_\ell$ of electrons in atoms. On the axis itself the phase is undefined, so the brightness has to vanish there
    when $m_\ell\neq0$: a twisted beam is a bright ring with a dark core, and the ring grows with $|m_\ell|$.

    A beam with polarization $m_s$ and winding $m_\ell$ is multiplied by both phases when it is turned, $e^{-i(m_\ell+m_s)\alpha}$, so

    $$m_j=m_\ell+m_s .$$

    Since $m_s$ is $+1$ or $-1$ and $m_\ell$ is any whole number, **the total $m_j$ can be any whole number.** This is where
    the "any integer" of the Noether section lives, for light. A photon in such a beam carries $\hbar m_j$ along the axis.

    In the figure the colour gives the phase of the wave and the brightness gives its size. Going once round the
    axis the colours cycle $|m_\ell|$ times.
    """)
    return


@app.cell
def _(mo):
    vortex_ml = mo.ui.slider(-4, 4, step=1, value=2, label="winding, m_ℓ", full_width=True)
    vortex_ms = mo.ui.radio(options=["+1", "-1"], value="+1", label="polarization, m_s", inline=True)
    mo.vstack([vortex_ml, vortex_ms])
    return vortex_ml, vortex_ms


@app.cell
def _(mo, plt, ring_radius, total_m, vortex_figure, vortex_ml, vortex_ms):
    _fig = vortex_figure(vortex_ml.value)
    _ms = int(vortex_ms.value)
    _ring = f"The brightest ring has radius {ring_radius(vortex_ml.value):.2f} (in units of the beam width)." if vortex_ml.value else "No winding: the beam is brightest on the axis."
    _out = mo.vstack([
        _fig,
        mo.md(
            f"$m_\\ell={vortex_ml.value}$ and $m_s={_ms:+d}$ give the total $m_j={total_m(vortex_ml.value, _ms):+d}$. " + _ring
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, total_m):
    _rows = ["| winding $m_\\ell$ | " + " | ".join(f"${m}$" for m in range(-3, 4)) + " |", "|---|" + "---|" * 7]
    for _ms in (1, -1):
        _rows.append(f"| total $m_j$ for $m_s={_ms:+d}$ | " + " | ".join(f"${total_m(m, _ms)}$" for m in range(-3, 4)) + " |")
    mo.md(
        "The total angular momentum for a few windings:\n\n" + "\n".join(_rows)
        + "\n\nThe same total can come from different pairs. For example $m_j=0$ comes from $(m_\\ell,m_s)=(-1,+1)$ and from "
        + "$(+1,-1)$: two beams with no net angular momentum, one a twisted beam with the opposite spin, the other with the "
        + "opposite twist."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Compared with the electron

    Light and the electron are both described by a pair of states, and in both cases a rotation multiplies each state
    by a phase $e^{-im_s\alpha}$. The difference is the value of $m_s$: $\pm1$ for light and $\pm\tfrac12$ for the
    electron. After a full turn $\alpha=2\pi$ the factor is $e^{\mp2\pi i}=+1$ for light and
    $e^{\mp i\pi}=-1$ for the electron. The figure follows the factors (the arrows) as the beam, or the electron, is
    turned through $\alpha$; the arcs show where each has been.

    In the basis $\mathbf e_+,\mathbf e_-$ the rotation of the polarization is the matrix

    $$\begin{pmatrix}e^{-i\alpha}&0\\ 0&e^{i\alpha}\end{pmatrix}=e^{-i\alpha\sigma_z},\qquad \sigma_z=\begin{pmatrix}1&0\\ 0&-1\end{pmatrix},$$

    while for the electron it is $e^{-i\alpha\sigma_z/2}$ (section 5 of the spinor notebook). The two look alike with the
    angle doubled. This is why a polarization can be handled like a spin-1/2 system,[^poincare] and why it must not be
    mistaken for one: for light the doubled angle brings the state back after $2\pi$, so there is no sign change, no
    $4\pi$ rotation and no belt trick. Light is a boson, and the electron is not.

    [^poincare]: The pair of amplitudes $(a_+,a_-)$ can be drawn as a point on a sphere, the Poincaré sphere. Because of
        the doubled angle, turning the light through $\alpha$ turns that point through $2\alpha$, where the point of
        the electron's Bloch sphere turns through $\alpha$.
    """)
    return


@app.cell
def _(mo):
    clock_alpha = mo.ui.slider(0, 4, step=0.02, value=1, label="rotation angle α / π", full_width=True)
    clock_alpha
    return (clock_alpha,)


@app.cell
def _(clock_alpha, mo, phase_clock_figure, plt):
    _fig = phase_clock_figure(clock_alpha.value)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - The two circular polarizations are the states of definite angular momentum along the beam, $m_s=+1$ and
      $m_s=-1$. They are the eigenvectors of $J=i\,dR/d\alpha$ for the rotation of the plane across the beam.
    - Linear polarization is an equal combination of the two, and its angle is half the relative phase of the two
      amplitudes. Unequal combinations are ellipses.
    - The spin of light is limited to $\pm1$ because the electric field is an arrow (spin 1) and a wave in empty space
      has no component along its direction of travel (no $m_s=0$).
    - A circular beam carries $\hbar$ per photon, which a wave plate or an absorber can take. The plate works because
      it breaks the rotation symmetry.
    - A twisted beam adds any whole number $m_\ell$ of orbital angular momentum, so the total $m_j=m_\ell+m_s$ can be any
      whole number.
    - Light returns to itself after a turn of $2\pi$ (integer spin); the electron needs $4\pi$.
    """)
    return


if __name__ == "__main__":
    app.run()
