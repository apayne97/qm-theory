import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from spinors import (
        bloch_vector,
        interference_intensity,
        phasor_pair,
        rotation,
        rotation_axis_angle,
        rotation_matrix_so3,
        spin_matrices,
        su2_to_so3,
    )
    from belt_plots import belt_figure, double_turn_ball_figure, single_turn_ball_figure
    from dirac_belt import belt_frames
    from rotation_plots import (
        circulation_figure,
        component_planes_figure,
        stern_gerlach_figure,
        x_oscillation_figure,
        z_rotation_figure,
    )

    def latex_matrix(m, digits=2):
        """Format a small real/complex matrix as a LaTeX pmatrix."""

        def _fmt(x):
            x = complex(np.round(x, digits))
            re, im = x.real + 0.0, x.imag + 0.0
            if im == 0:
                return f"{re:g}"
            if re == 0:
                return f"{im:g}i"
            return f"{re:g}{im:+g}i"

        rows = r" \\ ".join(" & ".join(_fmt(x) for x in row) for row in np.atleast_2d(m))
        return r"\begin{pmatrix}" + rows + r"\end{pmatrix}"

    return (
        belt_figure,
        component_planes_figure,
        belt_frames,
        bloch_vector,
        circulation_figure,
        double_turn_ball_figure,
        interference_intensity,
        latex_matrix,
        mo,
        np,
        phasor_pair,
        plt,
        rotation,
        rotation_axis_angle,
        rotation_matrix_so3,
        single_turn_ball_figure,
        spin_matrices,
        stern_gerlach_figure,
        su2_to_so3,
        x_oscillation_figure,
        z_rotation_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Mathematical Foundations of Bosons and Fermions

    All known particles fall into two classes. **Bosons** (photons, for example) can occupy the same quantum
    state in any number. **Fermions** (electrons, protons, neutrons) cannot: this is the Pauli exclusion
    principle, which gives atoms their shell structure and underlies chemistry.

    The two classes are distinguished by two related properties:

    - **Rotation.** The quantum state of a boson is unchanged by a rotation through $2\pi$ ($360^\circ$). The
      state of a fermion changes sign, and a rotation through $4\pi$ ($720^\circ$) is required to restore it.
      Bosons have integer spin and fermions have half-integer spin.
    - **Exchange.** Interchanging two identical bosons leaves the joint state unchanged (it is symmetric);
      interchanging two identical fermions changes its sign (it is antisymmetric).

    The spin-statistics theorem of relativistic quantum field theory links the two: integer spin goes with
    symmetric exchange and half-integer spin with antisymmetric exchange. This notebook develops the
    mathematics behind these statements step by step, starting from the rotation of ordinary vectors.

    **Read first:** nothing; this notebook starts from the rotation of ordinary vectors and is independent of
    the wave notebooks.

    **Contents**

    **Part I: Rotations and spin**

    1. Rotations and why they matter: symmetry and conservation laws, rotation matrices, and the fact that rotations about different axes do not commute.
    2. The $J$ matrices obtained from rotation matrices.
    3. The spin number of ordinary vectors (spin 1).
    4. Why this construction does not describe the electron, and why the electron matters.
    5. Building the description of spin 1/2: amplitudes and sign, a two-amplitude model, the matrices, the $2\pi$ rotation, and the double cover.
    6. The belt trick: the $2\pi$ and $4\pi$ rotations as a shape, and the space of rotations as a ball.

    Later parts will treat the exchange of identical particles and the link between exchange and spin.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Matrix**: a rectangular array of numbers. Multiplying a vector by a matrix produces a new vector.
        - **Complex number**: a number $a+ib$, which can be drawn as a two-dimensional arrow. Its length is its
          size and its angle is its **phase**. The number $e^{i\theta}$ has size 1 and phase $\theta$.
        - **State**: the list of complex numbers (**amplitudes**) that describes a quantum system. For a
          two-level system such as the electron's spin, the state has two amplitudes and is also called a **spinor**.
        - **Eigenvalue**: for a matrix $J$, a number $\lambda$ for which $Jv=\lambda v$ holds for some nonzero
          vector $v$. When $J$ represents a measurement, its eigenvalues are the possible results.
        - **Boson** and **fermion**: the two classes of particle described in the introduction; bosons have
          integer spin and fermions half-integer spin.
        - **$J$** denotes angular momentum. **$m_s$** denotes the value of the component of spin along the $z$ axis.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Part I: Rotations and spin

    The first part develops the mathematics of rotations and shows how spin, and the sign change of
    half-integer spin under a $2\pi$ rotation, follow from it.

    ## 1 Rotations and why they matter

    **Why study rotations?** Symmetries of a physical situation are tied to conserved quantities
    (Noether's theorem). If the physics is unchanged when everything is rotated about some axis, then the
    component of angular momentum, and hence of spin, along that axis does not change with time. The way a
    quantum state transforms under rotations therefore determines what spin is, and the conservation of
    angular momentum follows from rotational symmetry. This section sets up the mathematics of rotations; a
    test of the symmetry argument closes the section.

    A **matrix** is a rectangular array of numbers; multiplying a vector $(x, y)$ by a matrix produces a new
    vector. The matrix that rotates every vector in the plane through an angle $\alpha$ is

    $$R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}.$$

    The columns of $R$ are the images of the unit vectors $\hat x=(1,0)$ and $\hat y=(0,1)$. The slider
    below sets the angle; the corresponding matrix is displayed beneath the figure.
    """)
    return


@app.cell
def _(mo):
    angle2d = mo.ui.slider(0, 360, step=5, value=30, label="rotation angle (degrees)", full_width=True)
    angle2d
    return (angle2d,)


@app.cell
def _(angle2d, latex_matrix, mo, np, plt):
    _t = np.radians(angle2d.value)
    _r = np.array([[np.cos(_t), -np.sin(_t)], [np.sin(_t), np.cos(_t)]])
    _fig, _ax = plt.subplots(figsize=(3.8, 3.8))
    for _col, _color, _name in [(0, "tab:red", "x̂"), (1, "tab:green", "ŷ")]:
        _e = np.eye(2)[:, _col]
        _v = _r @ _e
        _ax.annotate("", xy=_e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.75", lw=1.5))
        _ax.annotate("", xy=_v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5))
        _ax.text(_v[0] * 1.15, _v[1] * 1.15, _name + "'", color=_color, ha="center", va="center")
    _ax.set_xlim(-1.4, 1.4)
    _ax.set_ylim(-1.4, 1.4)
    _ax.set_aspect("equal")
    _ax.axhline(0, color="0.9", lw=0.8)
    _ax.axvline(0, color="0.9", lw=0.8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("$$R = " + latex_matrix(_r) + "$$")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The same rotation acting on complex numbers.** The matrix $R(\alpha)$ has real entries, but it can also be
    applied to a vector $(x,y)$ whose components are complex numbers. Each component is then an arrow in its own
    complex plane, and the matrix combines the two arrows: the new $x$ is $x\cos\alpha-y\sin\alpha$ and the new
    $y$ is $x\sin\alpha+y\cos\alpha$. For $\hat x=(1,0)$, $\hat y=(0,1)$ or $(1,1)/\sqrt2$ both components stay on the real
    axis and only change size. For a vector with a complex component the result is different. Take
    $(1,\,i)$: since $\sin\alpha+i\cos\alpha=i\,e^{-i\alpha}$,

    $$R(\alpha)\begin{pmatrix}1\\ i\end{pmatrix}=\begin{pmatrix}\cos\alpha-i\sin\alpha\\ \sin\alpha+i\cos\alpha\end{pmatrix}
    =e^{-i\alpha}\begin{pmatrix}1\\ i\end{pmatrix},$$

    so both arrows turn through the same angle $-\alpha$ and keep their lengths, and the vector is only multiplied
    by a phase. The same holds for $(1,-i)$ with $+\alpha$. These are the two vectors that a rotation in the plane
    multiplies by a number; section 2 shows where they come from.

    The left panel of the figure below is the plane of the figure above. A vector with complex components is a
    pair of ordinary arrows in this plane, its real part (solid) and its imaginary part (dashed), and the
    rotation turns each of them through $\alpha$ like any other arrow. The two complex planes to the right show
    the resulting $x$ and $y$ components.
    """)
    return


@app.cell
def _(mo):
    state_choice_2d = mo.ui.dropdown(
        options={
            "x̂ = (1, 0)": (1, 0),
            "ŷ = (0, 1)": (0, 1),
            "(x̂ + iŷ)/√2": (1 / 2**0.5, 1j / 2**0.5),
            "(x̂ − iŷ)/√2": (1 / 2**0.5, -1j / 2**0.5),
            "(x̂ + ŷ)/√2 = (1, 1)/√2": (1 / 2**0.5, 1 / 2**0.5),
            "(1, 1 + i)/√3": (1 / 3**0.5, (1 + 1j) / 3**0.5),
        },
        value="(x̂ + iŷ)/√2",
        label="vector before the rotation",
    )
    angle_2d_c = mo.ui.slider(0, 360, step=5, value=50, label="rotation angle (degrees)", full_width=True)
    mo.vstack([state_choice_2d, angle_2d_c])
    return angle_2d_c, state_choice_2d


@app.cell
def _(angle_2d_c, component_planes_figure, latex_matrix, mo, np, plt, state_choice_2d):
    _v0 = np.array(state_choice_2d.value, dtype=complex)
    _t = np.radians(angle_2d_c.value)
    _r = np.array([[np.cos(_t), -np.sin(_t)], [np.sin(_t), np.cos(_t)]])
    _v = _r @ _v0
    _fig = component_planes_figure(_v0, angle_2d_c.value)
    _out = mo.vstack([
        _fig,
        mo.md(r"$$R(\alpha)\,v=" + latex_matrix(_r) + latex_matrix(_v0.reshape(-1, 1)) + "=" + latex_matrix(_v.reshape(-1, 1)) + "$$"),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Performing two rotations in succession corresponds to multiplying their matrices. In the plane, a rotation
    through $a$ followed by one through $b$ is a rotation through $a+b$, so the order is immaterial.

    In three dimensions there is a basic rotation about each axis, and each leaves its own axis fixed:

    $$R_x(\alpha)=\begin{pmatrix}1&0&0\\0&\cos\alpha&-\sin\alpha\\0&\sin\alpha&\cos\alpha\end{pmatrix},\;
    R_y(\alpha)=\begin{pmatrix}\cos\alpha&0&\sin\alpha\\0&1&0\\-\sin\alpha&0&\cos\alpha\end{pmatrix},\;
    R_z(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha&0\\\sin\alpha&\cos\alpha&0\\0&0&1\end{pmatrix}.$$

    The matrix $R_z$ contains the plane rotation $R(\alpha)$ in its upper left block, so a rotation about $z$ is
    the plane rotation applied to the $x$ and $y$ coordinates, with the $z$ coordinate unchanged. The figure
    shows both views of one rotation, applied to the letter F (an asymmetric shape, so that the turn is easy
    to follow): in the plane, where the origin is the only fixed point, and in space, where every point of the $z$
    axis stays fixed. The dashed outline marks the initial position.
    """)
    return


@app.cell
def _(mo):
    angle_z = mo.ui.slider(0, 360, step=5, value=40, label="rotation about z (degrees)", full_width=True)
    angle_z
    return (angle_z,)


@app.cell
def _(angle_z, mo, plt, z_rotation_figure):
    _fig = z_rotation_figure(angle_z.value)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **In three dimensions the order of rotations matters.** The figure shows an object (represented by its
    three axes: $\hat x$ red, $\hat y$ green, $\hat z$ blue) rotated about the fixed $x$ and $y$ axes in the two
    possible orders. The dashed triad marks the initial orientation.
    """)
    return


@app.cell
def _(mo):
    alpha_deg = mo.ui.slider(0, 180, step=5, value=90, label="rotation about x (degrees)", full_width=True)
    beta_deg = mo.ui.slider(0, 180, step=5, value=90, label="rotation about y (degrees)", full_width=True)
    mo.vstack([alpha_deg, beta_deg])
    return alpha_deg, beta_deg


@app.cell
def _(alpha_deg, beta_deg, mo, np, plt, rotation_axis_angle, rotation_matrix_so3):
    _rx = rotation_matrix_so3((1, 0, 0), np.radians(alpha_deg.value))
    _ry = rotation_matrix_so3((0, 1, 0), np.radians(beta_deg.value))
    _orders = [("x first, then y", _ry @ _rx), ("y first, then x", _rx @ _ry)]

    _fig = plt.figure(figsize=(10, 4.8))
    for _i, (_title, _r) in enumerate(_orders):
        _ax = _fig.add_subplot(1, 2, _i + 1, projection="3d")
        for _k, _color in enumerate(["tab:red", "tab:green", "tab:blue"]):
            _e = np.eye(3)[:, _k]
            _ax.plot(*zip([0, 0, 0], _e), color="0.6", ls="--", lw=1)
            _v = _r @ _e
            _ax.plot(*zip([0, 0, 0], _v), color=_color, lw=4)
            _ax.plot([_v[0]], [_v[1]], [_v[2]], "o", color=_color, ms=7)
        _ax.set_xlim(-1, 1)
        _ax.set_ylim(-1, 1)
        _ax.set_zlim(-1, 1)
        _ax.set_box_aspect((1, 1, 1))
        _ax.view_init(elev=20, azim=35)
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.set_zlabel("z")
        _ax.set_title(_title, fontsize=10)
    _fig.tight_layout()

    _axis, _angle = rotation_axis_angle(_orders[0][1] @ _orders[1][1].T)
    mo.vstack([
        _fig,
        mo.md(
            f"The two final orientations differ by a rotation through **{np.degrees(_angle):.1f}°** about "
            f"the direction ({_axis[0]:+.2f}, {_axis[1]:+.2f}, {_axis[2]:+.2f}). "
            "The two orders agree when either angle is 0° or 180°."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The only property of three-dimensional rotations required below is therefore that **rotations about
    different axes do not commute**.[^noncommute]

    [^noncommute]: For small rotations the mismatch can be computed exactly: rotating by $\epsilon$ about $x$,
        then $\epsilon$ about $y$, then back about $x$ and back about $y$ gives a rotation of approximately
        $\epsilon^2$ about $z$. This is the group-level form of the commutation relation
        $[J_x, J_y]=iJ_z$. The calculation, and a check that it holds for every spin, are in the companion
        notebook `archived_commutator_loop_and_ladder.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A test of the symmetry argument.** A spinning object with angular momentum $\mathbf L$ in a magnetic
    field along $\hat n$ experiences a torque that makes $\mathbf L$ precess. The motion is a steady rotation
    about $\hat n$, so $\mathbf L(t)=R_{\hat n}(\omega t)\,\mathbf L(0)$, with the rotation matrices above. The
    situation is unchanged by any rotation about $\hat n$ (a symmetry), so the prediction is that the component
    of $\mathbf L$ along $\hat n$ is constant while the other components oscillate. The slider sets the
    direction of the field in the $xz$-plane. The same conclusion holds for quantum spin, where the conserved
    quantity is the $J$ matrix of section 2.[^noether]

    [^noether]: The precise quantum statement is that the Hamiltonian $\hat H$ commutes with $J$, so that
        $d\langle J\rangle/dt=i\langle[\hat H,J]\rangle=0$. An argument that does not use commutators is given in
        section 3. A quantum calculation of this example, with the
        commutator norms for each component, is in the companion notebook
        `archived_commutator_loop_and_ladder.py`.
    """)
    return


@app.cell
def _(mo):
    field_beta = mo.ui.slider(0, 1, step=0.05, value=0.25, label="field direction β / π (angle from z, in the xz-plane)", full_width=True)
    field_beta
    return (field_beta,)


@app.cell
def _(field_beta, np, plt, rotation_matrix_so3):
    _beta = field_beta.value * np.pi
    _n = np.array([np.sin(_beta), 0.0, np.cos(_beta)])
    _l0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # arbitrary starting direction
    _omega = 2.0
    _ts = np.linspace(0, 2 * np.pi, 300)
    _l = np.array([rotation_matrix_so3(_n, _omega * _t) @ _l0 for _t in _ts])

    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _k, _label in enumerate(["L along x", "L along y", "L along z"]):
        _ax.plot(_ts, _l[:, _k], label=_label)
    _ax.plot(_ts, _l @ _n, "k--", lw=2, label="L along the field")
    _ax.set_xlabel("time")
    _ax.legend(fontsize=8, ncol=2, loc="lower center")
    _ax.set_title("only the component along the field is constant", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2 The $J$ matrices obtained from rotation matrices

    Quantum spin is described by three matrices $J_x$, $J_y$, $J_z$, one for each axis. They are obtained
    directly from the rotation matrices.

    For a rotation through a very small angle $\epsilon$, $\cos\epsilon\approx 1$ and $\sin\epsilon\approx\epsilon$,
    so the rotation matrix differs from the identity by a term proportional to $\epsilon$:
    $R(\epsilon)\approx 1+\epsilon K$. For a rotation about $z$,

    $$R_z(\epsilon)\approx\begin{pmatrix}1&-\epsilon&0\\ \epsilon&1&0\\0&0&1\end{pmatrix}
    =1+\epsilon\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix},\qquad
    K_z=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}.$$

    The same matrix follows by differentiating. The matrix $K$ is the derivative of the rotation matrix with
    respect to the angle, evaluated at zero angle: $K = dR/d\alpha\,\big|_{\alpha=0}$.[^jacobian] Differentiate
    each entry of $R_z(\alpha)$, using $\dfrac{d\cos\alpha}{d\alpha}=-\sin\alpha$ and
    $\dfrac{d\sin\alpha}{d\alpha}=\cos\alpha$; the constant entries (the $1$ and the zeros) have derivative $0$:

    $$\frac{dR_z}{d\alpha}=\begin{pmatrix}-\sin\alpha&-\cos\alpha&0\\ \cos\alpha&-\sin\alpha&0\\0&0&0\end{pmatrix}.$$

    At $\alpha=0$ we have $\sin 0=0$ and $\cos 0=1$, so
    $K_z=\dfrac{dR_z}{d\alpha}\Big|_{\alpha=0}=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}$, as above. A rotation through a finite angle $\alpha$ is a composition
    of many small ones, $R(\alpha)=\lim_{n\to\infty}(1+\tfrac{\alpha}{n}K)^n=e^{\alpha K}$, the matrix analogue
    of continuously compounded interest.

    The $J$ matrix is defined as $J=iK$, so that $R(\alpha)=e^{-i\alpha J}$. The factor $i$ ensures that $J$
    has real eigenvalues, as required of a quantity whose measured values are real. For the rotation about $z$,
    $J_z=iK_z=\begin{pmatrix}0&-i&0\\ i&0&0\\0&0&0\end{pmatrix}$. Hence **each entry of a
    $J$ matrix is $i$ times the derivative, at zero angle, of the corresponding entry of the rotation matrix.**
    Select an axis below; the matrices are computed by numerical differentiation of the rotation matrices
    of section 1 at zero angle.

    [^jacobian]: $J$ is the standard symbol for angular momentum and should not be confused with the
        Jacobian. There is nevertheless a connection: if every point moves with velocity $Kv$ under a steady
        rotation, then $K$ is the Jacobian of that velocity field, the matrix of all derivatives
        $\partial(\text{velocity})/\partial(\text{position})$.
    """)
    return


@app.cell
def _(mo):
    axis_choice = mo.ui.dropdown(options=["x", "y", "z"], value="z", label="rotation axis")
    axis_choice
    return (axis_choice,)


@app.cell
def _(axis_choice, latex_matrix, mo, np, rotation_matrix_so3):
    _axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis_choice.value]
    _h = 1e-5
    _k = (rotation_matrix_so3(_axis, _h) - rotation_matrix_so3(_axis, -_h)) / (2 * _h)
    _j = 1j * _k
    _ev = np.linalg.eigvalsh(_j)

    mo.md(
        f"**Rotation about {axis_choice.value}**\n\n"
        "$$K = " + latex_matrix(_k) + r"\quad\Rightarrow\quad J_" + axis_choice.value + "=iK=" + latex_matrix(_j) + "$$\n"
        f"Eigenvalues of $J_{axis_choice.value}$: **{', '.join(f'{x:+.0f}' for x in _ev)}**\n\n"
        "The eigenvalues, which are the possible measured values of spin along the chosen axis, are the same "
        "for every axis, as expected since no direction in space is special."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3 The spin number of ordinary vectors

    The eigenvalues of $J_z$ are the possible values of the component of spin along $z$, denoted $m_s$. For the
    matrices of section 2, $m_s=+1,0,-1$. A system with these three values is said to have **spin 1**; the
    spin number $j$ is the largest value of $m_s$.

    A vector $v=(v_1,v_2,v_3)$ is an **eigenvector** of $J_z$, with **eigenvalue** $\lambda$, if $J_z$ only
    multiplies it by a number: $J_zv=\lambda v$. With the matrix $J_z$ of section 2 this reads

    $$\begin{pmatrix}0&-i&0\\ i&0&0\\ 0&0&0\end{pmatrix}\begin{pmatrix}v_1\\ v_2\\ v_3\end{pmatrix}
    =\lambda\begin{pmatrix}v_1\\ v_2\\ v_3\end{pmatrix}
    \quad\Longleftrightarrow\quad -i\,v_2=\lambda v_1,\qquad i\,v_1=\lambda v_2,\qquad 0=\lambda v_3.$$

    *Eigenvalues.* Suppose $\lambda\neq0$. The third equation gives $v_3=0$, and the second gives
    $v_2=iv_1/\lambda$. Substituting into the first, $-i\,(iv_1/\lambda)=v_1/\lambda=\lambda v_1$, so
    $\lambda^2=1$ (otherwise $v_1=0$ and then $v_2=0$ as well, which is no vector at all). Hence $\lambda=+1$ or
    $\lambda=-1$. For $\lambda=0$ the first two equations give $v_1=v_2=0$, while $v_3$ is unrestricted. The
    eigenvalues are therefore $+1,\ 0,\ -1$, the values of $m_s$ quoted above.

    *Eigenvectors.* For $\lambda=+1$ the equation $v_2=iv_1/\lambda$ gives $v_2=iv_1$ and $v_3=0$, so $v\propto(1,i,0)$.
    For $\lambda=-1$ it gives $v_2=-iv_1$, so $v\propto(1,-i,0)$. For $\lambda=0$, $v\propto(0,0,1)$. Dividing the
    first two by $\sqrt2$ makes each vector have length $1$:

    $$m_s=+1:\ \tfrac1{\sqrt2}\begin{pmatrix}1\\ i\\ 0\end{pmatrix},\qquad
    m_s=0:\ \begin{pmatrix}0\\ 0\\ 1\end{pmatrix},\qquad
    m_s=-1:\ \tfrac1{\sqrt2}\begin{pmatrix}1\\ -i\\ 0\end{pmatrix}.$$

    Only the vector along $z$ is one of the basic vectors $\hat x,\hat y,\hat z$. A rotation about $z$ turns $\hat x$ into
    a mixture of $\hat x$ and $\hat y$, so $\hat x$ is not multiplied by a number; the vectors with $m_s=\pm1$ are the
    combinations $\hat x\pm i\hat y$.

    **What the eigenvectors mean.** Multiply an eigenvector by the steady phase $e^{-i\omega t}$ and take the real
    part, which is the arrow itself as it would be seen over time. For $m_s=+1$,

    $$\mathrm{Re}\left[e^{-i\omega t}\,\tfrac1{\sqrt2}\begin{pmatrix}1\\ i\\ 0\end{pmatrix}\right]
    =\tfrac1{\sqrt2}\begin{pmatrix}\cos\omega t\\ \sin\omega t\\ 0\end{pmatrix},$$

    an arrow of constant length turning **counter-clockwise** about $z$ at a constant rate. For $m_s=-1$ the sign of
    the $y$ component is reversed and the arrow turns **clockwise**. For $m_s=0$ the arrow only points along $z$,
    so it does not circulate about $z$ at all. The states of definite $m_s$ are therefore the states of constant
    circulation about the axis: counter-clockwise for $m_s=+1$ (angular momentum along $+z$, by the right-hand
    rule), clockwise for $m_s=-1$, and none for $m_s=0$.

    This explains the rotation rule. An arrow turning steadily about $z$ looks the same after a rotation about
    $z$, only at a different moment of its cycle. Turning the picture through $\alpha$ is the same as advancing
    $\omega t$ by $\alpha$ for $m_s=+1$ (and by $-\alpha$ for $m_s=-1$), which multiplies the vector by the phase
    $e^{-im_s\alpha}$. A back-and-forth oscillation along $x$ behaves differently; the example after the figure
    shows why it is not an eigenvector of $J_z$.
    """)
    return


@app.cell
def _(mo):
    circ_phase = mo.ui.slider(0, 360, step=5, value=40, label="time, ωt (degrees)", full_width=True)
    circ_phase
    return (circ_phase,)


@app.cell
def _(circ_phase, circulation_figure, mo, plt):
    _fig = circulation_figure(circ_phase.value)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **An example: why $\hat x$ is not an eigenvector.** The vector $\hat x$, drawn over time, is an arrow
    oscillating back and forth along $x$. Write $e_{+1}$ and $e_{-1}$ for the eigenvectors with $m_s=+1$ and $m_s=-1$.
    Adding them,

    $$\hat x=\frac1{\sqrt2}\,(e_{+1}+e_{-1}),\qquad e_{\pm1}=\tfrac1{\sqrt2}\begin{pmatrix}1\\ \pm i\\ 0\end{pmatrix},$$

    because the $y$ entries $i$ and $-i$ cancel and the $x$ entries add. In the circulation picture, a
    counter-clockwise arrow and a clockwise arrow of the same length always have opposite $y$ components, so their
    sum has no $y$ component: it only moves back and forth along $x$ (the last panel of the figure above).

    Now rotate through an angle $\alpha$ about $z$. Each part is an eigenvector, so each is multiplied by its
    own phase, $e^{-i\alpha}$ for $m_s=+1$ and $e^{+i\alpha}$ for $m_s=-1$:

    $$R_z(\alpha)\,\hat x=\frac1{\sqrt2}\left(e^{-i\alpha}e_{+1}+e^{i\alpha}e_{-1}\right)
    =\begin{pmatrix}\cos\alpha\\ \sin\alpha\\ 0\end{pmatrix}=\cos\alpha\,\hat x+\sin\alpha\,\hat y.$$

    The two phases are different, so the result is not the original vector times one number; it is a different
    vector, unless $\alpha$ is a multiple of $\pi$. An eigenvector would come back as itself times a phase. In the
    picture, turning both arrows through $\alpha$ in the plane puts the counter-clockwise arrow ahead in its cycle
    by $\alpha$ and the clockwise arrow behind by $\alpha$. Their $y$ components then no longer cancel. The sum is
    again an oscillation, but along the direction at angle $\alpha$ from the $x$ axis. The figure shows the
    oscillation before and after the turn.
    """)
    return


@app.cell
def _(mo):
    xosc_wt = mo.ui.slider(0, 360, step=5, value=50, label="time, ωt (degrees)", full_width=True)
    xosc_alpha = mo.ui.slider(0, 360, step=5, value=60, label="rotation about z, α (degrees)", full_width=True)
    mo.vstack([xosc_wt, xosc_alpha])
    return xosc_alpha, xosc_wt


@app.cell
def _(mo, plt, x_oscillation_figure, xosc_alpha, xosc_wt):
    _fig = x_oscillation_figure(xosc_wt.value, xosc_alpha.value)
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The rotation applied to these vectors.** The matrix $R_z(\alpha)$ of this section sends the basic vectors to

    $$R_z\hat x=\cos\alpha\,\hat x+\sin\alpha\,\hat y,\qquad R_z\hat y=-\sin\alpha\,\hat x+\cos\alpha\,\hat y,$$

    and neither result is a number times the original. For the combination $\hat x+i\hat y$, which has amplitudes
    $(1,\,i,\,0)$, the sum of the two results is

    $$R_z(\hat x+i\hat y)=(\cos\alpha-i\sin\alpha)\,\hat x+(\sin\alpha+i\cos\alpha)\,\hat y
    =e^{-i\alpha}\,(\hat x+i\hat y),$$

    since $\sin\alpha+i\cos\alpha=i(\cos\alpha-i\sin\alpha)=i\,e^{-i\alpha}$. In the same way,
    $R_z(\hat x-i\hat y)=e^{+i\alpha}(\hat x-i\hat y)$. These are the phases $e^{-im_s\alpha}$ for
    $m_s=+1$ and $m_s=-1$.

    The same result follows from $R_z(\alpha)=e^{-i\alpha J_z}$. For a vector $v$ with $J_zv=m_sv$ we have
    $J_z^nv=m_s^nv$ for every power $n$, so, writing the exponential as its series,

    $$e^{-i\alpha J_z}v=\sum_{n=0}^{\infty}\frac{(-i\alpha)^n}{n!}J_z^nv
    =\sum_{n=0}^{\infty}\frac{(-i\alpha m_s)^n}{n!}\,v=e^{-im_s\alpha}\,v.$$

    For $m_s=0$ the phase is $1$, which is the statement that $\hat z$ is unchanged by a rotation about $z$.

    **The components as complex numbers.** In general the three components $(v_1,v_2,v_3)$ of a state are complex
    numbers, and each can be drawn as an arrow in its own complex plane. For an ordinary arrow in space, such as
    $\hat x$, the components are real and stay on the real axis: a rotation about $z$ only moves amplitude between
    the $x$ and $y$ components. For the states with $m_s=\pm1$ the components are complex. A rotation turns the $x$
    and $y$ arrows together through the same angle, $-\alpha$ for $m_s=+1$ and $+\alpha$ for $m_s=-1$, and the $y$
    arrow stays a quarter turn ahead of the $x$ arrow for $m_s=+1$ and behind it for $m_s=-1$. The lengths of the
    arrows, the sizes of the amplitudes, stay constant and only their common phase advances: this is what it means
    for a vector to be an eigenvector of $J_z$, since the whole vector is multiplied by a single phase. The $z$
    component does not change. The figure applies $R_z(\alpha)$ to the chosen state: grey arrows show the components before
    the rotation, coloured arrows after it, and the curves the paths followed as $\alpha$ grows.
    """)
    return


@app.cell
def _(mo):
    state_choice_c = mo.ui.dropdown(
        options={
            "x̂, an arrow along x": (1, 0, 0),
            "ŷ, an arrow along y": (0, 1, 0),
            "ẑ, an arrow along z (m_s = 0)": (0, 0, 1),
            "(x̂ + iŷ)/√2 (m_s = +1)": (1 / 2**0.5, 1j / 2**0.5, 0),
            "(x̂ − iŷ)/√2 (m_s = −1)": (1 / 2**0.5, -1j / 2**0.5, 0),
            "(1, 1, 1)/√3": (1 / 3**0.5, 1 / 3**0.5, 1 / 3**0.5),
            "(1, 1 + i, 1)/2": (0.5, (1 + 1j) / 2, 0.5),
        },
        value="(x̂ + iŷ)/√2 (m_s = +1)",
        label="state before the rotation",
    )
    angle_c = mo.ui.slider(0, 360, step=5, value=50, label="rotation about z (degrees)", full_width=True)
    mo.vstack([state_choice_c, angle_c])
    return angle_c, state_choice_c


@app.cell
def _(angle_c, component_planes_figure, latex_matrix, mo, np, plt, rotation_matrix_so3, state_choice_c):
    _v0 = np.array(state_choice_c.value, dtype=complex)
    _v = rotation_matrix_so3((0, 0, 1), np.radians(angle_c.value)) @ _v0
    _fig = component_planes_figure(_v0, angle_c.value)
    _out = mo.vstack([
        _fig,
        mo.md(
            r"$$R_z(\alpha)\,v=" + latex_matrix(_v.reshape(-1, 1)) + "$$"
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Express a state through its amplitudes on the three eigenvectors of $J_z$. Since $R(\alpha)=e^{-i\alpha J}$
    and $J_z$ has the value $m_s$ on each of these eigenvectors, **a rotation through $\alpha$ about $z$
    multiplies the amplitude of the level with value $m_s$ by the phase $e^{-im_s\alpha}$.** For integer $m_s$ these
    phases are all equal to $1$ at $\alpha=2\pi$, so the state returns to itself after a full turn, as expected
    for an ordinary vector.

    **Rotational symmetry and the conservation of $m_s$.** The phase $e^{-im_s\alpha}$ gives the quantum form of the
    conservation law of section 1. The argument has four steps. Write $R_z(\alpha)$ for a turn about $z$ and $U$
    for letting a state evolve for a time.

    1. **Symmetry.** Suppose the physical situation is unchanged by rotations about $z$. Then turning the state
       and afterwards letting it evolve gives the same result as letting it evolve and afterwards turning it:
       $R_z(\alpha)\,U=U\,R_z(\alpha)$ for every $\alpha$.
    2. **A state of definite $m_s$.** This is an eigenvector of $J_z$. A rotation about $z$ multiplies all of its
       components by the same phase $e^{-im_s\alpha}$, as in the figure above. The sizes of its components, and
       the phases between them (for $m_s=\pm1$, $v_y=\pm i\,v_x$), do not change under the turn.
    3. **Evolution keeps this property.** Let $v$ have definite $m_s$ and let $w=Uv$ be the evolved state. Then
       $R_z w=R_zUv=UR_zv=U\,(e^{-im_s\alpha}v)=e^{-im_s\alpha}\,w$. The evolved state is multiplied by the same
       phase, so all of its components again turn together and it has the same $m_s$.
    4. **Conservation.** The value of $m_s$, the component of angular momentum along $z$, therefore does not change,
       whatever the details of the interaction: **a state of definite $m_s$ keeps that value of $m_s$ for all
       time.**[^translations]

    If the symmetry is broken, step 1 fails and the rest does not follow. With a field pointing along $x$, for
    example, turning and then evolving differs from evolving and then turning. The evolved state is then a mixture
    of different values of $m_s$, so its components no longer turn together under $R_z$: angular momentum along
    $z$ has been exchanged with the source of the field, as in the classical precession above.

    The figure shows this for the state with $m_s=+1$. The state is turned through an angle about $z$, which
    respects the symmetry, or about $x$, which does not (a turn about $x$ stands in for the effect of a field
    along $x$). Under the turn about $z$ the $x$ and $y$ arrows keep turning together. Under the turn about $x$
    they do not, and the probabilities of finding each value of $m_s$, shown beneath the figure, spread out from
    $m_s=+1$.

    [^translations]: The same argument applied to shifts of position and of time gives the conservation of
        momentum and of energy; see section 5 of the companion notebook
        `../plane_waves/plane_waves_and_packets.py`. In the language of matrices, the statement is that the
        operation of letting a state evolve commutes with the rotation matrices, which is the commutation of
        $\hat H$ with $J$ mentioned in section 1.
    """)
    return


@app.cell
def _(mo):
    evolve_axis = mo.ui.radio(
        options={
            "about z (the symmetry is respected)": (0, 0, 1),
            "about x (the symmetry is broken)": (1, 0, 0),
        },
        value="about z (the symmetry is respected)",
        label="turn",
    )
    evolve_angle = mo.ui.slider(0, 360, step=5, value=60, label="angle of the turn (degrees)", full_width=True)
    mo.vstack([evolve_axis, evolve_angle])
    return evolve_angle, evolve_axis


@app.cell
def _(component_planes_figure, evolve_angle, evolve_axis, mo, np, plt, rotation_matrix_so3):
    _s = 1 / np.sqrt(2)
    _v0 = np.array([_s, 1j * _s, 0])
    _w = rotation_matrix_so3(evolve_axis.value, np.radians(evolve_angle.value)) @ _v0
    _eig = {"+1": np.array([_s, 1j * _s, 0]), "0": np.array([0, 0, 1]), "-1": np.array([_s, -1j * _s, 0])}
    _p = {k: abs(np.vdot(e, _w)) ** 2 for k, e in _eig.items()}
    _fig = component_planes_figure(_v0, evolve_angle.value, axis=evolve_axis.value)
    _out = mo.vstack([
        _fig,
        mo.md(
            "Probability of $m_s=+1$: " + f"**{_p['+1']:.2f}**" + ", of $m_s=0$: " + f"**{_p['0']:.2f}**"
            + ", of $m_s=-1$: " + f"**{_p['-1']:.2f}**"
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **An example: light.** The electric field of a light wave is a vector, so light is a spin-1 system. Take a wave
    travelling along $z$. At a fixed point in space its electric field is an arrow in the $xy$-plane. In
    **circularly polarised** light this arrow turns at a steady rate, making one full turn per period of the wave.
    There are two kinds, according to the sense of the turning. Write the arrow as the complex number
    $E_x+iE_y$. For one sense it is $e^{i\omega t}$, and for the other it is $e^{-i\omega t}$.

    Now rotate the whole setup through an angle $\alpha$ about $z$. The turning arrow is the same arrow at a different
    moment of its cycle, so the amplitude is only multiplied by a phase, $e^{i\alpha}$ for one sense and $e^{-i\alpha}$
    for the other. By the rule above, these are the states with $m_s=\mp1$: **the two senses of circular
    polarisation are the states of definite $m_s=+1$ and $m_s=-1$.** After a rotation through $2\pi$ both phases
    equal $1$, so the field returns to itself, as expected for an ordinary vector. Light has no state with $m_s=0$,
    since that would be a field pointing along $z$, the direction of travel, and the field of a wave is always
    perpendicular to it.[^photon0]

    The figure follows a spin-1 state whose spin points along $+x$ (amplitudes $\tfrac12,\ \tfrac1{\sqrt2},\
    \tfrac12$ on $m_s=+1,0,-1$). Left: the direction of the spin in the $xy$-plane. Right: the three amplitudes,
    each drawn as an arrow in the complex plane.

    [^photon0]: A system with spin 1 normally has all three levels $m_s=+1,0,-1$. Light is a special case because
        it travels at the fastest possible speed; this is why it has only two.
    """)
    return


@app.cell
def _(mo):
    theta_one = mo.ui.slider(0, 2, step=0.02, value=0, label="rotation angle α / π", full_width=True)
    theta_one
    return (theta_one,)


@app.cell
def _(np, plt, rotation, spin_matrices, theta_one):
    _theta = theta_one.value * np.pi
    _psi0 = np.array([1, np.sqrt(2), 1], dtype=complex) / 2
    _psi = rotation(1, (0, 0, 1), _theta) @ _psi0
    _jx, _jy, _ = spin_matrices(1)
    _dir = np.array([np.vdot(_psi, _jx @ _psi).real, np.vdot(_psi, _jy @ _psi).real])
    _circ = np.linspace(0, 2 * np.pi, 200)

    _fig, (_ax_d, _ax_a) = plt.subplots(1, 2, figsize=(8, 4))
    for _ax, _title in [(_ax_d, "direction of the spin (xy-plane)"), (_ax_a, "the three amplitudes")]:
        _ax.plot(np.cos(_circ), np.sin(_circ), color="0.8", lw=1)
        _ax.axhline(0, color="0.9", lw=0.8)
        _ax.axvline(0, color="0.9", lw=0.8)
        _ax.set_xlim(-1.3, 1.3)
        _ax.set_ylim(-1.3, 1.3)
        _ax.set_aspect("equal")
        _ax.set_title(_title, fontsize=10)
    _ax_d.annotate("", xy=_dir, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.5))
    for _c, _label, _color in zip(_psi, ["$m_s=+1$", "$m_s=0$", "$m_s=-1$"], ["tab:orange", "tab:red", "tab:brown"]):
        _tip = _c * 2  # |c| is 1/2 or 1/sqrt(2); a scale of 2 keeps every arrow inside the circle
        _ax_a.annotate("", xy=(_tip.real, _tip.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5))
        _ax_a.text(_tip.real * 1.15 + 0.05, _tip.imag * 1.15 + 0.05, _label, color=_color, fontsize=8)
    _ax_a.set_xlabel(f"overlap with initial state = {np.vdot(_psi0, _psi).real:+.2f}", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Definite spin refers to an axis

    The values $m_s$ above are values of the spin component along $z$. A state with a definite value of the
    component along $z$ does not in general have a definite value along another axis. The eigenvectors of $J_x$
    follow in the same way as those of $J_z$, with $J_x=iK_x$ taken from the rotation about $x$:
    $m_x=+1$ for $\tfrac1{\sqrt2}(0,1,i)$, $m_x=0$ for $\hat x$, and $m_x=-1$ for $\tfrac1{\sqrt2}(0,1,-i)$.

    - The vector $\hat z$ has $m_z=0$ with certainty. Written with the eigenvectors of $J_x$ it is
      $\hat z=\tfrac1{i\sqrt2}\left[\tfrac1{\sqrt2}(0,1,i)-\tfrac1{\sqrt2}(0,1,-i)\right]$, an equal combination of $m_x=+1$
      and $m_x=-1$. A measurement of the component along $x$ gives $+1$ or $-1$, each with probability $\tfrac12$,
      and never $0$.
    - The state $e_{+1}=\tfrac1{\sqrt2}(1,i,0)$ has $m_z=+1$ with certainty. Along $x$ the probabilities of
      $m_x=+1,0,-1$ are $\tfrac14,\tfrac12,\tfrac14$.
    - The state in the last figure points along $+x$, so it has $m_x=+1$ with certainty. Along $z$ the
      probabilities of $m_z=+1,0,-1$ are the squares of its amplitudes, $\tfrac14,\tfrac12,\tfrac14$.

    The reason is the one found in section 1: rotations about different axes do not commute, so the matrices
    $J_x$ and $J_z$ derived from them do not commute either. A state with a definite value of both would have to be
    an eigenvector of both matrices, and no state of this system is.

    "The spin points along $z$" therefore has a precise meaning: **the component along $z$ is definite**, and so is the
    total spin. The components along $x$ and $y$ are then uncertain, with average $0$. It does not mean that the
    three components $x$, $y$, $z$ all have definite values at once, as they would for an ordinary arrow. The one
    exception is a system with spin $0$, for which every component is definitely $0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4 Why this construction does not describe the electron, and why the electron matters

    Sections 2 and 3 started from the rotation of ordinary three-dimensional vectors and therefore produced a
    system with three levels, $m_s=+1,0,-1$. They also showed that a definite spin component along one axis leaves
    the components along the other axes uncertain. The electron is not of this kind.

    **The device.** A Stern-Gerlach magnet has a non-uniform magnetic field along one axis. It deflects each atom
    of a beam according to the value of its spin component along that axis, so it splits the beam into one beam
    for each possible value. Silver atoms are used, because their magnetic moment is that of a single unpaired
    electron.

    **Experiment.** For a spin-1 system, a magnet along $z$ would produce three beams, one for each of
    $m_z=+1,0,-1$. Experimentally **exactly two beams emerge**, for any orientation of the magnet. The spin
    component of the electron along any axis takes only two values, so its state has two amplitudes.

    **Magnets in sequence.** The figure shows three magnets in a row, along $z$, then $x$, then $z$. After each of
    the first two, only the upper beam is kept; the blocked beams are marked ✗. Each label gives the fraction of the
    beam entering a magnet that leaves in that beam. The sequence tests the previous section:

    - A second magnet along $z$ directly after the first would send all of the kept beam into the same upper beam,
      since $m_z$ is definite.
    - The magnet along $x$ splits that beam into equal parts, since a definite $m_z$ leaves $m_x$ uncertain.
    - The last magnet, along $z$, splits the $m_x=+\tfrac12$ beam into equal parts again. Making $m_x$ definite
      destroyed the definite value of $m_z$ prepared by the first magnet.

    The lower row shows the same sequence for spin 1, with three beams and the fractions $\tfrac14,\tfrac12,\tfrac14$
    that were found in section 3. The electron behaves in the same way, except that it has two levels instead of
    three, and a state with two levels needs two amplitudes.
    """)
    return


@app.cell
def _(mo, plt, stern_gerlach_figure):
    _fig = stern_gerlach_figure()
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why it matters.** As the introduction noted, electrons are fermions. Three separate statements are
    involved, and only the first can be addressed with rotations:

    1. **Which spins can exist.** Rotation symmetry, together with the unobservability of the overall sign of a
       state (section 5.1), allows $j=0,\tfrac12,1,\tfrac32,\dots$. Half-integer spin is *permitted*, but rotation
       symmetry does not *require* nature to use it. Experiment does: the two beams above.
    2. **Half-integer spin implies a fermion.** This is the spin-statistics theorem (Pauli, 1940), proved in
       relativistic quantum field theory. It does not follow from rotation symmetry alone and is outside the scope
       of this notebook.
    3. **Consequences.** The exclusion principle produces atomic shells and chemistry.

    What rotations can show is what a two-level system requires of its rotation matrices, and that half-integer
    spin brings a sign change under a $2\pi$ rotation. That is the content of section 5.

    A two-component state cannot be rotated by the $3\times3$ matrices of section 1, which act on
    three-component objects. What is required is a family of $2\times2$ matrices that combine under
    multiplication in the same way as the rotations do. The existence of such matrices is not guaranteed by the
    $3\times3$ case, and finding them is the content of section 5, which ends with $J$ obtained exactly as in
    section 2.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5 Building the description of spin 1/2

    The description is assembled in five steps: how amplitudes and probabilities treat a sign (5.1); a
    minimal model with two amplitudes, which fixes $m_s=\pm\tfrac12$ (5.2); the resulting matrices and $J$
    (5.3); the effect of a $2\pi$ rotation (5.4); and why two matrices describe each rotation (5.5).

    ### 5.1 Amplitudes, probabilities and the sign of a state

    A quantum state is described by a list of complex numbers called **amplitudes**. The probability of an
    outcome is the squared size of the corresponding amplitude,

    $$P = |a|^2 .$$

    Writing $a=|a|e^{i\theta}$, the phase $\theta$ does not enter $P$. In particular, **$a$ and $-a$ give the
    same probability.**

    The sign does matter when amplitudes are *combined*. If a particle can reach a detector by two routes, the
    amplitudes are added before squaring. For two routes of equal weight with amplitudes $a_1$ and $a_2$,

    $$P = \left|\frac{a_1 + a_2}{2}\right|^2 .$$

    This depends on the relative phase of the two amplitudes. Taking $a_1=1$ and $a_2=e^{i\delta}$ gives
    $P=\cos^2(\delta/2)$: the routes reinforce at $\delta=0$ ($P=1$) and cancel at $\delta=\pi$, that is for
    $a_2=-1$ ($P=0$), even though $|a_2|^2=1$ in both cases.

    **An example.** A hydrogen molecular ion consists of two protons sharing one electron. Write its state as
    the pair of amplitudes $(a_1,a_2)$ for finding the electron near proton 1 or near proton 2. Reflecting the
    whole system in the plane halfway between the protons exchanges the two positions, so it turns
    $(a_1,a_2)$ into $(a_2,a_1)$. The state $\tfrac1{\sqrt2}(1,1)$ is unchanged by the reflection. The state
    $\tfrac1{\sqrt2}(1,-1)$ becomes $\tfrac1{\sqrt2}(-1,1)$, which is minus itself. In both states the
    probability of finding the electron near either proton is $\tfrac12$, before and after the reflection, so no
    probability of either state is altered. Yet the second state has had its sign reversed, and a sign is exactly
    what shows itself when the state is combined with another. Two reflections restore any state, so the only
    factors by which a reflection can multiply a state are $+1$ and $-1$. The same argument returns for the
    exchange of identical particles.

    The consequence for rotations is the following. **A rotation that multiplies the amplitudes of a state by
    $-1$ leaves every probability of that state unchanged, yet the change can be detected by interfering the
    rotated state with an unrotated copy.** Such a sign is therefore permitted in the rotation matrices of a
    quantum system, and it is exactly what a rotation through $2\pi$ does to the electron (section 5.4). The
    slider below varies the phase of $a_2$.
    """)
    return


@app.cell
def _(mo):
    amp_phase = mo.ui.slider(0, 2, step=0.02, value=0, label="phase of the second amplitude, δ / π", full_width=True)
    amp_phase
    return (amp_phase,)


@app.cell
def _(amp_phase, np, plt):
    _phi = amp_phase.value * np.pi
    _a1, _a2 = 1.0 + 0j, np.exp(1j * _phi)
    _comb = (_a1 + _a2) / 2
    _phis = np.linspace(0, 2 * np.pi, 300)

    _fig, (_ax_c, _ax_p) = plt.subplots(1, 2, figsize=(10, 4))

    _circ = np.linspace(0, 2 * np.pi, 200)
    _ax_c.plot(np.cos(_circ), np.sin(_circ), color="0.85")
    _ax_c.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.6", lw=5, alpha=0.5))
    _ax_c.text(1.1, -0.12, "a₁", color="0.45")
    _ax_c.annotate("", xy=(_a2.real, _a2.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.5))
    _ax_c.text(_a2.real * 1.15, _a2.imag * 1.15, "a₂", color="tab:blue", ha="center", va="center")
    _ax_c.annotate("", xy=(_comb.real, _comb.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2, ls="--"))
    _ax_c.text(_comb.real * 0.5, _comb.imag * 0.5 - 0.13, "(a₁+a₂)/2", fontsize=8, ha="center")
    _ax_c.set_xlim(-1.4, 1.4)
    _ax_c.set_ylim(-1.4, 1.4)
    _ax_c.set_aspect("equal")
    _ax_c.set_title("amplitudes in the complex plane", fontsize=10)

    _ax_p.plot(_phis / np.pi, np.ones_like(_phis), color="tab:blue", lw=2.5, label="|a₂|²  (a₂ alone)")
    _ax_p.plot(_phis / np.pi, np.cos(_phis / 2) ** 2, color="k", lw=2.5, label="|(a₁+a₂)/2|²  (two routes combined)")
    _ax_p.axvline(_phi / np.pi, color="0.5", ls="--", lw=1)
    _ax_p.set_xlabel("δ / π")
    _ax_p.set_ylabel("probability")
    _ax_p.set_ylim(-0.05, 1.15)
    _ax_p.legend(fontsize=8, loc="lower center")
    _ax_p.set_title(
        f"a₂ alone: P = {abs(_a2) ** 2:.2f}      combined: P = {abs(_comb) ** 2:.2f}", fontsize=10
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 A model with two amplitudes

    An electron has two levels, so the smallest model has two amplitudes of unit size, written as complex numbers
    with relative phase $\delta$ and common phase $\gamma$:

    $$a_1 = e^{i(\gamma - \delta/2)}, \qquad a_2 = e^{i(\gamma + \delta/2)}.$$

    - **Probability.** Combining the two as in section 5.1 gives $P=\left|\tfrac{a_1+a_2}{2}\right|^2=\cos^2(\delta/2)$:
      equal to 1 at $\delta=0$, to 0 at $\delta=\pi$, and repeating with period $2\pi$ in $\delta$. The common phase
      $\gamma$ does not appear.
    - **A complete description requires both $(\gamma,\delta)$**, because $\gamma$ matters as soon as the state is
      compared with a reference. Two descriptions represent the same state only if they give the same pair
      $(a_1,a_2)$. This is *not* equivalent to "same $\gamma$ and same $\delta$": the pair is also unchanged by
      $(\gamma,\delta)\to(\gamma+\pi,\ \delta+2\pi)$. Advancing $\delta$ by $2\pi$ changes the sign of the pair, and
      the same sign change is produced by advancing $\gamma$ by $\pi$.

    Set $\gamma=0$ and advance $\delta$ to $2\pi$: the point does not lie on a star, so the pair is not the initial
    one. Then set $\gamma=\pi$ and it does. Only $\delta=4\pi$ with $\gamma=0$ returns to the initial pair directly.
    """)
    return


@app.cell
def _(mo):
    toy_phi = mo.ui.slider(0, 4, step=0.02, value=0, label="relative phase δ / π", full_width=True)
    toy_gamma = mo.ui.slider(0, 2, step=0.02, value=0, label="common phase γ / π", full_width=True)
    mo.vstack([toy_phi, toy_gamma])
    return toy_phi, toy_gamma


@app.cell
def _(np, phasor_pair, plt, toy_phi, toy_gamma):
    _phi = toy_phi.value * np.pi
    _gamma = toy_gamma.value * np.pi
    _a1, _a2 = phasor_pair(_gamma, _phi)

    _fig, (_ax_plane, _ax_c) = plt.subplots(1, 2, figsize=(10, 4.6))

    # Left: the (gamma, phi) plane. Stars are descriptions with the same amplitudes as (0, 0).
    _stars = [(2 * np.pi * a + np.pi * b, 2 * np.pi * b) for a in range(-2, 3) for b in range(0, 3)]
    _stars = [(t, p) for t, p in _stars if -1e-9 <= t <= 2 * np.pi + 1e-9 and -1e-9 <= p <= 4 * np.pi + 1e-9]
    _ax_plane.plot(*zip(*_stars), "*", color="gold", mec="k", ms=15, label="same pair (a₁, a₂) as (0, 0)")
    _ax_plane.plot([_gamma, _gamma], [0, _phi], color="tab:blue", lw=2)
    _ax_plane.plot([_gamma], [_phi], "o", color="tab:blue", ms=10)
    _ax_plane.set_xlim(-0.4, 2 * np.pi + 0.4)
    _ax_plane.set_ylim(-0.4, 4 * np.pi + 0.4)
    _ax_plane.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    _ax_plane.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi], ["0", "π", "2π", "3π", "4π"])
    _ax_plane.set_xlabel("common phase γ")
    _ax_plane.set_ylabel("relative phase δ")
    _ax_plane.set_title("(γ, δ) plane", fontsize=10)
    _ax_plane.legend(loc="upper right", fontsize=8)

    # Right: the two amplitudes, their average, and the reference (a1 = a2 = 1)
    _circ = np.linspace(0, 2 * np.pi, 200)
    _ax_c.plot(np.cos(_circ), np.sin(_circ), color="0.85")
    _ax_c.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.6", lw=6, alpha=0.5))
    _ax_c.text(1.12, -0.12, "reference", color="0.5", fontsize=8)
    for _a, _name, _color in [(_a1, "a₁", "tab:blue"), (_a2, "a₂", "tab:orange")]:
        _ax_c.annotate("", xy=(_a.real, _a.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5))
        _ax_c.text(_a.real * 1.15, _a.imag * 1.15, _name, color=_color, ha="center", va="center")
    _mean = (_a1 + _a2) / 2
    _ax_c.annotate("", xy=(_mean.real, _mean.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2, ls="--"))
    _ax_c.set_xlim(-1.4, 1.4)
    _ax_c.set_ylim(-1.4, 1.4)
    _ax_c.set_aspect("equal")
    _ax_c.set_title("the two amplitudes (black: (a₁+a₂)/2)", fontsize=10)
    _ax_c.set_xlabel(
        f"probability |(a₁+a₂)/2|² = {abs(_mean) ** 2:.2f}     "
        f"amplitude (a₁+a₂)/2 = {_mean.real:+.2f}{_mean.imag:+.2f}i"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The value of $m_s$.** The observable quantity is the relative phase $\delta$, that is, the direction of the
    arrow $a_1^* a_2=e^{i\delta}$ (for the electron, the direction of the spin in the $xy$-plane), and a rotation
    of space through $\alpha$ must advance it by $\alpha$. If the rotation multiplies the two amplitudes by
    $e^{\mp im_s\alpha}$ (section 3), then $\delta$ advances by $2m_s\alpha$, so $2m_s=1$ and $m_s=\pm\tfrac12$.[^mvalues]
    The amplitudes are the square root of the observable arrow. Advancing both phases together ($\gamma$) is the
    overall phase, which is unobservable on its own; only the difference between the two phase rates is
    physical, and it must equal 1.

    **Limitation.** In a two-dimensional world the same argument permits *any* real value of $m_s$ (such
    particles are called anyons). Restricting $m_s$ to whole and half-integers requires a third dimension, in
    which rotations about different axes do not commute (section 1).

    [^mvalues]: The argument for any number of levels (the spacing between neighbouring levels must be 1 and the
        set of values symmetric about zero), with an interactive check and the ladder-operator derivation, is
        in the companion notebook `archived_commutator_loop_and_ladder.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.3 The matrices and $J$ for spin 1/2

    With $m_s=\pm\tfrac12$ (section 5.2), a rotation through $\alpha$ about $z$ multiplies the two amplitudes by
    $e^{\mp i\alpha/2}$:

    $$U_z(\alpha)=\begin{pmatrix}e^{-i\alpha/2}&0\\0&e^{i\alpha/2}\end{pmatrix}.$$

    The matrices for rotations about $x$ and $y$ are those $2\times2$ matrices that combine under
    multiplication in the same way as the ordinary rotations of section 1:

    $$U_x(\alpha)=\begin{pmatrix}\cos\frac\alpha2&-i\sin\frac\alpha2\\-i\sin\frac\alpha2&\cos\frac\alpha2\end{pmatrix},\qquad
    U_y(\alpha)=\begin{pmatrix}\cos\frac\alpha2&-\sin\frac\alpha2\\\sin\frac\alpha2&\cos\frac\alpha2\end{pmatrix}.$$

    $J$ is now obtained exactly as in section 2: $K=dU/d\alpha$ at zero angle and $J=iK$. Select an axis
    below. The cell also checks the requirement stated above: it computes the ordinary $3\times3$ rotation
    implemented by $U$ and compares it with the matrix of section 1.[^convention]

    [^convention]: These matrices turn the state through $\alpha$. Some texts instead turn the coordinate axes
        through $\alpha$, which is the same as turning the state through $-\alpha$. Each matrix is then replaced
        by its inverse (for $z$, $\mathrm{diag}(e^{+i\alpha/2},e^{-i\alpha/2})$) and $R(\alpha)=e^{-i\alpha J}$
        is written $e^{+i\alpha J}$. The physics is identical.
    """)
    return


@app.cell
def _(mo):
    axis_choice_half = mo.ui.dropdown(options=["x", "y", "z"], value="z", label="rotation axis")
    axis_choice_half
    return (axis_choice_half,)


@app.cell
def _(axis_choice_half, latex_matrix, mo, np, rotation, rotation_matrix_so3, su2_to_so3):
    _axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis_choice_half.value]
    _h = 1e-5
    _k = (rotation(0.5, _axis, _h) - rotation(0.5, _axis, -_h)) / (2 * _h)
    _j = 1j * _k
    _ev = np.linalg.eigvalsh(_j)
    _err = np.abs(su2_to_so3(rotation(0.5, _axis, 0.7)) - rotation_matrix_so3(_axis, 0.7)).max()

    mo.md(
        f"**Rotation about {axis_choice_half.value}, spin 1/2**\n\n"
        "$$K = " + latex_matrix(_k) + r"\quad\Rightarrow\quad J_" + axis_choice_half.value + "=iK=" + latex_matrix(_j) + "$$\n"
        f"Eigenvalues of $J_{axis_choice_half.value}$: **{', '.join(f'{x:+.1f}' for x in _ev)}**\n\n"
        f"Largest difference between the rotation implemented by $U$ (at $\\alpha=0.7$) and the matrix of section 1: "
        f"{_err:.1e}.\n\n"
        "The three matrices $J_x$, $J_y$, $J_z$ are the Pauli matrices divided by 2."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.4 The effect of a $2\pi$ rotation

    With $m_s=\pm\tfrac12$, a rotation through $2\pi$ multiplies both amplitudes by $e^{\mp i\pi}=-1$. For integer
    $m_s$ it multiplies them by $1$. By section 5.1 the sign change alters no probability of the rotated state
    on its own; it can only be detected by interference with an unrotated copy.

    The slider runs from $0$ to $4\pi$. The panels show
    (1) the direction in which the spin points, which returns after $2\pi$ for both spins;
    (2, 3) the amplitudes themselves, drawn as arrows in the complex plane, which return after $2\pi$ for
    spin 1 but with reversed sign for spin 1/2; and (4) the probability of detection when a beam is split,
    one part is rotated through $\alpha$, and the parts are recombined. For spin 1/2 this probability repeats
    only every $4\pi$.
    """)
    return


@app.cell
def _(mo):
    theta_over_pi = mo.ui.slider(0, 4, step=0.02, value=0, label="rotation angle α / π", full_width=True)
    theta_over_pi
    return (theta_over_pi,)


@app.cell
def _(bloch_vector, interference_intensity, np, rotation, spin_matrices, theta_over_pi):
    theta = theta_over_pi.value * np.pi
    _z = (0, 0, 1)

    # Spin-1/2 pointing along +x: (|up> + |down>) / sqrt(2)
    psi_half0 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    # Spin-1 pointing along +x: (|+1> + sqrt(2)|0> + |-1>) / 2
    psi_one0 = np.array([1, np.sqrt(2), 1], dtype=complex) / 2

    psi_half = rotation(0.5, _z, theta) @ psi_half0
    psi_one = rotation(1, _z, theta) @ psi_one0

    # Spin direction in the xy-plane (unit length), for both spins
    dir_half = bloch_vector(psi_half)[:2]
    dir_one = np.array(
        [np.vdot(psi_one, _j @ psi_one).real for _j in spin_matrices(1)[:2]]
    )

    # Interference curves over the full range, plus the value at the current angle
    thetas = np.linspace(0, 4 * np.pi, 400)
    i_half = [interference_intensity(psi_half0, 0.5, _z, _t) for _t in thetas]
    i_one = [interference_intensity(psi_one0, 1, _z, _t) for _t in thetas]
    return (
        dir_half,
        dir_one,
        i_half,
        i_one,
        psi_half,
        psi_half0,
        psi_one,
        psi_one0,
        theta,
        thetas,
    )


@app.cell
def _(
    dir_half,
    dir_one,
    i_half,
    i_one,
    np,
    plt,
    psi_half,
    psi_half0,
    psi_one,
    psi_one0,
    theta,
    thetas,
):
    _fig, (_ax_dir, _ax_half, _ax_one, _ax_int) = plt.subplots(1, 4, figsize=(17, 4.2))
    _circle = np.linspace(0, 2 * np.pi, 200)

    def _unit_circle(ax, title):
        ax.plot(np.cos(_circle), np.sin(_circle), color="0.8", lw=1)
        ax.axhline(0, color="0.9", lw=0.8)
        ax.axvline(0, color="0.9", lw=0.8)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)

    # 1. Where the spin points (observable direction)
    _unit_circle(_ax_dir, "direction of the spin (xy-plane)")
    _ax_dir.annotate("", xy=dir_half, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.5))
    _ax_dir.annotate("", xy=dir_one * 0.8, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.5))
    _ax_dir.legend(
        handles=[
            plt.Line2D([], [], color="tab:blue", lw=2.5, label="spin 1/2"),
            plt.Line2D([], [], color="tab:orange", lw=2.5, label="spin 1"),
        ],
        loc="lower left",
        fontsize=8,
    )

    # 2. Spin-1/2 amplitudes in the complex plane
    _unit_circle(_ax_half, "spin 1/2: the two amplitudes")
    for _c, _label, _color in zip(psi_half, ["$m_s=+1/2$", "$m_s=-1/2$"], ["tab:blue", "tab:cyan"]):
        _tip = _c * np.sqrt(2)  # rescale so |c| = 1/sqrt(2) draws at radius 1
        _ax_half.annotate("", xy=(_tip.real, _tip.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5))
        _ax_half.text(_tip.real * 1.12, _tip.imag * 1.12, _label, color=_color, fontsize=8, ha="center", va="center")
    _overlap_half = np.vdot(psi_half0, psi_half)
    _ax_half.set_xlabel(f"overlap with initial state = {_overlap_half.real:+.2f}", fontsize=10)

    # 3. Spin-1 amplitudes
    _unit_circle(_ax_one, "spin 1: the three amplitudes")
    for _c, _label, _color in zip(psi_one, ["$m_s=+1$", "$m_s=0$", "$m_s=-1$"], ["tab:orange", "tab:red", "tab:brown"]):
        _tip = _c * 2  # |c| is 1/2 or 1/sqrt(2); a scale of 2 keeps everything inside the circle
        _ax_one.annotate("", xy=(_tip.real, _tip.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5))
        _ax_one.text(_tip.real * 1.15 + 0.05, _tip.imag * 1.15 + 0.05, _label, color=_color, fontsize=8)
    _ax_one.set_xlabel(f"overlap with initial state = {np.vdot(psi_one0, psi_one).real:+.2f}", fontsize=10)

    # 4. Interference: split the beam, rotate one part by theta, recombine
    _ax_int.plot(thetas / np.pi, i_half, color="tab:blue", label="spin 1/2")
    _ax_int.plot(thetas / np.pi, i_one, color="tab:orange", label="spin 1")
    _ax_int.axvline(theta / np.pi, color="k", lw=1, ls="--")
    _ax_int.set_xlabel("α / π")
    _ax_int.set_ylabel("detection probability")
    _ax_int.set_xticks([0, 1, 2, 3, 4])
    _ax_int.set_title("interference with an unrotated copy", fontsize=10)
    _ax_int.legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.5 Two matrices for every rotation

    By section 5.1, multiplying all amplitudes of a state by $-1$ changes no probability. A physical rotation
    therefore cannot distinguish a matrix $U$ from $-U$: **every rotation has two valid $2\times2$ matrices.**
    The slider below runs $\alpha$ from $0$ to $4\pi$. Left: the two numbers
    $(a,b)=(\cos\frac\alpha2,\sin\frac\alpha2)$ from which the electron's matrix
    $U_z=\mathrm{diag}(a-ib,\,a+ib)$ is built trace out a circle, and a full circuit takes $4\pi$. Right: the
    ordinary $3\times3$ rotation implemented by this matrix completes a circuit every $2\pi$.
    """)
    return


@app.cell
def _(mo):
    phi_over_pi = mo.ui.slider(0, 4, step=0.02, value=0, label="rotation angle α / π", full_width=True)
    phi_over_pi
    return (phi_over_pi,)


@app.cell
def _(latex_matrix, mo, np, phi_over_pi, plt, rotation, rotation_matrix_so3, su2_to_so3):
    _phi = phi_over_pi.value * np.pi
    _z = (0, 0, 1)
    _u = rotation(0.5, _z, _phi)  # the electron's 2x2 matrix
    _r = su2_to_so3(_u)  # the ordinary 3x3 rotation it implements
    assert np.allclose(_r, rotation_matrix_so3(_z, _phi))  # same rotation, two descriptions

    _fig, (_ax_su2, _ax_so3) = plt.subplots(1, 2, figsize=(9, 4.4))
    _circ = np.linspace(0, 2 * np.pi, 200)
    _trail = np.linspace(0, _phi, 200)

    _ax_su2.plot(np.cos(_circ), np.sin(_circ), color="0.85")
    _ax_su2.plot(np.cos(_trail / 2), np.sin(_trail / 2), color="tab:blue", lw=3)
    _ax_su2.plot([np.cos(_phi / 2)], [np.sin(_phi / 2)], "o", color="tab:blue", ms=10)
    _ax_su2.text(1.08, 0, "U=+1\nα=0, 4π", va="center", fontsize=8)
    _ax_su2.text(-1.08, 0, "U=−1\nα=2π", va="center", ha="right", fontsize=8)
    _ax_su2.set_xlim(-1.6, 1.6)
    _ax_su2.set_ylim(-1.3, 1.3)
    _ax_su2.set_aspect("equal")
    _ax_su2.set_xlabel("a = cos(α/2)")
    _ax_su2.set_ylabel("b = sin(α/2)")
    _ax_su2.set_title("electron matrix: one circuit takes 4π", fontsize=10)

    _ax_so3.plot(np.cos(_circ), np.sin(_circ), color="0.85")
    for _col, _color, _name in [(0, "tab:red", "x̂'"), (1, "tab:green", "ŷ'")]:
        _ax_so3.annotate(
            "", xy=(_r[0, _col], _r[1, _col]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=_color, lw=2.5),
        )
        _ax_so3.text(_r[0, _col] * 1.15, _r[1, _col] * 1.15, _name, color=_color, ha="center", va="center")
    _ax_so3.set_xlim(-1.3, 1.3)
    _ax_so3.set_ylim(-1.3, 1.3)
    _ax_so3.set_aspect("equal")
    _ax_so3.set_title("ordinary 3×3 rotation: one circuit takes 2π", fontsize=10)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "$$U = " + latex_matrix(_u) + r"\qquad R = " + latex_matrix(_r) + "$$\n"
            "At $\\alpha=2\\pi$, **$R$ is the identity while $U=-1$.** The matrices $U$ and $-U$ implement the "
            "same $R$, so the correspondence from $2\\times2$ matrices to $3\\times3$ rotations is two-to-one. "
            "(These two families of matrices are known as SU(2) and SO(3), and SU(2) is called the *double "
            "cover* of SO(3).)"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6 The belt trick: a picture of the $4\pi$ rotation

    Sections 5.4 and 5.5 show with formulas that a rotation through $2\pi$ reverses the sign of the spin-1/2
    matrix and that $4\pi$ is needed to restore it. A physical model shows the same fact as a shape.

    **The model.** A belt joins a fixed wall to a plate, and the plate is turned about the vertical axis. At each
    point along the belt the cross-section has an *orientation*, a rotation measured from the wall. At the wall
    the orientation is the identity, and at the plate it is the rotation through the angle $\alpha$ by which the
    plate has been turned. The belt is therefore a continuous path of orientations from the identity to the
    plate's orientation. The two ends do not fix the orientations in between, because twists can be added without
    changing the ends, so a choice has to be made for each $\alpha$. The belt shown is given by one explicit
    formula[^beltformula] that changes smoothly as the plate turns. Its centre line follows the direction of the
    belt's own axis, so as the belt curls the plate is carried around a loop. (This arrangement is known as the
    plate trick: a plate carried on the hand stays level while the arm turns beneath it.) The red and blue sides
    are the two faces of the belt, and the pointer on the plate shows the plate's orientation.

    [^beltformula]: The orientations are written as unit quaternions, which are the $2\times2$ matrices of
        section 5.3 in other notation: $U = w - i(x\sigma_x + y\sigma_y + z\sigma_z)$, where $\sigma_x,\sigma_y,\sigma_z$
        are twice the spin-1/2 $J$ matrices. The formula and its construction are in `dirac_belt.py`, and
        `test_dirac_belt.py` checks that the wall and plate conditions hold, that the belt changes continuously
        with $\alpha$, that it repeats after $4\pi$ but not after $2\pi$, that the belt at $2\pi$ is exactly one
        full twist about its own axis, and that the quaternions agree with the matrices of section 5.3.

    ### 6.1 Turning the plate

    The slider sets the angle $\alpha$ through which the plate has been turned, from $0$ to $4\pi$. The readout
    under the figure states whether the plate and the belt are back in the state they started in.
    """)
    return


@app.cell
def _(mo):
    belt_phi_over_pi = mo.ui.slider(0, 4, step=0.02, value=0, label="plate rotation α / π", full_width=True)
    belt_phi_over_pi
    return (belt_phi_over_pi,)


@app.cell
def _(belt_figure, belt_frames, belt_phi_over_pi, mo, np, plt):
    _phi = belt_phi_over_pi.value * np.pi
    _fig = belt_figure(_phi)
    _plate_back = bool(np.isclose(np.exp(1j * _phi), 1, atol=1e-6))
    _belt_flat = bool(np.allclose(belt_frames(_phi, 60), np.eye(3), atol=1e-6))
    _out = mo.vstack([
        _fig,
        mo.md(
            f"Plate in its starting orientation: **{'yes' if _plate_back else 'no'}**. "
            f"Belt flat, as at the start: **{'yes' if _belt_flat else 'no'}**."
        ),
    ])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the figure shows.**

    - At $\alpha=2\pi$ the plate is back in its starting orientation, but the belt is not: it has one full
      twist. The two record different things. The orientation of the plate says only that it faces as it did at
      the start. The belt also records that a full turn has been made.
    - Turning the plate further in the same direction does not add a second twist. The belt untwists, and at
      $\alpha=4\pi$ the plate and the belt are both as they started.
    - No smooth way of turning the plate leaves the belt flat at both $\alpha=0$ and $\alpha=2\pi$, so the
      twist at $2\pi$ cannot be avoided by a cleverer choice of belt. For $4\pi$ there is such a way, and it is
      the one shown.

    **The same fact as the sign of the spinor.** Each orientation along the belt can be described by a
    $2\times2$ matrix $U$ of section 5.3, and two matrices $\pm U$ describe each orientation (section 5.5).
    Following the belt from the wall, where $U=1$, fixes a continuous choice of sign. At the plate this choice is
    the electron's matrix $U_z(\alpha)$, which is $-1$ at $\alpha=2\pi$ and $+1$ at $\alpha=4\pi$. A belt with
    one twist is the picture of the sign $-1$, and a belt that has untwisted is the picture of the sign $+1$.

    ### 6.2 Why the twist at $2\pi$ cannot be removed and the twist at $4\pi$ can

    Every orientation is a rotation through some angle between $0$ and $\pi$ about some axis. (A rotation through
    more than $\pi$ about an axis is the same as a rotation through less than $\pi$ about the opposite axis.)
    Each orientation can therefore be drawn as a point in a ball of radius $\pi$: the direction from the centre
    is the axis and the distance from the centre is the angle. The centre is the identity. A rotation through
    $\pi$ about an axis and a rotation through $\pi$ about the opposite axis are the same rotation, so
    **opposite points on the surface of the ball are the same orientation**, and a path that leaves the ball at
    one point continues from the opposite point.

    Turning the plate about the vertical axis moves its orientation straight up from the centre. At
    $\alpha=\pi$ it reaches the surface, continues from the opposite point at the bottom, and is back at the
    centre at $\alpha=2\pi$. This is a closed path that crosses the surface once (left figure). Turning through
    $4\pi$ repeats the path and crosses the surface twice (right figure; the second turn retraces the first, so
    at the start only one line is visible).

    A closed path can be deformed continuously while its start and end stay at the centre. The belt is flat
    exactly when the path has shrunk to the single point at the centre. A deformation can create or remove
    crossings of the surface only in pairs, because two crossings can merge and cancel, whereas a lone crossing
    cannot disappear. Hence the path with one crossing can never be shrunk to the centre, and the path with two
    crossings can. The slider shrinks the $4\pi$ path: the two crossings approach each other and cancel, and the
    path then contracts to a point.
    """)
    return


@app.cell
def _(mo):
    shrink_fraction = mo.ui.slider(0, 1, step=0.01, value=0, label="shrinking of the 4π path", full_width=True)
    shrink_fraction
    return (shrink_fraction,)


@app.cell
def _(double_turn_ball_figure, mo, np, plt, shrink_fraction, single_turn_ball_figure):
    _kappa = (np.pi / 2) * (1 - shrink_fraction.value)
    _left = single_turn_ball_figure()
    _right = double_turn_ball_figure(_kappa)
    _out = mo.hstack([_left, _right], justify="center")
    plt.close(_left)
    plt.close(_right)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The belt and the ball are two views of one fact. The path traced by the plate's orientation is a closed
    path of rotations. The belt at a given $\alpha$ is a path from the identity to the plate's orientation, and as
    the plate turns through $4\pi$ these belts sweep out a surface bounded by the plate's path. The shrinking
    paths in the ball are the same surface cut the other way. Such a surface exists for $4\pi$ and cannot exist for
    $2\pi$. In the language of section 5.5, the $4\pi$ path lifts to a closed path of matrices $U$ (it ends at
    $U=+1$) and can be shrunk, whereas the $2\pi$ path lifts to a path from $U=+1$ to $U=-1$, which is not closed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Summary of Part I.** A system with two levels has $m_s=\pm\tfrac12$ and the $2\times2$ rotation matrices above. A
    rotation through $2\pi$ multiplies its amplitudes by $-1$, which changes no probability of the state itself
    but can be detected by interference, and the state returns only after $4\pi$. Ordinary vectors return after
    $2\pi$, and the two descriptions are related two-to-one: each ordinary rotation corresponds to a pair of
    matrices $\pm U$. The belt trick shows the same fact as a shape: a plate turned through $2\pi$ leaves a belt
    with one twist, and the twist disappears only after $4\pi$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Sources.** The treatment of spin-1/2 rotations and of the symmetry argument for conservation laws follows
    Chapters 6 and 17 of R. P. Feynman, R. B. Leighton and M. Sands, *The Feynman Lectures on Physics*, Vol. III: Quantum Mechanics (Caltech online edition, edited by M. A. Gottlieb and R. Pfeiffer, 2013), <https://www.feynmanlectures.caltech.edu/III_toc.html>. The explicit belt in Section 6 is a construction written for this notebook
    (`dirac_belt.py`).
    """)
    return

if __name__ == "__main__":
    app.run()
