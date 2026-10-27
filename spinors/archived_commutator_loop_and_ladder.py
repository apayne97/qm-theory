import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from spinors import (
        commutator_loop,
        commutator_norms,
        expectation_vs_time,
        raising_operator,
        rotation,
        rotation_axis_angle,
        rotation_matrix_so3,
        spin_matrices,
    )

    return (
        commutator_loop,
        commutator_norms,
        expectation_vs_time,
        mo,
        np,
        plt,
        raising_operator,
        rotation,
        rotation_axis_angle,
        rotation_matrix_so3,
        spin_matrices,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Archived: the technical version

    Material cut from `spinor_rotation.py` to keep that notebook readable without advanced linear algebra,
    group theory or quantum mechanics. It is all correct and tested; it just assumes more background.

    Contents:
    1. The commutator of small rotations ($R_xR_yR_x^{-1}R_y^{-1}\approx R_z(\epsilon^2)$) and why it is the same for every spin.
    2. The allowed values of $m$ (phase argument for the spacing between neighbouring levels, with an interactive check).
    3. The ladder-operator derivation that $m$ comes in integer steps and $2j$ is an integer.
    4. Noether's theorem for angular momentum, in the form $[H, J]=0 \Rightarrow dJ/dt=0$, with commutator norms.
    5. Technical remarks on why the half-integer spins are allowed (projective representations).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The commutator of two small rotations is a third rotation

    Do $x$ by $\epsilon$, then $y$ by $\epsilon$, then undo $x$, then undo $y$. If rotations commuted you would
    be back where you started. Instead you land on a small rotation **about $z$ by $\approx\epsilon^2$**:

    $$R_x(\epsilon)\,R_y(\epsilon)\,R_x(-\epsilon)\,R_y(-\epsilon) \;\approx\; R_z(\epsilon^2).$$

    This is the group version of $[J_x, J_y] = iJ_z$. It is exactly why Feynman can build $R_x$ from $R_y$ and
    $R_z$ (and why $J_z$ is traceless, and why $m$ must come in integer steps): the rotations about
    different axes are tied together, which is what pins down the allowed spins.
    """)
    return


@app.cell
def _(mo):
    eps_deg = mo.ui.slider(1, 30, step=1, value=10, label="ε (degrees)", full_width=True)
    eps_deg
    return (eps_deg,)


@app.cell
def _(eps_deg, mo, np, rotation_axis_angle, rotation_matrix_so3):
    _e = np.radians(eps_deg.value)
    _x, _y = (1, 0, 0), (0, 1, 0)
    _loop = (
        rotation_matrix_so3(_x, _e) @ rotation_matrix_so3(_y, _e)
        @ rotation_matrix_so3(_x, -_e) @ rotation_matrix_so3(_y, -_e)
    )
    _axis, _angle = rotation_axis_angle(_loop)
    mo.md(
        f"Net rotation: **{np.degrees(_angle):.3f}°** about ({_axis[0]:+.2f}, {_axis[1]:+.2f}, {_axis[2]:+.2f}). "
        f"Compare ε² = {np.degrees(_e ** 2):.3f}° (ε² in radians, shown in degrees)."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The same loop works for every spin

    The commutator loop does not care which representation you use. Build it from spin-$j$ rotation matrices
    $e^{-i\epsilon J_x}$, $e^{-i\epsilon J_y}$ and compare with $e^{-i\epsilon^2 J_z}$. The error shrinks like
    $\epsilon^3$ for $j = \tfrac12, 1, \tfrac32, 2$ alike. Also, for small $\epsilon$ the loop is close to $+1$,
    including for half-integer $j$: the sign problem only appears for *large* paths like a full $2\pi$ turn.

    So what is representation-independent is the **non-commuting structure** (the Lie algebra
    $[J_x,J_y]=iJ_z$ and its cyclic partners). The spin number $j$ labels which representation of that
    structure you have.
    """)
    return


@app.cell
def _(commutator_loop, np, plt, rotation):
    _eps = np.logspace(-2.5, -0.5, 25)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _j in [0.5, 1, 1.5, 2]:
        _err = [np.linalg.norm(commutator_loop(_j, _e) - rotation(_j, (0, 0, 1), _e**2)) for _e in _eps]
        _ax.loglog(_eps, _err, "o-", ms=3, label=f"j = {_j:g}")
    _ax.loglog(_eps, _eps**3, "k--", label="ε³")
    _ax.set_xlabel("ε")
    _ax.set_ylabel("‖ loop − R_z(ε²) ‖")
    _ax.set_title("same non-commuting structure for every spin", fontsize=10)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where the quantisation comes from

    Non-commutation gives $J_z = -i[J_x, J_y]$ and, with $J_\pm = J_x \pm iJ_y$,

    $$[J_z, J_\pm] = \pm J_\pm.$$

    So if $J_z|m\rangle = m|m\rangle$, then $J_+|m\rangle$ has $J_z$-eigenvalue exactly $m+1$ and $J_-$ gives $m-1$:
    the $m$ values come in **steps of 1**. In a finite number of states the ladder must stop at a top state
    ($J_+|m_{\max}\rangle = 0$) and a bottom state ($J_-|m_{\min}\rangle = 0$). Requiring that
    $\|J_\pm|m\rangle\|^2 = j(j+1) - m(m\pm1)$ vanish at the ends forces $m_{\min} = -m_{\max} \equiv -j$.
    Taking $N$ steps of 1 from $-j$ to $+j$ gives $2j = N-1$, an integer.

    That is the derivation behind Part 2's claim "adjacent $m$ differ by 1, centred on zero".
    Pick a spin below and look at $J_+$: every nonzero entry connects $m$ to $m+1$ and nothing else.
    """)
    return


@app.cell
def _(mo):
    ladder_j = mo.ui.dropdown(
        options={"1/2": 0.5, "1": 1.0, "3/2": 1.5, "2": 2.0}, value="3/2", label="spin j"
    )
    ladder_j
    return (ladder_j,)


@app.cell
def _(ladder_j, mo, np, raising_operator, spin_matrices):
    _j = ladder_j.value
    _jz = spin_matrices(_j)[2]
    _jp = raising_operator(_j)
    _m = np.diag(_jz).real
    _check = np.linalg.norm(_jz @ _jp - _jp @ _jz - _jp)
    mo.md(
        f"m values (row/column order): `{', '.join(f'{_x:+g}' for _x in _m)}`\n\n"
        "**J₊** (nonzero entries sit one step above the diagonal: they take the column's m to the row's m + 1)\n"
        f"```\n{np.array2string(np.round(_jp.real, 3) + 0, precision=3)}\n```\n"
        f"‖[J_z, J₊] − J₊‖ = {_check:.1e}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Part 3: Noether's theorem and conserved angular momentum

    The generator in $U(\hat n,\theta) = e^{-i\theta\,\hat n\cdot\mathbf J}$ is $\mathbf J$ itself. If the
    Hamiltonian is rotationally invariant, $U H U^\dagger = H$ for every rotation, and expanding to first
    order in $\theta$ gives $[H, \hat n\cdot\mathbf J] = 0$. Then the Heisenberg equation

    $$\frac{d\langle \mathbf J\rangle}{dt} = i\langle [H, \mathbf J]\rangle = 0.$$

    So the conserved quantity is *literally the generator of the symmetry*: the same $m$ that labels the
    phase $e^{im\phi}$ in Feynman's argument is the eigenvalue of the conserved angular momentum. In quantum
    mechanics Noether's theorem is almost a one-liner. (The classical version, symmetry of the Lagrangian
    $\Rightarrow$ conserved charge, needs the Euler-Lagrange equations to get there.)

    **Test it.** A magnetic field breaks rotation symmetry, but not entirely: $H = \omega\,\hat n\cdot\mathbf J$
    is still invariant under rotations about $\hat n$. Prediction: only $\hat n\cdot\mathbf J$ is conserved.
    The other components precess. (And note the time evolution $e^{-iHt}$ is itself a rotation about
    $\hat n$ by $\omega t$.)
    """)
    return


@app.cell
def _(mo):
    beta_over_pi = mo.ui.slider(0, 1, step=0.05, value=0.25, label="field direction β / π (angle from z)", full_width=True)
    beta_over_pi
    return (beta_over_pi,)


@app.cell
def _(
    beta_over_pi,
    commutator_norms,
    expectation_vs_time,
    np,
    plt,
    spin_matrices,
):
    _beta = beta_over_pi.value * np.pi
    _n = np.array([np.sin(_beta), 0.0, np.cos(_beta)])
    _jn = np.tensordot(_n, spin_matrices(0.5), axes=1)  # n.J
    _h = 2.0 * _jn  # H = omega n.J with omega = 2

    # an arbitrary starting direction, so nothing is accidentally an eigenstate
    _t0, _p0 = np.arccos(1 / np.sqrt(3)), np.pi / 4
    _psi0 = np.array([np.cos(_t0 / 2), np.sin(_t0 / 2) * np.exp(1j * _p0)])

    _ts = np.linspace(0, 2 * np.pi, 300)
    _jexp = expectation_vs_time(_h, _psi0, 0.5, _ts)
    _norms = np.append(commutator_norms(_h, 0.5), np.linalg.norm(_h @ _jn - _jn @ _h))

    _fig, (_ax_t, _ax_c) = plt.subplots(1, 2, figsize=(10, 3.8))
    for _k, _label in enumerate(["⟨Jx⟩", "⟨Jy⟩", "⟨Jz⟩"]):
        _ax_t.plot(_ts, _jexp[:, _k], label=_label)
    _ax_t.plot(_ts, _jexp @ _n, "k--", lw=2, label="⟨n·J⟩")
    _ax_t.set_xlabel("time")
    _ax_t.legend(fontsize=8, ncol=4, loc="lower center")
    _ax_t.set_title("only the component along the field is constant", fontsize=10)

    _ax_c.bar(["Jx", "Jy", "Jz", "n·J"], _norms, color=["tab:blue", "tab:orange", "tab:green", "k"])
    _ax_c.set_ylabel("‖[H, J]‖")
    _ax_c.set_title("commutator with H  (0 = conserved)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the sign is allowed

    The amplitudes $C_+, C_-$ are only defined up to a common phase, so $-1$ changes no probability.
    Rotation symmetry therefore only demands that $R(\text{rot}_2)R(\text{rot}_1) = \pm R(\text{rot}_2\circ\text{rot}_1)$.
    Mathematically: the group acts on *rays*, so we need **projective** representations of SO(3). Because
    SO(3) is not simply connected ($\pi_1 = \mathbb{Z}_2$: a $2\pi$ rotation is a loop that cannot be shrunk
    to a point, but $4\pi$ can), its projective representations are exactly the ordinary representations of
    its simply connected double cover, **SU(2)**. That is where the half-integer spins come from, and it is
    the loop the Dirac belt trick will show.

    ## Fixing $m$: Feynman's 180° argument, as a picture

    Rotate the apparatus by $\phi$ about $z$. The two amplitudes get $e^{-im\phi}$ and $e^{+im\phi}$, so the
    *relative* phase is $e^{2im\phi}$, and the relative phase is the azimuth of the spin direction.
    A spin that started along $+x$ must point along the rotated axis, i.e. at azimuth $\phi$.
    So the blue curve must lie on the dashed line.

    Try $m = 1$ first, since "rotating by $2\pi$ should do nothing" suggests it. At $\phi = \pi$ both
    amplitudes get $e^{\mp i\pi} = -1$, which is an unobservable common sign, so the state looks unrotated
    and $+x$ never becomes $-x$. That contradicts geometry, so $m = 1$ is out.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # The allowed values of $m$ (cut from the main notebook)

    Section numbers below refer to `spinor_rotation.py`. The ladder-operator derivation further down makes the
    neighbouring-level argument rigorous.

    A rotation about $z$ cannot change the probabilities of the outcomes along $z$, so it acts on each
    amplitude only through a phase. Phases add when rotation angles add, so the phase of the level with
    value $m$ must be $e^{-im\theta}$, as in section 3. The values of $m$, however, are no longer taken
    from the $3\times3$ matrices; they are to be determined.

    The direction of the spin (its $x$ and $y$ components) is determined by the products of amplitudes of
    **neighbouring levels**, and two levels with values $m_1$ and $m_2$ have a relative phase
    $e^{i(m_1-m_2)\theta}$. For two levels with $m=\pm m_0$ this is the phase of $a_+^*a_-$, namely
    $e^{i\,2m_0\theta}$. This relative phase is the angle of the spin direction, which must equal the rotation
    angle $\theta$. The spacing between neighbouring levels must therefore be **exactly 1**.

    The levels must also be symmetric about zero. Rotating the apparatus through $180^\circ$ about $y$
    interchanges "up" and "down", so the set of $m$ values must be unchanged by $m\to-m$.

    For $N$ levels with spacing 1, symmetric about zero, the largest value is $j=(N-1)/2$. In the figure below,
    choose the number of levels and a trial spacing $s$. The blue line is the angle through which the spin
    direction actually turns; the dashed line is the angle required by geometry. **Only $s=1$ agrees.** The
    initial setting, $N=2$ with $s=2$ (that is, $m=\pm1$), is the natural first guess for two levels and it fails.
    """)
    return


@app.cell
def _(mo):
    n_levels = mo.ui.slider(2, 5, step=1, value=2, label="number of levels N", full_width=True)
    step_size = mo.ui.slider(0.25, 2.5, step=0.25, value=2.0, label="trial spacing s between neighbouring levels", full_width=True)
    mo.vstack([n_levels, step_size])
    return n_levels, step_size


@app.cell
def _(n_levels, np, plt, step_size):
    _n, _s = n_levels.value, step_size.value
    # N levels centred on zero, spaced by s, highest first
    _ms = _s * (np.arange(_n)[::-1] - (_n - 1) / 2)
    _thetas = np.linspace(0, 2 * np.pi, 300)
    # equal mix of all levels; rotating by theta multiplies level m by exp(-i m theta)
    _c = np.exp(-1j * np.outer(_thetas, _ms))
    # product of neighbouring levels has phase exp(i s theta): that is the spin-direction angle
    _angle = np.unwrap(np.angle((np.conj(_c[:, :-1]) * _c[:, 1:]).mean(axis=1)))

    _fig, _ax = plt.subplots(figsize=(5.5, 4))
    _ax.plot(_thetas / np.pi, _thetas / np.pi, "k--", label="required by geometry: angle = θ")
    _ax.plot(_thetas / np.pi, _angle / np.pi, color="tab:blue", lw=2.5, label=f"spin direction (spacing s = {_s:g})")
    _ax.set_xlabel("rotation θ / π")
    _ax.set_ylabel("angle of spin direction / π")
    _ax.legend(fontsize=8)
    _ax.set_title(
        "levels m = " + ", ".join(f"{_x:+g}" for _x in _ms) + ("   ✓" if np.isclose(_s, 1) else "   ✗"),
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    With spacing 1 and symmetry about zero, the number of levels determines the spin: $j=(N-1)/2$. Three
    levels reproduce the spin-1 values of section 3, and **two levels give $m=\pm\tfrac12$, which is the electron.**
    A rotation through $2\pi$ multiplies each level by $e^{-im2\pi}$:
    """)
    return


@app.cell
def _(mo, np, rotation, spin_matrices):
    _rows = ["| N levels | spin j = (N−1)/2 | m values | effect of a 2π rotation |", "|---|---|---|---|"]
    for _n in range(1, 7):
        _j = (_n - 1) / 2
        _ms = ", ".join(f"{_x:+g}" for _x in np.diag(spin_matrices(_j)[2]).real)
        _sign = rotation(_j, (0, 0, 1), 2 * np.pi)[0, 0].real
        _rows.append(f"| {_n} | {_j:g} | {_ms} | {'unchanged' if _sign > 0 else 'amplitudes change sign'} |")
    mo.md("\n".join(_rows) + "\n\nAn even number of levels (the electron has two) always produces a sign change.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## The matrices for spin 1/2 from the commutation relations

    (Cut from section 5.4 of the main notebook, where $U_x$ and $U_y$ are obtained by turning the axis.)
    The matrices $J_x,J_y,J_z$ obtained from the ordinary $3\times3$ rotations satisfy

    $$[J_x,J_y]=iJ_z,\qquad [J_y,J_z]=iJ_x,\qquad [J_z,J_x]=iJ_y,\qquad [A,B]=AB-BA.$$

    These relations express the fact that rotations about different axes do not commute; the cell below checks them
    numerically. Requiring that $2\times2$ matrices satisfy the same relations, with $J_z=\mathrm{diag}(\tfrac12,-\tfrac12)$
    (from $m_s=\pm\tfrac12$), determines them.

    Write $J_x=\begin{pmatrix}p&b\\c&d\end{pmatrix}$ with $J_x$ Hermitian, so that $p,d$ are real and $c=b^*$. Since $J_z$
    is diagonal, $[J_z,J_x]=\begin{pmatrix}0&b\\-b^*&0\end{pmatrix}$, and $[J_z,J_x]=iJ_y$ gives

    $$J_y=\begin{pmatrix}0&-ib\\ib^*&0\end{pmatrix}.$$

    Then $[J_y,J_z]=\begin{pmatrix}0&ib\\ib^*&0\end{pmatrix}$ must equal $iJ_x=\begin{pmatrix}ip&ib\\ib^*&id\end{pmatrix}$, so
    $p=d=0$ and $J_x=\begin{pmatrix}0&b\\b^*&0\end{pmatrix}$. Finally

    $$[J_x,J_y]=\begin{pmatrix}2i|b|^2&0\\0&-2i|b|^2\end{pmatrix}=iJ_z=\begin{pmatrix}\tfrac i2&0\\0&-\tfrac i2\end{pmatrix}
    \quad\Rightarrow\quad |b|=\tfrac12.$$

    The phase of $b$ can be removed by redefining the phase of the second basis state, which leaves $J_z$ unchanged.
    With $b=\tfrac12$, $J_x=\tfrac12\sigma_x$ and $J_y=\tfrac12\sigma_y$, the Pauli matrices divided by 2, and
    $U=e^{-i\alpha J}$ gives the matrices $U_x$, $U_y$ of the main notebook.
    """)
    return


@app.cell
def _(mo, np, rotation_matrix_so3, spin_matrices):
    _h = 1e-5

    def _j3(axis):
        return 1j * (rotation_matrix_so3(axis, _h) - rotation_matrix_so3(axis, -_h)) / (2 * _h)

    def _cyclic_error(jx, jy, jz):
        return max(
            np.abs(jx @ jy - jy @ jx - 1j * jz).max(),
            np.abs(jy @ jz - jz @ jy - 1j * jx).max(),
            np.abs(jz @ jx - jx @ jz - 1j * jy).max(),
        )

    _e3 = _cyclic_error(_j3((1, 0, 0)), _j3((0, 1, 0)), _j3((0, 0, 1)))
    _e2 = _cyclic_error(*spin_matrices(0.5))
    _pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]) / 2
    _ep = np.abs(np.array(spin_matrices(0.5)) - _pauli).max()
    mo.md(
        f"Largest violation of the commutation relations by the $3\\times3$ matrices: {_e3:.1e}; by the $2\\times2$ "
        f"matrices: {_e2:.1e}. Largest difference between these $2\\times2$ matrices and the Pauli matrices "
        f"divided by 2: {_ep:.1e}."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Technical remarks (cut from the main notebook)

    - **Projective representations.** Quantum states are rays, so symmetry only requires
      $U(\text{rot}_2)U(\text{rot}_1)=\pm U(\text{rot}_2\circ\text{rot}_1)$. SO(3) is not simply connected
      ($\pi_1=\mathbb Z_2$: a $2\pi$ rotation is a loop that cannot be shrunk to a point, but $4\pi$ can), so its
      projective representations are exactly the ordinary representations of its simply connected double cover
      SU(2). That is where the half-integer spins come from; the Dirac belt trick shows the loop.
    - **Why only half-integers.** In 2D the rotation group is abelian and any real spin is allowed (anyons).
      Quantisation to $j\in\{0,\tfrac12,1,\dots\}$ needs non-commuting rotations in 3D.
    - **Relativistic version.** The Lorentz group's double cover $SL(2,\mathbb C)$ is where Weyl and Dirac spinors live.
    - **Spin-statistics.** Half-integer spin $\Rightarrow$ fermion is the Pauli theorem (1940), proved in relativistic QFT
      from Lorentz invariance, locality and positive energy. Not derivable from rotations alone.
    """)
    return


if __name__ == "__main__":
    app.run()
