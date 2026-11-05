import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="The Kohn-Sham Matrix")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from dft_data import load
    from ks_plots import (
        levels_figure,
        matrices_figure,
        pieces_figure,
        scf_loop_figure,
        short_label,
        solve_generalized,
    )

    ks = load("ks")
    S = np.array(ks["S"])
    F = np.array(ks["F"])
    C = np.array(ks["C"])
    D = np.array(ks["D"])
    eps = np.array(ks["eps_Ha"])
    pieces = {k: np.array(v) for k, v in ks["pieces"].items()}
    n_occ = ks["n_occ"]
    labels = [short_label(x) for x in ks["ao_labels"]]
    return (mo, np, ks, S, F, C, D, eps, pieces, n_occ, labels, levels_figure, matrices_figure, pieces_figure, scf_loop_figure, solve_generalized)


@app.cell
def _(mo):
    mo.md(r"""
    # The Kohn-Sham Matrix $F$ and the Equation $F C = S C \varepsilon$

    A DFT program solves the equation $F C = S C \varepsilon$ in every cycle of its self-consistent-field (SCF) loop.
    This notebook explains where that equation comes from, what each symbol is, and what the matrix $F$ is built from.
    It uses one real calculation as the example: water at ωB97M-V/def2-TZVPD, the same calculation as in
    `reading_a_dft_label.py` (58 basis functions, 5 occupied orbitals, total energy $-76.4334$ Ha).

    **Contents**

    1. Where the equation comes from
    2. The symbols
    3. A two-function example, solved by hand
    4. What $F$ is made of
    5. Why the loop must be repeated, and the loop itself
    6. Numerical checks on water
    7. What is verified and what is not

    The numbers come from `compute_ks_matrix.py` (PySCF 2.14), which writes `data_ks.json`; this notebook only reads
    that file. Energies are in hartree (Ha) unless stated. A hartree is the atomic unit of energy, 27.2114 eV.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Where the equation comes from

    **The Kohn-Sham equation for the orbitals.** In Kohn-Sham DFT each electron is described by an orbital
    $\psi_i(\mathbf r)$, a function of the position $\mathbf r$, that satisfies

    $$
    \hat f\,\psi_i(\mathbf r) = \varepsilon_i\,\psi_i(\mathbf r),
    \qquad
    \hat f = -\tfrac12\nabla^2 + v_{\text{eff}}(\mathbf r).
    $$

    Here $\hat f$ is the *Kohn-Sham operator*: a rule that takes a function and returns another function.
    The symbol $\nabla^2$ means the sum of the second derivatives with respect to $x$, $y$ and $z$, so
    $-\tfrac12\nabla^2$ is the kinetic-energy operator. The function $v_{\text{eff}}(\mathbf r)$ is the potential
    energy of one electron at $\mathbf r$ in the field of the nuclei and of all the other electrons. The number
    $\varepsilon_i$ is the energy of orbital $i$. This equation says: applying $\hat f$ to the orbital gives back the
    same orbital multiplied by a number.

    **Writing the orbitals with a basis.** A computer cannot store a function of $\mathbf r$ exactly. Instead, each
    orbital is written as a sum of $N$ fixed, known *basis functions* $\phi_\mu(\mathbf r)$ (Gaussian-type functions
    centred on the atoms), with unknown numbers $C_{\mu i}$ as the weights:

    $$
    \psi_i(\mathbf r) = \sum_{\mu=1}^{N} C_{\mu i}\,\phi_\mu(\mathbf r).
    $$

    Now only the numbers $C_{\mu i}$ have to be found. The steps from the operator equation to a matrix equation are:

    - **(a) Insert the sum.** Put the sum into $\hat f\psi_i=\varepsilon_i\psi_i$. The operator acts on each
      $\phi_\mu$ separately, because the weights $C_{\mu i}$ are just numbers:
      $\sum_\mu C_{\mu i}\,\hat f\phi_\mu(\mathbf r)=\varepsilon_i\sum_\mu C_{\mu i}\,\phi_\mu(\mathbf r)$.
    - **(b) Turn a function equation into number equations.** The two sides are functions of $\mathbf r$. Multiply both
      sides by one basis function $\phi_\nu(\mathbf r)$ and integrate over all of space. This gives one number per side
      for each choice of $\nu$, and there are $N$ choices:
      $\sum_\mu C_{\mu i}\int\phi_\nu\,\hat f\,\phi_\mu\,d\mathbf r=\varepsilon_i\sum_\mu C_{\mu i}\int\phi_\nu\,\phi_\mu\,d\mathbf r$.
    - **(c) Name the two integrals.**

    $$
    F_{\nu\mu}=\int\phi_\nu(\mathbf r)\,\hat f\,\phi_\mu(\mathbf r)\,d\mathbf r,
    \qquad
    S_{\nu\mu}=\int\phi_\nu(\mathbf r)\,\phi_\mu(\mathbf r)\,d\mathbf r .
    $$

    - **(d) Recognise matrix multiplication.** For two matrices, $(AB)_{\nu i}=\sum_\mu A_{\nu\mu}B_{\mu i}$: the entry in
      row $\nu$ and column $i$ of the product is the sum over $\mu$ of row $\nu$ of $A$ times column $i$ of $B$. Step (b)
      therefore reads $(FC)_{\nu i}=\varepsilon_i\,(SC)_{\nu i}$. Collecting all $i$ at once, with the energies placed
      on the diagonal of a matrix $\varepsilon$, gives

    $$
    \boxed{\,F\,C = S\,C\,\varepsilon\,}
    $$

    Column $i$ of $C$ holds the weights of orbital $i$, and $\varepsilon$ is a diagonal matrix, so multiplying $C$ by it from the right multiplies column $i$ by $\varepsilon_i$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The symbols

    | Symbol | Meaning | Size for water |
    |---|---|---|
    | $\phi_\mu$ | basis function number $\mu$ ($\mu=1,\dots,N$) | $N=58$ |
    | $F$ | Kohn-Sham matrix, $F_{\mu\nu}=\int\phi_\mu\hat f\phi_\nu\,d\mathbf r$: the operator $\hat f$ written in the basis | $58\times58$ |
    | $S$ | overlap matrix, $S_{\mu\nu}=\int\phi_\mu\phi_\nu\,d\mathbf r$; equals 1 on the diagonal because each basis function is normalised | $58\times58$ |
    | $C$ | coefficient matrix; column $i$ holds the weights $C_{\mu i}$ of orbital $\psi_i$ | $58\times58$ |
    | $\varepsilon$ | diagonal matrix of the orbital energies $\varepsilon_i$ | 58 numbers |
    | $P$ | density matrix of the occupied orbitals, $P_{\mu\nu}=\sum_{i}^{\text{occ}}C_{\mu i}C_{\nu i}$ | $58\times58$, built from 5 columns of $C$ |
    | $D$ | total density matrix $D=2P$ (the factor 2 counts spin up and spin down) | $58\times58$ |

    Two facts about these symbols are used below.

    - **The density is a sum over $D$.** The electron density is
      $\rho(\mathbf r)=2\sum_i^{\text{occ}}|\psi_i(\mathbf r)|^2=\sum_{\mu\nu}D_{\mu\nu}\,\phi_\mu(\mathbf r)\,\phi_\nu(\mathbf r)$.
      The matrix $D$ is therefore the density, in the language of the basis, and every term of $F$ that depends on
      the density depends on it through $D$.
    - **$S$ is not the identity.** The basis functions overlap, so $S_{\mu\nu}\neq0$ for $\mu\neq\nu$. The orbitals are
      orthonormal ($\int\psi_i\psi_j\,d\mathbf r=\delta_{ij}$), which in matrix form reads $C^{T}SC=1$.
      Here $C^T$ is the transpose of $C$ (rows and columns exchanged), $1$ is the identity matrix and $\delta_{ij}$ is 1 if
      $i=j$ and 0 otherwise.

    The 58 solutions of the equation are 58 orbitals: the 5 lowest are occupied by the 10 electrons of water
    (two electrons each) and the other 53 are empty (virtual) orbitals.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. A two-function example, solved by hand

    The equation $FC=SC\varepsilon$ is called a *generalized eigenproblem*. If $S$ were the identity matrix it would
    be an ordinary eigenproblem $FC=C\varepsilon$: find the vectors that $F$ only stretches, and the stretch factors.
    The overlap $S$ is what makes it generalized. The smallest example has two basis functions that overlap,
    and a symmetric $F$:

    $$
    F=\begin{pmatrix}a&b\\ b&a\end{pmatrix},\qquad
    S=\begin{pmatrix}1&s\\ s&1\end{pmatrix}.
    $$

    Think of two identical $1s$ functions on two atoms, with $a$ the energy of the function on its own, $b$ the
    interaction between the two and $s$ their overlap. Because the matrices are symmetric under swapping the two functions, the solutions are the
    in-phase and out-of-phase combinations, $c=(1,\,1)$ and $c=(1,\,-1)$ (up to a common factor).

    - **(a)** For $c=(1,\pm1)$: $Fc=(a\pm b)\,(1,\pm1)$, because the first entry is $a\pm b$ and the second is $b\pm a=\pm(a\pm b)$.
    - **(b)** For the same $c$: $Sc=(1\pm s)\,(1,\pm1)$, by the same argument.
    - **(c)** $Fc=\varepsilon\,Sc$ then requires $a\pm b=\varepsilon\,(1\pm s)$, so

    $$
    \varepsilon_\pm=\frac{a\pm b}{1\pm s}.
    $$

    The factor $1\pm s$ is the effect of the overlap. With $s=0$ the energies would be $a\pm b$.
    """)
    return


@app.cell
def _(mo, np, solve_generalized):
    _a, _b, _s = -0.5, -0.4, 0.6
    _F = np.array([[_a, _b], [_b, _a]])
    _S = np.array([[1.0, _s], [_s, 1.0]])
    _eps, _C = solve_generalized(_F, _S)
    _by_hand = sorted([(_a + _b) / (1 + _s), (_a - _b) / (1 - _s)])
    _resid = np.abs(_F @ _C - _S @ _C * _eps).max()
    mo.md(
        f"With $a={_a}$, $b={_b}$, $s={_s}$ (arbitrary numbers): the formula gives "
        f"$\\varepsilon_+=({_a}{_b:+})/(1+{_s})={_by_hand[0]:.4f}$ and "
        f"$\\varepsilon_-=({_a}{-_b:+})/(1-{_s})={_by_hand[1]:.4f}$. "
        f"A numerical solver (the recipe below) returns {_eps[0]:.4f} and {_eps[1]:.4f}, and "
        f"$\\max|FC-SC\\varepsilon|={_resid:.1e}$. "
        f"The in-phase combination is lower. Without the overlap the energies would be $a+b={_a + _b:.1f}$ and $a-b={_a - _b:.1f}$; "
        "the factors $1\\pm s$ move them to the values above."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How a program solves it for 58 functions.** There is no need for the symmetry used above. The standard route
    turns the generalized problem into an ordinary one, in four steps (all of them are in `ks_plots.solve_generalized`):

    - **(a)** Diagonalise the overlap matrix: $S=U\,\mathrm{diag}(s_k)\,U^T$, where the $s_k$ are the eigenvalues of $S$ (all
      positive) and the columns of $U$ the eigenvectors.
    - **(b)** Form $X=S^{-1/2}=U\,\mathrm{diag}(s_k^{-1/2})\,U^T$. By construction $XSX=1$.
    - **(c)** Form $F'=XFX$, which is symmetric, and solve the *ordinary* problem $F'C'=C'\varepsilon$.
    - **(d)** Transform back: $C=XC'$. Check: $FC=F X C'=X^{-1}F'C'=X^{-1}C'\varepsilon=SXC'\varepsilon=SC\varepsilon$, using
      $X^{-1}=SX$ (which follows from $XSX=1$ after multiplying by $X^{-1}$ from the left).

    The same $\varepsilon$ come out, and $C^TSC=1$ holds automatically. Section 6 repeats this for water.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. What $F$ is made of

    The operator $\hat f=-\tfrac12\nabla^2+v_{\text{eff}}$ has a kinetic part and a potential part, and the potential
    part splits further into the nuclear attraction, the electron repulsion and the exchange and correlation. Taking
    the integral $\int\phi_\mu\,(\cdot)\,\phi_\nu\,d\mathbf r$ of each part gives one matrix per term. For a closed-shell
    molecule with a hybrid, range-separated functional such as ωB97M-V:

    $$
    F \;=\; T \;+\; V \;+\; J[D] \;+\; V^{\text{xc}}[D] \;+\; V^{\text{nlc}}[D] \;-\; \tfrac12\,K^{\text{eff}}[D].
    $$

    The square brackets mark dependence on the density matrix $D$. The terms are:

    **$T$, kinetic energy.** $T_{\mu\nu}=\int\phi_\mu\,(-\tfrac12\nabla^2)\,\phi_\nu\,d\mathbf r$. It depends only on the
    basis functions.

    **$V$, nuclear attraction.** $V_{\mu\nu}=-\sum_A Z_A\int\phi_\mu\,\phi_\nu\,\lvert\mathbf r-\mathbf R_A\rvert^{-1}\,d\mathbf r$,
    summed over the nuclei $A$ with charge $Z_A$ at position $\mathbf R_A$. It depends only on the geometry and the basis.
    $T$ and $V$ together are called the core Hamiltonian $h=T+V$.

    **$J$, Coulomb (also called Hartree) repulsion.**

    $$
    J_{\mu\nu}=\sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)\,D_{\lambda\sigma},\qquad
    (\mu\nu|\lambda\sigma)=\iint\frac{\phi_\mu(\mathbf r_1)\,\phi_\nu(\mathbf r_1)\;\phi_\lambda(\mathbf r_2)\,\phi_\sigma(\mathbf r_2)}{\lvert\mathbf r_1-\mathbf r_2\rvert}\,d\mathbf r_1\,d\mathbf r_2 .
    $$

    The four-index number $(\mu\nu|\lambda\sigma)$ is an *electron-repulsion integral*: the repulsion between a charge
    distribution $\phi_\mu\phi_\nu$ and a charge distribution $\phi_\lambda\phi_\sigma$. Summing over all pairs weighted by
    $D_{\lambda\sigma}$ gives the repulsion between an electron and the whole density. $J$ is **linear** in $D$.

    **$V^{\text{xc}}$, exchange and correlation.** $V^{\text{xc}}_{\mu\nu}=\int\phi_\mu\,v_{xc}(\mathbf r)\,\phi_\nu\,d\mathbf r$,
    where $v_{xc}=\delta E_{xc}/\delta\rho$ is the *functional derivative* of the exchange-correlation energy: how
    $E_{xc}$ changes when the density is changed at the point $\mathbf r$. For the functionals in these labels it depends
    on the density $\rho$, its gradient and the kinetic-energy density at $\mathbf r$. There is no closed formula for the
    integral, so the program evaluates $\rho$ and the other inputs at a grid of points $\mathbf r_g$ with weights $w_g$ and sums:

    $$
    V^{\text{xc}}_{\mu\nu}\approx\sum_g w_g\,v_{xc}(\mathbf r_g)\,\phi_\mu(\mathbf r_g)\,\phi_\nu(\mathbf r_g)
    \quad(\text{plus terms with gradients of }\phi_\mu,\phi_\nu\text{ for GGA and meta-GGA functionals}).
    $$

    For water the grid has 99 radial and 590 angular points per atom. $V^{\text{xc}}$ depends on $D$ **non-linearly**,
    through $\rho(\mathbf r_g)$ inside $v_{xc}$.

    **$K^{\text{eff}}$, exact exchange.** The exact-exchange matrix is
    $K_{\mu\nu}=\sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)\,D_{\lambda\sigma}$. A hybrid functional includes a fraction of it. A
    range-separated hybrid uses a different fraction for short and long distances between the electrons. For ωB97M-V the
    fraction is 15 % at short distance and 100 % at long distance, with the crossover set by $\omega=0.3$ bohr$^{-1}$
    (see section 5 of `reading_a_dft_label.py`). $K^{\text{eff}}$ is that weighted mixture; it is **linear** in $D$. The factor $-\tfrac12$
    belongs to the closed-shell form (each electron exchanges only with electrons of the same spin).

    **$V^{\text{nlc}}$, the VV10 non-local term.** A correlation correction that mimics dispersion (the "V" in ωB97M-V). It
    is evaluated on its own, coarser grid and is numerically small.
    """)
    return


@app.cell
def _(mo, np, pieces, F, labels):
    _terms = [("T", pieces["T"]), ("V", pieces["V"]), ("J", pieces["J"]), ("Vxc", pieces["Vxc"]),
              ("Vnlc", pieces["Vnlc"]), ("-K/2", pieces["mK"])]
    _idx = [(0, 0), (0, 1), (1, 1)]
    _rows = ["| Term | " + " | ".join(f"[{i},{j}]" for i, j in _idx) + " |", "|---|---|---|---|"]
    for _name, _m in _terms:
        _rows.append(f"| {_name} | " + " | ".join(f"{_m[i, j]:.4f}" for i, j in _idx) + " |")
    _total = sum(m for _, m in _terms)
    _rows.append("| **sum** | " + " | ".join(f"**{_total[i, j]:.4f}**" for i, j in _idx) + " |")
    _rows.append("| F from PySCF | " + " | ".join(f"{F[i, j]:.4f}" for i, j in _idx) + " |")
    mo.vstack([
        mo.md(
            "**The terms for water, numerically.** The table shows three elements (in hartree). Indices count from 0: "
            f"element [0,0] couples basis function 0 ({labels[0]}) with itself, [0,1] couples it with function 1 "
            f"({labels[1]}). The sum of the six matrices equals the matrix PySCF itself builds, over all "
            f"{F.shape[0]}×{F.shape[1]} elements, to within {np.abs(_total - F).max():.0e}.\n\n" + "\n".join(_rows)
            + f"\n\nFor element [0,0] the three large terms nearly cancel: $T+V+J={_total[0, 0] - pieces['Vxc'][0, 0] - pieces['Vnlc'][0, 0] - pieces['mK'][0, 0]:.4f}$ Ha, "
            f"while $V^{{\\text{{xc}}}}$ ({pieces['Vxc'][0, 0]:.4f}) and $-K/2$ ({pieces['mK'][0, 0]:.4f}) are not small compared with that remainder."
        ),
    ])
    return


@app.cell
def _(mo, pieces_figure, ks, labels):
    mo.vstack([
        pieces_figure(ks),
        mo.md(
            "Each panel is one of the $58\\times58$ matrices; the colour is the value of the element (blue negative, red "
            "positive) on a signed logarithmic scale, so that small elements remain visible. Rows and columns are ordered "
            f"by atom: the first {sum(x.startswith('O ') for x in labels)} belong to the oxygen basis functions and the remaining "
            f"{sum(not x.startswith('O ') for x in labels)} to the two hydrogens. "
            "All panels are symmetric about the diagonal."
        ),
    ])
    return


@app.cell
def _(mo, ks):
    _p = ks["energy_parts_Ha"]
    _total = sum(_p.values())
    mo.md(
        "**The same split for the energy.** Each matrix has a matching energy term, obtained by tracing it with $D$: "
        r"$E_T=\mathrm{tr}(DT)$ and so on, where $\mathrm{tr}(AB)=\sum_{\mu\nu}A_{\mu\nu}B_{\nu\mu}$. For water: "
        f"kinetic {_p['kinetic']:.4f} Ha, nuclear attraction {_p['nuclear_attraction']:.4f} Ha, Coulomb "
        f"$\\tfrac12\\mathrm{{tr}}(DJ)$ = {_p['coulomb']:.4f} Ha, nuclear repulsion {_p['nuclear_repulsion']:.4f} Ha. "
        f"Exchange-correlation (from the grid) {_p['xc_energy']:.4f} Ha, VV10 {_p['nlc_energy']:.4f} Ha, exact exchange "
        f"{_p['exact_exchange_energy']:.4f} Ha. The seven numbers add up to {_total:.4f} Ha, against {ks['e_tot_Ha']:.4f} Ha "
        "from the converged calculation. The factor $\\tfrac12$ on the Coulomb energy avoids counting each electron pair twice."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Why the loop must be repeated, and the loop itself

    The terms $T$ and $V$ are fixed once the molecule and the basis are chosen. The terms $J$, $V^{\text{xc}}$ and $K$ all
    contain $D$, and $D$ is built from the orbitals $C$ that come out of solving $FC=SC\varepsilon$. So $F$ depends on its own
    solution. The way out is to iterate; this is the self-consistent-field (SCF) loop:

    1. **Guess** a density matrix $D$ (PySCF superimposes atomic densities).
    2. **Build** $F[D]$ from the six terms above.
    3. **Solve** $FC=SC\varepsilon$ (section 3).
    4. **Form** a new density from the five lowest orbitals: $D_{\text{new}}=2\,C_{\text{occ}}C_{\text{occ}}^T$.
    5. **Mix** $D_{\text{new}}$ with the old $D$ and go back to step 2, until $D$ no longer changes.

    At the end, building $F$ from the final $D$ and solving gives back the same $D$: the density is *self-consistent*.
    Production programs mix with a smarter scheme than the simple average used below (DIIS), and so need fewer cycles.

    **Hartree-Fock for comparison.** The loop and the equation $FC=SC\varepsilon$ are identical. Only the matrix differs:
    $F^{\text{HF}}=T+V+J-\tfrac12K$, with the full ($100\,\%$) exact exchange $K$ and no $V^{\text{xc}}$ or $V^{\text{nlc}}$. DFT replaces
    some or all of $K$ by $V^{\text{xc}}$, which also contains correlation.
    """)
    return


@app.cell
def _(mo, ks):
    _shift = ks["shift_from_guess"]
    _rows = ["| Term | change between the starting guess and the converged density (Frobenius norm) |", "|---|---|"]
    for _k, _name in [("T", "T"), ("V", "V"), ("J", "J"), ("Vxc", "Vxc"), ("Vnlc", "Vnlc"), ("mK", "-K/2")]:
        _rows.append(f"| {_name} | {_shift[_k]:.4f} |")
    mo.md(
        "**Which terms depend on $D$.** The table gives how much each matrix changes when $F$ is built from the starting "
        "guess instead of from the converged density. The Frobenius norm of a matrix is the square root of the sum of the "
        "squares of its elements, a single number for its size. $T$ and $V$ do not change at all; the other four do, and the "
        "Coulomb and exchange terms change most.\n\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo, scf_loop_figure, ks):
    _h = ks["hand_scf"]
    mo.vstack([
        mo.md(
            f"**The loop run by hand on water.** `compute_ks_matrix.py` does steps 1 to 5 explicitly: PySCF builds $F$ from "
            f"a density, a plain eigensolver does step 3, and the new density is averaged 50:50 with the old one. It reached "
            f"a largest change of $D$ below $10^{{-6}}$ after {len(_h)} cycles. The energy of the first cycle (from the "
            f"guess) is {_h[0]['energy_Ha']:.4f} Ha; after the last it is {_h[-1]['energy_Ha']:.6f} Ha, against "
            f"{ks['e_tot_Ha']:.6f} Ha from PySCF's own converged calculation."
        ),
        scf_loop_figure(ks),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Numerical checks on water

    All of these use the converged matrices stored in `data_ks.json`.
    """)
    return


@app.cell
def _(mo, matrices_figure, ks, F, S, labels):
    _n = 4
    def _fmt(m):
        return "\n".join("    " + "  ".join(f"{m[i, j]:9.4f}" for j in range(_n)) for i in range(_n))
    mo.vstack([
        mo.md(
            f"**The shape of $F$ and $S$, and a corner of each.** Both are {F.shape[0]}×{F.shape[1]}. The first four basis "
            f"functions are all s-type functions on the oxygen ({', '.join(labels[:4])}); the large off-diagonal elements "
            "of $S$ show that they overlap strongly, which is why $S$ cannot be dropped.\n\n"
            f"```\nF[:4, :4] (hartree)\n{_fmt(F)}\n\nS[:4, :4]\n{_fmt(S)}\n```"
        ),
        matrices_figure(ks),
    ])
    return


@app.cell
def _(mo, np, F, S, C, D, eps, n_occ, solve_generalized):
    _res = np.abs(F @ C - S @ C * eps).max()
    _scale = np.abs(F @ C).max()
    _orth = np.abs(C.T @ S @ C - np.eye(len(eps))).max()
    _diag = np.abs(C.T @ F @ C - np.diag(eps)).max()
    _e2, _C2 = solve_generalized(F, S)
    _D2 = 2 * _C2[:, :n_occ] @ _C2[:, :n_occ].T
    mo.md(
        "**Does $FC=SC\\varepsilon$ hold?** At the converged density, using the stored $F$, $S$, $C$, $\\varepsilon$:\n\n"
        f"- $\\max|FC-SC\\varepsilon|={_res:.1e}$, where the elements of $FC$ are as large as {_scale:.1f}. The residual is at the level "
        "of the convergence threshold of the calculation.\n"
        f"- $C^TSC=1$ to {_orth:.0e}: the orbitals are orthonormal.\n"
        f"- $C^TFC$ is the diagonal matrix of the $\\varepsilon_i$ to {_diag:.0e}: in the basis of the orbitals, $F$ is diagonal and the "
        "eigenvalues are its diagonal entries.\n"
        f"- Solving again from $F$ and $S$ alone with the four-step recipe of section 3 (numpy only) reproduces all 58 $\\varepsilon$ to "
        f"{np.abs(_e2 - eps).max():.0e}, and the density $D=2C_{{\\text{{occ}}}}C_{{\\text{{occ}}}}^T$ built from it differs from the "
        f"stored converged $D$ by at most {np.abs(_D2 - D).max():.0e}. That is the statement that the density is self-consistent."
    )
    return


@app.cell
def _(mo, levels_figure, ks, eps, n_occ):
    _ev = eps * 27.211386
    _rows = ["| Orbital | ε (Ha) | ε (eV) |", "|---|---|---|"]
    _names = ["1 (O 1s core)", "2", "3", "4", "5 (highest occupied)"]
    for _i in range(n_occ):
        _rows.append(f"| {_names[_i]} | {eps[_i]:.4f} | {_ev[_i]:.1f} |")
    _rows.append(f"| 6 (lowest empty) | {eps[n_occ]:.4f} | {_ev[n_occ]:.1f} |")
    mo.vstack([
        mo.md(
            "**The eigenvalues are the orbital energies.** They are the diagonal entries of $C^TFC$ and are what a program prints "
            "as orbital energies.\n\n" + "\n".join(_rows)
        ),
        levels_figure(ks),
    ])
    return


@app.cell
def _(mo, ks):
    mo.md(
        f"**Hartree-Fock comparison.** PySCF's Hartree-Fock Fock matrix for the same water equals $T+V+J-\\tfrac12K$ rebuilt from the "
        f"Hartree-Fock density to {ks['hf']['rebuild_max_diff']:.0e}; the Hartree-Fock energy is {ks['hf']['e_tot_Ha']:.4f} Ha, against {ks['e_tot_Ha']:.4f} Ha for ωB97M-V."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. What is verified and what is not

    **Computed in this notebook's data (PySCF 2.14, water, ωB97M-V/def2-TZVPD, grid (99, 590)):** the shapes, the six-term
    sum equal to $F$, the residual of $FC=SC\varepsilon$, orthonormality, diagonalisation of $F$ in the orbital basis,
    the numpy solver, the hand-written loop and its convergence, the Hartree-Fock rebuild, and the changes of each term
    between the guess and the converged density. The two-function example is checked against a numerical solver.

    **Stated from the usual textbook form, not derived here:** that $V^{\text{xc}}$ for GGA and meta-GGA functionals contains
    gradient terms of the basis functions; the closed-shell factor $-\tfrac12$ on $K$ (the code's sum with the stated weights
    reproduces $F$, which supports it but does not prove the physical argument); that the first ionisation energy is approximated by $-\varepsilon_{\text{HOMO}}$ (not tested here).

    **Settings not stated by the label:** the VV10 non-local term used PySCF's default coarser grid, (50, 194), not (99, 590).
    """)
    return


if __name__ == "__main__":
    app.run()
