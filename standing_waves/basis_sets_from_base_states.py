import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Basis Sets: From Base States to def2")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from basis_sets import (
        function_count,
        load_data,
        load_water,
        shell_string,
        sto_ng_fit,
    )
    import basis_comparison as bc
    from basis_sets_plots import (
        N_MAX,
        comparison_error_figure,
        comparison_shape_figure,
        h2plus_figure,
        hydrogen_basis_figure,
        morse_figure,
        product_figure,
        shells_figure,
        split_valence_figure,
        sto_fit_figure,
        water_basis_figure,
    )
    from qmstyle import ANGSTROM_PER_BOHR

    data = load_data()
    water = load_water()
    return (
        ANGSTROM_PER_BOHR,
        N_MAX,
        bc,
        comparison_error_figure,
        comparison_shape_figure,
        data,
        function_count,
        h2plus_figure,
        hydrogen_basis_figure,
        mo,
        morse_figure,
        np,
        plt,
        product_figure,
        shell_string,
        shells_figure,
        split_valence_figure,
        sto_fit_figure,
        sto_ng_fit,
        water,
        water_basis_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Basis Sets: From Base States to def2

    The previous lecture ended with a promise: *identifying useful base states for electrons bound to a nucleus will be
    the goal of the next lecture*. The hydrogen notebook (`spherical_harmonics_and_hydrogen.py`) kept that promise for one
    electron and one nucleus. This notebook asks what to do when the exact base states are not available, which is the
    case for every molecule. The previous notebook (`from_atoms_to_molecules.py`) named LCAO, orbitals built from
    atom-centred functions, as the third approximation of chemistry and used it for methane with invented numbers. This notebook
    asks which functions to use and how the coefficients are found. The answer, **a basis set**, is the thing named in the
    second half of a label such as `def2-TZVPD`, and the DFT notebooks that follow start from it.

    **Read first:** `spherical_harmonics_and_hydrogen.py` (this folder), for the hydrogen orbitals and the real spherical
    harmonics that reappear in section 5, and `from_atoms_to_molecules.py`, sections 1 and 5, for why molecules have no
    closed-form base states and what LCAO looks like.

    **Contents**

    1. From stationary states to base states
    2. A truncated expansion on a toy problem
    3. Many nuclei: atom-centred functions and LCAO
    3b. Why atom-centred functions? A test against standing waves in a box
    4. Which functions? Slater and Gaussian, and the contraction STO-$n$G
    5. Shells, zeta, polarization and diffuse functions
    6. Where this leads: the `def2` label
    7. What was checked, and what was not

    Units are atomic units, as in `from_atoms_to_molecules.py` (bohr, hartree, $\hbar=m_e=1$; 1 bohr $=0.529$ Å), but the figures show lengths in ångström. The numbers that need PySCF come from `compute_basis_sets.py`, which writes `data_basis_sets.json`;
    the notebook only reads that file.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Base state**: one of the fixed waves $\psi_n$ out of which a general wave is built as a sum.
        - **Eigenfunction of $H$**: a wave $\psi_E$ with $H\psi_E = E\psi_E$. The exact base states of last lecture.
        - **Basis function** $\chi_\mu$: a fixed function chosen for convenience. It need not be an eigenfunction of $H$.
          A **basis set** is a collection of them.
        - **Coefficient** $c_\mu$: the weight of one basis function in the sum.
        - **Variational principle**: for any trial wave the energy computed from it is never below the true lowest energy.
        - **LCAO**: linear combination of atomic orbitals, a molecular wave written as a sum of functions centred on the atoms.
        - **Slater-type function**: $e^{-\zeta r}$. **Gaussian function**: $e^{-\alpha r^2}$. The letters $\zeta$ and $\alpha$
          (the *exponents*) set how fast the function falls with the distance $r$ from its centre.
        - **Contraction**: a fixed sum of Gaussians, used as one basis function.
        - **Shell**: all basis functions that share a radial part and an angular momentum $\ell$.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 From stationary states to base states

    The last lecture assumed a time-independent Hamiltonian $H$ and looked for solutions of the form
    $\psi(x,t)=\psi_E(x)\,T(t)$. Separating the variables gave the equation for the spatial part,

    $$
    H\,\psi_E(x) = E\,\psi_E(x),
    $$

    and a time factor $T(t)=e^{-iE t/\hbar}$. Such a state has a probability density $\lvert\psi(x,t)\rvert^2=\lvert\psi_E(x)\rvert^2$
    that does not change in time. A general wave is a sum of these states, one for each allowed energy $E_n$,

    $$
    \psi(x,t)=\sum_n c_n\,\psi_{E_n}(x)\,e^{-iE_n t/\hbar},
    $$

    and the sum itself does change in time, because the terms turn with different frequencies $E_n/\hbar$. The numbers
    $c_n$ say how much of each stationary state is present at the start. The functions $\psi_{E_n}$ are the **base states**.
    Three things follow.

    1. **The base states are the eigenfunctions of $H$.** Knowing them solves the problem for all times: the
       time dependence is written out, and only the numbers $c_n$ have to be found from the starting wave.
    2. **The set must be complete.** Every wave the system can have must be expressible as such a sum. Completeness
       is what makes the word *base* fair; the exact eigenfunctions of $H$ always have it.
    3. **In practice the sum is cut off.** A computer can hold a finite number $N$ of terms. The question becomes how
       large an error the cut-off causes and how to choose the first $N$ terms well.

    For hydrogen the exact eigenfunctions are known in closed form: the orbitals $\psi_{n\ell m_\ell}$ of the hydrogen
    notebook, with energies $-\tfrac{1}{2n^2}$ hartree. They are the natural base states for one electron bound to one
    nucleus. The rest of this notebook is about a different situation: the exact eigenfunctions are *not* known, so
    we build the sum out of other fixed functions $\chi_\mu$ that we are free to choose, and let the coefficients
    adjust. This section sets up the equation that finds the coefficients.

    **The equation for the coefficients.** Write the unknown wave of the time-independent problem as a sum of $N$ fixed
    functions, $\psi=\sum_{\nu} c_\nu\chi_\nu$, and put it into $H\psi=E\psi$:

    $$
    \sum_\nu c_\nu\,H\chi_\nu = E\sum_\nu c_\nu\,\chi_\nu .
    $$

    The equation holds at every point $x$. To turn it into a finite set of equations, multiply by one of the functions,
    $\chi_\mu$ (taken real here), and integrate over all $x$. This gives one equation for each $\mu$:

    $$
    \sum_\nu H_{\mu\nu}\,c_\nu = E\sum_\nu S_{\mu\nu}\,c_\nu,
    \qquad
    H_{\mu\nu}=\int\chi_\mu\,H\chi_\nu\,dx,
    \quad
    S_{\mu\nu}=\int\chi_\mu\chi_\nu\,dx .
    $$

    In matrix form this is $\mathbf H\mathbf c = E\,\mathbf S\mathbf c$, with $\mathbf H$ and $\mathbf S$ tables of numbers
    that depend only on the chosen functions, and $\mathbf c$ the column of coefficients. If the functions are orthonormal
    ($S_{\mu\nu}=1$ for $\mu=\nu$ and $0$ otherwise), $\mathbf S$ is the unit matrix and the equation reduces to the
    familiar eigenvalue equation $\mathbf H\mathbf c = E\mathbf c$: finding the $N$ pairs $(E,\mathbf c)$ is a standard
    computer task. The overlap matrix $\mathbf S$ is needed only when the functions overlap, as atom-centred
    functions on neighbouring atoms do (sections 3 to 5).

    **The variational principle** gives the answers their meaning. For any trial wave $\psi$, however chosen, the number
    $\langle\psi\vert H\vert\psi\rangle/\langle\psi\vert\psi\rangle$ is at least the lowest true energy $E_0$. The matrix
    equation finds the trial wave inside the span of the $N$ functions that makes this number as low as possible.
    So a truncated calculation gives an energy that is **never too low**, and adding a function can only lower it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 A truncated expansion on a toy problem

    To watch the cut-off at work we need a problem whose exact answer is known and which is not itself the basis. Take a
    particle (mass $1$) on a line in a **Morse well**,

    $$
    V(x)=D\left(1-e^{-a x}\right)^2,\qquad D=6\ \text{hartree},\quad a=0.5\ \text{bohr}^{-1}=0.94\ \text{Å}^{-1}.
    $$

    It is the standard one-dimensional picture of a chemical bond: steep on the left where the atoms push apart,
    flattening to the energy $D$ on the right where the bond is broken. Its levels are known exactly,
    $E_n=\omega(n+\tfrac12)-\omega^2(n+\tfrac12)^2/(4D)$ with $\omega=a\sqrt{2D}$, and there are seven bound ones. The exact
    ground state is also known in closed form, so we can draw it.

    **The fixed functions.** We use the first $N$ eigenfunctions $\phi_0,\dots,\phi_{N-1}$ of a harmonic oscillator with the
    frequency $\omega$ of the bottom of the well. These are exact base states of a *different* Hamiltonian, the one that
    touches the Morse well at its minimum. They are orthonormal and, taken all together, complete, so with enough of them
    any wave on the line can be written as a sum. The matrix element $H_{mn}$ is the integral of $\phi_m H\phi_n$. The
    oscillator part of it is diagonal, $\omega(n+\tfrac12)$, and the rest comes from the difference
    $V(x)-\tfrac12\omega^2x^2$.

    Use the slider to choose $N$.
    """)
    return


@app.cell
def _(mo):
    n_morse = mo.ui.slider(1, 12, step=1, value=3, label=r"$N$", show_value=True)
    return (n_morse,)


@app.cell
def _(mo, morse_figure, n_morse, plt):
    _fig = morse_figure(n_morse.value)
    _out = mo.vstack([n_morse, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    What to read from the figure.

    - **Left:** the ground state drawn on top of the well at its own energy. With $N=1$ the wave is a symmetric Gaussian.
      The true wave is lopsided, and each added function lets the sum lean toward the gentle right-hand side.
    - **Middle:** $c_n^2$ for the ground state. The oscillator functions are orthonormal, so these weights add to $1$, and
      $c_n$ is the amount of function $\phi_n$ in the answer.
    - **Right:** how far each of the three lowest levels lies above its exact value (logarithmic axis). Every point is
      positive: **the truncated energy is never below the exact one**. Every curve falls as functions are added, because
      the larger set contains the smaller one. Higher levels need more functions, since they are more
      wiggly and extend farther.

    Two further points are tested rather than just drawn. If the oscillator functions are replaced by the *non-orthogonal*
    functions $x^k e^{-\omega x^2/2}$, the matrix $\mathbf S$ is no longer the unit matrix, the same general equation
    $\mathbf H\mathbf c=E\mathbf S\mathbf c$ is solved, and the energies agree with the figure. What matters is the span of the functions (the set of all
    sums of them), not their individual shapes. Conversely, the *choice* of functions matters for how
    fast the answer converges, which is the whole point of the next sections.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 Many nuclei: atom-centred functions and LCAO

    Why not simply use the exact hydrogen orbitals for every atom in every molecule? They exist in closed form only because
    there is one electron and one centre of force. With several electrons pushing on each other and several nuclei pulling
    from different centres, nothing separates (`from_atoms_to_molecules.py`, section 1). So there are no closed-form
    base states for a molecule, and the strategy of section 2 takes over: choose fixed
    functions, build the matrices, and let the coefficients find the best combination. The choice made in chemistry is to centre
    the functions $\chi_\mu$ **on the atoms** and to let each molecular wave be

    $$
    \psi(\mathbf r)=\sum_\mu c_\mu\,\chi_\mu(\mathbf r).
    $$

    This is the **LCAO** idea, already used for methane in `from_atoms_to_molecules.py`, section 5. The functions on an atom look like atomic orbitals, because an electron near a
    nucleus mostly feels that nucleus, and the coefficients mix them into bonds. Whether the shapes are
    well chosen is checked by the variational principle: lower energy is better.

    **The smallest example: H$_2^+$.** One electron and two protons $A$ and $B$ a distance $R$ apart, with
    $H=-\tfrac12\nabla^2-1/r_A-1/r_B$, where $r_A$ and $r_B$ are the distances of the electron from the two protons. Take one function
    on each proton, the hydrogen 1s function $\chi_A=e^{-r_A}/\sqrt\pi$ (and $\chi_B$ likewise). The $2\times2$ equation
    $\mathbf H\mathbf c=E\mathbf S\mathbf c$ has the matrices

    $$
    \mathbf H=\begin{pmatrix}H_{AA}&H_{AB}\\H_{AB}&H_{AA}\end{pmatrix},\qquad
    \mathbf S=\begin{pmatrix}1&S\\S&1\end{pmatrix},
    $$

    with the closed-form integrals

    $$
    S=e^{-R}\left(1+R+\tfrac13R^2\right),\quad
    J=\tfrac1R-e^{-2R}\left(1+\tfrac1R\right),\quad
    K=e^{-R}(1+R),\quad
    H_{AA}=-\tfrac12-J,\quad H_{AB}=-\tfrac12 S-K .
    $$

    Here $S$ is the overlap of the two functions, $J$ the attraction of the electron cloud on one proton to the other proton,
    and $K$ a similar integral for the product $\chi_A\chi_B$. The two protons are identical, so the solutions have
    $c_A=c_B$ or $c_A=-c_B$. Adding the two rows of the equation for $c_A=c_B=c$ gives $(H_{AA}+H_{AB})c=E(1+S)c$, and likewise with a minus sign:

    $$
    E_\pm=\frac{H_{AA}\pm H_{AB}}{1\pm S}\quad\text{(electron only)},\qquad
    E_{\text{total}}=E_\pm+\frac1R .
    $$

    The last term is the repulsion of the two protons. Use the slider to change $R$. The formulas are in atomic units, so $R$ and $r$ in them are in bohr; the slider and the figure show $R$ in ångström ($R_{\text{bohr}}=R_{\text{Å}}/0.529$).
    """)
    return


@app.cell
def _(mo):
    r_h2 = mo.ui.slider(0.55, 3.2, step=0.05, value=1.05, label=r"$R$ (Å)", show_value=True)
    return (r_h2,)


@app.cell
def _(ANGSTROM_PER_BOHR, h2plus_figure, mo, plt, r_h2):
    _fig = h2plus_figure(r_h2.value / ANGSTROM_PER_BOHR)  # the figure function works in bohr
    _out = mo.vstack([r_h2, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The sum $c_A=c_B$ puts extra amplitude between the protons (**bonding**), the difference puts a node there
    (**antibonding**). Only the bonding energy dips below the energy $-\tfrac12$ of a separated hydrogen atom and a
    proton. Two functions give a minimum at $R\approx 1.32$ Å and a binding energy of about 1.8 eV. The dots are
    a calculation with a very large basis on each proton (PySCF, `aug-cc-pV5Z`; the same points as the energy curve of
    `from_atoms_to_molecules.py`, section 2): it has its minimum near
    $R=1.06$ Å and a binding energy of about 2.8 eV, so two functions give roughly 63 % of the bond energy. The 2-function curve lies above the dots at every $R$, as the variational principle
    requires. **Making the basis richer on each atom is what the rest of the notebook is about.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3b Why atom-centred functions?

    Section 1 said that *any* complete set of functions works if enough terms are kept, and the box standing waves of
    `standing_waves_in_boxes.py` are a complete set. Why, then, does every molecular program use functions centred on the
    atoms? The reason is the number of terms needed for a given accuracy. This section measures it on the two smallest
    problems of this notebook, with exact answers to compare with.

    **Setup.** One electron, atomic units. The two problems are the hydrogen atom (exact energy $-\tfrac12$ hartree) and
    H$_2^+$ with the protons $1.06$ Å ($2$ bohr) apart (exact energy $-0.6026$ hartree, including the repulsion of the protons).
    Two kinds of basis are compared, each solved by the equation $\mathbf H\mathbf c=E\mathbf S\mathbf c$ of section 1.

    - **Box waves.** A cube of side $L$ with the nucleus at its centre (the protons on the vertical axis for H$_2^+$) and
      hard walls, where $\psi=0$. The functions are products $\cos(k_xx)\cos(k_yy)\cos(k_zz)$ with
      $k=\pi/L,\,3\pi/L,\,5\pi/L,\dots$, the standing waves of the box notebook that are even about the centre. The ground
      state is even, so the odd waves play no role and are left out; the count $N$ below therefore *favours* the box (the full set of
      standing waves of the cube is about eight times larger). They are added in order of their kinetic energy $\tfrac12(k_x^2+k_y^2+k_z^2)$,
      lowest first, which is the same rule as the energy cutoff of a plane-wave program. The attraction $-1/r$ is the only
      integral that is not a product of one-dimensional integrals; it is written as a sum of
      Gaussians, $1/r=\tfrac{2}{\sqrt\pi}\int_0^\infty e^{-s^2r^2}\,ds$, which makes every matrix element a sum of products.
    - **Atom-centred Gaussians.** s functions $e^{-\alpha r^2}$ on each nucleus, as in section 4 below, with the exponents $\alpha$
      chosen to give the lowest hydrogen-atom energy for the number of functions used. For H$_2^+$ the same exponents are put on each
      proton, without readjusting them. $N$ counts all functions on both protons.

    The slider sets the side $L$ of the box. A larger box lowers the energy that the walls take away (a wave squeezed into a box has extra kinetic energy) but spreads the same
    number of waves over more volume. Computing a new box takes several seconds.
    """)
    return


@app.cell
def _(mo):
    box_slider = mo.ui.slider(4.0, 12.0, step=1.0, value=8.0, label=r"box side $L$ (Å)", show_value=True, debounce=True)
    return (box_slider,)


@app.cell
def _(ANGSTROM_PER_BOHR, box_slider, comparison_error_figure, mo, plt):
    _fig = comparison_error_figure(box_slider.value / ANGSTROM_PER_BOHR)  # the numerics work in bohr
    _out = mo.vstack([box_slider, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the left panel.** The vertical axis is how far above the exact energy each calculation lies, on a
    logarithmic scale, and the dotted line is $1.6$ millihartree, about 1 kcal/mol, a usual target for chemical accuracy.
    Four Gaussians on the nucleus are already close to that line for the hydrogen atom, and eight are within $10^{-5}$ hartree. Over a thousand box waves are *still*
    between $4\times10^{-3}$ and $2\times10^{-2}$ hartree above the exact value, depending on the box size on the slider (the best of the
    sizes offered is 5 to 6 Å). The curve of the box flattens at large $N$ for two separate reasons, and the slider separates them:
    in a small box the walls themselves cost energy, and in a large box the waves are too coarse to follow the sharp
    structure at the nucleus.

    **Reading the right panel.** The same for H$_2^+$. With only s functions the Gaussians stop improving at about $10^{-2}$ hartree: the
    bonding orbital is stretched along the bond, which s functions on the protons cannot describe. Adding one $p_z$ function
    on each proton (the polarization functions of section 5) removes this, and the error continues to fall to the
    millihartree range at $N=14$. The box has no such problem, and no such help either: its error falls slowly, by a factor of about 3 to 5 for
    every factor of 10 in $N$.

    **Why, in words.** Two things in the exact wave are hard for a plane or standing wave and easy for an atom-centred function.
    """)
    return


@app.cell
def _(mo):
    n_box_shape = mo.ui.dropdown({str(_n): _n for _n in (10, 46, 215, 1000)}, value="1000", label="box waves $N$")
    n_gauss_shape = mo.ui.slider(1, 6, step=1, value=3, label="Gaussians $N$", show_value=True)
    return n_box_shape, n_gauss_shape


@app.cell
def _(ANGSTROM_PER_BOHR, box_slider, comparison_shape_figure, mo, n_box_shape, n_gauss_shape, plt):
    _fig = comparison_shape_figure(box_slider.value / ANGSTROM_PER_BOHR, n_box_shape.value, n_gauss_shape.value)
    _out = mo.vstack([mo.hstack([n_box_shape, n_gauss_shape], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The figure shows the hydrogen 1s electron density along a line through the nucleus (box side from the slider above), exact and as
    the best sum of the chosen number of functions of each kind.

    1. **The nucleus (left panel).** The exact density has a corner at the nucleus, the cusp of section 4,
       and a waves-and-nodes basis has to build a corner out of smooth functions of ever shorter wavelength. The box waves
       reach the top of the density only slowly with $N$: to resolve a small region they need short wavelengths in all three directions
       at once, and the number of waves grows quickly with the shortest wavelength included. A handful of Gaussians of different widths, all centred on the nucleus, already follow the peak, because the width of each is a free
       parameter that the calculation chooses. The Gaussians are not exact either: they stay flat at $r=0$ (section 4).
    2. **The tail (right panel).** Far from the nucleus the exact density falls as a straight line on this logarithmic scale. Neither basis
       follows it all the way. The box waves show ripples, because a sum of waves of fixed wavelengths has to cancel
       itself out over a long distance, and end at the wall. The Gaussians fall away faster than the exact line (section 4).
       Both errors sit where the density is below about $10^{-3}$ of its peak value, so they cost little energy: the large energy errors of the box come from the nucleus.

    There are two further reasons, neither tested here.

    3. **Atom-centred functions follow the atoms.** Move a nucleus and its functions move with it, so a functions-per-atom budget needs no
       change. The box is fixed in space: a molecule that is twice as long needs a box twice as long in one direction, and the number of
       waves needed at the same resolution grows with the *volume of the box*, which is counted in the tests (eight times the number of functions for twice the side at a
       fixed cutoff). Most of that volume is empty space, where the electron density is
       negligible. A localized basis grows with the number of atoms.
    4. **The orbital is a few atomic orbitals mixed.** The bonding orbital of H$_2^+$ at the top of this section was already roughly right with
       two functions (the minimal basis of section 3, about 63 % of the bond energy), the rest being refinement. A chemical bond is a small change of the atom's electron cloud, so the atom's own functions
       are a good starting point.

    **The answer is not universal.** Two cautions follow.

    - *Plane waves are the standard choice for crystals.* In a periodic solid the box is the repeating cell, the waves fit it exactly, there
      are no walls, the matrix elements are simple, and the cost no longer depends on where the atoms are. The
      price is the nuclear region: the sharp structure of the electrons near each nucleus (the cusp and the oscillating inner
      shells) needs a very short wavelength, so plane-wave programs replace the nucleus and its inner electrons
      by a smooth effective potential, a *pseudopotential* (or use the projector-augmented-wave variant of it), and keep only the valence electrons. It is the same observation as above,
      the shape near the nucleus decides the basis, handled by changing the problem and not the basis.
    - *Atom-centred sets have their own flaws.* The basis is never complete, and the error depends on the choice of functions (the
      subject of sections 4 to 6). Two molecules that approach each other borrow each other's functions to lower their energy, the
      **basis-set superposition error**, and wide (diffuse) functions on neighbouring atoms can become nearly the same function, a
      near **linear dependence** that makes $\mathbf S$ almost singular. A box or plane-wave basis has neither
      problem, and a single parameter, the cutoff, improves it systematically.

    Atom-centred Gaussians are therefore the pragmatic choice for isolated molecules and clusters, where they need the fewest functions, and
    plane waves for periodic solids.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Which functions? Slater and Gaussian, and the contraction STO-$n$G

    The 1s function of hydrogen is $e^{-r}$ times a constant. A function of this kind, $e^{-\zeta r}$, is called
    **Slater-type**. It has two features that the exact solutions of atoms always have.

    - **A corner at the nucleus (the cusp).** The potential $-Z/r$ goes to minus infinity at the nucleus, which forces the
      derivative of the wave there to be $\psi'(0)=-Z\,\psi(0)$ (stated here; checked numerically for the hydrogen 1s function in the tests). The wave is
      not smooth at $r=0$.
    - **A slow, exponential tail.** Far out the wave falls like $e^{-\sqrt{2\lvert E\rvert}\,r}$, a straight line on a logarithmic plot.

    A **Gaussian function** $e^{-\alpha r^2}$ has neither: its derivative at the centre is zero, and it falls faster and
    faster with distance. So it is a poor copy of an atomic orbital. Gaussians are used anyway, because of how they
    multiply. Every matrix element in $\mathbf H$ and $\mathbf S$ is an integral of products of two or more basis
    functions, and for atom-centred functions there are millions of them.

    **The product of two Gaussians is a Gaussian.** In one dimension, with centres $A$ and $B$,

    $$
    e^{-\alpha(x-A)^2}\,e^{-\beta(x-B)^2}=K\,e^{-(\alpha+\beta)(x-P)^2},\qquad
    P=\frac{\alpha A+\beta B}{\alpha+\beta},\quad
    K=e^{-\frac{\alpha\beta}{\alpha+\beta}(A-B)^2}.
    $$

    To see it, add the exponents and expand: $\alpha(x-A)^2+\beta(x-B)^2=(\alpha+\beta)x^2-2(\alpha A+\beta B)x+\alpha A^2+\beta B^2$.
    Divide by $\alpha+\beta$ and **complete the square**, $x^2-2Px=(x-P)^2-P^2$, with $P$ as given. What is left over,
    $\alpha A^2+\beta B^2-(\alpha+\beta)P^2$, simplifies to $\frac{\alpha\beta}{\alpha+\beta}(A-B)^2$, which does not depend on $x$ and is the
    number in $K$. In three dimensions the same holds for each of $x,y,z$.
    The product of two functions on different atoms is therefore a *single* Gaussian at a point $P$ between the
    atoms, and integrals of Gaussians are known in closed form. For two exponentials the product has kinks at both
    centres and no such simple form (right-hand panel).
    """)
    return


@app.cell
def _(mo, plt, product_figure):
    _fig = product_figure()
    plt.close(_fig)
    mo.vstack([_fig, mo.md(
        r"The product theorem is checked numerically at random points in three dimensions in `test_basis_sets.py`."
    )])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Contraction.** To get back some of the shape of a Slater function, several Gaussians with fixed relative weights
    are added up and used as *one* basis function, a **contracted** function. The standard set of this kind is
    **STO-$n$G**: $n$ Gaussians whose exponents and weights are fitted to a Slater 1s function. Here the fit
    maximises the overlap with $e^{-r}$ (this is also how the published sets were made), computed in `basis_sets.py`.
    Choose $n$.
    """)
    return


@app.cell
def _(mo):
    n_sto = mo.ui.slider(1, 6, step=1, value=3, label=r"$n$ Gaussians", show_value=True)
    return (n_sto,)


@app.cell
def _(mo, n_sto, plt, sto_fit_figure):
    _fig = sto_fit_figure(n_sto.value)
    _out = mo.vstack([n_sto, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, n_sto, np, sto_ng_fit):
    _a, _c, _ = sto_ng_fit(n_sto.value)
    _rows = ["| $k$ | exponent $\\alpha_k$ | coefficient $c_k$ |", "|---|---|---|"]
    _rows += [f"| {_k + 1} | {_x:.4f} | {_y:.4f} |" for _k, (_x, _y) in enumerate(zip(_a, _c))]
    mo.md(
        "The fitted exponents and coefficients for the chosen $n$ (for a Slater function with $\\zeta=1$, normalised "
        "Gaussians, the contraction normalised to 1). For $n=3$ and $n=6$ they agree with the tabulated STO-3G and "
        "STO-6G hydrogen sets after scaling the exponents by $\\zeta^2$:\n\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Reading the four panels:

    - **The whole function and the nucleus.** The Gaussians cannot make a corner: their sum has a flat top at $r=0$, but with more terms the flat region shrinks.
    - **The tail.** On the logarithmic plot the Slater function is a straight line. The sum of Gaussians follows it for
      a while and then bends away downward, because the *widest* Gaussian in the set sets how far the tail is
      represented. This is why wide, low-exponent functions are added when the tail matters (section 5).
    - **The error.** The overlap deficit $1-\text{overlap}^2$ and the hydrogen-atom energy computed from the contraction
      both fall steadily with $n$. The energy is never below $-\tfrac12$ hartree, the exact value, as the variational principle says. One Gaussian
      gives about $-0.424$, three about $-0.495$, six about $-0.4998$.

    The cost of a basis function is a cost in *how many Gaussians appear in each integral*, and the benefit is
    the improvement in the fit. A contracted function with $n$ terms is evaluated as a sum of $n$ simple pieces.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Shells, zeta, polarization and diffuse functions

    So far each atom carried a single spherical function. A useful basis has several per atom, of several kinds.
    This section goes through the kinds that appear in the label `def2-TZVPD`.

    ### Angular parts: shells and counting

    As in the hydrogen notebook (section 6), a basis function is a radial part times an angular part, and the angular
    part is a real spherical harmonic $Y_{\ell m}$. A **shell** of angular momentum $\ell$ is the set of functions with the same radial part and all
    $2\ell+1$ harmonics of that $\ell$:

    | shell | $\ell$ | functions in the shell |
    |---|---|---|
    | s | 0 | 1 |
    | p | 1 | 3 ($p_x,p_y,p_z$) |
    | d | 2 | 5 |
    | f | 3 | 7 |

    A common choice for the radial part of a function with angular momentum $\ell$ is $r^\ell e^{-\alpha r^2}$. The
    figure shows cross sections through the nucleus (for $\alpha=0.45$ bohr$^{-2}=1.6$ Å$^{-2}$, with lengths in ångström): the s function is round, $p_z$ has one nodal plane and two
    lobes of opposite sign, $d_{z^2}$ has two nodal cones. The right-hand panel shows what the p function is for. A
    pure s function gives a cloud that is round about the nucleus (dashed contours). Adding some $p_z$ shifts the cloud along the $z$ axis (solid contours).
    That is **polarization**: functions of higher angular momentum than the atom's occupied orbitals let the cloud move off-centre, as it must in a bond,
    where the neighbour pulls it to one side.
    """)
    return


@app.cell
def _(mo, plt, shells_figure):
    _fig = shells_figure()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(function_count, mo, shell_string, water):
    _rows = ["| Basis | shells on O | shells on each H | functions on O | on H | H$_2$O total |", "|---|---|---|---|---|---|"]
    for _name, _v in water.items():
        _o, _h = function_count(_v["O_shells"]), function_count(_v["H_shells"])
        _rows.append(f"| {_name} | {shell_string(_v['O_shells'])} | {shell_string(_v['H_shells'])} | {_o} | {_h} | {_o} + 2·{_h} = {_o + 2 * _h} |")
    mo.md(
        "**Counting functions.** Each shell of angular momentum $\\ell$ contributes $2\\ell+1$ functions, so a molecule's "
        "basis size is a sum over atoms and shells. The table counts the shells of water in several basis sets (shell "
        "data from `../dft/data_basis.json`) and checks the sum against PySCF's own count; a shell here means "
        "*contracted* shells, so `3s` is three radial functions with s symmetry:\n\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Zeta: more than one radial function per shell

    In a molecule the valence orbital of an atom swells or shrinks depending on the bonds. A single contracted function
    has a fixed shape, so it cannot follow. **Split valence** gives the valence shell more than one radial function: a
    **double-zeta** set has two (one tight, one wide), a **triple-zeta** set (the TZ in `def2-TZVPD`) has three. The calculation then
    chooses their relative weights. The letter $\zeta$ is the Slater exponent in $e^{-\zeta r}$, so "zeta" counts the
    number of radial functions of different width.

    The figure imitates this with the three Gaussians of the STO-3G fit of the previous section. The target is a 1s
    function with exponent $\zeta$ (large means tight). A *frozen* function (single zeta) keeps the shape that was fitted at
    $\zeta=1$. The *double* set splits off the widest Gaussian as its own function, and the *triple* set lets all
    three weights vary. This is an illustration of the idea with a tiny set, not the actual `def2` functions. Move $\zeta$.
    """)
    return


@app.cell
def _(mo):
    zeta_slider = mo.ui.slider(0.6, 2.0, step=0.05, value=1.3, label=r"target exponent $\zeta$", show_value=True)
    return (zeta_slider,)


@app.cell
def _(mo, plt, split_valence_figure, zeta_slider):
    _fig = split_valence_figure(zeta_slider.value)
    _out = mo.vstack([zeta_slider, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    At $\zeta=1$ every option is as good as the original fit. Away from it the frozen function degrades fastest, the
    double set recovers most of the loss, and the triple set is better still. Each step costs extra functions, which is
    why the sizes in the table above grow from SVP to TZVP to QZVP.

    ### Polarization and diffuse functions: the hydrogen atom as a test

    Two more ingredients, **P** and **D**, can be tested on the hydrogen atom, where the exact answer is
    known. With one electron, the lowest energy of the Hartree-Fock equations (the UHF energy; the method is explained in
    `../dft/reading_a_dft_label.py`, section 2, and for one electron it just solves the one-electron problem in the basis)
    is exact *within* the basis, so every difference from $-\tfrac12$ hartree is a basis-set error. The figure shows, for several basis sets (PySCF):

    - **Left:** the polarizability $\alpha_{\text{pol}}$, how much dipole an electric field $F$ induces. The atom's
      energy in a weak field is $-\tfrac12\alpha_{\text{pol}}F^2$, and it was obtained here from small fields
      $\pm0.002$. The exact value, $\tfrac92$ in atomic units (bohr$^3$), i.e. $0.67$ Å$^3$, is the textbook value for hydrogen (from memory); a basis can only
      give a value at or below it.
    - **Right:** the electron density along a line out from the nucleus, on a logarithmic scale, against the exact
      $e^{-2r}/\pi$.
    """)
    return


@app.cell
def _(data, hydrogen_basis_figure, mo, plt):
    _fig = hydrogen_basis_figure(data)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    What the numbers say.

    - **Without a p function the atom cannot respond to a field at all:** `def2-TZVP` with its p shell removed and STO-3G
      both give $\alpha_{\text{pol}}=0$. The energy of the *undisturbed* atom does not change when the p shell is added
      (the two `def2-TZVP` energies agree), because the field-free atom is round. Polarization functions are for
      the *shape change*, not for the atom.
    - **A polarization function is not automatically a good one for every purpose.** The single p shell of `def2-TZVP` captures only
      about $0.07$ Å$^3$ of the exact $0.67$ Å$^3$. The `D` version of the set, `def2-TZVPD`, adds one more, wide p function on
      hydrogen and reaches about $0.64$ Å$^3$: a field-distorted cloud reaches far from the nucleus, and only wide functions describe it. `def2-TZVPPD`, with two p shells and a d shell, gets about $0.65$ Å$^3$.
    - **Diffuse functions are about the far region.** The tails in the right panel fall away from the exact line at large $r$ for each basis, earlier for smaller
      sets. For hydrogen the D adds a p function only, so the spherical density of `def2-TZVPD` is identical to that of `def2-TZVP`
      (the two lines coincide; the notebook shows one). On oxygen the table above shows that the D adds one extra s, p and d
      shell (`5s3p2d` to `6s4p3d`), so the diffuse tail of the atom's cloud is represented there too.

    This is the reason for the letter D in the labels of datasets that deal with hydrogen bonds, anions and other
    contacts between molecules: what happens far from the nuclei is controlled by the widest functions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Water: energy against basis size

    The water numbers used elsewhere in these notebooks (ωB97M-V energies from `../dft/data_basis.json`) show the same story at
    the molecular scale. The figure plots each basis's energy above the energy of `def2-QZVP`, the largest basis in
    the set. This is **not** the complete-basis limit, so the plotted numbers understate the true errors. The basis sizes
    are 7 (STO-3G), 24, 43, 58, 59, 74 and 117 functions.
    """)
    return


@app.cell
def _(mo, plt, water, water_basis_figure):
    _fig = water_basis_figure(water)
    plt.close(_fig)
    mo.vstack([_fig, mo.md(
        r"Blue points: no diffuse functions. Orange points: the D sets (`TZVPD`, `TZVPPD`). Energies fall as functions are added, "
        r"steeply at first (STO-3G to SVP to TZVP) and then in small steps: the usual diminishing return of adding functions."
    )])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Where this leads: the `def2` label

    Everything in the label `def2-TZVPD` now has a meaning:

    | Piece | Meaning, in the words of this notebook |
    |---|---|
    | `def2` | one consistent family of atom-centred Gaussian basis sets |
    | `TZ` | triple zeta: three radial functions of different width for each valence shell (section 5, zeta) |
    | `V` | the extra functions are for the valence shells; core shells get fewer |
    | `P` | polarization: shells of higher angular momentum than the atom's occupied ones, such as p on hydrogen (section 5, polarization) |
    | `D` | diffuse: wide functions for the far region (section 5, diffuse) |

    The coefficients $c_\mu$ of the molecular orbitals come from an equation of the same form as $\mathbf H\mathbf c=E\mathbf S\mathbf c$
    (section 1), with the Fock or Kohn-Sham matrix in place of $\mathbf H$ (stated here, not derived). That matrix
    is the subject of the DFT notebooks.

    **To continue:** `../dft/reading_a_dft_label.py` takes the other half of the label. Its section 2 introduces Hartree-Fock,
    the first method that supplies that matrix; its sections 3 to 7 explain the functional and the dispersion term; and its
    section 4 returns to `def2-TZVPD` and `def2-TZVPPD` as used by OMol25 and SPICE II, adding only the counterpoise
    test of why the diffuse functions matter between molecules. What this notebook explains is not repeated there.

    **Takeaways for a slide**

    1. Base states are the eigenfunctions of $H$. When they are not known, any fixed functions can serve, with a matrix equation
       $\mathbf H\mathbf c=E\mathbf S\mathbf c$ for the coefficients; the variational principle says that more functions never raise the energy.
    2. For molecules the functions are centred on the atoms (LCAO), because there are no closed-form molecular eigenfunctions.
    3. Gaussians replace Slater functions because their products are Gaussians, so integrals are fast; contraction repairs the shape.
    4. Atom-centred functions need far fewer terms than box waves for one nucleus or one bond (section 3b): they carry the
       cusp and the decay of the atom and move with the nuclei. Plane waves remain the standard for periodic solids.
    5. Zeta gives flexible widths, P gives flexible shapes, D gives reach into the tails.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 What was checked, and what was not

    **Checked numerically** (`pixi run python -m pytest -q standing_waves/test_basis_sets.py`, 45 tests):

    - *Section 2.* The Morse levels agree with a finite-difference solution, and the closed-form ground state satisfies
      $H\psi=E\psi$. The oscillator functions are orthonormal ($\mathbf S=\mathbf 1$). Truncated energies are never below
      the exact ones and never rise when a function is added. The non-orthogonal set $x^ke^{-\omega x^2/2}$ solved with
      $\mathbf S\ne\mathbf 1$ gives the same levels; the solver's residual $\mathbf H\mathbf c-E\mathbf S\mathbf c$ and normalisation
      $\mathbf c^T\mathbf S\mathbf c=1$ are tested.
    - *Section 3.* The integrals $S$, $J$, $K$ and the matrix element $H_{AB}$ match numerical quadrature in spheroidal
      coordinates; the coefficients are $c_A=\pm c_B$ with the stated normalisation; the minimum of the two-function
      curve is at $R\approx1.32$ Å ($2.49$ bohr) with $E\approx-0.5648$ hartree (binding about 1.76 eV); it lies above the PySCF
      large-basis curve at all 14 distances; that curve gives about 2.79 eV.
    - *Section 3b.* The sum of Gaussians for $1/r$ against $1/r$ from $10^{-3}$ to 30 bohr; the box hamiltonian against the closed-form
      $-4/(3\pi)$ for a Gaussian trial function; box energies variational (never below $-\tfrac12$ or $-0.6026$ hartree) and never
      rising with $N$; the Gaussian hydrogen energies against $-4/(3\pi)$ for one function, at or below the tabulated STO-3G and STO-6G
      energies, and the general Gaussian code (including $p_z$) against closed-form s integrals; H$_2^+$ Gaussian energies variational;
      four Gaussians beating 1331 box waves by a factor of at least 5; no cusp in either expansion; box function count growing as $L^3$.
    - *Section 4.* Normalisation of the Slater and Gaussian functions; the analytic Gaussian overlap, kinetic and
      attraction integrals against numerical radial integrals; the STO-3G and STO-6G exponents and coefficients against
      PySCF's tabulated sets; overlap and energy improving with $n$ and staying variational; zero cusp derivative for
      Gaussians and $-1$ for the exact hydrogen 1s function; the tail ratio; the Gaussian product theorem.
    - *Section 5.* Shell sizes $2\ell+1$ and the function counts, which reproduce every water basis size (7, 24, 43, 58,
      59, 74, 117); parity of the $s$, $p_z$, $d_{z^2}$ shell functions; the ordering of the split-valence errors and that the projections
      reproduce them; the PySCF hydrogen numbers (variational energies, polarizability zero without p, TZVPD much larger than TZVP, equal TZVP and TZVPD
      densities, density tail at $r=5.3$ Å, 10 bohr); water energies falling with basis size.

    **Not checked here, or stated from memory.**

    - *Section 3b.* Only these two problems were computed, with one box shape (a cube, even functions only), nuclei at the centre, and s
      Gaussians optimised for the atom. The reasons offered for the result are explanations consistent with the figures, not separate tests,
      apart from the volume count. Statements from general knowledge, checked only against general reading and not against the original
      papers: that plane waves are standard for periodic solids, that pseudopotentials or projector-augmented waves are used because of the nuclear
      region, and the descriptions of basis-set superposition error and linear dependence.

    - The cusp condition $\psi'(0)=-Z\psi(0)$ and the tail $e^{-\sqrt{2\lvert E\rvert}r}$ in general; only the hydrogen 1s case is tested.
    - That Gaussian integrals are analytic in general (for higher angular momentum and several centres); only the
      s-type integrals at one centre are tested, and the product theorem.
    - The exact hydrogen polarizability $\tfrac92$ and the exact H$_2^+$ energy $-0.602634$ hartree at $R=1.06$ Å, 2 bohr (the latter agrees with the large-basis calculation within $3\times10^{-5}$).
    - How the `def2` sets were designed (Weigend and Ahlrichs) and what each letter promises in general; only the shell counts for
      water and hydrogen are data.
    - The PySCF numbers are stored in `data_basis_sets.json` and were not recomputed by the tests; the water energies are
      those of the DFT notebook, and the notebook shows them relative to `def2-QZVP`, not to the complete-basis limit.
    - The split-valence figure illustrates the idea with the three STO-3G Gaussians; it is not a model of `def2`.
    """)
    return


if __name__ == "__main__":
    app.run()
