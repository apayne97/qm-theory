import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="From Atoms to Molecules: Three Approximations")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import molecules as mol
    from molecules_plots import (
        MASS_RANGE,
        STATE_MAX,
        born_oppenheimer_figure,
        helium_figure,
        hybrid_figure,
        methane_figure,
        orbital_size_figure,
        screening_figure,
    )

    return (
        MASS_RANGE,
        STATE_MAX,
        born_oppenheimer_figure,
        helium_figure,
        hybrid_figure,
        methane_figure,
        mo,
        mol,
        orbital_size_figure,
        plt,
        screening_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # From Atoms to Molecules: Three Approximations

    The hydrogen notebook solved one electron around one fixed nucleus exactly. A molecule has several electrons and
    several nuclei, and no exact solution. Chemistry gets by with three approximations that are made one after the other:
    **fixed nuclei** (Born-Oppenheimer), **one orbital per electron** (the orbital approximation), and **orbitals built from
    atom-centred functions** (LCAO). This notebook shows what each one assumes, what it costs, and how the resulting orbitals
    (hybrids, bonds) look. It is the bridge between hydrogen (`spherical_harmonics_and_hydrogen.py`) and the rest of the path:
    the third approximation, which functions to put on the atoms, is taken up in `basis_sets_from_base_states.py` (next),
    and the Hartree-Fock and DFT notebooks (`../dft/`) follow after that.

    **Read first:** `spherical_harmonics_and_hydrogen.py` (s, p shapes and the real spherical harmonics). Section 2 plots
    the H$_2^+$ energy curve that `basis_sets_from_base_states.py`, section 3, computes with two basis functions; here it is
    used as given. Worked calculations that use these ideas are in `../../qm-calculations`: `carbon` (the carbon atom, its density and a p
    orbital), `methane` (its molecular orbitals and a C-H bond orbital, used in section 5) and `basis-functions` (the shapes of the basis
    functions on hydrogen and oxygen, as in the next notebook).

    **Contents**

    1. Why hydrogen's recipe stops
    2. Fixed nuclei: the Born-Oppenheimer approximation
    3. One orbital per electron: helium, and why $2s$ lies below $2p$
    4. Hybrid orbitals: how many $p$ in an $s$-$p$ mixture, and the angle
    5. Methane: one level, then three
    6. What was checked, and what was not

    Atomic units throughout, as in `spherical_harmonics_and_hydrogen.py` and every later notebook: lengths in bohr (1 bohr $=0.529$ Å; the figures are drawn in Å), energies in
    hartree (1 hartree $=27.2$ eV), $\hbar=m_e=1$. Energies of the methane toy are in eV. Everything is computed in `molecules.py` and tested in `test_molecules.py`; the H$_2^+$ energies
    are the PySCF numbers of `data_basis_sets.json`.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Born-Oppenheimer approximation**: solve for the electrons with the nuclei held at fixed positions, then let the
          nuclei move in the energy surface that results.
        - **Orbital**: a one-electron wave $\varphi(\mathbf r)$. The orbital approximation writes the many-electron wave as a
          product (made antisymmetric in `../dft/reading_a_dft_label.py`, section 2) of one orbital per electron.
        - **Mean field**: each electron feels the nucleus plus the *average* charge cloud of the others, not their instantaneous positions.
        - **Screening**: the pull of the nucleus on an outer electron is reduced by the inner electrons.
        - **LCAO**: linear combination of atom-centred functions, $\psi=\sum_\mu c_\mu\chi_\mu$.
        - **Hybrid orbital**: a mixture of $s$ and $p$ functions on one atom that points in one direction.
        - **Symmetry labels** $a_1$, $t_2$: the names of the levels of a tetrahedral molecule; $a_1$ is one function, $t_2$ is three that
          always share an energy.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 Why hydrogen's recipe stops

    In the hydrogen notebook, the wave depends on three numbers (one position), the potential depends on the distance $r$ only,
    and the variables separate into a radial and an angular problem. A molecule breaks this in three ways.

    - **Every particle adds three coordinates.** The wave $\Psi$ depends on the positions of all nuclei and all electrons at once.
    - **The electrons push on each other** through $1/\lvert\mathbf r_i-\mathbf r_j\rvert$, which ties their positions together.
    - **There is no single centre.** The pull of several nuclei has no spherical symmetry around any point.

    The size of the first problem is easy to count. A table of the wave on a grid with ten points per coordinate needs
    $10^{N}$ numbers for $N$ coordinates.
    """)
    return


@app.cell
def _(mo, mol):
    _rows = [("hydrogen atom", 1, 1), ("helium atom", 1, 2), ("H$_2^+$", 2, 1), ("water", 3, 10), ("methane", 5, 10), ("caffeine (C$_8$H$_{10}$N$_4$O$_2$)", 24, 102)]
    _table = "| system | nuclei | electrons | coordinates $N$ | grid values $10^N$ |\n|---|---|---|---|---|\n"
    for _name, _nuc, _el in _rows:
        _n = mol.coordinate_count(_nuc, _el)
        _table += f"| {_name} | {_nuc} | {_el} | {_n} | $10^{{{_n}}}$ |\n"
    mo.md(_table + r"""
    For comparison, the visible universe holds about $10^{80}$ atoms. So nobody stores the wave of methane on a grid, and the
    approximations below are what make a calculation possible at all. Each one removes coordinates or couples them in a
    simpler way.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 Fixed nuclei: the Born-Oppenheimer approximation

    A proton is 1836 times heavier than an electron. For the same forces, the nuclei therefore move far more slowly, and the
    electrons have time to settle into their lowest arrangement for each position of the nuclei. The approximation does two
    things:

    1. Freeze the nuclei at positions $\mathbf R$ and solve the electronic problem. This gives an energy
       $E_{\text{el}}(\mathbf R)$ plus the nuclear repulsion, together a function of the nuclear positions only.
    2. Treat that function as the potential for the nuclei and solve their (much slower) motion in it.

    This replaces one problem in $N$ coordinates by a family of electronic problems (the nuclei are numbers now, not
    coordinates), and then by a small problem for the nuclei. **The potential energy surface of every force field and every
    geometry optimisation is the Born-Oppenheimer energy.** The simplest example is H$_2^+$ (one electron, two protons a distance
    $R$ apart), whose energy in a very large basis is the set of points in the figure.

    The nuclear problem is a one-dimensional standing wave problem, like the particle in a box of Part 1, with the
    potential of the figure and the reduced mass $\mu=M_p/2$:

    $$
    \left[-\frac{1}{2\mu}\frac{d^2}{dR^2}+E(R)\right]\chi(R)=E_{\text{tot}}\,\chi(R).
    $$

    The levels have $0,1,2,\dots$ nodes, and their spacing is the **vibrational frequency** of the bond. Use the sliders to
    pick a level and to change the nuclear mass.
    """)
    return


@app.cell
def _(MASS_RANGE, STATE_MAX, mo):
    bo_state = mo.ui.slider(0, STATE_MAX, step=1, value=0, label="nuclear level", show_value=True)
    bo_mass = mo.ui.slider(MASS_RANGE[0], MASS_RANGE[1], step=0.25, value=1.0, label="nuclear mass (real = 1)", show_value=True)
    return bo_mass, bo_state


@app.cell
def _(bo_mass, bo_state, born_oppenheimer_figure, mo, plt):
    _fig = born_oppenheimer_figure(bo_state.value, bo_mass.value)
    _out = mo.vstack([mo.hstack([bo_state, bo_mass], justify="start"), _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the numbers say.**

    - The minimum of the curve is at $R\approx 1.06$ Å ($2.0$ bohr) with a binding energy of about 2.8 eV below H + proton.
    - The two lowest nuclear levels are $0.0100$ hartree $=0.27$ eV apart ($\approx 2190\ \text{cm}^{-1}$). This is about a tenth of the
      electron's binding energy, which is what "the electrons are fast" means in numbers.
    - The spacing falls as $1/\sqrt{M}$: lighter nuclei vibrate faster (the orange line, which is a straight line of slope $-\tfrac12$
      on these log axes). If the nuclei were four times lighter, the spacing would double, to a fifth of the electron's binding energy; at some mass the
      split into "electrons first, nuclei second" stops being a good idea. That is where the approximation breaks down,
      for example for hydrogen atoms moving near a metal surface or for processes in which several electronic states come together.
    - Replacing the proton by a heavier nucleus (deuteron: factor 2) lowers the vibrational frequency by $1/\sqrt2$, but leaves the
      curve unchanged. This is why isotope effects shift frequencies and rates and not the equilibrium geometry (to this
      order).

    The calculations in the rest of the talk, wB97M-V energies and geometries from `def2` basis sets, are all Born-Oppenheimer
    calculations: $\mathbf R$ is an input.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 One orbital per electron: helium, and why $2s$ lies below $2p$

    Fixed nuclei leave the electron-electron term. For helium the exact wave depends on the two positions
    $\mathbf r_1,\mathbf r_2$ jointly. The **orbital approximation** takes a product of one-electron waves,

    $$
    \Psi(\mathbf r_1,\mathbf r_2)\approx\varphi(\mathbf r_1)\,\varphi(\mathbf r_2),
    $$

    so that each electron moves in the field of the nucleus plus the *average* cloud of the other. (Two electrons may share the
    same spatial orbital only if their spins are opposite; this is the Pauli principle, and `reading_a_dft_label.py`, section 2,
    makes the product antisymmetric in general.)

    Take the hydrogen-like function $\varphi\propto e^{-\zeta r}$ and let $\zeta$ be the one free number. The energy in
    hartree is a sum of three terms,

    $$
    E(\zeta)=\underbrace{\zeta^{2}}_{\text{kinetic}}\;\underbrace{-\,2Z\zeta}_{\text{attraction to the nucleus}}\;
    {}+\underbrace{\tfrac58\zeta}_{\text{repulsion of the two electrons}},\qquad Z=2 .
    $$

    Without the last term the minimum is at $\zeta=Z=2$ (the helium ion's orbital, twice over) and the energy is $-4$ hartree.
    With it, the minimum moves to $\zeta=Z-\tfrac5{16}=1.6875$ and the energy is $-2.8477$ hartree. The repulsion works
    as a **screening**: each electron sees a nucleus whose charge is reduced by a bit under one third of an electron
    charge, because the other electron sits partly between it and the nucleus. The orbital swells (right panel).
    """)
    return


@app.cell
def _(mo):
    zeta = mo.ui.slider(1.0, 2.6, step=0.025, value=1.6875, label=r"$\zeta$", show_value=True)
    return (zeta,)


@app.cell
def _(helium_figure, mo, plt, zeta):
    _fig = helium_figure(zeta.value)
    _out = mo.vstack([zeta, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.accordion({
        "Details: where the three terms come from": mo.md(r"""
        For $\varphi=\sqrt{\zeta^3/\pi}\,e^{-\zeta r}$ the kinetic energy of one electron is $\tfrac12\zeta^2$ and the attraction to
        a nucleus of charge $Z$ is $-Z\zeta$ (both are the hydrogen-like values with $Z\to\zeta$ and the pull $Z$); two electrons double
        them. The repulsion $\langle 1/r_{12}\rangle$ for two electrons in this orbital is $\tfrac58\zeta$, which is checked by direct
        integration in `test_molecules.py`. Setting $dE/d\zeta=2\zeta-2Z+\tfrac58=0$ gives $\zeta=Z-\tfrac5{16}$ and
        $E=-(Z-\tfrac5{16})^2$.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How good is it?** Against the exact non-relativistic energy of helium, $-2.9037$ hartree:

    | wave function | energy (hartree) |
    |---|---|
    | no repulsion between the electrons | $-4$ (not an approximation of anything: the term is missing) |
    | one orbital, bare nucleus, $\zeta=2$ | $-2.75$ |
    | one orbital with the best $\zeta=1.6875$ | $-2.8477$ |
    | best possible orbital product (Hartree-Fock limit) | $-2.8617$ |
    | exact | $-2.9037$ |

    The one-orbital picture recovers 98 % of the exact energy, but the missing $0.042$ hartree ($1.1$ eV) is larger than most
    chemical energy differences. That gap is the **correlation energy**: the electrons dodge each other, which no
    product of orbitals can express. Hartree-Fock stops at the orbital product; DFT and coupled cluster (Parts 4 to 6) are
    different ways of recovering some or all of the gap.

    **Why $2s$ is lower than $2p$.** In hydrogen the energy depends on $n$ only (section 5 of the hydrogen notebook), because the
    potential is exactly $-1/r$. In an atom with several electrons, an inner electron screens the nucleus, so the charge
    seen by an outer electron is large close in and small far out. The toy below models an atom with a filled $1s$ shell: the
    effective charge is $Z_{\text{eff}}(r)=1+2e^{-r/a}$ (nucleus 3, minus the two $1s$ electrons seen from outside), and $a$ is the
    distance over which the core screens. The radial equation of the hydrogen notebook is solved with this potential.
    """)
    return


@app.cell
def _(mo):
    decay = mo.ui.slider(0.05, 0.8, step=0.05, value=0.25, label=r"screening distance $a$ (Å)", show_value=True)
    return (decay,)


@app.cell
def _(decay, mo, plt, screening_figure):
    _fig = screening_figure(decay.value)
    _out = mo.vstack([decay, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    An $s$ wave is large near the nucleus (it has no centrifugal barrier), so it lies partly *inside* the screening cloud and feels the
    strong pull there. A $p$ wave is kept away from the nucleus by the barrier $\ell(\ell+1)/2r^2$, feels the screened charge, and lies
    higher. The order of each shell is $s<p<d$ at the same $n$, and the splitting grows with the screening distance. This is the origin of the
    familiar filling order $1s,2s,2p,3s,3p,4s,3d\dots$ (the $4s$ before $3d$ is the same effect, a bit stronger). The toy
    potential has one parameter, set by hand rather than fitted to any atom, so only the ordering is a result.

    For the slides: in hydrogen the shapes ($s$, $p$, $d$) *and* the degeneracy in $\ell$ come from the same function; in an atom with
    several electrons the shapes survive, and the degeneracy in $\ell$ does not.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How big is an orbital, and what does a heavier nucleus do to it?** The hydrogen orbitals of
    `spherical_harmonics_and_hydrogen.py` are large. One bohr is $0.529$ Å, and the mean distance of the electron from the nucleus is

    $$
    \langle r\rangle=\tfrac12\bigl[3n^{2}-\ell(\ell+1)\bigr]\ \text{bohr},
    $$

    where $n$ is the principal quantum number and $\ell$ the orbital angular momentum. For $1s$ this is $0.79$ Å ($1.5$ bohr); for $3s$ it is
    $7.1$ Å ($13.5$ bohr), with $99\,\%$ of the probability inside $13.5$ Å. The size grows as $n^{2}$, which is why the $3s$ plot in the hydrogen notebook
    reaches so far out. A molecule is a few Å across, so a hydrogen $3s$ orbital would not fit in it.

    **A larger nuclear charge $Z$ squeezes the orbitals.** For one electron around a nucleus of charge $Z$ the pull is $-Z/r$ instead of
    $-1/r$. Replacing $r$ by $Zr$ turns the radial equation back into hydrogen's, so the shapes are the same and only the scales change:

    $$
    \langle r\rangle=\frac{3n^{2}-\ell(\ell+1)}{2Z}\ \text{bohr},\qquad E_n=-\frac{Z^{2}}{2n^{2}}\ \text{hartree}.
    $$

    The radius falls as $1/Z$ and the energy rises in size as $Z^{2}$. The $1s$ orbital of the helium ion He$^{+}$ ($Z=2$) is half as large as hydrogen's
    and four times as tightly bound; this is the $\zeta=Z=2$ orbital of the helium discussion above. For uranium ($Z=92$) the $1s$ orbital would have $\langle r\rangle=0.016$ bohr $=0.0086$ Å.

    **Why atoms do not shrink to a point, or grow without limit.** Neither extreme happens, because the other electrons screen the
    nucleus (the effect of $\zeta=Z-\tfrac5{16}$ above, and of $Z_{\text{eff}}(r)$ in the toy atom). An outer electron sees the nucleus
    plus the inner electrons, a net charge $Z_{\text{eff}}$ that is only a little above $1$ for Li and Na and grows by roughly $0.6$ to $0.7$ for each
    step along a row of the periodic table. The orbital size is then roughly $n^{2}/Z_{\text{eff}}$: a larger $n$ makes it larger, a larger $Z_{\text{eff}}$
    makes it smaller, and the two partly cancel. The result is that the outermost orbitals of the atoms from H to Ar have mean radii
    between about $0.5$ and $2.2$ Å, instead of the $7$ Å of a hydrogen $3s$ orbital. Use the slider to change $Z$ in the left panel; the right panel shows the
    computed sizes of the neutral atoms.
    """)
    return


@app.cell
def _(mo):
    charge = mo.ui.slider(1, 18, step=1, value=1, label=r"nuclear charge $Z$", show_value=True)
    return (charge,)


@app.cell
def _(charge, mo, orbital_size_figure, plt):
    _fig = orbital_size_figure(charge.value)
    _out = mo.vstack([charge, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left:* the radial probability of the $1s$, $3s$ and $3d$ orbitals of a hydrogen-like ion (one electron, nuclear charge $Z$): each curve is
    hydrogen's curve (dashed) squeezed towards the nucleus by the factor $Z$. *Right:* the mean distance $\langle r\rangle$ of the outermost occupied orbital of the
    neutral atoms H to Ar, from Hartree-Fock calculations with the `def2-TZVP` basis set (`compute_atom_sizes.py`, a few seconds of PySCF);
    the circle marks the atom with $Z$ from the slider. The grey lines show what a bare nucleus ($Z_{\text{eff}}=Z$) would give for $n=1,2,3$; the atoms lie well above them, the
    amount of screening. Each new row (Li, Na) starts a new, larger shell, and within a row the orbital shrinks from Li ($2.0$ Å) to Ne ($0.51$ Å)
    as $Z_{\text{eff}}$ rises. A noble gas has the smallest orbital of its row.

    **Where this lives in a DFT calculation.** Nothing about screening is put in by hand. The Kohn-Sham operator
    (`../dft/kohn_sham_matrix.py`) for each electron contains the full attraction $-Z_A/r_A$ to every nucleus $A$, plus the Hartree
    potential (the average repulsion of all the electrons, which gives the screening) and the exchange-correlation potential (the
    rest of the electron-electron effect). The self-consistent loop starts from a guess for the orbitals, builds these potentials,
    finds new orbitals in them, and repeats; the shrunken, screened orbitals are what it converges to. What the user chooses is only
    the **basis set**, which has to be able to represent what comes out:

    - **Exponents per element.** A Gaussian $e^{-\alpha r^{2}}$ has a size $\sim\alpha^{-1/2}$. Each element has its own exponents, small for the diffuse valence
      orbital and large for the tight inner ones, as for the Slater exponent $\zeta$ of `basis_sets_from_base_states.py`.
    - **Tight core functions for heavy atoms.** The $1s$ orbital shrinks as $1/Z$ and needs exponents that grow as $Z^{2}$ (a $2\times$ smaller orbital needs
      a $4\times$ larger $\alpha$). In `def2-TZVP` the tightest Gaussian has $\alpha=34$ on hydrogen and about $7\times10^{4}$ on chlorine.
    - **Effective core potentials** for the heaviest elements. The core electrons of, for example, platinum (in the OPLS5 labels of
      `../dft/reading_a_dft_label.py`) are both numerous and very compact; they are replaced by a fixed potential that acts on the
      valence electrons, so that only the valence orbitals, with sizes like those in the right panel, are in the basis. (`def2-TZVP` does this for platinum:
      its tightest Gaussian is $\alpha=30$, not tens of thousands.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Hybrid orbitals: how many $p$ in an $s$-$p$ mixture, and the angle

    In a molecule an atom's orbitals are rearranged by its neighbours. A carbon atom bonded to four others has four
    equivalent bonds, but the atom's own $s$ and $p$ orbitals are not equivalent. Any mixture of orbitals of one atom is as
    legitimate as the originals (the energy of the whole set does not change), so one may choose mixtures that point to the
    neighbours. The simplest family mixes one $s$ with a $p$ pointing in direction $\hat n$:

    $$
    h_{\hat n}=\frac{s+\sqrt{n}\,p_{\hat n}}{\sqrt{1+n}},\qquad
    h(\gamma)\propto\frac{1+\sqrt{3n}\cos\gamma}{\sqrt{1+n}},
    $$

    where $\gamma$ is the angle from $\hat n$, $p_{\hat n}$ is the real $p$ function along $\hat n$, and the second formula uses the real
    harmonics of the hydrogen notebook ($s=1/\sqrt{4\pi}$, $p=\sqrt{3}\cos\gamma/\sqrt{4\pi}$). The mixture is large (positive) in front, where $s$ and $p$ add, and small
    (negative) behind. The number $n$ is how many times more $p$ than $s$ (as weight): $n=1,2,3$ gives sp, sp$^2$ and sp$^3$.

    **The angle comes from orthogonality.** Two such hybrids pointing along $\hat n_a$ and $\hat n_b$ at an angle $\theta$
    overlap by $\langle h_a\vert h_b\rangle=(1+n\cos\theta)/(1+n)$, since $\langle p_a\vert p_b\rangle=\cos\theta$ and $\langle s\vert p\rangle=0$. Orbitals of one atom
    must be orthogonal (they are different states of one electron), so

    $$
    1+n\cos\theta=0\quad\Longrightarrow\quad\theta=\arccos\!\left(-\frac1n\right)=180^\circ,\ 120^\circ,\ 109.47^\circ\quad\text{for }n=1,2,3.
    $$

    These are the angles of acetylene, ethene and methane. Four sp$^3$ hybrids, built from the $s$ and three $p$ with the coefficients
    $h_i=\tfrac12\bigl(s+\sqrt3\,\hat n_i\cdot\mathbf p\bigr)$ and $\hat n_i$ the corners of a tetrahedron, are orthonormal, and each has a quarter of
    $s$ character.
    """)
    return


@app.cell
def _(hybrid_figure, mo, plt):
    _fig = hybrid_figure()
    _out = mo.vstack([_fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    Hybrids are a **choice of description**, not a physical effect that happens to an atom: the same set of occupied orbitals can be
    written either way, and section 5 shows that for methane.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Methane: one level, then three

    **LCAO for a molecule.** Methane has a carbon atom with four hydrogens at the corners of a tetrahedron. Take as basis the
    carbon $2s$ and three $2p$ functions and one $1s$ function on each hydrogen: eight functions, so eight molecular orbitals
    $\psi=\sum_\mu c_\mu\chi_\mu$. The $1s$ core of carbon is left out; it is an orbital by itself and not involved in the
    bonding. To see the pattern without a calculation, build the simplest possible Hamiltonian: one energy for each kind of
    function (C $2s$, C $2p$, H $1s$), and one coupling strength $c$ between a function on carbon and a hydrogen $1s$ function. The coupling of
    a carbon $p$ function to a hydrogen is proportional to the cosine of the angle between the $p$ direction and the C-H direction. The
    numbers are illustrative; the **pattern does not depend on them**.

    Symmetry reduces the $8\times8$ problem to two-by-two ones:

    - the carbon $2s$ couples only to the sum of the four $1s$ functions: a pair ($a_1$) with energies
      $E_\pm=\tfrac12(\varepsilon_s+\varepsilon_H)\pm\sqrt{\tfrac14(\varepsilon_s-\varepsilon_H)^2+4c^2}$;
    - each carbon $2p_i$ couples only to the one combination of hydrogens that points the same way: three identical pairs ($t_2$),
      with the same formula for $\varepsilon_p$.

    The four lower levels are bonding (one $a_1$, three $t_2$ sharing an energy) and the four upper levels are antibonding. Methane has
    $4+4=8$ valence electrons, which fill the four bonding levels.
    """)
    return


@app.cell
def _(mo):
    bond = mo.ui.slider(0, 3, step=1, value=0, label="bond orbital (toward H1...H4)", show_value=True)
    return (bond,)


@app.cell
def _(bond, methane_figure, mo, plt):
    _fig = methane_figure(bond.value)
    _out = mo.vstack([bond, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A calculation of the real molecule has the same pattern.** The calculation in `../../qm-calculations/methane`
    (wB97M-V, `def2-TZVPD`, 73 functions) gives these occupied orbital energies in hartree:

    | orbital | energy |
    |---|---|
    | C $1s$ (core) | $-10.319$ |
    | $a_1$ | $-0.813$ |
    | $t_2$ (three, equal) | $-0.493$ |

    One level and then three degenerate ones, as in the toy. The canonical orbitals are spread over the whole molecule. To see
    four equivalent C-H bonds, mix them: the four combinations $b_j=\tfrac12\bigl(a_1+\sqrt3\,\hat n_j\cdot\mathbf t_2\bigr)$ (with $\mathbf t_2=(t_{2x},t_{2y},t_{2z})$ the three degenerate orbitals
    that transform like $x,y,z$) span exactly the same space and therefore the same electron density. The weight of $b_j$ is largest on
    hydrogen $j$ (right panel) and its carbon part points along the C-H direction. Each bond orbital is one quarter $a_1$ and three
    quarters $t_2$ by construction; the real calculation gives the same $0.25$ and $0.75$.

    **How much of the hybrid picture survives?** On carbon the real bond orbital is $20\,\%$ $s$, $76\,\%$ $p$ and $4\,\%$ $d$
    (sp$^3$ would be 25 : 75 : 0), and its carbon $p$ part points exactly at the hydrogen (angle $0.00^\circ$). The toy has $33\,\%$ $s$ and $67\,\%$ $p$ in the carbon part of its bond orbital, an $s{:}p$ ratio of $1{:}2$, again not exactly sp$^3$,
    because the $a_1$ and $t_2$ levels mix the carbon functions by different amounts. The textbook sp$^3$ numbers are the symmetric ideal; real
    bonds sit near it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways for a slide**

    1. A molecule has $3(N_{\text{nuclei}}+N_{\text{electrons}})$ coordinates and no closed-form solution: methane has 45, caffeine 378.
    2. **Born-Oppenheimer:** fix the nuclei, solve the electrons, and let the energy as a function of the positions be the potential for the
       nuclei. The nuclear problem is another standing-wave problem; its level spacing is the vibration. It works because the
       electrons are fast (here by a factor of about ten in energy).
    3. **Orbital approximation:** one orbital per electron in the mean field of the others. In helium it recovers 98 % of the energy; the rest is
       the correlation energy, the target of DFT and coupled cluster. Screening splits $s<p<d$ at the same $n$.
    4. **LCAO** (atom-centred functions; which ones to choose is the next notebook) gives molecular orbitals; the symmetry labels $a_1$ and $t_2$ say how
       many levels there are at what energies. Hybrids are a rewriting of occupied orbitals, and $1+n\cos\theta=0$ gives the bond angles
       $180^\circ,120^\circ,109.47^\circ$.

    **Next.** The third approximation, LCAO, was used here for methane with invented numbers. `basis_sets_from_base_states.py`
    asks what the atom-centred functions should be, and ends at the `def2-TZVPD` label that the DFT notebooks decode.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 What was checked, and what was not

    **Checked numerically** (`pixi run python -m pytest -q standing_waves/test_molecules.py`):

    - *Section 1.* The coordinate counts (6 for hydrogen, 45 for methane, 30 for its electrons with fixed nuclei).
    - *Section 2.* The fitted H$_2^+$ curve passes through the 14 PySCF points within $3\times10^{-4}$ hartree; its minimum is at
      $R=1.99$ bohr ($1.05$ Å) and $E=-0.6026$ hartree, as the lowest PySCF point; the harmonic wavenumber is 2312 cm$^{-1}$ and the spacing of the two lowest
      nuclear levels 2189 cm$^{-1}$ (smaller than harmonic, as for an anharmonic well); the nuclear levels have $0,1,2,3$ nodes, are
      normalised, and the spacing scales as $1/\sqrt{\text{mass}}$; vibration quantum over electron binding energy is $0.097$.
    - *Section 3.* The minimum of $E(\zeta)$ is at $\zeta=1.6875$ with $E=-2.84766$; the term $\tfrac58\zeta$ against a direct numerical
      integration of $\langle1/r_{12}\rangle$ (agreement to 0.2 %); the ordering of exact $-2.9037$ < HF limit $-2.8617$ < the one-orbital $-2.8477$; the
      screened-potential levels: $2s=2p$ when the screening disappears (and $=-0.125$ hartree), and $s<p<d$ at each $n$ otherwise.
      Orbital size: $\langle r\rangle=\tfrac12[3n^2-\ell(\ell+1)]/Z$ against numerical integration of the hydrogen-like radial density (six shells, $Z$ up to 18), the $1/Z$ scaling and normalisation of the squeezed density, and $13.5$ bohr $=7.14$ Å for $3s$. The H to Ar sizes are Hartree-Fock numbers read from `data_atom_sizes.json` (tests check hydrogen's $1.5$ bohr, the range, and the trend across row 2); the $99\,\%$ radius of $3s$ ($13.5$ Å) was computed once by integration and is not in a test.
    - *Section 4.* The angles $180^\circ,120^\circ,109.47^\circ$ and the tetrahedral angle between corner directions; the four sp$^3$ coefficient rows are
      orthonormal, each with $s$ weight $\tfrac14$; the same orthonormality by numerical integration over the sphere with the real $s$ and $p$ functions;
      the hybrid profile is normalised to 1 and largest along its axis.
    - *Section 5.* The eight toy levels equal the eigenvalues of the two-by-two blocks (one $a_1$ pair, three identical $t_2$ pairs) for three values of the
      coupling; the four bond orbitals are orthonormal, span the occupied space exactly, put the largest weight on their own hydrogen and point
      along the corners; the carbon $s{:}p$ ratio is not exactly $1{:}3$.

    **Not checked here, or stated from memory.**

    - The numbers of the real methane calculation (orbital energies, the $20/76/4$ composition) are copied from
      `qm-calculations/methane/methane_output.txt`, not recomputed here.
    - The helium values $-2.9037$ and $-2.8617$ hartree (exact and Hartree-Fock limit) are literature values, not computed.
    - The vibrational numbers for H$_2^+$ can be compared with experiment ($\omega_e\approx2321$ cm$^{-1}$, fundamental $\approx2191$ cm$^{-1}$, from
      memory), not tested.
    - That the $4s$ before $3d$ filling order is "the same effect"; the screening toy has only one parameter and shows only the
      ordering.
    - The toy for methane uses invented numbers, shows the pattern only, and ignores overlap between functions and any hydrogen-hydrogen coupling, which is why its
      bond orbitals are cleaner (almost no weight on other hydrogens, 0.15 %) than those of the real calculation (12 %).
    - The failure of Born-Oppenheimer near surfaces and at crossings of electronic states is stated, not shown.
    """)
    return


if __name__ == "__main__":
    app.run()
