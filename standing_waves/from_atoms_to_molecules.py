import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="From Atoms to Molecules: Fixed Nuclei, Hybrids, Methane")


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
        geometry_optimization_figure,
        hybrid_figure,
        methane_figure,
        optimisation_toy_figure,
        surface_uses_figure,
    )

    return (
        MASS_RANGE,
        STATE_MAX,
        born_oppenheimer_figure,
        geometry_optimization_figure,
        hybrid_figure,
        methane_figure,
        mo,
        mol,
        optimisation_toy_figure,
        plt,
        surface_uses_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # From Atoms to Molecules: Fixed Nuclei, Hybrids, Methane

    The hydrogen notebook solved one electron around one fixed nucleus exactly. A molecule has several electrons and
    several nuclei, and no exact solution. Chemistry gets by with three approximations that are made one after the other:
    **fixed nuclei** (Born-Oppenheimer), **one orbital per electron** (the orbital approximation), and **orbitals built from
    atom-centred functions** (LCAO). This notebook takes the first (section 2) and shows what the orbitals that the other two produce look like
    (hybrids and bond orbitals, sections 3 and 4). Here **two approximations** are worked out; the third, one orbital per electron, is
    the subject of the Hartree-Fock notebook (`hartree_fock_mean_field.py`), and which functions to put on the atoms is taken up in
    `basis_sets_from_base_states.py` (next). It is the bridge between hydrogen (`spherical_harmonics_and_hydrogen.py`) and the rest of the path.

    An **orbital** is a one-electron wave $\varphi(\mathbf r)$. The orbital approximation gives each electron its own orbital (two electrons, of opposite spin, to an orbital),
    each moving in the field of the nuclei plus the average cloud of the others; why this is a good idea, and what it costs, is in `hartree_fock_mean_field.py`.

    **Read first:** `spherical_harmonics_and_hydrogen.py` (s, p shapes and the real spherical harmonics). Section 2 plots
    the H$_2^+$ energy curve that `basis_sets_from_base_states.py`, section 3, computes with two basis functions; here it is
    used as given. Worked calculations that use these ideas are in `../../qm-calculations`: `carbon` (the carbon atom, its density and a p
    orbital), `methane` (its molecular orbitals and a C-H bond orbital, used in section 4) and `basis-functions` (the shapes of the basis
    functions on hydrogen and oxygen, as in the next notebook).

    **Contents**

    1. Why hydrogen's recipe stops
    2. Fixed nuclei: the Born-Oppenheimer approximation (2.1: how a geometry optimization uses it)
    3. Hybrid orbitals: how many $p$ in an $s$-$p$ mixture, and the angle
    4. Methane: one level, then three
    5. What was checked, and what was not

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
        - **Orbital**: a one-electron wave $\varphi(\mathbf r)$; the orbital approximation builds the many-electron wave from one orbital per electron (`hartree_fock_mean_field.py`).
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
    ### 2.1 How a geometry optimization uses Born-Oppenheimer

    A geometry optimization asks: for which nuclear positions $\mathbf R$ is the energy $E(\mathbf R)$ lowest? Born-Oppenheimer
    is what makes the question well posed, because it turns the energy into a function of the nuclear positions alone. The
    calculation is two loops, one inside the other. In the figure below, $\mathbf R$ stands for all nuclear positions at once, $E$ is the
    Born-Oppenheimer energy of section 2 (electrons plus nuclear repulsion), and $\mathbf F=-\partial E/\partial\mathbf R$ is the
    **force** on the nuclei: the negative slope of the energy surface, one vector per nucleus.

    **Inner loop: the electrons, at fixed $\mathbf R$.** The nuclei are fixed charges at $\mathbf R$, and the electrons are solved in their field by the
    **self-consistent field** (SCF) loop, which is described once, in `hartree_fock_mean_field.py`, section 7 (the orbitals and the field they make are iterated until they agree).
    It returns $E(\mathbf R)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Outer loop: the nuclei.** The SCF gives $E(\mathbf R)$. The force on each nucleus follows from the converged wave without a new SCF:
    the **Hellmann-Feynman** part is the force that the electron cloud exerts on the nucleus, as if the cloud were frozen; the **Pulay term**
    corrects it for the fact that the basis functions are centred on the atoms and move with them (it is zero for a complete basis).
    The optimiser then moves the nuclei downhill. It keeps an approximate matrix $H$ of second derivatives of $E$ (the Hessian), updated from the
    change in the forces from one step to the next (BFGS is the usual update), and takes the step $\Delta\mathbf R=-H^{-1}\mathbf g$ with
    $\mathbf g=-\mathbf F$ the gradient. Real programs write $\mathbf R$ in **internal coordinates** (bond lengths, angles, torsions), because $E$
    changes along these in a simpler way than along the Cartesian $x,y,z$. The loop stops when the largest and the root-mean-square force, the
    largest displacement and the change of energy are all below thresholds.

    The Born-Oppenheimer assumption sits in the box inside the dashed line: for the electrons, $\mathbf R_k$ is a set of fixed charges
    and nothing is said yet about how the nuclei move. The motion of the nuclei is then handled by the outer loop, as in the nuclear equation of section 2 but
    here without quantum waves: the optimiser only looks for the minimum of $E$ (the nuclear problem at zero temperature, in the classical limit).
    """)
    return


@app.cell
def _(geometry_optimization_figure, mo, plt):
    _fig = geometry_optimization_figure()
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    The same surface $E(\mathbf R)$ and the same forces serve three jobs. They differ in what they do with $\mathbf R$ after the SCF:

    - a **single point** stops after one SCF and reports the energy (and the forces, which cost little extra);
    - an **optimization** moves $\mathbf R$ downhill until the forces vanish;
    - **ab initio molecular dynamics** moves $\mathbf R$ by Newton's equation, $M\ddot{\mathbf R}=\mathbf F$ ($M$ the nuclear mass),
      and runs an SCF at every time step.

    Vibrational **frequencies** are taken from the second derivatives of $E$ at an optimized geometry.
    """)
    return


@app.cell
def _(mo, plt, surface_uses_figure):
    _fig = surface_uses_figure()
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A toy optimization: the H$_2^+$ bond length.** There is one coordinate, the proton separation $R$, so $H$ is a single number. Each
    step takes the energy and force from the H$_2^+$ curve of section 2 (standing in for one SCF), moves by $\Delta R=F/h$, and then
    replaces the approximate second derivative $h$ by the one measured from the last two forces (this is the BFGS update in one dimension).
    A step that would raise the energy is shortened, as a line search does in a real program. Choose the starting separation below.
    """)
    return


@app.cell
def _(mo):
    opt_start = mo.ui.slider(0.8, 2.0, step=0.1, value=1.6, label="starting $R$ (Å)", show_value=True)
    return (opt_start,)


@app.cell
def _(mo, mol, opt_start, optimisation_toy_figure, plt):
    _fig = optimisation_toy_figure(start=opt_start.value / mol.BOHR_ANGSTROM)
    _out = mo.vstack([opt_start, mo.as_html(_fig)])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo, mol):
    _run = mol.optimise_bond_length(1.6 / mol.BOHR_ANGSTROM)
    _rows = "".join(
        f"| {k} | {r * mol.BOHR_ANGSTROM:.3f} | {e:.5f} | {f:+.5f} | {h:.3f} |\n"
        for k, (r, e, f, h) in enumerate(zip(_run["r"], _run["energy"], _run["force"], _run["hessian"]))
    )
    mo.md(
        "Starting from $R=1.60$ Å, the steps are (force and second derivative in atomic units per bohr; $h$ is the "
        "approximate second derivative used for the next step):\n\n| step | $R$ (Å) | $E$ (hartree) | $F$ (hartree/bohr) | $h$ (hartree/bohr$^2$) |\n|---|---|---|---|---|\n"
        + _rows
        + "\nThe first step is a guess ($h=0.3$ is a typical starting value for a bond); the update then corrects $h$ towards the true second derivative "
        "(about $0.10$, which is what the vibrational frequency of section 2 measures), after which the steps are Newton steps and the energy falls "
        "quickly. A real molecule has $3N_{\\text{nuclei}}-6$ coordinates, so $H$ is a matrix and each SCF is much more expensive, but the loop is this one.\n"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Which calculations optimize the geometry?** The label of a dataset (`../dft/reading_a_dft_label.py`) says which method and basis were used,
    not whether the geometry was optimized; the two are separate choices. A dataset of energies and forces needs only single points, whereas
    a geometry for a force-field fit needs the loop above. Which of the datasets (SPICE, OMol25) and force-field jobs (OPLS5) optimize the geometry
    is tabulated next to their labels, in section 9 of that notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 Hybrid orbitals: how many $p$ in an $s$-$p$ mixture, and the angle

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
    written either way, and section 4 shows that for methane.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Methane: one level, then three

    **LCAO for a molecule.** Methane has a carbon atom with four hydrogens at the corners of a tetrahedron. Take as basis the
    carbon $2s$ and three $2p$ functions and one $1s$ function on each hydrogen: eight functions, so eight molecular orbitals
    $\psi=\sum_\mu c_\mu\chi_\mu$ (orbitals in the sense of `hartree_fock_mean_field.py`: each holds two electrons of opposite spin). The $1s$ core of carbon is left out; it is an orbital by itself and not involved in the
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
    3. **LCAO** (atom-centred functions; which ones to choose is the next notebook) gives molecular orbitals; the symmetry labels $a_1$ and $t_2$ say how
       many levels there are at what energies. Hybrids are a rewriting of occupied orbitals, and $1+n\cos\theta=0$ gives the bond angles
       $180^\circ,120^\circ,109.47^\circ$.

    **Next.** The third approximation, LCAO, was used here for methane with invented numbers. `basis_sets_from_base_states.py`
    asks what the atom-centred functions should be. After it, `hartree_fock_mean_field.py` justifies the second approximation, one orbital per electron
    in the average field of the others (helium, screening, the Fock operator), and the DFT notebooks then decode the `def2-TZVPD` label.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 What was checked, and what was not

    **Checked numerically** (`pixi run python -m pytest -q standing_waves/test_molecules.py`):

    - *Section 1.* The coordinate counts (6 for hydrogen, 45 for methane, 30 for its electrons with fixed nuclei).
    - *Section 2.* The fitted H$_2^+$ curve passes through the 14 PySCF points within $3\times10^{-4}$ hartree; its minimum is at
      $R=1.99$ bohr ($1.05$ Å) and $E=-0.6026$ hartree, as the lowest PySCF point; the harmonic wavenumber is 2312 cm$^{-1}$ and the spacing of the two lowest
      nuclear levels 2189 cm$^{-1}$ (smaller than harmonic, as for an anharmonic well); the nuclear levels have $0,1,2,3$ nodes, are
      normalised, and the spacing scales as $1/\sqrt{\text{mass}}$; vibration quantum over electron binding energy is $0.097$.
    - *Section 2.1.* The force $F=-dE/dR$ against a finite difference of the fitted curve; the toy optimization reaches the minimum
      of the curve ($R=1.99$ bohr) from four starting separations, never raises the energy, ends with $|F|<10^{-4}$ hartree/bohr in at most 12 SCFs, and stops at once if it starts at the minimum.
    - *Section 3.* The angles $180^\circ,120^\circ,109.47^\circ$ and the tetrahedral angle between corner directions; the four sp$^3$ coefficient rows are
      orthonormal, each with $s$ weight $\tfrac14$; the same orthonormality by numerical integration over the sphere with the real $s$ and $p$ functions;
      the hybrid profile is normalised to 1 and largest along its axis.
    - *Section 4.* The eight toy levels equal the eigenvalues of the two-by-two blocks (one $a_1$ pair, three identical $t_2$ pairs) for three values of the
      coupling; the four bond orbitals are orthonormal, span the occupied space exactly, put the largest weight on their own hydrogen and point
      along the corners; the carbon $s{:}p$ ratio is not exactly $1{:}3$.

    The helium, screening and orbital-size checks that used to be here are in `hartree_fock_mean_field.py`, section 9.

    **Not checked here, or stated from memory.**

    - *Section 2.1.* The flowchart is a textbook description (not computed): the Hellmann-Feynman and Pulay terms, BFGS, internal coordinates and the
      convergence criteria are stated from general knowledge. The table of which datasets optimize the geometry is in `../dft/reading_a_dft_label.py`, sections 9 and 10.
    - The numbers of the real methane calculation (orbital energies, the $20/76/4$ composition) are copied from
      `qm-calculations/methane/methane_output.txt`, not recomputed here.
    - The vibrational numbers for H$_2^+$ can be compared with experiment ($\omega_e\approx2321$ cm$^{-1}$, fundamental $\approx2191$ cm$^{-1}$, from
      memory), not tested.
    - The toy for methane uses invented numbers, shows the pattern only, and ignores overlap between functions and any hydrogen-hydrogen coupling, which is why its
      bond orbitals are cleaner (almost no weight on other hydrogens, 0.15 %) than those of the real calculation (12 %).
    - The failure of Born-Oppenheimer near surfaces and at crossings of electronic states is stated, not shown.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    Free sources, checked on 2026-11-13 (the full table, with what each covers, is in `SOURCES.md`).

    - [The H$_2^+$ prototypical species (LibreTexts, *Physical Chemistry* 9.2)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_%28LibreTexts%29/09%3A_Chemical_Bonding_in_Diatomic_Molecules/9.02%3A_The_H_Prototypical_Species): Born-Oppenheimer and LCAO for H$_2^+$ (intro).
    - [sp$^3$ hybrid orbitals and methane (LibreTexts, *Organic Chemistry* 1.6)](https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_%28Morsch_et_al.%29/01:_Structure_and_Bonding/1.06:_sp_Hybrid_Orbitals_and_the_Structure_of_Methane): the textbook hybrid picture (intro).
    - [Methane as molecular orbitals (LibreTexts, Rioux, 3.18)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Quantum_Tutorials_%28Rioux%29/03:_Chemical_Bonding/3.18:_A_Molecular_Orbital_Approach_to_Bonding_in_Methane) and [ligand group orbitals, the symmetry-adapted combinations (LibreTexts, Ursinus College, 5.4.1)](https://chem.libretexts.org/Courses/Ursinus_College/CHEM322:_Inorganic_Chemistry/05:_Molecular_Orbitals/5.04:_Polyatomic_MO_Diagrams/5.4.01:_Ligand_Group_Orbitals_and_Generator_Functions) (intro to derivation).
    """)
    return


if __name__ == "__main__":
    app.run()
