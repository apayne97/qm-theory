import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Hartree-Fock: Each Electron in the Field of the Others")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import molecules as mol
    from molecules_plots import (
        average_field_figure,
        helium_figure,
        orbital_size_figure,
        scf_loop_figure,
        screening_figure,
    )

    return (
        average_field_figure,
        helium_figure,
        mo,
        mol,
        orbital_size_figure,
        plt,
        scf_loop_figure,
        screening_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Hartree-Fock: Each Electron in the Field of the Others

    The previous notebooks fixed the nuclei (Born-Oppenheimer, `from_atoms_to_molecules.py`) and chose the functions that the
    orbitals are built from (`basis_sets_from_base_states.py`). What is left is the electron-electron repulsion
    $1/\lvert\mathbf r_i-\mathbf r_j\rvert$, which ties the positions of all the electrons together. This notebook is the third
    approximation of chemistry, **one orbital per electron**, carried through to a method: **Hartree-Fock** (HF). It follows the chain of
    questions of C. D. Sherrill's *An Introduction to Hartree-Fock Molecular Orbital Theory* (Georgia Tech, 2000; see *Further reading*), in our own words and
    notation, with our own examples added: *what problem are we solving* (section 1), *why a product of orbitals* (2), what it is as an average field (3 and 4),
    *why the product fails* and what replaces it (5), *what is the energy* (6), and *which orbitals minimise it*, the Fock operator, the matrix form and the loop that solves it (7).
    Density functional theory (the next notebooks) keeps this structure and changes one term.

    **Read first:** `from_atoms_to_molecules.py` (section 1 on why the exact wave cannot be written down, and the
    Born-Oppenheimer approximation) and `basis_sets_from_base_states.py`, sections 1 and 3 (the variational principle, and the
    matrix equation $\mathbf H\mathbf c=E\,\mathbf S\mathbf c$ for the coefficients). Atomic units, as before: lengths in bohr
    (1 bohr $=0.529$ Å; the figures are drawn in Å), energies in hartree (1 hartree $=27.2$ eV), $\hbar=m_e=1$. The numerics are in
    `molecules.py` and are tested in `test_molecules.py`.

    **Contents**

    1. Why orbitals, and what problem are we solving?
    2. A product of orbitals, and helium
    3. The average field of the other electrons
    4. Screening in atoms: $2s$ below $2p$, and the size of an orbital
    5. Why the product fails: spin orbitals, the Slater determinant and the Pauli principle
    6. The Hamiltonian in simpler notation, and the energy of a determinant
    7. Minimising the energy: the Fock operator (7.1), a basis set and the Roothaan equations (7.2), the self-consistent loop (7.3)
    8. What Hartree-Fock misses, and the two ways past it
    9. What was checked, and what was not
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Terms used in this notebook": mo.md(r"""
        - **Orbital** $\varphi_i(\mathbf r)$: a one-electron wave in space. A **spin orbital** $\psi_i(\mathbf x)$, a spatial orbital times the spin function $\alpha$ (up) or $\beta$ (down), adds the spin; $\mathbf x=\{\mathbf r,\sigma\}$ with $\sigma\in\{\alpha,\beta\}$ stands for position and spin together.
        - **Hartree product**: the product of one orbital per electron. **Orbital approximation**: the many-electron wave is built from one orbital per electron and nothing else.
        - **Density** $\rho(\mathbf r)$: the electron charge per unit volume, the sum of $\lvert\varphi_i\rvert^2$ over the occupied orbitals.
        - **Average (mean) field**: the potential energy of one electron in the *averaged* charge cloud of the others, not in their instantaneous positions.
        - **Screening**: the pull of the nucleus on an outer electron is reduced by the inner electrons; $Z_{\text{eff}}$ is the charge that is left.
        - **Slater determinant**: the antisymmetric product of the occupied spin orbitals.
        - **One- and two-electron operators** $h$ and $v$: the kinetic-plus-nuclear-attraction part and the pair repulsion $1/r_{ij}$ of the Hamiltonian.
        - **Fock operator** $\hat f$: the one-electron effective Hamiltonian of Hartree-Fock. **Orbital energy** $\varepsilon_i$: its eigenvalue for orbital $i$.
        - **$\hat J$, $\hat K$**: the Coulomb (average-field) and exchange terms of $\hat f$.
        - **SCF**: self-consistent field, the loop that repeats until the orbitals and the field they make agree.
        - **Correlation energy**: what the best orbital product still misses.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 Why orbitals, and what problem are we solving?

    **Orbitals are a construction.** Chemists talk about orbitals all the time, and the picture is useful, but only for a system with *one* electron (hydrogen, He$^+$) are
    orbitals the exact solutions of the problem. For more electrons, an orbital is a mathematical device: each electron's motion is described by a single-particle function that does not depend
    on where the other electrons are at that instant. Hartree-Fock (HF) is the theory that finds the best such orbitals. It is the foundation of molecular orbital theory, and the starting point of
    the better methods of the later notebooks (the correlation methods and, in spirit, DFT).

    **What problem is it solving?** After the Born-Oppenheimer approximation the nuclei are fixed at positions $\mathbf R$ and the electrons obey the electronic Schrödinger equation,

    $$
    \bigl[\hat T_e+\hat V_{eN}+\hat V_{NN}+\hat V_{ee}\bigr]\,\Psi(\mathbf r;\mathbf R)=E_{\text{el}}(\mathbf R)\,\Psi(\mathbf r;\mathbf R),
    $$

    with the kinetic energy of the electrons, their attraction to the nuclei, the repulsion of the nuclei with each other (a constant at fixed $\mathbf R$) and the repulsion
    $\hat V_{ee}=\sum_{i<j}1/r_{ij}$ of the electrons with each other. Two things come out of it. The energy $E_{\text{el}}(\mathbf R)$, with $\hat V_{NN}$ included, is the **potential energy surface**
    of `from_atoms_to_molecules.py`, section 2 (equilibrium geometries, vibration frequencies). The wave $\Psi$ gives molecular properties (dipole moment, polarizability, and so on). Everything that
    makes the problem hard is in $\hat V_{ee}$, the one term that ties the electrons together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 A product of orbitals, and helium

    **The thought experiment.** Hydrogen we can solve. Add a second electron to get H$^-$ (or He), and *pretend the electrons ignore each other*, $\hat V_{ee}=0$. Then the Hamiltonian is a sum of
    one-electron Hamiltonians, the problem separates, and the exact wave is a **product** of one-electron functions. This suggests starting from the general form

    $$
    \Psi(\mathbf r_1,\dots,\mathbf r_N)\approx\varphi_1(\mathbf r_1)\,\varphi_2(\mathbf r_2)\cdots\varphi_N(\mathbf r_N),
    $$

    called a **Hartree product**. Ignoring the repulsion entirely is a serious approximation. The improvement that keeps the product is to let each electron feel the repulsion of the others
    *on average* (sections 3 and 4). The product has one serious defect, which is not about energy but about symmetry (section 5).

    **Helium as the concrete case.** For helium the exact wave depends on two positions $\mathbf r_1,\mathbf r_2$ jointly. Take the product
    $\Psi\approx\varphi(\mathbf r_1)\,\varphi(\mathbf r_2)$ with the hydrogen-like function $\varphi\propto e^{-\zeta r}$, and let $\zeta$
    be the one free number. (Two electrons may share one spatial orbital only if their spins are opposite; section 5 makes this
    precise.) The energy in hartree is a sum of three terms,

    $$
    E(\zeta)=\underbrace{\zeta^{2}}_{\text{kinetic}}\;\underbrace{-\,2Z\zeta}_{\text{attraction to the nucleus}}\;
    {}+\underbrace{\tfrac58\zeta}_{\text{repulsion of the two electrons}},\qquad Z=2 .
    $$

    The best $\zeta$ is the one that makes $E$ lowest: this is the variational principle of `basis_sets_from_base_states.py`,
    section 1, applied to a trial wave with one free number (section 6 makes it the general principle). Without the last term the minimum is at $\zeta=Z=2$ (the helium ion's orbital,
    twice over) and the energy is $-4$ hartree. With it, the minimum moves to $\zeta=Z-\tfrac5{16}=1.6875$ and the energy is $-2.8477$ hartree.
    The repulsion works as a **screening**: each electron sees a nucleus whose charge is reduced by a bit under one third of an electron
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
        them. The repulsion $\langle 1/r_{12}\rangle$ for two electrons in this orbital is $\tfrac58\zeta$; section 3 shows what it is the
        energy *of*, and `test_molecules.py` checks the value by direct integration. Setting $dE/d\zeta=2\zeta-2Z+\tfrac58=0$ gives
        $\zeta=Z-\tfrac5{16}$ and $E=-(Z-\tfrac5{16})^2$.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 The average field of the other electrons

    In helium "the average of the other electrons" is one electron, so "the average of the others" and "the other one" sound like the same
    thing, and the number $\tfrac58\zeta$ does not look like a field. It is one. This section writes the field down.

    **The field of a charge cloud.** Let the other electron have the density $\rho_{\text{other}}(\mathbf r')=\lvert\varphi(\mathbf r')\rvert^2$:
    its charge per unit volume, averaged over where it may be. An electron at $\mathbf r$ is repelled by every small piece of that cloud, by
    the amount (charge)/(distance). Adding up all the pieces gives the **potential energy of an electron at $\mathbf r$** in the average field,

    $$
    v_J(\mathbf r)=\int\frac{\rho_{\text{other}}(\mathbf r')}{\lvert\mathbf r-\mathbf r'\rvert}\,d\mathbf r'.
    $$

    This is a potential energy like the nuclear attraction $-Z/r$: a function of position that multiplies the orbital in the Schrödinger
    equation. For the cloud $\varphi\propto e^{-\zeta r}$ the integral can be done in closed form (the charge inside a sphere of radius $r$
    acts as if it sat at the centre, and the charge outside adds a constant):

    $$
    v_J(r)=\frac1r\Bigl[\,1-e^{-2\zeta r}\,(1+\zeta r)\Bigr].
    $$

    Two limits show what the formula says. **Far away** ($r\gg1/\zeta$) the bracket tends to $1$ and $v_J\to1/r$: the cloud looks like a
    point charge of one electron. **Close in** ($r\to0$) it tends to the *finite* value $\zeta$: inside the cloud the charge lies mostly
    outside the electron, where it pulls equally in all directions and its effect cancels, so the repulsion does not diverge. The
    closed form is compared with a numerical integration over the cloud in `test_molecules.py`.
    """)
    return


@app.cell
def _(mo):
    zeta_field = mo.ui.slider(1.0, 3.0, step=0.025, value=1.6875, label=r"orbital exponent $\zeta$ of the cloud", show_value=True)
    return (zeta_field,)


@app.cell
def _(average_field_figure, mo, plt, zeta_field):
    _fig = average_field_figure(zeta_field.value)
    _out = mo.vstack([zeta_field, _fig])
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left:* the field $v_J(r)$ of the other helium electron (curve) and the cloud that makes it (shading, the probability per Å of finding
    the electron at distance $r$). *Right:* the same idea for lithium, treated in the next paragraph.

    **What $\tfrac58\zeta$ is the energy of.** The electron in orbital $\varphi$ moves in this field, so its energy in the field is the
    average of $v_J$ over its own cloud:

    $$
    \int\lvert\varphi(\mathbf r)\rvert^{2}\,v_J(\mathbf r)\,d\mathbf r=\tfrac58\zeta .
    $$

    This is exactly the repulsion term of $E(\zeta)$ in section 2: the energy of one electron in the average field of the other
    (checked by integration in `test_molecules.py`). Each electron feels the other, but the repulsion energy of the pair is one number, so it enters $E(\zeta)$
    once; the double counting returns in the orbital energies of section 7.1.

    **More electrons: lithium.** In lithium two electrons fill a compact $1s$ core, $\varphi_c\propto e^{-\zeta_c r}$ with
    $\zeta_c=Z-\tfrac5{16}=2.6875$ (the helium-like value for $Z=3$), and the third electron lives outside it. The core makes the field
    $2\,v_J(r)$ (two electrons), and the nuclear charge as seen by the outer electron at distance $r$ is the bare charge minus the part of
    the potential that the core takes away,

    $$
    Z_{\text{eff}}(r)=Z-2\,r\,v_J(r)=1+2\,e^{-2\zeta_c r}\,(1+\zeta_c r).
    $$

    It is $3$ at the nucleus (an electron inside the core sees the bare nucleus) and falls to $1$ outside it (the nucleus is screened
    by two core electrons), which is the right panel of the figure. The outer electron in turn makes a field on the core, and the two sets of
    orbitals have to agree: this is what the loop of section 7.3 does. **Neon** (ten electrons) is the same construction with all the
    other electrons in the field: each electron feels the nucleus of charge $10$ and the average cloud of the others.

    **The part that is not an average of the others.** The sum over all electrons in $v_J$ includes the electron's *own* cloud, because the
    field is built from the total density. An electron does not repel itself, and the exchange term $\hat K$ of section 7.1 removes exactly
    this self-part (and keeps electrons of the same spin apart). For helium this is a one-line statement: with both electrons in
    $\varphi$, $\hat J-\tfrac12\hat K$ is equal to the field of the *single other* electron.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Screening in atoms: $2s$ below $2p$, and the size of an orbital

    *This section uses the field of section 3 to explain two facts about atoms.*

    **Why $2s$ is lower than $2p$.** In hydrogen the energy depends on $n$ only (section 5 of the hydrogen notebook), because the
    potential is exactly $-1/r$. In an atom with several electrons the potential seen by one electron is $-Z_{\text{eff}}(r)/r$ of section 3:
    large close in, small far out. The toy below is lithium's $Z_{\text{eff}}(r)=1+2e^{-r/a}\bigl(1+r/2a\bigr)$ (the formula of section 3 with
    $\zeta_c=1/2a$), where the distance $a$ over which the core screens is the one free parameter: lithium's own core has $a=0.098$ Å, and the
    slider moves it. The radial equation of the hydrogen notebook is solved with this potential.
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
    familiar filling order $1s,2s,2p,3s,3p,4s,3d\dots$ (the $4s$ before $3d$ is the same effect, a bit stronger). With lithium's own core
    ($a=0.098$ Å) the toy gives $E_{2s}=-0.175$ and $E_{2p}=-0.126$ hartree (real lithium: about $-0.198$ and $-0.130$, from memory); the
    toy has the one parameter $a$ and ignores the outer electron's effect on the core, so only the ordering is a result.

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
    and four times as tightly bound; this is the $\zeta=Z=2$ orbital of section 2. For uranium ($Z=92$) the $1s$ orbital would have $\langle r\rangle=0.016$ bohr $=0.0086$ Å.

    **Why atoms do not shrink to a point, or grow without limit.** Neither extreme happens, because the other electrons screen the
    nucleus (the effect of $\zeta=Z-\tfrac5{16}$ in section 2, and of $Z_{\text{eff}}(r)$ in section 3). An outer electron sees the nucleus
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

    **What this means for the basis set.** Nothing about screening is put in by hand in a calculation: the average field of section 3 is
    built from the electrons themselves. What the user chooses is the basis set, which has to be able to represent the sizes that come out
    (tight exponents for the heavy atoms' $1s$ orbitals, which shrink as $1/Z$, and effective core potentials for the heaviest elements);
    that is the subject of `basis_sets_from_base_states.py`, section 5 (the paragraph on orbital sizes), and the effective core potential is
    met in the OPLS5 labels of `../dft/reading_a_dft_label.py`, section 9.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Why the product fails: spin orbitals, the Slater determinant and the Pauli principle

    **Spin orbitals.** An electron has a spin as well as a position $\mathbf r$, and the spin has two possible states, described by the spin functions $\alpha$ (up) and $\beta$ (down). Write $\mathbf x=\{\mathbf r,\sigma\}$ for position and spin together, with $\sigma\in\{\alpha,\beta\}$, and replace the spatial
    orbital $\varphi(\mathbf r)$ by a **spin orbital** $\psi(\mathbf x)$: a spatial orbital times either the spin function $\alpha$ or $\beta$. The Hartree product becomes $\psi_1(\mathbf x_1)\,\psi_2(\mathbf x_2)\cdots\psi_N(\mathbf x_N)$.

    **The failure.** Electrons are fermions: the wave must change sign when the space-spin coordinates of any two electrons are swapped (the spinor notebook shows where this comes from). Take two electrons.
    The product $\psi_1(\mathbf x_1)\psi_2(\mathbf x_2)$ becomes $\psi_1(\mathbf x_2)\psi_2(\mathbf x_1)$ after the swap, and that is the *negative* of the original only for special orbitals; in general it is
    a different function. So the Hartree product does not respect the antisymmetry that nature demands, however good its energy might look.

    **The fix: a determinant.** For two electrons the combination $\tfrac1{\sqrt2}\bigl[\psi_1(\mathbf x_1)\psi_2(\mathbf x_2)-\psi_1(\mathbf x_2)\psi_2(\mathbf x_1)\bigr]$ changes sign under the swap for *any* choice of the two orbitals. For $N$ electrons the
    same idea is a determinant,

    $$
    \Psi(\mathbf x_1,\dots,\mathbf x_N) = \frac{1}{\sqrt{N!}}
    \begin{vmatrix}
    \psi_1(\mathbf x_1) & \psi_2(\mathbf x_1) & \cdots & \psi_N(\mathbf x_1)\\
    \psi_1(\mathbf x_2) & \psi_2(\mathbf x_2) & \cdots & \psi_N(\mathbf x_2)\\
    \vdots & & & \vdots\\
    \psi_1(\mathbf x_N) & \psi_2(\mathbf x_N) & \cdots & \psi_N(\mathbf x_N)
    \end{vmatrix} ,
    $$

    a **Slater determinant**. Swapping two electrons swaps two rows, which flips the sign. Putting two electrons in the same spin orbital makes two columns equal, so $\Psi = 0$: the **Pauli exclusion principle** is a
    consequence of antisymmetry and comes out for free. Two electrons can share a *spatial* orbital only with opposite spins ($\varphi\alpha$ and $\varphi\beta$ are different spin orbitals), which is the rule used for helium in section 2.

    **Each electron is in every orbital.** The rows of the determinant make the electrons indistinguishable: no electron "owns" an orbital. This is easy to forget, because writing the whole determinant is cumbersome. The usual shorthand lists the occupied
    spin orbitals only, $\lvert\psi_i\psi_j\cdots\psi_k\rangle$ or $\lvert ij\cdots k\rangle$; the normalisation factor is implied.

    **What a single determinant means physically.** Assuming that the electrons are described by one determinant is equivalent to assuming that each electron moves independently of the others, except that it feels the Coulomb repulsion of the
    *average positions* of all the electrons, and a further interaction, **exchange**, that comes from the antisymmetry: electrons of the *same* spin are kept apart (the wave is zero when two of them are at the same point), electrons of opposite spin are not.
    Hartree-Fock is therefore an **independent-particle**, or **mean-field**, model. Kohn-Sham DFT, in the next notebooks, resembles it closely.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 The Hamiltonian in simpler notation, and the energy of a determinant

    **A simpler Hamiltonian.** Group the terms by how many electrons they involve. The **one-electron operator**
    $h(i)=-\tfrac12\nabla_i^2-\sum_A Z_A/r_{iA}$ is the kinetic energy of electron $i$ plus its attraction to all the nuclei, and the **two-electron operator** $v(i,j)=1/r_{ij}$ is the repulsion of a pair. Then

    $$
    \hat H_{\text{el}}=\sum_i h(i)+\sum_{i<j}v(i,j)+V_{NN}.
    $$

    $V_{NN}$ is a constant at fixed nuclei: it shifts the energy and changes no wave function, so it is left out until the end. (This notation hides the integrals that have to be done later.)

    **The energy.** With the wave restricted to a single determinant of orthonormal spin orbitals, the energy is the usual quantum-mechanical average $E=\langle\Psi\vert\hat H_{\text{el}}\vert\Psi\rangle$. The **variational principle**
    (`basis_sets_from_base_states.py`, section 1) says that this number is never below the true ground-state energy, so **the best orbitals are the ones that make $E$ lowest**. The orbitals can be found on a numerical grid or,
    far more commonly, as combinations of fixed basis functions (7.2).

    Working out the average for a determinant (algebra that is not repeated here) gives the **Hartree-Fock energy** in terms of integrals over the orbitals,

    $$
    E_{\text{HF}}=\sum_i\langle i\vert h\vert i\rangle+\tfrac12\sum_{i,j}\bigl(J_{ij}-K_{ij}\bigr),
    $$

    where $\langle i\vert h\vert i\rangle$ is the **one-electron integral** (the kinetic energy and nuclear attraction of an electron in spin orbital $i$), the **Coulomb integral**
    $J_{ij}=\iint\lvert\psi_i(\mathbf x_1)\rvert^2\,r_{12}^{-1}\,\lvert\psi_j(\mathbf x_2)\rvert^2\,d\mathbf x_1d\mathbf x_2$ is the classical repulsion of the charge cloud of electron $i$ with that of electron $j$, and the **exchange integral**
    $K_{ij}=\iint\psi_i^{*}(\mathbf x_1)\psi_j(\mathbf x_1)\,r_{12}^{-1}\,\psi_j^{*}(\mathbf x_2)\psi_i(\mathbf x_2)\,d\mathbf x_1d\mathbf x_2$ has no classical counterpart: it is the same integral with the two spin orbitals
    swapped, and it is non-zero only for two spin orbitals with the *same* spin. (The sum includes $i=j$, where $J_{ii}=K_{ii}$ and the two cancel: no electron repels itself.) These are the two-electron integrals in the "chemists' notation" $[ii\vert jj]$ and $[ij\vert ji]$ of the programs.

    **Helium as the check.** The two electrons occupy the spin orbitals $\varphi\alpha$ and $\varphi\beta$ with the same $\varphi\propto e^{-\zeta r}$. Each has $\langle i\vert h\vert i\rangle=\tfrac12\zeta^2-Z\zeta$. Their $J_{12}=\tfrac58\zeta$ is the repulsion of one cloud with the other,
    which is exactly the energy of one electron in the average field $v_J$ of section 3, and $K_{12}=0$ because the spins are opposite. So $E=2(\tfrac12\zeta^2-Z\zeta)+\tfrac58\zeta$, which is the $E(\zeta)$ of section 2. The general formula
    contains nothing new for helium; what it adds is that it holds for any number of electrons and any orbitals.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 Minimising the energy: the Fock operator, a basis set, and the loop

    ### 7.1 The Hartree-Fock equations and the Fock operator

    The best determinant is found by minimising $E_{\text{HF}}$ with respect to changes of the orbitals, while keeping them orthonormal. The constraint is handled by the method of Lagrange multipliers (a standard device
    for minimising under a condition, not developed here). Setting the first variation to zero gives one equation per orbital,

    $$
    \Bigl[h+\sum_{j\ne i}\hat J_j-\sum_{j\ne i}\hat K_j\Bigr]\psi_i=\varepsilon_i\,\psi_i .
    $$

    **The two new terms.** $\hat J_j$ is the **Coulomb operator**: the potential energy at the position of electron $i$ of the average charge cloud of the electron in orbital $j$,
    $\hat J_j(\mathbf x_1)=\int\lvert\psi_j(\mathbf x_2)\rvert^2\,r_{12}^{-1}\,d\mathbf x_2$, acting on an orbital by multiplication. The sum over $j$ is the average field of all the *other* electrons, the $v_J$ of section 3. $\hat K_j$ is the **exchange operator**, defined
    by its action on an orbital, $\hat K_j\psi_i(\mathbf x_1)=\bigl[\int\psi_j^{*}(\mathbf x_2)\,r_{12}^{-1}\,\psi_i(\mathbf x_2)\,d\mathbf x_2\bigr]\psi_j(\mathbf x_1)$: it looks like $\hat J_j$ with the two orbitals swapped. It is not a multiplication,
    it mixes the orbital $\psi_j$ into $\psi_i$, it has no classical analogue, and it exists because of antisymmetry. These are the operators whose averages are the integrals $J_{ij}$ and $K_{ij}$ of section 6.

    **Removing the restriction $j\neq i$.** For $j=i$ the two terms cancel, $(\hat J_i-\hat K_i)\psi_i=0$: this is the self-interaction cancelling. So the restriction can be dropped, and with the operator
    $\hat J=\sum_j\hat J_j$ (the average field of *all* electrons, own cloud included) and $\hat K=\sum_j\hat K_j$ one obtains the **Fock operator** and the **Hartree-Fock equation**:

    $$
    \hat f\,\psi_i=\varepsilon_i\,\psi_i,\qquad
    \hat f=-\tfrac12\nabla^{2}+v_{\text{nuc}}+\hat J-\hat K .
    $$

    Here $-\tfrac12\nabla^2+v_{\text{nuc}}$ is the operator $h$ of section 6. Check: for one electron (hydrogen) $\hat J-\hat K=0$ and this is the exact hydrogen equation.

    **What $\hat f$ is, and what it is not.** The symbol $\hat f$ is *not* an orbital. It is an **operator**: a rule that takes a function and returns
    another function. It is the **one-electron effective Hamiltonian**: the Hamiltonian that one electron would have if the other
    electrons were replaced by their average field (plus the exchange correction). The hydrogen equation $\hat H\psi=E\psi$ has the same shape, with the exact Hamiltonian $\hat H$ in
    place of $\hat f$. The Hartree-Fock equation is an **eigenvalue equation**: its solutions are the functions $\psi_i$ (the **orbitals**, the
    *eigenfunctions* of $\hat f$) and the numbers $\varepsilon_i$ (the **orbital energies**, its *eigenvalues*); applying $\hat f$ to an orbital returns
    the same orbital times a number.

    (The form above, summed over spin orbitals, is the general one. For a closed shell written with spatial orbitals,
    each holding two electrons, the same operator reads $v_{\text{nuc}}+\hat J[\rho]-\tfrac12\hat K[\rho]$ with the *total* density $\rho$; the
    $\tfrac12$ is the one met in `../dft/kohn_sham_matrix.py`.)

    **The orbital energy $\varepsilon_i$** is the energy of an electron in orbital $i$ in the field of all the others. It is not "the share of the
    total energy": the sum of the $\varepsilon_i$ counts every electron-electron repulsion twice (once from each electron's side). For helium in the
    one-orbital model of section 2: $\varepsilon=\tfrac12\zeta^2-Z\zeta+\tfrac58\zeta=-0.896$ hartree, $2\varepsilon=-1.793$, and the total energy is
    $2\varepsilon-\tfrac58\zeta=-2.848$ hartree (tested in `test_molecules.py`). By Koopmans' theorem $-\varepsilon_i$ approximates the energy needed to remove
    electron $i$ (met in `../dft/reading_a_dft_label.py`, section 6).

    **Why $\hat f$ depends on its own solutions.** $\hat J$ and $\hat K$ are built from the occupied orbitals $\psi_j$, and the occupied orbitals
    are what the equation is supposed to find. So $\hat f$ cannot be written down before the orbitals are known, and the orbitals cannot be found
    before $\hat f$ is known. The way out is to guess orbitals and refine them by iteration (7.3). This circularity is the one feature that makes the orbital approximation
    harder than the hydrogen equation, and it is the reason the method is called *self-consistent*.

    **Kohn-Sham DFT keeps this whole structure.** It has the same eigenvalue equation $\hat f\psi_i=\varepsilon_i\psi_i$, with the same $-\tfrac12\nabla^2$, $v_{\text{nuc}}$ and $\hat J$. The only change
    is that the exchange term $\hat K$, and the correlation that Hartree-Fock leaves out, are replaced by an approximate potential $v_{xc}[\rho]$
    that depends on the density (and, in a hybrid functional, a fraction of $\hat K$ is kept). Written compactly: $\hat f=-\tfrac12\nabla^2+v_{\text{eff}}[\rho]$ with
    $v_{\text{eff}}=v_{\text{nuc}}+v_J+v_{xc}$, where Hartree-Fock has $v_{\text{nuc}}+\hat J-\hat K$. The next notebook, `../dft/reading_a_dft_label.py`, takes this
    one step.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.2 A basis set: the Roothaan equations

    The operator equation can be solved on a grid (*numerical* Hartree-Fock) or, in practice, in the space spanned by fixed basis functions $\phi_\mu$ (called $\chi_\mu$ in `basis_sets_from_base_states.py`). Write each orbital as a sum,
    $\psi_i=\sum_\mu C_{\mu i}\,\phi_\mu$, put it into $\hat f\psi_i=\varepsilon_i\psi_i$, multiply by one basis function $\phi_\nu$ and integrate over all space, once for each $\nu$: the steps of
    `basis_sets_from_base_states.py`, section 1, with $\hat f$ in place of $H$. They give the **Hartree-Fock-Roothaan equations**,

    $$
    F_{\mu\nu}=\int\phi_\mu\,\hat f\,\phi_\nu\,d\mathbf r,\qquad
    S_{\mu\nu}=\int\phi_\mu\,\phi_\nu\,d\mathbf r,\qquad
    \mathbf F\mathbf C=\mathbf S\mathbf C\,\boldsymbol\varepsilon .
    $$

    $\mathbf F$ is the operator $\hat f$ written in the basis (the **Fock matrix**), $\mathbf S$ is the **overlap matrix** of the basis functions
    (it is not the identity because the functions overlap), column $i$ of $\mathbf C$ holds the weights of orbital $i$, and $\boldsymbol\varepsilon$ is the
    diagonal matrix of the orbital energies. Each term of $\hat f$ gives one matrix: $\mathbf F=\mathbf T+\mathbf V+\mathbf J-\mathbf K$.
    (For a closed shell the spin is integrated out, and the equation is written with the spatial parts of the orbitals.) This is an eigenvalue equation except for $\mathbf S$. Changing to an orthogonalised basis
    removes $\mathbf S$ (the recipe with $\mathbf S^{-1/2}$ is in `../dft/kohn_sham_matrix.py`, section 3), and what is left is the diagonalisation of a matrix. But $\mathbf F$ itself depends on the
    orbitals through $\hat J$ and $\hat K$, so the diagonalisation has to be repeated, which is the loop of 7.3. `../dft/kohn_sham_matrix.py` builds each matrix of $\mathbf F$ for water.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.3 The self-consistent loop

    The loop below is the same for Hartree-Fock and for Kohn-Sham; only the box that builds $\hat f$ differs. It runs at one fixed set of nuclear
    positions $\mathbf R$. In words:

    1. **Guess** the orbitals, or only the density (a program starts from a sum of atomic densities).
    2. **Build** $\hat f$ from them: $\hat J$ (and $\hat K$) need the density.
    3. **Solve** $\hat f\psi_i=\varepsilon_i\psi_i$, in a basis as $\mathbf F\mathbf C=\mathbf S\mathbf C\boldsymbol\varepsilon$ (7.2).
    4. **Form** the new density from the occupied orbitals (the lowest $N/2$ of them for a closed shell).
    5. **Compare** with the density that built $\hat f$. If it changed, go back to step 2; if not, the orbitals and the field they make agree: the density is
       *self-consistent*.

    The result is the Born-Oppenheimer energy $E(\mathbf R)$ of `from_atoms_to_molecules.py`, section 2. Programs mix the new and the old density with a
    smarter scheme than a plain average (DIIS) so that fewer cycles are needed. `../dft/kohn_sham_matrix.py`, section 5, runs the loop by hand on
    water, and `from_atoms_to_molecules.py`, section 2.1, shows how a geometry optimisation wraps one more loop around it.
    """)
    return


@app.cell
def _(mo, plt, scf_loop_figure):
    _fig = scf_loop_figure()
    _out = mo.as_html(_fig)
    plt.close(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 What Hartree-Fock misses, and the two ways past it

    **How good is the orbital product?** Against the exact non-relativistic energy of helium, $-2.9037$ hartree:

    | wave function | energy (hartree) |
    |---|---|
    | no repulsion between the electrons | $-4$ (not an approximation of anything: the term is missing) |
    | one orbital, bare nucleus, $\zeta=2$ | $-2.75$ |
    | one orbital with the best $\zeta=1.6875$ | $-2.8477$ |
    | best possible orbital product (Hartree-Fock limit) | $-2.8617$ |
    | exact | $-2.9037$ |

    The one-orbital picture recovers 98 % of the exact energy, but the missing $0.042$ hartree ($1.1$ eV) is larger than most
    chemical energy differences. That gap is the **correlation energy**. Each electron in HF sees only the *average* position of the others; real
    electrons also dodge each other instantly, and no product of orbitals can express this. It is a small fraction of the total energy, but of the same
    size as the energy differences chemistry cares about. Dispersion, the attraction between two molecules with no charge or hydrogen bond, is a pure
    correlation effect, so HF cannot produce it at all: HF finds two methane molecules repulsive (the numbers for water and methane dimers are in
    `../dft/reading_a_dft_label.py`, section 2).

    ### Two ways past Hartree-Fock

    - **Wave-function methods** (MP2, coupled cluster such as CCSD(T)) add correlation on top of the HF determinant. They
      can be very accurate, and CCSD(T) is the usual reference in the DFT notebook, but their cost grows steeply with
      system size. `../dft/coupled_cluster.py` (the last notebook) explains where CCSD(T) comes from.
    - **Density functional theory** drops the wave function and works with the density, keeping a cost close to
      Hartree-Fock's. The price is an *approximate* exchange-correlation term, which replaces $\hat K$ in the operator of section 7.1 and which the
      letters of a label describe. Next: `../dft/reading_a_dft_label.py`.

    **Takeaways for a slide**

    1. **Orbital approximation:** each electron moves in the field of the nuclei and the *average* cloud of the others, $v_J(\mathbf r)=\int\rho(\mathbf r')/\lvert\mathbf r-\mathbf r'\rvert\,d\mathbf r'$. In helium
       the repulsion term $\tfrac58\zeta$ is the energy of one electron in this field, and it recovers 98 % of the energy.
    2. **The Fock operator** $\hat f=-\tfrac12\nabla^2+v_{\text{nuc}}+\hat J-\hat K$ is the one-electron effective Hamiltonian; the orbitals are its eigenfunctions and
       $\varepsilon_i$ its eigenvalues. $\hat J$ is the average field, $\hat K$ the same-spin exchange that removes the self-interaction.
    3. $\hat f$ depends on its own solutions, hence the **SCF loop**; in a basis it is $\mathbf F\mathbf C=\mathbf S\mathbf C\boldsymbol\varepsilon$.
    4. The missing piece is the **correlation energy** (helium 1.1 eV). Kohn-Sham DFT keeps the structure and approximates $\hat K$ plus correlation; coupled
       cluster adds correlation to the determinant.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 What was checked, and what was not

    **Checked numerically** (`pixi run python -m pytest -q standing_waves/test_molecules.py`):

    - *Section 2.* The minimum of $E(\zeta)$ is at $\zeta=1.6875$ with $E=-2.84766$; the ordering of exact $-2.9037$ < HF limit $-2.8617$ < the one-orbital $-2.8477$.
    - *Section 3.* The closed form $v_J(r)=\tfrac1r[1-e^{-2\zeta r}(1+\zeta r)]$ against a numerical integral of the density over all space (four values of $\zeta$, five distances);
      its limits ($\zeta$ at $r=0$, $1/r$ far out, falling steadily and below $1/r$); the identity $\int\lvert\varphi\rvert^2v_J\,d\mathbf r=\tfrac58\zeta$ by integration (three values of $\zeta$), and with it
      $E(\zeta)=2\langle h\rangle+\langle v_J\rangle$; $Z_{\text{eff}}(r)=Z-2rv_J$ has the limits $3$ and $1$ for lithium and the closed form $1+2e^{-2\zeta_cr}(1+\zeta_cr)$, and equals $Z$ minus the core charge inside $r$ and $r$ times the
      potential of the core outside $r$, by direct integration. The statement that $\hat J-\tfrac12\hat K$ is the field of the other electron in helium is a check of the arithmetic of the weights ($2v-v=v$), not of the operators.
    - *Section 4.* The screened-potential levels: $2s=2p$ when the screening disappears (and $=-0.125$ hartree), $s<p<d$ at each $n$ otherwise, and the lithium numbers $E_{2s}=-0.175$, $E_{2p}=-0.126$ hartree.
      Orbital size: $\langle r\rangle=\tfrac12[3n^2-\ell(\ell+1)]/Z$ against numerical integration of the hydrogen-like radial density (six shells, $Z$ up to 18), the $1/Z$ scaling and normalisation of the squeezed density, and $13.5$ bohr $=7.14$ Å for $3s$.
      The H to Ar sizes are Hartree-Fock numbers read from `data_atom_sizes.json` (tests check hydrogen's $1.5$ bohr, the range, and the trend across row 2); the $99\,\%$ radius of $3s$ ($13.5$ Å) was computed once by integration and is not in a test.
    - *Section 6.* For helium the energy formula reduces to $E(\zeta)=2\langle h\rangle+\langle v_J\rangle$ with $\langle h\rangle=\tfrac12\zeta^2-Z\zeta$ and $J_{12}=\int\lvert\varphi\rvert^2v_J\,d\mathbf r=\tfrac58\zeta$, by integration (the same test as in section 3); $K_{12}=0$ for opposite spins is stated, not computed.
    - *Section 7.1.* The helium orbital energy $-0.8965$ hartree and $E=2\varepsilon-\tfrac58\zeta$, with $\varepsilon=\tfrac12\zeta^2-Z\zeta+\langle v_J\rangle$.

    **Not checked here, or stated from memory.**

    - The energy expression of a determinant, the Hartree-Fock equations (the Lagrange-multiplier minimisation), the form of $\hat K$ and the cancellation of the self-repulsion are the textbook forms, following Sherrill's notes; they are not derived in this notebook and the operators are not applied to functions in the tests. That a single determinant is equivalent to the mean-field-plus-exchange picture is stated, not shown.
    - The helium values $-2.9037$ and $-2.8617$ hartree (exact and Hartree-Fock limit) are literature values, not computed.
    - Real lithium's $2s$ and $2p$ energies ($-0.198$ and $-0.130$ hartree, $5.39$ and $3.54$ eV) are quoted from memory.
    - That the $4s$ before $3d$ filling order is "the same effect"; the screening toy has only one parameter and shows only the ordering.
    - Koopmans' theorem and the statement that HF finds two methane molecules repulsive are from the DFT notebook (computed there, `data_dispersion.json`), not recomputed here.
    - The SCF flowchart is a textbook description, not a computation (the loop is run for water in `../dft/kohn_sham_matrix.py`, section 5).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    Free sources, checked on 2026-11-13 (the full table, with what each covers, is in `SOURCES.md`).

    - [Helium, the variational method (LibreTexts, Pacific Union College, 8.2)](https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.02%3A_Perturbation_Theory_and_the_Variational_Method_for_Helium) (intro).
    - [Slater determinants (LibreTexts, Pacific Union College, 8.6)](https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.06%3A_Antisymmetric_Wavefunctions_can_be_Represented_by_Slater_Determinants) and [the self-consistent field method (8.3)](https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.03%3A_Hartree-Fock_Equations_are_Solved_by_the_Self-Consistent_Field_Method) (intro).
    - [An introduction to Hartree-Fock molecular orbital theory (C. D. Sherrill, Georgia Tech, June 2000, 8-page free PDF)](https://vergil.chemistry.gatech.edu/static/content/hf-intro.pdf): the order of this notebook's argument, from the problem and the Hartree product to the Roothaan equations and the SCF procedure; his $J_j$ and $K_j$ are the operators $\hat J_j$ and $\hat K_j$ of section 7.1 (intro level, derivation; read in full on 2026-11-13).
    - [Introduction to Hartree-Fock molecular orbital theory (C. D. Sherrill, Georgia Tech slides)](https://vergil.chemistry.gatech.edu/static/content/Hartree-Fock-Intro.pdf): the Hartree-Fock energy, exchange and the SCF procedure (intro).
    - [Psi4NumPy tutorial 03, Hartree-Fock](https://github.com/psi4/psi4numpy/tree/master/Tutorials/03_Hartree-Fock): the Fock matrix and the SCF loop in numpy (derivation).
    """)
    return


if __name__ == "__main__":
    app.run()
