import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Reading a DFT Label (OMol25, SPICE II)")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dft_data import load, shell_label
    from dft_plots import (
        basis_figure,
        bsse_figure,
        damping_figure,
        dispersion_figure,
        eint_figure,
        grid_figure,
        homo_figure,
        line_fields_figure,
        scf_figure,
        split_figure,
    )

    scf_data = load("scf")
    basis_data = load("basis")
    ladder_data = load("ladder")
    disp_data = load("dispersion")
    return (mo, plt, np, load, shell_label, basis_figure, bsse_figure, damping_figure, dispersion_figure, eint_figure, grid_figure, homo_figure, line_fields_figure, scf_figure, split_figure, scf_data, basis_data, ladder_data, disp_data,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reading a DFT Label: ωB97M-V/def2-TZVPD and ωB97M-D3(BJ)/def2-TZVPPD

    The datasets used to train machine-learned force fields are labelled with strings such as **ωB97M-V/def2-TZVPD**
    (OMol25) and **ωB97M-D3(BJ)/def2-TZVPPD** (SPICE II). Each piece of the string is a separate choice made by whoever
    ran the calculation. This notebook takes the labels apart one piece at a time, and each piece comes with a real
    calculation on water, so that you can see what the choice does.

    **Read first:** `../quantization` (sections 3 and 5) for the idea that boundary conditions give energy levels. The
    worked PySCF examples in `../../qm-calculations` (water, basis functions, dimers) are what the numbers here extend.

    **Contents**

    1. The label has three parts
    2. Before DFT: Hartree-Fock
    3. What the calculation solves: Kohn-Sham DFT
    4. The basis set: def2, TZ, P, D
    5. The functional: Jacob's ladder, B97, M
    6. Exact exchange and range separation: ω
    7. Dispersion: -V against D3(BJ)
    8. Settings that are not in the label: the grid
    9. Putting the two labels together
    10. What I have not verified

    All numbers in the figures were computed with PySCF 2.14 (`compute_dft_data.py`, which writes the `data_*.json` files
    this notebook reads). Energies are in hartree (Ha), electronvolts (eV) or kcal/mol as stated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The label has three parts

    $$
    \underbrace{\omega\text{B97M-V}}_{\text{functional (with its dispersion)}}\;/\;\underbrace{\text{def2-TZVPD}}_{\text{basis set}}
    $$

    | | OMol25 | SPICE II |
    |---|---|---|
    | Functional | ωB97M-V | ωB97M |
    | Dispersion | part of the functional (VV10) | added afterwards (D3 with BJ damping) |
    | Basis set | def2-TZVPD | def2-TZVPPD |
    | Written as | ωB97M-V/def2-TZVPD | ωB97M-D3(BJ)/def2-TZVPPD |

    (Both labels are as I remember them; see section 10.)

    Three choices therefore define the "level of theory":

    - the **functional** decides how the electrons' interaction is approximated (sections 5 and 6),
    - the **dispersion treatment** decides how the weak attraction between neighbouring molecules is described (section 7),
    - the **basis set** decides what shapes the orbitals are built from (section 4).

    Underneath are settings that are usually *not* in the label but still change the number: the integration grid, the
    convergence thresholds and the program (section 8).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Before DFT: Hartree-Fock

    With $N$ electrons the Schrödinger equation cannot be separated, because the repulsion $1/r_{12}$ between two electrons
    depends on both positions at once. The first practical approximation is to give each electron **its own orbital**
    $\psi_i$ and to let it feel only the *average* of the others. Two ideas turn this into a method.

    **Antisymmetry.** Electrons are fermions: swapping any two of them must flip the sign of the wave function (the
    spinor notebook shows where this comes from). A product of orbitals does not do this, but a determinant does:

    $$
    \Psi(1,2,\dots,N) = \frac{1}{\sqrt{N!}}
    \begin{vmatrix}
    \psi_1(1) & \psi_2(1) & \cdots & \psi_N(1)\\
    \psi_1(2) & \psi_2(2) & \cdots & \psi_N(2)\\
    \vdots & & & \vdots\\
    \psi_1(N) & \psi_2(N) & \cdots & \psi_N(N)
    \end{vmatrix} .
    $$

    Swapping two electrons swaps two rows, which flips the sign. Putting two electrons in the same orbital makes two
    columns equal, so $\Psi = 0$: that is the Pauli exclusion principle, and it comes out for free.

    **The variational principle.** Any trial wave function has an energy at or above the true ground-state energy. So the
    best orbitals are the ones that make the energy of the determinant as low as possible. Carrying this out gives the
    **Hartree-Fock (HF) equations**, one equation per orbital:

    $$
    \hat f\,\psi_i = \varepsilon_i\,\psi_i,
    \qquad
    \hat f = -\tfrac12\nabla^2 + v_{\text{nuc}} + \hat J - \hat K .
    $$

    Here $\hat J$ is the classical repulsion from the average density of all electrons, and $\hat K$ is **exchange**: a
    correction, with no classical counterpart, that comes entirely from the antisymmetry. It removes the unphysical
    repulsion of an electron with itself that $\hat J$ alone contains, and it keeps electrons of the same spin apart.

    Written in a basis set the HF equations become the matrix equation $\mathbf F\mathbf C = \mathbf S\mathbf C\boldsymbol\varepsilon$
    of the next section, solved by the same self-consistent loop. **Kohn-Sham DFT keeps this whole structure.** The only
    difference is that the exchange term $\hat K$ (and what HF leaves out) is replaced, entirely or in part, by an
    approximate exchange-correlation term.

    ### What Hartree-Fock misses

    Each electron in HF sees only the *average* position of the others. Real electrons also dodge each other instantly,
    and the energy this saves is the **correlation energy**. It is a small fraction of the total energy, but of the same size
    as the energy differences chemistry cares about. Dispersion, the attraction between two molecules with no charge or
    hydrogen bond, is a pure correlation effect, so HF cannot produce it at all. The numbers below come from the
    calculations of this notebook (section 7 for the methane dimer, section 6 for the others).
    """)
    return


@app.cell
def _(disp_data, ladder_data, mo):
    _hf, _wb = ladder_data["functionals"]["HF"], ladder_data["functionals"]["wb97m_v"]
    _rows = [
        "| Quantity | Hartree-Fock | ωB97M-V | Reference |",
        "|---|---|---|---|",
        f"| Water dimer, interaction energy (kcal/mol) | {_hf['cp']:.2f} | {_wb['cp']:.2f} | "
        f"{ladder_data['ccsdt']['cp']:.2f} (CCSD(T)) |",
        f"| Methane dimer at 3.8 Å (kcal/mol) | {disp_data['methods']['HF']['cp']:+.2f} | "
        f"{disp_data['methods']['wb97m_v']['cp']:+.2f} | {disp_data['ccsdt']['cp']:+.2f} (CCSD(T)) |",
        f"| Water: −(energy of the highest occupied orbital) (eV) | {-_hf['homo_eV']:.1f} | {-_wb['homo_eV']:.1f} | "
        "12.6 (experiment) |",
    ]
    mo.vstack([
        mo.md("\n".join(_rows)),
        mo.md(
            "Hartree-Fock under-binds the water dimer, finds the methane dimer repulsive, and gives an orbital energy that "
            "overshoots the ionisation energy. A range-separated hybrid functional does better on all three, at a cost of "
            "the same order as Hartree-Fock's."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Two ways past Hartree-Fock

    - **Wave-function methods** (MP2, coupled cluster such as CCSD(T)) add correlation on top of the HF determinant. They
      can be very accurate, and CCSD(T) is the usual reference in this notebook, but their cost grows steeply with
      system size.
    - **Density functional theory** drops the wave function and works with the density, keeping a cost close to
      Hartree-Fock's. The price is an *approximate* exchange-correlation term, and that approximation is what the
      letters of the label describe.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. What the calculation solves: Kohn-Sham DFT

    For $N$ electrons the wave function depends on $3N$ coordinates, which is far too many for a molecule of any size.
    Density functional theory (DFT) replaces it with the electron density $\rho(\mathbf r)$, which depends on only three.
    Two facts make this possible.

    - **Hohenberg-Kohn:** the ground-state energy is determined by $\rho(\mathbf r)$ alone, so there is an energy
      *functional* $E[\rho]$.
    - **Kohn-Sham:** nobody knows $E[\rho]$ exactly, but it can be split so that only a small, well-defined piece is unknown.
      Imagine non-interacting electrons that happen to have the same density as the real ones. Their orbitals
      $\varphi_i$ give

    $$
    \rho(\mathbf r) = \sum_{i}^{\text{occupied}} \lvert\varphi_i(\mathbf r)\rvert^2 ,
    \qquad
    E[\rho] = T_s[\rho] + E_{\text{nuc}}[\rho] + E_{\text{H}}[\rho] + E_{xc}[\rho] .
    $$

    Here $T_s$ is the kinetic energy of the non-interacting electrons, $E_{\text{nuc}}$ the attraction to the nuclei and
    $E_{\text{H}}$ the classical repulsion of the density with itself. Everything that is left over, which is the part
    that makes electrons avoid each other (exchange and correlation), is gathered in $E_{xc}[\rho]$.

    **The "functional" in a DFT label is a specific approximation to $E_{xc}$.** Everything else in the sum is exact.
    The orbitals are found by the self-consistent-field (SCF) loop:

    1. guess a density,
    2. build the Kohn-Sham matrix $F$ from it,
    3. solve $F C = S C \varepsilon$ for the orbital coefficients $C$ and energies $\varepsilon$ ($S$ is the overlap of the basis functions),
    4. form the new density, and repeat until the energy stops changing.

    ### What is in the matrix $\mathbf F$

    $\mathbf F$ is the Kohn-Sham operator $\hat f = -\tfrac12\nabla^2 + v_{\text{eff}}[\rho]$ written in the basis functions
    $\phi_\mu$: $F_{\mu\nu} = \int\phi_\mu\,\hat f\,\phi_\nu\,d\mathbf r$. It is a sum of pieces,

    $$
    \mathbf F = \mathbf T + \mathbf V + \mathbf J[\mathbf D] + \mathbf V^{xc}[\mathbf D] + a\,\mathbf K[\mathbf D],
    $$

    where $\mathbf D$ is the density matrix of the occupied orbitals.

    - $\mathbf T$ is the kinetic energy and $\mathbf V$ the attraction to the nuclei: they depend only on the geometry and the basis.
    - $\mathbf J$ is the Coulomb repulsion of an electron with the whole density.
    - $\mathbf V^{xc}$ is the exchange-correlation term, evaluated by summing over the numerical grid of section 8.
    - $a\,\mathbf K$ is the fraction $a$ of exact (Hartree-Fock) exchange that a hybrid functional mixes in; $a=0$ for a pure
      functional and $a=1$ with no $\mathbf V^{xc}$ is Hartree-Fock itself. A range-separated functional lets $a$ depend on the
      distance between the two electrons (section 6).
    - The $-\mathrm{V}$ in ωB97M-V adds one more, non-local term.

    Everything except $\mathbf T$ and $\mathbf V$ depends on $\mathbf D$, which comes from solving with $\mathbf F$. That
    circularity is the reason the loop has to be repeated until it is self-consistent.

    For the first element $F_{11}$ of water at ωB97M-V/def2-TZVPD (hartree; a separate one-off PySCF check, not stored
    in the `data_*.json` files), the pieces are $T=+76.978$, $V=-99.580$, $J=+19.827$, $V^{xc}=-3.611$, VV10 $+0.005$ and
    exact exchange $-1.024$, adding up to $-7.404$. The big terms $T$, $V$ and $J$ nearly cancel; exchange-correlation is a
    small correction on top, which is why its approximation matters so much for energy *differences*.
    """)
    return


@app.cell
def _(mo, scf_figure, scf_data):
    _e = scf_data["orbital_energies_Ha"]
    mo.vstack([
        mo.md(
            f"Water at ωB97M-V/def2-TZVPD ({scf_data['n_basis']} basis functions, {scf_data['n_electrons']} electrons): "
            f"total energy **{scf_data['e_tot_Ha']:.4f} Ha**, converged in {len(scf_data['scf_history_Ha'])} cycles. "
            f"The five occupied orbitals have energies "
            f"{', '.join(f'{x:.3f}' for x in _e[:5])} Ha, and the first empty one {_e[5]:.3f} Ha."
        ),
        scf_figure(scf_data),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two sets of choices enter here that you will meet again in the labels: **how the orbitals are written down** (the
    basis set, next section) and **which approximation to $E_{xc}$ is used** (sections 5 to 7).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The basis set: def2, TZ, P, D

    The orbitals $\varphi_i$ are written as sums of fixed, atom-centred functions (the basis functions $\chi_\mu$):

    $$
    \varphi_i(\mathbf r) = \sum_\mu C_{\mu i}\,\chi_\mu(\mathbf r),
    \qquad
    \rho(\mathbf r) = \sum_{\mu\nu} P_{\mu\nu}\,\chi_\mu(\mathbf r)\,\chi_\nu(\mathbf r),
    $$

    where $P_{\mu\nu}$ is the density matrix built from the coefficients. The calculation only chooses the numbers
    $C_{\mu i}$; a bigger basis gives it more freedom. The letters describe how that freedom was built.

    | Piece | Means | Why |
    |---|---|---|
    | **def2** | the family of basis sets designed by Weigend and Ahlrichs, defined for most of the periodic table | one consistent family, so the labels `def2-SVP`, `def2-TZVP`, ... are comparable |
    | **SV, TZ, QZ** | split-valence, triple-zeta, quadruple-zeta: the valence orbital is described by 2, 3 or 4 functions of different widths, so it can swell or shrink in a bond | flexibility of the valence region |
    | **P** | polarization functions: higher angular momentum than the atom has in its ground state ($d$ on carbon, $p$ on hydrogen) | lets the density shift off the nucleus and form a bond |
    | **PP** | a second set of polarization functions (on hydrogen too) | more freedom for the bond region |
    | **D** | diffuse functions: very wide, very low-exponent functions | anions, hydrogen bonds and dispersion all live far from the nuclei |

    Each basis function is a *contracted* Gaussian, a fixed sum of several simple Gaussians
    $e^{-\alpha r^2}$ with fixed coefficients, so that it has roughly the right shape. The table below counts contracted
    shells **for one atom**; an $s$ shell is 1 function, $p$ is 3, $d$ is 5, $f$ is 7.
    """)
    return


@app.cell
def _(mo, shell_label, basis_figure, basis_data):
    _rows = ["| Basis | O shells | H shells (each) | Functions in H₂O |", "|---|---|---|---|"]
    for _b, _v in basis_data["basis"].items():
        _rows.append(f"| {_b} | {shell_label(_v['O_shells'])} | {shell_label(_v['H_shells'])} | {_v['n_basis']} |")
    mo.vstack([mo.md("\n".join(_rows)), basis_figure(basis_data)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Compare `def2-TZVPD` with `def2-TZVPPD`: they are identical on oxygen, and the difference is entirely on hydrogen,
    which gets one more $p$ shell and a $d$ shell (3s2p against 3s3p1d in the table). That is why SPICE's water has more basis functions (74) than OMol25's (58).

    ### Why the D matters: counterpoise and the basis-set superposition error

    When two molecules are close, each can "borrow" the other's basis functions to describe its own electrons better. The
    energy drops for a reason that has nothing to do with chemistry. This is the basis-set superposition error (BSSE). The
    **counterpoise** correction recomputes each monomer in the full dimer basis, using *ghost atoms* (basis functions without
    nuclei or electrons). If the raw and corrected interaction energies agree, the basis set is large enough in the
    region that matters.

    **Why a bigger basis shrinks the error.** The error does not come from the overlap of the basis functions of the two
    molecules; it comes from how much *energy a molecule gains* by using the other's functions. A molecule gains energy
    only if its own basis is missing something. The part of molecule A's electron cloud that reaches toward molecule B
    is the faint tail far from A's nuclei. A basis without diffuse functions describes that tail badly, and the nearest
    functions that can describe it are B's, so A borrows them. A diffuse function on A itself describes the tail, so
    there is nothing left to borrow. In the limit of a complete basis the gain is zero. More functions give more overlap,
    but each extra function gains less energy. The figure below shows the effect: most of the drop from `def2-TZVP` to
    `def2-TZVPD`, where only diffuse functions are added, is the removal of borrowing.

    **What a diffuse function is, and is not.** It is still centred on an atom: a Gaussian with a small exponent, so it
    is very wide. It does not sit in the space between the molecules; it reaches that space from its own atom. On oxygen the
    widest Gaussian in `def2-TZVP` falls to $1/e$ of its height at about 1.3 Å, and the one that `def2-TZVPD` adds does
    so at about 2.3 Å, which at an O···O distance of 2.9 Å nearly reaches the neighbour. Little electron density sits
    between two molecules, in contrast to a covalent bond (where the valence and polarization functions do the work).
    What matters in a non-bonded contact is:

    - **the tails of each molecule's cloud**, for example the lone pair of a hydrogen-bond acceptor reaching toward the donor
      hydrogen;
    - **how the cloud responds** to the neighbour: it is polarized by the other molecule's field, and the fluctuations of
      the two clouds produce dispersion. Distorting a cloud outward needs functions that extend far from the nucleus.

    Anions need diffuse functions for the same reason: the extra electron is loosely bound, so its density extends far
    from the nuclei.
    """)
    return


@app.cell
def _(mo, bsse_figure, basis_data):
    mo.vstack([
        bsse_figure(basis_data),
        mo.md("Same geometry as `../../qm-calculations/dimer` (rigid, not optimized). The small basis over-binds by borrowing; "
              "the diffuse functions in `TZVPD` and `TZVPPD` leave very little to correct."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The functional: Jacob's ladder, B97, M

    The approximations to $E_{xc}$ can be sorted by **what they are allowed to look at**. The standard picture is a ladder,
    where each rung adds a new ingredient:

    | Rung | Looks at | Examples |
    |---|---|---|
    | 1. LDA | the density $\rho$ at each point | SVWN |
    | 2. GGA | $\rho$ and its gradient $\lvert\nabla\rho\rvert$ | PBE, B97 |
    | 3. meta-GGA | also the kinetic energy density $\tau$ | r2SCAN, **ωB97M** (the M) |
    | 4. hybrid | also some exact (Hartree-Fock) exchange built from the orbitals | B3LYP, ωB97X, **ωB97M** |
    | 5. double hybrid | also virtual orbitals | (not used for these datasets) |

    **B97** names the *form* of the GGA part: Becke's 1997 functional writes the exchange and correlation energies as
    power series in the reduced gradient, with coefficients fitted to data. The fit is what distinguishes members of
    the family (B97, ωB97, ωB97X, ωB97X-V, ωB97M-V), since they share a shape and differ in how the
    coefficients were chosen.

    **M** means meta: the functional also uses the kinetic energy density

    $$
    \tau(\mathbf r) = \frac12\sum_i \lvert\nabla\varphi_i(\mathbf r)\rvert^2 .
    $$

    The figure shows the three ingredients on a line from the oxygen nucleus through one hydrogen. The reduced gradient
    $s$ tells the functional how rapidly the density changes (large in the tails, small in the bonding region). The
    kinetic energy density $\tau$, compared with its value $\tau_W$ for a single orbital, tells it what kind of region it
    is in: where a single orbital dominates (far outside the molecule, and in the 1s core of the oxygen) $\tau\approx\tau_W$;
    where several orbitals overlap, as in the bonding region between the nuclei, $\tau>\tau_W$. A GGA cannot make that
    distinction, a meta-GGA can.
    """)
    return


@app.cell
def _(line_fields_figure, scf_data):
    line_fields_figure(scf_data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Exact exchange and range separation: ω

    Exchange between electrons of the same spin is known exactly from Hartree-Fock theory. In DFT it is only
    approximated, and the approximation has a known flaw: an electron partly repels itself (self-interaction error). The
    cure is to mix in some exact exchange. That gives a **hybrid**: B3LYP uses 20% everywhere.

    A **range-separated hybrid** is smarter about *where*. It splits the electron repulsion in two,

    $$
    \frac1r \;=\; \underbrace{\frac{\operatorname{erfc}(\omega r)}{r}}_{\text{short range}}
    \;+\; \underbrace{\frac{\operatorname{erf}(\omega r)}{r}}_{\text{long range}},
    $$

    where $\omega$ is the splitting parameter (the **ω in ωB97M-V**, unrelated to angular frequency) and $r$ is the
    distance between the two electrons. Up close the density functional does the job; far away,
    the exact $1/r$ tail matters, so the long-range part uses 100% exact exchange. The program reports the values
    for ωB97M-V directly.
    """)
    return


@app.cell
def _(mo, split_figure, scf_data):
    mo.vstack([
        mo.md(f"For ωB97M-V PySCF reports: ω = **{scf_data['omega']:.1f}** bohr⁻¹, long-range exact exchange "
              f"**{scf_data['long_range_exchange']:.0%}**, short-range exact exchange "
              f"**{scf_data['short_range_exchange']:.0%}**."),
        split_figure(scf_data["omega"], scf_data["short_range_exchange"], scf_data["long_range_exchange"]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    A simple test of why this matters. For the exact functional, the energy of the highest occupied orbital equals minus
    the ionisation energy. Local functionals and B3LYP miss by 3 to 5 eV, because their wrong $1/r$ tail lets the outermost
    electron drift away too easily. Range-separated hybrids put the tail back and land within about 1 eV. (Hartree-Fock
    is a different case: by Koopmans' theorem its orbital energy approximates the ionisation energy, and here it overshoots.)
    """)
    return


@app.cell
def _(homo_figure, ladder_data):
    homo_figure(ladder_data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The same functionals applied to the water dimer interaction energy (counterpoise corrected, `def2-TZVPD`), against
    the coupled-cluster CCSD(T) result in the same basis, which is the usual reference:
    """)
    return


@app.cell
def _(eint_figure, ladder_data):
    eint_figure(ladder_data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The reference is itself limited by the basis (CCSD(T) in `def2-TZVPD`, not the complete-basis limit), so read the bars
    as a comparison, not as a ranking of truth. The local functionals (LDA, GGA) over-bind the hydrogen bond,
    Hartree-Fock under-binds it, and B3LYP lands closest here; its agreement owes partly to cancelling errors, since it
    has no dispersion term. ωB97M-V binds about 0.5 kcal/mol more than the reference. The weak interaction that none of
    these functionals describes without help is the subject of the next section.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Dispersion: -V against D3(BJ)

    London dispersion is an attraction between instantaneous fluctuations of the electron cloud of two molecules, falling
    off as $-C_6/r^6$. It is a *nonlocal* effect: it connects density in one place with density in another. Local
    functionals (rungs 1 to 3) and hybrid exact exchange cannot produce it. Two remedies exist, and they are the difference between
    the two labels:

    - **-V (VV10)** adds a *nonlocal correlation* term to the functional, computed from the density on the grid. The
      functional's own parameters were fitted together with it, so it is part of ωB97M-V.
    - **-D3(BJ)** adds an *empirical* pair-wise term $-\sum C_6/r^6+\dots$ after the electronic calculation, using
      tabulated $C_6$ coefficients. A damping function (here Becke-Johnson, "BJ") turns the correction off at short distance,
      where the functional already describes the electrons.

    Test case: two methanes 3.8 Å apart. They have no hydrogen bond or charge; whatever binding they have is dispersion.
    """)
    return


@app.cell
def _(dispersion_figure, disp_data):
    dispersion_figure(disp_data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Hartree-Fock and B3LYP find the methanes repulsive. Switching VV10 off in ωB97M-V leaves almost nothing. With VV10 the
    dimer binds by 0.42 kcal/mol, close to the CCSD(T) value of 0.39.

    The damped form is why "BJ" appears in the label: the bare $-C_6/r^6$ would diverge at short range.
    """)
    return


@app.cell
def _(damping_figure):
    damping_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Settings that are not in the label: the grid

    DFT energies need integrals of $E_{xc}$ over space, done numerically on a grid of points around each atom. The grid is
    given as (radial points, angular points). A calculation needs a grid fine enough that the number does not depend on it; my own water calculations use (99, 590).
    """)
    return


@app.cell
def _(mo, grid_figure, scf_data):
    _rows = ["| Grid (radial, angular) | Points for H₂O | Energy (Ha) | Time (s) |", "|---|---|---|---|"]
    for _k, _v in scf_data["grid_study"].items():
        _rows.append(f"| ({_k}) | {_v['n_points']} | {_v['e_Ha']:.6f} | {_v['seconds']:.0f} |")
    mo.vstack([mo.md("\n".join(_rows)), grid_figure(scf_data)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Meta-GGAs and nonlocal terms tend to be more sensitive to the grid than plain GGAs. The grid, the convergence criteria and the
    program (PySCF, ORCA, Psi4, Q-Chem) are not part of the label, so two groups with the "same" level of theory may differ by small amounts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Putting the two labels together

    | Piece | OMol25 | SPICE II | What it means |
    |---|---|---|---|
    | ω | yes | yes | range-separated hybrid: 15% exact exchange up close, 100% far away (ω = 0.3 bohr⁻¹) |
    | B97 | yes | yes | Becke's 1997 flexible functional form, refitted with different coefficients in each member |
    | M | yes | yes | meta: also uses the kinetic energy density $\tau$ |
    | -V | yes | no | nonlocal VV10 dispersion built into the functional |
    | D3(BJ) | no | yes | empirical pair-wise dispersion with BJ damping, added to ωB97M |
    | def2- | yes | yes | Weigend-Ahlrichs basis family |
    | TZ | yes | yes | triple-zeta valence |
    | VP | yes | no (VPP) | one set of polarization functions |
    | PP | no | yes | second set of polarization functions, which on hydrogen means $p$ and $d$ |
    | D | yes | yes | diffuse functions |

    So the two datasets use the same kind of functional and nearly the same basis. The functional parameters themselves
    differ, because ωB97M-D3(BJ) is a separate fit and not ωB97M-V with the dispersion swapped, so energies from the two
    are not directly interchangeable.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. What I have not verified

    These are from memory and should be checked against the papers before they go on a slide:

    - the exact labels: ωB97M-V/def2-TZVPD for OMol25 and ωB97M-D3(BJ)/def2-TZVPPD for SPICE II;
    - the sources for the functionals: Becke 1997 (B97), Chai and Head-Gordon 2008 (ωB97X), Mardirossian and Head-Gordon
      2016 (ωB97M-V), Vydrov and Van Voorhis 2010 (VV10), Grimme and co-workers (D3), Najibi and Goerigk 2018 (the D3(BJ)
      refit of ωB97M), Weigend and Ahlrichs 2005 (def2), Rappoport and Furche 2010 (diffuse def2);
    - that the SPICE II dispersion-corrected functional has the same exact-exchange fractions and ω as ωB97M-V (I assumed
      the same family, but did not check);
    - PySCF has no ωB97M-D3(BJ) built in, so no D3(BJ) calculation appears here; the damping figure shows the shape only.
    - program versions, convergence settings and grids used by the OMol25 and SPICE authors.
    - the pieces of $F_{11}$ quoted in section 3 come from a one-off PySCF script that is not part of `compute_dft_data.py`, and
      the statements about the cost of Hartree-Fock, DFT and CCSD(T) are general knowledge, not measured here.

    **Computed here** (not from memory): all numbers and figures in sections 2 to 8, from PySCF 2.14 on the rigid
    geometries in `compute_dft_data.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
