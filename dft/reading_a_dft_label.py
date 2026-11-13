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
    from qmstyle import ANGSTROM_PER_BOHR

    scf_data = load("scf")
    basis_data = load("basis")
    ladder_data = load("ladder")
    disp_data = load("dispersion")
    opls5_data = load("opls5")
    return (ANGSTROM_PER_BOHR, mo, plt, np, load, shell_label, basis_figure, bsse_figure, damping_figure, dispersion_figure, eint_figure, grid_figure, homo_figure, line_fields_figure, scf_figure, split_figure, scf_data, basis_data, ladder_data, disp_data, opls5_data,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reading a DFT Label: ωB97M-V/def2-TZVPD and ωB97M-D3(BJ)/def2-TZVPPD

    The datasets used to train machine-learned force fields are labelled with strings such as **ωB97M-V/def2-TZVPD**
    (OMol25) and **ωB97M-D3(BJ)/def2-TZVPPD** (SPICE II). Each piece of the string is a separate choice made by whoever
    ran the calculation. This notebook takes the labels apart one piece at a time, and each piece comes with a real
    calculation on water, so that you can see what the choice does.

    **Read first:** `../standing_waves/hartree_fock_mean_field.py` (the previous notebook: one orbital per electron in the average field of the others, the Fock
    operator $\hat f$, the SCF loop and what Hartree-Fock misses) and, behind it, `../standing_waves/basis_sets_from_base_states.py` (base states, the matrix
    equation $\mathbf H\mathbf c=E\mathbf S\mathbf c$, the variational principle, and what a basis set is). Those two
    notebooks explain the method that DFT modifies and the second half of the label; this one adds the first half and puts the pieces together. The worked PySCF
    examples in `../../qm-calculations` (water, basis functions, dimers) are what the numbers here extend.

    **Contents**

    1. The label has three parts
    2. Before DFT: Hartree-Fock (what it misses, in numbers)
    3. What the calculation solves: Kohn-Sham DFT
    4. The basis set: def2, TZ, P, D (a short return to the previous notebook, plus counterpoise)
    5. The functional: Jacob's ladder, B97, M
    6. Exact exchange and range separation: ω
    7. Dispersion: -V against D3(BJ)
    8. Settings that are not in the label: the grid
    9. Putting the two labels together (and a third source, the labels in the OPLS5 paper; which calculations optimise the geometry)
    10. What was checked, and what was not

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

    (The OMol25 label was checked against its paper and the SPICE II label against the dataset's published description; see section 10.)

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

    The method that DFT modifies is Hartree-Fock (HF), explained in the previous notebook, `hartree_fock_mean_field.py`: one orbital per electron in the average field of
    the others, the determinant that makes the product antisymmetric, the Fock operator $\hat f=-\tfrac12\nabla^2+v_{\text{nuc}}+\hat J-\hat K$ whose eigenfunctions are the orbitals
    (in a basis, $\mathbf F\mathbf C=\mathbf S\mathbf C\boldsymbol\varepsilon$), and the self-consistent loop that solves it. **Kohn-Sham DFT keeps this whole structure.**
    The only difference is that the exchange term $\hat K$ (and what HF leaves out) is replaced, entirely or in part, by an
    approximate exchange-correlation term.

    ### What Hartree-Fock misses, in numbers

    Each electron in HF sees only the *average* position of the others. The energy that real electrons save by dodging each other is the **correlation energy**
    (helium's was $1.1$ eV in `hartree_fock_mean_field.py`, section 8). It is a small fraction of the total energy, but of the same size
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
    The two ways past Hartree-Fock, wave-function methods and density functional theory, are described in `hartree_fock_mean_field.py`, section 8. This notebook follows the
    second; `coupled_cluster.py` (the last notebook) explains where the CCSD(T) reference of the table comes from.
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
    The orbitals are found by the self-consistent-field (SCF) loop of `hartree_fock_mean_field.py`, section 7: build the Kohn-Sham matrix $F$ from a density, solve
    $F C = S C \varepsilon$ for the coefficients $C$ and energies $\varepsilon$ ($S$ is the overlap of the basis functions), form the new density, and repeat until it stops changing.

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
    circularity is the reason the loop has to be repeated until it is self-consistent. `kohn_sham_matrix.py` (the next notebook)
    derives the equation and builds each of these pieces for this same water calculation, with the numbers of one matrix element; there the big terms $T$, $V$ and $J$
    nearly cancel and exchange-correlation is a small correction on top, which is why its approximation matters so much for energy *differences*.
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
    basis set, next section, which builds on the previous notebook) and **which approximation to $E_{xc}$ is used** (sections 5 to 7).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The basis set: def2, TZ, P, D

    `basis_sets_from_base_states.py` explained what a basis set is and where each letter of `def2-TZVPD` comes from (its section 6 is the decoding table, and its
    section 5 counts the functions of water). Only what a DFT label adds is covered here: how the density is written in the basis, the rows `SV`, `QZ` and `PP`,
    the basis sizes of the two datasets, and counterpoise. The orbitals $\varphi_i$ are sums of the atom-centred basis functions $\chi_\mu$ (LCAO), and the density is built from the
    density matrix $P_{\mu\nu}$ of the coefficients:

    $$
    \varphi_i(\mathbf r) = \sum_\mu C_{\mu i}\,\chi_\mu(\mathbf r),
    \qquad
    \rho(\mathbf r) = \sum_{\mu\nu} P_{\mu\nu}\,\chi_\mu(\mathbf r)\,\chi_\nu(\mathbf r).
    $$

    The calculation only chooses the numbers $C_{\mu i}$; a bigger basis gives it more freedom. The letters that this notebook adds to the table of
    `basis_sets_from_base_states.py`, section 6:

    | Piece | Means | Why |
    |---|---|---|
    | **SV, QZ** | split-valence (the valence orbital is described by 2 functions of different widths), quadruple-zeta (4 functions) | the TZ of the earlier table has 3: flexibility of the valence region |
    | **PP** | a second set of polarization functions (on hydrogen too) | more freedom for the bond region |

    The shells of each basis for water (an $s$ shell is 1 function, $p$ is 3, $d$ is 5, $f$ is 7) are in the table of `basis_sets_from_base_states.py`, section 5; the figure plots
    the resulting number of functions in H$_2$O (the same PySCF counts).
    """)
    return


@app.cell
def _(basis_figure, basis_data):
    basis_figure(basis_data)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Compare `def2-TZVPD` with `def2-TZVPPD`: they are identical on oxygen, and the difference is entirely on hydrogen,
    which gets one more $p$ shell and a $d$ shell (3s2p against 3s3p1d in the table above). That is why SPICE's water has more basis functions (74) than OMol25's (58).

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

    **What a diffuse function is, and is not.** As the previous notebook showed for hydrogen, it is still centred on an atom: a Gaussian with a small exponent, so it
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

    This is the point of the letter D in `def2-TZVPD`. The next sections leave the basis set and turn to the second ingredient of the label, the functional.
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

    The figure shows the three ingredients on a line from the oxygen nucleus through one hydrogen (the O–H bond is 0.96 Å long; the figure shows lengths in ångström, while the program works in atomic units, where 1 bohr $=0.529$ Å). The reduced gradient
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
    distance between the two electrons. Since $\omega r$ must be dimensionless, $\omega$ is an inverse length, quoted in bohr$^{-1}$ (1 bohr $=0.529$ Å, so 1 bohr$^{-1}=1.89$ Å$^{-1}$), and the figure shows $r$ in ångström. Up close the density functional does the job; far away,
    the exact $1/r$ tail matters, so the long-range part uses 100% exact exchange. The program reports the values
    for ωB97M-V directly.
    """)
    return


@app.cell
def _(ANGSTROM_PER_BOHR, mo, split_figure, scf_data):
    mo.vstack([
        mo.md(f"For ωB97M-V PySCF reports: ω = **{scf_data['omega']:.1f}** bohr⁻¹ (= {scf_data['omega'] / ANGSTROM_PER_BOHR:.2f} Å⁻¹), long-range exact exchange "
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
    | ω | yes | yes | range-separated hybrid: 15% exact exchange up close, 100% far away (ω = 0.3 bohr⁻¹ = 0.57 Å⁻¹) |
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
    ### A third source: the labels in the OPLS5 paper

    The two datasets above are large collections of DFT energies. A third, smaller example comes from a force-field paper:
    Damm et al., *OPLS5: Addition of Polarizability and Improved Treatment of Metals* (ChemRxiv 2024, doi
    10.26434/chemrxiv-2024-2svr7, a preprint, not peer reviewed). A force field is a cheap formula for the energy of
    a molecule, with parameters fitted to reference calculations. This paper uses **four different levels of theory**, one for each job:

    | Job in the paper | Label as printed | Geometry optimised in the calculation? |
    |---|---|---|
    | (a) optimise the geometries of about 500 metal complexes; electrostatic potential (ESP) with and without a uniform electric field of $0.02$ a.u. | ωB97x-D3/lacvp+* | yes: "optimise the geometries" |
    | (b) energy scans of the metal-ligand distance and of ligand rotations | ωB97x-D3/def2-tzvpd(-f) | scan points are prescribed geometries; whether the other coordinates were relaxed at each point is not stated in the source I read |
    | (c) energy of a $+1$ point charge held $3.5$ Å and $4.0$ Å above the centre of a benzene ring | ωB97X-D/6-311++G** | no: the charge is placed at fixed, prescribed positions; how the benzene geometry was obtained is not stated in the source I read |
    | (d) reference interaction energies of charged groups (acetate, guanidine, ammonium ions) | CCSD(T)/CBS | not stated in the source; the origin of the geometries is not stated in the source I read |

    The new pieces are the functional ωB97X-D3, the basis `lacvp+*` with an effective core potential, the suffix
    `(-f)`, the Pople basis `6-311++G**`, and `CBS`, which is not a basis set at all. They are taken in turn.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### ωB97X-D3 and ωB97X-D against ωB97M-V

    - **X against M-V.** ωB97X is a range-separated hybrid on the B97 form of section 5 but it has *no* kinetic energy
      density $\tau$, so it is a hybrid built on a GGA, without the meta ingredient of ωB97M. It also has no built-in
      nonlocal term (no -V). Everything it knows about dispersion has to be added from outside, which is the role of the suffix.
    - **-D against -D3.** Both add an empirical atom-pair term $-\sum C_6/R^6+\dots$ after the electronic calculation, as
      D3(BJ) did in section 7. The original **-D** uses one fixed set of $C_6$ coefficients per pair of elements; **-D3** is
      Grimme's later scheme, in which $C_6$ also depends on how many neighbours an atom has. In both, the functional's own
      coefficients are refitted together with the correction, so **ωB97X, ωB97X-D and ωB97X-D3 are three different functionals**.
    - **Range separation.** The same two exchange fractions as in section 6, now with different values for each member of the family:
    """)
    return


@app.cell
def _(ANGSTROM_PER_BOHR, mo, opls5_data):
    _rows = ["| Functional | $\\omega$ (bohr⁻¹) | $\\omega$ (Å⁻¹) | exact exchange, short range | exact exchange, long range |", "|---|---|---|---|---|"]
    for _name, _v in opls5_data["rsh"].items():
        _rows.append(f"| {_name} | {_v['omega']:.2f} | {_v['omega'] / ANGSTROM_PER_BOHR:.2f} | {_v['short_range']:.1%} | {_v['long_range']:.0%} |")
    mo.vstack([
        mo.md("\n".join(_rows)),
        mo.md("These are the values PySCF (through the libxc library) reports for the functional part of each. The dispersion terms "
              "are not in these numbers. The $\\omega=0.25$ and $19.6\\%$ of ωB97X-D3 are what I remembered, and the calculation agrees."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### lacvp+*: a Pople basis with a core potential

    Jaguar, Schrödinger's quantum chemistry program, names its basis sets with letters. According to the Jaguar manual,
    **LACVP** means: *LA* for the Los Alamos effective core potentials of Hay and Wadt, *CV* for core and valence, and *P* for the Pople
    6-31G set on the other atoms. In the manual's table, the atoms from hydrogen to argon use 6-31G, and an effective
    core potential (ECP) is used for the heavier elements: potassium to copper, rubidium to silver, caesium to lanthanum, and hafnium to gold (so both
    iridium and platinum, the metals of the paper). The remaining symbols have their usual Pople meaning:

    - **\*** adds one set of $d$ polarization functions to the atoms heavier than hydrogen (`**` would add $p$ on hydrogen as well),
    - **+** adds one set of diffuse functions on the heavy atoms (`++` would add one on hydrogen too).

    So for carbon, nitrogen or chlorine `lacvp+*` is the basis `6-31+G*`, and the metal is described by an ECP with its own valence basis.
    (The manual lists the symbols `*`, `**`, `+` and `++` for LACVP but does not define them; the definitions above are the standard Pople ones.)

    **What an ECP is.** A heavy atom such as platinum has 78 electrons. The 60 innermost are bound tightly, take no part in
    chemistry, and would need many basis functions. An effective core potential replaces them by a smooth
    potential $V_{\text{ECP}}(\mathbf r)$ added to the nuclear attraction, and the calculation treats only the remaining valence
    and semicore electrons explicitly. This helps in two ways: it removes many electrons, and the potential can be fitted to
    a relativistic atomic calculation, which includes the effect (the contraction of the heavy atom's inner orbitals) that a non-relativistic
    calculation misses. This second reason is general knowledge, not something I checked in a source for this notebook.
    Without an ECP, a transition-metal complex of a few hundred atoms would be both costlier and less accurate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### def2-tzvpd(-f)

    This is `def2-TZVPD` of section 4 with the suffix `(-f)`. The Jaguar manual says that the suffix means that **the $f$ and
    higher functions are removed**. They are the largest and most expensive shells for the main-group atoms, and
    the metal-ligand scans of job (b) need many energies. The manual lists the set for H to Kr with analytic integrals,
    and for the heavier atoms together with an ECP. What the manual does not say, in the part I read, is whether the
    metal keeps its own $f$ functions. The counts below remove them from the main-group atoms only.

    #### 6-311++G** and CBS

    - **6-311++G\*\*** is a Pople triple-zeta set: each core orbital is one contracted function of six Gaussians, and each valence
      orbital is split into **three** (the `311`). The `++` adds diffuse functions on every atom including hydrogen, and `**`
      adds polarization functions on every atom including hydrogen ($d$ on the heavy atoms, $p$ on hydrogen). This is the
      Pople counterpart of `def2-TZVPPD` in SPICE II.
    - **CBS** means complete basis set. No basis is meant: `CCSD(T)/CBS` is a coupled-cluster energy extrapolated from
      calculations in a series of growing basis sets to the limit of an infinite one. A standard form is
      $E_X = E_{\text{CBS}} + A\,X^{-3}$ for the correlation energy, where $X$ is the size index of the basis (2 for double zeta, 3 for triple, and so on). This is the textbook
      form; which basis sets and which formula the reference of Table 3 used I did not check.

    Counts of basis functions follow from these definitions. Benzene has 12 atoms and 42 electrons; the platinum complex is
    $\text{Pt(NH}_3)_2\text{Cl}_2$ in an illustrative square-planar geometry (the counts do not depend on the geometry).
    PySCF uses spherical $d$ and $f$ functions (5 and 7); a program using Cartesian Pople functions would count more.
    """)
    return


@app.cell
def _(mo, opls5_data):
    _b = opls5_data["benzene"]
    _rows = ["| Basis | Functions in benzene | Electrons treated explicitly |", "|---|---|---|"]
    for _k, _v in _b.items():
        _rows.append(f"| {_k} | {_v['n_basis']} | {_v['n_electrons']} of {_v['n_electrons_all']} |")
    _rows2 = ["| Basis for the platinum complex | Functions | Electrons treated explicitly |", "|---|---|---|"]
    for _k, _v in opls5_data["pt_complex"].items():
        _rows2.append(f"| {_k} | {_v['n_basis']} | {_v['n_electrons']} of {_v['n_electrons_all']} |")
    _d = _b["def2-TZVPD"]["n_basis"] - _b["def2-TZVPD(-f)"]["n_basis"]
    mo.vstack([
        mo.md("\n".join(_rows)),
        mo.md(f"Removing the $f$ functions takes {_d} functions out of benzene, which is $6\\times 7$: one set of seven on each carbon. "
              "`6-31+G*` is the light-atom part of `lacvp+*`; its count is for a metal-free molecule, where the ECP plays no role."),
        mo.md("\n".join(_rows2)),
        mo.md("In both ECP cases the platinum atom is left with 18 explicit electrons: 60 of 132 are replaced by the potential. "
              "The LANL2DZ row is the closest PySCF has to the Jaguar set: it is the Hay-Wadt potential with its standard valence "
              "basis on platinum and `6-31+G*` on the others, but it is **not** a check of Jaguar's `lacvp+*`, whose valence basis on the metal I did not compare."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A check we can run: the probe charge on benzene (job c)

    Job (c) puts a point charge of $+1$ above the centre of the ring, perpendicular to its plane, and asks for the
    interaction energy $E(\text{benzene}+q) - E(\text{benzene})$. The force field has to reproduce this attraction of the
    charge to the $\pi$ cloud, which it can only do with a polarizable site; the paper tunes the Drude polarizability until the numbers fit.

    PySCF has the ωB97X-D functional (under its libxc name) but cannot add the -D term. That does not matter here, because
    the -D term is a sum over pairs of *atoms* and a point charge has none, so the electronic part gives the whole
    interaction energy. This is my reasoning, not a calculation. The geometry is a standard D6h benzene
    (C-C 1.397 Å, C-H 1.087 Å) rather than the paper's, which is not given in the text.
    """)
    return


@app.cell
def _(mo, opls5_data):
    _p = opls5_data["probe"]
    _rows = ["| Distance | Paper: QM (kcal/mol) | Here, ωB97X-D/6-311++G** (kcal/mol) |", "|---|---|---|"]
    for _z in ("3.5", "4.0"):
        _rows.append(f"| {_z} Å | {_p['paper_QM'][_z]:.1f} | {_p['computed'][_z]:.2f} |")
    mo.vstack([
        mo.md("\n".join(_rows)),
        mo.md("The printed values are reproduced. A force field with no polarizability ($\\alpha=0$ in the paper's Table 1) gets "
              "only $-7.4$ and $-5.4$, about half of this; polarizability of $1.0$ Å$^3$ on the ring gives $-12.7$ and $-8.8$, "
              "the closest of the values the paper tried. The agreement to every printed digit is helped by rounding: the paper's grid and convergence settings are not stated."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Why one paper uses four levels of theory

    A force-field fit needs the *cheapest* calculation that is accurate enough for each quantity.

    - **Geometry (a)** needs thousands of optimisation steps on complexes with heavy metals: a hybrid with a modest basis and an ECP.
    - **Energy scans (b)** need accurate *relative* energies along a path, hence the larger triple-zeta basis with diffuse functions (`def2-tzvpd(-f)`),
      with the $f$ functions dropped to save cost.
    - **Probe charge (c)** is a small benzene and a charge, so the Pople basis with diffuse and polarization functions on
      every atom is cheap, and the diffuse functions are what let the $\pi$ cloud respond to the charge.
    - **Reference (d)** is the target against which everything else is judged, so it gets the most accurate method, CCSD(T) at the complete-basis limit,
      only affordable for a small cluster of a charged group and its partner.

    This is also the reason to read each label as a **pair** (method, basis) and to ask what job it was chosen for. The paper does not
    explain these choices in this way; the paragraph above is my reading of what each job needs.

    #### All labels in this notebook

    | | OMol25 | SPICE II | OPLS5 (a) | OPLS5 (b) | OPLS5 (c) | OPLS5 (d) |
    |---|---|---|---|---|---|---|
    | Method | ωB97M-V | ωB97M-D3(BJ) | ωB97X-D3 | ωB97X-D3 | ωB97X-D | CCSD(T) |
    | Type | meta-GGA, range-separated hybrid | meta-GGA, range-separated hybrid | GGA, range-separated hybrid | GGA, range-separated hybrid | GGA, range-separated hybrid | wave function |
    | Dispersion | VV10 inside | D3(BJ) added | D3 added | D3 added | -D (no effect for a point charge) | not needed |
    | Basis | def2-TZVPD | def2-TZVPPD | lacvp+* | def2-tzvpd(-f) | 6-311++G** | extrapolated to CBS |
    | ECP | yes, heavy elements | not recorded here | yes, Hay-Wadt, for the metals | yes for heavy atoms (def2 ECP) | none, no heavy atoms | none |
    | Program | ORCA 6.0.0 | Psi4 | not stated in the paper | not stated in the paper | not stated in the paper | not stated |
    | Use | training labels for ML potentials | training labels for ML potentials | geometries, ESP | metal-ligand scans | polarizability fit | reference energies |
    | Geometry optimised in the calculation? | mostly no (short optimisations of at most five steps for some subsets) | not stated in the source I could read | yes | prescribed scan points | no (fixed charge positions) | not stated |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Which calculations optimise the geometry?

    A label says which method and basis were used, not whether the geometry was optimised. The two are separate choices: a **single point** stops after one SCF at the
    given $\mathbf R$ (`hartree_fock_mean_field.py`, section 7), gives the energy and, almost for free, the forces, and takes $\mathbf R$ as an input; an **optimisation** runs many SCFs,
    one per step of the outer loop of `from_atoms_to_molecules.py`, section 2.1, and gives $\mathbf R_{\text{opt}}$ and its energy. A dataset of energies and forces needs only
    single points, whereas a geometry for a force-field fit needs the loop. What the sources say about the two datasets (read on 2026-11-12, short quotes in quotation marks;
    the OPLS5 jobs are in the table above):

    | Dataset | Geometry optimised in the calculation? | What the source says |
    |---|---|---|
    | **SPICE** (original paper, Eastman et al., *Sci. Data* 10, 11, 2023) | no DFT optimisation is stated; energies and gradients are computed on given structures | "Forces and energies for all conformations were calculated" with Psi4. Structures: 10 RDKit conformers, then "100 ps of molecular dynamics at a temperature of 500 K" (force-field MD, OpenMM/Amber14) for the high-energy set; "five iterations of L-BFGS energy minimization" (partial, "did not fully minimize it") plus 1 ps at 100 K for the low-energy partner; ion pairs on a grid of distances |
    | **SPICE II** (Eastman et al., arXiv 2406.13112) | not stated in the source | I could read only the abstract of this paper. Whether its sampling differs from SPICE 1 is unchecked |
    | **OMol25** (the dataset paper, arXiv 2505.08762) | mostly no, but not all subsets | "over 140 million DFT single-point calculations". Structures come from MD (classical and machine-learned potentials), ring-polymer MD, docking followed by MD, generated metal complexes, reaction paths and a few "rattled" structures. Some subsets were "optimized for up to five steps" (metal complexes, 3 Å electrolyte clusters, small-molecule combinations); the text I read does not say with which method |

    So the usual picture, that SPICE and OMol25 are energies and forces on structures made elsewhere (by force-field or machine-learned MD, conformer
    generators and reaction-path tools), holds, with one correction: the OMol25 paper describes short optimisations of at most five steps for some of
    its subsets. Neither dataset is a set of relaxed minima, and that is deliberate: a force field or a machine-learned potential has to be accurate away from
    the minimum too, because MD visits strained geometries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. What was checked, and what was not

    **Checked on 2026-11-05 against published sources:**

    - **OMol25** (the Open Molecules 2025 paper, arXiv 2505.08762): the level of theory is ωB97M-V/def2-TZVPD, run with
      ORCA 6.0.0 using RI-J and COSX acceleration, ORCA's `DEFGRID3` grid (a pruned grid with 590 angular points for
      exchange-correlation and 302 for the COSX grid; the radial count is not stated), tight convergence, and effective core
      potentials for the heavy elements. Open-shell and charged systems are run with unrestricted Kohn-Sham. So the grid
      of this notebook, (99, 590) in PySCF, is similar in spirit but not the same setting.
    - **SPICE II:** the published dataset records give ωB97M-D3(BJ)/def2-TZVPPD, run in Psi4. I could not read the
      SPICE II paper itself (the fetched PDF was unreadable), so the Psi4 version (1.8.x in the paper, against 1.4.1 in
      another record) is not settled.
    - **ωB97M-V:** Mardirossian and Head-Gordon, J. Chem. Phys. 144 (2016), with VV10 nonlocal correlation, as the label says.
    - **ωB97M-D3(BJ) in Psi4** is described as "ωB97M-V with D3(BJ) instead of VV10", and Psi4 lists the same ω (0.3) and
      the same short-range exact-exchange fraction (0.15) as for ωB97M-V.

    **Still from memory:**

    - whether the other fitted parameters of ωB97M-D3(BJ) were also refit when D3(BJ) replaced VV10, and whether Najibi
      and Goerigk is where that functional is defined (the reference itself now exists, see the next list: J. Chem. Theory
      Comput. 14, 5725, 2018, but I have not read it);
    - the damping figure in section 7 shows the shape only: PySCF has no ωB97M-D3(BJ) built in, so no D3(BJ) calculation
      appears here;
    - the statements about the cost of Hartree-Fock, DFT and CCSD(T) are general knowledge, not measured here. (The matrix elements of the Kohn-Sham matrix are in
      `data_ks.json`, section 4 of `kohn_sham_matrix.py`.)

    **Bibliographic records checked on 2026-11-13** (title, authors, year, journal, volume and pages compared with the
    Crossref record of the DOI; the papers themselves were **not** read, so only the references are confirmed, not what they say):

    - Becke 1997 (B97), J. Chem. Phys. 107, 8554, doi 10.1063/1.475007;
    - Chai and Head-Gordon 2008 (ωB97X), J. Chem. Phys. 128, 084106, doi 10.1063/1.2834918, and (ωB97X-D) Phys. Chem. Chem. Phys. 10, 6615, doi 10.1039/b810189b;
    - Vydrov and Van Voorhis 2010 (VV10), J. Chem. Phys. 133, 244103, doi 10.1063/1.3521275;
    - Grimme, Antony, Ehrlich and Krieg 2010 (D3), J. Chem. Phys. 132, 154104, doi 10.1063/1.3382344, and Grimme, Ehrlich and Goerigk 2011 (the Becke-Johnson damping, D3(BJ)), J. Comput. Chem. 32, 1456, doi 10.1002/jcc.21759;
    - Weigend and Ahlrichs 2005 (def2), Phys. Chem. Chem. Phys. 7, 3297, doi 10.1039/b508541a; Rappoport and Furche 2010 (diffuse def2 functions), J. Chem. Phys. 133, 134105, doi 10.1063/1.3484283; Weigend 2006 (Coulomb-fitting sets), Phys. Chem. Chem. Phys. 8, 1057, doi 10.1039/b515623h;
    - Najibi and Goerigk 2018, J. Chem. Theory Comput. 14, 5725, doi 10.1021/acs.jctc.8b00842, and Lin, Li, Mao and Chai (ωB97X-D3), J. Chem. Theory Comput. 9, 263, doi 10.1021/ct300715s;
    - Hay and Wadt 1985, J. Chem. Phys. 82, 270, doi 10.1063/1.448799 (the transition-metal potentials, Sc to Hg);
    - the OMol25 paper (arXiv 2505.08762), SPICE (Sci. Data 10, 11, 2023, doi 10.1038/s41597-022-01882-6) and OPLS5 (ChemRxiv, doi 10.26434/chemrxiv-2024-2svr7) exist with those titles. The SPICE II paper is probably the JCTC 2024 article "Nutmeg and SPICE" (doi 10.1021/acs.jctc.4c00794), but that is not confirmed.

    **The OPLS5 labels (section 9, third source), checked on 2026-11-05:**

    - the four labels and what each was used for were read from the text of the ChemRxiv preprint (doi
      10.26434/chemrxiv-2024-2svr7, version 1, 13 August 2024; I read a local PDF copy and found the title and date through a search);
      the programs used are not named there.
    - **Jaguar manual** (a mirror of Schrödinger's user manual, page "Basis Sets", no version number on the page): LACVP is the Los Alamos
      (Hay-Wadt) ECP with 6-31G on H to Ar; the ECP elements include Hf to Au; the suffix `(-f)` removes $f$ and higher functions; `def2-tzvpd(-f)` is a listed option.
    - **Lin, Li, Mao and Chai,** J. Chem. Theory Comput. 9 (2013) 263 (ωB97X-D3): the title, journal, volume, pages and the statement that it
      reoptimises ωB97X-D with improved D3 dispersion were confirmed by search results (arXiv 1211.0387); I did not read the paper.
    - **Computed here:** exchange fractions of ωB97X, ωB97X-D, ωB97X-D3 and ωB97M-V (PySCF/libxc), the benzene and platinum-complex function counts,
      the 60-electron core of Pt in both ECPs, and the probe-charge energies ($-13.61$ and $-9.52$ kcal/mol), which match the paper's Table 1.

    **Still from memory in the OPLS5 part:** the meaning of `*` and `+` (standard Pople notation; the manual does not define them for LACVP);
    the original references of ωB97X and ωB97X-D (Chai and Head-Gordon, J. Chem. Phys. 2008 and Phys. Chem. Chem. Phys. 2008) and of the Pople
    6-311G set; the claim that D3 coefficients depend on the number of neighbours; the reasons an ECP is used (relativistic effects);
    the form $A\,X^{-3}$ of the extrapolation, and which one the CCSD(T)/CBS values of the paper's Table 3 (taken from the OPLS4 paper by Lu et al.) used.
    The paper cites Hay and Wadt, J. Chem. Phys. 82 (1985) 270 and Weigend, Phys. Chem. Chem. Phys. 8 (2006) 1057 (a Coulomb-fitting set, an
    auxiliary basis for the density) for its labels, but not the Weigend-Ahlrichs paper that defines def2 itself. I did not open the ACS, ChemRxiv or Wiley
    pages of these papers on 2026-11-05; the bibliographic records of the Chai-Head-Gordon, Hay-Wadt and Weigend papers were checked on 2026-11-13 (previous list), but their contents are not verified.

    **Which calculations optimise the geometry (section 9), read on 2026-11-12:** the SPICE paper (PMC full text) and the OMol25 paper (arXiv HTML version 2), each quoted
    in the table. For SPICE II the sampling is not stated in any source I could read (only the abstract). For OMol25 I could not read the appendix, so which method does
    the five-step optimisations is open; OPLS5 (b) to (d) were not re-read in the preprint beyond what is quoted above.

    **Computed here** (not from memory): all numbers and figures in sections 2 to 8 and the tables of section 9 marked as computed, from PySCF 2.14 on the rigid
    geometries in `compute_dft_data.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Next.** Section 3 stated the equation $FC=SC\varepsilon$ and listed what is in $F$ without deriving either.
    `kohn_sham_matrix.py` does that for the same water calculation, term by term. After it, `coupled_cluster.py` explains the
    method that this notebook used as its reference (CCSD(T)).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading

    Free or open-access sources where possible; papers are given by DOI and marked when they need publisher access. Checked on 2026-11-13; the full table, with what each source covers, is in `SOURCES.md`. For papers only the bibliographic record was checked, not the content.

    **Textbook-style** (Hartree-Fock itself is in the further reading of `hartree_fock_mean_field.py`)

    - [The ABC of DFT (K. Burke, free draft)](https://dft.uci.edu/learnDFT.php) and [Review of approximations for the exchange-correlation energy (J. Toulouse, arXiv 2103.02645)](https://arxiv.org/abs/2103.02645): Kohn-Sham theory, Jacob's ladder, range separation (intro to reference).
    - [Best-practice DFT protocols for basic molecular computational chemistry (Bursch, Mewes, Hansen, Grimme, 2022)](https://doi.org/10.1002/anie.202205735) and [Thirty years of density functional theory in computational chemistry (Mardirossian and Head-Gordon, 2017)](https://doi.org/10.1080/00268976.2017.1333644): choosing functionals, basis sets and dispersion corrections (reference).
    - [PySCF: density functional theory](https://pyscf.org/user/dft.html) (integration grids), [PySCF: molecular structure](https://pyscf.org/user/gto.html) (basis sets and ECPs), [Psi4: DFT functionals](https://psicode.org/psi4manual/master/dft_byfunctional.html) (what each label means in code) and [Psi4: basis set superposition corrections](https://psicode.org/psi4manual/master/nbody.html) (counterpoise).
    - [Basis Set Exchange](https://www.basissetexchange.org/) and [Basis sets in quantum chemistry (Sherrill)](https://vergil.chemistry.gatech.edu/basis-sets.pdf): look up and understand a basis set (reference, intro).

    **Primary papers (DOI)**

    - Kohn-Sham theory: [Hohenberg and Kohn 1964](https://doi.org/10.1103/PhysRev.136.B864), [Kohn and Sham 1965](https://doi.org/10.1103/PhysRev.140.A1133), [Perdew's Jacob's ladder, 2001](https://doi.org/10.1063/1.1390175).
    - Functionals: [B97 (Becke 1997)](https://doi.org/10.1063/1.475007), [ωB97X (Chai and Head-Gordon 2008)](https://doi.org/10.1063/1.2834918), [ωB97X-D](https://doi.org/10.1039/b810189b), [ωB97X-D3 (Lin et al. 2013)](https://doi.org/10.1021/ct300715s), [ωB97M-V (Mardirossian and Head-Gordon 2016)](https://doi.org/10.1063/1.4952647), [Najibi and Goerigk 2018](https://doi.org/10.1021/acs.jctc.8b00842).
    - Dispersion: [VV10 (Vydrov and Van Voorhis 2010)](https://doi.org/10.1063/1.3521275), [D3 (Grimme et al. 2010)](https://doi.org/10.1063/1.3382344), [D3(BJ) (Grimme, Ehrlich and Goerigk 2011)](https://doi.org/10.1002/jcc.21759), [review (Grimme et al. 2016)](https://doi.org/10.1021/acs.chemrev.5b00533).
    - Basis sets: [def2 (Weigend and Ahlrichs 2005)](https://doi.org/10.1039/b508541a), [diffuse def2 (Rappoport and Furche 2010)](https://doi.org/10.1063/1.3484283), [Coulomb fitting (Weigend 2006)](https://doi.org/10.1039/b515623h), [counterpoise (Boys and Bernardi 1970)](https://doi.org/10.1080/00268977000101561).
    - Grids and core potentials: [Treutler and Ahlrichs 1995](https://doi.org/10.1063/1.469408), [SG-1 (Gill, Johnson and Pople 1993)](https://doi.org/10.1016/0009-2614%2893%2980125-9), [SG-2 and SG-3 (Dasgupta and Herbert 2017)](https://doi.org/10.1002/jcc.24761), [Hay and Wadt 1985](https://doi.org/10.1063/1.448799), [relativistic pseudopotentials review (Dolg and Cao 2012)](https://doi.org/10.1021/cr2001383).
    - Datasets and force field: [OMol25 (arXiv)](https://arxiv.org/abs/2505.08762), [SPICE (Eastman et al. 2023)](https://doi.org/10.1038/s41597-022-01882-6), [Nutmeg and SPICE (Eastman et al. 2024)](https://doi.org/10.1021/acs.jctc.4c00794), [OPLS5 (ChemRxiv)](https://doi.org/10.26434/chemrxiv-2024-2svr7).
    """)
    return


if __name__ == "__main__":
    app.run()
