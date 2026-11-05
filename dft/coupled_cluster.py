import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from cc_numerics import (
        ccsd,
        coefficient,
        combine_noninteracting,
        ci_solve,
        excitation_level_weights,
        fci_dimension,
        load_cc_json,
        load_system,
        mp2_energy,
        triples_correction,
    )
    from cc_plots import (
        convergence_figure,
        determinant_growth_figure,
        interaction_figure,
        level_weights_figure,
        product_figure,
        recovery_figure,
        scaling_figure,
        size_consistency_figure,
    )
    from dft_data import HA2KCAL, load

    cc_data = load_cc_json()
    ladder_data = load("ladder")
    disp_data = load("dispersion")
    sys_h2 = load_system("h2_631g")
    sys_w1 = load_system("water_sto3g")
    sys_w2 = load_system("water_631g")
    res_h2, res_w1, res_w2 = ccsd(sys_h2), ccsd(sys_w1), ccsd(sys_w2)
    res_w2_plain = ccsd(sys_w2, diis=False)
    return (
        HA2KCAL, cc_data, ccsd, ci_solve, coefficient, combine_noninteracting, convergence_figure,
        determinant_growth_figure, disp_data, excitation_level_weights, fci_dimension, interaction_figure,
        ladder_data, level_weights_figure, mo, mp2_energy, np, product_figure, recovery_figure, res_h2, res_w1,
        res_w2, res_w2_plain, scaling_figure, size_consistency_figure, sys_h2, sys_w1, sys_w2, triples_correction,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Coupled Cluster Theory: Why CCSD(T) Is the Reference

    The Hartree-Fock section of the DFT notebook ends with two ways past Hartree-Fock: density functional theory, and
    wave-function methods that add the missing electron correlation on top of the Hartree-Fock determinant. The DFT
    notebook uses CCSD(T), a coupled cluster method, as the reference for the water dimer and the methane dimer. This
    notebook explains what that name means: where the method comes from, the equations behind it (with the derivation
    in small steps), a short working implementation, and what it costs.

    **Read first:** `reading_a_dft_label.py`, section 2 (Hartree-Fock: Slater determinant, exchange, what HF misses).

    **Contents**

    1. The problem: one determinant, the others, and why exact is too expensive
    2. Truncating the exact answer (CISD) breaks size consistency
    3. The coupled cluster idea: an exponential of excitations
    4. Deriving the working equations
    5. A from-scratch implementation, checked against PySCF
    6. The (T) correction
    7. What it costs
    8. How accurate: the ladder, and the link back to DFT
    9. What I have not verified, and four lines for a slide

    **How to read the claims.** *Checked here* means reproduced by the numbers in this notebook, which come from a
    numpy implementation (`cc_numerics.py`) tested against PySCF 2.14 in `test_cc.py`, or from the stored PySCF runs
    (`compute_cc_data.py`). *From memory* means textbook knowledge that this notebook does not test, such as the cost
    exponents. Energies are in hartree (Ha) unless stated.

    **Notation.** Occupied orbitals are labelled $i,j,k,l$, unoccupied ("virtual") orbitals $a,b,c,d$, and $p,q,r,s$
    stand for either. These letters are orbital labels here, and not the complex amplitudes $a$ of the other
    notebooks. $\Phi_0$ is the Hartree-Fock determinant (the notebooks on spin call a determinant $\Psi$; here the
    exact wave function is $\Psi$ and a single determinant is $\Phi$).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The problem: one determinant, the others, and why exact is too expensive

    Hartree-Fock gives the best single Slater determinant $\Phi_0$ built from $N$ occupied spin orbitals (an orbital
    together with a spin, up or down). In a basis set of $M$ spatial orbitals, Hartree-Fock also produces
    $2M-N$ **virtual** (unoccupied) spin orbitals. Every way of moving electrons from occupied to virtual orbitals
    gives another determinant:

    - $\Phi_i^a$: one electron moved from occupied $i$ to virtual $a$ (a **single** excitation),
    - $\Phi_{ij}^{ab}$: two electrons moved, from $i,j$ to $a,b$ (a **double**),
    - triples $\Phi_{ijk}^{abc}$, quadruples, and so on up to all $N$ electrons moved.

    Together these determinants span every antisymmetric wave function that can be written in the basis. The exact
    answer *in that basis* is a combination of all of them,

    $$
    \Psi_{\text{FCI}} = c_0\,\Phi_0 + \sum_{ia} c_i^a\,\Phi_i^a + \sum_{i<j,\,a<b} c_{ij}^{ab}\,\Phi_{ij}^{ab}
    + \sum c_{ijk}^{abc}\,\Phi_{ijk}^{abc} + \cdots,
    $$

    and finding the coefficients is called **full configuration interaction (FCI)**: write the Hamiltonian as a matrix
    between all determinants and take its lowest eigenvector. The **correlation energy** in the basis is
    $E_{\text{corr}} = E_{\text{FCI}} - E_{\text{HF}}$, the energy that the single determinant misses.

    The cost is the problem. The number of determinants for $N_\alpha$ up-spin and $N_\beta$ down-spin electrons in $M$
    spatial orbitals is $\binom{M}{N_\alpha}\binom{M}{N_\beta}$, which grows faster than any power of $M$. Below, the
    count for water ($N_\alpha=N_\beta=5$) in the basis sets used in this notebook.
    """)
    return


@app.cell
def _(cc_data, ci_solve, determinant_growth_figure, excitation_level_weights, fci_dimension, level_weights_figure, mo, sys_w1, HA2KCAL):
    _fci = ci_solve(sys_w1)
    _w = excitation_level_weights(_fci, 10)
    _rows = ["| Basis for water | Spatial orbitals $M$ | Determinants in FCI ($\\alpha=\\beta=5$) |", "|---|---|---|"]
    for _r in cc_data["fci_counts"]:
        _rows.append(f"| {_r['basis']} | {_r['n_basis']} | {_r['n_det']:,} |")
    fci_w1_dim = len(_fci["dets"])
    mo.vstack([
        mo.md("\n".join(_rows)),
        determinant_growth_figure(cc_data["fci_counts"]),
        mo.md(
            f"**Checked here.** Water in the smallest basis (STO-3G, {sys_w1['n_so'] // 2} spatial orbitals) is small enough for "
            f"FCI with the numpy code: {fci_w1_dim} determinants (equal numbers of up and down electrons; "
            f"$\\binom{{7}}{{5}}^2 = {fci_dimension(7, 5, 5)}$). The result is {_fci['e_corr']:+.5f} Ha of correlation energy "
            f"({_fci['e_corr'] * HA2KCAL:+.1f} kcal/mol), the same as PySCF's FCI to better than $10^{{-8}}$ Ha. "
            f"For the 6-31G basis the table gives 1.7 million determinants, which PySCF solves in a couple of seconds; "
            f"for cc-pVDZ it is about $10^9$, and for cc-pVTZ about $10^{{13}}$."
        ),
        level_weights_figure(_w),
        mo.md(
            f"The weights show what a good approximation must capture. The reference determinant carries "
            f"{100 * _w[0]:.1f}% of the wave function, the double excitations almost all of the remainder "
            f"({100 * _w[2]:.2f}%), and singles, triples and quadruples together about {100 * (_w[1] + _w[3] + _w[4]):.2f}%. "
            f"(Singles are small because Hartree-Fock orbitals are already the best single-determinant ones; they do "
            f"matter once the doubles are present.) So most of the exact answer is *the reference plus pairs of electrons "
            f"moved together*. The question for the next two sections is how to keep the doubles and drop the rest "
            f"without breaking something else."
        ),
    ])
    return (fci_w1_dim,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Truncating the exact answer (CISD) breaks size consistency

    The obvious cut is to keep only the determinants up to doubles and find the best coefficients in that smaller space:

    $$
    \Psi_{\text{CISD}} = c_0\,\Phi_0 + \sum_{ia} c_i^a\,\Phi_i^a + \sum_{i<j,\,a<b} c_{ij}^{ab}\,\Phi_{ij}^{ab},
    $$

    with the coefficients from the lowest eigenvector of the Hamiltonian matrix in this space. This is **CISD**
    (configuration interaction with singles and doubles). The energy is variational (never below the exact one in the
    basis), and the cost is far below FCI.

    It has a serious flaw. A method is **size consistent** if the energy of two molecules infinitely far apart equals the
    sum of the energies of each alone, $E(A+B) = E(A)+E(B)$. This is not optional for chemistry: an interaction energy
    is $E(AB)-E(A)-E(B)$, and the same method has to give zero when the molecules do not interact.

    **Why CISD fails.** For two far-apart molecules $A$ and $B$ the exact wave function contains the situation "$A$ has a
    double excitation *and* $B$ has a double excitation at the same time". Seen from the pair, that is one *quadruple*
    excitation, which CISD does not contain. CISD can describe correlation in $A$ or in $B$, but not both at once,
    so correlation in $A$ gets squeezed out by correlation in $B$ and the energy of the pair is too high.

    **A test with two H$_2$ molecules.** The numpy code builds the pair as two copies of an H$_2$ molecule
    (bond length 1.0 Å, 6-31G basis) with *no* integral connecting them, which is exactly the limit of infinite separation.
    """)
    return


@app.cell
def _(ci_solve, ccsd, combine_noninteracting, mo, mp2_energy, res_h2, sys_h2):
    pair = combine_noninteracting(sys_h2, sys_h2)
    _one_cisd = ci_solve(sys_h2, level=2)["e_corr"]
    _two_cisd = ci_solve(pair, level=2)["e_corr"]
    res_pair = ccsd(pair)
    _one_fci = ci_solve(sys_h2)["e_corr"]
    _two_fci = ci_solve(pair)["e_corr"]
    _rows = [
        "| Correlation energy (Ha) | One H$_2$ | Two H$_2$, no interaction | Two minus twice one (mHa) |",
        "|---|---|---|---|",
        f"| MP2 | {mp2_energy(sys_h2):.6f} | {mp2_energy(pair):.6f} | {1000 * (mp2_energy(pair) - 2 * mp2_energy(sys_h2)):+.4f} |",
        f"| CISD | {_one_cisd:.6f} | {_two_cisd:.6f} | **{1000 * (_two_cisd - 2 * _one_cisd):+.4f}** |",
        f"| CCSD | {res_h2['e_corr']:.6f} | {res_pair['e_corr']:.6f} | **{1000 * (res_pair['e_corr'] - 2 * res_h2['e_corr']):+.6f}** |",
        f"| FCI | {_one_fci:.6f} | {_two_fci:.6f} | {1000 * (_two_fci - 2 * _one_fci):+.6f} |",
    ]
    mo.md(
        "\n".join(_rows) + f"""

**Checked here.** The Hartree-Fock energy of the pair is exactly twice that of one molecule. MP2, CCSD and FCI are additive. CISD
is not: the pair comes out {1000 * (_two_cisd - 2 * _one_cisd):.2f} mHa too high ({(_two_cisd - 2 * _one_cisd) * 627.5:.2f} kcal/mol), an error
for molecules that do not touch. (For one H$_2$, which has two electrons, CISD is exact, so the one-molecule columns of CISD, CCSD and FCI agree.)
CCSD, the method of the next section, is additive to $10^{{-9}}$ Ha and equals FCI for the pair.
"""
    )
    return pair, res_pair


@app.cell
def _(cc_data, mo, size_consistency_figure):
    mo.vstack([
        mo.md(
            "The same effect with real molecules: PySCF supermolecule calculations of two H$_2$ molecules at growing "
            "separation (the pair is one calculation with all four atoms). The plotted quantity is the correlation energy "
            "of the pair minus twice that of a single molecule. CCSD tends to zero, as it should. CISD levels off at a "
            "positive value and stays there, which is the size-consistency error."
        ),
        size_consistency_figure(cc_data["h2_dimer"]["scan"], cc_data["h2_dimer"]["monomer"]["e_ccsd_corr"]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The coupled cluster idea: an exponential of excitations

    The pair example shows what a good wave function must do: when $A$ and $B$ do not interact, the wave function must
    be a **product**, $\Psi_{AB}=\Psi_A\Psi_B$. Products of excitations on different molecules then appear on their own,
    and the amplitude of "double on $A$ and double on $B$" is the product of the two separate double amplitudes.
    The exponential function is the one that turns sums into products, $e^{x+y}=e^xe^y$, so coupled cluster
    writes the wave function as an exponential of excitation operators.

    **Excitation operators.** Let $a_p^\dagger$ put an electron into the spin orbital $p$ and $a_p$ remove one from it.
    The operator that makes all single excitations, each with its own number (an **amplitude**) $t_i^a$, is
    $\hat T_1$, and the one for double excitations, with amplitudes $t_{ij}^{ab}$, is $\hat T_2$:

    $$
    \hat T_1 = \sum_{ia} \textcolor{#6B913C}{t_i^a}\, a_a^\dagger a_i ,
    \qquad
    \hat T_2 = \frac14\sum_{ijab} \textcolor{#F14124}{t_{ij}^{ab}}\, a_a^\dagger a_b^\dagger a_j a_i ,
    \qquad
    \hat T = \hat T_1 + \hat T_2 + \hat T_3 + \cdots
    $$

    so that $\hat T_1\Phi_0=\sum t_i^a\Phi_i^a$ and $\hat T_2\Phi_0=\sum_{i<j,\,a<b} t_{ij}^{ab}\Phi_{ij}^{ab}$. The amplitudes change
    sign when two occupied or two virtual labels are swapped, $t_{ij}^{ab}=-t_{ji}^{ab}=-t_{ij}^{ba}$, which is why the factor
    $\tfrac14$ appears: the sum over all labels counts each distinct double four times. The coupled cluster wave function is

    $$
    \Psi_{\text{CC}} = e^{\hat T}\,\textcolor{#4E67C8}{\Phi_0}
    = \Bigl(1 + \hat T + \tfrac12\hat T^2 + \tfrac1{3!}\hat T^3 + \cdots\Bigr)\textcolor{#4E67C8}{\Phi_0}.
    $$

    The wave function is normalised so that $\langle\Phi_0\vert\Psi_{\text{CC}}\rangle=1$ (called intermediate
    normalisation); the coefficient of $\Phi_0$ is one and everything else is relative to it.

    ### What the exponential contains

    Keep only singles and doubles, $\hat T=\hat T_1+\hat T_2$ (**CCSD**), and expand by the number of electrons moved.
    The operators commute with each other (every one of them only moves electrons from occupied to virtual orbitals), so the
    powers can be expanded like numbers:

    | Excitation level | Terms of $e^{\hat T_1+\hat T_2}\Phi_0$ at that level |
    |---|---|
    | 1 (singles) | $\hat T_1$ |
    | 2 (doubles) | $\hat T_2 + \tfrac12\hat T_1^2$ |
    | 3 (triples) | $\hat T_1\hat T_2 + \tfrac1{6}\hat T_1^3$ |
    | 4 (quadruples) | $\tfrac12\hat T_2^2 + \tfrac12\hat T_1^2\hat T_2 + \tfrac1{24}\hat T_1^4$ |
    | 5, 6, ... | further products, all the way to every electron moved |

    Two things follow. First, with only $\hat T_1$ and $\hat T_2$ to determine, **CCSD has the same number of unknowns as
    CISD** (one per single and per double), yet its wave function reaches determinants with any number of excitations.
    The coefficients of triples and higher are not free; they are *products* of the singles and doubles amplitudes, such as
    $\tfrac12\hat T_2^2$ for the quadruples. These are the "disconnected" contributions: two pairs of electrons that
    are excited independently. What CCSD leaves out are the *connected* triples, quadruples and so on, which have their
    own independent amplitudes ($\hat T_3$, $\hat T_4$, ...).

    Second, **size consistency is automatic.** If molecules $A$ and $B$ do not interact, then $\hat H=\hat H_A+\hat H_B$
    and the cluster operator splits into $\hat T=\hat T_A+\hat T_B$ (excitations on $A$ and excitations on $B$). These
    commute, so

    $$
    e^{\hat T}\,\Phi_0 = e^{\hat T_A+\hat T_B}\,\Phi_A\Phi_B = \bigl(e^{\hat T_A}\Phi_A\bigr)\bigl(e^{\hat T_B}\Phi_B\bigr),
    $$

    a product of the wave functions of $A$ and $B$, whose energy is the sum. If $\hat T_A$ and $\hat T_B$ contain only singles
    and doubles then so does their sum, so truncating at CCSD does not spoil this. (CISD, in contrast, is a *linear*
    combination whose truncation is not preserved by products.) This is the numerical content of section 2.
    """)
    return


@app.cell
def _(ci_solve, coefficient, mo, pair, product_figure, res_h2, res_pair):
    _fci = ci_solve(pair)
    c_a = coefficient(_fci, [(0, 4), (1, 5)])
    c_b = coefficient(_fci, [(2, 10), (3, 11)])
    c_quad = coefficient(_fci, [(0, 4), (1, 5), (2, 10), (3, 11)])
    mo.vstack([
        product_figure(c_a, c_b, c_quad),
        mo.md(
            f"**Checked here.** In the exact (FCI) wave function of two independent H$_2$ molecules, the coefficient of the "
            f"determinant with *both* molecules doubly excited is {abs(c_quad):.6f} in size, and the product of the two "
            f"single-molecule double coefficients, {abs(c_a):.6f} and {abs(c_b):.6f}, is {abs(c_a * c_b):.6f}: they agree to "
            f"six digits. This is the $\\tfrac12\\hat T_2^2$ term at work, and it is why CCSD needs no quadruples of its own for the pair. "
            f"The CCSD energy of the pair, {res_pair['e_corr']:.6f} Ha, is twice that of one molecule ({2 * res_h2['e_corr']:.6f} Ha)."
        ),
    ])
    return c_a, c_b, c_quad


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Deriving the working equations

    The wave function $e^{\hat T}\Phi_0$ has to satisfy the Schrödinger equation, and the unknowns are the amplitudes.
    This section gets the equations for them in steps. All quantities are in the spin-orbital basis.

    ### Step 1. The Hamiltonian in terms of integrals

    For a given set of spin orbitals $\chi_p$ the electronic Hamiltonian is

    $$
    \hat H = \sum_{pq} h_{pq}\,a_p^\dagger a_q + \frac14\sum_{pqrs}\langle pq\Vert rs\rangle\,a_p^\dagger a_q^\dagger a_s a_r ,
    \qquad
    \langle pq\Vert rs\rangle = \langle pq\vert rs\rangle - \langle pq\vert sr\rangle ,
    $$

    where $h_{pq}$ holds the kinetic energy of an electron and its attraction to the nuclei, and
    $\langle pq\vert rs\rangle=\iint\chi_p^*(1)\chi_q^*(2)\,r_{12}^{-1}\,\chi_r(1)\chi_s(2)\,d1\,d2$ is the repulsion between two
    electrons (a pair in $p,q$ going to $r,s$). The combination $\langle pq\Vert rs\rangle$, called the **antisymmetrised
    integral**, builds the exchange term in at once. The Fock matrix is $f_{pq}=h_{pq}+\sum_k\langle pk\Vert qk\rangle$ (sum over
    occupied $k$). With Hartree-Fock orbitals $f$ is diagonal, $f_{pp}=\varepsilon_p$ (the orbital energies), and
    $f_{ia}=0$.

    ### Step 2. Move the exponential to the Hamiltonian

    Put $\Psi=e^{\hat T}\Phi_0$ into $\hat H\Psi=E\Psi$ and multiply from the left by $e^{-\hat T}$:

    $$
    \bar H\,\Phi_0 = E\,\Phi_0,
    \qquad
    \bar H = e^{-\hat T}\,\hat H\,e^{\hat T} .
    $$

    $\bar H$ is the **similarity-transformed Hamiltonian**. A similarity transformation does not change the eigenvalues,
    so $E$ is still the energy. The price is that $\bar H$ is not Hermitian. The gain is the next step.

    ### Step 3. Project

    $\bar H\Phi_0=E\Phi_0$ says that $\bar H\Phi_0$ is $E$ times the reference and has no component on any excited determinant.
    Take the overlap with each determinant in turn. Because $\Phi_0$ is orthogonal to the excited ones, the right side only
    survives for $\Phi_0$:

    $$
    E = \langle\Phi_0\vert\bar H\vert\Phi_0\rangle ,
    \qquad
    0 = \langle\Phi_i^a\vert\bar H\vert\Phi_0\rangle ,
    \qquad
    0 = \langle\Phi_{ij}^{ab}\vert\bar H\vert\Phi_0\rangle .
    $$

    The second and third lines are one equation for each single and for each double, the same number as unknown
    amplitudes $t_i^a$ and $t_{ij}^{ab}$. Together they determine the amplitudes (CCSD); leaving out the singles
    equations and the $\hat T_1$ amplitudes gives **CCD**. We do *not* require the equations for triples and beyond,
    since there are no triples amplitudes to fix.

    These projected equations do not minimise the energy, so unlike CISD the coupled cluster energy is not an upper
    bound to the exact energy (*from memory*; section 8 shows an example where it goes below the exact value).
    That choice is what keeps the equations finite, as the next step shows.

    ### Step 4. Expand $\bar H$: the series stops

    The Baker-Campbell-Hausdorff formula expands $\bar H$ into nested commutators:

    $$
    \bar H = \hat H + [\hat H,\hat T] + \frac1{2!}\bigl[[\hat H,\hat T],\hat T\bigr]
    + \frac1{3!}\Bigl[\bigl[[\hat H,\hat T],\hat T\bigr],\hat T\Bigr]
    + \frac1{4!}\Bigl[\Bigl[\bigl[[\hat H,\hat T],\hat T\bigr],\hat T\Bigr],\hat T\Bigr] .
    $$

    The general series would go on forever, but here it **ends after the fourth commutator** (*from memory*). A commutator
    of $\hat H$ with $\hat T$ is non-zero only if at least one creation or annihilation operator of $\hat H$ is
    contracted with an operator in $\hat T$ (shares an orbital index). $\hat H$ contains at most four such operators,
    because it is a two-electron operator, and each $\hat T$ in the product must use up at least one of them, so no term
    with five or more factors of $\hat T$ survives. As a consequence the equations are at most **quartic** in the amplitudes,
    and the energy and the singles and doubles equations are finite expressions. Working them out term by term, which
    is a bookkeeping exercise of contractions, gives the results below. (The energy formula can be seen directly:
    $\hat H$ can only connect $\Phi_0$ to determinants at most doubly excited, so only $\hat T_1$ and $\hat T_2$ matter.)

    ### Step 5. The energy

    $$
    E_{\text{CC}} = E_{\text{HF}} + E_{\text{corr}},
    \qquad
    E_{\text{corr}} = \sum_{ia} f_{ia}\,\textcolor{#6B913C}{t_i^a}
    + \frac14\sum_{ijab}\langle ij\Vert ab\rangle\,\textcolor{#F14124}{t_{ij}^{ab}}
    + \frac12\sum_{ijab}\langle ij\Vert ab\rangle\,\textcolor{#6B913C}{t_i^a}\textcolor{#6B913C}{t_j^b} .
    $$

    The first sum is zero for Hartree-Fock orbitals. Knowing the amplitudes gives the energy directly; all the work is in the
    amplitudes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 6. The doubles equation for CCD

    The $\langle\Phi_{ij}^{ab}\vert\bar H\vert\Phi_0\rangle=0$ equation, with $\hat T=\hat T_2$ only, written out. Let $P(ij)\,X_{ij}=X_{ij}-X_{ji}$
    and likewise $P(ab)$ for $a$ and $b$. Then, with $\varepsilon_p$ the orbital energies,

    $$
    \begin{aligned}
    \bigl(\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b\bigr)\,t_{ij}^{ab}
    &= \langle ij\Vert ab\rangle
    + \tfrac12\sum_{cd}\langle ab\Vert cd\rangle\,t_{ij}^{cd}
    + \tfrac12\sum_{kl}\langle kl\Vert ij\rangle\,t_{kl}^{ab}
    + P(ij)P(ab)\sum_{kc}\langle kb\Vert cj\rangle\,t_{ik}^{ac}\\
    &\quad + \tfrac14\sum_{klcd}\langle kl\Vert cd\rangle\,t_{ij}^{cd}\,t_{kl}^{ab}
    - \tfrac12\,P(ab)\sum_{klcd}\langle kl\Vert cd\rangle\,t_{ij}^{ac}\,t_{kl}^{bd}\\
    &\quad - \tfrac12\,P(ij)\sum_{klcd}\langle kl\Vert cd\rangle\,t_{ik}^{ab}\,t_{jl}^{cd}
    - \tfrac12\,P(ij)P(ab)\sum_{klcd}\langle kl\Vert cd\rangle\,t_{ik}^{ac}\,t_{jl}^{db} .
    \end{aligned}
    $$

    Reading the terms (the left side is the diagonal part of the Fock operator acting on the amplitude, and every other
    term has been moved to the right):

    - $\langle ij\Vert ab\rangle$: the *driving term*. It is the direct coupling of $\Phi_0$ to the double excitation.
      With every amplitude on the right set to zero, the equation gives $t_{ij}^{ab}=\langle ij\Vert ab\rangle/(\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b)$,
      the amplitude of second-order perturbation theory (MP2).
    - The three linear sums let each pair of electrons scatter into other pairs of orbitals: virtual to virtual
      (the $\langle ab\Vert cd\rangle$ "particle-particle ladder"), occupied to occupied, and one occupied and one virtual
      each (the "ring" term). They give the pair correlation more room than MP2 allows.
    - The four quadratic sums are products of two amplitudes, the way different pairs of electrons influence each other.

    **Checked here.** The equation above is implemented as a separate function (`ccd_update_expanded`), and
    `test_cc.py` shows that it gives the same amplitudes as the intermediate form below (with $t_i^a=0$) to $10^{-12}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 7. The CCSD equations

    With singles, the number of terms is much larger, so the equations are written with intermediates, which are
    partial sums that are built once and reused. With $\tau_{ij}^{ab}=t_{ij}^{ab}+t_i^at_j^b-t_i^bt_j^a$ and
    $\tilde\tau_{ij}^{ab}=t_{ij}^{ab}+\tfrac12\bigl(t_i^at_j^b-t_i^bt_j^a\bigr)$, with $m,n$ occupied and $e,f$ virtual:

    $$
    \begin{aligned}
    F_{ae} &= (1-\delta_{ae})f_{ae} - \tfrac12\sum_m f_{me}t_m^a + \sum_{mf}t_m^f\langle ma\Vert fe\rangle
      - \tfrac12\sum_{mnf}\tilde\tau_{mn}^{af}\langle mn\Vert ef\rangle ,\\
    F_{mi} &= (1-\delta_{mi})f_{mi} + \tfrac12\sum_e t_i^e f_{me} + \sum_{ne}t_n^e\langle mn\Vert ie\rangle
      + \tfrac12\sum_{nef}\tilde\tau_{in}^{ef}\langle mn\Vert ef\rangle ,\\
    F_{me} &= f_{me} + \sum_{nf}t_n^f\langle mn\Vert ef\rangle ,\\
    W_{mnij} &= \langle mn\Vert ij\rangle + P(ij)\sum_e t_j^e\langle mn\Vert ie\rangle + \tfrac14\sum_{ef}\tau_{ij}^{ef}\langle mn\Vert ef\rangle ,\\
    W_{abef} &= \langle ab\Vert ef\rangle - P(ab)\sum_m t_m^b\langle am\Vert ef\rangle + \tfrac14\sum_{mn}\tau_{mn}^{ab}\langle mn\Vert ef\rangle ,\\
    W_{mbej} &= \langle mb\Vert ej\rangle + \sum_f t_j^f\langle mb\Vert ef\rangle - \sum_n t_n^b\langle mn\Vert ej\rangle
      - \sum_{nf}\bigl(\tfrac12 t_{jn}^{fb}+t_j^ft_n^b\bigr)\langle mn\Vert ef\rangle .
    \end{aligned}
    $$

    The singles and doubles equations, with $D_i^a=\varepsilon_i-\varepsilon_a$ and $D_{ij}^{ab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b$:

    $$
    \begin{aligned}
    D_i^a\,\textcolor{#6B913C}{t_i^a} &= f_{ia} + \sum_e t_i^eF_{ae} - \sum_m t_m^aF_{mi} + \sum_{me}t_{im}^{ae}F_{me}
      - \sum_{nf}t_n^f\langle na\Vert if\rangle - \tfrac12\sum_{mef}t_{im}^{ef}\langle ma\Vert ef\rangle
      - \tfrac12\sum_{men}t_{mn}^{ae}\langle nm\Vert ei\rangle ,\\[4pt]
    D_{ij}^{ab}\,\textcolor{#F14124}{t_{ij}^{ab}} &= \langle ij\Vert ab\rangle
      + P(ab)\sum_e t_{ij}^{ae}\bigl(F_{be}-\tfrac12\sum_m t_m^bF_{me}\bigr)
      - P(ij)\sum_m t_{im}^{ab}\bigl(F_{mj}+\tfrac12\sum_e t_j^eF_{me}\bigr)\\
    &\quad + \tfrac12\sum_{mn}\tau_{mn}^{ab}W_{mnij} + \tfrac12\sum_{ef}\tau_{ij}^{ef}W_{abef}
      + P(ij)P(ab)\sum_{me}\bigl(t_{im}^{ae}W_{mbej} - t_i^et_m^a\langle mb\Vert ej\rangle\bigr)\\
    &\quad + P(ij)\sum_e t_i^e\langle ab\Vert ej\rangle - P(ab)\sum_m t_m^a\langle mb\Vert ij\rangle .
    \end{aligned}
    $$

    With $t_i^a=0$ these reduce to the CCD equation above. The diagonal part of $f$ was moved to the left into the
    $D$'s, which is the reason for the factors $(1-\delta)$: for Hartree-Fock orbitals the remaining off-diagonal $f$
    terms are zero, and they would only matter for other kinds of orbitals.

    ### Step 8. Solving them: iterate

    The unknowns appear on both sides, but each amplitude on the left is multiplied by a plain number, the orbital-energy
    difference $D$, so the equations can be solved by repeated substitution (a **Jacobi iteration**):

    1. Start from $t_i^a=0$ and the MP2 amplitudes $t_{ij}^{ab}=\langle ij\Vert ab\rangle/D_{ij}^{ab}$.
    2. Evaluate the right-hand side with the current amplitudes (the intermediates first, then the two equations) and divide by $D$. These are the new amplitudes.
    3. Compute $E_{\text{corr}}$ from step 5; stop when it changes by less than $10^{-10}$ Ha, otherwise go back to 2.

    The first energy of the iteration is the MP2 energy, so the iteration can be read as MP2 improved until the pairs
    are consistent with each other. Mixing the last few iterates with a least-squares weighting (DIIS, *from memory*
    a standard acceleration device) reaches the same answer in fewer steps; both are shown in section 5.
    The cost of one iteration is set by the sums with the most free indices. The largest is
    $\tfrac12\sum_{ef}\tau_{ij}^{ef}W_{abef}$: for fixed $i,j,a,b$ it sums over $e,f$, which with $o$ occupied and $v$ virtual
    labels is about $o^2v^4$ multiplications. That is section 7.
    """)
    return


@app.cell
def _(HA2KCAL, cc_data, convergence_figure, mo, res_h2, res_w1, res_w2, res_w2_plain):
    _rows = [
        "| System | Spin orbitals (occupied) | CCSD $E_{\\text{corr}}$, numpy (Ha) | CCSD $E_{\\text{corr}}$, PySCF (Ha) | Difference (Ha) | Iterations |",
        "|---|---|---|---|---|---|",
    ]
    for _name, _label, _res, _n in (("h2_631g", "H$_2$, 6-31G", res_h2, "8 (2)"), ("water_sto3g", "H$_2$O, STO-3G", res_w1, "14 (10)"),
                                    ("water_631g", "H$_2$O, 6-31G", res_w2, "26 (10)")):
        _ref = cc_data["systems"][_name]["e_ccsd_corr"]
        _rows.append(f"| {_label} | {_n} | {_res['e_corr']:.10f} | {_ref:.10f} | {_res['e_corr'] - _ref:+.1e} | {_res['iterations']} |")
    mo.vstack([
        mo.md(r"""
        ## 5. A from-scratch implementation, checked against PySCF

        `cc_numerics.py` is a short numpy program for the equations above (`ccsd_update` is the right-hand sides;
        `ccsd` is the iteration with its DIIS mixing). It works in the spin-orbital basis of section 4:
        the molecular-orbital integrals of a Hartree-Fock calculation are read from `data_cc_integrals.npz` (written by
        `compute_cc_data.py`, which is the only part that needs PySCF) and turned into $h_{pq}$ and $\langle pq\Vert rs\rangle$.
        Everything after that is `numpy.einsum`, one line per term of the equations. The three test systems are
        H$_2$ (two electrons), water in STO-3G and water in 6-31G, with all electrons correlated.
        """),
        mo.md("\n".join(_rows)),
        mo.md(
            f"**Checked here.** For water in 6-31G the numpy CCSD correlation energy is {res_w2['e_corr']:.8f} Ha "
            f"({res_w2['e_corr'] * HA2KCAL:.2f} kcal/mol) and agrees with PySCF's CCSD to {abs(res_w2['e_corr'] - cc_data['systems']['water_631g']['e_ccsd_corr']):.0e} Ha; "
            f"the tests require $10^{{-8}}$ Ha for all three systems. The agreement tests the intermediates and every term of the equations "
            f"in the singles and doubles, since one wrong sign would move the energy by far more than that. The plain iteration takes "
            f"{res_w2_plain['iterations']} steps and the DIIS-mixed one {res_w2['iterations']}, with the same energy."
        ),
        convergence_figure(res_w2["history"], res_w2_plain["history"]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The (T) correction

    CCSD contains triple excitations only as products of singles and doubles (section 3). The *connected* triples, the
    ones that need their own amplitudes $t_{ijk}^{abc}$, are missing, and they are the largest correction left at
    the usual single-reference level. Solving for them properly (CCSDT) is much more expensive, so the standard
    compromise is **CCSD(T)**: solve CCSD first, then estimate the connected triples by perturbation theory *once*, from the
    converged $t_i^a$ and $t_{ij}^{ab}$ (*from memory*: the "T" in parentheses means "perturbative").

    With $D_{ijk}^{abc}=\varepsilon_i+\varepsilon_j+\varepsilon_k-\varepsilon_a-\varepsilon_b-\varepsilon_c$ and
    $P(i/jk)\,X_{ijk}=X_{ijk}-X_{jik}-X_{kji}$ (and the same for $a/bc$), the first-order triples amplitude has a
    **connected** part, built from the doubles, and a **disconnected** part, built from the singles:

    $$
    \begin{aligned}
    D_{ijk}^{abc}\,t_{ijk}^{abc}(\text{c}) &= P(i/jk)P(a/bc)\Bigl[\sum_d t_{jk}^{ad}\langle bc\Vert di\rangle - \sum_l t_{il}^{bc}\langle la\Vert jk\rangle\Bigr],\\
    D_{ijk}^{abc}\,t_{ijk}^{abc}(\text{d}) &= P(i/jk)P(a/bc)\; t_i^a\,\langle jk\Vert bc\rangle ,\\
    E_{(\text{T})} &= \frac1{36}\sum_{ijkabc}\;t_{ijk}^{abc}(\text{c})\,D_{ijk}^{abc}\Bigl[t_{ijk}^{abc}(\text{c}) + t_{ijk}^{abc}(\text{d})\Bigr] .
    \end{aligned}
    $$

    In words: a triple excitation is reached from a double by adding one more excited electron through the integrals
    (the two terms of the connected part: one virtual orbital replaced by two, or one occupied replaced by two); the energy
    is the square of that amplitude weighted by the denominator, the usual second-order form, plus a cross term with the
    disconnected piece. The factor $\tfrac1{36}=\bigl(\tfrac1{3!}\bigr)^2$ undoes the over-counting of orderings of $ijk$ and $abc$.
    The correction is computed once at the end, and not iterated, so it is called *non-iterative*.
    """)
    return


@app.cell
def _(HA2KCAL, cc_data, mo, res_h2, res_w1, res_w2, sys_h2, sys_w1, sys_w2, triples_correction):
    _rows = [
        "| System | $E_{(\\text{T})}$ numpy (Ha) | $E_{(\\text{T})}$ PySCF (Ha) | Difference (Ha) | $E_{(\\text{T})}$ (kcal/mol) |",
        "|---|---|---|---|---|",
    ]
    for _name, _label, _sys, _res in (("h2_631g", "H$_2$, 6-31G", sys_h2, res_h2), ("water_sto3g", "H$_2$O, STO-3G", sys_w1, res_w1),
                                      ("water_631g", "H$_2$O, 6-31G", sys_w2, res_w2)):
        _et = triples_correction(_sys, _res["t1"], _res["t2"])
        _ref = cc_data["systems"][_name]["e_t"]
        _rows.append(f"| {_label} | {_et:.10f} | {_ref:.10f} | {_et - _ref:+.1e} | {_et * HA2KCAL:.3f} |")
    mo.vstack([
        mo.md("\n".join(_rows)),
        mo.md(
            "**Checked here.** I implemented the formulas above (`triples_correction`, a few lines of `einsum` on arrays with six "
            "indices) and it reproduces PySCF's `ccsd_t` to better than $10^{-8}$ Ha for both water systems. For H$_2$ the correction is "
            "zero, as it must be: with two electrons there are no triple excitations. The implementation stores the whole "
            "array $t_{ijk}^{abc}$, which is fine for these sizes and would be hopeless for a real molecule, where programs loop over "
            "$i,j,k$ and never hold it."
        ),
    ])
    return


@app.cell
def _(mo, np, scaling_figure, cc_data):
    _t = cc_data["timing"]
    _slope_u = np.diff(np.log([r["t_update_s"] for r in _t]))[-1] / np.diff(np.log([r["n_virt_so"] for r in _t]))[-1]
    _tt = [r for r in _t if "t_triples_s" in r]
    _slope_t = np.diff(np.log([r["t_triples_s"] for r in _tt]))[-1] / np.diff(np.log([r["n_virt_so"] for r in _tt]))[-1]
    mo.vstack([
        mo.md(r"""
        ## 7. What it costs

        The number of operations for a method is set by the number of orbital labels in its most expensive sum. With $o$ occupied
        and $v$ virtual orbitals, and $N=o+v$ the number of basis functions (for a molecule in a fixed type of basis, $o$ and $v$ grow in
        proportion to $N$):

        | Method | Most expensive step | Cost as a function of $N$ | Source of the exponent |
        |---|---|---|---|
        | HF | two-electron integrals, Fock build | formally $N^4$, less with screening | memory |
        | DFT (hybrid) | same plus the grid | about $N^3$ to $N^4$ | memory |
        | MP2 | transform integrals, then $o^2v^2$ amplitudes | $N^5$ | memory |
        | CCSD | $\tfrac12\sum_{ef}\tau_{ij}^{ef}W_{abef}$ ($o^2v^4$), per iteration | $N^6$ | checked: counted from the sums above |
        | CCSD(T) | $t_{ijk}^{abc}$: $o^3v^3$ elements, each a sum over $v$ or $o$ terms ($o^3v^4$), once | $N^7$ | checked: counted from the sums above |
        | FCI | all determinants | factorial: $\binom{M}{N_\alpha}\binom{M}{N_\beta}$ | checked: section 1 |

        So doubling the system size multiplies the cost of CCSD by about $2^6=64$ and of CCSD(T) by about $2^7=128$,
        against 8 to 16 for DFT. Memory matters as much as time: CCSD stores arrays with four virtual indices
        ($v^4$ numbers, *from memory* the usual bottleneck).

        Left: the counts $o^2v^3$, $o^2v^4$ and $o^3v^4$ for a molecule that grows with $v=4o$ (water in a double-zeta basis has
        $o=5$, $v=19$); FCI, whose count is in section 1, would lie far above them. Right: wall-clock time of one iteration of *this notebook's*
        numpy code, and of the (T) evaluation, for eight occupied spin orbitals and a growing number of virtual ones.
        """),
        scaling_figure(cc_data["timing"]),
        mo.md(
            f"Between the last two sizes the measured slope of the CCSD iteration is {_slope_u:.1f} (it climbs towards the counted 4 as "
            f"$v$ grows) and that of (T) is {_slope_t:.1f} between its last two sizes (lower than the counted 4 at these small sizes, where "
            f"creating the six-index array costs about as much as the contractions). Only the trend should be read from the right-hand "
            f"panel and not a fitted power. Time per iteration: " + ", ".join(
                f"{r['t_update_s'] * 1000:.0f} ms at $v={r['n_virt_so']}$" for r in cc_data["timing"][2::2]) + "."
        ),
    ])
    return


@app.cell
def _(HA2KCAL, cc_data, mo, recovery_figure):
    _eq = cc_data["water_series"]["equilibrium"]
    _rows = [
        "| Water | $E_{\\text{HF}}$ (Ha) | MP2 | CCSD | CCSD(T) | FCI | CCSD(T) $-$ FCI (kcal/mol) |",
        "|---|---|---|---|---|---|---|",
    ]
    _cases = [("STO-3G, equilibrium", _eq["sto-3g"]), ("6-31G, equilibrium", _eq["6-31g"])] + [
        (f"6-31G, O–H bonds ×{k}", cc_data["water_series"]["stretched"][k]) for k in ("1.5", "2.0")
    ]
    for _lab, _e in _cases:
        _t = _e["e_ccsd_corr"] + _e["e_t"]
        _rows.append(
            f"| {_lab} | {_e['e_hf']:.4f} | {_e['e_mp2_corr']:.5f} | {_e['e_ccsd_corr']:.5f} | {_t:.5f} | {_e['e_fci_corr']:.5f} | "
            f"{(_t - _e['e_fci_corr']) * HA2KCAL:+.2f} |"
        )
    _stretch = {"equilibrium": _eq["6-31g"], "O–H ×1.5": cc_data["water_series"]["stretched"]["1.5"],
                "O–H ×2": cc_data["water_series"]["stretched"]["2.0"]}
    _e2 = cc_data["water_series"]["stretched"]["2.0"]
    _e0 = _eq["6-31g"]
    ladder_summary = {"mp2": 100 * _e0["e_mp2_corr"] / _e0["e_fci_corr"], "ccsd": 100 * _e0["e_ccsd_corr"] / _e0["e_fci_corr"],
                      "ccsdt": 100 * (_e0["e_ccsd_corr"] + _e0["e_t"]) / _e0["e_fci_corr"],
                      "mp2_2": 100 * _e2["e_mp2_corr"] / _e2["e_fci_corr"], "ccsd_2": 100 * _e2["e_ccsd_corr"] / _e2["e_fci_corr"],
                      "ccsdt_2": 100 * (_e2["e_ccsd_corr"] + _e2["e_t"]) / _e2["e_fci_corr"]}
    mo.vstack([
        mo.md(r"""
        ## 8. How accurate: the ladder, and the link back to DFT

        ### A series on water, with the exact answer to compare

        Water is small enough in the 6-31G basis for FCI (PySCF, 1.7 million determinants), so each method can be
        compared with the exact correlation energy *in the same basis*. Correlation energies in hartree (all electrons correlated):
        """),
        mo.md("\n".join(_rows)),
        recovery_figure(_stretch),
        mo.md(
            f"**Checked here.** At the equilibrium geometry MP2 recovers {ladder_summary['mp2']:.1f}% of the exact correlation energy, "
            f"CCSD {ladder_summary['ccsd']:.1f}% and CCSD(T) {ladder_summary['ccsdt']:.1f}%; the remaining difference is "
            f"{abs((_e0['e_ccsd_corr'] + _e0['e_t'] - _e0['e_fci_corr']) * HA2KCAL):.2f} kcal/mol. Each step up the ladder removes roughly two-thirds or more "
            f"of the error of the one before. When the two O–H bonds are stretched to twice their length, a stress test in which the single "
            f"determinant is a poor starting point, MP2 falls to {ladder_summary['mp2_2']:.1f}% and CCSD to {ladder_summary['ccsd_2']:.1f}%, while CCSD(T) overshoots to "
            f"{ladder_summary['ccsdt_2']:.1f}%: it is *below* the exact energy, which is allowed because the method is not variational, and a warning "
            f"that (T) is a correction for a determinant that is already a good start. (This test is for a small basis set; the "
            f"exact numbers would change in a larger one, but the trend with stretching is the point.)"
        ),
    ])
    return (ladder_summary,)


@app.cell
def _(disp_data, interaction_figure, ladder_data, mo):
    _hf, _b3, _wb = ladder_data["functionals"]["HF"], ladder_data["functionals"]["b3lyp"], ladder_data["functionals"]["wb97m_v"]
    _ref = ladder_data["ccsdt"]
    _dm = disp_data["methods"]
    _rows = [
        "| Interaction energy (kcal/mol) | HF | B3LYP | ωB97M-V | CCSD(T) |",
        "|---|---|---|---|---|",
        f"| Water dimer | {_hf['cp']:+.2f} | {_b3['cp']:+.2f} | {_wb['cp']:+.2f} | {_ref['cp']:+.2f} |",
        f"| Methane dimer at 3.8 Å | {_dm['HF']['cp']:+.2f} | {_dm['b3lyp']['cp']:+.2f} | {_dm['wb97m_v']['cp']:+.2f} | {disp_data['ccsdt']['cp']:+.2f} |",
    ]
    mo.vstack([
        mo.md(r"""
        ### The dimers of the DFT notebook

        The numbers below are the ones already in the DFT notebook (`data_ladder.json`, `data_dispersion.json`), all with the
        def2-TZVPD basis and counterpoise correction. The CCSD(T) column is the method of this notebook, run through the
        PySCF CCSD and `ccsd_t` calls; the DFT columns are from the functionals of that notebook.
        """),
        mo.md("\n".join(_rows)),
        interaction_figure(ladder_data, disp_data),
        mo.md(
            f"**Why CCSD(T) is used as the reference there.** The ladder of this section is the reason: starting from "
            f"Hartree-Fock, CCSD brings in the pair correlation and (T) the leading connected triples, and compared with exact "
            f"energies the remaining error in the correlation energy is a fraction of a kcal/mol at the equilibrium geometry (checked above for water "
            f"in one small basis; the general claim for interaction energies in large basis sets is *from memory*). It is also a "
            f"method with no adjustable parameters, so it does not share the choices that distinguish one functional from another.\n\n"
            f"**Dispersion is correlation.** For the methane dimer Hartree-Fock gives {_dm['HF']['cp']:+.2f} kcal/mol: it is repulsive, "
            f"because there is no charge, hydrogen bond or permanent moment to attract the molecules, and what holds them together "
            f"is the correlated motion of electrons in the two molecules, which a single determinant cannot describe. CCSD(T) gives "
            f"{disp_data['ccsdt']['cp']:+.2f} kcal/mol, attractive. The difference between a functional with a dispersion term and without, "
            f"in the DFT notebook section 7, is the density-functional counterpart of what the doubles amplitudes of this notebook "
            f"capture: the pair of electrons on one molecule and the pair on the other moving in step. In the amplitude "
            f"language of section 4 it is the *double excitations on the two molecules at the same time*, which appear in CCSD "
            f"through the products of section 3."
        ),
    ])
    return


@app.cell
def _(ladder_summary, mo):
    mo.md(f"""
    ## 9. What I have not verified, and four lines for a slide

    ### Not verified

    - **From memory, not tested here:** the cost exponents of HF, DFT and MP2 ($N^4$, $N^3$ to $N^4$, $N^5$), that DIIS is the
      usual acceleration, that the equations end at fourth order (the argument is given in section 4, but the full
      term-by-term expansion was not redone; only the *result* is tested, by the agreement with PySCF), that the memory
      is dominated by arrays with four virtual indices, and the statement that CCSD(T) reaches a fraction of a kcal/mol for
      interaction energies in large basis sets.
    - **Small cases only:** the accuracy ladder is for one molecule (water) in the 6-31G basis, and the size-consistency
      demonstration is for H$_2$ pairs, where the exact answer is available. Neither says how large the errors are for
      a bigger basis, a transition metal, or a bond being broken (the stretched water cases suggest that they grow).
    - **The cost figure** is measured with the numpy code on random integrals of the right shape, at small sizes, on one
      machine. It shows the trend but is not a fitted exponent, and PySCF's own code is much faster than this one.
    - **Interaction energies:** the CCSD(T) values for the dimers are PySCF runs stored from the DFT notebook; they are not
      reproduced by the numpy code, which is only run on water and H$_2$. They are in one basis (def2-TZVPD) and are not extrapolated
      to the complete-basis limit, so they are a reference for the same basis, not the exact answer.
    - **Only closed-shell, canonical Hartree-Fock orbitals** are handled; open-shell molecules, frozen-core approximations and
      the local-correlation methods used for large systems are not covered.
    - **Rendering:** the displayed equations were written for KaTeX and have not been looked at in a browser; the figures were
      rendered and looked at.

    ### Four lines for a slide

    1. Hartree-Fock misses the correlation energy; the exact answer needs every excitation from $\\Phi_0$ and costs a factorial number of determinants.
    2. Truncating the excitations (CISD) is not size consistent; writing the wave function as $e^{{\\hat T}}\\Phi_0$ with $\\hat T=\\hat T_1+\\hat T_2$ is.
    3. The amplitudes follow from projecting $e^{{-\\hat T}}\\hat H e^{{\\hat T}}\\Phi_0=E\\Phi_0$; the series ends at fourth order, and the cost is $N^6$ for CCSD and $N^7$ with the perturbative triples (T).
    4. In water, 6-31G, CCSD(T) recovers {ladder_summary['ccsdt']:.1f}% of the exact correlation energy (MP2 {ladder_summary['mp2']:.0f}%), which is why it is the reference for the dimers, where dispersion is a correlation effect that Hartree-Fock cannot give.
    """)
    return


if __name__ == "__main__":
    app.run()
