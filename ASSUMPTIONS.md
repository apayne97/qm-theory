# Assumptions and approximations: reference list for the talk

Draft v2, 2026-11-05 (renumbered after adding the basis-set notebook and the OPLS5 labels). Refer to items on slides by number, e.g. "(A17)". Order follows `TALK_OUTLINE.md`.

- **Kind**: *Assumption* = a premise of the theory that is exact if true; *Approximation* = a deliberate simplification with an error; *Setting* = a choice made in the calculation, not in the physics.
- **Status**: *standard* = textbook; *checked* = confirmed in a notebook calculation or against a source I read (OMol25 paper, OPLS5 paper, Jaguar manual); *memory* = stated from memory, not verified (see `dft/reading_a_dft_label.py` §10).
- Numbers are from the notebooks (water/dimer geometries from `~/science/qm-calculations`, def2-TZVPD unless noted).

## Part 1-2. Waves, boxes, atoms

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A1 | Assumption | Non-relativistic: electrons obey the Schrödinger equation, not Dirac. | Fine for organic molecules; matters for heavy elements (see A30). | standard | all |
| A2 | Assumption | The Hamiltonian does not depend on time. | Allows $\psi(x,t)=\psi_E(x)T(t)$ and stationary states. | standard (last talk) | `basis_sets_from_base_states` §1 |
| A3 | Assumption | Any state is a superposition $\sum_n c_n\psi_{E_n}e^{-iE_nt/\hbar}$ of stationary states (the set is complete). | Justifies expanding in base states; in practice the sum is truncated (A14). | standard | `basis_sets_from_base_states` §1 |
| A4 | Assumption | Box with infinitely hard walls: $\psi=0$ at the boundary. | Gives whole numbers of half-wavelengths; an idealisation of a bound particle. | standard | `standing_waves_in_boxes` §1 |
| A5 | Assumption | Rectangular box, so the 2D/3D wave is a product of 1D waves and energies add. | Gives the $(k_x,k_y,k_z)$ lattice and the level pattern 1,3,3,3,1,6. | standard | `standing_waves_in_boxes` §2-5 |
| A6 | Assumption | The potential depends only on the distance $r$ from the nucleus (central potential). | Allows radial times angular parts; the angular part is the spherical harmonics. | standard | `spherical_harmonics_and_hydrogen` §1-4 |
| A7 | Approximation | Point nucleus of infinite mass (electron mass, not reduced mass). | Correction of order 1/1836 in the energy. | standard | `spherical_harmonics_and_hydrogen` §5 |
| A8 | Approximation | Hydrogen levels from the Coulomb attraction only: no spin-orbit coupling, Lamb shift or hyperfine structure. | $E=-13.6/n^2$ eV is the leading term; fine structure is much smaller. | standard | `spherical_harmonics_and_hydrogen` §5 |

## Part 3. From atoms to molecules

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A9 | Approximation | Born-Oppenheimer: nuclei are fixed, and the electrons are solved at each geometry. | Gives one energy surface $E(\text{geometry})$; fails near surface crossings. | standard | `spherical_harmonics_and_hydrogen` §7 |
| A10 | Setting | The label energy is a single electronic energy of one geometry, ground state, no vibrations, no temperature. | Zero-point energy and thermal terms are not included. | standard | `reading_a_dft_label` §1 |
| A11 | Approximation | Orbital approximation: each electron occupies its own orbital; the many-electron wavefunction is built from them. | Neglects correlated motion of electrons (see A18, A20). | standard | `spherical_harmonics_and_hydrogen` §7, `reading_a_dft_label` §2 |
| A12 | Requirement | The wavefunction must be antisymmetric under swapping two electrons (Pauli). | Not an approximation; a Slater determinant satisfies it. | standard | `reading_a_dft_label` §2 |
| A13 | Approximation | LCAO: molecular orbitals are linear combinations of atom-centred functions. | Reduces the equations to a matrix problem; accuracy depends on the functions chosen (A14-A17). | standard | `basis_sets_from_base_states` §3, `kohn_sham_matrix` §1 |

## Part 3b. Basis sets: where the finite sum comes from

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A14 | Approximation | Only $N$ fixed functions are used in the expansion instead of the complete set. | By the variational principle the energy is never below the exact one and never rises when a function is added. Toy check: two-function H$_2^+$ binds by about 1.76 eV, a large basis gives about 2.79 eV. | checked (toy model and H$_2^+$ numbers); exact H$_2^+$ energy from memory | `basis_sets_from_base_states` §2-3 |
| A15 | Setting | The basis functions are not eigenfunctions of $H$ and are not orthogonal, so the equation becomes $\mathbf{HC}=\mathbf{SC}\varepsilon$ with an overlap matrix $\mathbf S$. | Same levels are obtained from an orthonormal or a non-orthogonal set of the same span. | checked | `basis_sets_from_base_states` §2, `kohn_sham_matrix` |
| A16 | Approximation | Gaussians replace the exact exponential (Slater) shape; a fixed sum of Gaussians (contraction, e.g. STO-$n$G) mimics it. | A single Gaussian has zero derivative at the nucleus (exact hydrogen 1s: $-1$) and a tail that falls too fast; error shrinks with $n$. Used because Gaussian integrals are analytic. | checked (hydrogen 1s, s-type integrals); analytic integrals in general from memory | `basis_sets_from_base_states` §4 |
| A17 | Setting | Each atom carries shells with fixed angular parts ($2\ell+1$ functions per shell); flexibility is added by more radial functions (zeta), higher-$\ell$ shells (polarization, P) and wide functions (diffuse, D). | Water function counts: svp 24, tzvp 43, tzvpd 58, tzvpp 59, tzvppd 74, qzvp 117. How def2 was designed is from memory. | checked (counts, hydrogen tests); def2 design memory | `basis_sets_from_base_states` §5 |

## Part 4. Hartree-Fock

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A18 | Approximation | One Slater determinant; each electron feels the average field of the others (mean field). | Misses correlation. Water dimer CP $E_\text{int}$: HF −2.82 vs CCSD(T) −3.73 kcal/mol; methane dimer is unbound at HF (+0.28). | checked | `reading_a_dft_label` §2 |
| A19 | Setting | The equations are solved self-consistently (SCF loop) to a convergence threshold. | Orbitals depend on the field they create, so the loop is repeated; OMol25 used tight convergence. | checked | `kohn_sham_matrix` §5 |

## Part 5. DFT and the labels

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A20 | Assumption | Hohenberg-Kohn / Kohn-Sham: the energy is an exact functional of the density, and a fictitious non-interacting system reproduces that density. | Exact in principle; everything unknown is collected in the exchange-correlation term $E_{xc}$. | standard | `reading_a_dft_label` §3, `kohn_sham_matrix` |
| A21 | Approximation | Finite basis set (def2-TZVPD for OMol25, def2-TZVPPD for SPICE II). | Basis-set superposition error (BSSE) on the water dimer: ωB97M-V SVP −7.70 vs TZVPD −4.30 kcal/mol (raw). | checked | `reading_a_dft_label` §4 |
| A22 | Approximation | $E_{xc}$ is approximated: ωB97M-V is a meta-GGA range-separated hybrid (rung 4 on Jacob's ladder). | Functional error; no systematic way to improve. Water HOMO −11.65 eV vs experiment 12.6. | checked (HOMO numbers); label meaning memory | `reading_a_dft_label` §5 |
| A23 | Setting | Range separation: $\omega=0.3\ \text{bohr}^{-1}$, 15% exact exchange at short range, 100% at long range. | Corrects the long-range self-interaction error. Values from PySCF and the Psi4 docs. | checked (ωB97M-V); that SPICE's D3(BJ) variant shares them is memory (Psi4 lists the same values) | `reading_a_dft_label` §6 |
| A24 | Approximation | Dispersion is added by a separate term: nonlocal VV10 (OMol25) or the empirical D3(BJ) (SPICE II). | Methane dimer 3.8 Å: without VV10 +0.07, with VV10 −0.42, CCSD(T) −0.39 kcal/mol. | checked (VV10); no D3(BJ) run in PySCF | `reading_a_dft_label` §7 |
| A25 | Setting | Numerical integration of $E_{xc}$ on a finite grid. OMol25: DEFGRID3 (590 angular points for XC, 302 for COSX). | Small, controllable error; radial point count not stated in the paper. | checked (OMol25 paper) | `reading_a_dft_label` §8 |
| A26 | Approximation | OMol25 uses RI-J (density fitting of Coulomb) and COSX (chain-of-spheres exchange) in ORCA 6.0.0; SPICE II uses Psi4. | Speed-ups with small errors. | checked (OMol25); Psi4 version unresolved | `reading_a_dft_label` §8 |
| A27 | Setting | Open-shell and charged systems use unrestricted Kohn-Sham (UKS). | Different spatial orbitals for spin up and down; spin contamination is not discussed in the notebooks. | checked (OMol25) | `reading_a_dft_label` §8 |
| A28 | Setting | Protein-ligand interaction energy: $E(\text{ligand+pocket})-E(\text{ligand})-E(\text{pocket})$. | Counterpoise correction is a separate choice; check whether OMol25 applies it. | checked (definition); counterpoise not checked | `reading_a_dft_label` §4 |
| A29 | Assumption | The DFT label is treated as the truth when training a machine-learning potential or a force field. | The model inherits A21-A24; functional error vs CCSD(T) is about 0.5 kcal/mol for ωB97M-V on the water dimer here, and is not known for other systems. | memory (general statement) | `coupled_cluster` §8 |

## Part 5b. A force-field paper's labels (OPLS5)

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A30 | Approximation | An effective core potential (ECP) replaces the core electrons of heavy atoms; only the valence electrons are treated explicitly (`lacvp+*` for Ir, Pt). | Pt: 60 electrons removed. Cuts cost and absorbs the main relativistic effects into the potential (A1). The Jaguar basis definition was read from a manual mirror; "absorbs relativistic effects" is memory. | checked (Pt 60 e, LACVP definition); relativistic remark memory | `reading_a_dft_label` §9 |
| A31 | Setting | Different levels of theory for different jobs: ωB97X-D3/lacvp+* for geometries and ESP, ωB97X-D3/def2-TZVPD(-f) for scans, ωB97X-D/6-311++G** for a probe charge, CCSD(T)/CBS for charged-group references. | Energies from different levels are not comparable; the label tells which job a number belongs to. | checked (OPLS5 PDF) | `reading_a_dft_label` §9 |
| A32 | Approximation | ωB97X, ωB97X-D and ωB97X-D3 are three different functionals, each refitted with its own dispersion correction. | Range separation $\omega$ = 0.2 (-D) vs 0.25 (-D3) bohr⁻¹; short-range exact exchange 22.2% vs 19.57%. Probe charge over benzene with PySCF: −13.61 and −9.52 kcal/mol vs paper −13.6 and −9.5. | checked (libxc values, benzene probe); original references memory | `reading_a_dft_label` §9 |
| A33 | Approximation | CBS: the complete-basis-set limit is estimated by extrapolating results from finite basis sets. | Removes most basis-set error from the CCSD(T) references; the extrapolation formula the paper used is not known. | memory | `reading_a_dft_label` §9 |
| A34 | Approximation | A classical force field with fitted parameters stands in for the quantum calculation. | OPLS5 metal-ligand scan error against DFT: 5.4 kcal/mol (weighted RMSE) vs 32.6 for OPLS4. | checked (OPLS5 PDF) | `reading_a_dft_label` §9 |

## Part 6. Coupled cluster (reference)

| # | Kind | Statement | Consequence | Status | Notebook |
|---|------|-----------|-------------|--------|----------|
| A35 | Assumption | Single reference: one determinant (HF) dominates the exact wavefunction. | Fails for stretched bonds and some open-shell cases. | standard | `coupled_cluster` §1 |
| A36 | Approximation | CCSD truncates the excitation operator at singles and doubles; (T) adds triples perturbatively. | Cost $O(N^7)$ for (T); CCSD(T) is the reference for the dimer comparisons. | checked (CCSD vs PySCF to 1e-8 Ha) | `coupled_cluster` §3-7 |

## Not yet settled

- Whether OMol25/SPICE use effective core potentials for heavy elements (A30 is from the OPLS5 paper, not these datasets); whether they are gas-phase only; whether SPICE II refit any parameters (see `dft/reading_a_dft_label.py` §10).
- The Najibi-Goerigk 2018 reference for ωB97M-D3(BJ), the Chai-Head-Gordon references for ωB97X and ωB97X-D, and the older citations are not verified.
- The meaning of `*` and `+` in LACVP+*, the CBS extrapolation formula, and the D3 coordination-number dependence (see §10 of the label notebook).
- A29's error size comes only from the water dimer here; do not generalise on a slide.
- A34 is only loosely part of the talk: keep it if you show the OPLS5 example, otherwise drop it and renumber.
