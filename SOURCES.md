# Sources for further reading

Free, textbook-style sources for the topics of the current talk, one table per notebook. Checked on **2026-11-13**.

**What "checked" means here.** *Page* = the page was fetched, resolved, and its text contains the topic named in the "covers" column (I looked for the key terms and, where marked, read the passage). *Record* = the title, authors, year, journal, volume and pages were compared with the Crossref record of the DOI (or, for arXiv, the arXiv record); **the paper itself was not read**, so "covers" is taken from its title only. *Level*: intro (first exposure), derivation (steps shown), reference (look things up).

Access: LibreTexts, Psi4, PySCF, Psi4NumPy, Basis Set Exchange, Sherrill's notes and Burke's notes are open. Journal papers carry a DOI; several are behind a paywall (marked $), the rest are open access or have a free preprint.

The Feynman Lectures, Vol. III (online Caltech edition) remain the main reference for the wave picture; see `README.md` for the full citation.

## 01 `standing_waves_in_boxes`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Particle in a 1D box | LibreTexts, *Physical Chemistry* 3.5, "The Energy of a Particle in a Box is Quantized" | https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/03:_The_Schrodinger_Equation_and_a_Particle_in_a_Box/3.05:_The_Energy_of_a_Particle_in_a_Box_is_Quantized | Quantised energies, nodes, non-zero lowest energy | intro | page |
| 3D box, degeneracy | LibreTexts, *Physical Chemistry* 3.9, "A Particle in a Three-Dimensional Box" | https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/03:_The_Schrodinger_Equation_and_a_Particle_in_a_Box/3.09:_A_Particle_in_a_Three-Dimensional_Box | Sum of squares of three quantum numbers, degeneracy of the cube | intro | page |

## 02 `spherical_harmonics_and_hydrogen`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Laplacian in spherical coordinates, separation of variables | Physics LibreTexts (UC Davis, Physics 9HC) 7.3, "Central Forces" | https://phys.libretexts.org/Courses/University_of_California_Davis/UCD:_Physics_9HC__Introduction_to_Waves_Physical_Optics_and_Quantum_Theory/7:_Quantum_Theory_in_Three_Dimensions/7.3:_Central_Forces | Writes the Laplacian in $(r,\theta,\phi)$ and separates $R(r)\Theta(\theta)\Phi(\phi)$ (passage read) | derivation | page |
| Spherical harmonics | LibreTexts (Pacific Union College, *Quantum Chemistry*) 6.2, "The Wavefunctions of a Rigid Rotator are Called Spherical Harmonics" | https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/06:_The_Hydrogen_Atom/6.02:_The_Wavefunctions_of_a_Rigid_Rotator_are_Called_Spherical_Harmonics | Spherical harmonics as the angular solutions | intro | page |
| Hydrogen quantum numbers | LibreTexts (Pacific Union College) 6.4, "Hydrogen Atomic Orbitals Depend upon Three Quantum Numbers" | https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/06%3A_The_Hydrogen_Atom/6.04%3A_Hydrogen_Atomic_Orbitals_Depend_upon_Three_Quantum_Numbers | $n$, $\ell$, $m_\ell$ and the orbitals they label | intro | page |
| Hydrogen radial equation | Physics LibreTexts (Fowler, *Quantum Mechanics*) 4.6, "The Hydrogen Atom" | https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Quantum_Mechanics_(Fowler)/04:_Angular_Momentum_Spin_and_the_Hydrogen_Atom/4.06:_The_Hydrogen_Atom | Radial equation and Laguerre-polynomial solutions (terms found; passage not read in full) | derivation | page |
| Real versus complex orbitals | LibreTexts (Grinnell College, CHM 364) 6.6, "Orbital Angular Momentum and the p-Orbitals" | https://chem.libretexts.org/Courses/Grinnell_College/CHM_364:_Physical_Chemistry_2_(Grinnell_College)/06:_The_Hydrogen_Atom/6.06:_Orbital_Angular_Momentum_and_the_p-Orbitals | $p_x$, $p_y$ are linear combinations of the $m_\ell=\pm1$ functions, so $m_\ell=\pm1$ cannot be matched one-to-one with them (passage read) | intro | page |

## 03 `from_atoms_to_molecules`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Born-Oppenheimer, H$_2^+$, LCAO | LibreTexts, *Physical Chemistry* 9.2, "The H$_2^+$ Prototypical Species" | https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/09%3A_Chemical_Bonding_in_Diatomic_Molecules/9.02%3A_The_H_Prototypical_Species | Born-Oppenheimer approximation applied to H$_2^+$, LCAO orbitals | intro | page |
| Hybridisation (the valence-bond picture) | LibreTexts, *Organic Chemistry* (Morsch et al.) 1.6, "sp$^3$ Hybrid Orbitals and the Structure of Methane" | https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_(Morsch_et_al.)/01:_Structure_and_Bonding/1.06:_sp_Hybrid_Orbitals_and_the_Structure_of_Methane | The textbook sp$^3$ account of methane | intro | page |
| Methane as molecular orbitals, symmetry | LibreTexts, Rioux, *Quantum Tutorials* 3.18, "A Molecular Orbital Approach to Bonding in Methane" | https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Quantum_Tutorials_(Rioux)/03:_Chemical_Bonding/3.18:_A_Molecular_Orbital_Approach_to_Bonding_in_Methane | Methane MOs, $T_d$ labels (terms found; derivation not read in full) | derivation | page |
| Symmetry-adapted combinations | LibreTexts (Ursinus College, CHEM 322) 5.4.1, "Ligand Group Orbitals and Generator Functions" | https://chem.libretexts.org/Courses/Ursinus_College/CHEM322:_Inorganic_Chemistry/05:_Molecular_Orbitals/5.04:_Polyatomic_MO_Diagrams/5.4.01:_Ligand_Group_Orbitals_and_Generator_Functions | Ligand group orbitals (the inorganic name for symmetry-adapted combinations), methane worked as $T_d$ | intro | page |
| Geometry optimisation | Psi4NumPy, tutorial 13, "Geometry Optimization" | https://github.com/psi4/psi4numpy/tree/master/Tutorials/13_Geometry_Optimization | Worked geometry-optimisation code | derivation | page (exists; contents not read) |

No good free source found for: Born-Oppenheimer as a stand-alone derivation (9.2 above treats it inside H$_2^+$).

## 04 `basis_sets_from_base_states` (listed here only; the notebook has no Further reading cell)

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| LCAO | LibreTexts 9.2 (as in 03) | see 03 | LCAO for H$_2^+$ | intro | page |
| STO-nG, split valence, polarisation, diffuse, Gaussians versus Slater | Sherrill (Georgia Tech), "Basis Sets in Quantum Chemistry" (lecture slides) | https://vergil.chemistry.gatech.edu/basis-sets.pdf | STO-3G, 6-31G, polarisation, diffuse, cc-pVXZ (terms counted in the extracted text) | intro | page |
| STO-nG | Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969) | https://doi.org/10.1063/1.1672392 | "Self-Consistent Molecular-Orbital Methods. I. Use of Gaussian Expansions of Slater-Type Atomic Orbitals" | reference | record |
| def2 | Weigend, Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297 (2005) | https://doi.org/10.1039/b508541a | "Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn" | reference | record |
| def2 diffuse (-D) | Rappoport, Furche, J. Chem. Phys. 133, 134105 (2010) | https://doi.org/10.1063/1.3484283 | "Property-optimized Gaussian basis sets for molecular response calculations" | reference | record (page numbers not returned by Crossref) |
| Look up any basis set | Basis Set Exchange | https://www.basissetexchange.org/ | Browse and download basis sets; the site's reference page is not at `/references` (404) | reference | page |
| Citing the BSE | Pritchard et al., J. Chem. Inf. Model. 59, 4814 (2019) | https://doi.org/10.1021/acs.jcim.9b00725 | "New Basis Set Exchange: An Open, Up-to-Date Resource for the Molecular Sciences Community" | reference | record |
| Basis sets in PySCF | PySCF documentation, "Molecular structure" | https://pyscf.org/user/gto.html | How basis sets and ECPs are specified in PySCF | reference | page |
| BSSE, counterpoise | Boys, Bernardi, Mol. Phys. 19, 553 (1970) $ | https://doi.org/10.1080/00268977000101561 | "The calculation of small molecular interactions by the differences of separate total energies. Some procedures with reduced errors" | reference | record |
| BSSE in practice | Psi4 manual, "Basis Set Superposition Corrections" | https://psicode.org/psi4manual/master/nbody.html | Counterpoise correction as implemented in Psi4 | reference | page |

## 05 `hartree_fock_mean_field`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Helium, variational method | LibreTexts (Pacific Union College) 8.2, "Perturbation Theory and the Variational Method for Helium" | https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.02%3A_Perturbation_Theory_and_the_Variational_Method_for_Helium | Variational treatment of helium | intro | page |
| Orbital approximation, SCF | LibreTexts (Pacific Union College) 8.3, "Hartree-Fock Equations are Solved by the Self-Consistent Field Method" | https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.03%3A_Hartree-Fock_Equations_are_Solved_by_the_Self-Consistent_Field_Method | Self-consistent field idea | intro | page |
| Slater determinants | LibreTexts (Pacific Union College) 8.6, "Antisymmetric Wavefunctions can be Represented by Slater Determinants" | https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/08%3A_Multielectron_Atoms/8.06%3A_Antisymmetric_Wavefunctions_can_be_Represented_by_Slater_Determinants | Slater determinants and the Pauli principle | intro | page |
| Hartree-Fock and SCF in practice | Sherrill (Georgia Tech), "Introduction to Hartree-Fock Molecular Orbital Theory" (lecture slides) | https://vergil.chemistry.gatech.edu/static/content/Hartree-Fock-Intro.pdf | Hartree-Fock energy, Slater determinant, SCF (text extracted and read for its headings) | intro | page |
| Hartree-Fock, written notes (the order of the notebook follows it) | Sherrill (Georgia Tech), "An Introduction to Hartree-Fock Molecular Orbital Theory" (8-page free PDF, June 2000; suggested by Alex) | https://vergil.chemistry.gatech.edu/static/content/hf-intro.pdf | Whole chain: the electronic problem, Hartree product, antisymmetry, Slater determinant and Pauli, mean-field meaning, one- and two-electron operators, energy expression, Hartree-Fock equations with $J$, $K$ and the Fock operator, Roothaan equations, SCF | intro / derivation | read in full, 2026-11-13 |
| Fock matrix, SCF code | Psi4NumPy, tutorial 03, "Hartree-Fock" | https://github.com/psi4/psi4numpy/tree/master/Tutorials/03_Hartree-Fock | Worked HF/SCF code in numpy | derivation | page (exists, mentions HF and SCF; contents not read) |

No good free source found for: effective nuclear charge and screening in helium (the Pacific Union page has no mention of screening), the closed form of $v_J(r)$ for a 1s density (derived in the notebook from the shell theorem and compared with numerical integration), and the notebook's use of $Z_{\text{eff}}(r)=Z-2rv_J(r)$ for lithium (derived there, not taken from a source).

## 06 `reading_a_dft_label`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Kohn-Sham DFT | Hohenberg, Kohn, Phys. Rev. 136, B864 (1964) $ | https://doi.org/10.1103/PhysRev.136.B864 | "Inhomogeneous Electron Gas" | reference | record |
| Kohn-Sham DFT | Kohn, Sham, Phys. Rev. 140, A1133 (1965) $ | https://doi.org/10.1103/PhysRev.140.A1133 | "Self-Consistent Equations Including Exchange and Correlation Effects" | reference | record |
| Kohn-Sham DFT, textbook | Burke, *The ABC of DFT* (free draft, for individual use) | https://dft.uci.edu/learnDFT.php (PDF: https://dft.uci.edu/doc/g1.pdf) | Table of contents lists the Kohn-Sham equations and Jacob's ladder; PDF fetched (104 pages) | intro / derivation | page |
| Functionals, review | Toulouse, "Review of approximations for the exchange-correlation energy in density-functional theory" (arXiv 2103.02645) | https://arxiv.org/abs/2103.02645 | Kohn-Sham scheme, meta-GGA, range separation (terms counted in the PDF) | derivation | page |
| Jacob's ladder | Perdew, AIP Conf. Proc. 577, 1 (2001) | https://doi.org/10.1063/1.1390175 | "Jacob's ladder of density functional approximations for the exchange-correlation energy" | reference | record |
| Functionals, survey | Mardirossian, Head-Gordon, Mol. Phys. 115, 2315 (2017) | https://doi.org/10.1080/00268976.2017.1333644 | "Thirty years of density functional theory in computational chemistry: an overview and extensive assessment of 200 density functionals" | reference | record |
| Which functional/basis to use | Bursch, Mewes, Hansen, Grimme, Angew. Chem. Int. Ed. 61, e202205735 (2022) | https://doi.org/10.1002/anie.202205735 | "Best-Practice DFT Protocols for Basic Molecular Computational Chemistry" | intro / reference | record |
| B97 | Becke, J. Chem. Phys. 107, 8554 (1997) $ | https://doi.org/10.1063/1.475007 | "Density-functional thermochemistry. V. Systematic optimization of exchange-correlation functionals" | reference | record |
| $\omega$B97X | Chai, Head-Gordon, J. Chem. Phys. 128, 084106 (2008) | https://doi.org/10.1063/1.2834918 | "Systematic optimization of long-range corrected hybrid density functionals" | reference | record |
| $\omega$B97X-D | Chai, Head-Gordon, Phys. Chem. Chem. Phys. 10, 6615 (2008) | https://doi.org/10.1039/b810189b | "Long-range corrected hybrid density functionals with damped atom-atom dispersion corrections" | reference | record |
| $\omega$B97X-D3 | Lin, Li, Mao, Chai, J. Chem. Theory Comput. 9, 263 (2013) | https://doi.org/10.1021/ct300715s | "Long-Range Corrected Hybrid Density Functionals with Improved Dispersion Corrections" (online 2012) | reference | record |
| $\omega$B97M-V | Mardirossian, Head-Gordon, J. Chem. Phys. 144, 214110 (2016) | https://doi.org/10.1063/1.4952647 | "$\omega$B97M-V: A combinatorially optimized, range-separated hybrid, meta-GGA density functional with VV10 nonlocal correlation" | reference | record |
| VV10 | Vydrov, Van Voorhis, J. Chem. Phys. 133, 244103 (2010) | https://doi.org/10.1063/1.3521275 | "Nonlocal van der Waals density functional: The simpler the better" | reference | record |
| D3 | Grimme, Antony, Ehrlich, Krieg, J. Chem. Phys. 132, 154104 (2010) | https://doi.org/10.1063/1.3382344 | "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu" | reference | record |
| D3(BJ) | Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011) | https://doi.org/10.1002/jcc.21759 | "Effect of the damping function in dispersion corrected density functional theory" | reference | record |
| $\omega$B97M-D3(BJ) | Najibi, Goerigk, J. Chem. Theory Comput. 14, 5725 (2018) | https://doi.org/10.1021/acs.jctc.8b00842 | "The Nonlocal Kernel in van der Waals Density Functionals as an Additive Correction: An Extensive Analysis with Special Emphasis on the B97M-V and $\omega$B97M-V Approaches". The record matches the notebook's reference; that this paper defines the D3(BJ) variant is **not** confirmed (no abstract available, paper not read) | reference | record |
| Dispersion, review | Grimme, Hansen, Brandenburg, Bannwarth, Chem. Rev. 116, 5105 (2016) $ | https://doi.org/10.1021/acs.chemrev.5b00533 | "Dispersion-Corrected Mean-Field Electronic Structure Methods" | reference | record |
| Functional definitions in code | Psi4 manual, "DFT Functionals" | https://psicode.org/psi4manual/master/dft_byfunctional.html | List of functionals including wB97X, wB97M-V and the -D3 variants | reference | page |
| def2 | Weigend, Ahlrichs 2005; Rappoport, Furche 2010; Weigend 2006 (auxiliary fitting sets) | https://doi.org/10.1039/b508541a, https://doi.org/10.1063/1.3484283, https://doi.org/10.1039/b515623h | see 04; Weigend 2006 is "Accurate Coulomb-fitting basis sets for H to Rn" | reference | record |
| Integration grids (PySCF) | PySCF documentation, "Density functional theory (DFT)" | https://pyscf.org/user/dft.html | Grid level 0 to 9 with default 3; Bragg radii, Treutler-Ahlrichs radial grids, Becke partitioning (passage read) | reference | page |
| Integration grids (papers) | Treutler, Ahlrichs, J. Chem. Phys. 102, 346 (1995) $ | https://doi.org/10.1063/1.469408 | "Efficient molecular numerical integration schemes" | reference | record |
| Integration grids (papers) | Gill, Johnson, Pople, Chem. Phys. Lett. 209, 506 (1993) $ | https://doi.org/10.1016/0009-2614(93)80125-9 | "A standard grid for density functional calculations" (SG-1) | reference | record |
| Integration grids (papers) | Dasgupta, Herbert, J. Comput. Chem. 38, 869 (2017) | https://doi.org/10.1002/jcc.24761 | "Standard grids for high-precision integration of modern density functionals: SG-2 and SG-3" | reference | record |
| ECPs | Hay, Wadt, J. Chem. Phys. 82, 270 (1985) $ | https://doi.org/10.1063/1.448799 | "Ab initio effective core potentials for molecular calculations. Potentials for the transition metal atoms Sc to Hg" | reference | record |
| ECPs | Wadt, Hay, J. Chem. Phys. 82, 284 (1985) $ | https://doi.org/10.1063/1.448800 | "Ab initio effective core potentials for molecular calculations. Potentials for main group elements Na to Bi" (companion paper) | reference | record |
| ECPs, review | Dolg, Cao, Chem. Rev. 112, 403 (2012) $ | https://doi.org/10.1021/cr2001383 | "Relativistic Pseudopotentials: Their Development and Scope of Applications" | reference | record |
| ECPs in code | PySCF documentation, "Molecular structure", section "ECP" | https://pyscf.org/user/gto.html | How an ECP is attached to a molecule in PySCF | reference | page |
| OMol25 | Levine et al., arXiv 2505.08762 | https://arxiv.org/abs/2505.08762 | "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models" (first author Daniel S. Levine, first version 13 May 2025; the notebook's section 10 already records its contents) | reference | record |
| SPICE | Eastman et al., Sci. Data 10, 11 (2023) | https://doi.org/10.1038/s41597-022-01882-6 | "SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials" | reference | record |
| SPICE 2 | Eastman, Pritchard, Chodera, Markland, J. Chem. Theory Comput. 20, 8583 (2024) | https://doi.org/10.1021/acs.jctc.4c00794 | "Nutmeg and SPICE: Models and Data for Biomolecular Machine Learning". Whether this is the paper that describes SPICE 2 is **not** confirmed (no abstract returned), so it is listed as a question, not used in the notebook | reference | record |
| OPLS5 | Damm et al., ChemRxiv (2024) | https://doi.org/10.26434/chemrxiv-2024-2svr7 | "OPLS5: Addition of Polarizability and Improved Treatment of Metals" (Crossref; the notebook's section 10 notes that this preprint was read on 2026-11-05) | reference | record |

## 07 `kohn_sham_matrix`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Hartree-Fock, written notes | Sherrill (Georgia Tech), "An Introduction to Hartree-Fock Molecular Orbital Theory" (8-page notes, June 2000; suggested by Alex) | https://vergil.chemistry.gatech.edu/static/content/hf-intro.pdf | Slater determinant, Roothaan equations, SCF (terms counted in the extracted text: Slater determinant 7, Roothaan 3) | intro / derivation | page |
| Fock matrix, SCF code | Psi4NumPy, tutorial 03, "Hartree-Fock" | https://github.com/psi4/psi4numpy/tree/master/Tutorials/03_Hartree-Fock | Worked HF/SCF code in numpy | derivation | page (exists, mentions HF and SCF; contents not read) |
| Kohn-Sham matrix, DFT code | Psi4NumPy, tutorial 04, "Density Functional Theory" | https://github.com/psi4/psi4numpy/tree/master/Tutorials/04_Density_Functional_Theory | Worked DFT code | derivation | page (exists; contents not read) |
| Kohn-Sham equations | Toulouse (arXiv 2103.02645) and Burke, *ABC of DFT* | https://arxiv.org/abs/2103.02645, https://dft.uci.edu/learnDFT.php | Kohn-Sham equations (headings seen in the table of contents) | derivation | page |
| Original paper | Kohn, Sham 1965 | https://doi.org/10.1103/PhysRev.140.A1133 | see 06 | reference | record |
| Grids in the matrix build | PySCF documentation, "Density functional theory (DFT)" | https://pyscf.org/user/dft.html | see 06 | reference | page |

## 08 `coupled_cluster`

| Topic | Source | URL / DOI | Covers | Level | Checked |
|---|---|---|---|---|---|
| Coupled cluster, introduction | Crawford, Schaefer, in *Reviews in Computational Chemistry* 14, 33 (2000) $ | https://doi.org/10.1002/9780470125915.ch2 | "An Introduction to Coupled Cluster Theory for Computational Chemists" | intro / derivation | record |
| Coupled cluster, review | Bartlett, Musiał, Rev. Mod. Phys. 79, 291 (2007) $ | https://doi.org/10.1103/RevModPhys.79.291 | "Coupled-cluster theory in quantum chemistry" | reference | record |
| CCSD(T) | Raghavachari, Trucks, Pople, Head-Gordon, Chem. Phys. Lett. 157, 479 (1989) $ | https://doi.org/10.1016/S0009-2614(89)87395-6 | "A fifth-order perturbation comparison of electron correlation theories" (the paper that introduces (T)) | reference | record |
| CCD/CCSD code | Psi4NumPy, tutorial 08, "CEPA0 and CCD" | https://github.com/psi4/psi4numpy/tree/master/Tutorials/08_CEPA0_and_CCD | Worked CCD code in numpy (CCSD itself lives in the repository's Coupled-Cluster folder) | derivation | page (exists; contents not read) |
| CCSD code | Crawford Group, *Programming Projects* | https://github.com/CrawfordGroup/ProgrammingProjects | Step-by-step projects up to CCSD | derivation | page (mentions CCSD and Hartree-Fock) |
| CCSD code | Crawford Group, PyCC documentation | https://crawfordgrouppycc.readthedocs.io/en/latest/getting_started.html | Python coupled-cluster code | reference | page |
| CBS extrapolation | Helgaker, Klopper, Koch, Noga, J. Chem. Phys. 106, 9639 (1997) $ | https://doi.org/10.1063/1.473863 | "Basis-set convergence of correlated calculations on water" | reference | record |
| CBS extrapolation | Halkier, Helgaker, Jørgensen, Klopper, Koch, Olsen, Wilson, Chem. Phys. Lett. 286, 243 (1998) $ | https://doi.org/10.1016/S0009-2614(98)00111-0 | "Basis-set convergence in correlated calculations on Ne, N2, and H2O" | reference | record |
| CBS in code | Psi4 manual, `cbs()` function | https://psicode.org/psi4manual/master/api/psi4.driver.cbs.html | Extrapolation schemes named `corl_xtpl_helgaker_2` and `scf_xtpl_helgaker_2/3`, with a CCSD(T) delta-correction example (formulas shown; the $X^{-3}$ form itself is not on that page) | reference | page |

## Questions that the sources did not settle

- `reading_a_dft_label` §9: the CBS form $E_X=E_{\text{CBS}}+A\,X^{-3}$ for the correlation energy. The two Helgaker papers are about this, but I did not read them, and the Psi4 page names the scheme without printing the formula.
- `reading_a_dft_label` §10: that Najibi and Goerigk (2018) is where $\omega$B97M-D3(BJ) is defined, and which parameters were refit.
- `reading_a_dft_label` §10: whether the Psi4 SPICE II recipe ($\omega$B97M-D3(BJ)/def2-TZVPPD) is described in the Nutmeg and SPICE paper above.
- `hartree_fock_mean_field`: no free source found for the effective nuclear charge and screening in helium.
- Not fetchable: the Feynman Lectures site returned HTTP 403 to my fetches (all pages, including the table of contents), so the existing chapter citations were not re-checked and no new Feynman pages were added.
- Not used: Szabo and Ostlund, *Modern Quantum Chemistry* (not open), though the chapter lists it under optional reading.
