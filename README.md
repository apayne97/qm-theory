# qm-theory

Small numerical experiments for building intuition about quantum mechanics
fundamentals (theory, as opposed to `../qm-calculations`, which runs real DFT).

## Setup

Uses [pixi](https://pixi.sh) (`brew install pixi`); `pixi.lock` pins the exact versions.

```bash
cd ~/science/qm-theory
pixi install              # default environment: numpy, matplotlib, marimo, click, pytest
pixi run notebooks     # read-only gallery of the current talk (Parts 1-6), in talk order, at http://localhost:4871 (code hidden); order = numbered symlinks in gallery/, restart after changes
pixi run notebooks-part1   # the earlier lectures (Part I), in dependency order, at http://localhost:4873; symlinks in gallery_part1/
pixi run edit-all       # file browser for editing, like JupyterLab's file view, at http://localhost:4872
pixi run marimo edit dft/reading_a_dft_label.py   # one notebook
pixi run test             # all pytest checks
pixi install -e compute   # adds PySCF, only needed to rerun dft/compute_dft_data.py
```

## Reading order, notation and colours

Read the notebooks in this order: `plane_waves` → `oscillator` → `quantization` → `field_quanta`. `spinors` is independent of the wave notebooks and can be read at any point; `field_quanta` points forward to its exchange part. `polarization` applies the rotation argument of `spinors` (sections 2 and 3) to light and refers to `plane_waves` only for $E=\hbar\omega$.

**Notation (the same in every notebook, in the figures and in the knowledge-base chapter).**

| Symbol | Meaning | Not used for |
|---|---|---|
| $H$, $A$ | height of a real wave, and its size (largest height) | "amplitude" is reserved for the complex number below |
| $\theta$, $\theta_0$ | phase of a wave, and its starting phase | rotation angle, polar angle |
| $\delta$ | phase difference of two waves or amplitudes | |
| $\gamma$ | common (overall) phase of the two spin amplitudes | |
| $\psi$ | complex wave (water wave, then quantum wave function) | Planck's $h$, which is written $2\pi\hbar$ |
| $a$, $a_1$, $a_2$ | complex amplitude, as in $P=\lvert a_1+a_2\rvert^2$ | |
| $\alpha$ | angle of a rotation in space (spinors, belt trick) | |
| $\vartheta$, $\varphi$ | polar angle and azimuth (hydrogen, ring) | |
| $m$, $m_e$ | mass of a particle, and of the electron | |
| $m_s$, $m_\ell$ | spin component along $z$; orbital component along $z$ | |
| $\hat H$ | Hamiltonian | the height $H$ |

**Colours.** One colour per quantity, defined once in `qmstyle.py` (the study-group slide colours) and used by every plotting module. In notebooks and in the chapter, colour is used only in displayed equations and figures, never in running text, and rotation and spin equations are uncoloured. `qmstyle.latex_definitions()` prints the matching `\definecolor` lines for the chapter, and `python -m pytest qmstyle_test.py` checks that no notebook uses a colour outside the palette or colours running text.

## spinors/

Mathematical foundations of bosons and fermions, for readers without linear algebra, group theory or QM. Part I (written): rotations and spin, and why spin-1/2 needs a 4π rotation to return to itself. Later parts: exchange of identical particles and the spin-statistics link.

| File | What it is |
|---|---|
| `spinors.py` | Spin matrices, `exp(-iθ n·J)`, Bloch vector, interference, SU(2)→SO(3), phasor toy model, commutator helpers, length of the cycle of quarter turns for a given $m$, orientations along the loop of four small turns |
| `test_spinors.py` | pytest checks of all of the above |
| `cycle_plots.py` | Matplotlib figures for the cycle of quarter turns ($1\to i\to-1\to-i$, and the eight-step cycle of $m=\tfrac12$) and the loop of four small turns on the letter F |
| `dirac_belt.py` | The belt (plate) trick: explicit smooth family of belt orientations (unit quaternions) that returns after 4π but not 2π; rotation-vector map to the ball of rotations; the shrinking family of 4π loops |
| `test_dirac_belt.py` | pytest checks (wall/plate conditions, 4π periodicity, 2π belt is one twist about its axis, continuity, agreement with the 2×2 matrices, surface-crossing parity) |
| `belt_plots.py` | Matplotlib figures for section 6: the belt with its plate, and the ball of rotations |
| `rotations_spin_and_the_belt_trick.py` | Main marimo notebook: 1 rotations and why they matter (symmetry and conservation, rotation matrices, the cycle of quarter turns of $i$, order matters in 3D, the loop of four small turns, classical precession test), 2 J matrices from rotation matrices and the rule $[J_x,J_y]=iJ_z$, 3 spin number of ordinary vectors (spin 1), 4 why that fails for the electron and why the electron matters (Stern-Gerlach magnets in sequence, fermions), 5 building spin 1/2 (spinors and what their rotation matrices must do, amplitudes and sign, two-amplitude model, matrices for all three axes and J, the 2π rotation and the same thing as a cycle of 8 quarter turns, double cover), 6 the belt trick (plate and belt with a slider from 0 to 4π; the space of rotations as a ball and shrinking the 4π path) |
| `archived_commutator_loop_and_ladder.py` | Technical material cut from the main notebook: small-rotation commutator loop, ladder-operator derivation, allowed values of m (spacing argument), Noether with commutators and the quantum precession demo, projective-representation remarks |

```bash
cd spinors
python -m pytest -q
marimo edit rotations_spin_and_the_belt_trick.py
```


## polarization/

Light polarization and angular momentum, for readers who know the rotation argument of the spinor notebook: circular polarization as the states $m_s=\pm1$ along the beam, linear polarization as their combination (angle = half the relative phase), why light has only $m_s=\pm1$ (spin 1, transverse field) and not $0$ or larger, the angular momentum per photon and wave plates, twisted beams ($m_j=m_\ell+m_s$, any whole number), and the contrast with the electron (phase factors return after $2\pi$, not $4\pi$).

| File | What it is |
|---|---|
| `light.py` | Jones vectors, rotation matrix and $J$, circular states $\mathbf e_\pm$ and helicity amplitudes, field and polarization ellipse, wave-plate retarder and torque, twisted-beam field, orbital $m_\ell$ by FFT round the axis |
| `test_light.py` | pytest checks (eigenvectors of $J$, $R(\alpha)\mathbf e_\pm=e^{\mp i\alpha}\mathbf e_\pm$, linear light = equal mix with relative phase $2\alpha$, ellipse axes against the traced curve, half- and quarter-wave plates, $\langle m_\ell\rangle$ of single and superposed vortices, brightest-ring radius, $m_j$ takes every integer) |
| `light_plots.py` | Matplotlib figures used by the notebook (colours from `qmstyle.py`: the $m_s=+1$ and $m_s=-1$ amplitudes are the two shades of the complex-amplitude colour) |
| `light_polarization.py` | Marimo notebook: 1 what polarization is (Jones vector, ellipse), 2 turning the beam (eigenvectors of $J$), 3 linear light from two circular lights, 4 why only $m_s=\pm1$ (including why not $0$ or other integers), 5 angular momentum of light and wave plates, 6 twisted beams, 7 compared with the electron |

```bash
cd polarization
python -m pytest -q
marimo edit light_polarization.py
```

## plane_waves/

Energy, momentum and the plane wave, for readers who know only trigonometry: what ω and k mean (first for a real cosine wave, then as the rates at which a complex arrow turns in time and twists in space), E = ħω and p = ħk (Lorentz step in a footnote), and when a wave packet moves like a classical particle (k0·σx ≫ 1).

| File | What it is |
|---|---|
| `waves.py` | Plane wave, free-particle ω(k), exact Gaussian packet and its explicit sum over k, width/spreading time, k0·σx, Lorentz phase coefficients |
| `test_waves.py` | pytest checks (closed form vs explicit sum over k, packet width and velocity, uncertainty product σx·σk = 1/2, boosted rest-frame phase) |
| `wave_plots.py` | Matplotlib figures used by the notebook |
| `plane_waves_and_packets.py` | Marimo notebook: 1 a wave in trigonometric form (A, ω, T, f, k, λ, travelling wave), 2 adding two waves (two-hole interference, trig), 3 the same waves with complex numbers (arrow, Re e^{iθ} = cos θ, ω and k as turning rates), 4 why the waves may be complex (arrows add: interference again; uniform probability for a particle), 5 E and p, 6 wave packets (velocity, spreading, when classical) |

```bash
cd plane_waves
python -m pytest -q
marimo edit plane_waves_and_packets.py
```

## oscillator/

Bound states of the quantum harmonic oscillator, for readers who know only the plane-wave and spin notebooks: the allowed energies $E_n=(n+\tfrac12)\hbar\omega$ and the shapes of the states, their real and imaginary parts turning in time, a superposition of two levels (probability swinging at the classical frequency), and why only these energies are allowed (shooting: the solution must die away at both ends, like the fixed ends of a string). Units $\hbar=m=\omega=1$.

| File | What it is |
|---|---|
| `oscillator.py` | Energy levels, potential and turning points, the eigenfunctions $\psi_n$ by a stable recursion, stationary states and superpositions, and a Runge-Kutta shooting solver with `tail_value(E)` |
| `test_oscillator.py` | pytest checks (normalisation, orthogonality, the Schrödinger equation, n nodes and parity, stationary density, two-level period, shooting sign changes at the allowed energies) |
| `oscillator_plots.py` | Matplotlib figures (levels, state grid with real and imaginary parts, superposition, shooting) and `icon_figure` / `save_icon` |
| `harmonic_oscillator.py` | Marimo notebook: 1 allowed energies and shapes, 2 real and imaginary parts, 3 a superposition, 4 why only these energies |
| `make_icons.py` | Writes bare real-part curves (no axes, transparent background) to `icons/` as SVG and PNG, for use as icons in slides |

```bash
cd oscillator
python -m pytest -q
marimo edit harmonic_oscillator.py
python make_icons.py
```

## field_image/

A wireframe picture of a field with one localised excitation, for slides (a particle drawn as a bump of its field). Original artwork, in the study-group colours, on a transparent background.

| File | What it is |
|---|---|
| `field_image.py` | `field_height` (smooth background plus a bump: Bessel J0 rings under a Gaussian envelope) and `bessel_j0` and `field_figure` / `save_field` |
| `test_field_image.py` | pytest checks (bump at the origin, circular symmetry, dies away, small background, J0 values, several rings) |
| `make_field_images.py` | Writes `images/electron_field_{red,blue}[_labelled].{svg,png}` |

## chapter/

The knowledge-base chapter (`quantum-mechanics.tex`, its figures and PDF) is edited **here**; the lab's knowledge-base repo is only a destination. `lecture.cls` and `commons.sty` are vendored copies so the chapter builds standalone (the lab repo's versions are authoritative).

```bash
cd chapter
latexmk -pdf -outdir=.build quantum-mechanics.tex && cp .build/quantum-mechanics.pdf .
python ../make_chapter_figures.py        # regenerate figures/ after changing qmstyle.py or a plot
python sync_chapter.py ~/science/knowledge-base-qm   # copy tex, PDF, figures; then commit and push there
python -m pytest -q
```

`sync_chapter.py` never commits or pushes. It refuses to run if the destination changed since the last sync (recorded in `.last_sync.json`) or has uncommitted changes, unless `--force`.

## Shared files

| File | What it is |
|---|---|
| `qmstyle.py` | The one colour palette, with the meaning of each colour and a generator for the LaTeX colour definitions |
| `make_chapter_figures.py` | Regenerates the 17 figures of `chapter/figures/` from the plotting modules |
| `qmstyle_test.py` | pytest checks of the palette and of how notebooks use it |

## References

Feynman, R. P., Leighton, R. B., and Sands, M. *The Feynman Lectures on Physics, Vol. III: Quantum Mechanics.* Addison-Wesley, 1965 (New Millennium Edition, Basic Books, 2010-2011). Online edition: Caltech, edited by M. A. Gottlieb and R. Pfeiffer, 2013, <https://www.feynmanlectures.caltech.edu/III_toc.html>. Chapters used: 1 (quantum behavior, the water-wave interference example), 6 (spin one-half), 7 (amplitudes in time), 16 (amplitudes in position), 17 (symmetry and conservation laws). Cite specific results by chapter, section and equation number (the numbering is the same in the printed and online editions).

```bibtex
@book{feynman1965lectures3,
  author    = {Feynman, Richard P. and Leighton, Robert B. and Sands, Matthew},
  title     = {The {F}eynman Lectures on Physics, Vol.~{III}: Quantum Mechanics},
  publisher = {Addison-Wesley},
  address   = {Reading, MA},
  year      = {1965}
}

@online{feynman2013lectures3online,
  author  = {Feynman, Richard P. and Leighton, Robert B. and Sands, Matthew},
  title   = {The {F}eynman Lectures on Physics, Vol.~{III}: Quantum Mechanics},
  editor  = {Gottlieb, Michael A. and Pfeiffer, Rudolf},
  organization = {California Institute of Technology},
  year    = {2013},
  url     = {https://www.feynmanlectures.caltech.edu/III_toc.html},
  urldate = {2026-10-14}
}
```

## quantization/

Where whole numbers come from, for readers of the plane-wave and oscillator notebooks: a free wave allows every $k$ (and any overall phase); whole numbers appear only when the wave must satisfy a condition (one value after a full circuit of a ring, zero at the walls of a box, decay at both ends of an oscillator); hydrogen as three such conditions ($m$ from the azimuth, $\ell$ from the polar angle, $n$ from the radius). Units $\hbar=m_e=1$, plus eV and nm for the box.

| File | What it is |
|---|---|
| `quantization.py` | Free wave and energy, ring wave and seam mismatch $2|\sin\pi q|$, box states and energies (units and eV/nm), hydrogen levels and the $(\ell,m)$ count |
| `test_quantization.py` | pytest checks (free wave solves the Schrödinger equation for any $k$, overall phase leaves the density unchanged, seam matches only for whole $q$, box ends vanish at $kL=n\pi$, normalisation and orthogonality, $n^2$ degeneracy) |
| `quantization_plots.py` | Matplotlib figures (free wave, ring, box, box levels against length, ring/box/oscillator comparison, hydrogen levels); imports `hermite_function` from `../oscillator` |
| `where_whole_numbers_come_from.py` | Marimo notebook: 1 free wave, 2 ring, 3 box, 4 same pattern in the oscillator, 5 hydrogen set-up |

```bash
cd quantization
python -m pytest -q
marimo edit where_whole_numbers_come_from.py
```

## field_quanta/

Why every quantum of a field is the same, for readers of the oscillator and quantization notebooks. Opens with the double slit (the voltage fixes the energy of every electron, the intensity fixes only how many), separates the two meanings of "quantization" (levels of one particle versus the count of quanta of a field), shows that a free electron has no levels and a located one is a packet, then builds a ring of masses on springs: each mode is an oscillator with equally spaced levels $(n+\tfrac12)\hbar\omega_k$, so each added quantum costs the same $\hbar\omega_k$ and carries $\hbar k$. Properties (mass floor $m_ec^2$, charge as coupling) belong to the field's dispersion relation; counting of identical quanta; the 0-or-1 rule for electrons is stated, not derived. Units $\hbar=1$ for the chain, eV and nm for the electron.

| File | What it is |
|---|---|
| `field_quanta.py` | Ring-of-masses dispersion relation and modes (also from Newton's laws by matrix), ladder energies and the cost of one more quantum, field-state energy and momentum, counting of states (named, boson, fermion), electron wavelength from voltage, double-slit pattern, electron field energy with rest-energy floor, Gaussian packet; imports `energy` from `../oscillator` |
| `test_field_quanta.py` | pytest checks (dispersion against the matrix, running wave solves Newton's law, stiffer springs scale every frequency, equal ladder steps against growing box steps, energy and momentum add up, state counts, fringe spacing follows the wavelength, rest-energy floor, packet $\sigma_x\sigma_k=\tfrac12$) |
| `field_quanta_plots.py` | Matplotlib figures (simulated double-slit counts, packet in $x$ and $k$, chain mode and dispersion, ladder, stiffness, electron energy and floor, state counting); imports `hermite_function` from `../oscillator` |
| `what_makes_quanta_identical.py` | Marimo notebook: 1 the double slit, 2 two kinds of whole number, 3 one electron in vacuum (packets), 4 a ring of masses, 5 each mode counts quanta, 6 properties belong to the field, 7 indistinguishable quanta and the electron rule, 8 both together and the limits |

```bash
cd field_quanta
python -m pytest -q
marimo edit what_makes_quanta_identical.py
```

## dft/

How to read a DFT label such as ωB97M-V/def2-TZVPD (OMol25) or ωB97M-D3(BJ)/def2-TZVPPD (SPICE II), one piece at a time, with PySCF calculations on water, the water dimer and the methane dimer behind each piece.

| File | What it is |
|---|---|
| `reading_a_dft_label.py` | Marimo notebook: 1 the three parts of a label, 2 Hartree-Fock (Slater determinant, exchange, what HF misses), 3 Kohn-Sham DFT, the SCF loop and what is in the F matrix, 4 basis sets (def2, TZ, P, D, counterpoise), 5 Jacob's ladder, B97 and meta, 6 exact exchange and range separation (ω), 7 dispersion (-V against D3(BJ)), 8 the integration grid, 9 both labels decoded, 10 what is not verified |
| `compute_dft_data.py` | PySCF script that writes `data_scf.json`, `data_basis.json`, `data_ladder.json`, `data_dispersion.json` (about 5 to 20 min each); needs `pixi run -e compute` |
| `dft_data.py`, `dft_plots.py` | Loader and analytic helpers; figures |
| `kohn_sham_matrix.py` | Supplementary marimo notebook: where $FC=SC\varepsilon$ comes from, the symbols, a two-function example, the terms of the Kohn-Sham matrix $F$ ($T$, $V$, $J$, $V^{xc}$, VV10, exact exchange), the SCF loop, and numerical checks on water |
| `compute_ks_matrix.py`, `ks_plots.py`, `test_ks.py` | PySCF script writing `data_ks.json` (`pixi run -e compute python dft/compute_ks_matrix.py`, about 15 s); figures; pytest checks |
| `test_dft.py` | pytest checks of the helpers and of the computed numbers |
| `coupled_cluster.py` | Marimo notebook: 1 the problem (excited determinants, full CI and its cost), 2 CISD and size consistency (two H$_2$), 3 the coupled cluster idea ($e^{\hat T}\Phi_0$, what the exponential contains), 4 deriving the working equations (similarity-transformed Hamiltonian, projection, the series ending at fourth order, CCD and CCSD equations in spin-orbital form, the iteration), 5 a numpy CCSD checked against PySCF, 6 the (T) correction, 7 cost, 8 accuracy ladder (water, stretched water, the dimers of the DFT notebook), 9 what is not verified and four takeaway lines |
| `cc_numerics.py` | Pure-numpy spin-orbital toolkit: integrals to spin orbitals, MP2, CCSD/CCD iteration (with DIIS), the expanded CCD equation, (T), a determinant-based CI (full or truncated), non-interacting fragment pairs, counting |
| `compute_cc_data.py`, `cc_plots.py`, `test_cc.py` | PySCF script (click CLI, `--part` one of all, small, dimer, series, timing, about 1 min) writing `data_cc.json` and `data_cc_integrals.npz`; figures; pytest checks (CCSD, (T), MP2, FCI, CISD against PySCF to $10^{-8}$ Ha, two-electron CCSD = FCI, additivity of CCSD and non-additivity of CISD, quadruple coefficient = product of doubles) |

```bash
pixi run marimo edit dft/reading_a_dft_label.py
pixi run marimo edit dft/coupled_cluster.py
pixi run python -m pytest -q dft
pixi run -e compute python dft/compute_cc_data.py   # regenerates data_cc*.json / .npz
```

## standing_waves/

Standing waves of a particle in a box in one, two and three dimensions, and why confinement gives energy levels. First of two notebooks leading to hydrogen orbitals; the second (spherical harmonics and hydrogen) follows below the table. Atomic units in the second notebook.

| File | What it is |
|---|---|
| `standing_waves_in_boxes.py` | Marimo notebook: 1 one dimension (whole numbers of half-wavelengths), 2 two dimensions (product waves, node lines), 3 the allowed waves as a lattice in $(k_x,k_y)$ with energy as squared distance, 4 degeneracy and what breaks it (side-ratio fan), 5 three dimensions (slices, cube levels), 6 counting states in 1D/2D/3D, 7 from a box to an atom |
| `standing_waves.py` | Box waves, energies, level and degeneracy enumeration |
| `standing_waves_plots.py` | Matplotlib figures |
| `test_standing_waves.py` | pytest checks (zero on walls, orthonormality, Schrödinger equation by finite differences, square and cube degeneracies, state counts) |
| `spherical_harmonics_and_hydrogen.py` | Marimo notebook B: 1 from a box to an atom (spherical coordinates, $\psi=R\,Y$), 2 polar coordinates in 2D and 3D (conversions, coordinate grid and surfaces, area and volume elements; interactive point; slide figures from `export_figures.py`), 3 nodes and their direction (string points, drumhead circles and diameters, atom spheres, cones and planes; one table across 1D/2D/3D; animated string and drum modes), 4 standing waves on a sphere (why $m_\ell$ and $\ell$ are whole numbers, $\ell(\ell+1)$, degeneracy $2\ell+1$, node counts, where the nodal planes and latitude circles sit (interactive), gallery of real harmonics, and the same harmonics as a Fourier series on a circle and a sphere), 5 the radial part (effective potential, finite-difference solution against $-1/(2n^2)$, $n^2$ states per level), 6 hydrogen orbitals and reading an orbital from its nodes, 7 bridge to molecules and takeaways |
| `hydrogen.py` | numpy only (no scipy): associated Legendre functions and $Y_{\ell m}$ (complex and real), finite-difference angular operator, polar-axis shooting, exact radial functions (Laguerre recursion), finite-difference radial solver, orbitals, density sampling, Bessel functions and zeros for the drumhead, `node_geometry` (spheres, cones, planes of an orbital) |
| `coordinates.py` | numpy only: polar and spherical coordinates, conversions both ways, polar grid lines |
| `coordinates_plots.py` | Matplotlib figures of polar and spherical coordinates (notebook size and slide size) |
| `export_figures.py` | click CLI: writes the coordinate figures as PNG and SVG slide files to `standing_waves/figures/` (git-ignored) |
| `export_gifs.py` | click CLI: writes every string and drum animation as GIF to `standing_waves/gifs/` (git-ignored) |
| `from_atoms_to_molecules.py` | Marimo notebook, the bridge to molecules: 1 why hydrogen's recipe stops (coordinate counts), 2 Born-Oppenheimer (H$_2^+$ energy curve as the nuclear potential, nuclear levels, mass slider), 3 one orbital per electron (helium with one shared 1s orbital, screening, why $2s<2p$), 4 hybrid orbitals and the angles $180^\circ,120^\circ,109.47^\circ$ from orthogonality, 5 methane (symmetry toy with one $a_1$ and three $t_2$ levels, four equivalent bond orbitals, comparison with the real calculation in `qm-calculations`), 6 what was checked |
| `molecules.py`, `molecules_plots.py`, `test_molecules.py` | numpy numerics, figures and 25 pytest checks of the bridge notebook |
| `hydrogen_plots.py` | Matplotlib figures (coordinates, string, drumhead modes, node anatomy of an orbital, harmonic shape and map, node placement, gallery, polar-axis shooting, effective potential, radial states, level check, orbital views, orbital gallery) |
| `test_hydrogen.py` | pytest checks (orthonormality of $Y_{\ell m}$ and of whole orbitals, eigenvalue $\ell(\ell+1)$ by finite differences, node counts, finite on the axis only for whole $\ell$, radial equation, energies against $-1/(2n^2)$, normalisation, degeneracy $n^2$, $\langle r\rangle$, density sampling, drumhead modes and Bessel zeros, nodes of orbitals lie on the stated surfaces) |

```bash
pixi run marimo edit standing_waves/standing_waves_in_boxes.py
pixi run marimo edit standing_waves/spherical_harmonics_and_hydrogen.py
pixi run python -m pytest -q standing_waves
```
