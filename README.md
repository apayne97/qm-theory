# qm-theory

Small numerical experiments for building intuition about quantum mechanics
fundamentals (theory, as opposed to `../qm-calculations`, which runs real DFT).

## Setup

```bash
cd ~/science/qm-theory
conda env create -f environment.yaml   # creates the `qm-theory` env
conda activate qm-theory
```

## Reading order, notation and colours

Read the notebooks in this order: `plane_waves` → `oscillator` → `quantization` → `field_quanta`. `spinors` is independent of the wave notebooks and can be read at any point; `field_quanta` points forward to its exchange part.

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
| `spinors.py` | Spin matrices, `exp(-iθ n·J)`, Bloch vector, interference, SU(2)→SO(3), phasor toy model, commutator helpers |
| `test_spinors.py` | pytest checks of all of the above |
| `dirac_belt.py` | The belt (plate) trick: explicit smooth family of belt orientations (unit quaternions) that returns after 4π but not 2π; rotation-vector map to the ball of rotations; the shrinking family of 4π loops |
| `test_dirac_belt.py` | pytest checks (wall/plate conditions, 4π periodicity, 2π belt is one twist about its axis, continuity, agreement with the 2×2 matrices, surface-crossing parity) |
| `belt_plots.py` | Matplotlib figures for section 6: the belt with its plate, and the ball of rotations |
| `spinor_rotation.py` | Main marimo notebook: 1 rotations and why they matter (symmetry and conservation, rotation matrices, order matters in 3D, classical precession test), 2 J matrices from rotation matrices, 3 spin number of ordinary vectors (spin 1), 4 why that fails for the electron and why the electron matters (Stern-Gerlach magnets in sequence, fermions), 5 building spin 1/2 (spinors and what their rotation matrices must do, amplitudes and sign, two-amplitude model, matrices for all three axes and J, the 2π rotation, double cover), 6 the belt trick (plate and belt with a slider from 0 to 4π; the space of rotations as a ball and shrinking the 4π path) |
| `archived_commutator_loop_and_ladder.py` | Technical material cut from the main notebook: small-rotation commutator loop, ladder-operator derivation, allowed values of m (spacing argument), Noether with commutators and the quantum precession demo, projective-representation remarks |

```bash
cd spinors
python -m pytest -q
marimo edit spinor_rotation.py
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
| `why_whole_numbers.py` | Marimo notebook: 1 free wave, 2 ring, 3 box, 4 same pattern in the oscillator, 5 hydrogen set-up |

```bash
cd quantization
python -m pytest -q
marimo edit why_whole_numbers.py
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
