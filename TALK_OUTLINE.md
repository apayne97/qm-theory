# Talk outline: from standing waves to DFT labels

Draft 2026-11-05. Each part names the notebook and sections to draw on. Gallery: `pixi run notebooks` → http://localhost:4871 (cards in talk order: numbered symlinks in `gallery/`; to add a notebook, add the next-numbered symlink and restart). The earlier lectures: `pixi run notebooks-part1` → http://localhost:4873 (plane waves, oscillator, whole numbers, identical quanta, spin and belt trick, polarization, archived commutator).

## Part 1. Standing waves in a box (1D → 2D → 3D)
- Whole numbers of half-wavelengths, energy from wavelength — `standing_waves/standing_waves_in_boxes.py` §1
- Product waves, node lines, allowed (kx, ky) lattice, energy as squared distance — §2–3
- Degeneracy and what breaks it — §4
- 3D slices, cube levels 1,3,3,3,1,6 — §5
- Counting states: more dimensions, more levels — §6
- Box to atom: why new coordinates — §7

## Part 2. Spheres and atoms
- New coordinates and polar coordinates in 2D and 3D — `standing_waves/spherical_harmonics_and_hydrogen.py` §1–2
- Nodes and direction: string, drumhead, atom — §3
- Spherical harmonics as standing waves on a sphere (ℓ, m, nodal lines; s, p, d shapes) — §4
- Radial part, effective potential, E = −13.6/n² eV — §5
- Hydrogen orbital gallery, reading an orbital from its nodes — §6

## Part 2, slide order and figures (hydrogen section; slide numbers from the 2026-11-08 deck)

Figure files are written by `pixi run python standing_waves/export_figures.py` into `standing_waves/figures/` (PNG and SVG,
git-ignored); GIFs are `drum_fourier_*_build_up.gif` there and `standing_waves/gifs/` (`export_gifs.py`: strings and drums).

1. **Setup (slides 4-8).** Time-dependent equation (4), time-independent equation (6), sum over stationary states (7).
   Trim slide 8 to the Cartesian lines only (1D and 3D equations, ∇² = ∂ₓ² + ∂ᵧ² + ∂_z²): drop the "∇²ψ(r,ϑ,φ) = ???" line and the
   spherical equation, which are slide 25. Delete hidden slide 9 (it duplicates 26). Fix slide 8's first line: it should be
   −ħ²/2mₑ · ∂²ψ_E/∂x² + V(x)ψ_E (no ∇², no subscript t).
2. **1D and 2D standing waves (10-21).** As now. Add a polar-coordinates slide before or at 13:
   `polar_2d_point` (the same point as (x, y) and as (r, φ)) and `polar_2d_grid` (circles of constant r, rays of constant φ;
   a node at constant r is a circle, at constant φ a diameter). Notebook §2, §3.
3. **Sum of modes (17-21).** Replace the single M = 100 view by `drum_fourier_steps` (M = 1, 3, 10, 30) and `drum_fourier_error`;
   use `drum_fourier_modes_used` for the mode grid (ℓ axis in the angular purple); `drum_fourier_example` is one frame with
   the error curve; the closer can be `drum_fourier_rose_steps` (M = 1, 10, 50, 200) or the GIF `drum_fourier_rose_build_up`.
   Notebook §3, last subsection.
4. **3D coordinates and the Laplacian (new, then 25-27).** Order: a 3D-coordinates slide with `spherical_3d_coordinates`
   (r, ϑ, φ; x lower-left, y right, z up), then 25 (spherical potential, with the ??? line), 26 (cell sizes:
   `central_cartesian_cell` above `central_length_factors`; dV = dx dy dz and r² sinϑ dr dϑ dφ), 27 (∇² = ∂_r² + (2/r)∂_r + Λ/r²,
   −ΛY = ℓ(ℓ+1)Y). Fix slide 27: the left side is ∇²ψ(r,ϑ,φ) but the right side has no ψ (write ∇² = …); the last line
   runs into the References box. Notebook §2 (Laplacian subsection), §5.
5. **Nodes in 3D (23-24, after 27).** 23: nodes are surfaces: `spherical_3d_surfaces` (or the three `spherical_3d_const_*`
   files); in the bullet say "r, ϑ, φ" (not "r, φ, ϑ"). 24: write N_nodes = n − 1 = n_r + n_ϑ + n_φ and ℓ = n_ϑ + n_φ (same
   order as the bullets), ℓ and Y_ℓm in purple (ANGULAR_COLOR), R pink, ψ light blue. Complete the title ("Standing waves in
   3D - node counts"). Notebook §3 table, §4.
6. **Radial part (new).** The slide version is the short lead-in at the top of notebook §5: two eigenvalue problems side by side,
   angular −ΛY = ℓ(ℓ+1)Y (eigenfunctions Y_ℓm) and radial [−ħ²/(2m) d²/dr² + V_eff]u = E u with V_eff = V + ħ²ℓ(ℓ+1)/(2mr²)
   (eigenfunctions u_{n_r ℓ}, labelled by the radial nodes n_r); ℓ(ℓ+1) enters the radial problem as a fixed number, so each ℓ is
   its own 1D problem with its own ladder of E; the centrifugal barrier keeps ℓ > 0 away from r = 0; hydrogen, V = −1/r, gives
   E = −13.6/n² eV with n = n_r + ℓ + 1, shown with `central_levels_l0` to `central_levels_l2` (no series, no Laguerre, and Bessel functions are
   not explained). The derivations are folded under "Details" in the notebook; use them only as backup slides: separation of variables
   (the constant is ℓ(ℓ+1)); the hydrogen solution (limits r^(ℓ+1) and e^(−κr), the series must stop, 1/κ = n_r + ℓ + 1 = n,
   E_n = −1/(2n²)). Notebook §5.
7. **Orbitals (new).** Reading 3d, 4f, 3p from (n_r, n_ϑ, n_φ), the orbital gallery. Closing slide: the energy counts N_nodes (the
   ℓ degeneracy is special to 1/r; drum and oscillator differ). Notebook §6, "What it says" at the end of §5.

Colour keys for the figures: r pink, ϑ orange (cones), φ green (planes), ℓ purple (all angular nodes); Cartesian cell: x blue,
y dark cyan, z teal; positive blue, negative red.

## Part 3. From atoms to organic molecules (approximations)
- Why hydrogen's recipe stops (3(N_nuc+N_el) coordinates) — `standing_waves/from_atoms_to_molecules.py` §1
- Born-Oppenheimer: H2+ energy curve as the nuclear potential, vibration as a 1D standing wave, mass slider — §2
- Orbital approximation: helium with one shared 1s orbital (E(ζ), ζ = Z − 5/16), correlation energy, screening so 2s < 2p — §3
- Hybrids sp, sp², sp³ and the angles from 1 + n cos θ = 0 — §4
- Methane: LCAO symmetry toy (a1 + 3 t2), four equivalent bond orbitals, link to carbon/methane in `~/science/qm-calculations` — §5
- Short bridge also in `spherical_harmonics_and_hydrogen.py` §7. Takeaways for a slide are at the end of §5.

## Part 3b. Basis sets: which functions on the atoms (gallery 04, directly after Part 3)
- Base states, the matrix equation HC = ESC, the variational principle; truncated expansion on a toy well — `standing_waves/basis_sets_from_base_states.py` §1–2
- Atom-centred functions and LCAO, H2+ with two functions (the LCAO calculation behind Part 3's third approximation) — §3
- Slater vs Gaussian, contraction STO-nG — §4
- Shells, zeta, polarization P, diffuse D, water vs basis size — §5
- The `def2-TZVPD` label decoded — §6; leads into Part 4/5

## Part 4. Hartree-Fock: the bridge to DFT
- Slater determinant, J and K, what HF misses — `dft/reading_a_dft_label.py` §2 (needs Part 3 §3 orbital approximation and Part 3b §1 variational principle)

## Part 5. DFT and the labels (longest part)
- The label has three parts — `reading_a_dft_label.py` §1
- Kohn-Sham equations, what is in F — §3; derivation in `dft/kohn_sham_matrix.py` §1–5, numerics §6
- Basis sets in the label (sizes of the two datasets, counterpoise) — §4; the content is Part 3b
- Jacob's ladder, B97, meta — §5
- Range separation ω — §6
- Dispersion: -V vs D3(BJ) — §7
- Grid and settings not in the label — §8
- Both labels decoded (OMol25, SPICE II) — §9
- What is verified and what is not — §10

## Part 6. Reference: coupled cluster
- Why CCSD(T) is the benchmark, size consistency, accuracy ladder, link back to DFT dimers — `dft/coupled_cluster.py` §1, §6–8 (derivation §4 as backup slides)

## Open before the talk
- Review `coupled_cluster.py` (never reviewed)
- Check KaTeX and sliders in a browser
- Verify older citations and Najibi-Goerigk 2018 refit
- Decide on a Born-Oppenheimer bridge notebook
- Timing: Part 5 is the longest; decide what to cut from Parts 1–3
