# Talk outline: from standing waves to DFT labels

Draft 2026-11-05. Each part names the notebook and sections to draw on. Gallery: `pixi run notebooks` → http://localhost:4871.

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

## Part 3. From atoms to organic molecules (approximations)
- Born-Oppenheimer, orbital approximation, LCAO — `spherical_harmonics_and_hydrogen.py` §7 (short bridge only)
- Link to carbon/methane in `~/science/qm-calculations`
- **Gap:** only a short bridge; possible dedicated notebook.

## Part 4. Hartree-Fock: the bridge to DFT
- Slater determinant, J and K, what HF misses — `dft/reading_a_dft_label.py` §2

## Part 5. DFT and the labels (longest part)
- The label has three parts — `reading_a_dft_label.py` §1
- Kohn-Sham equations, what is in F — §3; derivation in `dft/kohn_sham_matrix.py` §1–5, numerics §6
- Basis sets (def2, TZ, P, D), counterpoise — §4
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
