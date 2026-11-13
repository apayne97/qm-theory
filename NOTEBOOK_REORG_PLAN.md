# Reorganisation plan for notebooks 03–06

Plan only; no notebook, chapter or code was changed. Read in full: 03 `from_atoms_to_molecules.py`, 04 `basis_sets_from_base_states.py`, 05 `reading_a_dft_label.py`, 06 `kohn_sham_matrix.py`. Skimmed: 02 §7, 07, the LaTeX chapter, README, ASSUMPTIONS, tests.

## 1. Recommendation (Option B below), and what it costs

**Move the electron–electron material out of 03 and into the Hartree–Fock section of 05, and remove the copies of things that 04, 05 and 06 each re-explain.** No notebook is renamed, added or renumbered, so the gallery symlinks, the pixi tasks and the file names in the chapter stay as they are. Section numbers change inside 03 and 05 only.

Cost:
- About 8 cells of editing in 03, 05 and 06, and about 1 screen of new text (the average-field subsection, §4 below).
- The chapter text for "One orbital per electron" (about 35 lines of LaTeX, two figures) moves from `sec:bridge` to `sec:hf`.
- Every cross-reference of the form "03 §3", "05 §2", "06 §1" must be rechecked (list in §6).
- `helium_figure`, `screening_figure` and `orbital_size_figure` are called from a notebook in `dft/` instead of `standing_waves/`. They import with the `notebooks` task's `PYTHONPATH`, but plain `marimo edit dft/...` may not find `molecules_plots`. Either put `standing_waves` on the path for those tasks or move the three figure functions.

Why this and not less or more:
- Alex's observation (1) is right about the problem and about the fix. See §3.
- Merging 05 §2–3 with 06 (Option C) removes the 05↔06 "see there / see here" loop, but it swaps the gallery order and touches almost every cross-reference. It can be a second step.
- Dedupe-only (Option A) is cheap and should be done anyway, but leaves the orbital approximation defined in the earliest place with the weakest example.

## 2. (a) Inventory of 03–06

"First here" lists concepts introduced in that section and nowhere earlier in 03–06 (earlier notebooks 01/02 aside).

### 03 `from_atoms_to_molecules.py` (742 lines, 10 figures)

| § | Content | First here |
|---|---|---|
| intro, terms | three approximations (fixed nuclei, one orbital per electron, LCAO); glossary | mean field, screening, hybrid, a₁/t₂ |
| 1 | why hydrogen's recipe stops; coordinate-count table (caffeine 378) | $10^N$ grid argument |
| 2 | Born–Oppenheimer; H₂⁺ curve; nuclear levels; mass slider | BO, potential energy surface, vibration as 1D standing wave |
| 2.1 | geometry optimisation: SCF inner loop (flowchart), outer loop (Hellmann–Feynman, Pulay, BFGS, internal coordinates), single point / optimisation / AIMD, 1D BFGS toy, table of which datasets optimise geometries (SPICE, OMol25, OPLS5 a–d) | SCF loop (first appearance), force, Hessian |
| 3 | orbital approximation: helium $E(\zeta)$ with $\tfrac58\zeta$; accuracy table; correlation energy 1.1 eV; screening toy ($2s<2p$); orbital size ∝ $1/Z$; "where this lives in DFT"; basis-set consequences (tight exponents, ECP) | orbital approximation, average field, screening, correlation energy, $Z_{\text{eff}}$ |
| 4 | hybrids sp/sp²/sp³, angle from $1+n\cos\theta=0$ | hybrid, orthogonality |
| 5 | methane LCAO toy (a₁ + t₂); real methane numbers; bond orbitals | LCAO (first use, invented numbers), symmetry labels |
| 6 | checked / not checked | |

### 04 `basis_sets_from_base_states.py` (829 lines, 10 figures)

| § | Content | First here |
|---|---|---|
| 1 | stationary states and base states (recap of the previous lecture); $\mathbf{Hc}=E\mathbf{Sc}$ derived; variational principle | basis function, overlap matrix $\mathbf S$, variational principle |
| 2 | Morse well, truncated oscillator basis; energy never below exact | convergence with $N$ |
| 3 | LCAO and H₂⁺ with two functions, $E_\pm=(H_{AA}\pm H_{AB})/(1\pm S)$ | atom-centred basis, bonding/antibonding |
| 3b | box waves vs Gaussians for H and H₂⁺ (4 reasons; plane waves for solids; BSSE and linear dependence named) | cusp, why localized functions |
| 4 | Slater vs Gaussian; product theorem; STO-nG contraction | cusp, tail, contraction |
| 5 | shells ($2\ell+1$), polarization, zeta, diffuse; water function counts; H polarizability test; water energy vs basis | shell, zeta, P, D |
| 6 | `def2-TZVPD` decoding table; takeaways | |
| 7 | checked / not checked | |

### 05 `reading_a_dft_label.py` (865 lines, 10 figures)

| § | Content | First here |
|---|---|---|
| 1 | the label has three parts; OMol25 vs SPICE II | level of theory |
| 2 | Hartree–Fock: Slater determinant, Pauli, variational principle, $\hat f=-\tfrac12\nabla^2+v+\hat J-\hat K$, matrix form, what HF misses, table HF vs ωB97M-V vs CCSD(T), two ways past HF | determinant, $\hat J$, $\hat K$ (exchange), Fock operator |
| 3 | Hohenberg–Kohn, Kohn–Sham, $E_{xc}$, SCF steps, pieces of $\mathbf F$, $F_{11}$ numbers, SCF figure | density functional, $E_{xc}$, hybrid fraction $a$ |
| 4 | def2/SV/TZ/QZ/P/PP/D table, shells table, BSSE and counterpoise, what a diffuse function is | counterpoise, ghost atoms |
| 5 | Jacob's ladder, B97, meta, reduced gradient, $\tau$ | functional rungs |
| 6 | exact exchange, self-interaction, range separation ω, HOMO test, water dimer bars | ω, self-interaction error |
| 7 | dispersion: VV10 vs D3(BJ), methane dimer, damping | London dispersion |
| 8 | grid | |
| 9 | both labels side by side; OPLS5 as a third source (ωB97X-D3, lacvp+*, ECP, `(-f)`, 6-311++G**, CBS, benzene probe check, "why four levels"); table of all labels | ECP, CBS |
| 10 | checked / not checked | |

### 06 `kohn_sham_matrix.py` (631 lines, 4 figures)

| § | Content | First here |
|---|---|---|
| 1 | the equation $\mathbf{FC}=\mathbf{SC}\varepsilon$ derived again, step by step | matrix-product step (d) |
| 2 | symbol table ($F,S,C,\varepsilon,P,D$); basis functions vs orbitals; two orbitals as recipes | density matrix |
| 3 | two-function example with $a,b,s$; four-step solver ($S^{-1/2}$) | generalized eigenproblem solver |
| 4 | what $F$ is made of: $T,V,J,V^{xc},K^{\text{eff}},V^{nlc}$; closer look at $J$ (potential of the density, products of basis functions); pairs vs orbitals; numerical table of $F_{00}$; energy split | $(\mu\nu\lvert\lambda\sigma)$, $v_J(\mathbf r)=\int\rho/\lvert\mathbf r-\mathbf r'\rvert$ |
| 5 | why SCF; five steps; which terms depend on $D$; hand-run loop; HF for comparison | |
| 6 | numerical checks on water; eigenvalues; HF rebuild | |
| 7 | checked / not checked | |

## 3. (b) Concepts explained more than once

"Canonical" is my proposal for the instance that carries the full explanation; the others should shrink to one sentence and a pointer.

| Concept | Where, and how deeply | Canonical |
|---|---|---|
| **Orbital approximation / average field** | 03 intro (1 para); 03 §3 (helium, one example, depth: formula only); 03 takeaway 3; 05 §2 opening ("of 03 §3, helium was the example") and "What HF misses" ("average position"); 06 §1 ("field of the nuclei and all other electrons"); 06 §4 "closer look at J" (**the only place where the average field is actually written as an integral**); chapter `sec:bridge` and `sec:hf` | 05 §2 (new subsection, §4 below), with 06 §4 as the detailed version |
| **Variational principle** | used in 03 §3 (minimising $E(\zeta)$) before it is stated; stated in 04 §1 (full); restated 05 §2 | 04 §1. The 03 use is an ordering fault, which Option B removes |
| **Matrix equation $\mathbf{Hc}=E\mathbf{Sc}$** | 04 §1 (derived), 04 §3 (H₂⁺ solved), 05 §2 and §3 (stated), 06 §1 (derived again, "slowly"), 06 §3 (H₂⁺-like $2\times2$, solved again), 03 §5 (methane $2\times2$ blocks) | 04 §1 and §3. 06 §1 and the 06 §3 example go; the matrix-product step 06 §1(d) and the $S^{-1/2}$ recipe 06 §3 stay |
| **SCF loop** | 03 §2.1 (flowchart, inner loop); 03 §3 "where this lives in DFT" (a second description); 05 §2 ("next section"); 05 §3 (4 steps, real convergence figure); 06 §5 (5 steps, hand-run loop, second figure with the same name) | one description, in the HF part of 05; 06 §5 keeps the numerics (hand-run loop) and refers to it |
| **Coulomb / Hartree term $J$** | 03 §3 (screening), 05 §2 ($\hat J$), 05 §3 ($E_{\text{H}}$, $\mathbf J$), 06 §4 (full) | 05 §2 for the idea; 06 §4 for the matrix element |
| **Correlation energy** | 03 §3 (1.1 eV, helium), 05 §2 (restated, cites 03), 05 §3 ($E_{xc}$), 07 intro | 05 §2 (with the helium number, which comes along) |
| **Pauli / antisymmetry** | 03 §3 (one parenthesis, points forward), 05 §2 (determinant) | 05 §2 |
| **Screening / $Z_{\text{eff}}$** | 03 §3 three times (helium $\zeta=Z-\tfrac5{16}$, the $2s/2p$ toy, orbital sizes); 02 slide note on $\ell$-degeneracy; chapter technical note | one place, next to $v_J$ (05 §2 in Option B) |
| **LCAO** | 03 §5 (toy, invented numbers), 04 §3 (H₂⁺, canonical), 05 §4 (restated with $\rho=\sum P\chi\chi$), 06 §1–2 (restated; "orbitals vs basis functions") | 04 §3 for the idea; the "basis functions ≠ orbitals" block (06 §2) should live in 04 §3, where the confusion starts |
| **Basis-set decoding table** | 04 §6 (`def2-TZVPD`), 05 §1 (three parts), 05 §4 (SV/TZ/QZ/P/PP/D table), 05 §9 (both labels) | 04 §6 for TZ/P/D; 05 §4 keeps only SV/QZ/PP and the counts |
| **Water shell/function counts** | 04 §5 table and 05 §4 table, both read `data_basis.json` | 04 §5 |
| **Exchange fraction and ω** (0.3 bohr⁻¹, 15 % / 100 %) | 05 §3, §6, §9; 06 §4; ASSUMPTIONS A23 | 05 §6 |
| **$F_{11}$ pieces ($T=76.978$, $V=-99.580$, $J=19.827$, …)** | 05 §3 ("one-off PySCF check") and 06 §4 table ($F_{00}$ read from `data_ks.json`). I compared them to `data_ks.json`: identical numbers | 06 §4. Delete from 05 §3 and from the "from memory" list in 05 §10, which is now out of date |
| **Born–Oppenheimer** | 03 §2 and 2.1; 03 intro; 02 §7 pointer. ASSUMPTIONS A9 cites 02 §7, which holds only a pointer | 03 §2 |
| **Orbital size and basis functions** (tight exponents, ECP) | 03 §3 end, 05 §9 (ECP, Pt), 04 §5/§6 | the basis-set consequences move to 04 §5; the ECP detail stays in 05 §9 |
| **"Which calculations optimise the geometry?"** (table with SPICE, OMol25, OPLS5 a–d) | 03 §2.1; the OPLS5 rows repeat the job table of 05 §9 | 05 §9, as one extra column of the existing job table |

Further flow problems found:
- 03 §2.1 uses the Kohn–Sham operator, $V_{\text{H}}$ and $V_{\text{xc}}$ before anything has introduced them (forward references to 06).
- 04 §3b refers forward to §4 and §5 five times (cusp, "Gaussians stay flat", "polarization functions of section 5"). It belongs after §5.
- 04 §1 spends its first half on stationary states and base states, which is a recap of the plane-wave lecture.
- 05 §1 says the OMol25/SPICE labels were "checked", while 05 §3 calls its own numbers a one-off script.
- ASSUMPTIONS A9 and A11 point to 02 §7 for BO and the orbital approximation; those live in 03. A11 and A18 are near duplicates.

## 4. (d, part 1) Observations (1) and (2)

**(1) Is the orbital approximation better placed in the HF part? Mostly yes.**
- Agree: the full statement (determinant, Pauli, $\hat J$, $\hat K$, SCF, correlation as what the product cannot express) is the HF method. 05 §2 already says so in its first paragraph and then defers to 03 §3 for the example, so the key idea has no home: it is defined in the earliest place (before the variational principle and before antisymmetry exist) and restated in three later places.
- The ordering fault is real: 03 §3 minimises $E(\zeta)$ before 04 §1 introduces the variational principle, and its Pauli remark points forward to 05.
- Nuance: 03 still needs a two-sentence version ("an orbital is a one-electron wave; electrons fill them two at a time; the justification is in the Hartree–Fock part") because 03 §4–5 (hybrids, methane's eight valence electrons filling four levels) use orbitals. The "three approximations" framing of 03 can stay as a preview.
- Nuance: moving helium alone is not enough. The screening toy ($2s<2p$) and the orbital-size discussion rest on the same average-field idea; if they stay behind in 03, the idea is still introduced there. They should move with it (as core or as backup).

**(2) Is it unclear that $\langle 1/r_{12}\rangle=\tfrac58\zeta$ stands for the average field of the others? Yes.**
- In helium "the other electrons" is one electron, so "average of the others" and "the other one" are the same thing, and nothing in the notebook shows a potential or a field. The Details box only says the value was checked by integration.
- Also hidden: with $N>2$ the Hartree term uses the *total* density, which includes the electron itself, and $\hat K$ removes that self-part. For helium, $\hat J-\tfrac12\hat K$ equals the field of the single other electron.

**Smallest change that makes the idea clear (one figure, about 15 lines):**
1. Define the field the other electron makes: $v_J(\mathbf r)=\int\rho_{\text{other}}(\mathbf r')\,/\lvert\mathbf r-\mathbf r'\rvert\,d\mathbf r'$. For $\rho_{\text{other}}=\lvert\varphi\rvert^2$ with $\varphi\propto e^{-\zeta r}$ it has the closed form $v_J(r)=\tfrac1r\bigl[1-e^{-2\zeta r}(1+\zeta r)\bigr]$. It tends to $\zeta$ at $r=0$ (inside the cloud the other electron's charge cancels out) and to $1/r$ far away (the cloud looks like a point charge).
2. Show the figure: $v_J(r)$ with the density $\lvert\varphi\rvert^2$ behind it.
3. State $\tfrac58\zeta=\int\lvert\varphi\rvert^2\,v_J\,d\mathbf r$: the energy of one electron in the average field of the other. I checked this integral by hand: $4\zeta^3\bigl[\tfrac1{4\zeta^2}-\tfrac1{16\zeta^2}-\tfrac1{32\zeta^2}\bigr]=\tfrac58\zeta$. `test_molecules.py` already integrates $\langle1/r_{12}\rangle$ numerically.
4. The lithium example costs nothing new: the screening toy *is* the many-electron case. For a $1s^2$ core of exponent $\zeta_c$ around nuclear charge $Z=3$, $Z_{\text{eff}}(r)=Z-2\,r\,v_J=1+2e^{-2\zeta_c r}(1+\zeta_c r)$. This is the toy's $1+2e^{-r/a}$ with the missing $(1+\zeta_c r)$ factor. The toy's "nucleus 3, minus the two $1s$ electrons seen from outside" becomes a derived statement. Neon (10 electrons) is then one sentence: each electron sees nine others' worth of $v_J$, and in HF the exchange term takes out its own contribution.

**Where it should live:** in the new first subsection of 05 §2 (next to the $\hat J$ definition, Option B). If Option A is chosen instead, in 03 §3 directly after the $E(\zeta)$ formula, with the lithium $Z_{\text{eff}}(r)$ replacing the hand-set toy. Either way 06 §4's "closer look at $J$" gets a one-line back-reference.

## 5. (c) Candidate reorganisations

### Option A: dedupe in place (smallest)

Notebook list (unchanged): 03 atoms→molecules, 04 basis sets, 05 label, 06 KS matrix.

- **Moves:** 03 §2.1 dataset table → 05 §9 (extra column); 04 §3b → after §5 (call it 5b so no numbers change); "basis functions ≠ orbitals" block 06 §2 → 04 §3.
- **Adds:** §4 average-field subsection in 03 §3, with $Z_{\text{eff}}(r)$ replacing the hand-set toy form.
- **Cuts:** 06 §1 (reduce to a recap plus step (d)); 06 §3 hand example; 05 §3 $F_{11}$ numbers and SCF step list; 05 §4 piece table and shell table; 05 §2 restatement of the variational principle; first half of 04 §1; 03 §3 "where this lives in DFT" paragraph (to 04 §5).
- **Bridging text:** about 6 sentences.
- **Talk time:** unchanged, since nothing is removed from the talk path. About −10 % in reading length.
- **Risk:** lowest. No file or symlink changes; few §-number changes (3b→5b only).
- **Does not address:** observation (1). The orbital approximation is still defined before the variational principle and before antisymmetry.

### Option B: HF-first within the existing notebooks (recommended)

Notebook list (same files, same numbers): 03 atoms→molecules, 04 basis sets, 05 label (with a larger §2), 06 KS matrix.

- **03 loses §3.** New 03: 1 coordinates, 2 BO (2.1 reduced to a pointer or marked backup), 3 hybrids, 4 methane. Keeps a two-sentence orbital statement in the intro and in the methane section, plus "the average field is the Hartree–Fock part".
- **05 §2 becomes "Hartree–Fock: each electron in the field of the others".** In order: helium $E(\zeta)$ (moved), §4 average-field subsection with $v_J(r)$, lithium $Z_{\text{eff}}(r)$ and the screening toy ($s<p<d$) (moved), orbital size (moved, basis consequences go to 04 §5), then determinant and Pauli, $\hat J$, $\hat K$, HF equation, matrix form, the SCF loop flowchart (moved from 03 §2.1, one description), what HF misses with the correlation numbers (helium 1.1 eV; the water/methane-dimer table can stay where §6 and §7 compute it), two ways past HF.
- **Cuts and dedupes:** same list as Option A, plus 03 §3's takeaway line and the 05 §2 backward references to 03 (now local).
- **Bridging text:** 03 intro rewritten ("two approximations here, the third is in Part 4"); a hand-over sentence at the end of 03 §4; 04 §6 "to continue" changed to point at the new 05 §2; about 10 sentences in all.
- **Talk time:** Part 3 shrinks by about half (03 had 10 figures; B leaves about 5, two of them backup). The HF part grows by helium, $v_J$ and screening, so the net change is small *unless* screening and orbital size are marked backup in the outline (my suggestion: the toy and sizes are backup; helium, $v_J$, determinant, $J/K$, SCF are core). Net effect with that choice: Part 3 −5 to −6 slides, Part 4 +3. Slide counts are my estimate from figures and tables, not Alex's deck.
- **Risk:** medium. The same files and symlinks; §-numbers change in 03 and 05; a figure-import path issue (see §1); chapter text moves one subsection; ASSUMPTIONS "Notebook" columns for A9, A11, A12, A18, A19 need editing (A-numbers do not).
- **Fallback if 05 gets too heavy** (about 950 lines): put the HF material in a new notebook `hartree_fock.py` between 04 and 05. That adds one gallery entry and renumbers 05→06, 06→07, 07→08 (symlinks, README, outline, 07's "Read first").

### Option C: B, then merge 05 §2–3 with 06 and swap their order

Notebook list: 03 atoms→molecules, 04 basis sets, 05 `kohn_sham_matrix.py` (extended with 05 §2 HF and §3 KS), 06 `reading_a_dft_label.py` (§1, §4–§10, shorter), 07 coupled cluster. Same file names; only the gallery symlinks swap.

- **Moves:** the 05 §2–3 text goes to the front of the KS notebook; its "which terms are in $\mathbf F$" list and the 06 §4 table become one section; the SCF material from 06 §5 follows the HF flowchart; the 05 HF-vs-functional table moves to the label notebook (where the other comparisons are).
- **Cuts:** all of Option B, plus the "stated, not derived" lists in 05 §3 and 06 §1, and a second symbol table.
- **Why consider it:** removes the 05↔06 deferral loop (05 states $\mathbf F$ and defers the derivation to 06; 06 "Read first" sends the reader back to 05). It also makes the label notebook purely decoding, which matches how it is used in the talk (Part 5 draws on 05 §1, §4–§9; 06 is derivation and backup).
- **Talk time:** same as B. The gain is less hopping, not fewer slides.
- **Risk:** highest. Symlink swap and restart of the gallery (card titles are file names); almost every "Read first / Next / see section" line in 03–07; outline Parts 4–5; README rows; ASSUMPTIONS columns A18–A20, A22–A24; chapter `\emph{Notebook:}` lines for `sec:hf`, `sec:dft`, `sec:label`. The merged notebook would be about 1,000 lines.

## 6. Where the numbers and names are cited (risk checklist)

| Place | Cites | Affected by |
|---|---|---|
| `TALK_OUTLINE.md` | Part 3 "03 §1–5", Part 3b "04 §1–6", Part 4 "05 §2 (needs 03 §3)", Part 5 "05 §1, §3, §4–§9; 06 §1–5, §6" | all options (A: 3b→5b only) |
| `README.md` rows for the four notebooks (lines 214, 217, 247 and the 04 row) | section lists | B, C |
| `ASSUMPTIONS.md` / `ASSUMPTIONS_statements.txt` "Notebook" column | A2, A3 (04 §1); A9 (02 §7, already wrong); A10 (05 §1); A11 (02 §7, 05 §2, already wrong); A12 (05 §2); A13 (04 §3, 06 §1); A14–A17 (04); A18 (05 §2); A19 (06 §5); A20 (05 §3, 06); A21 (05 §4); A22–A29 (05 §5–§9) | B, C. A-numbers can stay: keep A11 as the statement and A18 as the single-determinant item rather than merging them, since slides cite the numbers |
| Notebook text | 02 §7 → files; 03 → 05 §2 (×2), 06 (×3), 04 §3; 04 → 03 §1, §2, §5 (read first), 05 §2, 05 §4; 05 → 03 §3 (×3), 04 §1, §4, §6, 06; 06 → 05 §2, §3, §6, 04 §1, §3; 07 → 05 §2 (×3) | B (03 §3 and 05 §2 move), C (everything), A (04 §3b only) |
| `chapter2/quantum-chemistry.tex` | `\emph{Notebook:}` lines for `sec:bridge`, `sec:hf`, `sec:dft`, `sec:label`; bridge figures | B moves one subsection; C changes four lines; `\ref` uses are symbolic (9) so they survive |
| `make_chapter2_figures.py` | `bridge-helium.png`, `bridge-screening.png` call `molecules_plots` | keep file names; no change unless functions move |
| `standing_waves/test_molecules.py` | comments "section 2/3/5", "section 3: orbital size", "section 2: geometry optimisation" (comments only; the tests import `molecules`) | comment edits only |
| `standing_waves/test_basis_sets.py` line 358, `basis_comparison.py` docstring | "Section 3b" | only if 3b is renamed 5b |
| `gallery/` symlinks | `01`…`07` by file name | none in A and B; swap 05/06 in C; one more entry if the new HF notebook is made |
| `pixi.toml` | `gallery/*.py` glob, shared `PYTHONPATH` | none |
| `standing_waves/compute_atom_sizes.py` docstring | "orbital-size figure in `from_atoms_to_molecules.py`" | B (figure moves) |
| `dft/test_*.py` | no section references | none |

## 7. (e) Open questions for Alex

1. How long is the talk and how many minutes per part? That decides whether the screening toy, orbital sizes and 03 §2.1 are core, backup or cut. I would mark them backup in `TALK_OUTLINE.md` in any case.
2. Do you want the orbital approximation introduced *before* the basis sets (as now, but with a thin 03 §3) or *after* them in the HF part (Option B)? B means the audience first meets "orbital" in hybrids/methane with only a two-sentence justification.
3. Should the "which calculations optimise the geometry?" table (SPICE/OMol25/OPLS5) stay in the talk? It is about dataset content, not about BO, and fits better beside the label comparison in 05 §9.
4. 03 §2.1 (SCF flowchart, outer loop, BFGS toy): keep as backup in 03, move the SCF flowchart to the HF part and drop the rest, or drop all of it?
5. Is 04 §3b (box waves vs Gaussians) a main-talk slide or backup? It connects to Part 1 but is not needed to read a label. Move to 5b regardless?
6. Is it acceptable to put the new HF material in `dft/reading_a_dft_label.py` and import the helium/screening figure functions from `standing_waves/`, or should those functions move to `dft/`?
7. Is the merge-and-swap of Option C wanted now, or only after the talk?
8. May A9 and A11 in ASSUMPTIONS be corrected now (they cite 02 §7, which is only a pointer)?

## 8. What I did not check

- Talk-time effects are my estimates from figure and table counts; I did not see the deck.
- I did not run the notebooks or tests. The $\tfrac58\zeta$ integral was checked by hand, and the $F_{11}$ numbers were compared to `dft/data_ks.json`.
- I skimmed 02 and 07 and the chapter for references only; I did not review their content.
