"""Tests for molecules.py (the bridge notebook `from_atoms_to_molecules.py` and the Hartree-Fock notebook `hartree_fock_mean_field.py`).
Run from the repository root: pytest standing_waves."""

import numpy as np
import pytest

import molecules as m
import hydrogen as hy


# ---- from_atoms_to_molecules.py, section 1

def test_coordinate_counts():
    assert m.coordinate_count(1, 1) == 6  # hydrogen atom
    assert m.coordinate_count(5, 10) == 45  # methane
    assert m.coordinate_count(0, 10) == 30  # methane electrons alone (fixed nuclei)
    assert m.grid_values(2, 10) == 100


# ---- from_atoms_to_molecules.py, section 2

def test_h2plus_minimum_matches_the_large_basis_points():
    r_e, e_min = m.h2plus_minimum()
    r, e = m.h2plus_points()
    assert 1.9 < r_e < 2.1  # H2+ equilibrium is 2.00 bohr
    assert e_min == pytest.approx(e.min(), abs=2e-4)  # lowest computed point is at R = 2.0


def test_fit_passes_through_the_points():
    fit = m.h2plus_fit()
    r, e = m.h2plus_points()
    keep = (r >= 1.25) & (r <= 4.0)
    assert np.max(np.abs(fit(r[keep]) - e[keep])) < 3e-4


def test_vibration_is_about_2300_wavenumbers():
    w = m.harmonic_wavenumber()
    assert 2200 < w < 2400  # experiment: omega_e = 2321 cm^-1 for H2+
    spacing = m.vibrational_spacing_cm()
    assert 2100 < spacing < w  # the real well is anharmonic (wider than a parabola at the top), so levels get closer


def test_nuclear_levels_are_ordered_with_nodes():
    r, e, v = m.nuclear_levels(4)
    assert np.all(np.diff(e) > 0)
    nodes = [int(np.sum(np.sign(col[1:]) != np.sign(col[:-1]))) for col in v.T]
    assert nodes == [0, 1, 2, 3]
    assert np.allclose(np.sum(v**2, axis=0) * (r[1] - r[0]), 1.0)


def test_nuclear_levels_follow_the_mass():
    """Twice the mass: levels near the bottom of the well come closer by 1/sqrt(2)."""
    fit = m.h2plus_fit()
    _, e1, _ = m.nuclear_levels(2, fit, mass=m.REDUCED_MASS_H2PLUS)
    _, e2, _ = m.nuclear_levels(2, fit, mass=2 * m.REDUCED_MASS_H2PLUS)
    assert (e2[1] - e2[0]) / (e1[1] - e1[0]) == pytest.approx(1 / np.sqrt(2), rel=0.03)


def test_electrons_are_the_fast_ones():
    assert m.timescale_ratio() < 0.15


# ---- hartree_fock_mean_field.py, sections 2 and 3: helium

def test_helium_variational_minimum():
    zeta = np.linspace(1.0, 2.5, 150001)
    e = m.helium_energy(zeta)
    assert zeta[np.argmin(e)] == pytest.approx(m.helium_best_zeta(), abs=2e-5)
    assert e.min() == pytest.approx(m.helium_best_energy(), abs=1e-8)
    assert m.helium_best_energy() == pytest.approx(-2.84765625)


def test_helium_energy_sits_between_the_limits():
    assert m.HELIUM_EXACT < m.HELIUM_HF_LIMIT < m.helium_best_energy() < -2.0
    assert m.helium_energy(2.0) == pytest.approx(-4.0 + 1.25)  # no screening: -Z^2 plus 5Z/8


def test_helium_one_orbital_energy_by_numerical_integration():
    """Check 5 zeta / 8 by direct integration: <1/r12> = 2 int rho(r1) [ (1/r1) int_0^r1 rho 4 pi r^2 + int_r1^inf rho 4 pi r ] ..."""
    zeta = 1.6875
    r = np.linspace(1e-6, 25, 250001)
    rho = zeta**3 / np.pi * np.exp(-2 * zeta * r)  # density of one electron
    dr = r[1] - r[0]
    inner = np.cumsum(rho * 4 * np.pi * r**2) * dr  # charge inside r
    outer_integrand = rho * 4 * np.pi * r
    outer = (np.sum(outer_integrand) - np.cumsum(outer_integrand)) * dr  # int_r^inf rho 4 pi r' dr'
    v = np.sum(rho * 4 * np.pi * r**2 * (inner / r + outer)) * dr  # <1/r12> for two electrons in rho
    assert v == pytest.approx(5 * zeta / 8, rel=2e-3)


def test_pure_coulomb_levels_do_not_depend_on_l():
    e = m.screened_levels(decay=1e-9, z=1.0 + 2.0)  # z - 2 = 1: a proton seen from outside, core shrunk to nothing
    assert e[(2, 0)] == pytest.approx(e[(2, 1)], abs=2e-4)
    assert e[(2, 0)] == pytest.approx(-0.125, abs=5e-4)


def test_screening_puts_s_below_p_below_d():
    e = m.screened_levels(decay=0.5, z=3.0)
    assert e[(2, 0)] < e[(2, 1)]
    assert e[(3, 0)] < e[(3, 1)] < e[(3, 2)]
    assert e[(1, 0)] < e[(2, 0)] < e[(2, 1)]


# ---- from_atoms_to_molecules.py, section 3: hybrids

@pytest.mark.parametrize("n,angle", [(1, 180.0), (2, 120.0), (3, np.degrees(np.arccos(-1 / 3)))])
def test_hybrid_angles(n, angle):
    assert m.hybrid_angle_degrees(n) == pytest.approx(angle)


def test_tetrahedral_angle_is_109_47():
    assert m.tetrahedral_angle_degrees() == pytest.approx(109.4712, abs=1e-3)
    assert m.hybrid_angle_degrees(3) == pytest.approx(m.tetrahedral_angle_degrees())


def test_sp3_coefficients_are_orthonormal_and_a_quarter_s():
    c = m.sp3_coefficients()
    assert np.allclose(c @ c.T, np.eye(4))
    assert np.allclose(c[:, 0] ** 2, 0.25)  # s character 1/4
    assert np.allclose(np.sum(c[:, 1:] ** 2, axis=1), 0.75)


def test_sp3_hybrids_by_integration_over_the_sphere():
    """Orthonormality of the real hybrids built from the real harmonics of hydrogen.py, integrated over the sphere."""
    n_t, n_p = 200, 400
    th, ph = np.meshgrid((np.arange(n_t) + 0.5) * np.pi / n_t, (np.arange(n_p) + 0.5) * 2 * np.pi / n_p, indexing="ij")
    x, y, z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    s = np.full_like(th, 1 / np.sqrt(4 * np.pi))
    p = [np.sqrt(3 / (4 * np.pi)) * q for q in (x, y, z)]
    basis = [s, *p]
    c = m.sp3_coefficients()
    funcs = [sum(c[i, k] * basis[k] for k in range(4)) for i in range(4)]
    w = np.sin(th) * (np.pi / n_t) * (2 * np.pi / n_p)
    gram = np.array([[np.sum(f * g * w) for g in funcs] for f in funcs])
    assert np.allclose(gram, np.eye(4), atol=1e-4)


def test_hybrid_profile_is_largest_along_its_axis():
    for n in (1, 2, 3):
        g = np.linspace(0, np.pi, 1801)
        assert np.argmax(m.hybrid_profile(n, g)) == 0
    assert m.hybrid_profile(3, 0.0) == pytest.approx(2.0)  # (1 + 3) / 2


def test_hybrid_profile_normalised_over_the_sphere():
    g = (np.arange(4000) + 0.5) * np.pi / 4000
    for n in (1, 2, 3):
        # int |h|^2 dOmega with h = profile / sqrt(4 pi), axially symmetric: 2 pi int profile^2 sin g dg / (4 pi)
        assert np.sum(m.hybrid_profile(n, g) ** 2 * np.sin(g)) * np.pi / 4000 / 2 == pytest.approx(1.0, abs=1e-5)


# ---- from_atoms_to_molecules.py, section 4: methane

def test_toy_levels_have_the_a1_t2_pattern():
    e, _ = m.methane_toy_levels()
    a1, t2 = m.methane_toy_closed_form()
    assert np.allclose(e, np.sort([a1[0], a1[1], *([t2[0]] * 3), *([t2[1]] * 3)]), atol=1e-9)
    assert e[0] < e[1] and np.allclose(e[1:4], e[1], atol=1e-9)  # a1, then three degenerate t2 (the occupied ones)
    assert e[3] < m.TOY_ENERGIES_EV["H1s"] < e[4]  # four bonding below, four antibonding above the H 1s level


def test_toy_pattern_survives_other_parameters():
    for scale in (0.5, 1.0, 2.0):
        e, _ = m.methane_toy_levels(coupling=scale * m.TOY_COUPLING_EV)
        a1, t2 = m.methane_toy_closed_form(coupling=scale * m.TOY_COUPLING_EV)
        assert np.allclose(e, np.sort([a1[0], a1[1], *([t2[0]] * 3), *([t2[1]] * 3)]), atol=1e-9)
        assert np.allclose(e[1:4], e[1], atol=1e-9)


def test_four_equivalent_bond_orbitals():
    occ, bonds = m.localised_bond_orbitals()
    assert np.allclose(bonds.T @ bonds, np.eye(4), atol=1e-9)  # orthonormal
    proj_occ, proj_bonds = occ @ occ.T, bonds @ bonds.T
    assert np.allclose(proj_occ, proj_bonds, atol=1e-9)  # same occupied space, hence the same density
    for j in range(4):
        w = bonds[:, j] ** 2
        assert w[4 + j] == max(w[4:])  # largest weight on its own H
        assert w[4 + j] > 3 * np.mean([w[4 + k] for k in range(4) if k != j])
    # all four have the same weight on carbon (equivalent by symmetry)
    carbon = bonds[:4, :] ** 2
    assert np.allclose(carbon.sum(axis=0), carbon.sum(axis=0)[0])


def test_toy_bond_points_along_its_corner_and_s_to_p_ratio_follows_the_mixing():
    """The carbon part of bond j is (a/2, (sqrt3 a'/2) n_j): p along the corner, s:p = a^2 : 3 a'^2.
    It is 1:3 (sp3) only if a1 and t2 hold equal carbon amplitude; here they do not, as in the real calculation (20:76)."""
    occ, bonds = m.localised_bond_orbitals()
    a = occ[0, 0] * np.sign(occ[0, 0])
    a1 = np.linalg.norm(occ[:4, 0])
    for j in range(4):
        c = bonds[:, j]
        p_vec = c[1:4]
        assert np.allclose(p_vec / np.linalg.norm(p_vec), m.TETRAHEDRON[j], atol=1e-9)
        assert c[0] == pytest.approx(a / 2)
    ratio = bonds[0, 0] ** 2 / np.sum(bonds[1:4, 0] ** 2)
    assert ratio != pytest.approx(1 / 3, abs=1e-3)  # not exactly sp3


def test_real_methane_numbers_are_consistent():
    r = m.METHANE_REAL
    assert len(r["orbital_energies"]) == 5 and r["orbital_energies"][2] == r["orbital_energies"][4]
    assert sum(r["bond_orbital_on_carbon"].values()) == pytest.approx(1.0, abs=2e-3)


# ---- hartree_fock_mean_field.py, section 4: orbital size and nuclear charge

@pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_mean_radius_formula_against_integration(n, l):
    r = np.linspace(1e-4, 120, 240001)
    p = hy.radial_probability(n, l, r)
    assert np.trapezoid(p, r) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(p * r, r) == pytest.approx(m.hydrogen_like_radius(n, l), rel=1e-6)


def test_quoted_hydrogen_sizes():
    assert m.hydrogen_like_radius(3, 0) == 13.5
    assert m.hydrogen_like_radius(3, 0) * m.BOHR_ANGSTROM == pytest.approx(7.14, abs=0.01)
    assert m.hydrogen_like_radius(1, 0) * m.BOHR_ANGSTROM == pytest.approx(0.79, abs=0.01)


@pytest.mark.parametrize("z", [1.0, 2.0, 7.5, 18.0])
def test_radius_scales_as_one_over_z_and_energy_as_z_squared(z):
    assert m.hydrogen_like_radius(3, 1, z) == pytest.approx(m.hydrogen_like_radius(3, 1, 1.0) / z)
    assert m.hydrogen_like_energy(2, z) == pytest.approx(z**2 * m.hydrogen_like_energy(2, 1.0))
    r = np.linspace(1e-4, 150 / z, 150001)
    p = m.hydrogen_like_density(3, 1, z, r)
    assert np.trapezoid(p, r) == pytest.approx(1.0, abs=1e-5)
    assert np.trapezoid(p * r, r) == pytest.approx(m.hydrogen_like_radius(3, 1, z), rel=1e-5)


def test_helium_ion_orbital_is_half_the_size_of_hydrogen():
    assert m.hydrogen_like_radius(1, 0, 2.0) == pytest.approx(0.75)  # He+ 1s, and zeta = 2 above


def test_atom_sizes_data():
    d = m.atom_sizes()
    rr = dict(zip(d["symbols"], d["mean_radius_bohr"]))
    assert rr["H"] == pytest.approx(1.5, abs=0.02)  # exact hydrogen value
    assert all(0.8 < v < 4.5 for v in rr.values())  # valence orbitals stay between ~0.4 and ~2.3 angstrom
    assert rr["Li"] > rr["Be"] > rr["B"] > rr["C"] > rr["N"] > rr["O"] > rr["F"] > rr["Ne"] - 0.1  # contraction across period 2
    assert rr["Na"] > rr["Ne"] and rr["Li"] > rr["He"]  # a new shell makes the atom larger


# ---- from_atoms_to_molecules.py, section 2.1: geometry optimisation of the H2+ bond length

def test_force_is_minus_the_slope_of_the_curve():
    fit = m.h2plus_fit()
    h = 1e-5
    for r in (1.6, 2.0, 2.8):
        numeric = -(fit(r + h) - fit(r - h)) / (2 * h)
        assert m.h2plus_force(r, fit) == pytest.approx(numeric, abs=1e-8)
    assert m.h2plus_force(3.0, fit) < 0 < m.h2plus_force(1.5, fit)  # stretched bond pulls in, squeezed bond pushes out


@pytest.mark.parametrize("start", [1.5, 2.6, 3.0, 3.8])
def test_optimisation_finds_the_minimum_and_lowers_the_energy_each_step(start):
    fit = m.h2plus_fit()
    run = m.optimise_bond_length(start, fit)
    r_e, e_min = m.h2plus_minimum(fit)
    assert run["r"][-1] == pytest.approx(r_e, abs=2e-3)
    assert run["energy"][-1] == pytest.approx(e_min, abs=1e-6)
    assert abs(run["force"][-1]) < 1e-4
    assert all(b <= a + 1e-12 for a, b in zip(run["energy"], run["energy"][1:]))
    assert len(run["r"]) <= 12  # a quasi-Newton step needs a handful of evaluations in one dimension


def test_optimisation_from_the_minimum_stops_at_once():
    r_e, _ = m.h2plus_minimum()
    assert len(m.optimise_bond_length(r_e)["r"]) == 1


# ---- hartree_fock_mean_field.py, section 3: the average field of the other electron

def _radial_integral(f, upper=40.0, n=400001):
    r = np.linspace(1e-9, upper, n)
    return np.trapezoid(f(r), r)


@pytest.mark.parametrize("zeta", [0.8, 1.6875, 2.0, 2.6875])
def test_coulomb_potential_closed_form_against_numerical_integral(zeta):
    """v_J(r) = int rho(r') / |r - r'| d r' for rho = |phi|^2, by integrating over r' and the angle between r and r' (the angular integral of 1/|r - r'| is 2 pi (|r + r'| - |r - r'|) / (r r'))."""
    rp = np.linspace(1e-9, 30, 600001)
    rho = zeta**3 / np.pi * np.exp(-2 * zeta * rp)
    for r in (0.05, 0.4, 1.0, 2.5, 6.0):
        angular = 2 * np.pi * (r + rp - np.abs(r - rp)) / (r * rp)  # int dOmega' / |r - r'|, in closed form
        # the angular integral equals 4 pi / max(r, r'); written out above so the test does not repeat the shell theorem
        assert angular[1000] == pytest.approx(4 * np.pi / max(r, rp[1000]), rel=1e-9)
        numeric = np.trapezoid(rho * rp**2 * angular, rp)
        assert m.coulomb_potential_1s(r, zeta) == pytest.approx(numeric, rel=1e-6)


def test_coulomb_potential_limits():
    zeta = 1.6875
    assert m.coulomb_potential_1s(0.0, zeta) == pytest.approx(zeta)  # inside the cloud the charge outside cancels
    assert m.coulomb_potential_1s(1e-3, zeta) == pytest.approx(zeta, rel=1e-5)
    far = np.array([15.0, 30.0])
    assert np.allclose(m.coulomb_potential_1s(far, zeta), 1 / far, rtol=1e-9)  # the cloud looks like a point charge
    r = np.linspace(1e-6, 10, 2000)
    v = m.coulomb_potential_1s(r, zeta)
    assert np.all(np.diff(v) < 0) and np.all(v < 1 / r + 1e-12)  # falls steadily, always below the point-charge 1/r
    # the small-r series (used below zeta r = 1e-4) and the closed form join smoothly
    just_below, just_above = 0.99e-4 / zeta, 1.01e-4 / zeta
    assert m.coulomb_potential_1s(just_below, zeta) == pytest.approx(m.coulomb_potential_1s(just_above, zeta), rel=1e-7)


@pytest.mark.parametrize("zeta", [1.0, 1.6875, 2.5])
def test_five_eighths_zeta_is_the_orbital_in_the_field_of_the_other(zeta):
    """5 zeta / 8 = int |phi|^2 v_J d r, and the helium energy is 2 <h> + <phi| v_J |phi>."""
    dens = lambda r: 4 * np.pi * r**2 * zeta**3 / np.pi * np.exp(-2 * zeta * r)
    assert _radial_integral(lambda r: dens(r)) == pytest.approx(1.0, abs=1e-7)
    in_field = _radial_integral(lambda r: dens(r) * m.coulomb_potential_1s(r, zeta))
    assert in_field == pytest.approx(5 * zeta / 8, rel=1e-6)
    # one-electron part <h> = zeta^2/2 - Z zeta, from <-nabla^2/2> = zeta^2/2 and <1/r> = zeta
    mean_inverse_r = _radial_integral(lambda r: dens(r) / r)
    assert mean_inverse_r == pytest.approx(zeta, rel=1e-6)
    assert 2 * (zeta**2 / 2 - 2.0 * mean_inverse_r) + in_field == pytest.approx(m.helium_energy(zeta), rel=1e-6)


def test_helium_repulsion_term_is_the_other_electrons_field_with_the_self_part_removed():
    """Closed shell, one orbital: J[rho_total] phi = 2 v phi and the exchange term K phi = v phi (both electrons are in phi),
    so (J - K/2) phi = v phi: the field of the *one* other electron, with no self-repulsion left."""
    zeta = 1.6875
    r = np.linspace(0.01, 8, 50)
    v = m.coulomb_potential_1s(r, zeta)
    j_total, half_k = 2 * v, 0.5 * 2 * v  # J from the density of both electrons; K/2 with the density of both electrons
    assert np.allclose(j_total - half_k, v)


def test_effective_charge_limits_and_lithium_closed_form():
    z, zeta = 3.0, m.helium_best_zeta(3.0)
    assert zeta == pytest.approx(2.6875)
    assert m.effective_charge(0.0, z) == pytest.approx(3.0)  # the bare nucleus close in
    assert m.effective_charge(40.0, z) == pytest.approx(1.0, abs=1e-9)  # nucleus seen through two core electrons
    r = np.linspace(0.0, 6, 200)
    assert np.allclose(m.effective_charge(r, z), 1 + 2 * np.exp(-2 * zeta * r) * (1 + zeta * r))
    zeff = m.effective_charge(r, z)
    assert np.all(np.diff(zeff) <= 1e-12) and zeff.min() >= 1.0  # falls steadily from 3 to 1
    # the potential is -Zeff/r
    assert np.allclose(m.screened_potential(r[1:], 3.0, 1 / (2 * zeta)), -zeff[1:] / r[1:])
    # a point with z = 10 (neon-like core of 2): the same two-electron core, asymptote z - 2
    assert m.effective_charge(40.0, 10.0) == pytest.approx(8.0, abs=1e-9)


def test_lithium_effective_charge_is_the_integral_of_the_core_density():
    """Zeff(r) = Z - (charge of the core inside r) - r (field of the core charge outside r): the shell theorem, by direct integration."""
    z, zeta = 3.0, 2.6875
    rho = lambda r: 2 * zeta**3 / np.pi * np.exp(-2 * zeta * r)  # two 1s electrons
    for r0 in (0.1, 0.5, 1.5):
        rr = np.linspace(1e-9, 30, 600001)
        inside = np.trapezoid(np.where(rr < r0, 4 * np.pi * rr**2 * rho(rr), 0), rr)
        outside = np.trapezoid(np.where(rr >= r0, 4 * np.pi * rr * rho(rr), 0), rr)
        assert m.effective_charge(r0, z, zeta) == pytest.approx(z - inside - r0 * outside, rel=1e-4)


def test_helium_orbital_energy_and_the_double_counted_repulsion():
    zeta = 1.6875
    eps = m.helium_orbital_energy(zeta)
    assert eps == pytest.approx(-0.896484375)
    assert 2 * eps - 5 * zeta / 8 == pytest.approx(m.helium_energy(zeta))  # E = 2 eps - <v_J>
    assert m.helium_energy(zeta) < 2 * eps  # the sum of the orbital energies lies above the total energy: it counts the repulsion twice
    dens = lambda r: 4 * np.pi * r**2 * zeta**3 / np.pi * np.exp(-2 * zeta * r)
    in_field = _radial_integral(lambda r: dens(r) * m.coulomb_potential_1s(r, zeta))
    assert eps == pytest.approx(zeta**2 / 2 - 2.0 * zeta + in_field, rel=1e-6)


def test_lithium_core_in_the_screened_potential_puts_2s_below_2p():
    """The numbers quoted in the notebook: lithium's own core, zeta = 2.6875, i.e. a = 1/(2 zeta) = 0.0985 angstrom."""
    decay = 1 / (2 * m.helium_best_zeta(3.0))
    assert decay * m.BOHR_ANGSTROM == pytest.approx(0.0985, abs=5e-4)
    e = m.screened_levels(decay=decay, z=3.0)
    assert e[(2, 0)] == pytest.approx(-0.1749, abs=1e-3)
    assert e[(2, 1)] == pytest.approx(-0.1259, abs=1e-3)
    assert e[(1, 0)] < e[(2, 0)] < e[(2, 1)]
