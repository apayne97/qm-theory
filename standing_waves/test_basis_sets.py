import numpy as np
import pytest

import basis_sets as b
from hydrogen import radial_R


# ----------------------------------------------------------------------------------------------------------------
# Section 2: Morse well in oscillator functions

def fd_hamiltonian(x):
    """Three-point finite-difference Hamiltonian -1/2 d^2/dx^2 + V on a uniform grid with zero at both ends."""
    h = x[1] - x[0]
    n = x.size
    diag = 1 / h**2 + b.morse_potential(x)
    off = -0.5 / h**2 * np.ones(n - 1)
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


def test_morse_levels_agree_with_finite_differences():
    x = np.linspace(-6.0, 30.0, 3601)[1:-1]
    e = np.linalg.eigvalsh(fd_hamiltonian(x))[:5]
    assert np.allclose(e, b.morse_energy(np.arange(5)), atol=2e-4)


def test_morse_ground_state_is_an_eigenfunction():
    x = b.X_GRID
    psi = b.morse_psi0()
    h = x[1] - x[0]
    lap = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / h**2
    residual = -0.5 * lap + b.morse_potential(x[1:-1]) * psi[1:-1] - b.morse_energy(0) * psi[1:-1]
    assert np.max(np.abs(residual)) < 2e-4
    assert b.trapz(psi**2, x) == pytest.approx(1.0, abs=1e-9)


def test_number_of_bound_states():
    assert b.morse_bound_states() == 7
    assert np.all(np.diff(b.morse_energy(np.arange(7))) > 0)  # still rising for the last bound level
    assert b.morse_energy(6) < b.MORSE_D  # all below the dissociation limit


def test_oscillator_functions_are_orthonormal():
    phi = b.ho_functions(12)
    gram = phi @ phi.T * (b.X_GRID[1] - b.X_GRID[0])
    assert np.allclose(gram, np.eye(12), atol=1e-8)
    h, s = b.oscillator_matrices(12)
    assert np.allclose(s, np.eye(12), atol=1e-8)
    assert np.allclose(h, h.T)


def test_truncated_levels_are_upper_bounds_and_fall_with_basis_size():
    err = b.energy_errors(25)
    assert np.all(err[~np.isnan(err)] > -1e-9)  # variational principle
    for k in range(3):
        col = err[:, k][~np.isnan(err[:, k])]
        assert np.all(np.diff(col) <= 1e-9)  # adding a function never raises a level
    assert err[24, 0] < 1e-6
    assert err[24, 2] < 1e-2


def test_a_wide_basis_is_complete_enough_for_the_exact_ground_state():
    psi = b.expansion_wavefunction(30, 0)
    overlap = b.trapz(psi * b.morse_psi0(), b.X_GRID)
    assert abs(overlap) > 0.999999


def test_coefficients_are_amplitudes_with_unit_total_weight():
    e, c = b.variational_levels(8)
    assert np.allclose(np.sum(c**2, axis=0), 1.0)
    psi = b.expansion_wavefunction(8, 1)
    assert b.trapz(psi**2, b.X_GRID) == pytest.approx(1.0, abs=1e-8)


def test_generalised_solver_residual_and_normalisation():
    h, s = b.monomial_matrices(6)
    assert not np.allclose(s, np.eye(6))
    e, c = b.solve_generalised(h, s)
    assert np.allclose(h @ c, s @ c * e, atol=1e-7 * np.abs(h).max())
    assert np.allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-8)


def test_same_span_gives_same_energies_whatever_the_functions():
    n = 6
    e_ho = b.variational_levels(n)[0]
    e_mono = b.solve_generalised(*b.monomial_matrices(n))[0]
    assert np.allclose(e_ho, e_mono, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------
# Section 3: H2+

def prolate_expectation(r, left, right, operator):
    """<left| operator |right> for functions of (r_a, r_b), by Gauss-Legendre quadrature in prolate spheroidal coordinates."""
    mu_max = 1 + 45.0 / r
    xs, ws = np.polynomial.legendre.leggauss(300)
    mu = 0.5 * (mu_max - 1) * (xs + 1) + 1
    wmu = 0.5 * (mu_max - 1) * ws
    nu, wnu = xs, ws
    m, n = np.meshgrid(mu, nu, indexing="ij")
    ra, rb = r * (m + n) / 2, r * (m - n) / 2
    jac = (r / 2) ** 3 * (m**2 - n**2) * 2 * np.pi
    f = left(ra, rb) * operator(ra, rb) * right(ra, rb) * jac
    return float(np.einsum("i,j,ij->", wmu, wnu, f))


def s1(rr):
    return np.exp(-rr) / np.sqrt(np.pi)


@pytest.mark.parametrize("r", [1.0, 2.0, 3.5])
def test_h2plus_integrals_match_numerical_quadrature(r):
    a = lambda ra, rb: s1(ra)
    bb = lambda ra, rb: s1(rb)
    one = lambda ra, rb: 1.0 + 0 * ra
    assert prolate_expectation(r, a, bb, one) == pytest.approx(float(b.sto_overlap(r)), abs=1e-7)
    assert prolate_expectation(r, a, a, lambda ra, rb: 1 / rb) == pytest.approx(float(b.sto_coulomb(r)), abs=1e-7)
    assert prolate_expectation(r, a, bb, lambda ra, rb: 1 / ra) == pytest.approx(float(b.sto_exchange(r)), abs=1e-7)
    assert prolate_expectation(r, a, bb, lambda ra, rb: 1 / rb) == pytest.approx(float(b.sto_exchange(r)), abs=1e-7)


def test_one_s_function_is_an_eigenfunction_of_its_own_atom():
    # the kinetic + own-nucleus part gives -1/2 times the function: used in H_ab = -S/2 - K
    r = 2.0
    a = lambda ra, rb: s1(ra)
    bb = lambda ra, rb: s1(rb)
    # (-1/2 laplacian - 1/r_a) a = -1/2 a, so <a|T - 1/r_a|b> = -1/2 <a|b> by symmetry of the operator
    h_ab = -0.5 * prolate_expectation(r, a, bb, lambda ra, rb: 1.0 + 0 * ra) - prolate_expectation(r, a, bb, lambda ra, rb: 1 / rb)
    assert h_ab == pytest.approx(b.h2plus_matrices(r)[0][0, 1], abs=1e-7)


def test_h2plus_coefficients_are_symmetric_and_antisymmetric():
    r = 2.0
    e, c = b.h2plus_lcao(r)
    s = float(b.sto_overlap(r))
    assert abs(abs(c[0, 0]) - abs(c[1, 0])) < 1e-9 and c[0, 0] * c[1, 0] > 0
    assert abs(c[0, 0]) == pytest.approx(1 / np.sqrt(2 + 2 * s), abs=1e-9)
    assert c[0, 1] * c[1, 1] < 0
    assert abs(c[0, 1]) == pytest.approx(1 / np.sqrt(2 - 2 * s), abs=1e-9)
    h, s_mat = b.h2plus_matrices(r)
    assert e[0] == pytest.approx((h[0, 0] + h[0, 1]) / (1 + s))
    assert e[1] == pytest.approx((h[0, 0] - h[0, 1]) / (1 - s))


def test_h2plus_minimum_of_the_two_function_basis():
    r = np.linspace(1.5, 3.5, 2001)
    e = b.h2plus_total(r)
    i = int(np.argmin(e))
    assert r[i] == pytest.approx(2.49, abs=0.01)   # textbook LCAO-1s value
    assert e[i] == pytest.approx(-0.5648, abs=2e-4)
    assert np.all(b.h2plus_total(r, 1) > -0.5)      # antibonding never binds


def test_two_function_lcao_lies_above_the_large_basis_curve():
    d = b.load_data()["h2plus"]
    for r, e in zip(d["R"], d["energy"]):
        assert b.h2plus_total(r) > e - 1e-9
    # literature value of the exact H2+ ground state at R = 2 bohr, -0.602634 hartree (from memory)
    i = d["R"].index(2.0)
    assert d["energy"][i] == pytest.approx(-0.602634, abs=3e-5)
    # far apart: one hydrogen atom and a proton, -1/2
    assert d["energy"][-1] == pytest.approx(-0.5, abs=0.01)


# ----------------------------------------------------------------------------------------------------------------
# Section 4: Slater and Gaussian functions

def test_slater_and_gaussian_are_normalised():
    r = b.R_GRID
    assert b.trapz(4 * np.pi * r**2 * b.slater_1s(r) ** 2, r) == pytest.approx(1, abs=1e-8)
    for a in (0.1, 1.0, 20.0):
        assert b.trapz(4 * np.pi * r**2 * b.gauss_1s(r, a) ** 2, r) == pytest.approx(1, abs=1e-6)


def test_gaussian_matrix_elements_against_radial_integrals():
    r = b.R_GRID
    al = np.array([0.3, 1.7])
    g = [b.gauss_1s(r, a) for a in al]
    w = 4 * np.pi * r**2
    s = np.array([[b.trapz(w * gi * gj, r) for gj in g] for gi in g])
    v = np.array([[-b.trapz(4 * np.pi * r * gi * gj, r) for gj in g] for gi in g])
    # kinetic energy as 1/2 integral of the product of derivatives
    dg = [-2 * a * r * gi for a, gi in zip(al, g)]
    t = np.array([[0.5 * b.trapz(w * di * dj, r) for dj in dg] for di in dg])
    assert np.allclose(s, b.gauss_overlap_matrix(al), atol=1e-6)
    assert np.allclose(v, b.gauss_attraction_matrix(al), atol=1e-6)
    assert np.allclose(t, b.gauss_kinetic_matrix(al), atol=1e-6)


def test_sto_3g_and_sto_6g_agree_with_the_tabulated_basis_sets():
    ref = b.load_data()["sto_ng"]
    for n, name in ((3, "sto-3g"), (6, "sto-6g")):
        a, c, _ = b.sto_ng_fit(n)
        assert np.allclose(a, ref[name]["exponents"], rtol=2e-3)
        # tabulated coefficients belong to normalised primitives with <g|g> = 1 as well
        assert np.allclose(c, np.array(ref[name]["coefficients"]) / np.sqrt(
            np.array(ref[name]["coefficients"]) @ b.gauss_overlap_matrix(ref[name]["exponents"]) @ np.array(ref[name]["coefficients"])), atol=2e-3)


def test_more_gaussians_fit_better_and_give_better_energies():
    overlaps, energies = [], []
    for n in range(1, 7):
        a, c, o2 = b.sto_ng_fit(n)
        overlaps.append(1 - o2)
        energies.append(b.hydrogen_energy_of_contraction(a, c))
    assert np.all(np.diff(overlaps) < 0)
    assert np.all(np.array(energies) > -0.5)           # variational
    assert np.all(np.diff(energies) < 0)
    assert energies[0] == pytest.approx(-0.4242, abs=1e-3)   # one Gaussian: 85 % of the true binding energy
    assert energies[2] == pytest.approx(-0.4949, abs=1e-3)
    assert energies[5] == pytest.approx(-0.49983, abs=1e-4)


def test_energy_with_free_coefficients_is_lower_than_with_fitted_ones():
    a, c, _ = b.sto_ng_fit(3)
    assert b.best_energy_of_exponents(a) <= b.hydrogen_energy_of_contraction(a, c) + 1e-12
    assert b.best_energy_of_exponents(a) > -0.5


def test_cusp_condition_slater_yes_gaussian_no():
    assert b.cusp_ratio(b.slater_1s) == pytest.approx(-1.0, abs=1e-3)
    assert b.cusp_ratio(lambda r: radial_R(1, 0, r)) == pytest.approx(-1.0, abs=1e-3)   # the exact hydrogen 1s function
    a, c, _ = b.sto_ng_fit(6)
    assert abs(b.cusp_ratio(lambda r: b.contracted_radial(r, a, c))) < 1e-3


def test_gaussian_tails_fall_faster_than_exponentials():
    r = np.array([6.0, 10.0, 14.0])
    a, c, _ = b.sto_ng_fit(3)
    ratio = b.contracted_radial(r, a, c) / b.slater_1s(r)
    assert np.all(np.diff(ratio) < 0)
    assert ratio[0] < 0.9 and ratio[1] < 0.05 and ratio[-1] < 1e-3


def test_product_of_two_gaussians_is_one_gaussian():
    alpha, beta = 0.7, 1.9
    ca, cb = np.array([0.0, 0.0, 0.0]), np.array([1.1, -0.4, 0.8])
    k, p, gamma = b.gaussian_product(alpha, ca, beta, cb)
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50, 3))
    lhs = np.exp(-alpha * np.sum((pts - ca) ** 2, 1)) * np.exp(-beta * np.sum((pts - cb) ** 2, 1))
    rhs = k * np.exp(-gamma * np.sum((pts - p) ** 2, 1))
    assert np.allclose(lhs, rhs)
    assert gamma == alpha + beta
    # the centre lies on the line between A and B, closer to the tighter Gaussian
    assert np.linalg.norm(p - cb) < np.linalg.norm(p - ca)


# ----------------------------------------------------------------------------------------------------------------
# Section 5: shells, split valence, polarization, the PySCF numbers

def test_shell_sizes_and_function_counts():
    assert [b.shell_size(l) for l in range(5)] == [1, 3, 5, 7, 9]
    assert b.function_count({"s": 3, "p": 2, "d": 1}) == 14


def test_function_counts_reproduce_every_water_basis_size():
    for name, row in b.load_water().items():
        n = b.function_count(row["O_shells"]) + 2 * b.function_count(row["H_shells"])
        assert n == row["n_basis"], name


def test_known_water_counts():
    water = b.load_water()
    counts = {k: v["n_basis"] for k, v in water.items()}
    assert counts["def2-svp"] == 24 and counts["def2-tzvp"] == 43 and counts["def2-tzvpd"] == 58
    assert counts["def2-tzvpp"] == 59 and counts["def2-tzvppd"] == 74 and counts["def2-qzvp"] == 117


def test_water_energies_fall_with_basis_size_toward_the_largest_basis():
    water = b.load_water()
    order = ["sto-3g", "def2-svp", "def2-tzvp", "def2-tzvpd", "def2-tzvppd", "def2-qzvp"]
    e = [water[k]["e_water_Ha"] for k in order]
    assert np.all(np.diff(e) < 0)


def test_gaussian_shells_have_the_parity_of_their_harmonic():
    x = np.linspace(-2, 2, 41)
    xx, zz = np.meshgrid(x, x)
    p = b.gaussian_shell(1, 0.8, xx, zz)
    d = b.gaussian_shell(2, 0.8, xx, zz)
    s = b.gaussian_shell(0, 0.8, xx, zz)
    assert np.allclose(p, -p[::-1, :])          # odd under z -> -z
    assert np.allclose(d, d[::-1, :])           # even
    assert np.allclose(s, s[::-1, :]) and np.all(s > 0)
    assert np.allclose(p[20, :], 0)             # p_z is zero in the plane z = 0


def test_more_flexible_valence_follows_a_swollen_orbital_better():
    z = np.array([0.8, 1.0, 1.24, 1.6])
    single = b.split_valence_errors(z, "single")
    double = b.split_valence_errors(z, "double")
    triple = b.split_valence_errors(z, "triple")
    assert np.all(double <= single + 1e-12) and np.all(triple <= double + 1e-12)
    assert single[1] == pytest.approx(1 - b.sto_ng_fit(3)[2], abs=1e-9)   # single zeta at zeta = 1 is the STO-3G fit
    assert single[2] > 10 * triple[2]                                       # frozen shape fails when the orbital shrinks
    assert triple[0] < 1e-2 and triple[3] < 5e-3


def test_hydrogen_atom_pyscf_numbers():
    rows = {r["label"]: r for r in b.load_data()["hydrogen"]}
    for r in rows.values():
        assert r["energy"] >= -0.5 - 1e-9                    # variational
        assert r["polarizability"] <= 4.5 + 0.02             # variational lower bound of the exact 9/2
    assert rows["def2-TZVP, without its p shell"]["polarizability"] == pytest.approx(0.0, abs=1e-4)
    assert rows["STO-3G"]["polarizability"] == pytest.approx(0.0, abs=1e-4)
    # polarization alone helps a little, diffuse p functions are what the field-induced cloud needs
    assert rows["def2-TZVP"]["polarizability"] > 0.3
    assert rows["def2-TZVPD"]["polarizability"] > 5 * rows["def2-TZVP"]["polarizability"]
    assert rows["def2-TZVPPD"]["polarizability"] > 4.3
    # a spherical atom: its energy does not feel the p shell
    assert rows["def2-TZVP"]["energy"] == pytest.approx(rows["def2-TZVP, without its p shell"]["energy"], abs=1e-9)
    assert rows["def2-QZVP"]["energy"] < rows["def2-TZVP"]["energy"] < rows["def2-SVP"]["energy"] < rows["STO-3G"]["energy"]
    assert rows["def2-QZVP"]["energy"] == pytest.approx(-0.5, abs=3e-5)


def test_density_tail_needs_a_big_s_set_and_the_hydrogen_d_is_only_a_p_function():
    rows = {r["label"]: r for r in b.load_data()["hydrogen"]}
    r = np.array(rows["def2-TZVP"]["r"])
    exact = b.density_slater(r)
    i = int(np.argmin(np.abs(r - 10.0)))
    tz = rows["def2-TZVP"]["density"][i] / exact[i]
    qz = rows["def2-QZVP"]["density"][i] / exact[i]
    assert tz < 0.05 < qz                       # def2-TZVP has lost the tail of the density by 10 bohr
    # def2-TZVPD adds a diffuse p function on hydrogen, no new s function: the spherical density is unchanged
    assert np.allclose(rows["def2-TZVPD"]["density"], rows["def2-TZVP"]["density"], rtol=1e-6, atol=1e-12)
    near = 0
    assert rows["def2-QZVP"]["density"][near] == pytest.approx(exact[near], rel=0.05)


def test_split_valence_coefficients_reproduce_the_errors():
    a, _, _ = b.sto_ng_fit(3)
    for kind in ("single", "double", "triple"):
        for z in (0.8, 1.24):
            coef = b.split_valence_coefficients(z, kind)
            r = b.R_GRID
            f = b.contracted_radial(r, a, coef)
            err = b.trapz(4 * np.pi * r**2 * (f - b.slater_1s(r, z)) ** 2, r)   # ||target - projection||^2 = 1 - ||projection||^2
            assert err == pytest.approx(b.split_valence_errors([z], kind)[0], abs=2e-6)


def test_shell_string():
    assert b.shell_string({"d": 1, "s": 3, "p": 2}) == "3s2p1d"
    assert b.shell_string(b.load_water()["def2-tzvppd"]["H_shells"]) == "3s3p1d"


def test_binding_energies_quoted_in_the_notebook():
    hartree_ev = 27.211386
    r = np.linspace(1.5, 3.5, 2001)
    de_lcao = (-0.5 - b.h2plus_total(r).min()) * hartree_ev
    d = b.load_data()["h2plus"]
    de_big = (-0.5 - min(d["energy"])) * hartree_ev
    assert de_lcao == pytest.approx(1.76, abs=0.02)
    assert de_big == pytest.approx(2.79, abs=0.02)
    assert de_lcao / de_big == pytest.approx(0.63, abs=0.01)


# ----------------------------------------------------------------------------------------------------------------
# Section 5b: box standing waves against atom-centred Gaussians (basis_comparison.py)

import basis_comparison as bc  # noqa: E402

BOX = 16.0  # bohr


def test_gaussian_sum_stands_in_for_one_over_r():
    r = np.array([1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0, 30.0])
    assert np.allclose(bc.coulomb_gaussian_sum(r) * r, 1.0, atol=2e-4)


def test_box_matrices_reproduce_a_gaussian_expectation_value():
    """<g|H|g> for g = exp(-alpha r^2) projected onto the box functions, against the closed form -4/(3 pi) at alpha = 8/(9 pi)."""
    alpha = 8 / (9 * np.pi)
    c = bc.box_projection_of_gaussian(BOX, alpha)
    assert c @ c == pytest.approx(1.0, abs=1e-6)
    h = bc._box_hamiltonian(BOX, None, bc.M_BOX)
    assert c @ h @ c == pytest.approx(-4 / (3 * np.pi), abs=1e-4)


def test_box_energies_are_variational_and_fall_with_n():
    ns = bc.box_function_counts()
    for r in (None, bc.R_H2PLUS):
        e = bc.box_energies(BOX, ns, r)
        exact = bc.E_H_EXACT if r is None else bc.E_H2PLUS_EXACT
        assert np.all(e > exact)
        assert np.all(np.diff(e) < 1e-9)  # never rises when functions are added
    assert bc.box_energies(BOX, ns)[-1] - bc.E_H_EXACT < 0.02


def test_bigger_box_is_worse_for_few_functions():
    ns = (10, bc.M_BOX**3)
    small, big = bc.box_energies(10.0, ns), bc.box_energies(20.0, ns)
    assert big[0] > small[0]  # the same N spread over a larger volume


def test_box_counting_grows_with_volume():
    """At a fixed kinetic cutoff the number of functions grows as the volume L^3."""
    n1, n2 = bc.box_count_below_cutoff(10.0, 4.0), bc.box_count_below_cutoff(20.0, 4.0)
    assert n2 / n1 == pytest.approx(8.0, rel=0.15)


def test_gaussian_hydrogen_matches_known_values():
    assert bc.gaussian_atom_energy(1) == pytest.approx(-4 / (3 * np.pi), abs=1e-10)  # the notebook's "about -0.424"
    e = np.array([bc.gaussian_atom_energy(n) for n in range(1, 7)])
    assert np.all(e > -0.5) and np.all(np.diff(e) < 0)
    for n in (3, 6):  # free coefficients and optimised exponents do at least as well as the tabulated STO-nG contraction
        sto = b.hydrogen_energy_of_contraction(*b.sto_ng_fit(n)[:2])
        assert e[n - 1] <= sto + 1e-9
    assert e[2] == pytest.approx(-0.497, abs=0.001)
    assert e[5] == pytest.approx(-0.99995 / 2, abs=0.0002)


def test_general_gaussian_code_matches_the_closed_form_s_integrals():
    for n in (1, 3, 6):
        a = tuple(bc.gaussian_exponents(n))
        assert bc.gaussian_h2plus_general(a) == pytest.approx(bc.gaussian_h2plus_energy_s_analytic(n), abs=1e-4)


def test_gaussian_h2plus_is_variational_and_p_functions_help():
    assert all(bc.gaussian_h2plus_energy_s_analytic(n) > bc.E_H2PLUS_EXACT for n in (1, 3, 6))
    n = 4
    a = tuple(bc.gaussian_exponents(n))
    e_sp = bc.gaussian_h2plus_general(a, (bc.gaussian_p_exponent(n),))
    assert bc.E_H2PLUS_EXACT < e_sp < bc.gaussian_h2plus_energy_s_analytic(n) - 5e-3
    assert e_sp - bc.E_H2PLUS_EXACT < 3e-3


def test_few_gaussians_beat_a_thousand_box_waves():
    box_error = bc.box_energies(BOX, (bc.M_BOX**3,))[0] - bc.E_H_EXACT
    assert bc.gaussian_atom_energy(4) - bc.E_H_EXACT < box_error / 5


def test_neither_expansion_has_the_cusp():
    """Exact 1s density: relative slope -2 at the nucleus. Both finite expansions are much flatter there."""
    r = np.array([0.0, 0.02])
    for psi in (bc.box_psi_on_axis(BOX, 1000, r), bc.gaussian_psi(r, 4)):
        slope = (psi[1] ** 2 - psi[0] ** 2) / 0.02 / psi[0] ** 2
        assert slope > -1.0
