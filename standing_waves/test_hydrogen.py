import numpy as np
import pytest

from hydrogen import (
    HARTREE_EV,
    angular_degeneracy,
    angular_operator_fd,
    count_sign_changes,
    degeneracy,
    effective_potential,
    hydrogen_energy,
    hydrogen_energy_ev,
    latitude_nodes,
    legendre_assoc,
    mean_radius,
    nodal_structure,
    numerical_levels,
    polar_pole_coefficient,
    polar_solution,
    psi_real,
    quantum_numbers,
    radial_R,
    radial_u,
    real_ylm,
    sample_density,
    solve_radial,
    sphere_inner,
    sphere_quadrature,
    ylm,
)

LM = [(l, m) for l in range(5) for m in range(-l, l + 1)]


def test_quadrature_weights_sum_to_sphere_area():
    _, _, w = sphere_quadrature()
    assert w.sum() == pytest.approx(4 * np.pi)


def test_complex_harmonics_are_orthonormal():
    for a in LM:
        for b in LM:
            val = sphere_inner(lambda t, p: ylm(*a, t, p), lambda t, p: ylm(*b, t, p))
            assert val == pytest.approx(1.0 if a == b else 0.0, abs=1e-10), (a, b)


def test_real_harmonics_are_orthonormal():
    for a in LM:
        for b in LM:
            val = sphere_inner(lambda t, p: real_ylm(*a, t, p), lambda t, p: real_ylm(*b, t, p))
            assert val == pytest.approx(1.0 if a == b else 0.0, abs=1e-10), (a, b)


def test_known_values_of_low_harmonics():
    t, p = 0.7, 0.4
    assert ylm(0, 0, t, p) == pytest.approx(1 / np.sqrt(4 * np.pi))
    assert ylm(1, 0, t, p) == pytest.approx(np.sqrt(3 / (4 * np.pi)) * np.cos(t))
    assert real_ylm(1, 1, t, p) == pytest.approx(np.sqrt(3 / (4 * np.pi)) * np.sin(t) * np.cos(p))
    assert real_ylm(2, 0, t, p) == pytest.approx(np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(t) ** 2 - 1))


def test_legendre_matches_closed_forms():
    x = np.linspace(-0.9, 0.9, 7)
    assert legendre_assoc(2, 0, x) == pytest.approx((3 * x**2 - 1) / 2)
    assert legendre_assoc(3, 0, x) == pytest.approx((5 * x**3 - 3 * x) / 2)
    assert legendre_assoc(2, 1, x) == pytest.approx(3 * x * np.sqrt(1 - x**2))
    assert legendre_assoc(3, 2, x) == pytest.approx(15 * x * (1 - x**2))


@pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, -2), (3, 2), (4, -3)])
def test_angular_eigenvalue_is_l_times_l_plus_one(l, m):
    # Evaluate away from the poles and the mesh edges; the operator is built from finite differences.
    theta = np.linspace(0.4, np.pi - 0.4, 9)[:, None] * np.ones((1, 7))
    phi = np.ones((9, 1)) * np.linspace(0.1, 6.0, 7)[None, :]
    f = lambda t, p: real_ylm(l, m, t, p)  # noqa: E731
    lhs = angular_operator_fd(f, theta, phi, h=1e-3)
    rhs = l * (l + 1) * f(theta, phi)
    assert lhs == pytest.approx(rhs, abs=2e-4)


def test_noninteger_l_is_not_an_eigenvalue_of_a_harmonic():
    # The operator acting on a real harmonic of l = 2 does not give 3 * 4 = 12 times it, nor any other value.
    theta, phi = np.array([[0.9]]), np.array([[0.5]])
    f = lambda t, p: real_ylm(2, 1, t, p)  # noqa: E731
    ratio = angular_operator_fd(f, theta, phi) / f(theta, phi)
    assert ratio == pytest.approx(6.0, abs=1e-3)
    assert abs(ratio - 12.0) > 1


@pytest.mark.parametrize("l,m", [(1, 0), (2, 1), (3, 0), (3, 2), (4, 1), (4, 3), (5, 2)])
def test_node_counts(l, m):
    through_poles, latitudes = nodal_structure(l, m)
    assert through_poles + latitudes == l
    # circles of latitude: sign changes of Y along a meridian where it does not vanish identically
    theta = np.linspace(1e-4, np.pi - 1e-4, 4001)
    along_meridian = real_ylm(l, abs(m), theta, 0.0)
    assert count_sign_changes(along_meridian) == l - abs(m) == latitudes
    assert len(latitude_nodes(l, m)) == latitudes
    # circles through the poles: sign changes of cos(|m| phi) around the equator, two per meridian circle
    phi = np.linspace(0.0, 2 * np.pi, 8001)[:-1] + 1e-5
    around_equator = real_ylm(l, abs(m), np.pi / 2 + 0.3, phi) * np.ones_like(phi)
    changes = count_sign_changes(np.append(around_equator, around_equator[0]))
    assert changes == 2 * abs(m) if abs(m) > 0 else changes == 0


def test_equator_is_a_node_exactly_when_l_minus_m_is_odd():
    for l, m in LM:
        assert (abs(real_ylm(l, abs(m), np.pi / 2, 0.3)) < 1e-12) == ((l - abs(m)) % 2 == 1)


def test_degeneracy_of_an_angular_level():
    for l in range(7):
        assert angular_degeneracy(l) == 2 * l + 1


def test_finite_on_the_polar_axis_only_for_whole_l():
    # even start: c = 0 at even l; odd start: c = 0 at odd l; neither at half-integers or generic values
    whole = np.arange(0, 6)
    c_even = polar_pole_coefficient(whole, "even")
    c_odd = polar_pole_coefficient(whole, "odd")
    assert np.all(np.abs(c_even[::2]) < 1e-4)
    assert np.all(np.abs(c_odd[1::2]) < 1e-4)
    assert np.all(np.abs(c_even[1::2]) > 0.3)
    assert np.all(np.abs(c_odd[::2]) > 0.3)
    generic = np.array([0.5, 1.5, 2.5, 3.3, 0.8])
    for parity in ("even", "odd"):
        assert np.all(np.abs(polar_pole_coefficient(generic, parity)) > 0.05)


def test_pole_coefficient_has_a_zero_crossing_at_each_whole_l():
    ell = np.linspace(0.0, 5.0, 101)
    c = polar_pole_coefficient(ell, "even")
    for target in (2, 4):
        i = int(round(target / 0.05))
        assert c[i - 1] * c[i + 1] < 0 and abs(c[i]) < 1e-4


# ---------------------------------------------------------------------------------------------------------------
# radial part


def test_energy_values():
    assert hydrogen_energy(1) == -0.5
    assert hydrogen_energy_ev(1) == pytest.approx(-13.6057, abs=1e-3)
    assert HARTREE_EV / 2 == pytest.approx(13.6057, abs=1e-3)
    assert hydrogen_energy_ev(2) == pytest.approx(-3.4014, abs=1e-3)


def test_effective_potential_shape():
    r = np.array([0.5, 1.0, 2.0])
    assert effective_potential(r, 0) == pytest.approx(-1 / r)
    assert effective_potential(r, 1) == pytest.approx(-1 / r + 1 / r**2)
    # minimum of the l = 1 well at r = l (l + 1) = 2 with depth -1 / (2 l (l + 1)) = -1/4
    rr = np.linspace(0.5, 10, 20001)
    v = effective_potential(rr, 1)
    assert rr[np.argmin(v)] == pytest.approx(2.0, abs=1e-3)
    assert v.min() == pytest.approx(-0.25, abs=1e-6)


@pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3), (5, 2)])
def test_radial_functions_are_normalised(n, l):
    r = np.linspace(0, 150, 300001)
    assert np.trapezoid(radial_R(n, l, r) ** 2 * r**2, r) == pytest.approx(1.0, abs=1e-6)


def test_radial_functions_with_same_l_are_orthogonal():
    r = np.linspace(0, 150, 300001)
    for l in range(3):
        for n1 in range(l + 1, 5):
            for n2 in range(n1 + 1, 6):
                val = np.trapezoid(radial_u(n1, l, r) * radial_u(n2, l, r), r)
                assert val == pytest.approx(0.0, abs=1e-6)


def test_known_radial_functions():
    r = np.array([0.5, 1.0, 3.0])
    assert radial_R(1, 0, r) == pytest.approx(2 * np.exp(-r))
    assert radial_R(2, 0, r) == pytest.approx((1 / np.sqrt(2)) * (1 - r / 2) * np.exp(-r / 2))
    assert radial_R(2, 1, r) == pytest.approx((1 / (2 * np.sqrt(6))) * r * np.exp(-r / 2))


@pytest.mark.parametrize("n,l", [(2, 0), (3, 0), (3, 1), (4, 1), (4, 2), (5, 0)])
def test_radial_node_count_is_n_minus_l_minus_one(n, l):
    r = np.linspace(1e-3, 120, 200001)
    assert count_sign_changes(radial_R(n, l, r)) == n - l - 1


@pytest.mark.parametrize("n,l", [(1, 0), (2, 1), (3, 0), (4, 2)])
def test_exact_radial_function_solves_the_radial_equation(n, l):
    h = 1e-3
    r = np.linspace(0.5, 12, 11)
    u = lambda x: radial_u(n, l, x)  # noqa: E731
    second = (u(r + h) - 2 * u(r) + u(r - h)) / h**2
    lhs = -0.5 * second + effective_potential(r, l) * u(r)
    assert lhs == pytest.approx(hydrogen_energy(n) * u(r), abs=1e-5)


def test_finite_difference_levels_match_minus_one_over_two_n_squared():
    levels = numerical_levels(3, 4)
    assert set(levels) == {(n, l) for n in range(1, 5) for l in range(n)}
    for (n, l), e in levels.items():
        assert e == pytest.approx(-0.5 / n**2, rel=1.5e-3), (n, l)


def test_energy_does_not_depend_on_l_at_fixed_n():
    levels = numerical_levels(3, 4)
    for n in range(2, 5):
        es = [levels[(n, l)] for l in range(n)]
        assert max(es) - min(es) < 1e-3 * abs(np.mean(es))


def test_numerical_radial_states_are_normalised_and_have_the_right_nodes():
    for l in range(3):
        r, e, v = solve_radial(l)
        assert np.sum(v**2, axis=0) * 0.05 == pytest.approx(np.ones(v.shape[1]))
        for k in range(4):
            assert count_sign_changes(v[:, k][np.abs(v[:, k]) > 1e-9 * np.abs(v[:, k]).max()]) == k


def test_numerical_radial_function_agrees_with_the_exact_one():
    for n, l in [(1, 0), (2, 0), (2, 1), (3, 2), (3, 1)]:
        r, _, v = solve_radial(l)
        assert np.max(np.abs(v[:, n - l - 1] - radial_u(n, l, r))) < 2e-3, (n, l)


def test_level_counts_n_squared():
    for n in range(1, 8):
        assert degeneracy(n) == n * n == sum(2 * l + 1 for l in range(n))
        assert len(set(quantum_numbers(n))) == n * n


def test_numerical_degeneracy_is_n_squared():
    # count the numerical eigenvalues within 0.2 % of -1/(2 n^2), weighted by the 2 l + 1 values of m
    levels = numerical_levels(4, 4)
    for n in range(1, 5):
        total = sum(2 * l + 1 for (nn, l), e in levels.items() if nn == n and abs(e + 0.5 / n**2) < 2e-3 / n**2)
        assert total == n * n


def test_mean_radius_formula():
    r = np.linspace(0, 200, 400001)
    for n, l in [(1, 0), (2, 1), (3, 0), (3, 2)]:
        assert np.trapezoid(r * radial_u(n, l, r) ** 2, r) == pytest.approx(mean_radius(n, l), rel=1e-5)


def test_whole_orbitals_are_orthonormal_in_three_dimensions():
    x, wx = np.polynomial.legendre.leggauss(160)
    r = 0.5 * 80 * (x + 1)           # Gauss-Legendre on (0, 80)
    wr = 0.5 * 80 * wx
    th, ph, wa = sphere_quadrature(40, 80)
    rr = r[:, None, None]
    sets = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 1, -1), (3, 2, 0), (3, 2, 2), (3, 1, 0)]

    def psi(n, l, m):
        return radial_R(n, l, rr) * real_ylm(l, m, th[None], ph[None])

    for a in sets:
        for b in sets:
            val = np.sum(psi(*a) * psi(*b) * wa[None] * (wr[:, None, None] * rr**2))
            assert val == pytest.approx(1.0 if a == b else 0.0, abs=1e-6), (a, b)


def test_cartesian_orbital_matches_product_form():
    x, y, z = 0.8, -0.5, 1.1
    r = np.sqrt(x * x + y * y + z * z)
    th, ph = np.arccos(z / r), np.arctan2(y, x)
    for n, l, m in [(2, 1, 1), (2, 1, -1), (3, 2, 0), (3, 2, -2)]:
        assert psi_real(n, l, m, x, y, z) == pytest.approx(radial_R(n, l, r) * real_ylm(l, m, th, ph))


def test_2p_orbitals_are_the_cartesian_directions():
    x, y, z = 0.8, -0.5, 1.1
    r = np.sqrt(x * x + y * y + z * z)
    base = radial_R(2, 1, r) * np.sqrt(3 / (4 * np.pi)) / r
    assert psi_real(2, 1, 0, x, y, z) == pytest.approx(base * z)
    assert psi_real(2, 1, 1, x, y, z) == pytest.approx(base * x)
    assert psi_real(2, 1, -1, x, y, z) == pytest.approx(base * y)


def test_density_samples_follow_the_orbital():
    pts = sample_density(2, 1, 0, 20000, 30.0, seed=1)
    # for 2p_z: <r> = 5 bohr, <r^2> = n^2 (5 n^2 + 1 - 3 l (l+1)) / 2 = 30, and <cos^2 theta> = 3/5 for p_z
    r = np.linalg.norm(pts, axis=1)
    assert r.mean() == pytest.approx(5.0, rel=0.03)
    assert np.mean(r**2) == pytest.approx(30.0, rel=0.05)
    assert np.mean(pts[:, 2] ** 2) == pytest.approx(30.0 * 3 / 5, rel=0.05)


def test_polar_solution_is_a_legendre_polynomial_at_whole_l():
    # even start (y = 1 at the equator): y = P_l(cos t) / P_l(0) for even l
    # odd start (dy/dt = 1 at the equator, i.e. dP/dx = -1/y'... ): y = P_l(cos t) / (-P_l'(0)) for odd l
    h = 1e-5
    for ell, parity in [(2, "even"), (4, "even"), (3, "odd"), (1, "odd")]:
        t, y = polar_solution(ell, parity)
        p = legendre_assoc(ell, 0, np.cos(t))
        if parity == "even":
            expected = p / legendre_assoc(ell, 0, 0.0)
        else:
            expected = p / (-(legendre_assoc(ell, 0, h) - legendre_assoc(ell, 0, -h)) / (2 * h))
        assert y == pytest.approx(expected, abs=1e-4)
