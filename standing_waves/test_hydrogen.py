import numpy as np
import pytest

from hydrogen import (
    azimuth_nodes,
    latitude_nodes,
    latitude_nodes_predicted,
    count_sign_changes,
    circle_coefficients,
    circle_function,
    circle_reconstruct,
    sphere_coefficients,
    sphere_function,
    sphere_power,
    sphere_reconstruct,
    bessel_j,
    bessel_zeros,
    drum_frequency,
    drum_mode,
    drum_nodes,
    node_geometry,
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


# ---------------------------------------------------------------------------------------------------------------
# drumhead


def test_bessel_known_values_and_recurrence():
    assert bessel_j(0, 0.0) == pytest.approx(1.0)
    assert bessel_j(1, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert bessel_j(0, 1.0) == pytest.approx(0.7651976866, abs=1e-9)
    assert bessel_j(1, 2.0) == pytest.approx(0.5767248078, abs=1e-9)
    x = np.linspace(0.5, 9, 7)
    for m in (1, 2, 3):
        assert 2 * m / x * bessel_j(m, x) == pytest.approx(bessel_j(m - 1, x) + bessel_j(m + 1, x), abs=1e-10)


def test_bessel_zeros_known_values():
    assert bessel_zeros(0, 3) == pytest.approx([2.404826, 5.520078, 8.653728], abs=1e-5)
    assert bessel_zeros(1, 2) == pytest.approx([3.831706, 7.015587], abs=1e-5)
    assert bessel_zeros(2, 1) == pytest.approx([5.135622], abs=1e-5)


@pytest.mark.parametrize("m,k", [(0, 1), (0, 3), (1, 2), (2, 2), (3, 1), (-1, 2), (-3, 1), (-6, 1)])
def test_drum_mode_solves_the_wave_equation_and_vanishes_on_the_rim(m, k):
    j = drum_frequency(m, k)
    h = 1e-3
    x, y = 0.31, 0.22
    f = lambda a, b: drum_mode(m, k, np.hypot(a, b), np.arctan2(b, a))  # noqa: E731
    lap = (f(x + h, y) + f(x - h, y) + f(x, y + h) + f(x, y - h) - 4 * f(x, y)) / h**2
    assert -lap == pytest.approx(j**2 * f(x, y), abs=1e-3)
    assert abs(drum_mode(m, k, 1.0, 0.4)) < 1e-9


@pytest.mark.parametrize("m,k", [(0, 1), (0, 3), (1, 2), (2, 3), (3, 2), (4, 1)])
def test_drum_node_counts(m, k):
    radii, angles = drum_nodes(m, k)
    assert len(radii) == k - 1 and len(angles) == m
    r = np.linspace(1e-3, 1 - 1e-6, 20001)
    assert count_sign_changes(bessel_j(m, drum_frequency(m, k) * r)) == k - 1
    phi = np.linspace(0, 2 * np.pi, 4001)[:-1] + 1e-5
    ring = drum_mode(m, k, 0.5 * (1 + (radii[-1] if len(radii) else 0)) * 0.9, phi) * np.ones_like(phi)
    if m > 0:
        assert count_sign_changes(np.append(ring, ring[0])) == 2 * m  # m diameters cut the ring 2m times


@pytest.mark.parametrize("m,k", [(-1, 1), (-2, 3), (-5, 2)])
def test_drum_negative_m_is_the_sine_version_with_the_same_counts_and_frequency(m, k):
    a = -m
    radii, angles = drum_nodes(m, k)
    assert len(radii) == k - 1 and len(angles) == a
    assert drum_frequency(m, k) == pytest.approx(drum_frequency(a, k))
    for ang in angles:  # nodal diameters at the zeros of sin(a phi): phi = j pi / a
        assert abs(drum_mode(m, k, 0.4, ang)) < 1e-9
        assert abs(drum_mode(m, k, 0.4, ang + np.pi)) < 1e-9
    # the same shape turned by a quarter of the period 2 pi / a
    phi = np.linspace(0, 2 * np.pi, 37)
    assert drum_mode(m, k, 0.6, phi + np.pi / (2 * a)) == pytest.approx(drum_mode(a, k, 0.6, phi), abs=1e-9)


def test_drum_has_nodes_where_stated_and_frequencies_are_not_whole_numbers():
    for m, k in [(1, 3), (2, 2), (3, 2)]:
        radii, angles = drum_nodes(m, k)
        for rad in radii:
            assert abs(drum_mode(m, k, rad, 0.37)) < 1e-9
        for a in angles:
            assert abs(drum_mode(m, k, 0.4, a)) < 1e-9
            assert abs(drum_mode(m, k, 0.4, a + np.pi)) < 1e-9
    ratios = [drum_frequency(0, k) / drum_frequency(0, 1) for k in (1, 2, 3)]
    assert ratios[1] != pytest.approx(2.0, abs=0.1)  # a string would give 1, 2, 3


# ---------------------------------------------------------------------------------------------------------------
# node geometry of orbitals


ORBITALS = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 1, -1), (3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 2, 2),
            (3, 2, -2), (3, 2, 1), (4, 3, 2), (4, 2, -1), (4, 3, 0)]


@pytest.mark.parametrize("n,l,m", ORBITALS)
def test_orbital_has_n_minus_one_nodal_surfaces_of_three_kinds(n, l, m):
    g = node_geometry(n, l, m)
    assert len(g["spheres"]) == n - l - 1
    assert len(g["cones"]) == l - abs(m)
    assert len(g["planes"]) == abs(m)
    assert sum(len(v) for v in g.values()) == n - 1


@pytest.mark.parametrize("n,l,m", ORBITALS)
def test_orbital_vanishes_on_the_stated_nodal_surfaces(n, l, m):
    g = node_geometry(n, l, m)
    rng = np.random.default_rng(0)
    top = np.abs(psi_real(n, l, m, *rng.normal(size=(3, 4000)) * 2.0 * n)).max()
    for rad in g["spheres"]:
        d = rng.normal(size=(3, 20))
        d /= np.linalg.norm(d, axis=0)
        assert np.abs(psi_real(n, l, m, *(rad * d))).max() < 1e-5 * top
    for th in g["cones"]:
        rr, ph = rng.uniform(0.5, 3 * n, 20), rng.uniform(0, 2 * np.pi, 20)
        pts = (rr * np.sin(th) * np.cos(ph), rr * np.sin(th) * np.sin(ph), rr * np.cos(th))
        assert np.abs(psi_real(n, l, m, *pts)).max() < 1e-5 * top
    for az in g["planes"]:
        rho, z = rng.uniform(-3 * n, 3 * n, 20), rng.uniform(-3 * n, 3 * n, 20)
        assert np.abs(psi_real(n, l, m, rho * np.cos(az), rho * np.sin(az), z)).max() < 1e-5 * top


def test_named_orbitals_and_their_nodes():
    # 2p_z: one flat plane (the equator); 2p_x: one plane through the axis; 3d_z2: two cones; 3d_xy: two planes
    assert node_geometry(2, 1, 0)["cones"] == pytest.approx([np.pi / 2])
    assert node_geometry(2, 1, 1)["planes"] == pytest.approx([np.pi / 2])
    assert len(node_geometry(3, 2, 0)["cones"]) == 2
    assert len(node_geometry(3, 2, -2)["planes"]) == 2
    assert len(node_geometry(3, 0, 0)["spheres"]) == 2


@pytest.mark.parametrize("m,nr", [(0, 0), (2, 1), (-3, 2)])
def test_drum_figure_titles_use_n_l_nr(m, nr):
    import matplotlib.pyplot as plt

    from hydrogen_plots import drum_figure

    fig = drum_figure(m, nr)
    title = fig._suptitle.get_text()
    assert all("\\n" not in ax.get_title() for ax in fig.axes)  # no literal backslash-n from a raw string
    plt.close(fig)
    n, l = nr + abs(m) + 1, abs(m)
    assert f"n={n}" in title and rf"\ell={l}" in title and f"n_r={nr}" in title and f"m_\\ell={m}" in title
    assert f"N_\\mathrm{{nodes}}={n - 1}" in title and f"j_{{{l},{nr + 1}}}" in title


def test_animations_are_looping_gifs_with_one_period():
    import io

    from PIL import Image

    from hydrogen_plots import drum_gif, string_gif

    for data, frames in ((string_gif(3, n_frames=12, dpi=40), 12), (drum_gif(1, 1, n_frames=8, dpi=40), 8)):
        im = Image.open(io.BytesIO(data))
        assert im.format == "GIF" and im.n_frames == frames and im.info.get("loop") == 0
        first = np.asarray(im.convert("RGB")).astype(int)
        im.seek(frames // 2)  # half a period later the wave is upside down: the frames differ
        assert np.abs(np.asarray(im.convert("RGB")).astype(int) - first).sum() > 0


# ---------------------------------------------------------------------------------------------------------------
# Fourier series on a circle and on a sphere


def test_circle_fourier_picks_out_a_single_harmonic():
    a, b = circle_coefficients(lambda p: 0.5 + 2 * np.cos(3 * p) - 1.5 * np.sin(5 * p), 8)
    assert a[0] == pytest.approx(0.5, abs=1e-9)
    assert a[3] == pytest.approx(2.0, abs=1e-9) and b[5] == pytest.approx(-1.5, abs=1e-9)
    others = [a[m] for m in range(1, 9) if m != 3] + [b[m] for m in range(1, 9) if m != 5]
    assert np.abs(others).max() < 1e-9


def test_circle_series_converges_and_obeys_parseval():
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    a, b = circle_coefficients(circle_function, 60)
    errs = [np.mean((circle_reconstruct(a, b, phi, M) - circle_function(phi)) ** 2) for M in (1, 3, 10, 30, 60)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
    power = a[0] ** 2 + 0.5 * np.sum(a[1:] ** 2 + b[1:] ** 2)
    assert power == pytest.approx(np.mean(circle_function(phi) ** 2), rel=0.02)


def test_sphere_expansion_recovers_a_known_combination():
    th, ph, w = sphere_quadrature(90, 180)
    f = 2.0 * real_ylm(2, 1, th, ph) - 0.5 * real_ylm(3, -2, th, ph)
    got = {(l, m): float(np.sum(f * real_ylm(l, m, th, ph) * w)) for l in range(5) for m in range(-l, l + 1)}
    assert got[(2, 1)] == pytest.approx(2.0, abs=1e-9) and got[(3, -2)] == pytest.approx(-0.5, abs=1e-9)
    assert max(abs(v) for k, v in got.items() if k not in {(2, 1), (3, -2)}) < 1e-9


def test_sphere_series_converges_and_obeys_parseval():
    co = sphere_coefficients(20)
    th, ph, w = sphere_quadrature(60, 120)
    f = sphere_function(th, ph)
    errs = [np.sum((sphere_reconstruct(co, L, th, ph) - f) ** 2 * w) for L in (0, 2, 5, 10, 20)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
    assert errs[-1] < 1e-3 * errs[0]
    total = np.sum(f**2 * w)
    assert sphere_power(co).sum() == pytest.approx(total, rel=1e-3)
    assert sphere_power(co).sum() + errs[-1] == pytest.approx(total, rel=1e-3)


# ---------------------------------------------------------------------------------------------------------------
# Where the nodes of a spherical harmonic sit


LM_PAIRS = [(l, m) for l in range(0, 9) for m in range(0, l + 1)]


@pytest.mark.parametrize("l,m", LM_PAIRS)
def test_polar_sign_changes_equal_l_minus_m(l, m):
    theta = np.linspace(1e-4, np.pi - 1e-4, 4001)  # odd count would hit x=0 exactly; sign changes ignore zeros anyway
    p = legendre_assoc(l, m, np.cos(theta))
    assert count_sign_changes(p) == l - m


@pytest.mark.parametrize("l,m", LM_PAIRS)
def test_predicted_latitudes_match_zeros_of_wave(l, m):
    pred = latitude_nodes_predicted(l, m)
    assert len(pred) == l - m
    assert np.allclose(pred, latitude_nodes(l, m), atol=1e-4)
    # the real harmonic really vanishes there, at any azimuth
    for t0 in pred:
        for ph in (0.3, 1.1, 2.9):
            assert abs(real_ylm(l, m, t0, ph)) < 1e-9
        if m:
            assert abs(real_ylm(l, -m, t0, 0.7)) < 1e-9


@pytest.mark.parametrize("l,m", LM_PAIRS)
def test_latitudes_symmetric_about_equator(l, m):
    pred = latitude_nodes_predicted(l, m)
    assert np.allclose(np.sort(np.pi - pred), pred, atol=1e-9)
    on_equator = np.sum(np.abs(pred - np.pi / 2) < 1e-9)
    assert on_equator == (l - m) % 2


def test_worked_example_l4_m1():
    deg = np.degrees(latitude_nodes_predicted(4, 1))
    expected = [np.degrees(np.arccos(np.sqrt(3 / 7))), 90.0, 180 - np.degrees(np.arccos(np.sqrt(3 / 7)))]
    assert np.allclose(deg, expected, atol=1e-6)
    assert np.allclose(deg, [49.1, 90.0, 130.9], atol=0.05)
    # P_4^1 is proportional to sin(theta) (7 cos^3 theta - 3 cos theta)
    t = np.linspace(0.2, 2.9, 7)
    ratio = legendre_assoc(4, 1, np.cos(t)) / (np.sin(t) * (7 * np.cos(t) ** 3 - 3 * np.cos(t)))
    assert np.allclose(ratio, ratio[0])


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("kind", ["cos", "sin"])
def test_azimuth_planes_where_stated(m, kind):
    phi0 = azimuth_nodes(m, kind)
    assert len(phi0) == m
    assert np.allclose(np.diff(phi0), np.pi / m)  # equally spaced, pi/m apart
    assert phi0[0] == pytest.approx(np.pi / (2 * m) if kind == "cos" else 0.0)
    trig = np.cos if kind == "cos" else np.sin
    for p0 in phi0:
        for q in (p0, p0 + np.pi):
            assert abs(trig(m * q)) < 1e-12
    # and there are no other zeros of the real wave in [0, 2 pi): 2m of them
    phi = np.linspace(0, 2 * np.pi, 20001)[:-1] + 0.123
    v = trig(m * phi)
    assert count_sign_changes(np.r_[v, v[0]]) == 2 * m  # closed loop round the axis


@pytest.mark.parametrize("l,m", LM_PAIRS)
def test_total_nodal_surfaces_is_l_for_both_real_forms(l, m):
    for sign in (1, -1):
        mm = sign * m
        if m == 0 and sign == -1:
            continue
        kind = "cos" if mm >= 0 else "sin"
        assert len(azimuth_nodes(m, kind)) + len(latitude_nodes_predicted(l, m)) == l


def test_complex_harmonic_has_only_latitude_nodes():
    from hydrogen import ylm

    for l, m in [(3, 2), (4, 1), (5, 5)]:
        theta = np.linspace(0.05, np.pi - 0.05, 400)[:, None]
        phi = np.linspace(0, 2 * np.pi, 400)[None, :]
        mag = np.abs(ylm(l, m, theta, phi))
        # the magnitude is independent of phi, so it vanishes only on circles of latitude
        assert np.allclose(mag, mag[:, :1], atol=1e-12)
        # the phase winds m times round the axis
        ph = np.unwrap(np.angle(ylm(l, m, np.pi / 2 + 0.3, np.linspace(0, 2 * np.pi, 2000))))
        assert (ph[-1] - ph[0]) / (2 * np.pi) == pytest.approx(m, abs=0.01)


# ---------------------------------------------------------------------------------------------------------------
# polar and spherical coordinates


def test_polar_conversions_round_trip_and_ranges():
    from coordinates import cartesian_to_polar, polar_to_cartesian

    rng = np.random.default_rng(3)
    r, phi = rng.uniform(0.01, 3.0, 200), rng.uniform(0.0, 2 * np.pi, 200)
    x, y = polar_to_cartesian(r, phi)
    r2, phi2 = cartesian_to_polar(x, y)
    assert r2 == pytest.approx(r) and phi2 == pytest.approx(phi)
    assert np.all(r2 >= 0) and np.all((phi2 >= 0) & (phi2 < 2 * np.pi))
    # the four quarters of the plane, and the point that plain y/x cannot tell from its opposite
    assert cartesian_to_polar(1.0, 1.0)[1] == pytest.approx(np.pi / 4)
    assert cartesian_to_polar(-1.0, -1.0)[1] == pytest.approx(5 * np.pi / 4)
    assert cartesian_to_polar(0.0, -1.0)[1] == pytest.approx(3 * np.pi / 2)
    # the angle wraps: phi + 2 pi is the same point
    assert polar_to_cartesian(1.3, 0.4 + 2 * np.pi) == pytest.approx(polar_to_cartesian(1.3, 0.4))


def test_spherical_conversions_round_trip_and_ranges():
    from coordinates import cartesian_to_spherical, spherical_to_cartesian

    rng = np.random.default_rng(4)
    r, th, ph = rng.uniform(0.01, 3.0, 300), rng.uniform(0.01, np.pi - 0.01, 300), rng.uniform(0.0, 2 * np.pi, 300)
    x, y, z = spherical_to_cartesian(r, th, ph)
    r2, th2, ph2 = cartesian_to_spherical(x, y, z)
    assert r2 == pytest.approx(r) and th2 == pytest.approx(th) and ph2 == pytest.approx(ph)
    assert np.all((th2 >= 0) & (th2 <= np.pi)) and np.all((ph2 >= 0) & (ph2 < 2 * np.pi))
    assert np.sqrt(x**2 + y**2 + z**2) == pytest.approx(r)
    assert spherical_to_cartesian(2.0, 0.0, 1.0) == pytest.approx((0.0, 0.0, 2.0))  # north pole: on the z axis
    assert spherical_to_cartesian(2.0, np.pi / 2, 0.0) == pytest.approx((2.0, 0.0, 0.0))  # equator, phi = 0: on x
    assert cartesian_to_spherical(0.0, 0.0, -1.5)[1] == pytest.approx(np.pi)  # south pole


def test_coordinate_surfaces():
    from coordinates import spherical_to_cartesian

    rng = np.random.default_rng(5)
    th, ph = rng.uniform(0, np.pi, 50), rng.uniform(0, 2 * np.pi, 50)
    x, y, z = spherical_to_cartesian(1.7, th, ph)  # constant r: a sphere
    assert x**2 + y**2 + z**2 == pytest.approx(1.7**2)
    r = rng.uniform(0.1, 2, 50)
    x, y, z = spherical_to_cartesian(r, 0.9, ph)  # constant theta: a cone, z = sqrt(x^2 + y^2) / tan(theta)
    assert z == pytest.approx(np.hypot(x, y) / np.tan(0.9))
    x, y, z = spherical_to_cartesian(r, th, 0.7)  # constant phi: a half-plane through the z axis
    assert y * np.cos(0.7) - x * np.sin(0.7) == pytest.approx(0.0, abs=1e-12)
    assert np.all(x * np.cos(0.7) + y * np.sin(0.7) >= -1e-12)  # only the half on the phi side of the axis


def test_polar_grid_lines_count_and_spacing():
    from coordinates import polar_grid_lines

    radii, angles = polar_grid_lines(4, 12, 2.0)
    assert len(radii) == 4 and len(angles) == 12
    assert radii == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert np.diff(angles) == pytest.approx(np.full(11, np.pi / 6))
    assert angles[0] == 0.0 and angles[-1] < 2 * np.pi


def test_coordinate_figures_draw_expected_lines():
    import matplotlib.pyplot as plt

    from coordinates_plots import polar_grid_figure, polar_point_figure, spherical_surfaces_figure

    fig = polar_grid_figure(3, 8, slide=False)
    lines = fig.axes[0].lines
    assert len(lines) == 8 + 3  # eight rays and three circles
    plt.close(fig)
    fig = polar_point_figure(0.0, 0.0, slide=False)  # the centre: no angle to draw, no error
    plt.close(fig)
    fig = spherical_surfaces_figure(slide=False)
    assert len(fig.axes) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# a pattern on the drum as a sum of modes (drum_fourier.py)


def test_drum_modes_are_orthonormal_and_ordered_by_frequency():
    import drum_fourier as df

    modes, (_, _, w) = df.modes_on_quadrature(), df.quadrature()
    gram = np.tensordot(modes * w, modes, axes=([1, 2], [1, 2]))
    assert gram == pytest.approx(np.eye(df.M_MAX), abs=1e-9)
    freqs = [j for (_, _, _, j) in df.mode_list()]
    assert freqs == sorted(freqs)
    assert freqs[:4] == pytest.approx([2.404826, 3.831706, 3.831706, 5.135622], abs=1e-5)
    # the modes in the list agree with drum_mode(m, n_r + 1) up to the normalisation
    l, nr, m, j = df.mode_list()[6]
    r, phi = 0.37, 1.1
    ratio = df.mode_values(6, r, phi) / drum_mode(m, nr + 1, r, phi)
    assert abs(ratio) > 0
    assert df.mode_values(6, 0.62, 2.0) / drum_mode(m, nr + 1, 0.62, 2.0) == pytest.approx(ratio, rel=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 5, 9])
def test_a_pattern_that_is_one_mode_is_reproduced_by_that_mode(k):
    import drum_fourier as df

    modes = df.modes_on_quadrature()
    c = df._project(3.0 * modes[k])
    assert c[k] == pytest.approx(3.0, abs=1e-9)
    assert np.abs(np.delete(c, k)).max() < 1e-9


def test_drum_series_error_never_increases_and_parseval_holds():
    import drum_fourier as df

    _, _, w = df.quadrature()
    r, phi, _ = df.quadrature()
    for key in df.TARGETS:
        c, norm2, err = df.series(key)
        assert np.all(np.diff(err) <= 1e-12) and err[0] == 1.0 and err[-1] < 1
        f = df.target_function(key, r * np.cos(phi), r * np.sin(phi))
        assert np.sum(c**2) <= norm2 * (1 + 1e-12)  # Bessel's inequality
        # Parseval: the size of what is left is the size of the pattern minus the sizes of the coefficients
        modes = df.modes_on_quadrature()
        for M in (1, 7, 30):
            rest = f - np.tensordot(c[:M], modes[:M], axes=(0, 0))
            assert np.sum(rest**2 * w) == pytest.approx(norm2 * err[M] ** 2, rel=1e-8, abs=1e-12)


def test_drum_series_respects_symmetry():
    import drum_fourier as df

    c, _, _ = df.series("pluck")  # symmetric under y -> -y: no sine modes (m_l < 0), and none with odd l ... for x -> -x
    for ck, (l, nr, m, j) in zip(c, df.mode_list()):
        if m < 0:
            assert abs(ck) < 1e-10
        if l % 2 == 1:
            assert abs(ck) < 1e-10  # symmetric under x -> -x: cos(l phi) with odd l is odd
    c2, _, _ = df.series("poke")  # not symmetric: sine modes are present
    assert max(abs(ck) for ck, mode in zip(c2, df.mode_list()) if mode[2] < 0) > 1e-3


def test_rose_is_rings_of_petals_with_thin_gaps():
    import drum_fourier as df

    # the petals have one sign (negative, drawn red); between them are thin white gaps
    x = np.linspace(-1, 1, 401)
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    f = np.where(R <= 1, df.target_function("rose", X, Y), 0.0)  # only the disc counts
    assert f.max() <= 1e-12 and f.min() >= -1.0 - 1e-12
    gaps = np.abs(f[R < 0.9]) < 0.1 * np.abs(f).max()
    assert 0.03 < gaps.mean() < 0.25  # thin gaps: a small part of the disc, but there are some
    # seven outer petals: seven separate red regions round a circle through the outer ring
    phi = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    for rr in (0.55, 0.7):
        ring = df.target_function("rose", rr * np.cos(phi), rr * np.sin(phi))
        inside = ring < -0.5 * np.abs(ring).max()
        assert np.sum(inside & ~np.roll(inside, 1)) == 7
    # the envelope: the pattern fades to zero at the rim
    assert np.abs(f[(R > 0.995) & (R <= 1)]).max() < 0.1


def test_drum_targets_vanish_at_the_rim():
    import drum_fourier as df

    phi = np.linspace(0, 2 * np.pi, 50)
    for key in df.TARGETS:
        assert np.abs(df.target_function(key, np.cos(phi), np.sin(phi))).max() < 1e-12


def test_drum_fourier_figures_build():
    import matplotlib.pyplot as plt

    from drum_fourier_plots import drum_fourier_figure, drum_fourier_modes_figure

    for fig in (drum_fourier_figure("two", 7), drum_fourier_modes_figure(7)):
        assert len(fig.axes) >= 1
        plt.close(fig)
