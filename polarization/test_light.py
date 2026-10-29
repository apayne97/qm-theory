import numpy as np
import pytest

from light import (
    E_MINUS,
    E_PLUS,
    E_X,
    J_MATRIX,
    counter_rotating_arrows,
    ellipse_axes,
    field,
    from_helicity,
    helicity_amplitudes,
    jones,
    linear,
    mean_m,
    mean_orbital_m,
    phase_factor,
    plate_torque,
    polar_grid,
    retarder,
    ring_radius,
    rotation_2d,
    total_m,
    vortex_field,
)


def test_j_matrix_is_i_times_derivative_of_rotation():
    h = 1e-6
    numeric = 1j * (rotation_2d(h) - rotation_2d(-h)) / (2 * h)
    assert np.allclose(numeric, J_MATRIX, atol=1e-8)


def test_circular_states_are_eigenvectors_of_j_with_m_plus_minus_one():
    assert np.allclose(J_MATRIX @ E_PLUS, E_PLUS)
    assert np.allclose(J_MATRIX @ E_MINUS, -E_MINUS)


@pytest.mark.parametrize("alpha", [0.3, 1.0, np.pi, 4.0])
def test_rotation_multiplies_circular_states_by_phase(alpha):
    assert np.allclose(rotation_2d(alpha) @ E_PLUS, np.exp(-1j * alpha) * E_PLUS)
    assert np.allclose(rotation_2d(alpha) @ E_MINUS, np.exp(1j * alpha) * E_MINUS)


def test_rotation_through_2pi_returns_every_state_with_sign_plus_one():
    j = jones(0.4, 1.1)
    assert np.allclose(rotation_2d(2 * np.pi) @ j, j)
    assert np.allclose(rotation_2d(np.pi) @ j, -j)  # a half turn flips the sign: same line, not same state


def test_helicity_amplitudes_are_orthonormal_basis():
    for j in (E_X, jones(0.7, 0.3), linear(1.2)):
        a_plus, a_minus = helicity_amplitudes(j)
        assert np.allclose(from_helicity(a_plus, a_minus), j)
        assert np.isclose(abs(a_plus) ** 2 + abs(a_minus) ** 2, np.vdot(j, j).real)


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.3, 2.9])
def test_linear_light_is_equal_mix_with_relative_phase_two_alpha(alpha):
    a_plus, a_minus = helicity_amplitudes(linear(alpha))
    assert np.isclose(abs(a_plus), 1 / np.sqrt(2)) and np.isclose(abs(a_minus), 1 / np.sqrt(2))
    assert np.isclose(np.exp(1j * (np.angle(a_minus) - np.angle(a_plus))), np.exp(2j * alpha))
    assert np.isclose(mean_m(linear(alpha)), 0.0)


def test_circular_states_have_definite_m():
    assert np.isclose(mean_m(E_PLUS), 1) and np.isclose(mean_m(E_MINUS), -1)


def test_jones_quarter_turn_phase_difference_is_circular():
    assert np.isclose(mean_m(jones(np.pi / 4, np.pi / 2)), 1)   # y lags x by 90 degrees: turns counterclockwise
    assert np.isclose(mean_m(jones(np.pi / 4, -np.pi / 2)), -1)


def test_field_of_plus_state_turns_counterclockwise_once_per_period():
    wt = np.linspace(0, 2 * np.pi, 9)
    e = field(E_PLUS, wt)
    angles = np.unwrap(np.arctan2(e[1], e[0]))
    assert np.allclose(np.diff(angles), np.pi / 4)
    assert np.allclose(np.hypot(*e), 1 / np.sqrt(2))


def test_field_scalar_and_array_shapes():
    assert field(E_X, 0.3).shape == (2,)
    assert field(E_X, np.zeros(5)).shape == (2, 5)


@pytest.mark.parametrize("beta,delta", [(0.3, 0.9), (0.7, 2.0), (np.pi / 4, 0.0), (1.0, 4.5), (0.0, 0.0)])
def test_ellipse_axes_match_traced_curve(beta, delta):
    j = jones(beta, delta)
    major, minor, angle = ellipse_axes(j)
    e = field(j, np.linspace(0, 2 * np.pi, 4000, endpoint=False))
    radii = np.hypot(*e)
    assert np.isclose(radii.max(), major, atol=1e-3)
    assert np.isclose(radii.min(), minor, atol=1e-3)
    tip = e[:, radii.argmax()]
    assert np.isclose(abs(np.cos(np.arctan2(tip[1], tip[0]) - angle)), 1, atol=1e-3)


@pytest.mark.parametrize("alpha", [0.2, 0.9, 2.5])
def test_major_axis_direction_of_linear_light_is_alpha(alpha):
    major, minor, angle = ellipse_axes(linear(alpha))
    assert np.isclose(major, 1) and np.isclose(minor, 0, atol=1e-12)
    assert np.isclose(np.cos(angle - alpha) ** 2, 1)


def test_two_counter_rotating_arrows_add_to_the_field():
    j = jones(0.5, 1.2)
    for wt in (0.0, 0.7, 2.2):
        p, m = counter_rotating_arrows(j, wt)
        assert np.allclose(p + m, field(j, wt))


def test_half_wave_plate_flips_m_and_hands_two_hbar_to_the_plate():
    plate = retarder(np.pi)
    assert np.allclose(plate @ E_PLUS, E_MINUS)
    assert np.isclose(mean_m(plate @ E_PLUS), -1)
    assert np.isclose(plate_torque(E_PLUS, plate), 2)
    assert np.isclose(plate_torque(E_MINUS, plate), -2)


def test_quarter_wave_plate_turns_circular_into_linear_and_gives_one_hbar():
    plate = retarder(np.pi / 2)
    out = plate @ E_PLUS
    assert np.isclose(mean_m(out), 0)
    assert np.isclose(plate_torque(E_PLUS, plate), 1)
    assert np.isclose(ellipse_axes(out)[1], 0, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.3, 1.0, 2.2, 3.1])
def test_retarder_torque_on_circular_light_is_one_minus_cos_gamma(gamma):
    assert np.isclose(plate_torque(E_PLUS, retarder(gamma)), 1 - np.cos(gamma))


def test_plate_leaves_its_own_axes_alone():
    for gamma in (0.4, 2.0):
        assert np.isclose(plate_torque(E_X, retarder(gamma)), 0)


def test_phase_factor_returns_at_2pi_for_light_and_4pi_for_half_integer_spin():
    for m in (1, -1):
        assert np.isclose(phase_factor(m, 2 * np.pi), 1)
    for m in (0.5, -0.5):
        assert np.isclose(phase_factor(m, 2 * np.pi), -1)
        assert np.isclose(phase_factor(m, 4 * np.pi), 1)


@pytest.mark.parametrize("m_l", [-3, -1, 0, 1, 2, 5])
def test_vortex_has_definite_orbital_m(m_l):
    r, phi = polar_grid()
    psi = vortex_field(m_l, r * np.cos(phi), r * np.sin(phi))
    assert np.isclose(mean_orbital_m(psi), m_l)


def test_superposition_of_vortices_averages_the_orbital_m():
    r, phi = polar_grid()
    x, y = r * np.cos(phi), r * np.sin(phi)
    # equal weight on m_l = 1 and 3 (each normalised over the grid): the average is 2
    a, b = vortex_field(1, x, y), vortex_field(3, x, y)
    a, b = a / np.sqrt((abs(a) ** 2).sum()), b / np.sqrt((abs(b) ** 2).sum())
    assert np.isclose(mean_orbital_m(a + b), 2.0, atol=1e-6)
    assert np.isclose(mean_orbital_m(a + 0.5 * b), (1 + 0.25 * 3) / 1.25, atol=1e-6)


@pytest.mark.parametrize("m_l", [1, 2, 4])
def test_vortex_brightest_ring_radius(m_l):
    r = np.linspace(1e-3, 4, 40001)
    intensity = abs(vortex_field(m_l, r, 0 * r)) ** 2
    assert np.isclose(r[intensity.argmax()], ring_radius(m_l), atol=1e-3)


def test_vortex_is_dark_on_axis_only_for_nonzero_m():
    assert abs(vortex_field(0, 0.0, 0.0)) == pytest.approx(1.0)
    assert abs(vortex_field(2, 0.0, 0.0)) == pytest.approx(0.0)


def test_orbital_rotation_phase_matches_spin_rule():
    """Turning a twisted beam through alpha multiplies it by exp(-i m_l alpha): the same rule as for spin."""
    alpha = 0.8
    for m_l in (-2, 1, 3):
        x, y = 0.7, 0.4
        # rotated wave at (x, y) is the original wave at the point turned back by alpha
        xb, yb = np.cos(alpha) * x + np.sin(alpha) * y, -np.sin(alpha) * x + np.cos(alpha) * y
        assert np.isclose(vortex_field(m_l, xb, yb), phase_factor(m_l, alpha) * vortex_field(m_l, x, y))


def test_total_m_is_orbital_plus_spin_and_any_integer():
    assert {total_m(m_l, m_s) for m_l in range(-3, 4) for m_s in (1, -1)} == set(range(-4, 5))
    assert total_m(-1, 1) == 0
