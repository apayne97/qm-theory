import numpy as np
import pytest

from waves import (
    group_velocity,
    lorentz_phase_coefficients,
    omega_free,
    packet,
    packet_from_sum,
    phasor,
    plane_wave,
    sigma_k,
    spreading_time,
    wavelengths_in_lump,
    widths_travelled_before_spreading,
    width,
)

X = np.linspace(-40, 60, 20001)


def rms_width(x, density):
    density = density / np.trapezoid(density, x)
    mean = np.trapezoid(x * density, x)
    return mean, np.sqrt(np.trapezoid((x - mean) ** 2 * density, x))


def test_phasor_has_unit_length_and_turns_with_angle():
    theta = np.linspace(0, 10, 50)
    assert np.allclose(np.abs(phasor(theta)), 1)
    assert np.isclose(phasor(np.pi / 2), 1j)


def test_plane_wave_probability_is_uniform():
    x = np.linspace(-5, 5, 101)
    assert np.allclose(np.abs(plane_wave(x, 3.7, k=2.3, omega=1.1)) ** 2, 1)


def test_plane_wave_period_and_wavelength():
    k, omega = 3.0, 5.0
    assert np.isclose(plane_wave(0.0, 2 * np.pi / omega, k, omega), plane_wave(0.0, 0.0, k, omega))
    assert np.isclose(plane_wave(2 * np.pi / k, 0.0, k, omega), plane_wave(0.0, 0.0, k, omega))


def test_fixed_phase_moves_at_omega_over_k():
    k, omega = 3.0, 5.0
    t = 1.7
    assert np.isclose(plane_wave(omega / k * t, t, k, omega), 1)


def test_constant_energy_shift_only_changes_overall_phase():
    t, a = 2.2, 0.8
    e1, e2 = 1.0, 2.5
    p_plain = abs(np.exp(-1j * e1 * t) + np.exp(-1j * e2 * t)) ** 2
    p_shift = abs(np.exp(-1j * (e1 + a) * t) + np.exp(-1j * (e2 + a) * t)) ** 2
    assert np.isclose(p_plain, p_shift)


@pytest.mark.parametrize("v", [0.1, 0.5, 0.9])
def test_boosted_rest_phase_gives_energy_and_momentum(v):
    """E0 t = E' t' - p' x' for the Lorentz transformation t = (t' - x' v / c^2) / sqrt(1 - v^2/c^2)."""
    E0 = 1.3
    e_prime, p_prime = lorentz_phase_coefficients(E0, v)
    rng = np.random.default_rng(0)
    t_prime, x_prime = rng.normal(size=(2, 20))
    t = (t_prime - x_prime * v) / np.sqrt(1 - v**2)
    assert np.allclose(E0 * t, e_prime * t_prime - p_prime * x_prime)
    # relativistic energy-momentum relation
    assert np.isclose(e_prime**2, (p_prime) ** 2 + E0**2)


def test_low_speed_limit_is_classical():
    E0, v = 1.0, 1e-3  # E0 = M c^2 with c = 1
    e_prime, p_prime = lorentz_phase_coefficients(E0, v)
    assert np.isclose(p_prime, E0 * v, rtol=1e-5)  # p = M v
    assert np.isclose(e_prime - E0, 0.5 * E0 * v**2, rtol=1e-3)  # kinetic energy


def test_group_velocity_is_classical_velocity():
    assert np.isclose(group_velocity(5.0, hbar=1.0, m=2.0), 5.0 / 2.0)
    assert np.isclose(group_velocity(3.0, hbar=2.0, m=0.5), 2.0 * 3.0 / 0.5)


def test_packet_is_normalised():
    for t in (0.0, 3.0, 10.0):
        assert np.isclose(np.trapezoid(np.abs(packet(X, t, 4.0, 1.5)) ** 2, X), 1.0)


def test_packet_initial_width_is_sigma_x():
    _, w = rms_width(X, np.abs(packet(X, 0.0, 4.0, 1.5)) ** 2)
    assert np.isclose(w, 1.5, rtol=1e-4)


def test_packet_centroid_moves_at_classical_velocity():
    k0, hbar, m = 5.0, 1.0, 2.0
    for t in (0.0, 2.0, 6.0):
        mean, _ = rms_width(X, np.abs(packet(X, t, k0, 1.0, hbar, m)) ** 2)
        assert np.isclose(mean, hbar * k0 / m * t, atol=1e-3)


@pytest.mark.parametrize("t", [0.0, 2.0, 8.0])
def test_packet_width_follows_formula(t):
    _, w = rms_width(X, np.abs(packet(X, t, 4.0, 1.2)) ** 2)
    assert np.isclose(w, width(t, 1.2), rtol=1e-3)


def test_width_grows_by_sqrt2_at_spreading_time():
    s = 1.3
    assert np.isclose(width(spreading_time(s), s), s * np.sqrt(2))


@pytest.mark.parametrize("t", [0.0, 1.0, 5.0])
def test_closed_form_matches_explicit_sum_over_k(t):
    x = np.linspace(-10, 30, 801)
    exact = packet(x, t, 4.0, 1.0)
    summed = packet_from_sum(x, t, 4.0, 1.0)
    assert np.allclose(summed, exact, atol=1e-6)


def test_minimum_uncertainty_product():
    """sigma_x * sigma_k = 1/2, with sigma_k the rms width of |a(k)|^2 from the Fourier transform."""
    s, k0 = 1.4, 6.0
    x = np.linspace(-60, 60, 2**14, endpoint=False)
    psi = packet(x, 0.0, k0, s)
    a = np.fft.fftshift(np.fft.fft(psi))
    k = np.fft.fftshift(np.fft.fftfreq(len(x), d=x[1] - x[0])) * 2 * np.pi
    mean_k, w_k = rms_width(k, np.abs(a) ** 2)
    assert np.isclose(mean_k, k0, atol=1e-3)
    assert np.isclose(w_k, sigma_k(s), rtol=1e-3)
    assert np.isclose(s * w_k, 0.5, rtol=1e-3)


def test_wavelengths_in_lump_equals_p0_over_2_delta_p():
    hbar, k0, s = 1.0, 7.0, 0.9
    p0, delta_p = hbar * k0, hbar * sigma_k(s)
    assert np.isclose(wavelengths_in_lump(k0, s), p0 / (2 * delta_p))


def test_widths_travelled_before_spreading():
    hbar, m, k0, s = 1.0, 1.0, 10.0, 1.0
    v = hbar * k0 / m
    assert np.isclose(v * spreading_time(s, hbar, m) / s, widths_travelled_before_spreading(k0, s))


def test_omega_free_quadratic():
    assert np.isclose(omega_free(4.0, 1.0, 2.0), 4.0)


# --- two sources: interference ---------------------------------------------------

from waves import height, hole_amplitude, intensity, two_hole_distances  # noqa: E402


def test_height_is_real_part_of_plane_wave_form():
    k, omega, r, t = 2.3, 5.0, 1.7, 0.6
    h = hole_amplitude(r, k, size=1.4)
    assert np.isclose(height(h, t, omega), 1.4 * np.cos(k * r - omega * t))
    assert np.isclose(height(h, t, omega), np.real(1.4 * plane_wave(r, t, k, omega)))


def test_mean_square_height_is_half_intensity():
    omega = 2 * np.pi
    t = np.linspace(0, 1, 20001)[:-1]  # one full period
    h = 1.3 * np.exp(0.7j)
    assert np.isclose(np.mean(height(h, t, omega) ** 2), 0.5 * intensity(h), rtol=1e-6)


def test_sum_of_heights_is_height_of_sum_of_amplitudes():
    omega, t = 2 * np.pi, np.linspace(0, 2, 50)
    h1, h2 = 1.0 * np.exp(0.4j), 0.8 * np.exp(2.1j)
    assert np.allclose(height(h1, t, omega) + height(h2, t, omega), height(h1 + h2, t, omega))


def test_interference_formula():
    """I12 = I1 + I2 + 2 sqrt(I1 I2) cos(delta), delta = k (r2 - r1)."""
    k = 2 * np.pi / 0.9
    x = np.linspace(-15, 15, 301)
    r1, r2 = two_hole_distances(x, d=4.0, L=20.0)
    h1, h2 = hole_amplitude(r1, k, 1.0), hole_amplitude(r2, k, 0.6)
    delta = k * (r2 - r1)
    expected = intensity(h1) + intensity(h2) + 2 * np.sqrt(intensity(h1) * intensity(h2)) * np.cos(delta)
    assert np.allclose(intensity(h1 + h2), expected)


def test_bright_and_dark_positions_for_equal_waves():
    lam = 1.0
    k = 2 * np.pi / lam
    h1 = hole_amplitude(10.0, k)
    for n in range(4):
        assert np.isclose(intensity(h1 + hole_amplitude(10.0 + n * lam, k)), 4.0)  # whole wavelengths
        assert np.isclose(intensity(h1 + hole_amplitude(10.0 + (n + 0.5) * lam, k)), 0.0, atol=1e-12)


def test_equal_waves_give_four_cos_squared():
    k = 2 * np.pi
    x = np.linspace(-10, 10, 101)
    r1, r2 = two_hole_distances(x, 3.0, 20.0)
    total = intensity(hole_amplitude(r1, k) + hole_amplitude(r2, k))
    assert np.allclose(total, 4 * np.cos(k * (r2 - r1) / 2) ** 2)


def test_centre_of_pattern_is_bright_and_symmetric():
    k = 2 * np.pi / 1.1
    r1, r2 = two_hole_distances(np.array([-3.0, 0.0, 3.0]), 4.0, 20.0)
    total = intensity(hole_amplitude(r1, k) + hole_amplitude(r2, k))
    assert np.isclose(total[1], 4.0)
    assert np.isclose(total[0], total[2])
