import numpy as np
import pytest

from field_quanta import (
    HBAR2_OVER_2M_EV_NM2,
    box_energy_added,
    chain_frequency,
    chain_matrix_frequencies,
    chain_mode_frequencies,
    chain_running_wave,
    chain_wavenumbers,
    distinct_states,
    double_slit_intensity,
    electron_field_energy_ev,
    energy_added,
    fermion_occupation_allowed,
    field_state_energy,
    field_state_momentum,
    free_electron_energy_ev,
    fringe_spacing_nm,
    gaussian_packet,
    mode_energy,
    packet_sigma_k,
    packet_typical_energy_ev,
    sound_speed,
    wavelength_nm,
    wavenumber_from_voltage,
    ELECTRON_REST_EV,
)


@pytest.mark.parametrize("n_masses", [4, 7, 12])
def test_dispersion_matches_newtons_laws(n_masses):
    from_formula = np.sort(chain_mode_frequencies(n_masses, stiffness=2.5, mass=0.7))
    from_matrix = chain_matrix_frequencies(n_masses, stiffness=2.5, mass=0.7)
    assert np.allclose(from_formula, from_matrix, atol=1e-6)  # the zero mode comes out as sqrt(rounding error)


def test_long_wavelength_modes_travel_at_sound_speed():
    k = 1e-3
    assert chain_frequency(k, 3.0, 2.0) == pytest.approx(sound_speed(3.0, 2.0) * k, rel=1e-6)


def test_modes_come_in_pairs_k_and_minus_k():
    freqs = chain_mode_frequencies(10)
    assert freqs[3] == pytest.approx(freqs[7])
    assert freqs[0] == pytest.approx(0.0)


def test_running_wave_solves_equation_of_motion():
    # m x_i'' = K (x_{i+1} - 2 x_i + x_{i-1}) with wrap-around
    n, j, t, h = 12, 3, 0.4, 1e-4
    x = lambda tt: chain_running_wave(j, n, tt)
    accel = (x(t + h) - 2 * x(t) + x(t - h)) / h**2
    force = np.roll(x(t), -1) - 2 * x(t) + np.roll(x(t), 1)
    assert np.allclose(accel, force, atol=1e-5)


def test_stiffer_springs_raise_every_frequency_by_the_same_factor():
    base = chain_mode_frequencies(9, stiffness=1.0)
    stiff = chain_mode_frequencies(9, stiffness=4.0)
    assert np.allclose(stiff, 2.0 * base)


def test_every_added_quantum_costs_the_same():
    omega = 1.7
    steps = energy_added(np.arange(0, 50), omega)
    assert np.allclose(steps, omega)


def test_ladder_of_a_box_is_not_equally_spaced():
    steps = box_energy_added(np.arange(1, 6))
    assert np.all(np.diff(steps) > 0)


def test_mode_energy_includes_zero_point():
    assert mode_energy(0, 2.0) == pytest.approx(1.0)
    assert mode_energy(3, 2.0) == pytest.approx(7.0)


def test_state_energy_and_momentum_add_up():
    omegas = np.array([1.0, 2.0, 3.0])
    ks = np.array([0.5, 1.0, 1.5])
    occ = [2, 0, 1]
    assert field_state_energy(occ, omegas) == pytest.approx(5.0)
    assert field_state_momentum(occ, ks) == pytest.approx(2.5)
    # adding one quantum to mode 1 changes the energy by exactly omega_1 and the momentum by k_1
    assert field_state_energy([2, 1, 1], omegas) - 5.0 == pytest.approx(2.0)
    assert field_state_momentum([2, 1, 1], ks) - 2.5 == pytest.approx(1.0)


def test_counting_states():
    assert distinct_states(3, 2, "labelled") == 9
    assert distinct_states(3, 2, "boson") == 6
    assert distinct_states(3, 2, "fermion") == 3
    assert distinct_states(2, 3, "fermion") == 0  # three electrons cannot fit in two modes
    with pytest.raises(ValueError):
        distinct_states(3, 2, "anyon")


def test_boson_count_matches_explicit_enumeration():
    m, n = 4, 3
    count = sum(1 for a in range(n + 1) for b in range(n + 1 - a) for c in range(n + 1 - a - b))
    assert count == distinct_states(m, n, "boson")


def test_fermion_occupation_rule():
    assert fermion_occupation_allowed(0) and fermion_occupation_allowed(1)
    assert not fermion_occupation_allowed(2)


def test_voltage_sets_wavelength():
    assert wavelength_nm(100.0) == pytest.approx(0.1227, abs=1e-3)
    assert free_electron_energy_ev(wavenumber_from_voltage(250.0)) == pytest.approx(250.0)
    assert wavelength_nm(400.0) == pytest.approx(wavelength_nm(100.0) / 2.0)


def test_fringe_spacing_follows_wavelength():
    assert fringe_spacing_nm(25.0) == pytest.approx(2.0 * fringe_spacing_nm(100.0))
    # the pattern repeats with that period near the centre (envelope nearly flat): bright at y = 0 and y = spacing
    s = fringe_spacing_nm(100.0)
    assert double_slit_intensity(0.0, 100.0) == pytest.approx(1.0)
    assert double_slit_intensity(s, 100.0) > 0.9 * double_slit_intensity(0.0, 100.0) * np.sinc(100.0 * s / (wavelength_nm(100.0) * 5e8)) ** 2
    assert double_slit_intensity(0.5 * s, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_electron_field_energy_has_rest_energy_floor():
    assert electron_field_energy_ev(0.0) == pytest.approx(ELECTRON_REST_EV)
    k = 5.0  # 1/nm, slow compared with light
    assert electron_field_energy_ev(k) - ELECTRON_REST_EV == pytest.approx(free_electron_energy_ev(k), rel=1e-3)


@pytest.mark.parametrize("sigma_x", [0.5, 1.0, 3.0])
def test_packet_is_normalised_and_has_minimum_uncertainty(sigma_x):
    x = np.linspace(-12 * sigma_x, 12 * sigma_x, 8192)
    dx = x[1] - x[0]
    psi = gaussian_packet(x, sigma_x, k0=0.0)
    assert np.sum(np.abs(psi) ** 2) * dx == pytest.approx(1.0, rel=1e-6)
    phi = np.fft.fftshift(np.fft.fft(psi))
    k = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
    prob = np.abs(phi) ** 2
    prob /= prob.sum()
    sigma_k = np.sqrt(np.sum(prob * k**2))
    assert sigma_k == pytest.approx(packet_sigma_k(sigma_x), rel=1e-3)
    assert sigma_x * sigma_k == pytest.approx(0.5, rel=1e-3)


def test_narrower_packet_has_larger_energy_scale():
    assert packet_typical_energy_ev(0.5) == pytest.approx(4.0 * packet_typical_energy_ev(1.0))
    assert packet_typical_energy_ev(1.0) == pytest.approx(HBAR2_OVER_2M_EV_NM2 * 0.25)
