import numpy as np
import pytest

from quantization import (
    box_end_value,
    box_energy_ev,
    box_energy_units,
    box_state,
    free_energy,
    free_wave,
    hydrogen_energy_ev,
    hydrogen_quantum_numbers,
    ring_energy,
    ring_nodes_in_theta,
    ring_wave,
    seam_mismatch,
)


def second_derivative(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


@pytest.mark.parametrize("k", [0.3, 1.0, 2.7, 3.14159])
def test_free_wave_solves_schroedinger_for_any_k(k):
    # i d(psi)/dt = -1/2 psi''  for every k: nothing selects k
    x0, t0, h = 0.7, 0.4, 1e-5
    dt = (free_wave(k, x0, t0 + h) - free_wave(k, x0, t0 - h)) / (2 * h)
    lhs = 1j * dt
    rhs = -0.5 * second_derivative(lambda x: free_wave(k, x, t0), x0, h=1e-3)
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_free_wave_overall_phase_does_not_change_density():
    x = np.linspace(0, 10, 50)
    a = free_wave(1.7, x, 0.3, theta0=0.0)
    b = free_wave(1.7, x, 0.3, theta0=2.1)
    assert np.allclose(abs(a) ** 2, abs(b) ** 2)
    assert not np.allclose(a, b)


def test_free_energy_is_continuous():
    k = np.linspace(0.1, 3, 5)
    assert np.allclose(free_energy(k), 0.5 * k**2)


def test_ring_seam_matches_only_for_whole_q():
    for q in [0, 1, 2, -3, 5]:
        assert seam_mismatch(q) == pytest.approx(0, abs=1e-12)
        assert ring_wave(q, 2 * np.pi) == pytest.approx(ring_wave(q, 0.0))
    for q in [0.5, 1.3, 2.9]:
        assert seam_mismatch(q) > 0.1
        assert abs(ring_wave(q, 2 * np.pi) - ring_wave(q, 0.0)) == pytest.approx(seam_mismatch(q))


def test_ring_mismatch_largest_at_half_integers():
    q = np.linspace(0, 4, 4001)
    assert seam_mismatch(q).max() == pytest.approx(2.0, abs=1e-6)
    assert q[np.argmax(seam_mismatch(q))] % 1 == pytest.approx(0.5, abs=1e-3)


def test_ring_wave_is_eigenfunction_of_kinetic_energy():
    # -1/2 d^2/dphi^2 e^{i q phi} = q^2/2 e^{i q phi}
    for q in [1, 2, 3]:
        phi0 = 0.9
        lhs = -0.5 * second_derivative(lambda p: ring_wave(q, p), phi0, h=1e-3)
        assert lhs == pytest.approx(ring_energy(q) * ring_wave(q, phi0), rel=1e-5)


def test_ring_plus_minus_q_degenerate():
    assert ring_energy(2) == ring_energy(-2)


def test_box_end_value_zero_at_n_pi():
    for n in range(1, 8):
        assert box_end_value(n * np.pi) == pytest.approx(0, abs=1e-12)
    assert abs(box_end_value(1.5 * np.pi)) == pytest.approx(1.0)


def test_box_states_normalised_and_orthogonal():
    L = 2.3
    x = np.linspace(0, L, 20001)
    for n in (1, 2, 3):
        assert np.trapezoid(box_state(n, x, L) ** 2, x) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(box_state(1, x, L) * box_state(2, x, L), x) == pytest.approx(0, abs=1e-6)


def test_box_state_obeys_schroedinger_with_stated_energy():
    L, n, x0 = 1.7, 3, 0.55
    lhs = -0.5 * second_derivative(lambda x: box_state(n, x, L), x0, h=1e-3)
    assert lhs == pytest.approx(box_energy_units(n, L) * box_state(n, x0, L), rel=1e-5)


def test_box_energies_scale_as_n_squared_and_inverse_L_squared():
    assert box_energy_units(3) / box_energy_units(1) == pytest.approx(9)
    assert box_energy_units(1, 2.0) / box_energy_units(1, 1.0) == pytest.approx(0.25)


def test_electron_in_1nm_box_lowest_level():
    assert box_energy_ev(1, 1.0) == pytest.approx(0.376, abs=1e-3)


def test_hydrogen_levels():
    assert hydrogen_energy_ev(1) == pytest.approx(-13.6, abs=0.01)
    assert hydrogen_energy_ev(2) == pytest.approx(-3.40, abs=0.01)


def test_hydrogen_degeneracy_is_n_squared():
    for n in range(1, 7):
        assert len(hydrogen_quantum_numbers(n)) == n**2


def test_hydrogen_nodes_in_theta():
    assert ring_nodes_in_theta(2, 0) == 2
    assert ring_nodes_in_theta(2, 2) == 0
    assert all(ring_nodes_in_theta(l, m) >= 0 for l, m in hydrogen_quantum_numbers(5))
