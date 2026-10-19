import numpy as np
import pytest

from oscillator import (
    count_nodes,
    density,
    energy,
    hermite_function,
    potential,
    shoot,
    stationary_state,
    superposition,
    tail_value,
    turning_point,
)

X = np.linspace(-9, 9, 6001)


def test_energies_are_half_integers():
    assert np.allclose(energy(np.arange(5)), [0.5, 1.5, 2.5, 3.5, 4.5])


def test_potential_and_turning_point():
    assert np.isclose(potential(2.0), 2.0)
    assert np.isclose(potential(turning_point(3.5)), 3.5)


def test_ground_state_is_a_gaussian():
    assert np.allclose(hermite_function(0, X), np.pi**-0.25 * np.exp(-X**2 / 2))


@pytest.mark.parametrize("n", range(8))
def test_states_are_normalised(n):
    assert np.isclose(np.trapezoid(hermite_function(n, X) ** 2, X), 1.0, atol=1e-8)


def test_different_states_are_orthogonal():
    for m in range(6):
        for n in range(m):
            overlap = np.trapezoid(hermite_function(m, X) * hermite_function(n, X), X)
            assert abs(overlap) < 1e-8


@pytest.mark.parametrize("n", range(8))
def test_states_solve_the_schrodinger_equation(n):
    psi = hermite_function(n, X)
    h = X[1] - X[0]
    second = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / h**2
    lhs = -0.5 * second + potential(X[1:-1]) * psi[1:-1]
    assert np.allclose(lhs, energy(n) * psi[1:-1], atol=1e-4)


@pytest.mark.parametrize("n", range(8))
def test_state_n_has_n_nodes_and_definite_parity(n):
    psi = hermite_function(n, X)
    assert count_nodes(psi, x_limit=8, x=X) == n
    assert np.allclose(psi[::-1], (-1) ** n * psi, atol=1e-12)


def test_states_die_away_at_large_distance():
    for n in range(6):
        assert abs(hermite_function(n, np.array([8.0]))[0]) < 1e-6


def test_explicit_formulas_for_the_first_states():
    gauss = np.pi**-0.25 * np.exp(-X**2 / 2)
    assert np.allclose(hermite_function(1, X), np.sqrt(2) * X * gauss)
    assert np.allclose(hermite_function(2, X), (2 * X**2 - 1) / np.sqrt(2) * gauss)


def test_negative_n_is_rejected():
    with pytest.raises(ValueError):
        hermite_function(-1, X)


def test_stationary_state_changes_phase_but_not_probability():
    n = 3
    d0 = density(stationary_state(n, X, 0.0))
    for t in [0.4, 1.7, 5.0]:
        assert np.allclose(density(stationary_state(n, X, t)), d0)
    # the real and imaginary parts do change
    assert not np.allclose(stationary_state(n, X, 0.0).real, stationary_state(n, X, 1.0).real)


def test_stationary_state_is_real_at_time_zero_and_turns_at_rate_E():
    psi0 = stationary_state(2, X, 0.0)
    assert np.allclose(psi0.imag, 0)
    t = 0.7
    assert np.allclose(stationary_state(2, X, t), psi0 * np.exp(-1j * 2.5 * t))


def test_superposition_is_normalised():
    psi = superposition({0: 1.0, 1: 2.0, 3: 1j}, X, 0.9)
    assert np.isclose(np.trapezoid(density(psi), X), 1.0, atol=1e-8)


def test_two_state_superposition_repeats_after_one_period_and_moves_in_between():
    c = {0: 1.0, 1: 1.0}
    d0 = density(superposition(c, X, 0.0))
    period = 2 * np.pi  # the energies differ by 1, so the density repeats after 2 pi
    assert np.allclose(density(superposition(c, X, period)), d0, atol=1e-10)
    assert not np.allclose(density(superposition(c, X, period / 2)), d0)
    # the centre of the probability swings from one side to the other
    mean0 = np.trapezoid(X * d0, X)
    mean_half = np.trapezoid(X * density(superposition(c, X, period / 2)), X)
    assert mean0 > 0.3 and mean_half < -0.3


@pytest.mark.parametrize("n", range(4))
def test_only_the_allowed_energies_give_a_solution_that_dies_away(n):
    E = float(energy(n))
    below, above = tail_value(E - 0.05), tail_value(E + 0.05)
    assert below * above < 0  # the end value changes sign as E passes through E_n
    assert abs(tail_value(E)) < 0.05 * min(abs(below), abs(above))


def test_a_wrong_energy_blows_up_at_the_right_end():
    x, psi = shoot(1.0)
    assert abs(psi[-1]) > 20
    x, good = shoot(1.5)
    assert abs(good[-1]) < 1
