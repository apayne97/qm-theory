"""The quantum harmonic oscillator in one dimension, in units hbar = m = omega = 1.

The potential energy is V(x) = x^2 / 2 and the energy levels are E_n = n + 1/2. The state of
energy E_n has the amplitude

    psi_n(x) exp(-i E_n t),

where psi_n is real, has n nodes (points where it crosses zero) and dies away at large |x|.
"""

import numpy as np


def energy(n):
    """Energy of the n-th state: n + 1/2, in units of hbar * omega."""
    return np.asarray(n) + 0.5


def potential(x):
    """V(x) = x^2 / 2."""
    return 0.5 * np.asarray(x, dtype=float) ** 2


def turning_point(E):
    """The distance sqrt(2 E) at which a classical particle of energy E stops (V = E)."""
    return np.sqrt(2.0 * np.asarray(E, dtype=float))


def hermite_function(n, x):
    """The n-th energy eigenfunction psi_n(x), normalised so that the integral of psi_n^2 is 1.

    Built from psi_0 = pi^(-1/4) exp(-x^2/2) and psi_1 = sqrt(2) x psi_0 with the recursion
    psi_{k+1} = sqrt(2/(k+1)) x psi_k - sqrt(k/(k+1)) psi_{k-1}, which is stable for large n.
    """
    if n < 0:
        raise ValueError("n must be 0, 1, 2, ...")
    x = np.asarray(x, dtype=float)
    previous = np.pi ** -0.25 * np.exp(-0.5 * x**2)
    if n == 0:
        return previous
    current = np.sqrt(2.0) * x * previous
    for k in range(1, n):
        previous, current = current, np.sqrt(2.0 / (k + 1)) * x * current - np.sqrt(k / (k + 1)) * previous
    return current


def stationary_state(n, x, t):
    """Amplitude psi_n(x) exp(-i E_n t) of the state of definite energy E_n."""
    return hermite_function(n, x) * np.exp(-1j * energy(n) * t)


def superposition(coefficients, x, t):
    """Amplitude of the state sum_n c_n psi_n(x) exp(-i E_n t).

    `coefficients` maps n to the complex number c_n. They are rescaled so the total probability is 1.
    """
    norm = np.sqrt(sum(abs(c) ** 2 for c in coefficients.values()))
    total = np.zeros_like(np.asarray(x, dtype=complex))
    for n, c in coefficients.items():
        total = total + (c / norm) * stationary_state(n, x, t)
    return total


def density(psi):
    """Probability density |psi|^2."""
    return np.abs(psi) ** 2


def count_nodes(values, x_limit=None, x=None):
    """Number of sign changes of a real function sampled on a grid (nodes strictly inside the data)."""
    values = np.asarray(values, dtype=float)
    if x is not None and x_limit is not None:
        values = values[np.abs(np.asarray(x)) <= x_limit]
    signs = np.sign(values[np.abs(values) > 1e-12])
    return int(np.sum(signs[1:] != signs[:-1]))


def shoot(E, x_start=-6.0, x_end=6.0, steps=6000):
    """Integrate -psi''/2 + x^2 psi/2 = E psi from x_start to x_end, starting on the decaying left tail.

    The starting values (a tiny psi and the slope of a function that dies away to the left) are the
    only condition imposed at the left end. Whether the solution also dies away at the right end
    depends on E. Returns the grid x and the solution psi, rescaled so that its largest value inside
    the classically allowed region is 1.
    """
    x = np.linspace(x_start, x_end, steps + 1)
    h = x[1] - x[0]

    def rhs(xx, y):
        return np.array([y[1], (xx**2 - 2.0 * E) * y[0]])

    kappa = np.sqrt(max(x_start**2 - 2.0 * E, 1e-12))
    y = np.array([1e-8, 1e-8 * kappa])
    out = np.empty(len(x))
    out[0] = y[0]
    for i in range(steps):
        xi = x[i]
        k1 = rhs(xi, y)
        k2 = rhs(xi + h / 2, y + h / 2 * k1)
        k3 = rhs(xi + h / 2, y + h / 2 * k2)
        k4 = rhs(xi + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y[0]
    inside = np.abs(x) <= turning_point(E)
    scale = np.max(np.abs(out[inside])) if inside.any() else 1.0
    return x, out / scale


def tail_value(E, **kwargs):
    """The value at the right end of the solution found by `shoot`: zero only when E is an allowed energy."""
    _, psi = shoot(E, **kwargs)
    return float(psi[-1])
