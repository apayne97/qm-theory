"""Plane waves and wave packets for a free particle in one dimension.

A state of definite energy E and momentum p has the amplitude

    exp(i (k x - omega t)),   omega = E / hbar,   k = p / hbar.

A wave packet is a superposition of such waves with nearby k. Everything here
uses the nonrelativistic free-particle relation omega(k) = hbar k^2 / (2 m).
"""

import numpy as np


def phasor(theta):
    """The unit-length complex number exp(i theta): an arrow at angle theta."""
    return np.exp(1j * np.asarray(theta))


def plane_wave(x, t, k, omega):
    """Amplitude exp(i (k x - omega t)) of a state of definite k and omega."""
    return np.exp(1j * (k * np.asarray(x) - omega * t))


def omega_free(k, hbar=1.0, m=1.0):
    """Frequency of a free particle with wave number k: hbar k^2 / (2 m)."""
    return hbar * np.asarray(k) ** 2 / (2 * m)


def group_velocity(k0, hbar=1.0, m=1.0, dk=1e-6):
    """d(omega)/dk at k0, by central difference."""
    return (omega_free(k0 + dk, hbar, m) - omega_free(k0 - dk, hbar, m)) / (2 * dk)


def lorentz_phase_coefficients(E0, v, c=1.0):
    """Energy and momentum seen in a frame where the particle moves at speed v.

    The rest-frame phase E0 * t equals E' * t' - p' * x' with
    E' = E0 / sqrt(1 - v^2/c^2) and p' = E' v / c^2 (frame moving along -x).
    """
    gamma = 1.0 / np.sqrt(1.0 - (v / c) ** 2)
    e_prime = E0 * gamma
    return e_prime, e_prime * v / c**2


def packet(x, t, k0, sigma_x, hbar=1.0, m=1.0):
    """Gaussian wave packet: the sum over k of Gaussian amplitudes times plane waves.

    At t = 0 the amplitude is (2 pi sigma^2)^(-1/4) exp(-x^2 / 4 sigma^2) exp(i k0 x),
    so |psi|^2 is a Gaussian with root-mean-square width sigma_x. It then evolves
    under omega(k) = hbar k^2 / 2m (exact, in the limit of infinitely many k values).
    """
    x = np.asarray(x, dtype=float)
    tau = hbar * t / (2 * m * sigma_x**2)
    d = 1 + 1j * tau
    v = hbar * k0 / m
    envelope = np.exp(-((x - v * t) ** 2) / (4 * sigma_x**2 * d)) / np.sqrt(d)
    phase = np.exp(1j * (k0 * x - hbar * k0**2 * t / (2 * m)))
    return (2 * np.pi * sigma_x**2) ** -0.25 * envelope * phase


def packet_from_sum(x, t, k0, sigma_x, n_components=401, half_width=8.0, hbar=1.0, m=1.0):
    """The same packet built explicitly as a Riemann sum over wave numbers.

    The amplitude of each k is a Gaussian of width sigma_k = 1/(2 sigma_x) about k0,
    all with the same phase at t = 0. Used to check `packet`.
    """
    sigma_k = 1 / (2 * sigma_x)
    ks = k0 + np.linspace(-half_width, half_width, n_components) * sigma_k
    dk = ks[1] - ks[0]
    a = (2 * sigma_x**2 / np.pi) ** 0.25 * np.exp(-((ks - k0) ** 2) * sigma_x**2)
    phases = np.exp(1j * (ks[:, None] * np.asarray(x)[None, :] - omega_free(ks, hbar, m)[:, None] * t))
    return (a[:, None] * phases).sum(axis=0) * dk / np.sqrt(2 * np.pi)


def width(t, sigma_x, hbar=1.0, m=1.0):
    """Root-mean-square width of |psi|^2 at time t: sigma sqrt(1 + (hbar t / 2 m sigma^2)^2)."""
    return sigma_x * np.sqrt(1 + (hbar * t / (2 * m * sigma_x**2)) ** 2)


def spreading_time(sigma_x, hbar=1.0, m=1.0):
    """Time 2 m sigma^2 / hbar after which the width has grown by a factor sqrt(2)."""
    return 2 * m * sigma_x**2 / hbar


def sigma_k(sigma_x):
    """Width in wave number of the minimum-uncertainty packet: 1 / (2 sigma_x)."""
    return 1 / (2 * sigma_x)


def wavelengths_in_lump(k0, sigma_x):
    """k0 * sigma_x, which also equals p0 / (2 Delta p)."""
    return k0 * sigma_x


def widths_travelled_before_spreading(k0, sigma_x):
    """Distance v t_s measured in widths: 2 k0 sigma_x."""
    return 2 * k0 * sigma_x


# ---------------------------------------------------------------------------
# Two sources: adding waves
# ---------------------------------------------------------------------------


def hole_amplitude(r, k, size=1.0):
    """Complex amplitude size * exp(i k r) at distance r from a source of waves with wave number k."""
    return size * np.exp(1j * k * np.asarray(r, dtype=float))


def height(h, t, omega):
    """Instantaneous height Re[h exp(-i omega t)] of a wave with complex amplitude h."""
    return np.real(h * np.exp(-1j * omega * np.asarray(t, dtype=float)))


def two_hole_distances(x, d, L):
    """Distances from two holes at (0, +d/2) and (0, -d/2) to a detector at (L, x)."""
    x = np.asarray(x, dtype=float)
    return np.hypot(L, x - d / 2), np.hypot(L, x + d / 2)


def intensity(h):
    """Intensity |h|^2 (the constant of proportionality is dropped)."""
    return np.abs(h) ** 2


# ---------------------------------------------------------------------------
# The particle in a box as a matrix
# ---------------------------------------------------------------------------


def box_grid(n_points, length=1.0):
    """The n_points interior grid points x_j = j dx (j = 1 ... n_points) of a box 0 < x < length, and the spacing
    dx = length / (n_points + 1). The walls x = 0 and x = length carry psi = 0 and are not grid points."""
    dx = length / (n_points + 1)
    return dx * np.arange(1, n_points + 1), dx


def box_hamiltonian(n_points, length=1.0, hbar=1.0, mass=1.0):
    """The Hamiltonian of the particle in a box (V = 0) on the grid of box_grid, as an n_points x n_points matrix.

    psi'' at x_j is replaced by (psi_{j+1} - 2 psi_j + psi_{j-1}) / dx^2, with psi_0 = psi_{n+1} = 0 at the walls,
    so H = hbar^2 / (2 m dx^2) times the matrix with 2 on the diagonal and -1 on the two diagonals next to it."""
    _, dx = box_grid(n_points, length)
    off = -np.ones(n_points - 1)
    return hbar**2 / (2 * mass * dx**2) * (2 * np.eye(n_points) + np.diag(off, 1) + np.diag(off, -1))


def box_energy(n, length=1.0, hbar=1.0, mass=1.0):
    """The exact energy n^2 pi^2 hbar^2 / (2 m L^2) of the n-th state of the box."""
    return n**2 * np.pi**2 * hbar**2 / (2 * mass * length**2)
