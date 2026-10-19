"""Where whole numbers come from: a free wave, a wave on a ring, a wave in a box, and the hydrogen counting.

Units: hbar = m = 1 for the free wave and the ring (radius R = 1), L = 1 for the box trial wave. The
box-length figure uses an electron with lengths in nm and energies in eV.

A free plane wave exp(i(kx - omega t)) is a solution of the Schroedinger equation for every k. Whole
numbers appear only when the wave has to satisfy a condition at a seam or an edge.
"""

import numpy as np

# hbar^2 / (2 m_e) in eV nm^2, for an electron
HBAR2_OVER_2M_EV_NM2 = 0.0380998
RYDBERG_EV = 13.605693


# ---------- free wave ----------

def free_energy(k):
    """Energy k^2 / 2 of a free particle with wave number k (hbar = m = 1). Every k > 0 is allowed."""
    return 0.5 * np.asarray(k, dtype=float) ** 2


def free_wave(k, x, t=0.0, theta0=0.0):
    """Plane wave exp(i(k x - omega t + theta0)) with omega = k^2 / 2."""
    x = np.asarray(x, dtype=float)
    return np.exp(1j * (k * x - free_energy(k) * t + theta0))


# ---------- ring ----------

def ring_wave(q, phi):
    """Wave exp(i q phi) around a ring, where q = k R is the number of wavelengths around the circle."""
    return np.exp(1j * q * np.asarray(phi, dtype=float))


def seam_mismatch(q):
    """|psi(2 pi) - psi(0)| for the ring wave: 2 |sin(pi q)|. Zero exactly when q is a whole number."""
    return 2.0 * np.abs(np.sin(np.pi * np.asarray(q, dtype=float)))


def ring_energy(q):
    """Energy q^2 / 2 of the ring wave (hbar = m = R = 1). Only whole q satisfy the seam condition."""
    return 0.5 * np.asarray(q, dtype=float) ** 2


def winding_number(q):
    """How many times the phase arrow turns once around the ring: q (whole only for an allowed wave)."""
    return float(q)


# ---------- box ----------

def box_end_value(kL):
    """Value sin(kL) of the standing wave sin(kx) at the far wall x = L, as a function of kL."""
    return np.sin(np.asarray(kL, dtype=float))


def box_state(n, x, L=1.0):
    """Normalised standing wave sqrt(2/L) sin(n pi x / L) in a box 0 < x < L."""
    if n < 1:
        raise ValueError("n must be 1, 2, 3, ...")
    x = np.asarray(x, dtype=float)
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def box_energy_units(n, L=1.0):
    """Energy n^2 pi^2 / (2 L^2) in units hbar = m = 1."""
    return (np.asarray(n, dtype=float) * np.pi) ** 2 / (2.0 * L**2)


def box_energy_ev(n, L_nm):
    """Energy of an electron in a box of length L_nm (nm), in eV: n^2 pi^2 (hbar^2 / 2m) / L^2."""
    return (np.asarray(n, dtype=float) * np.pi) ** 2 * HBAR2_OVER_2M_EV_NM2 / np.asarray(L_nm, dtype=float) ** 2


# ---------- hydrogen ----------

def hydrogen_energy_ev(n):
    """Energy -13.6 eV / n^2 of the hydrogen level n = 1, 2, 3, ..."""
    return -RYDBERG_EV / np.asarray(n, dtype=float) ** 2


def hydrogen_quantum_numbers(n):
    """All (l, m) allowed for the level n: l = 0 .. n-1 and m = -l .. l."""
    return [(l, m) for l in range(n) for m in range(-l, l + 1)]


def ring_nodes_in_theta(l, m):
    """Number of nodes of the polar-angle part for the pair (l, m): l - |m|."""
    return l - abs(m)
