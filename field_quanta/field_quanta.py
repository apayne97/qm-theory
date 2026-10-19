"""Why every quantum of a field is the same: a chain of masses, the energy ladder of one mode, and counting.

Units: hbar = 1 for the chain and the ladder (frequencies in units of sqrt(K/m)). The double-slit and
free-electron functions use eV and nm for an electron.

The toy field is a ring of N equal masses m joined by identical springs K, spacing a. ("Ring" is the
periodic bookkeeping that removes the ends: the last mass is joined to the first.) Its normal modes are
running waves with k_j = 2 pi j / (N a) and frequency omega(k) = 2 sqrt(K/m) |sin(k a / 2)|. Each mode
is one harmonic oscillator, so its energy ladder is (n + 1/2) omega with equal steps.
"""

import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oscillator"))

from oscillator import energy as oscillator_energy  # noqa: E402

# hbar^2 / (2 m_e) in eV nm^2, for an electron
HBAR2_OVER_2M_EV_NM2 = 0.0380998
# hbar c in eV nm and the electron rest energy m_e c^2 in eV
HBARC_EV_NM = 197.3269804
ELECTRON_REST_EV = 510998.95


# ---------- the chain of masses (a toy field) ----------

def chain_wavenumbers(n_masses, spacing=1.0):
    """Wave numbers k_j = 2 pi j / (N a) of the N modes of a ring of N masses, j = 0 .. N-1."""
    return 2.0 * np.pi * np.arange(n_masses) / (n_masses * spacing)


def chain_frequency(k, stiffness=1.0, mass=1.0, spacing=1.0):
    """Dispersion relation omega(k) = 2 sqrt(K/m) |sin(k a / 2)| of the chain."""
    k = np.asarray(k, dtype=float)
    return 2.0 * np.sqrt(stiffness / mass) * np.abs(np.sin(0.5 * k * spacing))


def sound_speed(stiffness=1.0, mass=1.0, spacing=1.0):
    """Speed v = a sqrt(K/m) of long waves: omega = v |k| when k a << 1."""
    return spacing * np.sqrt(stiffness / mass)


def chain_mode_frequencies(n_masses, stiffness=1.0, mass=1.0):
    """Frequency of mode j = 0 .. N-1 from the dispersion relation."""
    return chain_frequency(chain_wavenumbers(n_masses), stiffness, mass)


def chain_matrix_frequencies(n_masses, stiffness=1.0, mass=1.0):
    """The same frequencies found directly from Newton's laws, as a check on `chain_frequency`.

    The equation m x_i'' = K (x_{i+1} - 2 x_i + x_{i-1}) (ring: indices wrap round) is x'' = -D x with D the
    matrix below; its eigenvalues are omega^2. Returned sorted.
    """
    d = np.zeros((n_masses, n_masses))
    for i in range(n_masses):
        d[i, i] = 2.0
        d[i, (i + 1) % n_masses] -= 1.0
        d[i, (i - 1) % n_masses] -= 1.0
    eigenvalues = np.linalg.eigvalsh(stiffness / mass * d)
    return np.sort(np.sqrt(np.clip(eigenvalues, 0.0, None)))


def chain_running_wave(j, n_masses, t=0.0, stiffness=1.0, mass=1.0, amplitude=1.0):
    """Displacement of each mass in mode j at time t: amplitude * cos(k x_i - omega t)."""
    k = chain_wavenumbers(n_masses)[j % n_masses] if j >= 0 else -chain_wavenumbers(n_masses)[-j % n_masses]
    omega = chain_frequency(k, stiffness, mass)
    return amplitude * np.cos(k * np.arange(n_masses) - omega * t)


# ---------- the energy ladder of one mode ----------

def mode_energy(n, omega):
    """Energy (n + 1/2) omega of a mode with n quanta (hbar = 1)."""
    return oscillator_energy(n) * omega


def energy_added(n, omega):
    """Energy needed to add one quantum to a mode that holds n: always omega."""
    return mode_energy(np.asarray(n) + 1, omega) - mode_energy(n, omega)


def box_energy_added(n):
    """For contrast: a particle in a box (L = 1, hbar = m = 1) has E_n = n^2 pi^2 / 2; the step n -> n+1 grows."""
    n = np.asarray(n, dtype=float)
    return 0.5 * np.pi**2 * ((n + 1) ** 2 - n**2)


def field_state_energy(occupations, omegas):
    """Energy above the empty field of the state with `occupations[k]` quanta in mode k: sum n_k omega_k."""
    return float(np.sum(np.asarray(occupations) * np.asarray(omegas)))


def field_state_momentum(occupations, wavenumbers):
    """Total momentum (hbar = 1) of running-wave quanta: sum n_k k."""
    return float(np.sum(np.asarray(occupations) * np.asarray(wavenumbers)))


# ---------- counting ----------

def distinct_states(n_modes, n_quanta, kind):
    """Number of different states of `n_quanta` quanta spread over `n_modes` modes.

    kind = "labelled":  each quantum carries its own name (so which one is where matters): M^N
    kind = "boson":     only how many are in each mode matters: C(M + N - 1, N)
    kind = "fermion":   as for bosons, with at most one per mode: C(M, N)
    """
    if kind == "labelled":
        return n_modes**n_quanta
    if kind == "boson":
        return comb(n_modes + n_quanta - 1, n_quanta)
    if kind == "fermion":
        return comb(n_modes, n_quanta)
    raise ValueError("kind must be 'labelled', 'boson' or 'fermion'")


def fermion_occupation_allowed(n):
    """An electron-field mode holds 0 or 1 electron."""
    return int(n) in (0, 1)


# ---------- a free electron and the electron field ----------

def free_electron_energy_ev(k_per_nm):
    """Kinetic energy hbar^2 k^2 / 2 m_e in eV, for k in 1/nm (every k allowed in vacuum)."""
    return HBAR2_OVER_2M_EV_NM2 * np.asarray(k_per_nm, dtype=float) ** 2


def electron_field_energy_ev(k_per_nm):
    """Energy of one electron with wave number k: sqrt((hbar c k)^2 + (m_e c^2)^2), in eV."""
    return np.hypot(HBARC_EV_NM * np.asarray(k_per_nm, dtype=float), ELECTRON_REST_EV)


def wavenumber_from_voltage(volts):
    """Wave number (1/nm) of an electron that gained kinetic energy e V: hbar^2 k^2 / 2 m_e = e V."""
    return np.sqrt(np.asarray(volts, dtype=float) / HBAR2_OVER_2M_EV_NM2)


def wavelength_nm(volts):
    """de Broglie wavelength 2 pi / k in nm for an electron accelerated through `volts`."""
    return 2.0 * np.pi / wavenumber_from_voltage(volts)


def double_slit_intensity(y_nm, volts, slit_separation_nm=500.0, slit_width_nm=100.0, screen_distance_nm=5e8):
    """Relative intensity on the screen at height y: two-slit fringes under the one-slit envelope (small angles)."""
    lam = wavelength_nm(volts)
    y = np.asarray(y_nm, dtype=float)
    fringe = np.cos(np.pi * slit_separation_nm * y / (lam * screen_distance_nm)) ** 2
    envelope = np.sinc(slit_width_nm * y / (lam * screen_distance_nm)) ** 2  # numpy sinc(x) = sin(pi x)/(pi x)
    return fringe * envelope


def fringe_spacing_nm(volts, slit_separation_nm=500.0, screen_distance_nm=5e8):
    """Distance between neighbouring bright fringes: lambda L / d."""
    return wavelength_nm(volts) * screen_distance_nm / slit_separation_nm


# ---------- a located electron is a packet ----------

def gaussian_packet(x, sigma_x, k0=0.0):
    """Amplitude exp(-x^2 / 4 sigma_x^2) exp(i k0 x), normalised so that |psi|^2 integrates to 1."""
    x = np.asarray(x, dtype=float)
    norm = (2.0 * np.pi * sigma_x**2) ** -0.25
    return norm * np.exp(-(x**2) / (4.0 * sigma_x**2)) * np.exp(1j * k0 * x)


def packet_sigma_k(sigma_x):
    """Spread in k of the packet above: sigma_k = 1 / (2 sigma_x), so sigma_x sigma_k = 1/2."""
    return 0.5 / sigma_x


def packet_typical_energy_ev(sigma_x_nm):
    """hbar^2 sigma_k^2 / 2 m_e in eV: the energy scale of a packet of width sigma_x (centred on k = 0)."""
    return HBAR2_OVER_2M_EV_NM2 * packet_sigma_k(sigma_x_nm) ** 2
