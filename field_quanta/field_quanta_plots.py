"""Figures for the field-quanta notebook. Each function returns a matplotlib Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oscillator"))

from field_quanta import (  # noqa: E402
    box_energy_added,
    chain_frequency,
    chain_mode_frequencies,
    chain_running_wave,
    chain_wavenumbers,
    distinct_states,
    double_slit_intensity,
    electron_field_energy_ev,
    free_electron_energy_ev,
    fringe_spacing_nm,
    gaussian_packet,
    mode_energy,
    packet_sigma_k,
    wavelength_nm,
    wavenumber_from_voltage,
    ELECTRON_REST_EV,
)
from oscillator import hermite_function  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import (  # noqa: E402
    A_COLOR,
    H_COLOR,
    INK,
    K_COLOR,
    LAMBDA_COLOR,
    OMEGA_COLOR,
    PROB_COLOR,
)

def double_slit_figure(volts, n_electrons, seed=1, y_max_mm=1.5):
    """Screen pattern for electrons of a chosen accelerating voltage: the curve, and a simulated set of clicks."""
    rng = np.random.default_rng(seed)
    y_nm = np.linspace(-y_max_mm, y_max_mm, 3000) * 1e6
    pattern = double_slit_intensity(y_nm, volts)
    probability = pattern / pattern.sum()
    clicks = rng.choice(y_nm, size=int(n_electrons), p=probability) / 1e6
    fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax.hist(clicks, bins=80, range=(-y_max_mm, y_max_mm), color=H_COLOR, alpha=0.85)
    expected_per_bin = probability * len(clicks) * len(y_nm) / 80  # the curve in units of counts per histogram bin
    ax.plot(y_nm / 1e6, expected_per_bin, color=A_COLOR, lw=1.6, label="prediction")
    ax.set_xlabel("position on the screen (mm)")
    ax.set_ylabel("number of electrons detected")
    ax.set_title(f"{int(n_electrons)} electrons, each of energy {volts:.0f} eV; fringe spacing "
                 f"{fringe_spacing_nm(volts) / 1e6:.3f} mm", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    v = np.linspace(5, 400, 200)
    ax_c.plot(v, wavelength_nm(v), color=LAMBDA_COLOR, lw=2.4)
    ax_c.plot([volts], [wavelength_nm(volts)], "o", color=A_COLOR, ms=9)
    ax_c.set_xlabel("accelerating voltage (V) = energy per electron (eV)")
    ax_c.set_ylabel(r"wavelength $\lambda=2\pi/k$ (nm)")
    ax_c.set_ylim(0, None)
    fig.tight_layout()
    return fig


def packet_figure(sigma_x):
    """A located electron is a spread of k: the packet in position (left) and in wave number (right), centred on k = 0."""
    x = np.linspace(-8, 8, 2000)
    psi = gaussian_packet(x, sigma_x, k0=4.0)
    k = np.linspace(-6, 10, 1200)
    sigma_k = packet_sigma_k(sigma_x)
    prob_k = np.exp(-((k - 4.0) ** 2) / (2 * sigma_k**2))
    prob_k /= np.trapezoid(prob_k, k)
    fig, (ax, ax_k) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax.fill_between(x, 0, np.abs(psi) ** 2, color=PROB_COLOR, alpha=0.7, label=r"probability $|\psi|^2$")
    ax.plot(x, psi.real / np.max(np.abs(psi)) * np.max(np.abs(psi) ** 2), color=H_COLOR, lw=1.4, label="real part (rescaled)")
    ax.set_xlabel("position $x$ (nm)")
    ax.set_yticks([])
    ax.set_title(rf"narrow in position: $\sigma_x={sigma_x:.2f}$ nm", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax_k.fill_between(k, 0, prob_k, color=K_COLOR, alpha=0.35)
    ax_k.plot(k, prob_k, color=K_COLOR, lw=2.2)
    ax_k.axvline(4.0, color=INK, lw=0.8, ls=":")
    ax_k.set_xlabel(r"wave number $k$ (1/nm)")
    ax_k.set_yticks([])
    ax_k.set_title(rf"spread in $k$: $\sigma_k=1/(2\sigma_x)={sigma_k:.2f}$ per nm", fontsize=10)
    fig.tight_layout()
    return fig


def chain_figure(n_masses, j, t):
    """A ring of masses carrying one running wave (left) and the dispersion relation with that mode marked (right)."""
    ks = chain_wavenumbers(n_masses)
    u = chain_running_wave(j, n_masses, t, amplitude=0.45)
    pos = np.arange(n_masses)
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax.plot(pos, u, "-", color=H_COLOR, lw=1)
    ax.plot(pos, u, "o", color=H_COLOR, ms=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, n_masses)
    ax.set_yticks([])
    ax.set_xlabel(f"mass number (the last mass is joined to the first) — {n_masses} masses")
    ax.set_title(rf"mode $j={j}$: $k={ks[j]:.2f}$, $\omega={chain_mode_frequencies(n_masses)[j]:.2f}$", fontsize=10)
    kk = np.linspace(0, 2 * np.pi, 400)
    ax_d.plot(kk, chain_frequency(kk), color=OMEGA_COLOR, lw=2.4)
    ax_d.plot(ks, chain_mode_frequencies(n_masses), "o", color=K_COLOR, ms=4)
    ax_d.plot([ks[j]], [chain_mode_frequencies(n_masses)[j]], "o", color=A_COLOR, ms=10)
    ax_d.set_xlabel(r"wave number $k$ (units of $1/a$)")
    ax_d.set_ylabel(r"frequency $\omega$ (units of $\sqrt{K/m}$)")
    ax_d.set_title(r"dispersion relation $\omega(k)=2\sqrt{K/m}\,|\sin(ka/2)|$", fontsize=10)
    fig.tight_layout()
    return fig


def ladder_figure(omega, n_quanta, n_show=6):
    """The energy ladder of one mode with n quanta (left), equal steps against a box (middle), and the mode's own wave function (right)."""
    fig, (ax, ax_s, ax_p) = plt.subplots(1, 3, figsize=(12, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for n in range(n_show + 1):
        e = mode_energy(n, omega)
        ax.plot([0.15, 0.85], [e, e], color=OMEGA_COLOR if n != n_quanta else A_COLOR, lw=3 if n == n_quanta else 2)
        ax.text(0.9, e, f"$n={n}$", va="center", fontsize=9)
    e_top = mode_energy(n_quanta, omega)
    ax.plot(0.15 + 0.12 * np.arange(n_quanta), np.full(n_quanta, e_top) + 0.12 * omega, "o", color=A_COLOR, ms=8)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_ylabel(r"energy (units $\hbar$)")
    ax.set_title(rf"{n_quanta} quanta in a mode of $\omega={omega:.2f}$: $E=(n+\frac{{1}}{{2}})\hbar\omega$", fontsize=9)
    n_axis = np.arange(0, 8)
    ax_s.plot(n_axis, np.full(len(n_axis), omega), "o-", color=OMEGA_COLOR, lw=2.2, label=r"field mode: always $\hbar\omega$")
    box = box_energy_added(np.arange(1, 9))
    ax_s.plot(np.arange(1, 9), box / box[0] * omega, "s--", color=H_COLOR, lw=1.8, label="level of a box: grows")
    ax_s.set_xlabel(r"number already present, $n$")
    ax_s.set_ylabel("energy cost of one more")
    ax_s.set_ylim(0, None)
    ax_s.legend(fontsize=8, loc="upper left")
    ax_s.set_title("cost of the next step", fontsize=9)
    a = np.linspace(-6, 6, 1200)
    ax_p.plot(a, 0.5 * a**2, color=INK, lw=1.2, alpha=0.5)
    e_n = mode_energy(n_quanta, 1.0)
    psi = hermite_function(n_quanta, a)
    ax_p.fill_between(a, e_n, e_n + 1.6 * psi**2, color=PROB_COLOR, alpha=0.7)
    ax_p.plot(a, e_n + 1.6 * psi, color=H_COLOR, lw=1.6)
    ax_p.axhline(e_n, color=A_COLOR, lw=1)
    ax_p.set_ylim(0, max(e_n + 2.5, 3.5))
    ax_p.set_xlim(-6, 6)
    ax_p.set_xlabel(r"size of the mode $u$ (units $\sqrt{\hbar/\omega}$), not a place in space")
    ax_p.set_yticks([])
    ax_p.set_title(rf"the mode's wave function $\psi_{{{n_quanta}}}(u)$ dies away at large $|u|$", fontsize=9)
    fig.tight_layout()
    return fig


def stiffness_figure(stiffness_factor, n_masses=16):
    """Changing the springs changes the dispersion relation, so every quantum of every mode changes together."""
    ks = chain_wavenumbers(n_masses)
    kk = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(kk, chain_frequency(kk), color=OMEGA_COLOR, lw=2.4, alpha=0.5, label="springs $K$")
    ax.plot(kk, chain_frequency(kk, stiffness=stiffness_factor), color=OMEGA_COLOR, lw=2.4,
            label=rf"springs ${stiffness_factor:.1f}K$")
    ax.plot(ks, chain_mode_frequencies(n_masses, stiffness=stiffness_factor), "o", color=K_COLOR, ms=5)
    ax.set_xlabel(r"wave number $k$ (units of $1/a$)")
    ax.set_ylabel(r"energy of one quantum, $\hbar\omega$")
    ax.set_ylim(0, 4.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def electron_field_figure(volts):
    """Energy of one electron against k: the rest-energy floor, and the voltage choosing a k."""
    k = np.linspace(0, 120, 600)
    kinetic = free_electron_energy_ev(k)
    fig, (ax, ax_z) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax.plot(k, kinetic, color=OMEGA_COLOR, lw=2.4)
    k0 = float(wavenumber_from_voltage(volts))
    ax.plot([k0], [volts], "o", color=A_COLOR, ms=9)
    ax.axvline(k0, color=K_COLOR, lw=1, ls=":")
    ax.axhline(volts, color=OMEGA_COLOR, lw=1, ls=":")
    ax.set_xlabel(r"wave number $k$ (1/nm)")
    ax.set_ylabel("kinetic energy of one electron (eV)")
    ax.set_title(rf"{volts:.0f} V selects $k={k0:.1f}$ per nm", fontsize=10)
    total = electron_field_energy_ev(k)
    ax_z.plot(k, total / 1e3, color=OMEGA_COLOR, lw=2.4)
    ax_z.axhline(ELECTRON_REST_EV / 1e3, color=INK, lw=1, ls="--")
    ax_z.text(60, ELECTRON_REST_EV / 1e3 + 0.002, r"floor: $m_ec^2$", fontsize=9, color=INK)
    ax_z.set_xlabel(r"wave number $k$ (1/nm)")
    ax_z.set_ylabel("energy of one electron (keV)")
    ax_z.ticklabel_format(useOffset=False)
    ax_z.set_ylim(ELECTRON_REST_EV / 1e3 - 0.003, ELECTRON_REST_EV / 1e3 + 0.07)
    ax_z.set_title("cost of creating one electron: at least the rest energy", fontsize=10)
    fig.tight_layout()
    return fig


def counting_figure(n_modes, n_quanta):
    """Number of different states of n quanta spread over m modes, for three kinds of quanta."""
    ns = np.arange(1, n_quanta + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for kind, color, label in (("labelled", H_COLOR, "each quantum has its own name"),
                               ("boson", OMEGA_COLOR, "identical, any number per mode"),
                               ("fermion", K_COLOR, "identical, at most one per mode")):
        counts = [distinct_states(n_modes, int(n), kind) for n in ns]
        ax.plot(ns, np.maximum(counts, 0.5), "o-", color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("number of quanta")
    ax.set_ylabel(f"number of different states ({n_modes} modes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
