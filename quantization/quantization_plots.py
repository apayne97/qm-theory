"""Figures for the quantization notebook. Each function returns a matplotlib Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oscillator"))

from oscillator import hermite_function  # noqa: E402
from quantization import (  # noqa: E402
    box_end_value,
    box_energy_ev,
    box_energy_units,
    box_state,
    free_energy,
    free_wave,
    hydrogen_energy_ev,
    ring_energy,
    ring_wave,
    seam_mismatch,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import (  # noqa: E402
    A_COLOR,
    H_COLOR,
    IM_COLOR,
    INK,
    K_COLOR,
    OMEGA_COLOR,
    PHASE_COLOR,
    PROB_COLOR,
    RE_COLOR,
)

def _cross_axes(ax, xlim, ylim):
    ax.axhline(0, color="black", lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks([])


def free_wave_figure(k, theta0):
    """A free wave for a chosen k and starting phase (left) and the free-particle energy for every k (right)."""
    x = np.linspace(0, 12, 1500)
    psi = free_wave(k, x, 0.0, theta0)
    fig, (ax, ax_e) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    _cross_axes(ax, (0, 12), (-1.3, 1.75))
    ax.fill_between(x, 0, np.abs(psi) ** 2, color=PROB_COLOR, alpha=0.6, label=r"probability density $|\psi|^2$")
    ax.plot(x, psi.imag, color=IM_COLOR, lw=2.2, label="imaginary part")
    ax.plot(x, psi.real, color=RE_COLOR, lw=2.4, label="real part")
    ax.set_xlabel(r"position $x$")
    ax.set_title(rf"$k={k:.2f}$, starting phase $\theta_0={theta0:.2f}$", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    kk = np.linspace(0, 4, 300)
    ax_e.plot(kk, free_energy(kk), color=OMEGA_COLOR, lw=2.4)
    ax_e.plot([k], [free_energy(k)], "o", color=A_COLOR, ms=9)
    ax_e.set_xlabel(r"wave number $k$")
    ax_e.set_ylabel(r"energy $E=\hbar^2k^2/2m_e$ (units $\hbar=m_e=1$)")
    ax_e.set_title("every value of $k$ is allowed: a continuous curve", fontsize=10)
    ax_e.set_xlim(0, 4)
    ax_e.set_ylim(0, 8.4)
    fig.tight_layout()
    return fig


def ring_figure(q):
    """The wave exp(i q phi) around a ring: drawn on the ring, unrolled, and its energy among the allowed values."""
    phi = np.linspace(0, 2 * np.pi, 1200)
    psi = ring_wave(q, phi)
    fig = plt.figure(figsize=(13, 4.2))
    ax_c = fig.add_subplot(1, 3, 1, projection="polar")
    ax_u = fig.add_subplot(1, 3, 2)
    ax_e = fig.add_subplot(1, 3, 3)

    # on the ring: radius = 1 + 0.25 * real part of psi
    ax_c.plot(np.linspace(0, 2 * np.pi, 400), np.ones(400), color="0.75", lw=1)
    ax_c.plot(phi, 1 + 0.25 * psi.real, color=RE_COLOR, lw=2.6)
    ax_c.plot([0], [1 + 0.25 * psi.real[0]], "o", color=PHASE_COLOR, ms=9, zorder=5)
    ax_c.plot([2 * np.pi], [1 + 0.25 * psi.real[-1]], "s", color=A_COLOR, ms=8, zorder=5)
    ax_c.set_ylim(0, 1.45)
    ax_c.set_yticks([])
    ax_c.set_xticks([])
    ax_c.spines["polar"].set_visible(False)
    ax_c.set_title("real part drawn on the ring\n(circle: start, square: end)", fontsize=10)

    # unrolled
    ax_u.axhline(0, color="black", lw=1)
    ax_u.plot(phi, psi.imag, color=IM_COLOR, lw=2.2, label="imaginary part")
    ax_u.plot(phi, psi.real, color=RE_COLOR, lw=2.4, label="real part")
    ax_u.plot([0], [psi.real[0]], "o", color=PHASE_COLOR, ms=9)
    ax_u.plot([2 * np.pi], [psi.real[-1]], "s", color=A_COLOR, ms=8)
    for edge in (0, 2 * np.pi):
        ax_u.axvline(edge, color="0.6", lw=1, ls="--")
    ax_u.set_xlim(-0.4, 2 * np.pi + 0.4)
    ax_u.set_ylim(-1.3, 1.3)
    ax_u.set_xticks([0, np.pi, 2 * np.pi])
    ax_u.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax_u.set_yticks([])
    ax_u.set_xlabel(r"angle $\varphi$ around the ring (the ends are the same point)")
    fits = seam_mismatch(q) < 1e-9
    verdict = "the ends agree: allowed" if fits else rf"mismatch at the seam $=${seam_mismatch(q):.2f}"
    ax_u.set_title(rf"$q={q:.2f}$ wavelengths around:  {verdict}", fontsize=10, color=INK if fits else A_COLOR)
    ax_u.legend(loc="lower left", fontsize=8)

    # energy
    qq = np.linspace(0, 5, 300)
    ax_e.plot(qq, ring_energy(qq), color="0.8", lw=2)
    ax_e.plot(np.arange(0, 6), ring_energy(np.arange(0, 6)), "o", color=OMEGA_COLOR, ms=9, label="allowed")
    ax_e.plot([q], [ring_energy(q)], "|", color=A_COLOR, ms=26, mew=3)
    ax_e.plot([q], [ring_energy(q)], "o", color=A_COLOR, ms=5)
    ax_e.set_xlim(0, 5)
    ax_e.set_ylim(0, 13)
    ax_e.set_xlabel(r"$q$ = wavelengths around the ring")
    ax_e.set_ylabel(r"energy, units $\hbar^2/m_eR^2$")
    ax_e.set_title("grey: energy of a wave that does not fit", fontsize=10)
    ax_e.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def box_figure(kL_over_pi):
    """The wave sin(kx) in a box (left) and its value at the far wall as kL changes (right)."""
    kL = kL_over_pi * np.pi
    x = np.linspace(0, 1, 800)
    psi = np.sin(kL * x)
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax.axvspan(-0.15, 0, color="0.85")
    ax.axvspan(1, 1.15, color="0.85")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(x, psi, color=RE_COLOR, lw=2.6)
    ax.plot([0], [0], "o", color=PHASE_COLOR, ms=9, zorder=5)
    ax.plot([1], [psi[-1]], "o", color=A_COLOR, ms=9, zorder=5)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", r"$L$"])
    ax.set_yticks([])
    ax.set_xlabel(r"position $x$ (grey: the walls)")
    fits = abs(box_end_value(kL)) < 1e-9
    verdict = "reaches zero at the wall: allowed" if fits else rf"value at the wall $=${psi[-1]:.2f}, not zero"
    ax.set_title(rf"$kL/\pi={kL_over_pi:.2f}$:  {verdict}", fontsize=10, color=INK if fits else A_COLOR)
    kk = np.linspace(0, 5.6, 600)
    ax_t.axhline(0, color="0.6", lw=0.8)
    ax_t.plot(kk, box_end_value(kk * np.pi), color=INK, lw=2)
    for n in range(1, 6):
        ax_t.axvline(n, color=OMEGA_COLOR, lw=1, ls="--")
    ax_t.axvline(kL_over_pi, color=A_COLOR, lw=2)
    ax_t.set_xlim(0, 5.6)
    ax_t.set_ylim(-1.3, 1.3)
    ax_t.set_xlabel(r"$kL/\pi$")
    ax_t.set_ylabel(r"value at the far wall, $\sin kL$")
    ax_t.set_title("zero at whole numbers (orange) and only there", fontsize=10)
    fig.tight_layout()
    return fig


def box_levels_figure(L_nm):
    """Energy levels of an electron in a box against its length: they crowd together as the box grows."""
    L = np.linspace(0.4, 6, 400)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for n in range(1, 9):
        ax.plot(L, box_energy_ev(n, L), color=OMEGA_COLOR, lw=1.8, alpha=0.85)
        ax.plot([L_nm], [box_energy_ev(n, L_nm)], "o", color=A_COLOR, ms=7, zorder=5)
    ax.axvline(L_nm, color=A_COLOR, lw=1.4)
    ax.set_xlim(0.4, 6)
    ax.set_ylim(0, 10)
    ax.set_xlabel("box length $L$ (nm)")
    ax.set_ylabel("energy of an electron (eV)")
    ax.text(5.95, 9.6, r"curves: $n=1,2,\dots,8$ from the bottom", ha="right", va="top", fontsize=9, color=OMEGA_COLOR)
    e1 = float(box_energy_ev(1, L_nm))
    e2 = float(box_energy_ev(2, L_nm))
    ax.set_title(rf"$L={L_nm:.1f}$ nm:  $E_1={e1:.2f}$ eV, gap $E_2-E_1={e2 - e1:.2f}$ eV", fontsize=10)
    fig.tight_layout()
    return fig


def comparison_figure(n_states=3):
    """The lowest states of a ring, a box and an oscillator, each drawn on its own energy level."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    ax = axes[0]
    phi = np.linspace(0, 2 * np.pi, 600)
    for m in range(n_states):
        E = float(ring_energy(m))
        ax.axhline(E, color=OMEGA_COLOR, lw=1.1, ls="--")
        ax.plot(phi, E + 0.45 * ring_wave(m, phi).real, color=RE_COLOR, lw=2.2)
        ax.text(2 * np.pi + 0.15, E, rf"$q={m}$", va="center", fontsize=9, color=OMEGA_COLOR)
    ax.set_xlim(0, 2 * np.pi + 1.4)
    ax.set_xticks([0, 2 * np.pi])
    ax.set_xticklabels(["0", r"$2\pi$"])
    ax.set_xlabel(r"angle $\varphi$")
    ax.set_ylabel(r"energy, units $\hbar^2/m_eR^2$")
    ax.set_title("ring: ends must agree", fontsize=10)

    ax = axes[1]
    x = np.linspace(0, 1, 600)
    for n in range(1, n_states + 1):
        E = float(box_energy_units(n))
        ax.axhline(E, color=OMEGA_COLOR, lw=1.1, ls="--")
        ax.plot(x, E + 2.5 * box_state(n, x), color=RE_COLOR, lw=2.2)
        ax.text(1.04, E, rf"$n={n}$", va="center", fontsize=9, color=OMEGA_COLOR)
    ax.axvspan(-0.1, 0, color="0.85")
    ax.axvspan(1, 1.1, color="0.85")
    ax.set_xlim(-0.1, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", r"$L$"])
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"energy, units $\hbar^2/m_eL^2$")
    ax.set_title("box: zero at both walls", fontsize=10)

    ax = axes[2]
    x = np.linspace(-5, 5, 800)
    ax.fill_between(x, 0.5 * x**2, 7, color="0.95")
    ax.plot(x, 0.5 * x**2, color=INK, lw=1.6)
    for n in range(n_states):
        E = n + 0.5
        ax.axhline(E, color=OMEGA_COLOR, lw=1.1, ls="--")
        ax.plot(x, E + 0.8 * hermite_function(n, x), color=RE_COLOR, lw=2.2)
        ax.text(5.1, E, rf"$n={n}$", va="center", fontsize=9, color=OMEGA_COLOR)
    ax.set_xlim(-5, 6.6)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"energy, units $\hbar\omega$")
    ax.set_title("oscillator: dies away at both ends", fontsize=10)

    for a in axes[:2]:
        a.set_ylim(a.get_ylim()[0], a.get_ylim()[1])
    axes[0].set_ylim(-0.4, 3.0)
    axes[1].set_ylim(0, 56)
    fig.tight_layout()
    return fig


def hydrogen_levels_figure(n_max=4):
    """Hydrogen levels -13.6 eV / n^2, split by l, with the number of m values (2l + 1) of each."""
    letters = "spdfg"
    fig, ax = plt.subplots(figsize=(9, 5.4))
    energies = [float(hydrogen_energy_ev(n)) for n in range(1, n_max + 1)]
    for n, E in enumerate(energies, start=1):
        ax.plot([-0.5, n_max - 0.5], [E, E], color=OMEGA_COLOR, lw=0.8, ls=":")
        for l in range(n):
            ax.plot([l - 0.32, l + 0.32], [E, E], color=OMEGA_COLOR, lw=4, solid_capstyle="butt")
            ax.text(l + 0.38, E, rf"${2 * l + 1}$", ha="left", va="center", fontsize=9, color=INK)
    ax.axhline(0, color="black", lw=1)
    ax.text(n_max - 0.55, 0.3, "above 0: a free electron, any energy", fontsize=9, color=INK, ha="right")
    ax.set_yticks(energies)
    ax.set_yticklabels([rf"$n={n}$:  {E:.2f}" for n, E in enumerate(energies, start=1)], fontsize=9)
    ax.set_xticks(range(n_max))
    ax.set_xticklabels([rf"$\ell={l}$ ({letters[l]})" for l in range(n_max)])
    ax.set_xlim(-0.5, n_max - 0.5)
    ax.set_ylim(-15, 2.4)
    ax.set_ylabel("energy (eV)")
    ax.set_title(r"number beside each level: how many values of $m$ ($2\ell+1$)", fontsize=10)
    fig.tight_layout()
    return fig
