"""Figures for the harmonic-oscillator notebook. Each function returns a matplotlib Figure."""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oscillator import (
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import (  # noqa: E402
    A_COLOR,
    H_COLOR,
    IM_COLOR,
    INK,
    K_COLOR,
    OMEGA_COLOR,
    PERIOD_COLOR,
    PHASE_COLOR,
    PROB_COLOR,
    RE_COLOR,
)

X = np.linspace(-6, 6, 1500)


def _cross_axes(ax, xlim, ylim):
    """Axes drawn as a black cross through the origin, without ticks."""
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])


def levels_figure(n_max=5, show_states=True):
    """The potential, the energy levels E_n = n + 1/2, and each state drawn on its own level."""
    x = np.linspace(-6, 6, 1500)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    top = float(energy(n_max)) + 1.2
    ax.fill_between(x, potential(x), top + 1, color="0.93", zorder=0)
    ax.plot(x, potential(x), color=INK, lw=2.2, zorder=3)
    for n in range(n_max + 1):
        E = float(energy(n))
        ax.plot([-turning_point(E), turning_point(E)], [E, E], color=OMEGA_COLOR, lw=1.4, ls="--", zorder=2)
        ax.text(6.15, E, rf"$E_{n}={E:g}$", va="center", color=OMEGA_COLOR, fontsize=10)
        if show_states:
            psi = hermite_function(n, x)
            ax.plot(x, E + 0.75 * psi, color=RE_COLOR, lw=2, zorder=4)
            ax.fill_between(x, E, E + 0.75 * psi, color=RE_COLOR, alpha=0.12, zorder=1)
    ax.set_xlim(-6, 7.7)
    ax.set_ylim(-0.1, top)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"energy, in units of $\hbar\omega$")
    ax.set_title(r"potential $V(x)=\frac{1}{2}x^2$ (dark), allowed energies (orange), states $\psi_n$ (blue)", fontsize=10)
    fig.tight_layout()
    return fig


def state_grid_figure(ns=(0, 1, 2, 3, 4, 5), t=0.0, columns=3):
    """Real part (blue) and imaginary part (green) of the state of definite energy, one small panel per n."""
    rows = int(np.ceil(len(ns) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.5 * rows), squeeze=False)
    x = np.linspace(-5.5, 5.5, 1200)
    for ax, n in zip(axes.flat, ns):
        psi = stationary_state(n, x, t)
        _cross_axes(ax, (-5.5, 5.5), (-1.0, 1.0))
        ax.plot(x, psi.imag, color=IM_COLOR, lw=2.4)
        ax.plot(x, psi.real, color=RE_COLOR, lw=2.4)
        ax.set_title(rf"$n={n}$", fontsize=11)
    for ax in list(axes.flat)[len(ns):]:
        ax.axis("off")
    fig.suptitle(rf"real part (blue) and imaginary part (green) at $t={t:.2f}$", fontsize=10)
    fig.tight_layout()
    return fig


def superposition_figure(coefficients, t):
    """Real and imaginary parts of a superposition (top) and its probability density (bottom)."""
    x = np.linspace(-6, 6, 1500)
    psi = superposition(coefficients, x, t)
    fig, (ax_psi, ax_rho) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))
    _cross_axes(ax_psi, (-6, 6), (-0.8, 0.8))
    ax_psi.plot(x, psi.imag, color=IM_COLOR, lw=2.2, label="imaginary part")
    ax_psi.plot(x, psi.real, color=RE_COLOR, lw=2.2, label="real part")
    ax_psi.set_yticks([-0.5, 0, 0.5])
    ax_psi.legend(loc="upper right", fontsize=9)
    ax_psi.set_ylabel(r"$\psi(x,t)$")
    rho = density(psi)
    ax_rho.fill_between(x, 0, rho, color=PROB_COLOR, alpha=0.9)
    ax_rho.plot(x, rho, color=H_COLOR, lw=2)
    mean = np.trapezoid(x * rho, x)
    ax_rho.axvline(mean, color=A_COLOR, lw=1.4, ls="--")
    ax_rho.text(mean, 0.97 * 0.9, r" average position", color=A_COLOR, fontsize=9, va="top")
    ax_rho.set_ylim(0, 0.9)
    ax_rho.set_xlim(-6, 6)
    ax_rho.set_xticks(np.arange(-6, 7, 2))
    ax_rho.set_xlabel(r"position $x$")
    ax_rho.set_ylabel(r"probability density $|\psi|^2$")
    fig.tight_layout()
    return fig


@lru_cache(maxsize=1)
def _tail_curve():
    """The value of the shot solution at the right end for many trial energies (computed once)."""
    Es = np.linspace(0.05, 5.8, 400)
    return Es, np.array([tail_value(e, steps=1500) for e in Es])


def shooting_figure(E):
    """Left: the solution of the Schrodinger equation for a chosen energy. Right: its value at the far end versus E."""
    x, psi = shoot(E)
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvspan(-turning_point(E), turning_point(E), color="0.94", zorder=0)
    ax.plot(x, np.clip(psi, -3, 3), color=RE_COLOR, lw=2.4)
    allowed = np.isclose(E - 0.5, np.round(E - 0.5), atol=1e-6) and E >= 0.5
    ax.set_ylim(-3, 3)
    ax.set_xlim(-6, 6)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"solution $\psi(x)$ (scaled)")
    ax.set_title(rf"$E={E:g}$:  " + ("an allowed energy, the solution dies away" if allowed else "the solution blows up at the right"), fontsize=10)
    Es, tails = _tail_curve()
    ax_t.axhline(0, color="0.6", lw=0.8)
    ax_t.plot(Es, np.sign(tails) * np.log10(1 + np.abs(tails)), color=INK, lw=2)
    for n in range(6):
        ax_t.axvline(float(energy(n)), color=OMEGA_COLOR, lw=1, ls="--")
    ax_t.axvline(E, color=A_COLOR, lw=2)
    ax_t.set_ylim(-7.5, 7.5)
    ax_t.set_xlabel(r"trial energy $E$ (in units of $\hbar\omega$)")
    ax_t.set_ylabel("value at the right end (sign $\\times\\ \\log_{10}(1+|\\mathrm{value}|)$)")
    ax_t.set_title("changes sign at each allowed energy (orange)", fontsize=10)
    fig.tight_layout()
    return fig


def icon_figure(coefficients=None, n=None, color=RE_COLOR, linewidth=14, size_inches=4.0):
    """The real part of a wave function as a bare curve: no axes, frame or background, for use as an icon.

    Give either a single state `n` or `coefficients` (a dict n -> c_n) for a superposition at t = 0.
    """
    x = np.linspace(-4.6, 4.6, 1200)
    if coefficients is not None:
        psi = superposition(coefficients, x, 0.0).real
    else:
        psi = hermite_function(0 if n is None else n, x)
    fig = plt.figure(figsize=(size_inches, size_inches * 0.62))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    ax.plot(x, psi, color=color, lw=linewidth, solid_capstyle="round", solid_joinstyle="round")
    pad = 0.12 * (psi.max() - psi.min())
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(psi.min() - pad, psi.max() + pad)
    return fig


def save_icon(fig, path_stem, dpi=400):
    """Write `<stem>.svg` and `<stem>.png`, both with a transparent background and no extra margin."""
    fig.savefig(f"{path_stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path_stem}.png", transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
