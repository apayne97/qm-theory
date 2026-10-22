"""Figures for the plane-wave notebook. Each function returns a matplotlib Figure."""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from waves import (
    box_energy,
    box_grid,
    box_hamiltonian,
    group_velocity,
    height,
    hole_amplitude,
    intensity,
    packet,
    phasor,
    plane_wave,
    two_hole_distances,
    width,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import (  # noqa: E402
    ARROW_COLOR,
    A_COLOR,
    CX2_COLOR,
    CX_COLOR,
    FREQ_COLOR,
    H2_COLOR,
    H_COLOR,
    IM_COLOR,
    INK,
    K2_COLOR,
    KIN_COLOR,
    K_COLOR,
    OMEGA2_COLOR,
    OMEGA_COLOR,
    PERIOD_COLOR,
    PHASE_COLOR,
    POT_COLOR,
    PROB_COLOR,
    RE_COLOR,
    R_COLOR,
)

_HEIGHT_CMAP = LinearSegmentedColormap.from_list("height", [H_COLOR, "#FFFFFF", A_COLOR])


def _unit_circle(ax):
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.6", lw=1))
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")


def arrow_figure(theta):
    """The amplitude exp(i theta) as an arrow on the unit circle, with its two components."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    _unit_circle(ax)
    z = phasor(theta)
    ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.5))
    ax.plot([z.real, z.real], [0, z.imag], color=IM_COLOR, lw=1.5, ls="--")
    ax.plot([0, z.real], [0, 0], color=RE_COLOR, lw=3)
    ax.set_title(rf"$\theta = {theta:.2f}$ rad  ({np.degrees(theta):.0f}$^\circ$)")
    fig.tight_layout()
    return fig


def time_rotation_figure(omega, t):
    """Amplitude at one fixed position: the arrow turns at rate omega as time passes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    _unit_circle(ax1)
    past = np.linspace(max(0, t - 2.0), t, 60) if t > 0 else np.array([0.0])
    trail = phasor(-omega * past)
    ax1.plot(trail.real, trail.imag, color=ARROW_COLOR, alpha=0.4, lw=2)
    z = phasor(-omega * t)
    ax1.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.5))
    ax1.set_title(r"arrow at one position, time $t$")

    tt = np.linspace(0, 10, 800)
    ax2.plot(tt, np.cos(omega * tt), color=RE_COLOR, label=r"real part  $\cos\omega t$")
    ax2.plot(tt, -np.sin(omega * tt), color=IM_COLOR, label=r"imaginary part  $-\sin\omega t$")
    ax2.plot([t], [np.cos(omega * t)], "o", color=RE_COLOR)
    ax2.plot([t], [-np.sin(omega * t)], "o", color=IM_COLOR)
    ax2.axvline(t, color="0.7", lw=0.8)
    if omega > 0:
        period = 2 * np.pi / omega
        ax2.annotate("", xy=(period, -1.45), xytext=(0, -1.45),
                     arrowprops=dict(arrowstyle="<->", color="0.3"))
        ax2.text(period / 2, -1.38, rf"$T = 2\pi/\omega = {period:.2f}$", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(-1.6, 1.4)
    ax2.set_xlabel("time $t$")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_title(r"the amplitude $e^{-i\omega t}$ at that position")
    fig.tight_layout()
    return fig


def space_twist_figure(k, n_arrows=14):
    """Amplitude at one instant: neighbouring arrows are turned by k per unit length."""
    x_max = 8.0
    x = np.linspace(0, x_max, 800)
    z = plane_wave(x, 0, k, 0)
    fig, (ax_dial, ax) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2]})
    # Top: the amplitude at evenly spaced positions, drawn as small dials.
    r = 0.25
    for xj in np.linspace(0, x_max, n_arrows):
        zj = plane_wave(xj, 0, k, 0)
        ax_dial.add_patch(plt.Circle((xj, 0), r, fill=False, color="0.7", lw=0.8))
        ax_dial.annotate("", xy=(xj + r * zj.real, r * zj.imag), xytext=(xj, 0),
                         arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.3, mutation_scale=8))
    ax_dial.set_aspect("equal", adjustable="datalim")
    ax_dial.axis("off")
    if k > 0:
        ax_dial.set_title(rf"the amplitude $e^{{ikx}}$ at one instant:  $\lambda = 2\pi/k = {2 * np.pi / k:.2f}$")
    # Bottom: the real and imaginary parts as continuous curves.
    ax.plot(x, z.real, color=RE_COLOR, label=r"real part  $\cos kx$")
    ax.plot(x, z.imag, color=IM_COLOR, label=r"imaginary part  $\sin kx$")
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlim(-0.3, x_max + 0.3)
    ax.set_xlabel("position $x$")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def moving_wave_figure(k, omega, t):
    """The real part of exp(i(kx - omega t)) at time t, with a point of fixed phase marked."""
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    z0 = plane_wave(x, 0, k, omega)
    z = plane_wave(x, t, k, omega)
    ax.plot(x, z0.real, color="0.8", lw=1.5, label=r"$t = 0$")
    ax.plot(x, z.real, color=RE_COLOR, lw=2, label=rf"$t = {t:.1f}$")
    # a crest: phase kx - omega t = 0 (mod 2 pi)
    crest = omega * t / k if k != 0 else 0.0
    crest = crest % (2 * np.pi / k) if k > 0 else 0.0
    ax.plot([crest], [1], "o", color=A_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("position $x$")
    ax.set_ylabel(r"$\cos(kx - \omega t)$")
    if k > 0:
        ax.set_title(rf"a point of fixed phase moves at $\omega/k = {omega / k:.2f}$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _density(ax, x, rho, label, ref=False):
    """Draw a probability density |psi|^2 in the palette colour: a filled curve, dashed and paler if it is the
    reference (earlier or comparison) curve."""
    ax.fill_between(x, 0, rho, color=PROB_COLOR, alpha=0.55 if ref else 1.0, label=label)
    ax.plot(x, rho, color="0.35", lw=1.5, ls="--" if ref else "-")


def probability_figure(k, omega, t):
    """Probability for the complex wave (uniform) versus the square of a real cosine (has nodes)."""
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _density(ax, x, np.abs(plane_wave(x, t, k, omega)) ** 2, r"complex wave $|e^{i(kx-\omega t)}|^2$")
    ax.plot(x, np.cos(k * x - omega * t) ** 2, color="0.35", lw=2, ls="--", label=r"real wave $\cos^2(kx-\omega t)$")
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("position $x$")
    ax.set_ylabel("relative probability")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def lump_buildup_figure(m, t=0.0, sigma_k=1.0, dk=0.5):
    """Build the lump from waves with weights a(k), the wave numbers spaced dk apart, at time t (hbar = m = 1).

    The waves k = -m dk ... m dk are included, and each turns at its own rate omega(k) = k^2 / 2. Left: the
    weights. Top right: Re psi, each wave pair (thin), their sum (thick), and the limit of very many closely
    spaced waves (dashed). Bottom right: the probability density of the sum and of that limit."""
    x = np.linspace(-8, 8, 1600)
    k_all = dk * np.arange(-12, 13)
    weight = lambda k: np.exp(-(k**2) / (4 * sigma_k**2)) * dk
    ks = dk * np.arange(-m, m + 1)
    psi = (weight(ks)[:, None] * np.exp(1j * (ks[:, None] * x[None, :] - 0.5 * ks[:, None] ** 2 * t))).sum(axis=0)
    sigma_x = 1 / (2 * sigma_k)
    scale = 2 * sigma_k * np.sqrt(np.pi) / packet(0.0, 0.0, 0.0, sigma_x).real
    exact = scale * packet(x, t, 0.0, sigma_x)

    fig = plt.figure(figsize=(10.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[1.25, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    ax1.vlines(k_all, 0, weight(k_all), color="0.85", lw=3)
    ax1.vlines(ks, 0, weight(ks), color=K_COLOR, lw=3)
    ax1.set_xlim(-6.6, 6.6)
    ax1.set_ylim(0, 0.55)
    ax1.set_xlabel("wave number $k$")
    ax1.set_ylabel(r"weight $a(k)$")
    ax1.set_title(f"{2 * m + 1} wave{'s' if m else ''} included", fontsize=10)

    for j in range(0, m + 1):  # the pair k and -k adds up to 2 w cos(kx) e^{-i omega t}; show its real part
        k = j * dk
        factor = 1 if j == 0 else 2
        ax2.plot(x, factor * weight(k) * np.cos(k * x) * np.cos(0.5 * k**2 * t), color="0.6", lw=0.8, alpha=0.7)
    ax2.plot(x, exact.real, color=INK, lw=1.3, ls="--", label="many closely spaced waves")
    ax2.plot(x, psi.real, color=RE_COLOR, lw=2.4, label="sum of the waves included")
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-1.2, 4.2)
    ax2.set_ylabel(r"real part of $\psi$")
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="upper right", fontsize=8)

    ax3.fill_between(x, 0, np.abs(psi) ** 2, color=PROB_COLOR, alpha=0.9)
    ax3.plot(x, np.abs(psi) ** 2, color="0.35", lw=1.5)
    ax3.plot(x, np.abs(exact) ** 2, color=INK, lw=1.3, ls="--")
    ax3.set_ylim(0, 14)
    ax3.set_xlabel("position $x$")
    ax3.set_ylabel(r"$|\psi|^2$")
    fig.tight_layout()
    return fig


def width_tradeoff_figure(sigma_k):
    """|a(k)|^2 and |psi(x)|^2 (each with total probability 1) with their widths, and the curve sigma_x = 1/(2 sigma_k)."""
    sigma_x = 1 / (2 * sigma_k)
    k = np.linspace(-7, 7, 1200)
    x = np.linspace(-5, 5, 1200)
    rho_k = np.exp(-(k**2) / (2 * sigma_k**2)) / (sigma_k * np.sqrt(2 * np.pi))
    rho_x = np.exp(-(x**2) / (2 * sigma_x**2)) / (sigma_x * np.sqrt(2 * np.pi))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    ax.fill_between(k, 0, rho_k, color=K_COLOR, alpha=0.35)
    ax.plot(k, rho_k, color=K_COLOR, lw=2)
    ax.annotate("", xy=(sigma_k, 0.55 * rho_k.max()), xytext=(-sigma_k, 0.55 * rho_k.max()),
                arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(0, 0.6 * rho_k.max(), r"$2\sigma_k$", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("wave number $k$")
    ax.set_title(rf"$|a(k)|^2$,  $\sigma_k = {sigma_k:.2f}$", fontsize=10)
    ax = axes[1]
    ax.fill_between(x, 0, rho_x, color=PROB_COLOR, alpha=0.8)
    ax.plot(x, rho_x, color="0.35", lw=2)
    ax.annotate("", xy=(sigma_x, 0.55 * rho_x.max()), xytext=(-sigma_x, 0.55 * rho_x.max()),
                arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(0, 0.6 * rho_x.max(), r"$2\sigma_x$", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("position $x$")
    ax.set_title(rf"$|\psi(x)|^2$,  $\sigma_x = 1/2\sigma_k = {sigma_x:.2f}$", fontsize=10)
    ax = axes[2]
    s = np.linspace(0.15, 3.0, 300)
    ax.fill_between(s, 0, 1 / (2 * s), color="0.88")
    ax.plot(s, 1 / (2 * s), color="0.35", lw=2)
    ax.plot([sigma_k], [sigma_x], "o", color=OMEGA_COLOR, ms=9, zorder=3)
    ax.text(0.8, 0.08, r"$\sigma_x\sigma_k<\frac{1}{2}$" + "\nnot possible", fontsize=8, ha="center", color="0.3")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r"$\sigma_k$")
    ax.set_ylabel(r"$\sigma_x$")
    ax.set_title(r"$\sigma_x\sigma_k = \frac{1}{2}$", fontsize=10)
    fig.tight_layout()
    return fig


def _entropy(rho, step):
    """Differential entropy, the integral of -rho ln(rho), of a density sampled on an evenly spaced grid."""
    positive = rho > 1e-300
    return -(rho[positive] * np.log(rho[positive])).sum() * step


@lru_cache(maxsize=1)
def _family_data():
    """The family a(k) = exp(-|k|^p / p), p = 1.5 ... 6 (p = 2 is the Gaussian).

    For each p: the densities |a|^2 in k and |psi|^2 in x (each with total probability 1), their
    root-mean-square widths and their entropies."""
    exps = np.linspace(1.5, 6.0, 19)
    k = np.linspace(-14, 14, 2001)
    dk = k[1] - k[0]
    x = np.linspace(-60, 60, 4001)
    dx = x[1] - x[0]
    kernel = np.exp(1j * np.outer(x, k))
    out = {"p": exps, "k": k, "x": x, "rho_k": [], "rho_x": [], "sigma_k": [], "sigma_x": [], "S_k": [], "S_x": []}
    for p in exps:
        a = np.exp(-(np.abs(k) ** p) / p)
        rho_k = a**2 / ((a**2).sum() * dk)
        rho_x = np.abs(kernel @ a * dk) ** 2
        rho_x /= rho_x.sum() * dx
        out["rho_k"].append(rho_k)
        out["rho_x"].append(rho_x)
        out["sigma_k"].append(np.sqrt((k**2 * rho_k).sum() * dk))
        out["sigma_x"].append(np.sqrt((x**2 * rho_x).sum() * dx))
        out["S_k"].append(_entropy(rho_k, dk))
        out["S_x"].append(_entropy(rho_x, dx))
    for key in ("sigma_k", "sigma_x", "S_k", "S_x"):
        out[key] = np.array(out[key])
    return out


def min_uncertainty_figure():
    """sigma_x sigma_k for a family of shapes a(k): the Gaussian (p = 2) gives the smallest product, 1/2."""
    data = _family_data()
    exps, prods = data["p"], data["sigma_x"] * data["sigma_k"]
    k = np.linspace(-3, 3, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))
    for p, c in ((1.5, "0.6"), (2.0, K_COLOR), (6.0, "0.3")):
        ax1.plot(k, np.exp(-(np.abs(k) ** p) / p), lw=2.2 if p == 2.0 else 1.5, color=c, label=rf"$p={p:g}$")
    ax1.set_xlabel("wave number $k$")
    ax1.set_ylabel(r"$a(k) = e^{-|k|^p/p}$")
    ax1.legend(fontsize=8)
    ax2.axhline(0.5, color="0.7", ls="--", lw=1)
    ax2.plot(exps, prods, "o-", color="0.35", lw=1.5, ms=4)
    ax2.plot([2.0], [prods[np.argmin(np.abs(exps - 2.0))]], "o", color=K_COLOR, ms=10, zorder=3, label="Gaussian")
    ax2.set_xlabel(r"shape exponent $p$")
    ax2.set_ylabel(r"$\sigma_x\,\sigma_k$")
    ax2.set_ylim(0.45, 0.75)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def entropy_family_figure(p):
    """For the shape a(k) = exp(-|k|^p / p): the densities in k and x with their entropies, and the sum of the
    entropies against the floor ln(pi e), with the bound ln(2 pi e sigma_x sigma_k) above it."""
    data = _family_data()
    i = int(np.argmin(np.abs(data["p"] - p)))
    k, x = data["k"], data["x"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    ax.fill_between(k, 0, data["rho_k"][i], color=K_COLOR, alpha=0.35)
    ax.plot(k, data["rho_k"][i], color=K_COLOR, lw=2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("wave number $k$")
    ax.set_title(rf"$|a(k)|^2$,  entropy $S_k={data['S_k'][i]:.3f}$", fontsize=10)
    ax = axes[1]
    ax.fill_between(x, 0, data["rho_x"][i], color=PROB_COLOR, alpha=0.8)
    ax.plot(x, data["rho_x"][i], color="0.35", lw=2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("position $x$")
    ax.set_title(rf"$|\psi(x)|^2$,  entropy $S_x={data['S_x'][i]:.3f}$", fontsize=10)
    ax = axes[2]
    floor = np.log(np.pi * np.e)
    total = data["S_x"] + data["S_k"]
    bound = np.log(2 * np.pi * np.e * data["sigma_x"] * data["sigma_k"])
    ax.axhline(floor, color="0.5", ls="--", lw=1.2)
    ax.text(5.95, floor - 0.012, r"floor $\ln(\pi e)$", ha="right", va="top", fontsize=8, color="0.4")
    ax.plot(data["p"], bound, color="0.75", lw=1.5, label=r"$\ln(2\pi e\,\sigma_x\sigma_k)$")
    ax.plot(data["p"], total, color="0.35", lw=2, label=r"$S_x+S_k$")
    ax.plot([data["p"][i]], [total[i]], "o", color=OMEGA_COLOR, ms=9, zorder=3)
    ax.set_ylim(2.12, 2.5)
    ax.set_xlabel(r"shape exponent $p$")
    ax.set_title(rf"$p={data['p'][i]:g}$:  $S_x+S_k={total[i]:.3f}$", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def schrodinger_check_figure(k0, sigma_x, t, dt=1e-3, n=4001):
    """Both sides of i dpsi/dt = -(1/2) d2psi/dx2 (hbar = m = 1) for the Gaussian packet, on a window that moves
    with the packet. Left: real parts. Right: imaginary parts. The relative difference is printed."""
    v = group_velocity(k0)
    x = np.linspace(v * t - 8, v * t + 8, n)
    dx = x[1] - x[0]
    psi = packet(x, t, k0, sigma_x)
    left = 1j * (packet(x, t + dt, k0, sigma_x) - packet(x, t - dt, k0, sigma_x)) / (2 * dt)
    right = np.zeros_like(psi)
    right[1:-1] = -0.5 * (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / dx**2
    inner = slice(1, -1)
    rel = np.max(np.abs(left[inner] - right[inner])) / np.max(np.abs(left[inner]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    d = x - v * t
    for ax, part, name in ((ax1, np.real, "real part"), (ax2, np.imag, "imaginary part")):
        ax.axhline(0, color="0.85", lw=0.8)
        ax.plot(d[inner], part(left[inner]), color=OMEGA_COLOR, lw=3.5, alpha=0.5, label=r"left side $i\,\partial\psi/\partial t$")
        ax.plot(d[inner], part(right[inner]), color=INK, lw=1.2, ls="--", label=r"right side $-\frac{1}{2}\,\partial^2\psi/\partial x^2$")
        ax.set_xlabel(r"position relative to the classical position, $x - vt$")
        ax.set_title(name, fontsize=10)
    for ax in (ax1, ax2):  # fixed axes: the largest value over the slider ranges is about 5.93
        ax.set_xlim(-8, 8)
        ax.set_ylim(-6.5, 6.5)
    ax1.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"largest difference between the two sides: {rel:.1e} of the largest value", fontsize=9, y=1.0)
    fig.tight_layout()
    return fig


def heat_vs_schrodinger_figure(t, sigma_k=1.0):
    """A Gaussian lump evolved with the Schroedinger equation (a factor i in the time derivative) and with the same
    equation without the i (the heat equation), hbar = m = 1. Left: the densities. Right: the total probability."""
    x = np.linspace(-12, 12, 6001)

    def amplitude(time, rate):
        # psi(x, t) = sqrt(pi / c) exp(-x^2 / 4c) with c = 1/(4 sigma_k^2) + rate * time / 2
        c = 1 / (4 * sigma_k**2) + rate * time / 2
        return np.sqrt(np.pi / c) * np.exp(-(x**2) / (4 * c))

    area0 = np.trapezoid(np.abs(amplitude(0.0, 1j)) ** 2, x)
    schr = np.abs(amplitude(t, 1j)) ** 2 / area0
    heat = np.abs(amplitude(t, 1.0)) ** 2 / area0
    ts = np.linspace(0, 3, 200)
    n_heat = 1 / np.sqrt(1 + 2 * sigma_k**2 * ts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    ax1.fill_between(x, 0, schr, color=PROB_COLOR, alpha=0.8, label="with the $i$ (Schr\u00f6dinger)")
    ax1.plot(x, schr, color="0.35", lw=1.5)
    ax1.plot(x, heat, color=OMEGA_COLOR, lw=2.2, label="without the $i$")
    ax1.set_ylim(0, 0.95)
    ax1.set_xlim(-12, 12)
    ax1.set_xlabel("position $x$")
    ax1.set_ylabel(r"$|\psi|^2$")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.axhline(1, color="0.5", lw=2.2, label="with the $i$: total stays 1")
    ax2.plot(ts, n_heat, color=OMEGA_COLOR, lw=2.2, label="without the $i$")
    ax2.plot([t], [1 / np.sqrt(1 + 2 * sigma_k**2 * t)], "o", color=OMEGA_COLOR, ms=8, zorder=3)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlim(0, 3)
    ax2.set_xlabel("time $t$")
    ax2.set_ylabel("total probability")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def conservation_figure(sigma, v, t):
    """A Gaussian probability density carried along by a flow of velocity v, so that the probability flow is j = v rho.

    Top: rho(x, t), with rho at t = 0 dashed. Middle: the flow j. Bottom: the rate of change d rho/dt = -d j/dx,
    positive where probability is gained and negative where it is lost. The areas under rho and under
    d rho/dt are printed: the first stays 1 and the second is 0."""
    x = np.linspace(-10, 10, 2001)

    def gaussian(centre):
        return np.exp(-((x - centre) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    rho = gaussian(v * t)
    j = v * rho
    rate = -np.gradient(j, x)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)

    _density(ax1, x, rho, rf"$\rho(x,t)$ at $t={t:g}$")
    ax1.plot(x, gaussian(0.0), color="0.5", lw=1.0, ls=":", label=r"$\rho$ at $t=0$")
    ax1.set_ylim(0, 0.6)
    ax1.set_ylabel(r"density $\rho$")
    ax1.set_title(rf"total probability $\int\rho\,dx = {np.trapezoid(rho, x):.3f}$", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.plot(x, j, color=OMEGA_COLOR, lw=2.2)
    ax2.fill_between(x, 0, j, color=OMEGA_COLOR, alpha=0.2)
    ax2.set_ylim(-0.9, 0.9)
    ax2.set_ylabel(r"flow $j = v\rho$")

    ax3.axhline(0, color="0.85", lw=0.8)
    ax3.fill_between(x, 0, rate, where=rate >= 0, color=CX_COLOR, alpha=0.8, label="probability gained")
    ax3.fill_between(x, 0, rate, where=rate < 0, color=OMEGA_COLOR, alpha=0.5, label="probability lost")
    ax3.plot(x, rate, color="0.35", lw=1.4)
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_xlim(-10, 10)
    ax3.set_xlabel("position $x$")
    ax3.set_ylabel(r"$\partial\rho/\partial t = -\partial j/\partial x$")
    ax3.set_title(rf"total change $\int \partial\rho/\partial t\,dx = {np.trapezoid(rate, x):+.3f}$", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def packet_flow_figure(k0, sigma0, t):
    """The Gaussian packet of section 6 (hbar = m = 1): density rho, probability flow j and local velocity v = j / rho.

    j = Im(psi* dpsi/dx). The exact result is v(x, t) = k0 + (x - k0 t) t / (t^2 + t_s^2), t_s = 2 sigma0^2:
    the centre moves at k0 and the velocity grows across the lump, which is why it widens. The axes are fixed."""
    ts = 2 * sigma0**2
    x = np.linspace(-12, 28, 4001)
    dx = x[1] - x[0]
    psi = packet(x, t, k0, sigma0)
    rho = np.abs(psi) ** 2
    j = np.imag(np.conj(psi) * np.gradient(psi, dx))
    visible = rho > 0.02 * rho.max()
    v_exact = k0 + (x - k0 * t) * t / (t**2 + ts**2)
    sigma_t = width(t, sigma0)
    mean_flow = np.trapezoid(j, x)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
    _density(ax1, x, rho, rf"$\rho = |\psi|^2$ at $t={t:g}$")
    ax1.axvline(k0 * t, color=ARROW_COLOR, ls="--", lw=1.2, label="centre, $x = v_0 t$")
    ax1.set_ylim(0, 0.45)
    ax1.set_ylabel(r"density $\rho$")
    ax1.set_title(rf"width $\sigma(t) = {sigma_t:.2f}$", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.plot(x, j, color=OMEGA_COLOR, lw=2.2)
    ax2.fill_between(x, 0, j, color=OMEGA_COLOR, alpha=0.2)
    ax2.set_ylim(-0.3, 1.0)
    ax2.set_ylabel(r"flow $j$")
    ax2.set_title(rf"average movement: $\int j\,dx = {mean_flow:.3f}$, the classical velocity $v_0 = {k0:g}$", fontsize=10)

    ax3.axhline(k0, color="0.6", lw=1.0, ls=":", label=r"$v_0$")
    ax3.plot(x[visible], (j[visible] / rho[visible]), color="0.35", lw=2.2, label=r"$v = j/\rho$ from the packet")
    ax3.plot(x[visible], v_exact[visible], color=INK, lw=1.2, ls="--", label=r"$v_0 + (x - v_0t)\,t/(t^2+t_s^2)$")
    ax3.axvline(k0 * t, color=ARROW_COLOR, ls="--", lw=1.0)
    ax3.set_ylim(-1.6, 3.2)
    ax3.set_xlim(-12, 28)
    ax3.set_xlabel("position $x$")
    ax3.set_ylabel(r"local velocity $v$")
    ax3.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def quantum_term_figure(sigma0, t, k0=1.0):
    """The Gaussian packet (hbar = m = 1) in the frame of its centre u = x - k0 t: the density, the quantum term
    Q = -(1/2) R''/R with R = |psi|, and the force -dQ/dx on the probability, which equals the acceleration
    dv/dt + v dv/dx of the probability computed from the packet (dashed). Q and the force are drawn where the density
    is not negligible. The axes are fixed."""
    ts = 2 * sigma0**2
    sigma_t = sigma0 * np.sqrt(1 + (t / ts) ** 2)
    u = np.linspace(-8, 8, 3201)
    x = u + k0 * t

    def psi_at(tt):
        return packet(x, tt, k0, sigma0)

    rho = np.abs(psi_at(t)) ** 2
    visible = rho > 0.02 * rho.max()

    def velocity(tt):
        p = psi_at(tt)
        return np.imag(np.conj(p) * np.gradient(p, x)) / np.abs(p) ** 2

    dt = 1e-3
    accel = (velocity(t + dt) - velocity(t - dt)) / (2 * dt) + velocity(t) * np.gradient(velocity(t), x)
    amplitude = np.abs(psi_at(t))
    q_numeric = -0.5 * np.gradient(np.gradient(amplitude, x), x) / amplitude
    q = (1 / (4 * sigma_t**2)) * (1 - u**2 / (2 * sigma_t**2))
    force = u / (4 * sigma_t**4)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
    _density(ax1, u, rho, rf"$\rho$ at $t={t:g}$")
    ax1.set_ylim(0, 0.45)
    ax1.set_ylabel(r"density $\rho$")
    ax1.set_title(rf"width $\sigma(t) = {sigma_t:.2f}$", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.plot(u[visible], q[visible], color=OMEGA_COLOR, lw=2.4, label=r"$Q = \frac{\hbar^2}{4m\sigma^2}\left(1-\frac{u^2}{2\sigma^2}\right)$")
    ax2.plot(u[visible][8:-8], q_numeric[visible][8:-8], color=INK, lw=1.1, ls="--", label=r"$-\frac{\hbar^2}{2m}R''/R$ from the packet")
    ax2.set_ylim(-0.85, 0.35)
    ax2.set_ylabel(r"quantum term $Q$")
    ax2.legend(loc="lower center", fontsize=8)

    ax3.axhline(0, color="0.85", lw=0.8)
    ax3.plot(u[visible], force[visible], color=OMEGA_COLOR, lw=2.4, label=r"$-\frac{1}{m}\,\partial Q/\partial x = \frac{\hbar^2 u}{4m^2\sigma^4}$")
    ax3.plot(u[visible][8:-8], accel[visible][8:-8], color=INK, lw=1.1, ls="--", label=r"$\partial v/\partial t + v\,\partial v/\partial x$ from the packet")
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_xlim(-8, 8)
    ax3.set_xlabel(r"position relative to the centre, $u = x - v_0 t$")
    ax3.set_ylabel("acceleration of the probability")
    ax3.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def box_states_figure(weight, t):
    """Particle in a box, 0 < x < 1 (hbar = m = L = 1): psi_n = sqrt(2) sin(n pi x), E_n = n^2 pi^2 / 2.

    A composite of the first two stationary states with probabilities 1 - weight and weight. Left: the energy levels,
    with the probability of each drawn as a bar and the average energy dashed. Right: the density |psi(x, t)|^2
    (solid) and its time average (dotted). The axes are fixed."""
    x = np.linspace(0, 1, 2001)
    energy = lambda n: n**2 * np.pi**2 / 2
    state = lambda n: np.sqrt(2) * np.sin(n * np.pi * x)
    c1, c2 = np.sqrt(1 - weight), np.sqrt(weight)
    omega12 = energy(2) - energy(1)
    density = c1**2 * state(1) ** 2 + c2**2 * state(2) ** 2 + 2 * c1 * c2 * state(1) * state(2) * np.cos(omega12 * t)
    average = c1**2 * state(1) ** 2 + c2**2 * state(2) ** 2
    mean_energy = (1 - weight) * energy(1) + weight * energy(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    for n in range(1, 4):
        ax1.hlines(energy(n), 0, 1, color="0.8", lw=1.5)
        ax1.text(1.03, energy(n), f"$n={n}$", va="center", fontsize=8, color="0.4")
    for n, prob in ((1, 1 - weight), (2, weight)):
        ax1.hlines(energy(n), 0, prob, color=OMEGA_COLOR, lw=6)
    ax1.axhline(mean_energy, color=INK, lw=1.2, ls="--")
    ax1.text(0.5, mean_energy + 0.6, rf"average energy $\langle E\rangle = {mean_energy:.2f}$", ha="center", fontsize=8)
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 50)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xlabel("probability of the energy")
    ax1.set_ylabel(r"energy $E$ (units $\hbar^2/mL^2$)")
    ax1.set_title("energy levels", fontsize=10)

    ax2.plot(x, average, color="0.5", lw=1.2, ls=":", label="time average")
    _density(ax2, x, density, rf"$|\psi|^2$ at $t={t:g}$")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 3.6)
    ax2.set_xlabel("position $x$ (in units of $L$)")
    ax2.set_ylabel(r"density $|\psi|^2$")
    ax2.set_title(rf"the density oscillates with period $2\pi\hbar/(E_2-E_1) = {2 * np.pi / omega12:.3f}$", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def box_matrix_figure(n_points, n):
    """The particle in a box (hbar = m = L = 1) as a matrix on n_points grid points. Left: the matrix H (orange:
    positive, blue: negative, white: zero). Middle: the n-th eigenvector as one number per grid point (dots), on the
    continuous state sqrt(2) sin(n pi x) (line). Right: the lowest eigenvalues of the matrix (dots) against the exact
    energies (lines)."""
    x, dx = box_grid(n_points)
    energies, vectors = np.linalg.eigh(box_hamiltonian(n_points))
    n = min(n, n_points)
    vector = vectors[:, n - 1] / np.sqrt(dx)  # unit vector -> values of psi_n(x_j), so that sum psi^2 dx = 1
    peak_j = np.argmax(np.abs(vector))
    vector = vector * np.sign(vector[peak_j] * np.sin(n * np.pi * x[peak_j]))  # eigh fixes the sign only up to +-1
    fine = np.linspace(0, 1, 801)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1.5, 1]})
    h = box_hamiltonian(n_points)
    limit = np.abs(h).max()
    ax1.imshow(h, cmap=LinearSegmentedColormap.from_list("h", [H_COLOR, "white", OMEGA_COLOR]), vmin=-limit, vmax=limit)
    ax1.set_title(rf"$H$: a {n_points}$\times${n_points} matrix", fontsize=10)
    ax1.set_xlabel("column $k$")
    ax1.set_ylabel("row $j$")
    ticks = list(range(0, n_points, max(1, n_points // 5)))
    ax1.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax1.set_yticks(ticks, [str(t + 1) for t in ticks])

    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.plot(fine, np.sqrt(2) * np.sin(n * np.pi * fine), color=CX_COLOR, lw=1.8, label=rf"$\psi_{{{n}}}(x)$, continuous")
    ax2.vlines(x, 0, vector, color=CX_COLOR, lw=1.0, alpha=0.6)
    ax2.plot(x, vector, "o", color=INK, ms=4.5, label=rf"eigenvector of $H$: $\psi_j$ at $x_j = j\,\Delta x$")
    ax2.plot([0, 1], [0, 0], "s", color=A_COLOR, ms=4, label=r"walls: $\psi = 0$")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-1.7, 2.7)
    ax2.set_xlabel("position $x$ (in units of $L$)")
    ax2.set_title(rf"state $n={n}$ with $N={n_points}$ grid points", fontsize=10)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=1)

    shown = min(n_points, 6)
    for m in range(1, shown + 1):
        ax3.hlines(box_energy(m), 0.05, 0.55, color="0.75", lw=1.5)
        ax3.plot([0.8], [energies[m - 1]], "o", color=OMEGA_COLOR if m == n else INK, ms=5)
        ax3.text(0.95, box_energy(m), f"$n={m}$", va="center", fontsize=8, color="0.4")
    ax3.set_xlim(0, 1.25)
    ax3.set_ylim(0, box_energy(shown) * 1.08)
    ax3.set_xticks([0.3, 0.8], ["exact", "matrix"])
    ax3.set_ylabel(r"energy $E$ (units $\hbar^2/mL^2$)")
    ax3.set_title("eigenvalues of $H$", fontsize=10)
    fig.tight_layout()
    return fig


def rest_energy_phase_figure(phase, k0=5.0, sigma_x=1.5, t=2.0):
    """A packet with and without the factor exp(-i phase) from the rest energy: Re psi changes, |psi|^2 does not."""
    v = group_velocity(k0)
    x = np.linspace(v * t - 6, v * t + 6, 2000)
    psi = packet(x, t, k0, sigma_x)
    turned = np.exp(-1j * phase) * psi
    scale = np.abs(packet(0.0, 0.0, k0, sigma_x))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    ax1.fill_between(x, 0, np.abs(psi) ** 2 / scale**2, color=PROB_COLOR, label=r"$|\psi|^2$ without the rest-energy phase")
    ax1.plot(x, np.abs(turned) ** 2 / scale**2, color=OMEGA_COLOR, lw=1.8, ls="--", label=r"$|\psi|^2$ with it")
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("probability density")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(x, psi.real / scale, color=RE_COLOR, lw=1.0, alpha=0.7, label=r"Re $\psi$ without")
    ax2.plot(x, turned.real / scale, color=OMEGA_COLOR, lw=1.0, label=r"Re $\psi$ with")
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_ylabel("real part")
    ax2.set_xlabel("position $x$")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def band_shapes_figure(sigma_k=1.0, half_range=10.0):
    """|a(k)|^2 for a top-hat and a Gaussian of equal rms width, and the |psi(x)|^2 each produces at t = 0."""
    k = np.linspace(-6 * sigma_k, 6 * sigma_k, 4001)
    dk = k[1] - k[0]
    w = np.sqrt(12) * sigma_k  # a uniform band of width w has rms width w / sqrt(12)
    amps = {
        "top-hat": np.where(np.abs(k) <= w / 2, 1.0, 0.0),
        "Gaussian": np.exp(-(k**2) / (4 * sigma_k**2)),
    }
    colors = {"top-hat": "0.6", "Gaussian": "0.35"}
    x = np.linspace(-half_range, half_range, 2001)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, a in amps.items():
        a = a / np.sqrt(np.sum(a**2) * dk)  # total probability over k equal to 1
        ax1.plot(k, a**2, color=colors[name], lw=2.2, label=name)
        if name == "Gaussian":
            ax1.fill_between(k, 0, a**2, color=PROB_COLOR)
        psi = (a[None, :] * np.exp(1j * np.outer(x, k))).sum(axis=1) * dk
        rho = np.abs(psi) ** 2
        ax2.plot(x, rho / rho.max(), color=colors[name], lw=2.2, label=name)
        if name == "Gaussian":
            ax2.fill_between(x, 1e-5, rho / rho.max(), color=PROB_COLOR)
    ax1.set_xlabel("wave number $k$")
    ax1.set_ylabel(r"probability of $k$, $|a(k)|^2$")
    ax1.legend(fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 2)
    ax2.set_xlabel("position $x$")
    ax2.set_ylabel(r"probability density $|\psi|^2$ (peak = 1)")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def flat_vs_lump_figure(k0, sigma_x, half_range=12.0):
    """|psi|^2 of a plane wave (flat) and a packet (lump), and the probability inside |x| < X as X grows."""
    x = np.linspace(-half_range, half_range, 4000)
    plane = np.abs(plane_wave(x, 0.0, k0, 0.0)) ** 2
    lump = np.abs(packet(x, 0.0, k0, sigma_x)) ** 2
    X = np.linspace(0.0, half_range, 400)
    inside_plane = 2 * X  # integral of the constant 1 over -X..X
    inside_lump = np.array([np.trapezoid(lump[np.abs(x) <= r], x[np.abs(x) <= r]) for r in X])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    _density(ax1, x, plane, r"plane wave $|e^{ikx}|^2$", ref=True)
    _density(ax1, x, lump, r"packet $|\psi|^2$")
    ax1.set_ylim(0, 1.3)
    ax1.set_xlabel("position $x$")
    ax1.set_ylabel("probability density")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(X, inside_plane, color="0.35", lw=2.2, ls="--", label="plane wave")
    ax2.plot(X, inside_lump, color="0.35", lw=2.2, label="packet")
    ax2.axhline(1, color="0.7", ls="--", lw=1)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 40)
    ax2.set_xlim(0, half_range)
    ax2.set_xlabel(r"half-width $X$ of the region $-X < x < X$")
    ax2.set_ylabel("total probability inside")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def packet_times_figure(k0, sigma_x, times, xlim=None):
    """|psi|^2 and Re psi of the packet at several times, with the classical position marked."""
    v = group_velocity(k0)
    if xlim is None:
        xlim = (-4 * sigma_x, v * max(times) + 4 * width(max(times), sigma_x))
    x = np.linspace(*xlim, 4000)
    fig, axes = plt.subplots(len(times), 1, figsize=(9, 1.9 * len(times)), sharex=True)
    for ax, t in zip(np.atleast_1d(axes), times):
        psi = packet(x, t, k0, sigma_x)
        scale = np.abs(packet(0.0, 0.0, k0, sigma_x))  # peak amplitude at t = 0
        ax.plot(x, psi.real / scale, color=RE_COLOR, lw=0.8, alpha=0.8, label=r"$\mathrm{Re}\,\psi$")
        ax.fill_between(x, 0, np.abs(psi) ** 2 / scale**2, color="0.75", alpha=0.7, label=r"$|\psi|^2$")
        ax.axvline(v * t, color=ARROW_COLOR, ls="--", lw=1.2, label=r"classical position $x = vt$")
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(rf"$t = {t:g}$")
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8, ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("position $x$")
    fig.tight_layout()
    return fig


def spacetime_figure(k0, sigma_x, t_max):
    """Probability density |psi(x, t)|^2 as a map in space and time, with the classical path."""
    v = group_velocity(k0)
    x = np.linspace(-4 * sigma_x, v * t_max + 4 * width(t_max, sigma_x), 900)
    t = np.linspace(0, t_max, 300)
    dens = np.array([np.abs(packet(x, ti, k0, sigma_x)) ** 2 for ti in t])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(dens, extent=[x[0], x[-1], t_max, 0], aspect="auto", cmap="magma")
    ax.plot(v * t, t, color="white", ls="--", lw=1.2, label=r"classical path $x = vt$")
    ax.set_xlabel("position $x$")
    ax.set_ylabel("time $t$")
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    fig.tight_layout()
    return fig


def spreading_figure(k0, sigma_x, t):
    """Probability of the packet at t = 0 and at time t, both drawn about the classical position (fixed axes)."""
    x = np.linspace(-20, 20, 4000)
    v = group_velocity(k0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for ti, is_ref in zip([0.0, t], [True, False]):
        d = np.abs(packet(x + v * ti, ti, k0, sigma_x)) ** 2
        _density(ax, x, d, rf"$|\psi|^2$ at $t={ti:g}$", ref=is_ref)
    ax.axvline(0, color=ARROW_COLOR, ls="--", lw=1.2, label=r"classical position")
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 1.4)
    ax.set_xlabel(r"position relative to the classical position, $x - vt$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def width_vs_distance_figure(k0, sigma0, d, d_max=20.0):
    """Uncertainty in position, Delta x, against the distance d = v t from the source (hbar = m = 1), and the packet
    at distance d. sigma0 is the width sigma_x at the source: Delta x(d) = sigma0 sqrt(1 + (d / d_s)^2) with
    d_s = v t_s = 2 k0 sigma0^2."""
    d_s = 2 * k0 * sigma0**2
    ds = np.linspace(0, d_max, 600)
    sig = sigma0 * np.sqrt(1 + (ds / d_s) ** 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.plot(ds, sig, color=RE_COLOR, lw=2.4, label=r"uncertainty $\Delta x(d)$")
    ax1.axhline(sigma0, color="0.6", lw=1.0, ls="--", label=r"at the source, $\sigma_x$")
    ax1.plot(ds, ds / (2 * k0 * sigma0), color="0.5", lw=1.0, ls=":", label=r"far from the source: $d/2k_0\sigma_x$")
    ax1.axvline(d_s, color=ARROW_COLOR, lw=1.0, ls="--")
    ax1.text(d_s, 11.4, r" $d_s=2k_0\sigma_x^2$", fontsize=8, va="top", color=ARROW_COLOR)
    ax1.plot([d], [sigma0 * np.sqrt(1 + (d / d_s) ** 2)], "o", color=OMEGA_COLOR, ms=9, zorder=3)
    ax1.set_xlim(0, d_max)
    ax1.set_ylim(0, 12)
    ax1.set_xlabel("distance from the source, $d = vt$")
    ax1.set_ylabel(r"uncertainty in position, $\Delta x$")
    ax1.legend(fontsize=7, loc="center right")
    x = np.linspace(-14, 14, 3000)
    now = np.abs(packet(x + d, d / group_velocity(k0), k0, sigma0)) ** 2
    start = np.abs(packet(x, 0.0, k0, sigma0)) ** 2
    _density(ax2, x, start, "at the source", ref=True)
    _density(ax2, x, now, f"at distance {d:g}")
    ax2.set_xlim(-14, 14)
    ax2.set_ylim(0, 0.9)
    ax2.set_xlabel(r"position relative to the classical position, $x - d$")
    ax2.set_ylabel(r"$|\psi|^2$")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def interference_figure(x, wavelength, d, L=20.0, x_range=15.0, show_arrows=True):
    """Two holes: the heights over two periods and the pattern, and (optionally) the complex amplitudes as arrows."""
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi  # period 1
    r1, r2 = two_hole_distances(x, d, L)
    h1, h2 = hole_amplitude(r1, k), hole_amplitude(r2, k)
    total = h1 + h2

    if show_arrows:
        fig = plt.figure(figsize=(10, 6.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[1.15, 1])
        ax_arrow = fig.add_subplot(gs[0, 0])
        ax_time = fig.add_subplot(gs[0, 1])
        ax_pat = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(10, 5.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_arrow = None
        ax_time = fig.add_subplot(gs[0, 0])
        ax_pat = fig.add_subplot(gs[1, 0])

    if ax_arrow is not None:
        # arrows: the sum (drawn first and thick, so it stays visible underneath), then h1 and h2 tip to tail
        _unit_circle(ax_arrow)
        ax_arrow.set_xlim(-2.3, 2.3)
        ax_arrow.set_ylim(-2.3, 2.3)
        arrow = dict(arrowstyle="-|>", lw=2.5, mutation_scale=14)
        ax_arrow.annotate("", xy=(total.real, total.imag), xytext=(0, 0),
                          arrowprops=dict(color=INK, arrowstyle="-|>", lw=7, mutation_scale=18), zorder=1)
        ax_arrow.annotate("", xy=(h1.real, h1.imag), xytext=(0, 0),
                          arrowprops=dict(color=ARROW_COLOR, **arrow), zorder=3)
        ax_arrow.annotate("", xy=(total.real, total.imag), xytext=(h1.real, h1.imag),
                          arrowprops=dict(color=CX2_COLOR, **arrow), zorder=3)
        ax_arrow.set_title(rf"$\psi_1$ + $\psi_2$ (tip to tail) = sum (dark);  $|\psi_1+\psi_2|^2 = {intensity(total):.2f}$", fontsize=9)

    # heights at the detector over two periods
    t = np.linspace(0, 2, 600)
    ax_time.plot(t, height(h1, t, omega), color=H_COLOR, lw=1.6, label=r"$H_1(t)$")
    ax_time.plot(t, height(h2, t, omega), color=H2_COLOR, lw=1.6, label=r"$H_2(t)$")
    ax_time.plot(t, height(total, t, omega), color=INK, lw=2.6, label=r"$H_1+H_2$")
    ax_time.axhline(0, color="0.8", lw=0.8)
    ax_time.set_ylim(-2.3, 2.9)
    ax_time.set_xlabel("time $t$ (in periods)")
    ax_time.set_ylabel("height at the detector")
    ax_time.legend(loc="upper right", fontsize=8, ncol=3)
    delta = k * (r2 - r1)
    ax_time.set_title(rf"$\delta = k(r_2-r_1) = {delta / (2 * np.pi):.2f}\times 2\pi$", fontsize=9)

    # intensity along the detector line
    xs = np.linspace(-x_range, x_range, 1500)
    a1, a2 = two_hole_distances(xs, d, L)
    g1, g2 = hole_amplitude(a1, k), hole_amplitude(a2, k)
    ax_pat.plot(xs, intensity(g1 + g2), color=INK, lw=2, label=r"both holes open: $|\psi_1+\psi_2|^2$" if show_arrows else r"both holes open: $I_{12}$")
    ax_pat.plot(xs, intensity(g1) + intensity(g2), color="0.55", lw=1.4, ls="--", label=r"sum of the separate intensities: $|\psi_1|^2+|\psi_2|^2$" if show_arrows else r"separate intensities added: $I_1+I_2$")
    ax_pat.axvline(x, color=R_COLOR, lw=1.2, alpha=0.8)
    ax_pat.plot([x], [intensity(total)], "o", color=R_COLOR, ms=8)
    ax_pat.set_ylim(-0.1, 4.6)
    ax_pat.set_xlabel("detector position $x$")
    ax_pat.set_ylabel("intensity")
    ax_pat.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def time_cosine_figure(amplitude, omega):
    """H(t) = A cos(omega t) at one position, with the amplitude and the period marked."""
    t = np.linspace(0, 10, 1200)
    period = 2 * np.pi / omega
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, amplitude * np.cos(omega * t), color=RE_COLOR, lw=2)
    ax.axhline(0, color="0.8", lw=0.8)
    for sign in (+1, -1):
        ax.axhline(sign * amplitude, color="0.7", ls=":", lw=1)
    ax.annotate("", xy=(0, amplitude), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color=A_COLOR, lw=1.6))
    ax.text(0.12, amplitude / 2, r"$A$", color=A_COLOR, va="center")
    if period <= 10:
        ax.annotate("", xy=(period, 1.25 * amplitude), xytext=(0, 1.25 * amplitude),
                    arrowprops=dict(arrowstyle="<->", color=PERIOD_COLOR, lw=1.4))
        ax.text(period / 2, 1.32 * amplitude, rf"period $T = 2\pi/\omega = {period:.2f}$", ha="center", va="bottom", fontsize=9, color=PERIOD_COLOR)
    ax.set_ylim(-3.0, 3.9)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$H(t) = A\cos(\omega t)$")
    fig.tight_layout()
    return fig


def space_cosine_figure(k):
    """H(x) = cos(kx) at one moment, with the wavelength marked."""
    x = np.linspace(0, 10, 1200)
    wavelength = 2 * np.pi / k
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, np.cos(k * x), color=RE_COLOR, lw=2)
    ax.axhline(0, color="0.8", lw=0.8)
    if wavelength <= 10:
        ax.annotate("", xy=(wavelength, 1.25), xytext=(0, 1.25), arrowprops=dict(arrowstyle="<->", color=PERIOD_COLOR, lw=1.4))
        ax.text(wavelength / 2, 1.32, rf"wavelength $\lambda = 2\pi/k = {wavelength:.2f}$", ha="center", va="bottom", fontsize=9, color=PERIOD_COLOR)
    ax.set_ylim(-1.4, 1.9)
    ax.set_xlabel("position $x$")
    ax.set_ylabel(r"$H(x) = \cos(kx)$")
    fig.tight_layout()
    return fig


def _phase_axis(ax, positions, labels, label):
    """Secondary axis along the top of `ax` showing the phase at the given positions."""
    top = ax.secondary_xaxis("top")
    top.set_xticks(positions)
    top.set_xticklabels(labels)
    top.set_xlabel(label, fontsize=10, color=PHASE_COLOR)
    top.tick_params(colors=PHASE_COLOR)
    return top


def _double_arrow(ax, p0, p1, color="0.25", lw=1.5):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=lw, shrinkA=0, shrinkB=0))


def time_features_figure(amplitude=1.5, omega=2.0, t1=0.8):
    """Where A, omega, t, H(t) and T appear on the graph of H(t) = A cos(omega t) at one position."""
    period = 2 * np.pi / omega
    t = np.linspace(0, 2.25 * period, 1500)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.4), gridspec_kw=dict(height_ratios=[1.6, 1]))

    ax.plot(t, amplitude * np.cos(omega * t), color=RE_COLOR, lw=2.2)
    ax.axhline(0, color="0.75", lw=0.8)
    for sign in (+1, -1):
        ax.axhline(sign * amplitude, color="0.7", ls=":", lw=1)
    # amplitude: from the axis to the first crest
    _double_arrow(ax, (0.12, 0), (0.12, amplitude), color=A_COLOR)
    ax.text(0.03, amplitude * 0.78, r"size $A$" "\n(largest height)", color=A_COLOR, va="center", ha="right", fontsize=10)
    # period: crest to crest
    ymark = 1.3 * amplitude
    _double_arrow(ax, (0, ymark), (period, ymark), color=PERIOD_COLOR)
    ax.text(period / 2, ymark + 0.08, r"period $T=2\pi/\omega$", ha="center", va="bottom", fontsize=10, color=PERIOD_COLOR)
    for x_c in (0, period):
        ax.plot([x_c, x_c], [amplitude, ymark], color="0.6", lw=0.8, ls="--")
    # one chosen time and the height at that time
    h1 = amplitude * np.cos(omega * t1)
    ax.plot([t1, t1], [0, h1], color="0.35", ls="--", lw=1)
    ax.plot([0, t1], [h1, h1], color="0.35", ls="--", lw=1)
    ax.plot([t1], [h1], "o", color="k", ms=7, zorder=5)
    ax.text(t1, -0.14 * amplitude, r"$t$", ha="center", va="top", fontsize=11)
    ax.text(-0.06, h1, r"$H(t)$", ha="right", va="center", fontsize=11, color=H_COLOR)
    ax.text(t1 + 0.1, h1 + 0.1, rf"$H(t)=A\cos(\omega t)$" "\n" rf"at this $t$, $\omega t={omega * t1:.2f}$ rad", fontsize=9, va="bottom")
    ax.set_xlim(-1.3, 2.25 * period)
    ax.set_ylim(-1.35 * amplitude, 1.75 * amplitude)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"height $H(t)$")
    n = 5
    _phase_axis(ax, [j * period / 2 for j in range(n)], ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"],
                r"phase $\omega t$: it reaches $2\pi$ when $t=T$, so $\omega=2\pi/T$")

    # comparison: doubling omega halves T
    t2 = np.linspace(0, 2.25 * period, 1500)
    ax2.plot(t2, np.cos(omega * t2), color=OMEGA_COLOR, lw=2, label=rf"$\omega={omega:g}$")
    ax2.plot(t2, np.cos(2 * omega * t2), color=OMEGA2_COLOR, lw=2, label=rf"$\omega={2 * omega:g}$ (twice as large)")
    ax2.axhline(0, color="0.75", lw=0.8)
    _double_arrow(ax2, (0, 1.3), (period, 1.3), color=PERIOD_COLOR)
    ax2.text(period / 2, 1.35, r"$T$", ha="center", va="bottom", color=PERIOD_COLOR)
    _double_arrow(ax2, (0, 1.9), (period / 2, 1.9), color=PERIOD_COLOR)
    ax2.text(period / 4, 1.95, r"$T/2$", ha="center", va="bottom", color=PERIOD_COLOR)
    ax2.set_xlim(-0.55, 2.25 * period)
    ax2.set_ylim(-1.3, 2.5)
    ax2.set_xlabel(r"time $t$")
    ax2.set_ylabel("height (A = 1)")
    ax2.set_title(r"a larger $\omega$ means a shorter period: more oscillations per unit time", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def space_features_figure(amplitude=1.5, k=1.5, x1=1.1):
    """Where A, k, x, H(x) and the wavelength appear on the graph of H(x) = A cos(kx) at one moment."""
    wavelength = 2 * np.pi / k
    x = np.linspace(0, 2.25 * wavelength, 1500)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.4), gridspec_kw=dict(height_ratios=[1.6, 1]))

    ax.plot(x, amplitude * np.cos(k * x), color=RE_COLOR, lw=2.2)
    ax.axhline(0, color="0.75", lw=0.8)
    for sign in (+1, -1):
        ax.axhline(sign * amplitude, color="0.7", ls=":", lw=1)
    _double_arrow(ax, (0.12, 0), (0.12, amplitude), color=A_COLOR)
    ax.text(0.03, amplitude * 0.78, r"size $A$" "\n(largest height)", color=A_COLOR, va="center", ha="right", fontsize=10)
    ymark = 1.3 * amplitude
    _double_arrow(ax, (0, ymark), (wavelength, ymark), color=PERIOD_COLOR)
    ax.text(wavelength / 2, ymark + 0.08, r"wavelength $\lambda=2\pi/k$", ha="center", va="bottom", fontsize=10, color=PERIOD_COLOR)
    for x_c in (0, wavelength):
        ax.plot([x_c, x_c], [amplitude, ymark], color="0.6", lw=0.8, ls="--")
    h1 = amplitude * np.cos(k * x1)
    ax.plot([x1, x1], [0, h1], color="0.35", ls="--", lw=1)
    ax.plot([0, x1], [h1, h1], color="0.35", ls="--", lw=1)
    ax.plot([x1], [h1], "o", color="k", ms=7, zorder=5)
    ax.text(x1, -0.14 * amplitude, r"$x$", ha="center", va="top", fontsize=11)
    ax.text(-0.06, h1, r"$H(x)$", ha="right", va="center", fontsize=11, color=H_COLOR)
    ax.text(x1 + 0.1, h1 + 0.1, rf"$H(x)=A\cos(kx)$" "\n" rf"at this $x$, $kx={k * x1:.2f}$ rad", fontsize=9, va="bottom")
    ax.set_xlim(-1.3, 2.25 * wavelength)
    ax.set_ylim(-1.35 * amplitude, 1.75 * amplitude)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"height $H(x)$")
    _phase_axis(ax, [j * wavelength / 2 for j in range(5)], ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"],
                r"phase $kx$: it reaches $2\pi$ when $x=\lambda$, so $k=2\pi/\lambda$")

    ax2.plot(x, np.cos(k * x), color=K_COLOR, lw=2, label=rf"$k={k:g}$")
    ax2.plot(x, np.cos(2 * k * x), color=K2_COLOR, lw=2, label=rf"$k={2 * k:g}$ (twice as large)")
    ax2.axhline(0, color="0.75", lw=0.8)
    _double_arrow(ax2, (0, 1.3), (wavelength, 1.3), color=PERIOD_COLOR)
    ax2.text(wavelength / 2, 1.35, r"$\lambda$", ha="center", va="bottom", color=PERIOD_COLOR)
    _double_arrow(ax2, (0, 1.9), (wavelength / 2, 1.9), color=PERIOD_COLOR)
    ax2.text(wavelength / 4, 1.95, r"$\lambda/2$", ha="center", va="bottom", color=PERIOD_COLOR)
    ax2.set_xlim(-0.55, 2.25 * wavelength)
    ax2.set_ylim(-1.3, 2.5)
    ax2.set_xlabel(r"position $x$")
    ax2.set_ylabel("height (A = 1)")
    ax2.set_title(r"a larger $k$ means a shorter wavelength: more oscillations per unit length", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def travelling_features_figure(amplitude=1.0, k=2.0, omega=3.0, dt=0.5, dx=0.6):
    """H(x, t) = A cos(kx - omega t): a crest moves in space as t advances, and arrives later at larger x."""
    v = omega / k
    fig, (ax_x, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.linspace(-0.2, 4.2, 1500)
    ax_x.plot(x, amplitude * np.cos(k * x), color="0.6", lw=2, label=r"at $t=0$")
    ax_x.plot(x, amplitude * np.cos(k * x - omega * dt), color=RE_COLOR, lw=2.2, label=rf"at $t={dt:g}$")
    crest0, crest1 = 0.0, v * dt
    crest0 = 2 * np.pi / k  # first crest to the right of the origin
    ax_x.plot([crest0], [amplitude], "o", color="0.4")
    ax_x.plot([crest0 + v * dt], [amplitude], "o", color=RE_COLOR)
    _double_arrow(ax_x, (crest0, 1.22 * amplitude), (crest0 + v * dt, 1.22 * amplitude), color=INK)
    ax_x.text(crest0 + v * dt / 2, 1.28 * amplitude, r"crest moves $\frac{\omega}{k}\,t$", color=INK, ha="center", fontsize=10)
    ax_x.axhline(0, color="0.8", lw=0.8)
    ax_x.set_ylim(-1.3 * amplitude, 1.75 * amplitude)
    ax_x.set_xlim(-0.2, 4.2)
    ax_x.set_xlabel(r"position $x$")
    ax_x.set_ylabel(r"height $H(x,t)$")
    ax_x.set_title("snapshots at two times: the pattern slides along $x$", fontsize=10)
    ax_x.legend(loc="lower right", fontsize=9)

    t = np.linspace(0, 6, 1500)
    ax_t.plot(t, amplitude * np.cos(-omega * t), color="0.6", lw=2, label=r"at $x=0$")
    ax_t.plot(t, amplitude * np.cos(k * dx - omega * t), color=RE_COLOR, lw=2.2, label=rf"at $x={dx:g}$")
    delay = k * dx / omega
    # first crests: x=0 at t=0, x=dx at t=delay (cos(k dx - omega t) = 1)
    ax_t.plot([0], [amplitude], "o", color="0.4")
    ax_t.plot([delay], [amplitude], "o", color=RE_COLOR)
    _double_arrow(ax_t, (0, 1.22 * amplitude), (delay, 1.22 * amplitude), color=INK)
    ax_t.text(delay / 2 + 0.05, 1.28 * amplitude, r"delay $\frac{k}{\omega}\,x$", color=INK, ha="left", fontsize=10)
    ax_t.axhline(0, color="0.8", lw=0.8)
    ax_t.set_ylim(-1.3 * amplitude, 1.75 * amplitude)
    ax_t.set_xlim(-0.2, 4.2)
    ax_t.set_xlabel(r"time $t$")
    ax_t.set_ylabel(r"height $H(x,t)$")
    ax_t.set_title("two fixed positions: the farther one oscillates later", fontsize=10)
    ax_t.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def phase_figure(theta=2.2, amplitude=1.5):
    """H(theta) = A cos(theta): the curve against the phase, and the same value as an arrow turning on a circle."""
    fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.9, 1]))
    th = np.linspace(0, 4 * np.pi, 1500)
    ax.plot(th, amplitude * np.cos(th), color=RE_COLOR, lw=2.2)
    ax.axhline(0, color="0.75", lw=0.8)
    for sign in (+1, -1):
        ax.axhline(sign * amplitude, color="0.7", ls=":", lw=1)
    _double_arrow(ax, (0.15, 0), (0.15, amplitude), color=A_COLOR)
    ax.text(0.03, amplitude * 0.62, r"$A$", color=A_COLOR, ha="right", va="center", fontsize=12)
    # the chosen phase and the height there
    h = amplitude * np.cos(theta)
    ax.plot([theta, theta], [0, h], color=PHASE_COLOR, ls="--", lw=1)
    ax.plot([0, theta], [h, h], color=H_COLOR, ls="--", lw=1)
    ax.plot([theta], [h], "o", color=H_COLOR, ms=8, zorder=5)
    ax.text(-0.1, h, r"$H(\theta)$", ha="right", va="center", fontsize=11, color=H_COLOR)
    ax.text(theta, 0.18 * amplitude * (-1 if h > 0 else 1), r"$\theta$", ha="center", va="top" if h > 0 else "bottom", fontsize=11, color=PHASE_COLOR)
    # landmarks of the cycle
    for x_c, label, y_c in [(0, "crest", amplitude), (np.pi, "trough", -amplitude), (2 * np.pi, "crest again", amplitude)]:
        ax.annotate(label, xy=(x_c, y_c), xytext=(x_c + 0.25, y_c + (0.32 if y_c > 0 else -0.45) * amplitude / 1.5),
                    fontsize=9, color="0.3", ha="left")
    ax.set_xticks([j * np.pi / 2 for j in range(9)])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$", r"$5\pi/2$", r"$3\pi$", r"$7\pi/2$", r"$4\pi$"])
    ax.set_xlim(-0.9, 4 * np.pi + 0.2)
    ax.set_ylim(-1.55 * amplitude, 1.6 * amplitude)
    ax.set_xlabel(r"phase $\theta$ (radians); the cosine repeats every $2\pi$", color=PHASE_COLOR)
    ax.tick_params(axis="x", colors=PHASE_COLOR)
    ax.set_ylabel(r"$H(\theta)=A\cos\theta$")

    circ = np.linspace(0, 2 * np.pi, 400)
    ax_c.plot(amplitude * np.cos(circ), amplitude * np.sin(circ), color="0.75", lw=1)
    ax_c.axhline(0, color="0.85", lw=0.8)
    ax_c.axvline(0, color="0.85", lw=0.8)
    tip = (amplitude * np.cos(theta), amplitude * np.sin(theta))
    ax_c.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.5))
    ax_c.plot([tip[0], tip[0]], [0, tip[1]], color="0.5", ls=":", lw=1)
    ax_c.plot([0, tip[0]], [0, 0], color=H_COLOR, lw=3.5, solid_capstyle="butt")
    arc = np.linspace(0, theta, 100) if theta > 0 else np.array([0.0])
    ax_c.plot(0.35 * amplitude * np.cos(arc), 0.35 * amplitude * np.sin(arc), color=PHASE_COLOR, lw=1.5)
    mid = theta / 2
    ax_c.text(0.5 * amplitude * np.cos(mid), 0.5 * amplitude * np.sin(mid), r"$\theta$", fontsize=11, ha="center", va="center", color=PHASE_COLOR)
    ax_c.text(tip[0] / 2, -0.2 * amplitude, r"$H=A\cos\theta$", ha="center", va="top", fontsize=10, color=H_COLOR)
    ax_c.set_xlim(-1.35 * amplitude, 1.35 * amplitude)
    ax_c.set_ylim(-1.35 * amplitude, 1.35 * amplitude)
    ax_c.set_aspect("equal")
    ax_c.set_title("arrow of length $A$ at angle $\\theta$", fontsize=10)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def wave_surface_figure(amplitude=1.5, k=1.5, omega=2.0, x_max=6.0, t_max=6.0, elev=34, azim=-50):
    """H(x, t) = A cos(kx - omega t) as a surface, with the t = 0 slice (space graph) and the x = 0 slice (time graph)."""
    x = np.linspace(0, x_max, 160)
    t = np.linspace(0, t_max, 160)
    xx, tt = np.meshgrid(x, t)
    hh = amplitude * np.cos(k * xx - omega * tt)
    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, tt, hh, cmap=_HEIGHT_CMAP, vmin=-2.5, vmax=2.5, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.9)
    # the t = 0 slice is the space graph, the x = 0 slice is the time graph
    ax.plot(x, np.zeros_like(x), amplitude * np.cos(k * x), color=K_COLOR, lw=3.2, label=r"$t=0$: height against $x$")
    ax.plot(np.zeros_like(t), t, amplitude * np.cos(-omega * t), color=OMEGA_COLOR, lw=3.2, label=r"$x=0$: height against $t$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, t_max)
    ax.set_zlim(-2.6, 2.6)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"time $t$")
    ax.set_zlabel(r"height $H(x,t)$")
    ax.set_box_aspect((1.2, 1.2, 0.8))
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(rf"$H=A\cos(kx-\omega t)$,  $A={amplitude:g}$, $k={k:g}$, $\omega={omega:g}$,  crest speed $\omega/k={omega / k:.2f}$", fontsize=10)
    return fig


def interference_schematic(x_det=2.0, wavelength=1.0, d=4.0, L=10.0, y_range=6.5):
    """Sketch of the two-hole experiment: wall, circular waves, distances r1 and r2, detector line and pattern."""
    from matplotlib.patches import Arc, Rectangle

    k = 2 * np.pi / wavelength
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-5.2, L + 7.5)
    ax.set_ylim(-y_range - 2.9, y_range + 1.0)

    y1, y2 = d / 2, -d / 2  # hole 1 above, hole 2 below
    gap = 0.3

    # incoming plane wave, to the left of the wall
    for j in range(1, 4):
        ax.plot([-1.4 * j * wavelength, -1.4 * j * wavelength], [-y_range + 0.8, y_range - 0.8], color=H_COLOR, lw=2, alpha=0.55)
    ax.annotate("", xy=(-2.0, 0), xytext=(-4.8, 0), arrowprops=dict(arrowstyle="-|>", color=H_COLOR, lw=2.2))
    ax.text(-3.4, 0.4, "incoming wave", color=H_COLOR, ha="center", fontsize=11)

    # circular waves from each hole (crests every wavelength)
    clip = Rectangle((0, -y_range), L + 0.6, 2 * y_range, transform=ax.transData, fill=False, visible=False)
    ax.add_patch(clip)
    for n in range(1, int((L + 1) / wavelength) + 2):
        for yc, col in [(y1, H_COLOR), (y2, H2_COLOR)]:
            arc = Arc((0, yc), 2 * n * wavelength, 2 * n * wavelength, theta1=-90, theta2=90, color=col, lw=1.6, alpha=0.8)
            ax.add_patch(arc)
            arc.set_clip_path(clip)

    # the wall with two holes
    wall = dict(facecolor=INK, edgecolor=INK)
    ax.add_patch(Rectangle((-0.18, y1 + gap), 0.36, y_range - (y1 + gap), **wall))
    ax.add_patch(Rectangle((-0.18, y2 + gap), 0.36, (y1 - gap) - (y2 + gap), **wall))
    ax.add_patch(Rectangle((-0.18, -y_range), 0.36, (y2 - gap) + y_range, **wall))
    ax.text(0.45, y1 + 0.35, "hole 1", color=INK, fontsize=10)
    ax.text(0.45, y2 - 0.7, "hole 2", color=INK, fontsize=10)

    # separation d
    ax.annotate("", xy=(-0.9, y1), xytext=(-0.9, y2), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(-1.15, -0.9, r"$d$", color=INK, ha="right", va="center", fontsize=12)
    ax.plot([-0.9, 0], [y1, y1], color=INK, lw=0.8, ls=":")
    ax.plot([-0.9, 0], [y2, y2], color=INK, lw=0.8, ls=":")

    # wavelength between two neighbouring crests along the axis of hole 1
    ax.annotate("", xy=(3 * wavelength, y1 + 1.0), xytext=(2 * wavelength, y1 + 1.0),
                arrowprops=dict(arrowstyle="<->", color=PERIOD_COLOR, lw=1.8))
    ax.text(2.5 * wavelength, y1 + 1.25, r"$\lambda$", color=PERIOD_COLOR, ha="center", fontsize=13)

    # detector line and the point where we look
    ax.plot([L, L], [-y_range, y_range], color=INK, lw=1.2, ls="--")
    ax.text(L, y_range + 0.3, "detector line", color=INK, ha="center", va="bottom", fontsize=10, zorder=10)
    ax.plot([L], [x_det], "o", color=INK, ms=8, zorder=6)
    ax.text(L + 0.3, x_det + 0.25, r"detector at $x$", color=INK, fontsize=10)

    # distances r1 and r2 from the holes to the detector
    r1 = np.hypot(L, x_det - y1)
    r2 = np.hypot(L, x_det - y2)
    ax.plot([0, L], [y1, x_det], color=R_COLOR, lw=2.6, zorder=5)
    ax.plot([0, L], [y2, x_det], color=R_COLOR, lw=2.6, zorder=5)
    ax.text(L * 0.62, y1 + (x_det - y1) * 0.62 + 0.55, r"$r_1$", color=R_COLOR, fontsize=15, weight="bold", ha="center")
    ax.text(L * 0.62, y2 + (x_det - y2) * 0.62 - 0.95, r"$r_2$", color=R_COLOR, fontsize=15, weight="bold", ha="center")

    # distance L to the detector line
    ax.annotate("", xy=(L, -y_range - 0.6), xytext=(0, -y_range - 0.6), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(L / 2, -y_range - 1.35, r"$L$", color=INK, ha="center", fontsize=12)

    # phase difference
    delta = k * (r2 - r1)
    ax.text(L / 2, -y_range - 2.2, rf"phase difference at the detector: $\delta = k(r_2-r_1) = {delta / (2 * np.pi):.2f}\times 2\pi$", color=PHASE_COLOR, fontsize=11, ha="center", va="top")

    # intensity pattern along the detector line, drawn sideways
    ys = np.linspace(-y_range, y_range, 700)
    q1 = np.hypot(L, ys - y1)
    q2 = np.hypot(L, ys - y2)
    inten = intensity(hole_amplitude(q1, k) + hole_amplitude(q2, k))
    base = L + 1.0
    scale = 1.35
    ax.plot(base + scale * inten, ys, color=INK, lw=2.2)
    ax.plot([base, base], [-y_range, y_range], color="0.75", lw=0.8)
    ax.plot([base + scale * 2, base + scale * 2], [-y_range, y_range], color="0.55", lw=1.2, ls="--")
    ax.text(base + scale * 2 + 0.15, y_range - 0.1, r"$I_1+I_2$", color="0.4", fontsize=9, va="top")
    ax.text(base + scale * 2.0, -y_range - 0.35, r"intensity $I_{12}$" "\n" r"along the detector line", color=INK, fontsize=10, ha="center", va="top")
    ax.plot([base], [x_det], "o", color=INK, ms=6)
    fig.tight_layout()
    return fig




def oscillator_energy_figure(amplitude=1.0, omega=2.0, mass=1.0, fixed_axes=False):
    """A harmonic oscillator H(t) = A cos(omega t): kinetic and potential energy trade off, the total is constant.

    With fixed_axes the time, height and energy axes do not depend on the sliders (A up to 2, omega up to 3)."""
    period = 2 * np.pi / omega
    t = np.linspace(0, 4 * np.pi if fixed_axes else 2 * period, 2400 if fixed_axes else 1200)
    h = amplitude * np.cos(omega * t)
    pot = 0.5 * mass * omega**2 * h**2
    kin = 0.5 * mass * (omega * amplitude * np.sin(omega * t)) ** 2
    total = 0.5 * mass * omega**2 * amplitude**2
    fig, (ax_h, ax_e) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))

    ax_h.plot(t, h, color=H_COLOR, lw=2.2)
    ax_h.axhline(0, color="0.75", lw=0.8)
    for sign in (+1, -1):
        ax_h.axhline(sign * amplitude, color="0.7", ls=":", lw=1)
    ax_h.set_ylim(*((-2.7, 2.7) if fixed_axes else (-1.35 * amplitude, 1.35 * amplitude)))
    ax_h.set_ylabel(r"height $H(t)=A\cos(\omega t)$")
    ax_h.set_title(rf"$A={amplitude:g}$, $\omega={omega:g}$, $m={mass:g}$", fontsize=10)

    ax_e.plot(t, pot, color=POT_COLOR, lw=2.2, label=r"potential $\frac{1}{2} m\omega^2H^2$")
    ax_e.plot(t, kin, color=KIN_COLOR, lw=2.2, label=r"kinetic $\frac{1}{2} m\dot H^2$")
    ax_e.axhline(total, color=INK, lw=2, ls="--", label=rf"total $=\frac{{1}}{{2}} m\omega^2A^2={total:.2f}$")
    ax_e.axhline(total / 2, color="0.6", lw=1, ls=":")
    ax_e.text(t[-1], total / 2, r"average of each $=\frac{1}{4} m\omega^2A^2$", ha="right", va="bottom", fontsize=9, color="0.3", bbox=dict(fc="white", ec="none", alpha=0.85))
    ax_e.set_ylim(0, 24.3 if fixed_axes else 1.35 * total)
    if fixed_axes:
        ax_e.set_xlim(0, t[-1])
    ax_e.set_xlabel(r"time $t$")
    ax_e.set_ylabel("energy")
    ax_e.legend(loc="upper right", fontsize=9, ncol=3, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig
