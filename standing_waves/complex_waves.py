"""Complex and real waves around a ring: the pure functions and the figure for the notebook.

A stationary state around the ring is  psi(phi, t) = f(phi) * exp(-i theta),  with theta = E t / hbar. Two choices of
f(phi) with the same energy are compared:

  * the complex wave  exp(i m phi)  : travels round the ring, its phase turns, |psi|^2 is the same everywhere;
  * the real wave     cos(m phi)    : stands still, with 2|m| nodes round the ring (|m| nodal planes in 3D).

Colours come from qmstyle.py: the real part is blue, the imaginary part green, the phase purple, |psi|^2 pale blue.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmstyle import A_COLOR, H_COLOR, IM_COLOR, INK, PROB_COLOR  # noqa: E402


def complex_wave(m, phi, theta=0.0):
    """exp(i (m phi - theta)): the travelling wave e^{i m phi} times the overall phase e^{-i theta}."""
    return np.exp(1j * (m * np.asarray(phi, dtype=float) - theta))


def real_wave(m, phi, theta=0.0):
    """cos(m phi) * exp(-i theta): the standing wave times the same overall phase."""
    return np.cos(m * np.asarray(phi, dtype=float)) * np.exp(-1j * theta)


def real_combination(m, phi):
    """(e^{i m phi} + e^{-i m phi}) / 2, the sum of the two degenerate waves +m and -m. Equal to cos(m phi)."""
    return 0.5 * (complex_wave(m, phi) + complex_wave(-m, phi))


def cos_nodes(m):
    """Azimuths in [0, 2 pi) where cos(m phi) = 0: pi/(2|m|) + k pi/|m|, k = 0 .. 2|m|-1 (empty for m = 0)."""
    m = abs(int(m))
    if m == 0:
        return np.array([])
    return np.pi / (2 * m) + np.arange(2 * m) * np.pi / m


def phase_winding(m, n=720):
    """How many times the phase of exp(i m phi) turns round while phi goes once round the ring (counted with sign)."""
    phi = np.linspace(0, 2 * np.pi, n + 1)
    step = np.diff(np.unwrap(np.angle(complex_wave(m, phi))))
    return float(step.sum() / (2 * np.pi))


def angular_momentum_z(m):
    """L_z / hbar of exp(i m phi): -i d/dphi gives m. The real wave cos(m phi) is a mix of +m and -m (no single value)."""
    return int(m)


def _ring(ax, values, cmap, vmin, vmax, lw):
    t = np.linspace(0, 2 * np.pi, 361)
    rad = np.array([1 - lw / 2, 1 + lw / 2])
    tt, rr = np.meshgrid(t, rad)
    ax.pcolormesh(rr * np.cos(tt), rr * np.sin(tt), values(0.5 * (t[:-1] + t[1:]))[None, :],
                  cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def complex_vs_real_figure(m=2, theta_deg=0.0):
    """Left: the complex wave e^{i m phi}, right: the real wave cos(m phi), both times e^{-i theta}.

    Top row: the ring (left coloured by the phase of psi, right by the real part). Bottom row: Re psi, Im psi and
    |psi|^2 against the azimuth. Changing theta moves the left curves round and leaves the right nodes where they are;
    |psi|^2 never changes.
    """
    theta = np.radians(theta_deg)
    phi = np.linspace(0, 2 * np.pi, 721)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white",
                         "axes.facecolor": "white", "mathtext.fontset": "dejavusans"}):
        fig = plt.figure(figsize=(9, 6.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.28, wspace=0.18)
        for col, (wave, title) in enumerate([
            (complex_wave, rf"complex: $e^{{i m_\ell\varphi}}$ (travels round)"),
            (real_wave, rf"real: $\cos(m_\ell\varphi)$ (stands still)"),
        ]):
            ax = fig.add_subplot(gs[0, col])
            if col == 0:
                _ring(ax, lambda t: np.angle(wave(m, t, theta)), "twilight", -np.pi, np.pi, 0.22)
                ax.set_title(title + "\nring coloured by the phase of $\\psi$")
            else:
                _ring(ax, lambda t: wave(m, t, theta).real, "coolwarm", -1, 1, 0.22)
                ax.set_title(title + "\nring coloured by $\\mathrm{Re}\\,\\psi$ (blue +, red $-$)")
                for p in cos_nodes(m):
                    ax.plot([0.8 * np.cos(p), 1.2 * np.cos(p)], [0.8 * np.sin(p), 1.2 * np.sin(p)],
                            color=INK, lw=2, solid_capstyle="round")
            ax.text(0, 0, rf"$m_\ell={m}$", ha="center", va="center", color=INK, fontsize=12)

            bx = fig.add_subplot(gs[1, col])
            v = wave(m, phi, theta)
            bx.fill_between(phi, np.abs(v) ** 2, color=PROB_COLOR, alpha=0.8, label=r"$|\psi|^2$")
            bx.plot(phi, v.real, color=H_COLOR, lw=2.2, label=r"$\mathrm{Re}\,\psi$")
            bx.plot(phi, v.imag, color=IM_COLOR, lw=2.2, label=r"$\mathrm{Im}\,\psi$")
            if col == 1:
                bx.plot(cos_nodes(m), np.zeros(2 * abs(m)), "o", color=A_COLOR, ms=6, zorder=5)
            bx.axhline(0, color="0.6", lw=0.8)
            bx.set_xlim(0, 2 * np.pi)
            bx.set_ylim(-1.15, 1.15)
            bx.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
            bx.set_xlabel(r"azimuth $\varphi$")
            for side in ("top", "right"):
                bx.spines[side].set_visible(False)
            if col == 0:
                bx.legend(loc="lower left", ncol=3, frameon=False, fontsize=9)
    return fig
