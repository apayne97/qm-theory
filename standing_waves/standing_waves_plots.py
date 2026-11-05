"""Figures for the standing-waves notebook. Each function returns a matplotlib Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmstyle import A_COLOR, H_COLOR, INK, OMEGA_COLOR, PROB_COLOR  # noqa: E402
from standing_waves import (  # noqa: E402
    box_energy_ev,
    box_wave,
    count_states_below,
    energy_units,
    levels,
    smooth_count,
)

# Positive values blue (the colour of the real part of psi), negative red, zero white.
WAVE_CMAP = LinearSegmentedColormap.from_list("wave", [A_COLOR, "#FFFFFF", H_COLOR])
PROB_CMAP = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", H_COLOR])


def line_figure(n):
    """The 1D standing wave with n half-wavelengths, its nodes, and its energy."""
    x = np.linspace(0, 1, 800)
    psi = np.sin(n * np.pi * x)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.axvspan(-0.08, 0, color="0.85")
    ax.axvspan(1, 1.08, color="0.85")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(x, psi, color=H_COLOR, lw=2.6)
    nodes = np.arange(1, n) / n
    ax.plot(nodes, np.zeros_like(nodes), "o", color=A_COLOR, ms=8, zorder=5)
    ax.plot([0, 1], [0, 0], "o", color=INK, ms=6, zorder=5)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([0, 1], ["0", r"$L$"])
    ax.set_yticks([])
    ax.set_title(
        rf"$n={n}$: {n} half-wavelength{'s' if n > 1 else ''} fit, {n - 1} node{'s' if n != 2 else ''} inside; "
        rf"$E = {n * n}\,E_1$  ({float(box_energy_ev(n, 1.0)):.2f} eV for $L=1$ nm)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def box2d_figure(nx, ny, ratio=1.0, resolution=300):
    """Signed wave (left) and probability density (right) of the (nx, ny) state of a box of sides 1 x ratio."""
    Ls = (1.0, float(ratio))
    x = np.linspace(0, Ls[0], resolution)[None, :]
    y = np.linspace(0, Ls[1], int(resolution * ratio))[:, None]
    psi = box_wave((nx, ny), (x, y), Ls)
    top = np.max(np.abs(psi))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9 * max(ratio, 0.6) / 1.0 + 0.4))
    ext = [0, Ls[0], 0, Ls[1]]
    axes[0].imshow(psi, origin="lower", extent=ext, cmap=WAVE_CMAP, vmin=-top, vmax=top, aspect="equal")
    axes[0].contour(x[0], y[:, 0], psi, levels=[0], colors=[INK], linewidths=1.2)
    axes[0].set_title(r"wave $\psi$ (blue +, red −); black lines are nodes", fontsize=9)
    axes[1].imshow(psi**2, origin="lower", extent=ext, cmap=PROB_CMAP, aspect="equal")
    axes[1].set_title(r"probability density $|\psi|^2$", fontsize=9)
    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    e = energy_units((nx, ny), Ls)
    fig.suptitle(
        rf"$(n_x,n_y)=({nx},{ny})$: {nx - 1} node line{'s' if nx != 2 else ''} across $x$, "
        rf"{ny - 1} across $y$;  $E={e:.2f}\,E_1$",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def kspace_figure(ratio, nx, ny, nmax=7):
    """The allowed wave vectors (k_x, k_y) = pi (n_x / L_x, n_y / L_y) form a lattice; energy is the squared distance."""
    Ls = (1.0, float(ratio))
    kx = lambda n: n / Ls[0]  # noqa: E731  (in units of pi)
    ky = lambda n: n / Ls[1]  # noqa: E731
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for a in range(1, nmax + 1):
        for b in range(1, nmax + 1):
            ax.plot(kx(a), ky(b), "o", color="0.65", ms=5)
    # states with the same energy as the chosen one
    e0 = energy_units((nx, ny), Ls)
    for e, states in levels(Ls, 60):
        if abs(e - e0) < 1e-9:
            for a, b in states:
                ax.plot(kx(a), ky(b), "o", color=A_COLOR, ms=9, zorder=5)
    r = np.sqrt(e0)
    t = np.linspace(0, np.pi / 2, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), color=A_COLOR, lw=1.2, ls="--")
    ax.plot(kx(nx), ky(ny), "o", mfc="none", mec=INK, ms=14, zorder=6)
    ax.set_xlim(0, nmax / Ls[0] + 0.5)
    ax.set_ylim(0, nmax / Ls[1] + 0.5)
    lim = min(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (in units of $\pi/L_x$)")
    ax.set_ylabel(r"$k_y$ (in units of $\pi/L_x$)")
    same = sum(len(s) for e, s in levels(Ls, 60) if abs(e - e0) < 1e-9)
    ax.set_title(f"Energy = squared distance from the corner; {same} state{'s' if same != 1 else ''} on the red circle",
                 fontsize=9)
    fig.tight_layout()
    return fig


def aspect_fan_figure(n_states=10):
    """Energies of the lowest (nx, ny) states of a 1 x ratio box, as the ratio of the sides changes."""
    ratios = np.linspace(0.75, 1.5, 300)
    states = sorted(((a, b) for a in range(1, 6) for b in range(1, 6)), key=lambda s: energy_units(s, (1.0, 1.0)))[:n_states]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for a, b in states:
        e = a**2 + (b / ratios) ** 2
        ax.plot(ratios, e, color=H_COLOR if a != b else OMEGA_COLOR, lw=1.6)
        ax.text(ratios[-1] + 0.01, e[-1], f"({a},{b})", fontsize=7, va="center")
    ax.axvline(1.0, color=A_COLOR, lw=1.2, ls="--")
    ax.set_xlim(0.75, 1.58)
    ax.set_ylim(0, 25)
    ax.set_xlabel(r"side ratio $L_y/L_x$  (dashed: the square)")
    ax.set_ylabel(r"energy ($E_1$ of side $L_x$)")
    ax.set_title("Degenerate pairs at the square split as soon as the sides differ", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def slices_figure(nx, ny, nz, resolution=200):
    """Three slices of a 3D standing wave in a unit cube, through the point (0.31, 0.43, 0.37) of the box."""
    Ls = (1.0, 1.0, 1.0)
    p = (0.31, 0.43, 0.37)
    u = np.linspace(0, 1, resolution)
    a, b = np.meshgrid(u, u, indexing="xy")
    slices = [
        (box_wave((nx, ny, nz), (a, b, p[2]), Ls), "x", "y", rf"$z={p[2]}L$"),
        (box_wave((nx, ny, nz), (a, p[1], b), Ls), "x", "z", rf"$y={p[1]}L$"),
        (box_wave((nx, ny, nz), (p[0], a, b), Ls), "y", "z", rf"$x={p[0]}L$"),
    ]
    top = max(np.max(np.abs(s[0])) for s in slices)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    for ax, (w, xl, yl, title) in zip(axes, slices):
        ax.imshow(w, origin="lower", extent=[0, 1, 0, 1], cmap=WAVE_CMAP, vmin=-top, vmax=top)
        ax.contour(u, u, w, levels=[0], colors=[INK], linewidths=1.0)
        ax.set_xlabel(f"${xl}$")
        ax.set_ylabel(f"${yl}$")
        ax.set_title(title, fontsize=9)
    e = energy_units((nx, ny, nz), Ls)
    fig.suptitle(rf"$(n_x,n_y,n_z)=({nx},{ny},{nz})$:  $E={e:.0f}\,E_1$   (slices of the wave; black = nodes)", fontsize=10)
    fig.tight_layout()
    return fig


def level_diagram_figure(dim, e_max):
    """Level diagram of a line, square or cube: one bar per level, with the number of states beside it."""
    Ls = (1.0,) * dim
    lv = levels(Ls, e_max)
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    for e, states in lv:
        ax.hlines(e, 0.2, 0.8, color=H_COLOR if len(states) == 1 else A_COLOR, lw=2.5)
        ax.text(0.85, e, f"×{len(states)}" if len(states) > 1 else "", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, e_max + 1)
    ax.set_xticks([])
    ax.set_ylabel(r"energy ($E_1$)")
    ax.set_title({1: "line", 2: "square", 3: "cube"}[dim] + "  (red: more than one state)", fontsize=9)
    fig.tight_layout()
    return fig


def counting_figure(e_max=120):
    """Number of states below energy E in 1, 2 and 3 dimensions (cube sides all equal)."""
    es = np.linspace(0.5, e_max, 600)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for dim, color in [(1, H_COLOR), (2, OMEGA_COLOR), (3, A_COLOR)]:
        exact = [count_states_below((1.0,) * dim, e) for e in es]
        ax.step(es, exact, where="post", color=color, lw=1.8, label=f"{dim}D (exact count)")
        ax.plot(es, smooth_count(dim, es), color=color, lw=1, ls=":")
    ax.set_xlabel(r"energy ($E_1$)")
    ax.set_ylabel("number of states below this energy")
    ax.set_yscale("log")
    ax.set_title("More dimensions, more states at every energy (dotted: smooth estimate)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig
