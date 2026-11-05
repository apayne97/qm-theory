"""Figures for the spherical-harmonics and hydrogen notebook. Each function returns a matplotlib Figure.

Colours come from qmstyle.py. Positive values of a wave are blue and negative values red, as in the box notebook.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydrogen import (  # noqa: E402
    cross_section_plane,
    effective_potential,
    hydrogen_energy,
    hydrogen_energy_ev,
    mean_radius,
    nodal_structure,
    numerical_levels,
    orbital_name,
    polar_pole_coefficient,
    polar_solution,
    psi_real,
    radial_R,
    radial_u,
    real_ylm,
    sample_density,
    solve_radial,
)
from qmstyle import (  # noqa: E402
    A_COLOR,
    H_COLOR,
    INK,
    KIN_COLOR,
    OMEGA_COLOR,
    POT_COLOR,
    PROB_COLOR,
    R_COLOR,
)
from standing_waves_plots import PROB_CMAP, WAVE_CMAP  # noqa: E402

PLUS, MINUS = H_COLOR, A_COLOR


def _shape_axes(ax, lim=1.0):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()


def _draw_shape(ax, l, m, n_theta=90, n_phi=180):
    """Polar plot of a real harmonic: the distance from the origin is |Y|, blue where Y > 0 and red where Y < 0."""
    theta = np.linspace(0, np.pi, n_theta)[:, None]
    phi = np.linspace(0, 2 * np.pi, n_phi)[None, :]
    y = real_ylm(l, m, theta, phi) * np.ones_like(phi)
    r = np.abs(y)
    x, yy, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta) * np.ones_like(phi)
    from matplotlib.colors import to_rgba

    face = np.where((y >= 0)[..., None], np.array(to_rgba(PLUS)), np.array(to_rgba(MINUS)))
    ax.plot_surface(x, yy, z, facecolors=face[:-1, :-1], rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    _shape_axes(ax, 0.78 * float(r.max()))
    return float(r.max())


def coordinates_figure():
    """The three spherical coordinates of one point: distance r, polar angle theta from the z axis, azimuth phi."""
    r, th, ph = 1.0, np.radians(52), np.radians(35)
    p = r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    foot = np.array([p[0], p[1], 0.0])
    fig = plt.figure(figsize=(4.2, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    for d, lab in zip(np.eye(3), ("x", "y", "z")):
        ax.plot(*zip(np.zeros(3), 1.1 * d), color="0.45", lw=1)
        ax.text(*(1.17 * d), f"${lab}$", color=INK, ha="center", va="center")
    ax.plot(*zip(np.zeros(3), p), color=R_COLOR, lw=2.6)
    ax.plot(*zip(p, foot), color="0.5", lw=1, ls="--")
    ax.plot(*zip(np.zeros(3), foot), color="0.5", lw=1, ls="--")
    ax.scatter(*p, color=A_COLOR, s=45, zorder=5)
    t = np.linspace(0, th, 40)
    ax.plot(0.45 * np.sin(t) * np.cos(ph), 0.45 * np.sin(t) * np.sin(ph), 0.45 * np.cos(t), color=INK, lw=1.4)
    f = np.linspace(0, ph, 40)
    ax.plot(0.4 * np.cos(f), 0.4 * np.sin(f), 0 * f, color=INK, lw=1.4)
    ax.text(0.2, 0.0, 0.62, r"$\vartheta$", color=INK, fontsize=13)
    ax.text(0.5, 0.1, -0.02, r"$\varphi$", color=INK, fontsize=13)
    ax.text(*(0.5 * p + np.array([-0.12, 0.1, 0.0])), r"$r$", color=R_COLOR, fontsize=13)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_zlim(0, 1.2)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-58)
    fig.tight_layout()
    return fig


def harmonic_figure(l, m):
    """A real spherical harmonic as a shape (left) and unrolled onto a map (right) with its node lines."""
    through, latitudes = nodal_structure(l, m)
    fig = plt.figure(figsize=(9.6, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_shape(ax, l, m)
    ax.view_init(elev=20, azim=-60)
    ax.set_title("shape: distance from centre $=|Y|$\n(blue $Y>0$, red $Y<0$)", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    theta = np.linspace(0, np.pi, 240)[:, None]
    phi = np.linspace(0, 2 * np.pi, 480)[None, :]
    y = real_ylm(l, m, theta, phi) * np.ones_like(phi)
    top = max(np.abs(y).max(), 1e-12)
    ax2.imshow(y, extent=[0, 360, 180, 0], cmap=WAVE_CMAP, vmin=-top, vmax=top, aspect="auto")
    if l > 0:
        ax2.contour(np.degrees(phi[0]), np.degrees(theta[3:-3, 0]), y[3:-3], levels=[0], colors=[INK], linewidths=1.3)
    ax2.set_xlabel(r"azimuth $\varphi$ (degrees)")
    ax2.set_ylabel(r"polar angle $\vartheta$ (degrees)")
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_title(
        rf"$\ell={l},\ m_\ell={m}$: {l} nodal line{'s' if l != 1 else ''} = {through} through the poles + "
        rf"{latitudes} of latitude" "\n" rf"$\ell(\ell+1)={l * (l + 1)}$",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def harmonic_gallery_figure(l_max=3):
    """All real harmonics up to l_max in rows (s, p, d, f), with the m values of each row side by side."""
    cols = 2 * l_max + 1
    fig = plt.figure(figsize=(1.55 * cols, 1.65 * (l_max + 1)))
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ax = fig.add_subplot(l_max + 1, cols, l * cols + (m + l_max) + 1, projection="3d")
            _draw_shape(ax, l, m, 60, 120)
            ax.view_init(elev=22, azim=-55)
            ax.set_title(f"$m_\\ell={m}$", fontsize=8, pad=-6)
        fig.text(0.005, 1 - (l + 0.5) / (l_max + 1), f"$\\ell={l}$\n{'spdf'[l]}", fontsize=10, va="center", color=INK)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.01, wspace=-0.12, hspace=0.05)
    return fig


def pole_figure(ell):
    """Why l is a whole number. Left: the polar-angle wave (m = 0) for the chosen l, even and odd start, against
    the polar angle on a logarithmic axis. Right: the coefficient c of ln(theta) near the pole against l."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    for parity, color, label in (("even", H_COLOR, "even about the equator"), ("odd", OMEGA_COLOR, "odd about the equator")):
        t, y = polar_solution(ell, parity)
        ax.plot(t, y, color=color, lw=2.2, label=label)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(-4, 4)
    ax.invert_xaxis()
    ax.set_xlabel(r"polar angle $\vartheta$ (radians, toward the pole at $0$)")
    ax.set_ylabel(r"wave $y(\vartheta)$ ($m_\ell=0$)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(rf"$\ell={ell:g}$: a wave that keeps growing near the pole is not allowed", fontsize=9)
    ax = axes[1]
    grid = np.linspace(0, 5, 251)
    ax.plot(grid, polar_pole_coefficient(grid, "even"), color=H_COLOR, lw=2, label="even start")
    ax.plot(grid, polar_pole_coefficient(grid, "odd"), color=OMEGA_COLOR, lw=2, label="odd start")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(ell, color=A_COLOR, lw=1.4, ls="--")
    ax.plot(np.arange(0, 6), np.zeros(6), "o", color=INK, ms=5, zorder=5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"growth near the pole, $c$")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(r"each curve reaches $c=0$ only at whole $\ell$ (dots)", fontsize=9)
    fig.tight_layout()
    return fig


def veff_figure():
    """Effective potential for l = 0, 1, 2 with the hydrogen levels."""
    r = np.linspace(0.12, 24, 1500)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for l, ls in zip((0, 1, 2), ("-", "--", ":")):
        ax.plot(r, np.clip(effective_potential(r, l), -1.2, 1.0), color=POT_COLOR, lw=2, ls=ls, label=rf"$\ell={l}$")
    for n in range(1, 6):
        ax.axhline(hydrogen_energy(n), color=OMEGA_COLOR, lw=1)
        if n <= 3:
            ax.text(23.8, hydrogen_energy(n) + 0.012, f"$n={n}$", color=INK, fontsize=8, ha="right")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.75, 0.2)
    ax.set_xlabel(r"distance $r$ (bohr)")
    ax.set_ylabel("energy (hartree)")
    ax.set_title(r"effective potential $-1/r+\ell(\ell+1)/(2r^2)$ and the levels $E_n=-1/(2n^2)$", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def radial_state_figure(n, l):
    """Left: the effective potential, the level, and the radial wave drawn at its energy (numerical, with the exact
    one on top). Right: the difference between numerical and exact radial waves."""
    r, e, v = solve_radial(l)
    k = n - l - 1
    u_num = v[:, k]
    u_ex = radial_u(n, l, r)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rmax = 2.2 * n * n + 4
    mask = r < rmax
    scale = 0.12 / np.abs(u_ex).max()  # height of the wave drawn on the energy axis, in hartree
    ax.plot(r[mask], np.clip(effective_potential(r[mask], l), -1.5, 1.0), color=POT_COLOR, lw=2, label="effective potential")
    ax.axhline(e[k], color=OMEGA_COLOR, lw=1.2, label=rf"numerical $E={e[k]:.5f}$ hartree")
    ax.plot(r[mask], e[k] + scale * u_num[mask], color=H_COLOR, lw=2.4, label=r"radial wave $u=rR$ (numerical)")
    ax.plot(r[mask][::25], e[k] + scale * u_ex[mask][::25], "o", color=A_COLOR, ms=3.5, label="exact")
    ax.axhline(0, color="black", lw=0.7)
    nodes = r[:-1][(u_num[:-1] * u_num[1:] < 0)]
    ax.plot(nodes, np.full_like(nodes, e[k]), "o", color=INK, ms=6, zorder=6)
    ax.set_xlim(0, rmax)
    lo = min(-0.6, e[k] - 0.2) if n == 1 else -0.3
    ax.set_ylim(max(lo, -0.7), 0.15)
    ax.set_xlabel(r"distance $r$ (bohr)")
    ax.set_ylabel("energy (hartree)")
    ax.set_title(
        rf"$n={n},\ \ell={l}$: {n - l - 1} radial node{'s' if n - l - 1 != 1 else ''} (black dots);  "
        rf"exact $E=-1/(2n^2)={hydrogen_energy(n):.5f}$ ({hydrogen_energy_ev(n):.3f} eV)",
        fontsize=8.5,
    )
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    ax.plot(r[mask], (u_num - u_ex)[mask], color=KIN_COLOR, lw=1.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel(r"distance $r$ (bohr)")
    ax.set_title(rf"numerical $-$ exact $u$ (largest: {np.abs(u_num - u_ex).max():.1e}; $u$ peaks near {np.abs(u_ex).max():.2f})", fontsize=8.5)
    fig.tight_layout()
    return fig


def level_check_figure(n_max=4):
    """Finite-difference energies against -1/(2 n^2): the points sit on the exact levels, whatever l is."""
    levels = numerical_levels(n_max - 1, n_max)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for n in range(1, n_max + 1):
        ax.hlines(hydrogen_energy(n), -0.4, n_max - 0.6, color=OMEGA_COLOR, lw=1)
        ax.text(n_max - 0.55, hydrogen_energy(n), rf"$n={n}$", va="center", fontsize=8, color=INK)
    for (n, l), e in levels.items():
        ax.plot(l, e, "o", color=H_COLOR, ms=7)
    ax.set_xticks(range(n_max), [f"$\\ell={l}$ ({'spdf'[l]})" for l in range(n_max)])
    ax.set_ylabel("energy (hartree)")
    ax.set_xlim(-0.4, n_max - 0.1)
    ax.set_title("dots: numerical radial equation; lines: $-1/(2n^2)$", fontsize=10)
    fig.tight_layout()
    return fig


def _extent(n):
    return 2.2 * n * n + 2 * n + 1


def orbital_figure(n, l, m, n_points=3500):
    """One real orbital: signed cross-section, density cross-section, sampled cloud, and radial probability."""
    ext = _extent(n)
    s, z, x, y = cross_section_plane(m, ext, 420)
    psi = psi_real(n, l, m, x, y, z)
    top = max(np.abs(psi).max(), 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    box = [-ext, ext, -ext, ext]
    axes[0].imshow(psi, origin="lower", extent=box, cmap=WAVE_CMAP, vmin=-top, vmax=top)
    axes[0].contour(s[0], z[:, 0], psi, levels=[0], colors=[INK], linewidths=1.0)
    axes[0].set_title(r"wave $\psi$ in a plane through the axis" "\n" "(blue +, red −, black = nodes)", fontsize=8.5)
    axes[1].imshow(psi**2, origin="lower", extent=box, cmap=PROB_CMAP, norm=PowerNorm(0.5, vmin=0, vmax=top**2))
    axes[1].set_title(r"density $|\psi|^2$, same plane" "\n" "(square-root shading)", fontsize=8.5)
    pts = sample_density(n, l, m, n_points, ext, seed=3)
    phi0 = 0.0 if m >= 0 else np.pi / (2 * abs(m))
    # side view: horizontal axis is the in-plane direction, depth is ignored
    horiz = pts[:, 0] * np.cos(phi0) + pts[:, 1] * np.sin(phi0)
    sign = psi_real(n, l, m, *pts.T) > 0
    axes[2].scatter(horiz[sign], pts[sign, 2], s=1.2, color=PLUS, alpha=0.5, linewidths=0)
    axes[2].scatter(horiz[~sign], pts[~sign, 2], s=1.2, color=MINUS, alpha=0.5, linewidths=0)
    axes[2].set_xlim(-ext, ext)
    axes[2].set_ylim(-ext, ext)
    axes[2].set_title(f"{n_points} points drawn with probability $|\\psi|^2$\n(side view, all depths)", fontsize=8.5)
    for ax in axes[:3]:
        ax.set_aspect("equal")
        ax.set_xlabel(r"in-plane distance from axis (bohr)")
    axes[0].set_ylabel(r"$z$ (bohr)")
    rr = np.linspace(0, ext * 1.1, 800)
    ax = axes[3]
    ax.fill_between(rr, radial_u(n, l, rr) ** 2, color=PROB_COLOR, alpha=0.8)
    ax.plot(rr, radial_u(n, l, rr) ** 2, color=R_COLOR, lw=2)
    ax.axvline(mean_radius(n, l), color=A_COLOR, ls="--", lw=1.2, label=rf"mean $\langle r\rangle={mean_radius(n, l):g}$")
    ax.set_xlabel(r"distance $r$ (bohr)")
    ax.set_title(r"probability per bohr, $r^2R^2$" "\n" rf"{n - l - 1} radial node{'s' if n - l - 1 != 1 else ''}", fontsize=8.5)
    ax.legend(fontsize=8)
    ax.set_xlim(0, ext * 1.1)
    fig.suptitle(
        rf"{orbital_name(n, l, m)}: $n={n},\ \ell={l},\ m_\ell={m}$;  $E=-1/(2n^2)={hydrogen_energy(n):.4f}$ hartree "
        rf"$={hydrogen_energy_ev(n):.2f}$ eV",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig


GALLERY = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0), (3, 2, 0)]


def orbital_gallery_figure():
    """1s, 2s, 2p, 3s, 3p, 3d_z2: wave in a plane through the axis (top) and radial probability (bottom)."""
    fig, axes = plt.subplots(2, 6, figsize=(14.0, 5.0), gridspec_kw={"height_ratios": [1.6, 1]})
    for j, (n, l, m) in enumerate(GALLERY):
        ext = _extent(n)
        s, z, x, y = cross_section_plane(m, ext, 360)
        psi = psi_real(n, l, m, x, y, z)
        top = np.abs(psi).max()
        ax = axes[0, j]
        ax.imshow(np.sign(psi) * np.abs(psi) ** 0.5, origin="lower", extent=[-ext, ext, -ext, ext],
                  cmap=WAVE_CMAP, vmin=-top**0.5, vmax=top**0.5)
        ax.contour(s[0], z[:, 0], psi, levels=[0], colors=[INK], linewidths=0.8)
        ax.set_title(f"{orbital_name(n, l, m)}  (" + (f"{n - l - 1} radial, {l} angular nodes") + ")", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        rr = np.linspace(0, ext * 1.1, 600)
        axb = axes[1, j]
        axb.fill_between(rr, radial_u(n, l, rr) ** 2, color=PROB_COLOR, alpha=0.8)
        axb.plot(rr, radial_u(n, l, rr) ** 2, color=R_COLOR, lw=1.8)
        axb.set_xlabel("$r$ (bohr)", fontsize=8)
        axb.set_yticks([])
        axb.set_xlim(0, ext * 1.1)
        axb.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(r"$\psi$ (plane through the $z$ axis)", fontsize=8)
    axes[1, 0].set_ylabel(r"$r^2R^2$", fontsize=8)
    fig.suptitle("Colour shows the signed wave (square-root scale so outer lobes are visible); each panel has its own length scale",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    return fig
