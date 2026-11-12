"""Figures for the spherical-harmonics and hydrogen notebook. Each function returns a matplotlib Figure.

Colours come from qmstyle.py. Positive values of a wave are blue and negative values red, as in the box notebook.
"""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordinates import VIEW_AZIM, VIEW_ELEV  # noqa: E402
from hydrogen import (  # noqa: E402
    azimuth_nodes,
    latitude_nodes_predicted,
    legendre_assoc,
    ylm_norm,
    circle_coefficients,
    circle_function,
    circle_reconstruct,
    sphere_coefficients,
    sphere_function,
    sphere_power,
    sphere_reconstruct,
    bessel_j,
    bessel_zeros,
    drum_frequency,
    drum_mode,
    drum_nodes,
    node_geometry,
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
    ANGSTROM_PER_BOHR as AB,
    ANGULAR_COLOR,
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
    """The three spherical coordinates of one point: distance r, polar angle theta from the z axis, azimuth phi. The same
    figure, in the same view and colours, as the slide figure of coordinates_plots.py."""
    from coordinates_plots import spherical_coordinates_figure

    return spherical_coordinates_figure(slide=False)


def harmonic_figure(l, m):
    """A real spherical harmonic as a shape (left) and unrolled onto a map (right) with the lines where its angular nodes
    (planes and cones) meet the sphere."""
    through, latitudes = nodal_structure(l, m)
    fig = plt.figure(figsize=(9.6, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_shape(ax, l, m)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
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
        rf"$\ell={l},\ m_\ell={m}$: $\ell={l}$ angular node{'s' if l != 1 else ''} = $n_\varphi={through}$ plane{'s' if through != 1 else ''} + "
        rf"$n_\vartheta={latitudes}$ cone{'s' if latitudes != 1 else ''}" "\n" rf"$\ell(\ell+1)={l * (l + 1)}$",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def node_placement_figure(l, m_abs, kind="cos"):
    """Where the nodes of a real harmonic sit. Left: the sphere with its nodes. Middle: the same as a map, the
    nodes found from the wave (solid white) and the predicted positions (dashed colour). Right: the polar part
    P_l^|m|(cos theta) against theta, with its zeros (the polar nodes, which are cones) marked."""
    m_abs = min(abs(m_abs), l)
    m = m_abs if (kind == "cos" or m_abs == 0) else -m_abs
    lat = latitude_nodes_predicted(l, m_abs)
    azi = azimuth_nodes(m_abs, "cos" if m >= 0 else "sin")
    fig = plt.figure(figsize=(12.6, 3.9))
    elev, azim = VIEW_ELEV, VIEW_AZIM
    # sphere
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    th = np.linspace(0, np.pi, 140)[:, None]
    ph = np.linspace(0, 2 * np.pi, 280)[None, :]
    y = real_ylm(l, m, th, ph) * np.ones_like(ph)
    from matplotlib.colors import to_rgba

    face = np.where((y >= 0)[..., None], np.array(to_rgba("#9db8f0")), np.array(to_rgba("#f2a79a")))
    for t0 in lat:  # paint the nodes onto the surface itself (3D line artists would be hidden behind it)
        face[np.abs(th - t0)[:, 0] < 0.018] = to_rgba(POLAR_NODE)
    for p0 in azi:
        face[np.abs(np.sin(th) * np.sin(ph - p0)) < 0.014] = to_rgba(AZIMUTH_NODE)
    ax.plot_surface(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th) * np.ones_like(ph),
                    facecolors=face[:-1, :-1], rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.plot([0, 0], [0, 0], [1.0, 1.25], color="0.35", lw=1)
    ax.text(0, 0, 1.33, "$z$", ha="center", color=INK)
    _shape_axes(ax, 0.8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("the sphere (blue $Y>0$, red $Y<0$)", fontsize=9, y=0.98)
    # map
    ax2 = fig.add_subplot(1, 3, 2)
    th2 = np.linspace(0, np.pi, 300)[:, None]
    ph2 = np.linspace(0, 2 * np.pi, 600)[None, :]
    y2 = real_ylm(l, m, th2, ph2) * np.ones_like(ph2)
    top = max(np.abs(y2).max(), 1e-12)
    ax2.imshow(y2, extent=[0, 360, 180, 0], cmap=WAVE_CMAP, vmin=-top, vmax=top, aspect="auto")
    if l > 0:
        ax2.contour(np.degrees(ph2[0]), np.degrees(th2[3:-3, 0]), y2[3:-3], levels=[0], colors=["white"], linewidths=4.5)
    for t0 in lat:
        ax2.axhline(np.degrees(t0), color=POLAR_NODE, ls="--", lw=1.6)
    for p0 in azi:
        for q in (p0, p0 + np.pi):
            ax2.axvline(np.degrees(q), color=AZIMUTH_NODE, ls="--", lw=1.6)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(180, 0)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_xlabel(r"azimuth $\varphi$ (degrees)")
    ax2.set_ylabel(r"polar angle $\vartheta$ (degrees)")
    trig = "cos" if m >= 0 else "sin"
    ax2.set_title(rf"white: zeros of the wave; dashed: predicted" "\n"
                  rf"$n_\varphi={m_abs}$ plane{'s' if m_abs != 1 else ''} (green) + $n_\vartheta={l - m_abs}$ cone{'s' if l - m_abs != 1 else ''} "
                  rf"(orange): $\ell={l}$ angular nodes", fontsize=9)
    # polar part
    ax3 = fig.add_subplot(1, 3, 3)
    tt = np.linspace(0, np.pi, 600)
    pl = ylm_norm(l, m_abs) * legendre_assoc(l, m_abs, np.cos(tt))
    ax3.plot(np.degrees(tt), pl, color=INK, lw=1.8)
    ax3.axhline(0, color="0.5", lw=0.8)
    ax3.axvline(90, color="0.7", lw=0.8, ls=":")
    amp = max(np.abs(pl).max(), 1e-12)
    for t0 in lat:
        ax3.plot(np.degrees(t0), 0, "o", color=POLAR_NODE, ms=7, zorder=5)
        ax3.annotate(f"{np.degrees(t0):.1f}°", (np.degrees(t0), 0), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color=INK)
    ax3.set_xlim(0, 180)
    ax3.set_ylim(-1.35 * amp, 1.35 * amp)
    ax3.set_xticks([0, 45, 90, 135, 180])
    ax3.set_xlabel(r"polar angle $\vartheta$ (degrees)")
    ax3.set_ylabel(rf"$N\,P_{{{l}}}^{{{m_abs}}}(\cos\vartheta)$")
    ax3.set_title(rf"polar part of the wave: its {l - m_abs} zero{'s' if l - m_abs != 1 else ''} are the polar nodes (cones)", fontsize=9)
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
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
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
        ax.plot(r * AB, np.clip(effective_potential(r, l), -1.2, 1.0), color=POT_COLOR, lw=2, ls=ls, label=rf"$\ell={l}$")
    for n in range(1, 6):
        ax.axhline(hydrogen_energy(n), color=OMEGA_COLOR, lw=1)
        if n <= 3:
            ax.text(23.8 * AB, hydrogen_energy(n) + 0.012, f"$n={n}$", color=INK, fontsize=8, ha="right")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlim(0, 24 * AB)
    ax.set_ylim(-0.75, 0.2)
    ax.set_xlabel(r"distance $r$ (Å)")
    ax.set_ylabel("energy (hartree)")
    ax.set_title(r"effective potential $-1/r+\ell(\ell+1)/(2r^2)$ and the levels $E_n=-1/(2n^2)$", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def central_levels_figure(l=0, slide=False):
    """The potential -1/r, the centrifugal term l(l+1)/(2r^2) and their sum V_eff, in atomic units, with the allowed
    levels E_n = -1/(2n^2) as horizontal lines and the radial waves u(r) of the allowed n_r = n - l - 1 drawn at their
    energies. Fixed axes, so that the figures for different l can be compared."""
    from coordinates_plots import _rc, _scale

    sc = _scale(slide)
    n_max = l + 3  # the three lowest allowed levels of this l (n_r = 0, 1, 2) and their waves
    r = np.linspace(0.05, 34, 3400)
    with _rc(slide):
        fig = plt.figure(figsize=(8 * sc if slide else 8, 6 * sc if slide else 4.4))
        ax = fig.add_axes([0.17, 0.15, 0.8, 0.78])
        ax.plot(r * AB, np.clip(-1.0 / r, -0.7, None), color=POT_COLOR, lw=2.2 * sc, alpha=0.55, label=r"$-1/r$")
        if l > 0:
            ax.plot(r * AB, np.clip(l * (l + 1) / (2 * r**2), None, 0.5), color=KIN_COLOR, lw=2.2 * sc, ls="--",
                    label=r"$\ell(\ell+1)/(2r^2)$")
        ax.plot(r * AB, np.clip(effective_potential(r, l), -0.7, 0.5), color=POT_COLOR, lw=3.2 * sc,
                label=r"$V_{\mathrm{eff}}$")
        ax.axhline(0, color="black", lw=0.8)
        for n in range(1, n_max + 1):
            e = float(hydrogen_energy(n))
            allowed = n >= l + 1
            ax.axhline(e, color=OMEGA_COLOR, lw=(2.0 if allowed else 0.8) * sc, alpha=1.0 if allowed else 0.25,
                       ls="-" if allowed else ":", zorder=1)
            if n <= 3:
                ax.text(33.7 * AB, e + 0.012, f"$n={n}$", color=INK, fontsize=(15 if slide else 9), ha="right", va="bottom")
            if allowed:
                u = radial_u(n, l, r)
                scale = 0.05 / np.abs(u).max()
                ax.plot(r * AB, e + scale * u, color=H_COLOR, lw=2.6 * sc)
        ax.set_xlim(0, 34 * AB)
        ax.set_ylim(-0.7, 0.2)
        ax.set_xlabel(r"distance $r$ (Å)")
        ax.set_ylabel("energy (hartree)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=(15 if slide else 9))
        ax.set_title(rf"$\ell={l}$", fontsize=(22 if slide else 11), color=ANGULAR_COLOR)
        ax.text(0.02, 0.97, r"$n_r=n-\ell-1=0,1,\dots$", transform=ax.transAxes, fontsize=(15 if slide else 8.5),
                va="top", color=INK)
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
    ax.plot(r[mask] * AB, np.clip(effective_potential(r[mask], l), -1.5, 1.0), color=POT_COLOR, lw=2, label="effective potential")
    ax.axhline(e[k], color=OMEGA_COLOR, lw=1.2, label=rf"numerical $E={e[k]:.5f}$ hartree")
    ax.plot(r[mask] * AB, e[k] + scale * u_num[mask], color=H_COLOR, lw=2.4, label=r"radial wave $u=rR$ (numerical)")
    ax.plot(r[mask][::25] * AB, e[k] + scale * u_ex[mask][::25], "o", color=A_COLOR, ms=3.5, label="exact")
    ax.axhline(0, color="black", lw=0.7)
    nodes = r[:-1][(u_num[:-1] * u_num[1:] < 0)]
    ax.plot(nodes * AB, np.full_like(nodes, e[k]), "o", color=INK, ms=6, zorder=6)
    ax.set_xlim(0, rmax * AB)
    lo = min(-0.6, e[k] - 0.2) if n == 1 else -0.3
    ax.set_ylim(max(lo, -0.7), 0.15)
    ax.set_xlabel(r"distance $r$ (Å)")
    ax.set_ylabel("energy (hartree)")
    ax.set_title(
        rf"$n={n},\ \ell={l}$: $n_r={n - l - 1}$ radial node{'s' if n - l - 1 != 1 else ''} (black dots);  "
        rf"exact $E=-1/(2n^2)={hydrogen_energy(n):.5f}$ ({hydrogen_energy_ev(n):.3f} eV)",
        fontsize=8.5,
    )
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    ax.plot(r[mask] * AB, (u_num - u_ex)[mask], color=KIN_COLOR, lw=1.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel(r"distance $r$ (Å)")
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
    box = [-ext * AB, ext * AB, -ext * AB, ext * AB]
    axes[0].imshow(psi, origin="lower", extent=box, cmap=WAVE_CMAP, vmin=-top, vmax=top)
    axes[0].contour(s[0] * AB, z[:, 0] * AB, psi, levels=[0], colors=[INK], linewidths=1.0)
    axes[0].set_title(r"wave $\psi$ in a plane through the axis" "\n" "(blue +, red −, black = nodes)", fontsize=8.5)
    axes[1].imshow(psi**2, origin="lower", extent=box, cmap=PROB_CMAP, norm=PowerNorm(0.5, vmin=0, vmax=top**2))
    axes[1].set_title(r"density $|\psi|^2$, same plane" "\n" "(square-root shading)", fontsize=8.5)
    pts = sample_density(n, l, m, n_points, ext, seed=3)
    phi0 = 0.0 if m >= 0 else np.pi / (2 * abs(m))
    # side view: horizontal axis is the in-plane direction, depth is ignored
    horiz = pts[:, 0] * np.cos(phi0) + pts[:, 1] * np.sin(phi0)
    sign = psi_real(n, l, m, *pts.T) > 0
    axes[2].scatter(horiz[sign] * AB, pts[sign, 2] * AB, s=1.2, color=PLUS, alpha=0.5, linewidths=0)
    axes[2].scatter(horiz[~sign] * AB, pts[~sign, 2] * AB, s=1.2, color=MINUS, alpha=0.5, linewidths=0)
    axes[2].set_xlim(-ext * AB, ext * AB)
    axes[2].set_ylim(-ext * AB, ext * AB)
    axes[2].set_title(f"{n_points} points drawn with probability $|\\psi|^2$\n(side view, all depths)", fontsize=8.5)
    for ax in axes[:3]:
        ax.set_aspect("equal")
        ax.set_xlabel(r"in-plane distance from axis (Å)")
    axes[0].set_ylabel(r"$z$ (Å)")
    rr = np.linspace(0, ext * 1.1, 800)
    ax = axes[3]
    ax.fill_between(rr * AB, radial_u(n, l, rr) ** 2 / AB, color=PROB_COLOR, alpha=0.8)
    ax.plot(rr * AB, radial_u(n, l, rr) ** 2 / AB, color=R_COLOR, lw=2)
    ax.axvline(mean_radius(n, l) * AB, color=A_COLOR, ls="--", lw=1.2,
               label=rf"mean $\langle r\rangle={mean_radius(n, l) * AB:.2f}$ Å")
    ax.set_xlabel(r"distance $r$ (Å)")
    ax.set_title(r"probability per Å, $r^2R^2$" "\n" rf"$n_r={n - l - 1}$ radial node{'s' if n - l - 1 != 1 else ''}", fontsize=8.5)
    ax.legend(fontsize=8)
    ax.set_xlim(0, ext * 1.1 * AB)
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
        ax.imshow(np.sign(psi) * np.abs(psi) ** 0.5, origin="lower", extent=[-ext * AB, ext * AB, -ext * AB, ext * AB],
                  cmap=WAVE_CMAP, vmin=-top**0.5, vmax=top**0.5)
        ax.contour(s[0] * AB, z[:, 0] * AB, psi, levels=[0], colors=[INK], linewidths=0.8)
        ax.set_title(f"{orbital_name(n, l, m)}  (" + (rf"$n_r={n - l - 1}$, $\ell={l}$") + ")", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        rr = np.linspace(0, ext * 1.1, 600)
        axb = axes[1, j]
        axb.fill_between(rr * AB, radial_u(n, l, rr) ** 2, color=PROB_COLOR, alpha=0.8)
        axb.plot(rr * AB, radial_u(n, l, rr) ** 2, color=R_COLOR, lw=1.8)
        axb.set_xlabel("$r$ (Å)", fontsize=8)
        axb.set_yticks([])
        axb.set_xlim(0, ext * 1.1 * AB)
        axb.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(r"$\psi$ (plane through the $z$ axis)", fontsize=8)
    axes[1, 0].set_ylabel(r"$r^2R^2$", fontsize=8)
    fig.suptitle("Colour shows the signed wave (square-root scale so outer lobes are visible); each panel has its own length scale",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    return fig


# Colours of the three kinds of node, used only to mark them in the figures of this notebook:
RADIAL_NODE = R_COLOR      # cut across the distance r (points on a string's r, circles on a drum, spheres in an atom)
POLAR_NODE = OMEGA_COLOR   # cut across the polar angle (cones; the flat plane z = 0 is the special case)
AZIMUTH_NODE = "#6B913C"   # cut across the azimuth (diameters on a drum, planes through the axis in an atom)


def string_figure(n):
    """A string fixed at both ends: n half-wavelengths, n - 1 nodes, each a point across the only direction."""
    x = np.linspace(0, 1, 800)
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(x, np.sin(n * np.pi * x), color=H_COLOR, lw=2.6)
    nodes = np.arange(1, n) / n
    ax.plot(nodes, np.zeros_like(nodes), "o", color=RADIAL_NODE, ms=9, zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-1.4, 1.4)
    ax.set_yticks([])
    ax.set_xticks([0, 1], ["0", "$L$"])
    ax.set_title(rf"string, $n={n}$: $N_\mathrm{{nodes}}={n - 1}$ node{'s' if n != 2 else ''}, each a point that cuts across $x$", fontsize=9)
    fig.tight_layout()
    return fig


def _drum_image(m, k, res=360):
    x = np.linspace(-1, 1, res)[None, :]
    y = np.linspace(-1, 1, res)[:, None]
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    psi = np.where(r <= 1, drum_mode(m, k, np.minimum(r, 1), phi), np.nan)
    return psi


def _draw_drum(ax, m, k, res=360):
    psi = _drum_image(m, k, res)
    top = np.nanmax(np.abs(psi))
    ax.imshow(psi, origin="lower", extent=[-1, 1, -1, 1], cmap=WAVE_CMAP, vmin=-top, vmax=top)
    radii, angles = drum_nodes(m, k)
    t = np.linspace(0, 2 * np.pi, 300)
    for rad in radii:
        ax.plot(rad * np.cos(t), rad * np.sin(t), color=RADIAL_NODE, lw=2.2)
    for a in angles:
        ax.plot([-np.cos(a), np.cos(a)], [-np.sin(a), np.sin(a)], color=AZIMUTH_NODE, lw=2.2)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def drum_figure(m, nr):
    """A circular drumhead mode seen from above (left), with its radial wave and its wave around the centre (right).
    m is the signed m_l (l = |m| diameters), nr the number of nodal circles; n = nr + l + 1."""
    k = nr + 1  # index of the Bessel zero that sets the frequency
    l = abs(m)
    n = nr + l + 1
    j = drum_frequency(m, k)
    fig = plt.figure(figsize=(9.8, 3.6))
    ax = fig.add_subplot(1, 3, 1)
    _draw_drum(ax, m, k)
    ax.set_title(f"drum, seen from above\n(blue up, red down; rim fixed)", fontsize=9)
    ax = fig.add_subplot(1, 3, 2)
    r = np.linspace(0, 1, 500)
    ax.plot(r, bessel_j(abs(m), j * r), color=H_COLOR, lw=2.4)
    ax.axhline(0, color="black", lw=0.7)
    radii, angles = drum_nodes(m, k)
    a = abs(m)
    ax.plot(radii, np.zeros_like(radii), "o", color=RADIAL_NODE, ms=8, zorder=5)
    ax.set_xlabel(r"distance from centre $r$ (rim at 1)")
    ax.set_title(rf"along a radius: $n_r={nr}$ nodal circle{'s' if nr != 1 else ''}" + f"\n(like the string, {k} half-wave{'s' if k != 1 else ''})", fontsize=9)
    ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 3)
    ph = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.degrees(ph), (np.cos if m >= 0 else np.sin)(a * ph), color=H_COLOR, lw=2.4)
    ax.axhline(0, color="black", lw=0.7)
    allang = np.sort(np.concatenate([angles, angles + np.pi])) if a > 0 else np.array([])
    ax.plot(np.degrees(allang), np.zeros_like(allang), "o", color=AZIMUTH_NODE, ms=8, zorder=5)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r"azimuth $\varphi$ (degrees)")
    ax.set_yticks([])
    form = r"\cos" if m >= 0 else r"\sin"
    ax.set_title(rf"around the centre: $\ell={a}$ nodal diameter{'s' if a != 1 else ''}" + "\n"
                 rf"(like the ring: ${form}({a}\varphi)$, {a} period{'s' if a != 1 else ''})", fontsize=9)
    fig.suptitle(rf"$n={n},\ \ell={a},\ n_r={nr},\ m_\ell={m}$: $N_\mathrm{{nodes}}={n - 1}$ = {nr} circle{'s' if nr != 1 else ''} + {a} diameter{'s' if a != 1 else ''};  "
                 rf"frequency $\propto j_{{{a},{k}}}={j:.3f}$", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def drum_gallery_figure(l_max=3, nr_max=2):
    """Drum modes in a grid: columns l = 0 ... l_max (diameters), rows n_r = 0 ... nr_max (circles)."""
    fig, axes = plt.subplots(nr_max + 1, l_max + 1, figsize=(2.2 * (l_max + 1), 2.2 * (nr_max + 1)))
    for nr in range(nr_max + 1):
        for l in range(l_max + 1):
            ax = axes[nr, l]
            _draw_drum(ax, l, nr + 1, 200)
            ax.set_title(rf"$n_r={nr}$, $\ell={l}$", fontsize=8)
    fig.suptitle("circles (pink): cut across $r$;  diameters (green): cut across $\\varphi$;  rim (dark): fixed", fontsize=9)
    fig.tight_layout()
    return fig


def node_anatomy_figure(n, l, m):
    """The nodal surfaces of an orbital, by kind. Left: a plane through the z axis (spheres appear as circles, cones as
    straight lines through the centre, planes through the axis as the z axis). Right: a horizontal slice (spheres as
    circles, planes through the axis as diameters, exactly as on the drum; cones as circles when the slice is off the
    equator)."""
    g = node_geometry(n, l, m)
    ext = _extent(n)
    flat = any(abs(t - np.pi / 2) < 1e-9 for t in g["cones"])
    z0 = 0.0 if not flat else 0.18 * ext
    s, z, x, y = cross_section_plane(m, ext, 360)
    psi_a = psi_real(n, l, m, x, y, z)
    sx = np.linspace(-ext, ext, 360)[None, :]
    sy = np.linspace(-ext, ext, 360)[:, None]
    psi_b = psi_real(n, l, m, sx + 0 * sy, sy + 0 * sx, z0 + 0 * sx + 0 * sy)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    t = np.linspace(0, 2 * np.pi, 400)
    E = ext * AB  # half-width of the displayed plane, in ångström
    box = [-E, E, -E, E]
    for ax, psi in zip(axes, (psi_a, psi_b)):
        top = max(np.abs(psi).max(), 1e-12)
        ax.imshow(psi, origin="lower", extent=box, cmap=WAVE_CMAP, vmin=-top, vmax=top, alpha=0.8)
        ax.set_aspect("equal")
    ax = axes[0]
    phi0 = 0.0 if m >= 0 else np.pi / (2 * abs(m))
    for rad in g["spheres"]:
        ax.plot(rad * AB * np.cos(t), rad * AB * np.sin(t), color=RADIAL_NODE, lw=2.4)
    for th in g["cones"]:
        if abs(th - np.pi / 2) < 1e-9:   # the flat plane z = 0
            ax.plot([-E, E], [0, 0], color=POLAR_NODE, lw=2.4)
        else:                            # a cone cuts the plane in two straight lines through the centre
            for sgn in (1, -1):
                ax.plot([0, sgn * 1.5 * E * np.sin(th)], [0, sgn * 1.5 * E * np.cos(th)], color=POLAR_NODE, lw=2.4)
    if len(g["planes"]):
        ax.plot([0, 0], [-E, E], color=AZIMUTH_NODE, lw=2.4)
    ax.set_xlim(-E, E)
    ax.set_ylim(-E, E)
    ax.set_xlabel("distance from the axis (Å)")
    ax.set_ylabel("$z$ (Å)")
    ax.set_title("plane through the $z$ axis", fontsize=9)
    ax = axes[1]
    for rad in g["spheres"]:
        if rad > abs(z0):
            rho = np.sqrt(rad**2 - z0**2) * AB
            ax.plot(rho * np.cos(t), rho * np.sin(t), color=RADIAL_NODE, lw=2.4)
    for th in g["cones"]:
        if abs(th - np.pi / 2) > 1e-9 and z0 > 0 and th < np.pi / 2:
            rho = z0 * AB * np.tan(th)
            ax.plot(rho * np.cos(t), rho * np.sin(t), color=POLAR_NODE, lw=2.4)
    for a in g["planes"]:
        ax.plot([-1.5 * E * np.cos(a), 1.5 * E * np.cos(a)], [-1.5 * E * np.sin(a), 1.5 * E * np.sin(a)],
                color=AZIMUTH_NODE, lw=2.4)
    ax.set_xlim(-E, E)
    ax.set_ylim(-E, E)
    ax.set_xlabel("$x$ (Å)")
    ax.set_ylabel("$y$ (Å)")
    ax.set_title("horizontal slice, $z=0$ (seen from above, like the drum)" if not flat
                 else rf"horizontal slice at $z={z0 * AB:.1f}$ Å (the plane $z=0$ is itself a node)", fontsize=9)
    ns, nc, npl = len(g["spheres"]), len(g["cones"]), len(g["planes"])
    fig.suptitle(
        rf"{orbital_name(n, l, m)}:  $n_r={ns}$ sphere{'s' if ns != 1 else ''} (pink) + $n_\vartheta={nc}$ cone{'s' if nc != 1 else ''} "
        rf"(orange) + $n_\varphi={npl}$ plane{'s' if npl != 1 else ''} through the axis (green) = $N_\mathrm{{nodes}}={n - 1}$",
        fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Animations (GIF bytes): one full period of a standing wave, psi(x, t) = psi(x) cos(2 pi t / T)


def _frames_to_gif(frames, duration_ms=45):
    """Pillow frames -> animated GIF bytes (looping forever)."""
    import io

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return buf.getvalue()


def _fig_to_image(fig, dpi):
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


@lru_cache(maxsize=None)
def string_gif(n, n_frames=36, dpi=70):
    """The string with n half-wavelengths vibrating through one period; the nodes stay put (pink dots)."""
    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axhline(0, color="black", lw=0.8)
    (line,) = ax.plot(x, np.sin(n * np.pi * x), color=H_COLOR, lw=2.6)
    nodes = np.arange(1, n) / n
    ax.plot(nodes, np.zeros_like(nodes), "o", color=RADIAL_NODE, ms=9, zorder=5)
    ax.plot([0, 1], [0, 0], "o", color=INK, ms=6, zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-1.4, 1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(rf"$n={n}$: the nodes (dots) never move", fontsize=9)
    fig.tight_layout()
    frames = []
    for i in range(n_frames):
        line.set_ydata(np.sin(n * np.pi * x) * np.cos(2 * np.pi * i / n_frames))
        frames.append(_fig_to_image(fig, dpi))
    plt.close(fig)
    return _frames_to_gif(frames)


@lru_cache(maxsize=None)
def drum_gif(m, nr, n_frames=30, dpi=60):
    """The (m_l = m, n_r = nr) drum mode vibrating through one period, seen obliquely. The colour is the sign of the
    mode (blue, red), so the nodal circles and diameters stay visible as white seams while the surface moves."""
    k = nr + 1
    r = np.linspace(0, 1, 36)[:, None]
    phi = np.linspace(0, 2 * np.pi, 97)[None, :]
    mode = drum_mode(m, k, r, phi)
    mode = mode / np.abs(mode).max()
    x, y = r * np.cos(phi), r * np.sin(phi)
    face = WAVE_CMAP(0.5 + 0.5 * np.clip(mode / 0.9, -1, 1))
    fig = plt.figure(figsize=(4.2, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.2, top=1.2)
    frames = []
    for i in range(n_frames):
        ax.clear()
        ax.plot_surface(x, y, 0.7 * mode * np.cos(2 * np.pi * i / n_frames), facecolors=face, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), 0 * t, color=INK, lw=2)
        ax.set_box_aspect((1, 1, 0.45))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.75, 0.75)
        ax.set_axis_off()
        ax.view_init(elev=38, azim=-60)
        frames.append(_fig_to_image(fig, dpi))
    plt.close(fig)
    return _frames_to_gif(frames, 50)


# ---------------------------------------------------------------------------------------------------------------
# Fourier series on a circle and on a sphere


def fourier_circle_figure(m_max):
    """A shape on a circle rebuilt from its first m_max harmonics. Left: unrolled. Middle: wrapped round the circle.
    Right: how large each harmonic's contribution is."""
    phi = np.linspace(0, 2 * np.pi, 1200)
    a, b = circle_coefficients(circle_function, 40)
    exact = circle_function(phi)
    approx = circle_reconstruct(a, b, phi, m_max)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
    ax = axes[0]
    ax.plot(np.degrees(phi), exact, color="0.65", lw=2, label="the shape")
    ax.plot(np.degrees(phi), approx, color=H_COLOR, lw=2.2, label=f"sum of the first {m_max} harmonics")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r"azimuth $\varphi$ (degrees)")
    ax.set_ylim(-0.4, 1.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("unrolled", fontsize=9)
    ax = axes[1]
    ax.plot(np.cos(phi), np.sin(phi), color="0.85", lw=1)
    ax.plot((1 + 0.5 * exact) * np.cos(phi), (1 + 0.5 * exact) * np.sin(phi), color="0.6", lw=2)
    ax.plot((1 + 0.5 * approx) * np.cos(phi), (1 + 0.5 * approx) * np.sin(phi), color=H_COLOR, lw=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the same, radius $=1+f/2$", fontsize=9)
    ax = axes[2]
    amp = np.hypot(a[1:], b[1:])[:20]
    ms = np.arange(1, len(amp) + 1)
    ax.bar(ms, amp, color=[H_COLOR if mm <= m_max else "0.8" for mm in ms])
    ax.set_xlabel(r"harmonic number $m_\ell$ (number of periods round the circle)")
    ax.set_title("size of each harmonic (blue: included)", fontsize=9)
    fig.tight_layout()
    return fig


def fourier_sphere_figure(l_max):
    """A shape on a sphere rebuilt from all harmonics up to l_max, as maps; and the strength of each l."""
    co = sphere_coefficients(20)
    theta = np.linspace(0, np.pi, 120)[:, None]
    phi = np.linspace(0, 2 * np.pi, 240)[None, :]
    exact = sphere_function(theta, phi)
    approx = sphere_reconstruct(co, l_max, theta, phi)
    top = max(np.abs(exact).max(), np.abs(approx).max())
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, data, title in ((axes[0], exact, "the shape (a map of bumps)"),
                            (axes[1], approx, rf"sum of all harmonics up to $\ell={l_max}$ ({(l_max + 1) ** 2} terms)")):
        ax.imshow(data, extent=[0, 360, 180, 0], cmap=WAVE_CMAP, vmin=-top, vmax=top, aspect="auto")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([0, 90, 180])
        ax.set_xlabel(r"azimuth $\varphi$ (degrees)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(r"polar angle $\vartheta$ (degrees)")
    power = sphere_power(co)
    ls = np.arange(len(power))
    axes[2].bar(ls, power, color=[H_COLOR if l <= l_max else "0.8" for l in ls])
    axes[2].set_xlabel(r"$\ell$ (number of nodal lines)")
    axes[2].set_title("strength of each $\\ell$ (blue: included)", fontsize=9)
    axes[2].set_yscale("log")
    fig.tight_layout()
    return fig
