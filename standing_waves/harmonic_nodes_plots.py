"""Real spherical harmonic with all its angular nodes drawn: cones (orange, vartheta) and planes (green, varphi).

A spherical harmonic does not depend on r, so it has no spherical (pink, r) nodes; those appear for the hydrogen
orbitals, where the radial function adds n - l - 1 of them."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordinates import VIEW_AZIM, VIEW_ELEV, spherical_to_cartesian  # noqa: E402
from coordinates_plots import AZIMUTH_C, POLAR_C, plane_piece  # noqa: E402
from hydrogen import azimuth_nodes, latitude_nodes_predicted, nodal_structure, real_ylm  # noqa: E402
from hydrogen_plots import INK, _draw_shape  # noqa: E402
from standing_waves_plots import WAVE_CMAP  # noqa: E402


def harmonic_node_angles(l, m):
    """(polar angles of the cones, azimuths in [0, pi) of the planes) of the real harmonic (l, m), in radians.

    m > 0 is the cos(m phi) harmonic, m < 0 the sin(|m| phi) one. Each plane contains the azimuths phi0 and phi0 + pi.
    The number of cones is l - |m| and of planes |m|, so the total is l."""
    thetas = latitude_nodes_predicted(l, m)
    phis = azimuth_nodes(m, "cos" if m > 0 else "sin")
    return np.asarray(thetas), np.asarray(phis)


def _draw_nodes(ax, l, m, radius):
    thetas, phis = harmonic_node_angles(l, m)
    for th in thetas:
        rr, ph = np.meshgrid(np.linspace(0, radius, 14), np.linspace(0, 2 * np.pi, 90))
        x, y, z = spherical_to_cartesian(rr, th, ph)
        ax.plot_surface(x, y, z, color=POLAR_C, alpha=0.5, shade=False, linewidth=0, rstride=1, cstride=1)
    for p0 in phis:
        for sign in (1, -1):
            for lower in (False, True):
                x, y, z = plane_piece(p0, sign, lower, n=20)
                ax.plot_surface(x * radius, y * radius, z * radius, color=AZIMUTH_C, alpha=0.5, shade=False, linewidth=0)


def harmonic_nodes_figure(l, m, show_nodes=True):
    """Like harmonic_figure: the shape (left) and the map (right); with show_nodes the nodes are drawn in the colours of
    the polar-coordinates section (cones orange, planes green) in both panels."""
    through, latitudes = nodal_structure(l, m)
    thetas, phis = harmonic_node_angles(l, m)
    fig = plt.figure(figsize=(9.6, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    extent = _draw_shape(ax, l, m)
    if show_nodes:
        for artist in list(ax.collections):  # translucent lobes, so that the nodes inside them show through
            artist.set_alpha(0.55)
        _draw_nodes(ax, l, m, 0.95 * extent)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_title("shape: distance from centre $=|Y|$\n(blue $Y>0$, red $Y<0$)", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    theta = np.linspace(0, np.pi, 240)[:, None]
    phi = np.linspace(0, 2 * np.pi, 480)[None, :]
    y = real_ylm(l, m, theta, phi) * np.ones_like(phi)
    top = max(np.abs(y).max(), 1e-12)
    ax2.imshow(y, extent=[0, 360, 180, 0], cmap=WAVE_CMAP, vmin=-top, vmax=top, aspect="auto")
    if show_nodes:
        for th in thetas:
            ax2.axhline(np.degrees(th), color=POLAR_C, lw=2.2)
        for p0 in phis:
            for p in (p0, p0 + np.pi):
                ax2.axvline(np.degrees(p), color=AZIMUTH_C, lw=2.2)
    elif l > 0:
        ax2.contour(np.degrees(phi[0]), np.degrees(theta[3:-3, 0]), y[3:-3], levels=[0], colors=[INK], linewidths=1.3)
    ax2.set_xlabel(r"azimuth $\varphi$ (degrees)")
    ax2.set_ylabel(r"polar angle $\vartheta$ (degrees)")
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_yticks([0, 45, 90, 135, 180])
    tag_c, tag_p = (" (orange)", " (green)") if show_nodes else ("", "")
    ax2.set_title(
        rf"$\ell={l},\ m_\ell={m}$: $n_\vartheta={latitudes}$ cone{'s' if latitudes != 1 else ''}{tag_c} + "
        rf"$n_\varphi={through}$ plane{'s' if through != 1 else ''}{tag_p}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def harmonic_nodes_gallery_figure(l_max=3, show_nodes=True):
    """All real harmonics up to l_max in rows (s, p, d, f), as harmonic_gallery_figure, with all their nodes drawn
    (cones orange, planes green) when show_nodes is True."""
    cols = 2 * l_max + 1
    fig = plt.figure(figsize=(1.55 * cols, 1.65 * (l_max + 1)))
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ax = fig.add_subplot(l_max + 1, cols, l * cols + (m + l_max) + 1, projection="3d")
            extent = _draw_shape(ax, l, m, 60, 120)
            if show_nodes:
                for artist in list(ax.collections):
                    artist.set_alpha(0.55)
                _draw_nodes(ax, l, m, 0.95 * extent)
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
            ax.set_title(f"$m_\\ell={m}$", fontsize=8, pad=-6)
        fig.text(0.005, 1 - (l + 0.5) / (l_max + 1), f"$\\ell={l}$\n{'spdf'[l]}", fontsize=10, va="center", color=INK)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.01, wspace=-0.12, hspace=0.05)
    return fig
