"""Figures for polar and spherical coordinates, for the notebook and for slides. Each function returns a Figure.

`slide=True` gives the slide style: white background, large fonts (20 pt), thick lines, height 6 inches.
`slide=False` gives the same figure smaller, for the notebook. Colours come from qmstyle.py (one colour per quantity:
the distance r is pink, as in the node figures; the polar angle and the cones are orange; the azimuth and the planes
through the axis are green; marks and the point are red; angle marks and axes are dark).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordinates import VIEW_AZIM, VIEW_ELEV, polar_grid_lines, polar_to_cartesian, spherical_to_cartesian  # noqa: E402
from qmstyle import A_COLOR, INK, K_COLOR, OMEGA_COLOR, R_COLOR  # noqa: E402

R_C, POLAR_C, AZIMUTH_C, POINT_C = R_COLOR, OMEGA_COLOR, K_COLOR, A_COLOR
PHI_PLANE_DEG = 100.0  # azimuth of the drawn half-plane: 65 degrees from the line of sight (azimuth VIEW_AZIM), clearly not edge-on
HEIGHT = 6.0  # inches; every slide figure has this height, 8 inches width (the three-panel figure 18)


def _scale(slide):
    return 1.0 if slide else 0.55


def _rc(slide):
    s = _scale(slide)
    return plt.rc_context({
        "font.size": 20 * s, "axes.titlesize": 20 * s, "lines.linewidth": 3.5 * s,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def _figure(width, slide):
    s = _scale(slide)
    return plt.figure(figsize=(width * s, HEIGHT * s))


def _arrow(ax, p, q, color=INK, lw=3.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def polar_point_figure(r=1.6, phi_deg=40.0, slide=True, lim=2.2):
    """One point described two ways: by (x, y) (dashed projections) and by (r, phi) (pink distance, dark angle)."""
    s = _scale(slide)
    phi = np.radians(phi_deg)
    x, y = (float(v) for v in polar_to_cartesian(r, phi))
    with _rc(slide):
        fig = _figure(8, slide)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.80])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        _arrow(ax, (-lim * 0.97, 0), (lim * 0.97, 0), lw=2.5 * s)
        _arrow(ax, (0, -lim * 0.97), (0, lim * 0.97), lw=2.5 * s)
        ax.text(lim * 0.97, -0.28, r"$x$", ha="right", va="top")
        ax.text(0.1, lim * 0.97, r"$y$", ha="left", va="top")
        ax.plot([x, x], [0, y], color="0.45", ls="--", lw=2.5 * s)
        ax.plot([0, x], [y, y], color="0.45", ls="--", lw=2.5 * s)
        ax.plot([0, x], [0, y], color=R_C, lw=4.5 * s, solid_capstyle="round")
        if r > 1e-9:
            a = np.linspace(0, phi, 60)
            ar = min(0.55, 0.6 * r)
            ax.plot(ar * np.cos(a), ar * np.sin(a), color=INK, lw=3 * s)
            ax.text(1.35 * ar * np.cos(phi / 2), 1.35 * ar * np.sin(phi / 2), r"$\varphi$", ha="center", va="center")
            ax.text(0.5 * x - 0.15 * np.sin(phi), 0.5 * y + 0.15 * np.cos(phi), r"$r$", color=R_C, ha="center", va="center")
        ax.scatter([x], [y], s=260 * s**2, color=POINT_C, zorder=5)
        ax.text(x, -0.28 if y >= 0 else 0.1, r"$x$" if abs(x) > 0.25 else "", ha="center", va="top" if y >= 0 else "bottom", color="0.35")
        ax.text(-0.1 if x >= 0 else 0.1, y, r"$y$" if abs(y) > 0.25 else "", ha="right" if x >= 0 else "left", va="center", color="0.35")
        fig.suptitle(rf"$(x,y)=({x:.2f},\ {y:.2f})$   $\Leftrightarrow$   $(r,\varphi)=({r:.2f},\ {phi_deg:.0f}^\circ)$",
                     y=0.97, fontsize=17 * s)
    return fig


def polar_grid_figure(n_circles=4, n_rays=12, slide=True, r_max=2.0):
    """The polar grid: circles of constant r (pink) and rays of constant phi (green) through the centre."""
    s = _scale(slide)
    radii, angles = polar_grid_lines(n_circles, n_rays, r_max)
    with _rc(slide):
        fig = _figure(8, slide)
        ax = fig.add_axes([0.04, 0.02, 0.92, 0.88])
        lim = 1.12 * r_max
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        for a in angles:
            ax.plot([0, r_max * np.cos(a)], [0, r_max * np.sin(a)], color=AZIMUTH_C, lw=3 * s)
        t = np.linspace(0, 2 * np.pi, 400)
        for rad in radii:
            ax.plot(rad * np.cos(t), rad * np.sin(t), color=R_C, lw=3.5 * s)
        ax.scatter([0], [0], s=150 * s**2, color=INK, zorder=5)
        fig.suptitle("circles: constant $r$,  rays: constant $\\varphi$", y=0.97, fontsize=17 * s)
    return fig


def _axes_3d(ax, lim=1.15, labels=True, zoom=0.85, lo=0.0, grid=False, floor=False):
    """Axes of the shared view (VIEW_ELEV, VIEW_AZIM): z up, y right, x towards the viewer. With grid=True the floor
    (z = 0) and the back wall (x = 0) carry a light grid, as in the textbook figure."""
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)
    ax.set_zlim(lo, lim)
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    if floor:  # a light floor grid over the whole range, so that x and y visibly lie in it (a filled floor would be
        # depth-sorted over the surfaces and wash them out)
        for v in np.linspace(lo, lim, 9):
            ax.plot([v, v], [lo, lim], [0, 0], color="0.8", lw=1.4)
            ax.plot([lo, lim], [v, v], [0, 0], color="0.8", lw=1.4)
    if grid:
        g = np.linspace(0, lim, 7)
        for v in g:
            ax.plot([v, v], [0, lim], [0, 0], color="0.85", lw=1.2)  # floor
            ax.plot([0, lim], [v, v], [0, 0], color="0.85", lw=1.2)
            ax.plot([0, 0], [v, v], [0, lim], color="0.85", lw=1.2)  # back wall x = 0
            ax.plot([0, 0], [0, lim], [v, v], color="0.85", lw=1.2)
    for d, lab in zip(np.eye(3), ("x", "y", "z")):
        ax.plot(*zip(np.zeros(3), 1.12 * d), color=INK, lw=2.2)
        if labels:
            ax.text(*(1.24 * d), f"${lab}$", color=INK, ha="center", va="center")


def spherical_coordinates_figure(r=1.0, theta_deg=50.0, phi_deg=68.0, slide=True):
    """The three coordinates of one point, in the textbook view (z up, y right, x towards the viewer): the distance r
    (pink, dashed), the polar angle theta from the z axis (orange arc) and the azimuth phi in the xy plane from the x
    axis (green arc). The point is above the xy plane; its drop to the plane and the line from the origin to the foot
    are dashed."""
    s = _scale(slide)
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    p = np.array([float(v) for v in spherical_to_cartesian(r, th, ph)])
    foot = np.array([p[0], p[1], 0.0])
    with _rc(slide):
        fig = _figure(8, slide)
        ax = fig.add_axes([0.02, -0.04, 0.96, 1.08], projection="3d")
        _axes_3d(ax, lim=1.25, lo=0.0, zoom=1.0, grid=True)
        ax.plot(*zip(np.zeros(3), p), color=R_C, lw=4.5 * s, ls=(0, (3, 1.6)))
        ax.plot(*zip(p, foot), color="0.45", lw=2.5 * s, ls="--")
        ax.plot(*zip(np.zeros(3), foot), color="0.45", lw=2.5 * s, ls="--")
        ax.scatter(*p, color=POINT_C, s=240 * s**2, zorder=5, depthshade=False)
        t = np.linspace(0, th, 40)
        ax.plot(*spherical_to_cartesian(0.5, t, ph), color=POLAR_C, lw=4 * s)
        f = np.linspace(0, ph, 40)
        ax.plot(*spherical_to_cartesian(0.45, np.pi / 2, f), color=AZIMUTH_C, lw=4 * s)
        ax.text(*spherical_to_cartesian(0.66, th / 2, ph), r"$\vartheta$", color=POLAR_C, fontsize=26 * s,
                ha="center", va="center")
        ax.text(*spherical_to_cartesian(0.6, np.pi / 2, ph / 2 + 0.05), r"$\varphi$", color=AZIMUTH_C, fontsize=26 * s,
                ha="center", va="center")
        ax.text(*(0.55 * p + np.array([0.0, 0.1, -0.04])), r"$r$", color=R_C, fontsize=26 * s, ha="center", va="center")
        ax.text(p[0], p[1] + 0.03, p[2] + 0.1, r"$(r,\vartheta,\varphi)$", color=INK, fontsize=22 * s, ha="center")
    return fig


def plane_piece(phi0, sign=1, lower=False, n=30):
    """Points of one quarter-disc of the plane through the z axis at azimuth phi0: sign = +1 for the half-plane at phi0 and
    -1 for the opposite one (phi0 + pi); lower=True for the part below the xy plane. Radius 1."""
    q, al = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, np.pi / 2, n) * (-1 if lower else 1))
    return sign * q * np.cos(al) * np.cos(phi0), sign * q * np.cos(al) * np.sin(phi0), q * np.sin(al)


def _surface(ax, kind, r0, theta0, phi0, color):
    if kind == "r":
        th, ph = np.meshgrid(np.linspace(0, np.pi, 60), np.linspace(0.5 * np.pi, 2 * np.pi, 90))
        x, y, z = spherical_to_cartesian(r0, th, ph)  # three quarters of the sphere: the missing front quarter shows the axes
    elif kind == "theta":
        rr, ph = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 2 * np.pi, 90))
        x, y, z = spherical_to_cartesian(rr, theta0, ph)
    else:
        # the full plane through the z axis, as two half-discs (z from -1 to 1): the half-plane at phi0 in full colour,
        # and the opposite half-plane (phi0 + pi), which belongs to the same node, lighter
        for sign, lower, alpha in ((1, False, 0.55), (1, True, 0.2), (-1, False, 0.22), (-1, True, 0.1)):
            # the part below the floor (z < 0) is drawn lighter, so that the plane reads as passing through the floor
            x, y, z = plane_piece(phi0, sign, lower)
            ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False, rstride=3, cstride=3, edgecolor="white",
                            linewidth=0.5)
        return x, y, z
    ax.plot_surface(x, y, z, color=color, alpha=0.55, shade=False, rstride=3, cstride=3, edgecolor="white", linewidth=0.5)
    return x, y, z


def _decorate_phi(ax, phi0, s):
    """Make the plane read as standing on the xy plane: its trace on the floor (dashed; lighter for the opposite half)
    and the angle phi from the +x axis to the trace (arc)."""
    ax.plot([0, np.cos(phi0)], [0, np.sin(phi0)], [0, 0], color=AZIMUTH_C, lw=3 * s, ls="--", zorder=6)
    ax.plot([0, -np.cos(phi0)], [0, -np.sin(phi0)], [0, 0], color=AZIMUTH_C, lw=3 * s, ls="--", alpha=0.35, zorder=6)
    f = np.linspace(0, phi0, 60)
    ax.plot(*spherical_to_cartesian(0.45, np.pi / 2, f), color=AZIMUTH_C, lw=4 * s, zorder=6)
    ax.text(*spherical_to_cartesian(0.62, np.pi / 2, phi0 / 2), r"$\varphi$", color=AZIMUTH_C, fontsize=24 * s,
            ha="center", va="center")


def spherical_surface_figure(kind="r", slide=True, r0=1.0, theta_deg=50.0, phi_deg=PHI_PLANE_DEG):
    """One constant-coordinate surface: kind 'r' a sphere (pink), 'theta' a cone (orange), 'phi' a half-plane
    through the z axis (green), drawn standing on the floor with its trace and the angle phi. The sphere is shown over three quarters of a turn, so that the axes stay visible."""
    s = _scale(slide)
    th0, ph0 = np.radians(theta_deg), np.radians(phi_deg)
    color, text = {"r": (R_C, r"$r=\mathrm{const}$: sphere"), "theta": (POLAR_C, r"$\vartheta=\mathrm{const}$: cone"),
                   "phi": (AZIMUTH_C, r"$\varphi=\mathrm{const}$: half-plane"+"\n(a node is the full plane)")}[kind]
    with _rc(slide):
        fig = _figure(8, slide)
        ax = fig.add_axes([0.0, -0.08, 1.0, 1.0], projection="3d")
        _axes_3d(ax, lim=1.2, lo=-1.2, zoom=1.12, floor=True)
        _surface(ax, kind, r0, th0, ph0, color)
        if kind == "phi":
            _decorate_phi(ax, ph0, s)
        fig.suptitle(text, y=0.97, fontsize=20 * s, color=color)
    return fig


def spherical_surfaces_figure(slide=True, r0=1.0, theta_deg=50.0, phi_deg=PHI_PLANE_DEG):
    """The three constant-coordinate surfaces side by side."""
    s = _scale(slide)
    th0, ph0 = np.radians(theta_deg), np.radians(phi_deg)
    items = [("r", R_C, r"$r=\mathrm{const}$: sphere"), ("theta", POLAR_C, r"$\vartheta=\mathrm{const}$: cone"),
             ("phi", AZIMUTH_C, r"$\varphi=\mathrm{const}$: half-plane" + "\n(a node is the full plane)")]
    with _rc(slide):
        fig = _figure(18, slide)
        for i, (kind, color, text) in enumerate(items):
            ax = fig.add_axes([i / 3, -0.08, 1 / 3, 1.0], projection="3d")
            _axes_3d(ax, lim=1.2, lo=-1.2, zoom=1.12, floor=True)
            _surface(ax, kind, r0, th0, ph0, color)
            if kind == "phi":
                _decorate_phi(ax, ph0, s)
            fig.text((i + 0.5) / 3, 0.93, text, ha="center", va="center", fontsize=20 * s, color=color)
    return fig
