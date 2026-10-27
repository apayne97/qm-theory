"""Figure for section 1: the same rotation about z seen in the plane and in three dimensions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spinors import rotation_matrix_so3

# An asymmetric letter F, so that the turning (and any reflection) of the plane is visible.
_F = np.array(
    [(0.25, 0.1), (0.5, 0.1), (0.5, 0.55), (0.9, 0.55), (0.9, 0.75), (0.5, 0.75),
     (0.5, 0.95), (1.1, 0.95), (1.1, 1.15), (0.25, 1.15)]
)
_COLORS = ("tab:red", "tab:green", "tab:blue")


def z_rotation_figure(angle_deg):
    """Left: the plane rotated through `angle_deg`. Right: the same turn about the z axis in space."""
    t = np.radians(angle_deg)
    r = rotation_matrix_so3((0, 0, 1), t)
    shape = np.c_[_F, np.zeros(len(_F))]
    turned = shape @ r.T

    fig = plt.figure(figsize=(10, 4.8))

    ax = fig.add_subplot(1, 2, 1)
    ax.fill(*_F.T, facecolor="none", edgecolor="0.6", ls="--", lw=1.2)
    ax.fill(*turned[:, :2].T, facecolor="#f4c27a", edgecolor="0.25", lw=1.2, alpha=0.9)
    for k, color, name in [(0, "tab:red", "x̂"), (1, "tab:green", "ŷ")]:
        e = np.eye(2)[:, k]
        v = r[:2, :2] @ e
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.75", lw=1.5))
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        ax.text(v[0] * 1.15, v[1] * 1.15, name + "'", color=color, ha="center", va="center")
    phis = np.linspace(0, t, 60)
    ax.plot(0.45 * np.cos(phis), 0.45 * np.sin(phis), color="0.3", lw=1.2)
    if angle_deg > 15:
        ax.text(0.62 * np.cos(t / 2), 0.62 * np.sin(t / 2), "α", ha="center", va="center", color="0.3")
    ax.plot([0], [0], "o", color="0.2", ms=5)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.9", lw=0.8)
    ax.axvline(0, color="0.9", lw=0.8)
    ax.set_title("In the plane: turn about the origin", fontsize=10)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.add_collection3d(Poly3DCollection([shape], facecolor="none", edgecolor="0.6", linestyles="--", lw=1.2))
    ax3.add_collection3d(Poly3DCollection([turned], facecolor="#f4c27a", edgecolor="0.25", lw=1.2, alpha=0.9))
    ax3.plot([0, 0], [0, 0], [-1.3, 1.3], color="tab:blue", lw=1.5, ls=":")
    for k, color in enumerate(_COLORS):
        e = np.eye(3)[:, k]
        v = r @ e
        ax3.plot(*zip([0, 0, 0], e), color="0.75", lw=1.5, ls="--")
        ax3.plot(*zip([0, 0, 0], v), color=color, lw=4)
        ax3.plot([v[0]], [v[1]], [v[2]], "o", color=color, ms=7)
    ax3.plot(0.45 * np.cos(phis), 0.45 * np.sin(phis), 0 * phis, color="0.3", lw=1.2)
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.set_zlim(-1.3, 1.3)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=22, azim=-60)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("In space: turn about the z axis (blue line stays fixed)", fontsize=10)
    fig.tight_layout()
    return fig


def _rotation(axis, a, n):
    """Rotation by angle a: the 2x2 plane rotation for n = 2, else about `axis` in three dimensions."""
    if n == 2:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])
    return rotation_matrix_so3(axis, a)


def _plane_panel(ax, v0, v, r):
    """The x-y plane: the rotated unit vectors, and the real and imaginary parts of the rotated vector."""
    for col, color, name in [(0, "tab:red", "x̂"), (1, "tab:green", "ŷ")]:
        e = np.eye(2)[:, col]
        w = r @ e
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.8", lw=1.2))
        ax.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, alpha=0.6))
        ax.text(w[0] * 1.2, w[1] * 1.2, name + "'", color=color, ha="center", va="center")
    for part, ls in [(v.real, "-"), (v.imag, "--")]:
        if np.hypot(*part) > 0.05:
            ax.annotate("", xy=part, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="0.15", lw=2.2, ls=ls))
    ax.legend(
        handles=[Line2D([], [], color="0.15", lw=2.2), Line2D([], [], color="0.15", lw=2.2, ls="--")],
        labels=["Re v", "Im v"], loc="lower right", fontsize=8, frameon=False,
    )
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.9", lw=0.8)
    ax.axvline(0, color="0.9", lw=0.8)
    ax.set_title("the x-y plane", fontsize=10)


def component_planes_figure(v0, angle_deg, axis=(0, 0, 1)):
    """The components of R(angle) v0, each as an arrow in its own complex plane.

    Three components: a rotation about `axis` (default z). Two components: the plane rotation R(angle),
    with the x-y plane drawn to the left.

    The grey arrow is the component before the rotation, the coloured arrow after it, and the thin curve the
    path followed as the angle grows from 0.
    """
    v0 = np.asarray(v0, dtype=complex)
    angles = np.radians(np.linspace(0, angle_deg, max(int(angle_deg) // 2, 2)))
    n = len(v0)
    path = np.array([_rotation(axis, a, n) @ v0 for a in angles])
    v = path[-1]

    fig, axes = plt.subplots(1, n + (n == 2), figsize=(3.5 * (n + (n == 2)), 3.6))
    if n == 2:
        _plane_panel(axes[0], v0, v, _rotation(axis, np.radians(angle_deg), n))
        axes = axes[1:]
    for k, (ax, color, name) in enumerate(zip(axes, _COLORS, ("x", "y", "z"))):
        ax.plot(path[:, k].real, path[:, k].imag, color=color, lw=1.2, alpha=0.6)
        start, end = v0[k], v[k]
        ax.annotate("", xy=(start.real, start.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.75", lw=1.5))
        ax.annotate("", xy=(end.real, end.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlabel("real part")
        ax.set_ylabel("imaginary part")
        ax.set_title(f"{name} component", fontsize=10, color=color)
    fig.tight_layout()
    return fig



def circulation_figure(wt_deg):
    """The real arrow Re[v exp(-i w t)] for the three eigenvectors of J_z, and for x-hat as their sum."""
    wt = np.radians(wt_deg)
    s = 1 / np.sqrt(2)
    sweep = np.linspace(0, wt, max(int(wt_deg) // 2, 2))

    def arrow(v, ph):
        return (np.asarray(v, dtype=complex) * np.exp(-1j * ph)).real

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    cases = [
        (axes[0], (s, 1j * s), "tab:purple", "m = +1: counter-clockwise"),
        (axes[1], (s, -1j * s), "tab:orange", "m = −1: clockwise"),
    ]
    for ax, v, color, title in cases:
        r = s
        ang = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(ang), r * np.sin(ang), color="0.85", lw=1, ls=":")
        trail = np.array([arrow(v, p) for p in sweep])
        ax.plot(trail[:, 0], trail[:, 1], color=color, lw=1.5, alpha=0.6)
        tip = arrow(v, wt)
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.8))
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax = axes[2]
    ax.plot([0, 0], [-1.1, 1.1], color="0.85", lw=1, ls=":")
    zc = np.cos(wt)
    ax.annotate("", xy=(0, zc), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.8))
    ax.set_title("m = 0: along z, no circulation", fontsize=10, color="tab:blue")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax = axes[3]
    a = arrow((s, 1j * s), wt) / np.sqrt(2)
    b = arrow((s, -1j * s), wt) / np.sqrt(2)
    ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=1.8, alpha=0.7))
    ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=1.8, alpha=0.7))
    ax.annotate("", xy=a + b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="0.15", lw=2.8))
    ax.plot(a[0] + np.zeros(2), [a[1], -a[1]], color="0.7", lw=0.8, ls=":")
    ax.set_title("sum of the two: oscillation along x", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for ax in axes:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
    fig.tight_layout()
    return fig
