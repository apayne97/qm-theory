"""Figures for the cycle of quarter turns and the loop of small turns. Each function returns a matplotlib Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spinors import cycle_length, loop_frames, rotation_axis_angle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import A_COLOR, CX_COLOR, CX2_COLOR  # noqa: E402

_NAMES = {1: "1", 1j: "i", -1: "−1", -1j: "−i"}


def _value_label(z):
    for value, name in _NAMES.items():
        if abs(z - value) < 1e-9:
            return name
    return None


def cycle_figure(m, k, value_labels=False):
    """The phase factor exp(-i m alpha) after k quarter turns, and the numbers visited so far.

    All the different numbers of the cycle are marked (grey dots). The arrows join the numbers visited in
    successive quarter turns; the thick arrow from the origin is the phase factor now. With `value_labels` the
    dots carry the values 1, i, -1, -i (only for the cycle of i); otherwise they carry the number of quarter
    turns at which they are first reached.
    """
    n = cycle_length(m)
    factor = np.exp(-1j * m * np.pi / 2)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="0.85", lw=1)
    ax.axhline(0, color="0.92", lw=0.8)
    ax.axvline(0, color="0.92", lw=0.8)
    for j in range(n):
        z = factor**j
        ax.plot([z.real], [z.imag], "o", color="0.65", ms=7, zorder=2)
        label = (_value_label(z) if value_labels else None) or str(j)
        ax.text(1.2 * z.real, 1.2 * z.imag, label, ha="center", va="center", fontsize=11, color="0.3")
    for j in range(k):
        a, b = factor**j, factor ** (j + 1)
        ax.annotate(
            "", xy=(b.real, b.imag), xytext=(a.real, a.imag), zorder=3,
            arrowprops=dict(arrowstyle="-|>", color=CX2_COLOR, lw=1.6, alpha=0.8, shrinkA=6, shrinkB=6,
                            connectionstyle="arc3,rad=0.25"),
        )
    z = factor**k
    ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=CX_COLOR, lw=3, shrinkA=0, shrinkB=0))
    ax.plot([z.real], [z.imag], "o", color=A_COLOR, ms=9, zorder=5)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"after {k} quarter turns", fontsize=10)
    fig.tight_layout()
    return fig


_F = np.array(
    [(-0.5, -0.7), (-0.2, -0.7), (-0.2, -0.05), (0.3, -0.05), (0.3, 0.15), (-0.2, 0.15), (-0.2, 0.45),
     (0.5, 0.45), (0.5, 0.7), (-0.5, 0.7)]
)
_SHAPE = np.c_[_F, np.zeros(len(_F))]
_COLORS = ("tab:red", "tab:green", "tab:blue")


def _draw_plate(ax, r, title, ghost=True):
    if ghost:
        ax.add_collection3d(Poly3DCollection([_SHAPE], facecolor="none", edgecolor="0.65", linestyles="--", lw=1.2))
    ax.add_collection3d(Poly3DCollection([_SHAPE @ r.T], facecolor="#f4c27a", edgecolor="0.25", lw=1.2, alpha=0.9))
    for k, (color, name) in enumerate(zip(_COLORS, ["x'", "y'", "z'"])):
        v = r[:, k]
        ax.plot(*zip([0, 0, 0], v), color=color, lw=3)
        ax.text(*(1.18 * v), name, color=color, fontsize=9, ha="center")
        e = np.eye(3)[:, k]
        ax.plot(*zip([0, 0, 0], e), color="0.8", lw=1, ls=":")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.view_init(elev=40, azim=-30)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def small_loop_figure(eps_deg):
    """A loop of four small turns applied to the letter F, and the net result seen from above."""
    eps = np.radians(eps_deg)
    frames = loop_frames(eps)
    titles = [
        "start",
        "1. $-\\epsilon$ about $y$",
        "2. $-\\epsilon$ about $x$",
        "3. $+\\epsilon$ about $y$",
        "4. $+\\epsilon$ about $x$",
    ]
    fig = plt.figure(figsize=(10, 6.6))
    for i, (title, r) in enumerate(zip(titles, frames)):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        _draw_plate(ax, r, title, ghost=bool(i))
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    _draw_plate(ax, frames[-1], "the net result, seen from above")
    ax.view_init(elev=88, azim=-90)
    axis, angle = rotation_axis_angle(frames[-1])
    phis = np.linspace(0, angle, 40)
    ax.plot(0.62 * np.cos(phis), 0.62 * np.sin(phis), 0 * phis + 0.01, color="tab:blue", lw=3)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0, hspace=0.08)
    return fig
