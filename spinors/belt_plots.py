"""Figures for the belt trick and the space of rotations. Each function returns a matplotlib Figure."""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dirac_belt import (
    belt_centerline,
    belt_frames,
    double_turn_quaternions,
    loop_quaternions,
    ribbon_edges,
    rotation_vector,
    single_turn_quaternions,
    split_at_surface,
    surface_crossings,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import BACK_COLOR, FRONT_COLOR, PLATE_COLOR  # noqa: E402


def _belt_geometry(phi, n=120, length=3.0, width=0.9):
    frames = belt_frames(phi, n)
    center = belt_centerline(frames, length=length)
    left, right = ribbon_edges(frames, center, width=width)
    normal = frames[:, :, 2]
    return frames, center, left, right, normal


@lru_cache(maxsize=None)
def _extent(length=3.0, width=0.9, samples=240):
    """Bounding box of the belt over a full 4*pi turn, so the axes do not move between frames."""
    pts = []
    for phi in np.linspace(0, 4 * np.pi, samples):
        _, c, a, b, _ = _belt_geometry(phi, n=40, length=length, width=width)
        pts.append(np.vstack([c, a, b]))
    pts = np.vstack(pts)
    return pts.min(axis=0), pts.max(axis=0)


def _draw_ribbon(ax, left, right, normal, thickness=0.012):
    """Draw the belt with a red front side and a blue back side."""
    quads, colors = [], []
    for sign, color in ((+1, FRONT_COLOR), (-1, BACK_COLOR)):
        off = sign * thickness * normal
        for i in range(len(left) - 1):
            quads.append([left[i] + off[i], right[i] + off[i], right[i + 1] + off[i + 1], left[i + 1] + off[i + 1]])
            colors.append(color)
    # One collection, so that the two faces are depth-sorted together and the side facing the viewer shows.
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors=colors, linewidths=0.3))


def _draw_plate(ax, position, phi, size=(1.1, 0.7), thickness=0.04):
    """The plate: a rectangle lying in the xy-plane, turned through phi about z, with a pointer along its x axis."""
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    hx, hy = size[0] / 2, size[1] / 2
    corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]) @ rot.T
    top = [np.r_[position[:2] + p, position[2] + thickness] for p in corners]
    ax.add_collection3d(Poly3DCollection([top], facecolor=PLATE_COLOR, edgecolor="black", linewidth=0.8, alpha=0.95))
    tip = np.r_[position[:2] + rot @ np.array([hx * 1.5, 0]), position[2] + thickness]
    base = np.r_[position[:2], position[2] + thickness]
    ax.plot(*zip(base, tip), color="black", lw=2)
    ax.plot([tip[0]], [tip[1]], [tip[2]], "o", color="black", ms=5)


def belt_figure(phi, length=3.0, width=0.9, view=(28, -28)):
    """The belt between a fixed wall and a plate that has been turned through phi about the vertical axis."""
    lo, hi = _extent(length, width)
    frames, center, left, right, normal = _belt_geometry(phi, length=length, width=width)
    fig = plt.figure(figsize=(7.5, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    _draw_ribbon(ax, left, right, normal)
    ax.plot(*center.T, color="black", lw=0.6, alpha=0.5)
    _draw_plate(ax, center[-1], phi)
    # the wall: a small block at the fixed end
    wall = np.array([[-0.6, -0.12], [0.6, -0.12], [0.6, 0.0], [-0.6, 0.0]])
    ax.add_collection3d(Poly3DCollection([[(x, y, -0.35) for x, y in wall]], facecolor="#4d5656", alpha=0.9))
    ax.add_collection3d(Poly3DCollection([[(x, y, 0.35) for x, y in wall]], facecolor="#4d5656", alpha=0.9))
    for z in (-0.35, 0.35):
        ax.plot([-0.6, 0.6], [0, 0], [z, z], color="#4d5656", lw=4)
    pad = 0.5
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect((hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, hi[2] - lo[2] + 2 * pad), zoom=1.8)
    ax.set_axis_off()
    ax.view_init(*view)
    turns = phi / (2 * np.pi)
    ax.set_title(rf"plate turned through $\alpha = {turns:.2f}\times 2\pi$  ({np.degrees(phi):.0f}$^\circ$)")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    return fig


# ---------------------------------------------------------------------------
# The space of rotations as a ball
# ---------------------------------------------------------------------------


def _draw_ball(ax):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.pi * np.outer(np.cos(u), np.sin(v))
    y = np.pi * np.outer(np.sin(u), np.sin(v))
    z = np.pi * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="0.85", alpha=0.18, linewidth=0, shade=False)
    for ang in np.linspace(0, np.pi, 4, endpoint=False):
        t = np.linspace(0, 2 * np.pi, 120)
        ax.plot(np.pi * np.cos(t) * np.cos(ang), np.pi * np.cos(t) * np.sin(ang), np.pi * np.sin(t), color="0.8", lw=0.5)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_axis_off()


def _draw_loop(ax, q, color, lw=2.5, label=None):
    """Draw a path of rotations in the ball; where it leaves one surface point it re-enters at the opposite one."""
    v = rotation_vector(q)
    pieces = split_at_surface(v, q)
    for k, piece in enumerate(pieces):
        ax.plot(*piece.T, color=color, lw=lw, label=label if k == 0 else None)
    for a, b in zip(pieces[:-1], pieces[1:]):
        ax.plot([a[-1, 0], b[0, 0]], [a[-1, 1], b[0, 1]], [a[-1, 2], b[0, 2]], color=color, lw=1.2, ls=":")
        ax.plot(*a[-1:].T, "o", color=color, ms=6)
        ax.plot(*b[:1].T, "o", color=color, ms=6, mfc="white")
    ax.plot([0], [0], [0], "k+", ms=10)


def single_turn_ball_figure(view=(18, -55)):
    """The plate turned through 2*pi: a path that leaves the ball and re-enters at the opposite point."""
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_ball(ax)
    _draw_loop(ax, single_turn_quaternions(), FRONT_COLOR)
    ax.view_init(*view)
    ax.set_title(r"plate turned through $2\pi$, loop crosses the surface 1 time")
    fig.tight_layout()
    return fig


def double_turn_ball_figure(kappa, view=(18, -55)):
    """The plate turned through 4*pi, with the loop shrunk by the amount set by kappa."""
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_ball(ax)
    q = double_turn_quaternions() if np.isclose(kappa, np.pi / 2) else loop_quaternions(kappa)
    _draw_loop(ax, q, BACK_COLOR)
    ax.view_init(*view)
    n = surface_crossings(q)
    ax.set_title(rf"plate turned through $4\pi$, loop crosses the surface {n} time{'' if n == 1 else 's'}")
    fig.tight_layout()
    return fig
