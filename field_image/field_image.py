"""A wireframe picture of a field with one localised excitation, for slides.

The field is a gently rippling surface. A particle is drawn as a localised bump of that surface: a
two-dimensional wave packet: concentric rings (a Bessel function) under a Gaussian envelope.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qmstyle import (  # noqa: E402
    A_COLOR,
    H_COLOR,
    INK,
)

def bessel_j0(x, samples=256):
    """Bessel function J0(x) from its integral (1/pi) * integral over [0, pi] of cos(x sin t) dt (periodic, so accurate)."""
    x = np.asarray(x, dtype=float)
    t = np.linspace(0, np.pi, samples, endpoint=False) + np.pi / (2 * samples)
    return np.mean(np.cos(x[..., None] * np.sin(t)), axis=-1)


def field_height(x, y, amplitude=1.0, width=3.0, ripple=2.2, ripple_amplitude=0.0, seed=3):
    """Height of the field at (x, y): a smooth background plus a bump at the origin.

    The bump is amplitude * J0(ripple * r) * exp(-r^2 / (2 width^2)), with r the distance from the origin: a
    sharp central peak surrounded by rings that fade with distance (J0 is the circular wave, and the Gaussian
    keeps it localised). The background is a few slow sine waves of small amplitude, so the field looks gently
    uneven far from the bump.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    bump = amplitude * bessel_j0(ripple * r) * np.exp(-(r**2) / (2 * width**2))
    rng = np.random.default_rng(seed)
    background = np.zeros_like(x)
    for _ in range(4):
        kx, ky = rng.uniform(-0.35, 0.35, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        background = background + ripple_amplitude * np.sin(kx * x + ky * y + phase) / 4
    return bump + background


def field_figure(color=A_COLOR, label=None, size=8.0, grid=80, extent=8.0, amplitude=1.0, width=3.0,
                 ripple=2.2, ripple_amplitude=0.14, elev=24, azim=-62, linewidth=0.7):
    """Wireframe picture of the field with one excitation, on a transparent background."""
    x = np.linspace(-extent, extent, grid)
    xx, yy = np.meshgrid(x, x)
    zz = field_height(xx, yy, amplitude, width, ripple, ripple_amplitude)
    fig = plt.figure(figsize=(size, size * 0.62))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    ax.plot_wireframe(xx, yy, zz, rstride=1, cstride=1, color=color, linewidth=linewidth, alpha=0.9)
    z_low, z_high = -amplitude * 0.7, amplitude * (2.6 if label else 1.35)
    ax.set_box_aspect((1, 1, 0.55 * (z_high - z_low) / 2.05))  # keeps the height of the bump the same with or without a label
    ax.set_proj_type("persp", focal_length=0.5)
    ax.view_init(elev=elev, azim=azim)
    top = float(field_height(0.0, 0.0, amplitude, width, ripple, ripple_amplitude))
    if label:
        stem = amplitude * 1.1
        ax.plot([0, 0], [0, 0], [top, top + stem], color=color, lw=1.8)
        ax.plot([0], [0], [top], "o", color=color, ms=6)
        ax.plot([0], [0], [top + stem], "o", color=color, ms=5)
        ax.text(0.4, 0, top + stem + 0.05 * amplitude, label, color=color, fontsize=15, fontweight="bold",
                family="serif", va="bottom", ha="left")
    ax.set_zlim(z_low, z_high)
    return fig


def save_field(fig, path_stem, dpi=300):
    """Write `<stem>.svg` and `<stem>.png` with a transparent background and no extra margin."""
    fig.savefig(f"{path_stem}.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path_stem}.png", transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
