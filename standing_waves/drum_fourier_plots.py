"""Figures for the drum Fourier example (a pattern as a sum of drum modes), notebook size and slide size."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordinates_plots import _rc, _scale  # noqa: E402
from drum_fourier import (  # noqa: E402
    M_MAX,
    RES,
    RES_SLIDE,
    TARGETS,
    approximation,
    mode_list,
    series,
    target_on_display,
)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from qmstyle import A_COLOR, ANGULAR_COLOR, H_COLOR, INK  # noqa: E402

# A stronger diverging colour map than the one of the box notebook: blue positive, red negative, white zero, with dark
# ends. The value is also shown through a square root (GAMMA = 0.5), so that small values are not washed out. The limits
# are fixed and symmetric: plus and minus the largest size of the target.
DRUM_CMAP = LinearSegmentedColormap.from_list("drum", ["#7A1405", A_COLOR, "#FFFFFF", H_COLOR, "#1F2C78"])
GAMMA = 0.5


def _show(ax, img, top, title=None, fs=None):
    z = np.sign(img) * np.minimum(np.abs(img) / top, 1.0) ** GAMMA
    ax.imshow(z, origin="lower", extent=[-1, 1, -1, 1], cmap=DRUM_CMAP, vmin=-1, vmax=1)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fs)


def _error_axes(ax, key, M, slide=False):
    s = _scale(slide)
    err = series(key)[2]
    ax.plot(np.arange(0, M_MAX + 1), err, color=H_COLOR, lw=3.5 * s)
    ax.plot([M], [err[M]], "o", color=A_COLOR, ms=12 * s, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(0, M_MAX)
    ax.set_ylim(max(err.min() * 0.7, 1e-3), 1.3)
    ax.set_xlabel("number of modes $M$")
    ax.set_ylabel("relative error")
    ax.spines[["top", "right"]].set_visible(False)


def drum_fourier_figure(key="poke", M=10, slide=False):
    """Top views of the target, of the sum of the first M modes and of the difference (one colour scale: blue positive,
    red negative), with the relative error against M."""
    s = _scale(slide)
    res = RES_SLIDE if slide else RES
    tgt, app = target_on_display(key, res), approximation(key, M, res)
    top = float(np.nanmax(np.abs(tgt)))
    with _rc(slide):
        fig = plt.figure(figsize=(18 * s, 4.4 * s) if slide else (10.5, 2.7))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.3, left=0.01, right=0.99, bottom=0.2, top=0.82)
        for i, (img, title) in enumerate(((tgt, "target"), (app, f"sum of {M} mode{'s' if M != 1 else ''}"),
                                          (tgt - app, "difference"))):
            _show(fig.add_subplot(gs[0, i]), img, top, title, 20 * s)
        ax = fig.add_subplot(gs[0, 3])
        _error_axes(ax, key, M, slide)
    return fig


def drum_fourier_modes_figure(M=10, slide=False, l_max=24, nr_max=10):
    """Which modes are in the sum: a grid of the number of angular nodes l (columns) against the number of nodal circles
    n_r (rows). Each cell holds the cosine mode (left, m_l = +l) and the sine mode (right, m_l = -l)."""
    s = _scale(slide)
    modes = mode_list()
    used = {(l, nr, m) for (l, nr, m, _) in modes[:M]}
    last = modes[M - 1][:3] if M > 0 else None
    with _rc(slide):
        fig = plt.figure(figsize=(8 * s, 4.6 * s) if slide else (6.2, 3.2))
        ax = fig.add_axes([0.1, 0.17, 0.88, 0.75])
        for l in range(l_max + 1):
            for nr in range(nr_max + 1):
                for m, dx in ((l, -0.17), (-l, 0.17)) if l > 0 else ((0, 0.0),):
                    inc = (l, nr, m) in used
                    col = A_COLOR if (l, nr, m) == last else H_COLOR
                    ax.scatter([l + dx], [nr], s=((300 if slide else 90) if inc else (130 if slide else 40)) * s**2 * (1 if l_max < 12 else 0.35),
                               color=col if inc else "white", edgecolor=col if inc else "0.75", linewidth=2 * s, zorder=3)
        ax.set_xticks(range(0, l_max + 1, 1 if l_max < 12 else 2))
        ax.set_yticks(range(nr_max + 1))
        ax.set_xlim(-0.6, l_max + 0.6)
        ax.set_ylim(-0.6, nr_max + 0.6)
        ax.set_xlabel(r"$\ell$ (angular nodes)", color=ANGULAR_COLOR)
        ax.tick_params(axis="x", colors=ANGULAR_COLOR)
        ax.set_ylabel(r"$n_r$ (circles)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"the first {M} modes (by frequency)", fontsize=18 * s)
    return fig


def drum_fourier_steps_figure(key="poke", steps=(1, 3, 10, 30), slide=True):
    """The target and the sums of the first M modes for a few M, one colour scale, in one row."""
    s = _scale(slide)
    tgt = target_on_display(key, RES_SLIDE)
    top = float(np.nanmax(np.abs(tgt)))
    with _rc(slide):
        fig = plt.figure(figsize=(18 * s, 4.4 * s))
        gs = fig.add_gridspec(1, len(steps) + 1, wspace=0.05, left=0.01, right=0.99, bottom=0.02, top=0.8)
        _show(fig.add_subplot(gs[0, 0]), tgt, top, "target", 22 * s)
        for i, M in enumerate(steps):
            _show(fig.add_subplot(gs[0, i + 1]), approximation(key, M, RES_SLIDE), top, f"$M={M}$", 22 * s)
    return fig


def drum_fourier_error_figure(key="poke", slide=True):
    """Relative error against the number of modes, for all four targets."""
    s = _scale(slide)
    colors = {"poke": H_COLOR, "pluck": A_COLOR, "two": "#FF8021", "blob": "#6B913C", "rose": "#7030A0"}
    with _rc(slide):
        fig = plt.figure(figsize=(8 * s, 6 * s))
        ax = fig.add_axes([0.17, 0.15, 0.8, 0.78])
        for k in TARGETS:
            ax.plot(np.arange(0, M_MAX + 1), series(k)[2], color=colors[k], lw=3.5 * s, label=TARGETS[k].split(" (")[0])
        ax.set_yscale("log")
        ax.set_xlabel("number of modes $M$")
        ax.set_ylabel("relative error")
        ax.set_xlim(0, M_MAX)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=13 * s, loc="lower left")
    return fig


def drum_fourier_gif_frames(key="poke", dpi=60, ms=(1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 32, 37, 42, 48, 54, 60)):
    """Frames of the sum growing with M (target on the left, sum on the right); returns Pillow frames."""
    from hydrogen_plots import _fig_to_image

    tgt = target_on_display(key)
    top = float(np.nanmax(np.abs(tgt)))
    err = series(key)[2]
    frames = []
    for M in list(ms) + [ms[-1]] * 4:
        with _rc(True):
            fig = plt.figure(figsize=(8.0, 3.7))
            gs = fig.add_gridspec(1, 2, wspace=0.05, left=0.02, right=0.98, bottom=0.02, top=0.82)
            _show(fig.add_subplot(gs[0, 0]), tgt, top, "target", 20)
            _show(fig.add_subplot(gs[0, 1]), approximation(key, M), top, f"$M={M}$,  error {100 * err[M]:.0f}%", 20)
        frames.append(_fig_to_image(fig, dpi))
        plt.close(fig)
    return frames
