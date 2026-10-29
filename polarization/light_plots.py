"""Matplotlib figures of the polarisation notebook. Colours come from qmstyle (one colour per quantity)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import qmstyle as qs  # noqa: E402

from light import (  # noqa: E402
    counter_rotating_arrows,
    ellipse_axes,
    field,
    from_helicity,
    helicity_amplitudes,
    jones,
    phase_factor,
    retarder,
    ring_radius,
    rotation_2d,
    vortex_field,
)

PLUS_COLOR = qs.CX_COLOR      # the m_s = +1 amplitude (a complex number, drawn as an arrow)
MINUS_COLOR = qs.CX2_COLOR    # the m_s = -1 amplitude: a darker shade of the same quantity
FIELD_COLOR = qs.H_COLOR      # the real field: a height
FIELD2_COLOR = qs.H2_COLOR
MARK_COLOR = qs.A_COLOR       # highlights: the field tip now, the plate's torque

_ARROW = dict(arrowstyle="-|>", lw=2.4, shrinkA=0, shrinkB=0)


def _arrow(ax, tail, head, color, lw=2.4, alpha=1.0, ls="-"):
    ax.annotate(
        "", xy=head, xytext=tail,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0, ls=ls),
    )


def _plane(ax, lim=1.3, title=None):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.9", lw=0.8)
    ax.axvline(0, color="0.9", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def _trace(j, n=400):
    e = field(j, np.linspace(0, 2 * np.pi, n))
    return e[0], e[1]


def _draw_ellipse(ax, j, wt=0.0, color=FIELD_COLOR, tip=True, ls="-", lw=2.0):
    ax.plot(*_trace(j), color=color, lw=lw, ls=ls)
    if tip:
        e = field(j, wt)
        _arrow(ax, (0, 0), tuple(e), MARK_COLOR, lw=2.6)
        ax.plot([e[0]], [e[1]], "o", color=MARK_COLOR, ms=6)


def phasor_panel(ax, j, title="the two amplitudes"):
    """The amplitudes on m_s = +1 and m_s = -1 as arrows in the complex plane, with the angle between them."""
    a_plus, a_minus = helicity_amplitudes(j)
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.sin(circ), color="0.85", lw=1)
    _plane(ax, 1.25, title)
    for a, color, label in [(a_plus, PLUS_COLOR, "$m_s=+1$"), (a_minus, MINUS_COLOR, "$m_s=-1$")]:
        if abs(a) > 1e-9:
            _arrow(ax, (0, 0), (a.real, a.imag), color)
            ax.text(a.real * 1.22 + 0.04, a.imag * 1.22 + 0.04, label, color=color, fontsize=9, ha="center")
    ax.set_xlabel(f"$P(+1)={abs(a_plus) ** 2:.2f}$,  $P(-1)={abs(a_minus) ** 2:.2f}$", fontsize=9)


def ellipse_figure(beta_deg, delta_deg, wt_deg):
    """Left: the path of the field tip in the plane across the beam. Right: E_x and E_y against time."""
    j = jones(np.radians(beta_deg), np.radians(delta_deg))
    wt = np.radians(wt_deg)
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    _plane(ax, 1.15, "the field in the plane across the beam")
    ax.annotate("", xy=(1.1, 0), xytext=(-1.1, 0), arrowprops=dict(arrowstyle="-", color="0.9"))
    _draw_ellipse(ax, j, wt)
    ax.set_xlabel("$E_x$ (right), $E_y$ (up)", fontsize=9)

    t = np.linspace(0, 2 * np.pi, 400)
    e = field(j, t)
    ax_t.plot(np.degrees(t), e[0], color=FIELD_COLOR, lw=2, label="$E_x$")
    ax_t.plot(np.degrees(t), e[1], color=FIELD2_COLOR, lw=2, label="$E_y$")
    cur = field(j, wt)
    ax_t.axvline(wt_deg, color=MARK_COLOR, lw=1.2)
    ax_t.plot([wt_deg, wt_deg], cur, "o", color=MARK_COLOR, ms=6)
    ax_t.set_xlim(0, 360)
    ax_t.set_ylim(-1.15, 1.15)
    ax_t.set_xticks([0, 90, 180, 270, 360])
    ax_t.axhline(0, color="0.9", lw=0.8)
    ax_t.set_xlabel("time, $\\omega t$ (degrees)", fontsize=9)
    ax_t.legend(loc="upper right", fontsize=9, ncol=2)
    ax_t.set_title("the two components over one period", fontsize=10)
    fig.tight_layout()
    return fig


def rotated_state_figure(j0, alpha_deg, label=""):
    """Turn a state through alpha about the beam. Left: its ellipse (dashed: before). Right: the two amplitudes."""
    alpha = np.radians(alpha_deg)
    j = rotation_2d(alpha) @ j0
    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(8.4, 4))
    _plane(ax, 1.15, "the field in the plane across the beam" + (f"\n{label}" if label else ""))
    _draw_ellipse(ax, j0, tip=False, color="0.7", ls="--", lw=1.5)
    _draw_ellipse(ax, j, wt=0.0)
    ax.set_xlabel("arrow: the field at $\\omega t=0$", fontsize=9)
    phasor_panel(ax_p, j)
    fig.tight_layout()
    return fig


def two_circles_figure(alpha_deg, chi_deg, wt_deg):
    """The field as the sum of an m_s=+1 arrow (counterclockwise) and an m_s=-1 arrow (clockwise)."""
    alpha, chi, wt = np.radians(alpha_deg), np.radians(chi_deg), np.radians(wt_deg)
    j = from_helicity(np.cos(chi) * np.exp(-1j * alpha), np.sin(chi) * np.exp(1j * alpha))
    p, m = counter_rotating_arrows(j, wt)
    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(8.8, 4.2))
    _plane(ax, 1.15, "two turning arrows add up to the field")
    # the path of each arrow's tip, and of the sum
    ts = np.linspace(0, 2 * np.pi, 300)
    pa = np.array([counter_rotating_arrows(j, t)[0] for t in ts])
    ax.plot(pa[:, 0], pa[:, 1], color=PLUS_COLOR, lw=1, alpha=0.35)
    ax.plot(*_trace(j), color=FIELD_COLOR, lw=2)
    _arrow(ax, (0, 0), tuple(p), PLUS_COLOR)
    _arrow(ax, tuple(p), tuple(p + m), MINUS_COLOR)
    _arrow(ax, (0, 0), tuple(p + m), MARK_COLOR, lw=2.8)
    ax.plot([p[0] + m[0]], [p[1] + m[1]], "o", color=MARK_COLOR, ms=6)
    major, minor, angle = ellipse_axes(j)
    if minor < 0.999 * major:
        ax.plot([-major * np.cos(angle), major * np.cos(angle)], [-major * np.sin(angle), major * np.sin(angle)],
                color=qs.PHASE_COLOR, lw=1, ls=":")
    ax.set_xlabel(f"major axis at ${np.degrees(angle) % 180:.0f}^\\circ$;  relative phase "
                  f"${(2 * alpha_deg) % 360:.0f}^\\circ$", fontsize=9)
    phasor_panel(ax_p, j)
    fig.tight_layout()
    return fig


def longitudinal_figure(wt_deg):
    """A plane wave along z with the field across the beam (allowed) and along it (needs charge)."""
    wt = np.radians(wt_deg)
    z = np.linspace(0, 4 * np.pi, 17)
    zz = np.linspace(0, 4 * np.pi, 400)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True)
    for ax in (ax1, ax2):
        ax.axhline(0, color="0.85", lw=1)
        ax.set_yticks([])
        ax.set_xlim(-0.6, 4 * np.pi + 0.6)
    ax1.set_ylim(-1.5, 1.5)
    ax1.plot(zz, np.cos(zz - wt), color=FIELD_COLOR, lw=1, alpha=0.5)
    for zi in z:
        v = np.cos(zi - wt)
        _arrow(ax1, (zi, 0), (zi, v), FIELD_COLOR, lw=2)
    ax1.set_title("field across the direction of travel ($E_x$, drawn up and down): allowed", fontsize=10)

    ax2.set_ylim(-1.5, 1.5)
    scale = 0.75
    for zi in z:
        v = scale * np.cos(zi - wt)
        _arrow(ax2, (zi, 0), (zi + v, 0), FIELD_COLOR, lw=2)
    ax2.plot(zz, -np.sin(zz - wt), color=MARK_COLOR, lw=2)
    ax2.text(4 * np.pi + 0.5, 1.15, "divergence $\\partial E_z/\\partial z$", color=MARK_COLOR, fontsize=9, ha="right")
    ax2.set_title("field along the direction of travel ($E_z$, drawn along the beam): arrows pile up", fontsize=10)
    ax2.set_xlabel("position along the beam, $z$", fontsize=9)
    fig.tight_layout()
    return fig


def plate_figure(j_in, gamma_deg, label=""):
    """A wave plate in the beam: the field before (dashed) and after, and the probabilities of m_s = +1, -1."""
    gamma = np.radians(gamma_deg)
    j_out = retarder(gamma) @ j_in
    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(8.4, 4), gridspec_kw=dict(width_ratios=[1, 1]))
    _plane(ax, 1.15, "the field across the beam" + (f"\n{label}" if label else ""))
    _draw_ellipse(ax, j_in, tip=False, color="0.7", ls="--", lw=1.6)
    _draw_ellipse(ax, j_out, wt=0.0)
    ax.annotate("", xy=(1.1, 0), xytext=(-1.1, 0), arrowprops=dict(arrowstyle="-", color=qs.PLATE_COLOR, lw=2, ls=":"))
    ax.text(1.12, 0.05, "fast axis", fontsize=8, ha="right", va="bottom", color="0.4")
    ax.set_xlabel("dashed: before the plate; solid: after", fontsize=9)

    pi_ = [abs(a) ** 2 for a in helicity_amplitudes(j_in)]
    po_ = [abs(a) ** 2 for a in helicity_amplitudes(j_out)]
    pi_ = np.array(pi_) / sum(pi_)
    po_ = np.array(po_) / sum(po_)
    x = np.arange(2)
    ax_b.bar(x - 0.18, pi_, 0.34, color="0.8", label="before")
    ax_b.bar(x + 0.18, po_, 0.34, color=[PLUS_COLOR, MINUS_COLOR], label="after")
    ax_b.set_xticks(x, ["$m_s=+1$", "$m_s=-1$"])
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("probability of finding each value", fontsize=9)
    ax_b.set_title("which value of $m_s$ a photon has", fontsize=10)
    ax_b.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def vortex_figure(m_l):
    """A twisted beam with orbital m_l. Left: phase (colour) and brightness. Right: brightness along a radius."""
    n = 241
    ext = 2.8
    xs = np.linspace(-ext, ext, n)
    x, y = np.meshgrid(xs, xs)
    psi = vortex_field(m_l, x, y)
    amp = np.abs(psi)
    amp = amp / max(amp.max(), 1e-12)
    hue = (np.angle(psi) % (2 * np.pi)) / (2 * np.pi)
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), amp ** 0.8], axis=-1))
    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(8.6, 4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax.imshow(rgb, extent=[-ext, ext, -ext, ext], origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("phase (colour round the axis) and brightness", fontsize=10)
    r = np.linspace(0, ext, 400)
    inten = np.abs(vortex_field(m_l, r, 0 * r)) ** 2
    ax_r.plot(r, inten / max(inten.max(), 1e-12), color=qs.PROB_COLOR, lw=3)
    if m_l:
        ax_r.axvline(ring_radius(m_l), color=MARK_COLOR, lw=1, ls=":")
    ax_r.set_xlabel("distance from the axis (units of the beam width)", fontsize=9)
    ax_r.set_ylabel("brightness", fontsize=9)
    ax_r.set_ylim(0, 1.1)
    ax_r.set_title("brightness along a radius", fontsize=10)
    fig.tight_layout()
    return fig


def phase_clock_figure(alpha_over_pi):
    """Phase factors exp(-i m alpha) of light (m = +-1) and of the electron (m = +-1/2) after a turn through alpha."""
    alpha = alpha_over_pi * np.pi
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    circ = np.linspace(0, 2 * np.pi, 200)
    for ax, ms, title in [(axes[0], (1, -1), "light: $m_s=\\pm1$"), (axes[1], (0.5, -0.5), "electron: $m_s=\\pm\\frac{1}{2}$")]:
        ax.plot(np.cos(circ), np.sin(circ), color="0.85", lw=1)
        _plane(ax, 1.3, title)
        for m, color in zip(ms, (PLUS_COLOR, MINUS_COLOR)):
            ts = np.linspace(0, alpha, max(int(abs(alpha) * 40), 2))
            ax.plot(np.cos(m * ts), -np.sin(m * ts), color=color, lw=1.2, alpha=0.4)
            f = phase_factor(m, alpha)
            _arrow(ax, (0, 0), (f.real, f.imag), color, lw=2.6)
        ax.plot([1], [0], "o", color="0.3", ms=4)
        ax.text(1.05, -0.12, "start", fontsize=8, color="0.3")
    fig.tight_layout()
    return fig
