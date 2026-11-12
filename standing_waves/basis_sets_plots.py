"""Figures for the basis-set notebook. Each function returns a matplotlib Figure. Colours come from qmstyle.py."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import basis_sets as bs  # noqa: E402
from qmstyle import (  # noqa: E402
    ANGSTROM_PER_BOHR as AB,
    A_COLOR,
    H2_COLOR,
    H_COLOR,
    INK,
    K_COLOR,
    OMEGA_COLOR,
    PHASE_COLOR,
    POT_COLOR,
    PROB_COLOR,
)

PLUS, MINUS = H_COLOR, A_COLOR
N_MAX = 12  # largest oscillator basis in the slider
GREY = "0.55"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def morse_figure(n_basis):
    """Morse well, the ground state built from n_basis oscillator functions, its coefficients, and energy errors."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.25, 0.9, 1.0]})
    ax, bx, cx = axes
    x = bs.X_GRID
    ax.plot(x * AB, bs.morse_potential(x), color=POT_COLOR, lw=2)
    for n in range(bs.morse_bound_states()):
        ax.axhline(bs.morse_energy(n), color=GREY, lw=0.6, xmin=0.12, xmax=0.95)
    e, _ = bs.variational_levels(n_basis)
    scale = 1.6
    ax.plot(x * AB, bs.morse_energy(0) + scale * bs.morse_psi0(), color=INK, lw=1.6, ls="--", label="exact")
    ax.plot(x * AB, e[0] + scale * bs.expansion_wavefunction(n_basis, 0), color=H_COLOR, lw=2,
            label=f"{n_basis} function{'s' if n_basis > 1 else ''}")
    ax.set_xlim(-2.0 * AB, 13.0 * AB)
    ax.set_ylim(0, 7.0)
    ax.set_xlabel("$x$ (Å)")
    ax.set_ylabel("energy (hartree)")
    ax.set_title("ground state: exact and expanded", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    _style(ax)

    c = bs.variational_levels(n_basis)[1][:, 0]
    full = np.zeros(N_MAX)
    full[:n_basis] = c**2
    bx.bar(np.arange(N_MAX), full, color=H_COLOR)
    bx.set_xlim(-0.7, N_MAX - 0.3)
    bx.set_ylim(0, 1.0)
    bx.set_xticks(range(0, N_MAX, 2))
    bx.set_xlabel("$n$")
    bx.set_ylabel(r"$c_n^2$")
    bx.set_title("weight of each function", color=INK, fontsize=10)
    _style(bx)

    err = bs.energy_errors(N_MAX)
    for k, col in zip(range(3), (H_COLOR, K_COLOR, OMEGA_COLOR)):
        cx.semilogy(np.arange(1, N_MAX + 1), err[:, k], "o-", color=col, ms=4, lw=1.4, label=f"level {k}")
    cx.axvline(n_basis, color=A_COLOR, lw=1.2, ls=":")
    cx.set_xlim(0.5, N_MAX + 0.5)
    cx.set_ylim(1e-7, 20)
    cx.set_xticks(range(1, N_MAX + 1, 2))
    cx.set_xlabel("number of functions $N$")
    cx.set_ylabel("energy above exact (hartree)")
    cx.set_title("never below, and falling", color=INK, fontsize=10)
    cx.legend(frameon=False, fontsize=8, loc="lower left")
    _style(cx)
    fig.tight_layout()
    return fig


def h2plus_figure(r):
    """Left: the two atomic 1s functions and the bonding and antibonding combinations along the axis. Right: energies against R."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    z = np.linspace(-6.5, 6.5, 600)
    fa, fb = bs.slater_1s(np.abs(z + r / 2)), bs.slater_1s(np.abs(z - r / 2))
    _, c = bs.h2plus_lcao(r)
    ax.plot(z * AB, fa, color=GREY, lw=1.2, ls="--")
    ax.plot(z * AB, fb, color=GREY, lw=1.2, ls="--")
    ax.plot(z * AB, c[0, 0] * fa + c[1, 0] * fb, color=H_COLOR, lw=2.2, label="bonding $c_A=c_B$")
    ax.plot(z * AB, c[0, 1] * fa + c[1, 1] * fb, color=A_COLOR, lw=2.2, label="antibonding $c_A=-c_B$")
    ax.axhline(0, color=INK, lw=0.6)
    for pos in (-r / 2, r / 2):
        ax.plot([pos * AB], [0], "o", color=INK, ms=6)
    ax.set_xlim(-6.5 * AB, 6.5 * AB)
    ax.set_ylim(-0.7, 0.8)
    ax.set_xlabel("position along the axis (Å)")
    ax.set_ylabel(r"$\psi$")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)

    rs = np.linspace(0.9, 8.0, 300)
    bx.plot(rs * AB, bs.h2plus_total(rs, 0), color=H_COLOR, lw=2, label="bonding (2 functions)")
    bx.plot(rs * AB, bs.h2plus_total(rs, 1), color=A_COLOR, lw=2, label="antibonding (2 functions)")
    d = bs.load_data()["h2plus"]
    bx.plot(np.array(d["R"]) * AB, d["energy"], "o", color=INK, ms=4, label="large basis (PySCF)")
    bx.axhline(-0.5, color=GREY, lw=0.8, ls="--")
    bx.text(7.9 * AB, -0.485, "H + proton", ha="right", va="bottom", fontsize=8, color=INK)
    bx.axvline(r * AB, color=PHASE_COLOR, lw=1.1, ls=":")
    bx.set_xlim(0.9 * AB, 8.0 * AB)
    bx.set_ylim(-0.7, -0.2)
    bx.set_xlabel("$R$ (Å)")
    bx.set_ylabel("total energy (hartree)")
    bx.legend(frameon=False, fontsize=8, loc="lower right")
    _style(bx)
    fig.tight_layout()
    return fig


def sto_fit_figure(n):
    """Slater 1s function against its n-Gaussian fit: whole curve, near the nucleus, far tail, and the error against n."""
    alphas, coeffs, _ = bs.sto_ng_fit(n)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    ax, bx, cx, dx = axes
    r = np.linspace(0, 10, 2001)
    slater = bs.slater_1s(r)
    fit = bs.contracted_radial(r, alphas, coeffs)
    for a_, lim in ((ax, 4.0), (bx, 0.6)):
        m = r <= lim
        a_.plot(r[m] * AB, slater[m], color=INK, lw=2, ls="--", label="Slater $e^{-r}$")
        a_.plot(r[m] * AB, fit[m], color=H_COLOR, lw=2, label=f"{n} Gaussian{'s' if n > 1 else ''}")
        a_.set_xlim(0, lim * AB)
        _style(a_)
    ax.set_ylim(0, 0.6)
    ax.set_title("the whole function", color=INK, fontsize=10)
    ax.set_xlabel("$r$ (Å)")
    ax.legend(frameon=False, fontsize=9)
    bx.set_ylim(0.3, 0.6)
    bx.set_title("at the nucleus", color=INK, fontsize=10)
    bx.set_xlabel("$r$ (Å)")
    cx.semilogy(r * AB, slater, color=INK, lw=2, ls="--")
    cx.semilogy(r * AB, np.maximum(fit, 1e-30), color=H_COLOR, lw=2)
    cx.set_xlim(0, 10 * AB)
    cx.set_ylim(1e-8, 1)
    cx.set_xlabel("$r$ (Å)")
    cx.set_title("far from the nucleus (log scale)", color=INK, fontsize=10)
    _style(cx)
    ns = np.arange(1, 7)
    miss = [1 - bs.sto_ng_fit(int(k))[2] for k in ns]
    en = [bs.hydrogen_energy_of_contraction(*bs.sto_ng_fit(int(k))[:2]) + 0.5 for k in ns]
    dx.semilogy(ns, miss, "o-", color=H_COLOR, label="$1-$overlap$^2$")
    dx.semilogy(ns, en, "s-", color=OMEGA_COLOR, label="energy above $-\\frac{1}{2}$")
    dx.axvline(n, color=A_COLOR, lw=1.2, ls=":")
    dx.set_xlim(0.5, 6.5)
    dx.set_ylim(1e-7, 1)
    dx.set_xlabel("number of Gaussians $n$")
    dx.set_title("error of the fit", color=INK, fontsize=10)
    dx.legend(frameon=False, fontsize=8, loc="lower left")
    _style(dx)
    fig.tight_layout()
    return fig


def product_figure():
    """One-dimensional picture: two Gaussians multiply to a Gaussian; two exponentials multiply to a kinked shape."""
    x = np.linspace(-4, 5, 900)
    xa, xb, al, be = 0.0, 1.8, 0.9, 0.5
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    ga, gb = np.exp(-al * (x - xa) ** 2), np.exp(-be * (x - xb) ** 2)
    k, p, gam = bs.gaussian_product(al, [xa], be, [xb])
    ax.plot(x, ga, color=H2_COLOR, lw=1.6, label="Gaussian at $A$")
    ax.plot(x, gb, color=OMEGA_COLOR, lw=1.6, label="Gaussian at $B$")
    ax.plot(x, ga * gb, color=H_COLOR, lw=2.4, label="product")
    ax.plot(x[::45], (k * np.exp(-gam * (x - p[0]) ** 2))[::45], "o", color=INK, ms=4, label="one Gaussian at $P$")
    ax.set_title("Gaussians: the product is a Gaussian", color=INK, fontsize=10)
    ea, eb = np.exp(-2 * np.abs(x - xa)), np.exp(-2 * np.abs(x - xb))
    bx.plot(x, ea, color=H2_COLOR, lw=1.6)
    bx.plot(x, eb, color=OMEGA_COLOR, lw=1.6)
    prod = ea * eb
    bx.plot(x, prod / prod.max(), color=A_COLOR, lw=2.4, label="product (rescaled to 1)")
    bx.set_title("exponentials: kinks, no simple centre", color=INK, fontsize=10)
    for a_ in (ax, bx):
        a_.set_xlabel("$x$")
        a_.set_ylim(0, 1.15)
        a_.set_xlim(-4, 5)
        _style(a_)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    bx.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def shells_figure():
    """Cross sections (xz plane) of s, p_z and d_z2 shell functions, and of 1s with and without a little p_z mixed in."""
    n = 241
    xs = np.linspace(-3, 3, n)
    xx, zz = np.meshgrid(xs, xs)
    alpha = 0.45
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    names = [("s", 0, "s shell, 1 function"), ("p", 1, "p shell, 3 functions: $p_z$"), ("d", 2, "d shell, 5 functions: $d_{z^2}$")]
    for ax, (_, ell, title) in zip(axes[:3], names):
        f = bs.gaussian_shell(ell, alpha, xx, zz)
        f = f / np.abs(f).max()
        ax.contourf(xx * AB, zz * AB, f, levels=np.linspace(-1, 1, 21), cmap=_cmap(), vmin=-1, vmax=1)
        ax.set_title(title, color=INK, fontsize=10)
    s = bs.gaussian_shell(0, alpha, xx, zz)
    p = bs.gaussian_shell(1, alpha, xx, zz)
    s, p = s / np.abs(s).max(), p / np.abs(p).max()
    rho0 = s**2
    rho1 = (s + 0.5 * p) ** 2
    ax = axes[3]
    levels = np.linspace(0.05, 1.0, 8) * rho1.max()
    ax.contour(xx * AB, zz * AB, rho0, levels=levels, colors=GREY, linewidths=1.0, linestyles="--")
    ax.contour(xx * AB, zz * AB, rho1, levels=levels, colors=PLUS, linewidths=1.4)
    ax.plot([0], [0], "o", color=INK, ms=4)
    ax.set_title("density of $s$ (dashed) and $s+\\frac{1}{2} p_z$", color=INK, fontsize=10)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (Å)")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-3 * AB, 3 * AB)
        ax.set_ylim(-3 * AB, 3 * AB)
    axes[0].set_ylabel("$z$ (Å)")
    fig.tight_layout()
    return fig


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("pm", [MINUS, "#FFFFFF", PLUS])


def split_valence_figure(zeta):
    """A 1s function that swells or shrinks (exponent zeta), followed by a frozen, a double and a triple basis."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    r = np.linspace(0, 4, 600)
    a, _, _ = bs.sto_ng_fit(3)
    ax.plot(r * AB, bs.slater_1s(r, zeta), color=INK, lw=2.2, ls="--", label=f"target $e^{{-{zeta:.2f}\\,r}}$")
    styles = {"single": (A_COLOR, "single zeta (weights frozen)"), "double": (OMEGA_COLOR, "double zeta (2 weights free)"),
              "triple": (H_COLOR, "3 weights free")}
    for kind, (col, lab) in styles.items():
        ax.plot(r * AB, bs.contracted_radial(r, a, bs.split_valence_coefficients(zeta, kind)), color=col, lw=1.6, label=lab)
    ax.set_xlim(0, 4 * AB)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("$r$ (Å)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    zs = np.linspace(0.6, 2.0, 57)
    for kind, (col, lab) in styles.items():
        bx.semilogy(zs, np.maximum(bs.split_valence_errors(zs, kind), 1e-9), color=col, lw=1.8)
    bx.axvline(zeta, color=PHASE_COLOR, lw=1.1, ls=":")
    bx.axvline(1.0, color=GREY, lw=0.8)
    bx.text(1.02, 0.5, "frozen function\nfitted for $\\zeta=1$", fontsize=8, color=INK, va="top")
    bx.set_xlim(0.6, 2.0)
    bx.set_ylim(1e-6, 1)
    bx.set_xlabel(r"orbital exponent $\zeta$ (large = tight)")
    bx.set_ylabel("$1-$overlap$^2$ with the target")
    _style(bx)
    fig.tight_layout()
    return fig


def hydrogen_basis_figure(data):
    """PySCF hydrogen atom: polarizability by basis (left) and density far from the nucleus (right)."""
    rows = data["hydrogen"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    labels = [r["label"].replace(", without its p shell", "\n(p removed)").replace("def2-", "") for r in rows]
    alphas = [r["polarizability"] * AB**3 for r in rows]  # bohr^3 -> Å^3
    ax.barh(range(len(rows)), alphas, color=H_COLOR)
    ax.axvline(4.5 * AB**3, color=A_COLOR, lw=1.4, ls="--")
    ax.text(4.45 * AB**3, -0.45, "exact $\\frac{9}{2}$", color=INK, ha="right", va="bottom", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 5 * AB**3)
    ax.set_xlabel(r"polarizability $\alpha$ of the H atom (Å$^3$)")
    _style(ax)
    r = np.array(rows[0]["r"])
    bx.semilogy(r * AB, bs.density_slater(r) / AB**3, color=INK, lw=2.2, ls="--", label="exact")
    for row, col in zip(rows, (GREY, A_COLOR, None, H_COLOR, None, None, K_COLOR)):
        if col is None:
            continue
        bx.semilogy(r * AB, np.maximum(np.array(row["density"]) / AB**3, 1e-30), color=col, lw=1.6, label=row["label"])
    bx.set_xlim(0, 12 * AB)
    bx.set_ylim(1e-11 / AB**3, 1 / AB**3)
    bx.set_xlabel("distance from the nucleus $r$ (Å)")
    bx.set_ylabel(r"electron density $\rho$ (Å$^{-3}$)")
    bx.legend(frameon=False, fontsize=8, loc="upper right")
    _style(bx)
    fig.tight_layout()
    return fig


def water_basis_figure(water):
    """wB97M-V energy of water against the number of basis functions, relative to the largest basis, def2-QZVP."""
    order = ["sto-3g", "def2-svp", "def2-tzvp", "def2-tzvpd", "def2-tzvpp", "def2-tzvppd"]
    ref = water["def2-qzvp"]["e_water_Ha"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k in order:
        e = (water[k]["e_water_Ha"] - ref) * 1000
        diffuse = k.endswith("d")
        ax.semilogy([water[k]["n_basis"]], [e], "o", ms=8, color=OMEGA_COLOR if diffuse else H_COLOR)
        offset = {"def2-tzvpd": (-34, -12), "def2-tzvpp": (6, 5)}.get(k, (6, 5))
        ax.annotate(k.replace("def2-", "").upper() if k != "sto-3g" else "STO-3G", (water[k]["n_basis"], e),
                    textcoords="offset points", xytext=offset, fontsize=8, color=INK)
    ax.set_xlim(0, 125)
    ax.set_ylim(1, 2000)
    ax.set_xlabel("number of basis functions in H$_2$O")
    ax.set_ylabel("energy above def2-QZVP (milli-hartree)")
    _style(ax)
    fig.tight_layout()
    return fig
