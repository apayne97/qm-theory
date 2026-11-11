"""Figures for the bridge notebook `from_atoms_to_molecules.py`. Each function returns a matplotlib Figure. Colours come from qmstyle.py."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import molecules as mol  # noqa: E402
from qmstyle import A_COLOR, H_COLOR, INK, K_COLOR, OMEGA_COLOR, PHASE_COLOR, POT_COLOR  # noqa: E402

PLUS, MINUS = H_COLOR, A_COLOR
GREY = "0.55"
STATE_MAX = 2  # highest nuclear level in the slider
MASS_RANGE = (0.25, 4.0)  # nuclear mass in units of the real one


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


# ---------------------------------------------------------------------------------------------------------------
# Section 2

def born_oppenheimer_figure(state=0, mass_factor=1.0):
    """Left: the electronic energy of H2+ as a potential for the nuclei, the nuclear levels in it and one nuclear wave.
    Right: spacing of the two lowest nuclear levels against nuclear mass, next to the binding energy of the electron."""
    fit = mol.h2plus_fit()
    r_pts, e_pts = mol.h2plus_points()
    r, levels, chi = mol.nuclear_levels(STATE_MAX + 1, fit, mass=mol.REDUCED_MASS_H2PLUS * mass_factor)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    ax.plot(r, fit(r), color=POT_COLOR, lw=2, label=r"$E_{\mathrm{el}}(R)+1/R$ (electron in the field of the fixed protons)")
    ax.plot(r_pts, e_pts, "o", color=INK, ms=3.5)
    for k, e in enumerate(levels):
        ax.plot([r[0], r[-1]], [e, e], color=GREY if k != state else OMEGA_COLOR, lw=0.8 if k != state else 1.6)
    scale = 0.55 * (levels[1] - levels[0]) / np.abs(chi[:, state]).max()
    ax.plot(r, levels[state] + scale * chi[:, state], color=OMEGA_COLOR, lw=2, label=f"nuclear wave, level {state}")
    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(-0.625, -0.46)
    ax.set_xlabel("proton separation $R$ (bohr)")
    ax.set_ylabel("energy (hartree)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)

    factors = np.geomspace(*MASS_RANGE, 12)
    spacing = []
    for f in factors:
        _, e2, _ = mol.nuclear_levels(2, fit, mass=mol.REDUCED_MASS_H2PLUS * f)
        spacing.append(e2[1] - e2[0])
    binding = -0.5 - mol.h2plus_minimum(fit)[1]
    bx.loglog(factors, spacing, color=OMEGA_COLOR, lw=2, label="spacing of the two lowest nuclear levels")
    bx.axhline(binding, color=H_COLOR, lw=1.4, label="binding energy of the electron")
    _, e2, _ = mol.nuclear_levels(2, fit, mass=mol.REDUCED_MASS_H2PLUS * mass_factor)
    bx.plot([mass_factor], [e2[1] - e2[0]], "o", color=PHASE_COLOR, ms=8)
    bx.axvline(1.0, color=GREY, lw=0.8, ls=":")
    bx.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
    bx.minorticks_off()
    bx.set_xlabel("nuclear mass (real mass = 1)")
    bx.set_ylabel("energy (hartree)")
    bx.legend(frameon=False, fontsize=8, loc="center right")
    _style(bx)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Section 3

def helium_figure(zeta=1.6875):
    """Left: energy of the helium atom against the orbital exponent zeta, with and without electron repulsion.
    Right: radial density 4 pi r^2 rho of the orbital for the chosen zeta against the unscreened zeta = 2."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9))
    zs = np.linspace(1.0, 2.6, 400)
    ax.plot(zs, mol.helium_energy(zs, 2.0) - 5 * zs / 8, color=GREY, lw=1.6, label="without the repulsion between the electrons")
    ax.plot(zs, mol.helium_energy(zs), color=H_COLOR, lw=2, label="with repulsion, both electrons in one orbital")
    ax.axhline(mol.HELIUM_HF_LIMIT, color=K_COLOR, lw=1, ls="--", label="best possible one-orbital-per-electron wave (HF limit)")
    ax.axhline(mol.HELIUM_EXACT, color=A_COLOR, lw=1, ls="--", label="exact")
    ax.plot([zeta], [mol.helium_energy(zeta)], "o", color=PHASE_COLOR, ms=8)
    ax.plot([mol.helium_best_zeta()], [mol.helium_best_energy()], "x", color=INK, ms=8)
    ax.set_xlim(1.0, 2.6)
    ax.set_ylim(-4.3, -0.4)
    ax.set_xlabel(r"orbital exponent $\zeta$ ($\varphi\propto e^{-\zeta r}$)")
    ax.set_ylabel("energy (hartree)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
    _style(ax)

    r = np.linspace(0, 4, 400)
    dens = lambda z: 4 * z**3 * r**2 * np.exp(-2 * z * r)  # 4 pi r^2 (z^3/pi) exp(-2 z r)
    bx.plot(r, dens(2.0), color=GREY, lw=1.6, ls="--", label=r"$\zeta=2$ (bare nucleus)")
    bx.plot(r, dens(zeta), color=PHASE_COLOR, lw=2.2, label=rf"$\zeta={zeta:.3f}$")
    bx.set_xlabel("$r$ (bohr)")
    bx.set_ylabel(r"probability per bohr, $4\pi r^2|\varphi|^2$")
    bx.legend(frameon=False, fontsize=9)
    _style(bx)
    fig.tight_layout()
    return fig


def screening_figure(decay=0.5):
    """Energy levels of a toy atom with a filled 1s shell: s, p, d columns, against the hydrogen levels -1/(2 n^2)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    levels = mol.screened_levels(decay=decay)
    for l, name in enumerate("spd"):
        for n in range(l + 1, 5):
            e = levels[(n, l)]
            ax.plot([l - 0.35, l + 0.35], [e, e], color=H_COLOR, lw=2.2)
            ax.text(l + 0.38, e, f"{n}{name}", fontsize=8, va="center", color=INK)
    for n in (2, 3, 4):
        ax.axhline(-0.5 / n**2, color=GREY, lw=0.6, ls=":")
        ax.text(2.8, -0.5 / n**2, f"n={n}", fontsize=7, color=GREY, va="bottom", ha="right")
    ax.set_xlim(-0.6, 2.9)
    ax.set_ylim(-0.5, 0.0)
    ax.set_xticks([0, 1, 2], ["s ($\\ell=0$)", "p ($\\ell=1$)", "d ($\\ell=2$)"])
    ax.set_ylabel("energy (hartree)")
    ax.set_title("dotted: levels of hydrogen; the 1s level is below the frame", fontsize=9, color=INK)
    _style(ax)

    r = np.linspace(0.02, 8, 400)
    bx.plot(r, -mol.screened_potential(r, 3.0, decay) * r, color=POT_COLOR, lw=2)
    bx.axhline(1.0, color=GREY, lw=0.8, ls="--")
    bx.axhline(3.0, color=GREY, lw=0.8, ls="--")
    bx.text(7.9, 3.05, "bare nucleus (charge 3)", ha="right", fontsize=8, color=INK)
    bx.text(7.9, 0.95, "seen through the 1s shell (charge 1)", ha="right", va="top", fontsize=8, color=INK)
    bx.set_ylim(0, 3.4)
    bx.set_xlabel("$r$ (bohr)")
    bx.set_ylabel(r"effective charge $Z_{\mathrm{eff}}(r)$")
    _style(bx)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.92, wspace=0.25)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Section 4

def hybrid_figure():
    """Hybrids s + sqrt(n) p for n = 1, 2, 3 in the plane that holds them: the angular amplitude, blue positive, red negative."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    gamma = np.linspace(0, 2 * np.pi, 721)
    for ax, n, name in zip(axes, (1, 2, 3), ("sp", "sp$^2$", "sp$^3$")):
        theta = np.radians(mol.hybrid_angle_degrees(n))
        count = {1: 2, 2: 3, 3: 2}[n]  # lobes in the plane (sp3 has two more, out of it)
        step = {1: np.pi, 2: 2 * np.pi / 3, 3: theta}[n]
        for i in range(count):
            alpha = (i - (count - 1) / 2) * step + np.pi / 2
            f = mol.hybrid_profile(n, gamma)
            x, y = np.abs(f) * np.cos(gamma + alpha), np.abs(f) * np.sin(gamma + alpha)
            pos = f >= 0
            for sel, colour in ((pos, PLUS), (~pos, MINUS)):
                xs, ys = np.where(sel, x, np.nan), np.where(sel, y, np.nan)
                ax.fill(xs, ys, color=colour, alpha=0.35, lw=0)
                ax.plot(xs, ys, color=colour, lw=1.4)
        ax.plot([0], [0], "o", color=INK, ms=4)
        ax.set_aspect("equal")
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.3, 2.3)
        ax.axis("off")
        extra = " (+2 out of the plane)" if n == 3 else ""
        ax.set_title(f"{name}: {mol.hybrid_angle_degrees(n):.1f}°{extra}", fontsize=10, color=INK)
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Section 5

def methane_figure(bond=0):
    """Left: levels of the symmetry toy for methane. Right: weight of one localised bond orbital on the eight functions."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    e, vec = mol.methane_toy_levels()
    en = mol.TOY_ENERGIES_EV
    # left column: C 2s, C 2p; right column: H 1s
    for x0, x1, val, label in ((0.0, 0.7, en["C2s"], "C 2s"), (0.0, 0.7, en["C2p"], "C 2p (x3)"), (3.3, 4.0, en["H1s"], "H 1s (x4)")):
        ax.plot([x0, x1], [val, val], color=GREY, lw=2)
        ax.text(x0 - 0.05 if x0 == 0 else x1 + 0.05, val, label, ha="right" if x0 == 0 else "left", va="center", fontsize=8, color=INK)
    seen = set()
    for k, val in enumerate(e):
        key = round(val, 6)
        if key in seen:
            continue
        seen.add(key)
        deg = int(np.sum(np.abs(e - val) < 1e-6))
        occupied = val < en["H1s"]
        xs = np.linspace(1.6, 2.4, deg + 2)[1:-1] if deg > 1 else [2.0]
        for x in xs:
            ax.plot([x - 0.12, x + 0.12], [val, val], color=H_COLOR if occupied else MINUS, lw=2.6)
            if occupied:
                ax.text(x, val + 0.5, "↑↓", ha="center", fontsize=9, color=INK)
        label = ("a$_1$" if deg == 1 else "t$_2$") + (" bonding" if occupied else " antibonding")
        ax.text(2.55, val, label, va="center", fontsize=8, color=INK)
        for x_end, src in ((0.7, en["C2s"] if deg == 1 else en["C2p"]),):
            ax.plot([x_end, 1.45], [src, val], color=GREY, lw=0.5, ls=":")
        ax.plot([3.3, 2.7], [en["H1s"], val], color=GREY, lw=0.5, ls=":")
    ax.set_xlim(-1.0, 4.9)
    ax.set_xticks([])
    ax.set_ylabel("energy (eV)  [toy parameters]")
    ax.set_title("8 electrons fill the four bonding levels (1 + 3)", fontsize=9, color=INK)
    _style(ax)

    _, bonds = mol.localised_bond_orbitals(vec)
    weights = bonds[:, bond] ** 2
    labels = ["C 2s", "C 2px", "C 2py", "C 2pz", "H1", "H2", "H3", "H4"]
    colours = [K_COLOR] * 4 + [OMEGA_COLOR] * 4
    colours[4 + bond] = A_COLOR
    bx.bar(range(8), weights, color=colours)
    bx.set_xticks(range(8), labels, rotation=45, fontsize=8)
    bx.set_ylabel("weight in the bond orbital")
    bx.set_title(f"one localised C-H bond orbital (toward H{bond + 1})", fontsize=9, color=INK)
    _style(bx)
    fig.tight_layout()
    return fig
