"""Figures for the bridge notebook `from_atoms_to_molecules.py`. Each function returns a matplotlib Figure. Colours come from qmstyle.py."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import molecules as mol  # noqa: E402
from qmstyle import A_COLOR, H_COLOR, INK, K_COLOR, OMEGA2_COLOR, OMEGA_COLOR, PHASE_COLOR, POT_COLOR  # noqa: E402

PLUS, MINUS = H_COLOR, A_COLOR
GREY = "0.55"
AB = mol.BOHR_ANGSTROM  # angstrom per bohr; lengths are computed in bohr and drawn in angstrom
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

    ax.plot(r * AB, fit(r), color=POT_COLOR, lw=2, label=r"$E_{\mathrm{el}}(R)+1/R$ (electron in the field of the fixed protons)")
    ax.plot(r_pts * AB, e_pts, "o", color=INK, ms=3.5)
    for k, e in enumerate(levels):
        ax.plot([r[0] * AB, r[-1] * AB], [e, e], color=GREY if k != state else OMEGA_COLOR, lw=0.8 if k != state else 1.6)
    scale = 0.55 * (levels[1] - levels[0]) / np.abs(chi[:, state]).max()
    ax.plot(r * AB, levels[state] + scale * chi[:, state], color=OMEGA_COLOR, lw=2, label=f"nuclear wave, level {state}")
    ax.set_xlim(r[0] * AB, r[-1] * AB)
    ax.set_ylim(-0.625, -0.46)
    ax.set_xlabel("proton separation $R$ (Å)")
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
# Section 2: flowcharts of a geometry optimisation

def _canvas(width, height, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _box(ax, cx, cy, w, h, text, colour=H_COLOR, size=11, dashed=False, fill=0.16, bold=False):
    """Rounded box centred at (cx, cy). Returns its edge midpoints {'l','r','t','b'}."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=colour, ec=colour, alpha=fill, lw=0, zorder=1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=colour, lw=2.0, ls="--" if dashed else "-", zorder=2))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.35)
    return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}


def _diamond(ax, cx, cy, w, h, text, colour=PHASE_COLOR, size=10.5):
    from matplotlib.patches import Polygon
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=colour, ec="none", alpha=0.16, zorder=1))
    ax.add_patch(Polygon(pts, closed=True, fc="none", ec=colour, lw=2.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, zorder=3, linespacing=1.3)
    return {"l": pts[3], "r": pts[1], "t": pts[0], "b": pts[2]}


def _arrow(ax, p, q, label=None, label_at=None, rad=0.0, colour=INK, size=10.5, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=0,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8, shrinkA=0, shrinkB=0, ls=ls,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=16))
    if label:
        x, y = label_at if label_at else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(x, y, label, fontsize=size, color=colour, ha="center", va="center", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=4)


def scf_loop_figure():
    """The self-consistent-field loop at one fixed set of nuclear positions R."""
    fig, ax = _canvas(100, 62, (11, 6.8))
    inp = _box(ax, 15, 54, 26, 9, "fixed nuclei at $\\mathbf{R}$\n(charges at fixed points)", POT_COLOR)
    guess = _box(ax, 15, 38, 26, 9, "guess a density $n(\\mathbf{r})$\n(e.g. sum of atoms)", H_COLOR)
    build = _box(ax, 50, 46, 34, 14, "build the Kohn-Sham operator $\\hat f$\n$\\hat f=T+V_{\\mathrm{nuc}}(\\mathbf{R})$\n$+\\,V_{\\mathrm{H}}[n]+V_{\\mathrm{xc}}[n]$", H_COLOR)
    diag = _box(ax, 83, 46, 28, 14, "diagonalise $\\hat f$\n$\\hat f\\varphi_i=\\varepsilon_i\\varphi_i$", H_COLOR)
    new = _box(ax, 83, 22, 28, 15, "new density from the\noccupied orbitals\n\n$n=\\sum_{i\\in\\mathrm{occ}}|\\varphi_i|^2$", H_COLOR)
    conv = _diamond(ax, 50, 22, 30, 15, "converged?\n$n$ and $E$ stopped\nchanging")
    out = _box(ax, 50, 5.5, 52, 9, "$E(\\mathbf{R})=E_{\\mathrm{el}}(\\mathbf{R})+E_{\\mathrm{nn}}(\\mathbf{R})$", OMEGA_COLOR, bold=True)
    _arrow(ax, inp["b"], guess["t"])
    _arrow(ax, guess["r"], build["l"], rad=0.0)
    _arrow(ax, build["r"], diag["l"])
    _arrow(ax, diag["b"], new["t"])
    _arrow(ax, new["l"], conv["r"])
    _arrow(ax, conv["t"], build["b"], "no: use the new $n$", (50, 33.5))
    _arrow(ax, conv["b"], out["t"], "yes", (53, 12.6))
    fig.tight_layout(pad=0.4)
    return fig


def geometry_optimization_figure():
    """The outer loop (positions -> energy and forces -> step), with the SCF loop nested inside the energy step."""
    fig, ax = _canvas(100, 70, (11, 7.6))
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((1, 1), 98, 68, boxstyle="round,pad=0,rounding_size=2", fc=OMEGA_COLOR, ec=OMEGA_COLOR,
                                alpha=0.05, lw=0, zorder=0))
    ax.add_patch(FancyBboxPatch((1, 1), 98, 68, boxstyle="round,pad=0,rounding_size=2", fc="none", ec=OMEGA_COLOR, lw=2, zorder=0))
    ax.text(3, 66, "geometry optimisation (outer loop, runs over nuclear positions)", fontsize=11, color=OMEGA2_COLOR, fontweight="bold", va="center")
    pos = _box(ax, 13, 52, 21, 11, "positions\n$\\mathbf{R}_k$ (Å)", POT_COLOR)
    # nested SCF box
    ax.add_patch(FancyBboxPatch((28, 40), 44, 22, boxstyle="round,pad=0,rounding_size=1.5", fc=H_COLOR, ec=H_COLOR, alpha=0.10, lw=0, zorder=0.5))
    ax.add_patch(FancyBboxPatch((28, 40), 44, 22, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=H_COLOR, lw=2, ls="--", zorder=0.5))
    ax.text(50, 59.2, "SCF loop (inner loop, previous figure)", fontsize=10.5, color=H_COLOR, ha="center", va="center", fontweight="bold")
    scf = _box(ax, 50, 49, 38, 12, "electrons solved with the nuclei held\nat the fixed points $\\mathbf{R}_k$\n(Born-Oppenheimer)", POT_COLOR, fill=0.10)
    energy = _box(ax, 87, 52, 21, 11, "energy\n$E(\\mathbf{R}_k)$", OMEGA_COLOR)
    forces = _box(ax, 87, 30, 21, 15, "forces on nuclei\n$\\mathbf{F}=-\\partial E/\\partial\\mathbf{R}$\nHellmann-Feynman\n+ Pulay term", OMEGA_COLOR, size=10)
    step = _box(ax, 50, 28, 36, 15, "optimiser step\n$\\Delta\\mathbf{R}=-H^{-1}\\mathbf{g}$ (internal coordinates)\n$H$ approximate, updated by BFGS", OMEGA_COLOR, size=10.5)
    conv = _diamond(ax, 14, 28, 24, 18, "converged?\nmax / RMS force,\ndisplacement,\n$\\Delta E$", PHASE_COLOR, size=9.5)
    stop = _box(ax, 14, 8, 21, 8, "stop: $\\mathbf{R}_{\\mathrm{opt}}$", K_COLOR, bold=True)
    _arrow(ax, pos["r"], (28, 52))
    _arrow(ax, (72, 52), energy["l"])
    _arrow(ax, energy["b"], forces["t"])
    _arrow(ax, forces["l"], step["r"])
    _arrow(ax, step["l"], conv["r"])
    _arrow(ax, conv["b"], stop["t"], "yes", (14.0, 15.5))
    _arrow(ax, conv["t"], pos["b"], "no: $\\mathbf{R}_{k+1}=\\mathbf{R}_k+\\Delta\\mathbf{R}$", (14, 40.5), size=10)
    ax.text(62, 11, "Each pass of the outer loop runs one full SCF; the analytic forces\nare then computed from the converged wave, without a new SCF.",
            fontsize=10.5, color=INK, ha="center", va="center")
    fig.tight_layout(pad=0.4)
    return fig


def surface_uses_figure():
    """One energy surface E(R), three things done with it."""
    fig, ax = _canvas(100, 50, (11, 5.4))
    surf = _box(ax, 50, 40, 52, 11, "$E(\\mathbf{R})$ and forces $\\mathbf{F}(\\mathbf{R})$ from an SCF\n(Born-Oppenheimer surface)", POT_COLOR, bold=True)
    one = _box(ax, 16, 17, 29, 20, "single point\none $\\mathbf{R}$, one SCF\n$\\rightarrow E$ (and $\\mathbf{F}$)\nused as a training label", H_COLOR, size=12.5)
    opt = _box(ax, 50, 17, 29, 20, "optimisation\nmany SCFs, $\\mathbf{R}$ moves\ndownhill until $\\mathbf{F}\\approx0$\n$\\rightarrow\\mathbf{R}_{\\mathrm{opt}}$", OMEGA_COLOR, size=12.5)
    md = _box(ax, 84, 17, 29, 20, "ab initio dynamics\nmany SCFs, $\\mathbf{R}$ moves by\nNewton's law, $m\\ddot{\\mathbf{R}}=\\mathbf{F}$\n$\\rightarrow$ a trajectory", K_COLOR, size=12.5)
    for b in (one, opt, md):
        _arrow(ax, (surf["b"][0] + (b["t"][0] - 50) * 0.4, surf["b"][1]), b["t"])
    ax.text(50, 2.5, "Frequencies: second derivatives of $E$ at $\\mathbf{R}_{\\mathrm{opt}}$ (the optimised geometry) give the vibrations of section 2.",
            fontsize=11, color=INK, ha="center", va="center")
    fig.tight_layout(pad=0.4)
    return fig


def optimisation_toy_figure(start=3.0, hessian0=0.3):
    """Optimising the H2+ bond length with the loop above: the steps on the energy curve (left) and the energy per step (right)."""
    fit = mol.h2plus_fit()
    run = mol.optimise_bond_length(start, fit, hessian0=hessian0)
    r = np.linspace(1.25, 4.0, 400)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.plot(r * AB, fit(r), color=POT_COLOR, lw=2)
    rs, es = np.array(run["r"]) * AB, np.array(run["energy"])
    ax.plot(rs, es, "-", color=GREY, lw=0.8)
    ax.scatter(rs, es, c=np.arange(len(rs)), cmap="viridis", s=60, zorder=3)
    for k, (x, y) in enumerate(zip(rs, es)):
        ax.annotate(str(k), (x, y), textcoords="offset points", xytext=(5, 7), fontsize=9, color=INK)
    ax.set_xlabel("proton separation $R$ (Å)")
    ax.set_ylabel("energy $E(R)$ (hartree)")
    ax.set_title("each numbered point is one SCF calculation", fontsize=10, color=INK)
    _style(ax)
    e_min = mol.h2plus_minimum(fit)[1]
    bx.semilogy(range(len(es)), es - e_min + 1e-7, "o-", color=H_COLOR, lw=1.8)
    bx.set_xlabel("optimisation step $k$ (one SCF each)")
    bx.set_ylabel(r"energy above the minimum (hartree)")
    bx.set_xticks(range(len(es)))
    bx.set_title(f"stopped at $|F|<10^{{-4}}$ hartree/bohr after {len(es)} SCFs", fontsize=10, color=INK)
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

    r = np.linspace(0, 4, 400)  # bohr
    dens = lambda z: 4 * z**3 * r**2 * np.exp(-2 * z * r) / AB  # 4 pi r^2 (z^3/pi) exp(-2 z r), per angstrom
    bx.plot(r * AB, dens(2.0), color=GREY, lw=1.6, ls="--", label=r"$\zeta=2$ (bare nucleus)")
    bx.plot(r * AB, dens(zeta), color=PHASE_COLOR, lw=2.2, label=rf"$\zeta={zeta:.3f}$")
    bx.set_xlabel("$r$ (Å)")
    bx.set_ylabel(r"probability per Å, $4\pi r^2|\varphi|^2$")
    bx.legend(frameon=False, fontsize=9)
    _style(bx)
    fig.tight_layout()
    return fig


def screening_figure(decay=0.25):
    """`decay` is the screening distance in angstrom. Energy levels of a toy atom with a filled 1s shell: s, p, d columns, against the hydrogen levels -1/(2 n^2)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    decay_bohr = decay / AB
    levels = mol.screened_levels(decay=decay_bohr)
    for l, name in enumerate("spd"):
        for n in range(l + 1, 5):
            e = levels[(n, l)]
            ax.plot([l - 0.35, l + 0.35], [e, e], color=H_COLOR, lw=2.2)
            ax.text(l + 0.38, e, f"{n}{name}", fontsize=8, va="center", color=INK, clip_on=True)
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
    bx.plot(r * AB, -mol.screened_potential(r, 3.0, decay_bohr) * r, color=POT_COLOR, lw=2)
    bx.axhline(1.0, color=GREY, lw=0.8, ls="--")
    bx.axhline(3.0, color=GREY, lw=0.8, ls="--")
    bx.text(7.9 * AB, 3.05, "bare nucleus (charge 3)", ha="right", fontsize=8, color=INK)
    bx.text(7.9 * AB, 0.95, "seen through the 1s shell (charge 1)", ha="right", va="top", fontsize=8, color=INK)
    bx.set_ylim(0, 3.4)
    bx.set_xlabel("$r$ (Å)")
    bx.set_ylabel(r"effective charge $Z_{\mathrm{eff}}(r)$")
    _style(bx)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.92, wspace=0.25)
    return fig


def orbital_size_figure(z=1.0):
    """Left: radial probability of the hydrogen-like 1s, 3s and 3d orbitals for nuclear charge z, hydrogen's (z = 1) as thin lines.
    Right: <r> of the highest occupied orbital of the neutral atoms H to Ar (PySCF), against the hydrogen-like value for n = 1, 2, 3 at the same Z."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    r = np.linspace(0, 24, 1200)  # bohr
    for (n, l), colour in zip(((1, 0), (3, 0), (3, 2)), (H_COLOR, A_COLOR, OMEGA_COLOR)):
        name = f"{n}{'spd'[l]}"
        ax.plot(r * AB, mol.hydrogen_like_density(n, l, 1.0, r) / AB, color=colour, lw=0.9, ls="--", alpha=0.7)
        ax.plot(r * AB, mol.hydrogen_like_density(n, l, z, r) / AB, color=colour, lw=2.2,
                label=rf"{name}: $\langle r\rangle={mol.hydrogen_like_radius(n, l, z) * AB:.2f}$ Å")
    ax.set_xlim(0, 24 * AB)
    ax.set_xlabel("$r$ (Å)")
    ax.set_ylabel("probability per Å, $r^2R^2$")
    ax.set_title(f"nuclear charge $Z={z:g}$ (dashed: $Z=1$)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    d = mol.atom_sizes()
    zs = np.array(d["z"])
    rr = np.array(d["mean_radius_bohr"]) * AB
    zz = np.linspace(1, 18, 200)
    for n, ls in ((1, ":"), (2, "-."), (3, "--")):
        bx.plot(zz, mol.hydrogen_like_radius(n, 0, zz) * AB, color=GREY, lw=1, ls=ls, label=rf"bare nucleus, $n={n}$ ($\langle r\rangle=0.79\,n^2/Z$ Å)")
    bx.plot(zs, rr, "o-", color=PHASE_COLOR, lw=1.6, ms=5, label="neutral atom, outermost orbital (PySCF)")
    k = int(np.clip(round(z), 1, 18)) - 1
    bx.plot([zs[k]], [rr[k]], "o", color=INK, ms=10, mfc="none", mew=1.5)
    for zi, sym in zip(zs, d["symbols"]):
        if sym in ("H", "He", "Li", "Ne", "Na", "Ar"):
            bx.annotate(sym, (zi, rr[zi - 1]), textcoords="offset points", xytext=(4, 5), fontsize=8, color=INK)
    bx.set_xlabel("nuclear charge $Z$")
    bx.set_ylabel(r"mean distance $\langle r\rangle$ (Å)")
    bx.set_ylim(0, 3.2)
    bx.legend(frameon=False, fontsize=7, loc="upper right")
    _style(bx)
    fig.tight_layout()
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
