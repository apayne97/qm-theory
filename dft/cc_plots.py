"""Figures for the coupled-cluster notebook. Each function returns a matplotlib Figure.

Colours come from qmstyle.py. Within this notebook the excitation levels keep one colour throughout (singles,
doubles, triples), and each method keeps one colour in every figure.
"""

import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmstyle import (  # noqa: E402
    A_COLOR, H_COLOR, INK, K_COLOR, OMEGA_COLOR, PERIOD_COLOR, PHASE_COLOR, POT_COLOR, PROB_COLOR,
)

LEVEL_COLORS = {0: H_COLOR, 1: K_COLOR, 2: A_COLOR, 3: OMEGA_COLOR, 4: PERIOD_COLOR}
LEVEL_NAMES = {0: "reference", 1: "singles", 2: "doubles", 3: "triples", 4: "quadruples"}
METHOD_COLORS = {"HF": H_COLOR, "MP2": K_COLOR, "CISD": POT_COLOR, "CCSD": A_COLOR, "CCSD(T)": OMEGA_COLOR, "FCI": INK,
                 "DFT": PHASE_COLOR}


def level_weights_figure(weights):
    """Weight of each excitation level in the exact (full CI) wave function. `weights`: dict level -> weight."""
    levels = sorted(weights)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([LEVEL_NAMES.get(k, str(k)) for k in levels], [weights[k] for k in levels],
           color=[LEVEL_COLORS.get(k, INK) for k in levels])
    ax.set_yscale("log")
    ax.set_ylabel("weight in the exact wave function")
    ax.set_title("Water (STO-3G): weight of each kind of determinant", fontsize=10)
    for i, k in enumerate(levels):
        ax.text(i, weights[k] * 1.4, f"{weights[k]:.1e}" if weights[k] < 0.01 else f"{weights[k]:.3f}", ha="center", fontsize=8)
    ax.set_ylim(min(weights.values()) / 100, 3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def determinant_growth_figure(rows):
    """Number of determinants of full CI for water-like electron counts (5 alpha + 5 beta) against basis size."""
    n = np.arange(5, 120)
    dets = np.array([float(comb(int(x), 5)) ** 2 for x in n])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(n, dets, color=INK, lw=1.5)
    for r in rows:
        ax.plot(r["n_basis"], r["n_det"], "o", color=A_COLOR)
        ax.annotate(r["basis"], (r["n_basis"], r["n_det"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.axhline(1e9, color="gray", ls=":", lw=1)
    ax.text(60, 2.5e9, r"$10^9$ determinants", fontsize=8, color="gray")
    ax.set_xlabel("number of basis functions N")
    ax.set_ylabel("determinants in full CI")
    ax.set_title("Full CI for ten electrons (water): the size of the problem", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def size_consistency_figure(scan, mono_cisd_ccsd):
    """Error of the pair energy, E(pair) - 2 E(single), against separation, for CISD and CCSD (PySCF supermolecules)."""
    r = sorted(float(k) for k in scan if float(k) >= 3.0)
    cisd = [scan[str(x)]["e_cisd_corr"] - 2 * mono_cisd_ccsd for x in r]
    ccsd = [scan[str(x)]["e_ccsd_corr"] - 2 * mono_cisd_ccsd for x in r]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.semilogx(r, np.array(cisd) * 1000, "o-", color=METHOD_COLORS["CISD"], label="CISD")
    ax.semilogx(r, np.array(ccsd) * 1000, "s-", color=METHOD_COLORS["CCSD"], label="CCSD")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(r, [f"{x:g}" for x in r])
    ax.minorticks_off()
    ax.set_xlabel("distance between the two H$_2$ molecules (Å)")
    ax.set_ylabel("correlation energy of the pair\nminus twice that of one (mHa)")
    ax.set_title("Two H$_2$ moving apart: the CISD error stays, the CCSD error vanishes", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def product_figure(c_a, c_b, c_quad):
    """The coefficient of 'both molecules doubly excited' in the exact wave function against the product c_a c_b."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = ["doubles on A\n$c_A$", "doubles on B\n$c_B$", "product\n$c_A\\,c_B$", "quadruples on A and B\nexact wave function"]
    vals = [abs(c_a), abs(c_b), abs(c_a * c_b), abs(c_quad)]
    ax.bar(labels, vals, color=[A_COLOR, A_COLOR, INK, PERIOD_COLOR])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.5f}", ha="center", fontsize=8)
    ax.set_ylabel("size of coefficient")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Two independent H$_2$: the quadruple is the product", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def convergence_figure(history_diis, history_plain):
    """Error of the CCSD energy at each iteration (against the final value)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for hist, label, style, colour in ((history_plain, "plain iteration", "o-", H_COLOR), (history_diis, "with DIIS mixing", "s-", A_COLOR)):
        h = np.array(hist)
        err = np.abs(h - h[-1])[:-1]
        err = np.maximum(err, 1e-14)
        ax.semilogy(np.arange(len(err)), err, style, color=colour, ms=4, label=label)
    ax.set_xlabel("iteration (0 = MP2 amplitudes)")
    ax.set_ylabel("|E − E(final)| (hartree)")
    ax.set_title("Water (6-31G): the amplitude iteration converges in a few steps", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def scaling_figure(timing):
    """Left: operation counts for a growing molecule (o occupied orbitals, v = 4 o virtual). Right: measured time of this notebook's numpy code against the number of virtual orbitals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    o = np.arange(2, 40)
    v = 4 * o
    ax1.loglog(o, o**2 * v**3.0, color=METHOD_COLORS["MP2"], label=r"MP2: $o^2v^3\propto o^5$")
    ax1.loglog(o, o**2 * v**4.0, color=METHOD_COLORS["CCSD"], label=r"CCSD per iteration: $o^2v^4\propto o^6$")
    ax1.loglog(o, o**3 * v**4.0, color=METHOD_COLORS["CCSD(T)"], label=r"CCSD(T), once: $o^3v^4\propto o^7$")
    ax1.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax1.minorticks_off()
    ax1.set_xlabel("molecule size: occupied orbitals o (virtual v = 4 o)")
    ax1.set_ylabel("operations (up to a constant)")
    ax1.set_title("Counting operations (FCI is off this scale: see section 1)", fontsize=9)
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3, which="both")
    v_up = np.array([r["n_virt_so"] for r in timing])
    ax2.loglog(v_up, [r["t_update_s"] for r in timing], "o-", color=METHOD_COLORS["CCSD"], label="one CCSD iteration")
    vt = [r["n_virt_so"] for r in timing if "t_triples_s" in r]
    ax2.loglog(vt, [r["t_triples_s"] for r in timing if "t_triples_s" in r], "s-", color=METHOD_COLORS["CCSD(T)"], label="(T), once")
    ref = np.array([8.0, 40.0])
    ax2.loglog(ref, 3e-4 * (ref / 8) ** 4, ":", color="gray", label=r"slope 4 ($v^4$)")
    ax2.set_xlabel("virtual spin orbitals v (eight occupied)")
    ax2.set_ylabel("wall-clock time (s)")
    ax2.set_title("Measured with the numpy code of section 5", fontsize=10)
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def recovery_figure(equilibrium_and_stretch):
    """Share of the exact correlation energy recovered by MP2, CCSD and CCSD(T). Input: dict label -> energies dict."""
    labels = list(equilibrium_and_stretch)
    methods = ["MP2", "CCSD", "CCSD(T)"]
    width = 0.26
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for k, m in enumerate(methods):
        vals = []
        for lab in labels:
            e = equilibrium_and_stretch[lab]
            corr = {"MP2": e["e_mp2_corr"], "CCSD": e["e_ccsd_corr"], "CCSD(T)": e["e_ccsd_corr"] + e["e_t"]}[m]
            vals.append(100 * corr / e["e_fci_corr"])
        ax.bar(x + (k - 1) * width, vals, width, color=METHOD_COLORS[m], label=m)
        for xi, v in zip(x + (k - 1) * width, vals):
            ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.axhline(100, color=INK, lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(70, 112)
    ax.set_ylabel("% of the exact (FCI) correlation energy")
    ax.set_title("Water, 6-31G: how much correlation each method recovers", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def interaction_figure(ladder, dispersion):
    """Interaction energies (counterpoise corrected, kcal/mol) of the water dimer and the methane dimer, from the DFT notebook."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    names = [("HF", "HF", "HF"), ("pbe", "PBE", "DFT"), ("b3lyp", "B3LYP", "DFT"), ("wb97m_v", "ωB97M-V", "DFT")]
    vals1 = [ladder["functionals"][k]["cp"] for k, _, _ in names] + [ladder["ccsdt"]["cp"]]
    vals2 = [dispersion["methods"][k]["cp"] for k, _, _ in names if k in dispersion["methods"]] + [dispersion["ccsdt"]["cp"]]
    names2 = [lab for k, lab, _ in names if k in dispersion["methods"]]
    cols1 = [METHOD_COLORS[c] for _, _, c in names] + [METHOD_COLORS["CCSD(T)"]]
    cols2 = [METHOD_COLORS[c] for k, _, c in names if k in dispersion["methods"]] + [METHOD_COLORS["CCSD(T)"]]
    for ax, labs, vals, cols, title in (
        (ax1, [n[1] for n in names] + ["CCSD(T)"], vals1, cols1, "Water dimer (O···O 2.91 Å)"),
        (ax2, names2 + ["CCSD(T)"], vals2, cols2, "Methane dimer (C···C 3.8 Å)"),
    ):
        ax.bar(labs, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.08 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("interaction energy (kcal/mol)")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    ax1.set_ylim(min(vals1) - 1.0, 0.8)
    ax2.set_ylim(min(vals2) - 0.15, max(vals2) + 0.15)
    fig.tight_layout()
    return fig
