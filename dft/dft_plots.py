"""Figures for the DFT decoder notebook. Each function returns a matplotlib Figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_data import (  # noqa: E402
    EXPERIMENTAL_WATER_IP_EV,
    damped_dispersion,
    erf_split,
    exact_exchange_fraction,
)
from qmstyle import A_COLOR, H2_COLOR, H_COLOR, INK, K_COLOR, KIN_COLOR, PHASE_COLOR, PROB_COLOR, R_COLOR  # noqa: E402

FUNCTIONAL_LABELS = {
    "HF": "Hartree-Fock", "svwn": "LDA (SVWN)", "pbe": "GGA (PBE)", "r2scan": "meta-GGA (r2SCAN)",
    "b3lyp": "hybrid (B3LYP)", "wb97x": "range-sep. (ωB97X)", "wb97m_v": "range-sep. meta (ωB97M-V)",
}


def scf_figure(scf):
    hist = np.array(scf["scf_history_Ha"])
    err = np.abs(hist - hist[-1])[:-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.semilogy(np.arange(1, len(err) + 1), err, "o-", color=H_COLOR)
    ax.set_xlabel("self-consistent-field cycle")
    ax.set_ylabel("|E − E(final)| (hartree)")
    ax.set_title("Water, ωB97M-V/def2-TZVPD: the energy settles after a few cycles")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def line_fields_figure(scf):
    line = scf["line"]
    s = np.array(line["s_bohr"])
    rho, grad_s = np.array(line["rho"]), np.array(line["reduced_gradient"])
    tau, tau_w = np.array(line["tau"]), np.array(line["tau_w"])
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    axes[0].semilogy(s, rho, color=H_COLOR)
    axes[0].set_ylabel("density ρ (bohr⁻³)")
    axes[0].set_title("Along the line from the oxygen through one hydrogen")
    axes[1].plot(s, np.clip(grad_s, 0, 6), color=A_COLOR)
    axes[1].set_ylabel("reduced gradient s\n(how fast ρ changes)")
    axes[2].semilogy(s, np.maximum(tau, 1e-8), color=KIN_COLOR, label="kinetic energy density τ")
    axes[2].semilogy(s, np.maximum(tau_w, 1e-8), color=INK, ls="--", label="τ for a single orbital (τ_W)")
    axes[2].set_ylabel("τ (hartree bohr⁻³)")
    axes[2].set_xlabel("distance from the oxygen nucleus (bohr)")
    axes[2].legend(fontsize=8, loc="upper right")
    h_pos = line["oh_length_bohr"]
    for ax in axes:
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.axvline(h_pos, color="gray", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
    axes[0].text(0.05, 0.88, "O", transform=axes[0].get_xaxis_transform(), color="gray")
    axes[0].text(h_pos + 0.05, 0.88, "H", transform=axes[0].get_xaxis_transform(), color="gray")
    fig.tight_layout()
    return fig


def homo_figure(ladder):
    names = list(ladder["functionals"])
    ip = [-ladder["functionals"][n]["homo_eV"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [H2_COLOR if n in ("HF", "svwn", "pbe", "r2scan", "b3lyp") else H_COLOR for n in names]
    ax.barh([FUNCTIONAL_LABELS[n] for n in names], ip, color=colors)
    ax.axvline(EXPERIMENTAL_WATER_IP_EV, color=A_COLOR, lw=2)
    ax.text(EXPERIMENTAL_WATER_IP_EV + 0.15, len(names) - 1, "experiment\n12.6 eV", color=A_COLOR, fontsize=8, ha="left", va="center")
    ax.set_xlim(0, 15.5)
    ax.invert_yaxis()
    ax.set_xlabel("−(energy of the highest occupied orbital) (eV)")
    ax.set_title("Water: top orbital energy against the ionisation energy", fontsize=10)
    fig.tight_layout()
    return fig


def eint_figure(ladder):
    names = [n for n in ladder["functionals"] if n != "HF"] + ["HF"]
    vals = [ladder["functionals"][n]["cp"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([FUNCTIONAL_LABELS[n] for n in names], vals, color=H_COLOR)
    ref = ladder["ccsdt"]["cp"]
    ax.axvline(ref, color=A_COLOR, lw=2)
    ax.text(ref - 0.05, len(names) - 0.4, f"CCSD(T)\n{ref:.2f}", color=A_COLOR, fontsize=8, ha="right", va="bottom")
    ax.invert_yaxis()
    ax.set_xlabel("counterpoise-corrected interaction energy (kcal/mol)")
    ax.set_title("Water dimer, def2-TZVPD, different functionals", fontsize=10)
    fig.tight_layout()
    return fig


def split_figure(omega, short_fraction, long_fraction):
    r = np.linspace(0.3, 12, 300)
    short, long_ = erf_split(r, omega)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    ax.plot(r, 1 / r, color=INK, lw=2, label="1/r")
    ax.plot(r, short, color=A_COLOR, label="short-range part erfc(ωr)/r")
    ax.plot(r, long_, color=H_COLOR, label="long-range part erf(ωr)/r")
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("distance between two electrons r (bohr)")
    ax.set_ylabel("repulsion (hartree)")
    ax.legend(fontsize=8)
    ax.set_title("The 1/r repulsion split in two")
    ax = axes[1]
    ax.plot(r, exact_exchange_fraction(r, omega, short_fraction, long_fraction), color=H_COLOR, lw=2)
    ax.axhline(short_fraction, color="gray", ls=":", lw=1)
    ax.axhline(long_fraction, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("distance between two electrons r (bohr)")
    ax.set_ylabel("fraction of exact exchange")
    ax.set_title(f"ωB97M-V: {short_fraction:.0%} up close, {long_fraction:.0%} far away")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def basis_figure(basis):
    names = list(basis["basis"])
    n = [basis["basis"][b]["n_basis"] for b in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(names, n, color=H_COLOR)
    for i, v in enumerate(n):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    ax.set_ylabel("basis functions for H₂O")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("More letters, more functions")
    fig.tight_layout()
    return fig


def bsse_figure(basis):
    names = list(basis["dimer"])
    raw = [basis["dimer"][b]["raw"] for b in names]
    cp = [basis["dimer"][b]["cp"] for b in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, raw, 0.4, color=H2_COLOR, label="raw: dimer − 2 monomers")
    ax.bar(x + 0.2, cp, 0.4, color=H_COLOR, label="counterpoise corrected")
    ax.set_xticks(x, names)
    ax.set_ylabel("interaction energy (kcal/mol)")
    ax.legend(fontsize=8)
    ax.set_title("Water dimer, ωB97M-V: the two answers agree once the basis has diffuse functions")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    return fig


def dispersion_figure(disp):
    order = ["HF", "b3lyp", "wb97m_v_novv10", "wb97m_v"]
    labels = ["Hartree-Fock", "B3LYP", "ωB97M-V\nVV10 switched off", "ωB97M-V\n(with VV10)"]
    vals = [disp["methods"][k]["cp"] for k in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, vals, color=[H2_COLOR, H2_COLOR, H2_COLOR, H_COLOR])
    ref = disp["ccsdt"]["cp"]
    ax.axhline(ref, color=A_COLOR, lw=2)
    ax.text(-0.45, ref - 0.02, f"CCSD(T) {ref:.2f}", color=A_COLOR, fontsize=8, ha="left", va="top")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction energy (kcal/mol)")
    ax.set_title(f"Methane dimer, C···C = {disp['r_cc']} Å: only a dispersion-aware method binds")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    return fig


def damping_figure():
    r = np.linspace(0.6, 6, 300)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(r, -1 / r**6, color=INK, ls="--", label="bare −C₆/r⁶")
    ax.plot(r, damped_dispersion(r, 1.0, 2.0), color=H_COLOR, label="damped −C₆/(r⁶ + R₀⁶)")
    ax.set_ylim(-0.1, 0.01)
    ax.set_xlabel("distance between two atoms (arbitrary units)")
    ax.set_ylabel("energy (arbitrary units)")
    ax.legend(fontsize=8)
    ax.set_title("Why a damping function is needed (shape only)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grid_figure(scf):
    g = scf["grid_study"]
    pts = np.array([v["n_points"] for v in g.values()])
    e = np.array([v["e_Ha"] for v in g.values()])
    ref = e[-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.loglog(pts[:-1], np.maximum(np.abs(e[:-1] - ref) * 627.5095, 1e-6), "o-", color=H_COLOR)
    ax.set_xlabel("grid points for H₂O")
    ax.set_ylabel("|E − E(largest grid)| (kcal/mol)")
    ax.set_title("The integration grid is a setting too")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig
