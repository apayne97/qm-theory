"""Figures and small numerical helpers for the Kohn-Sham matrix notebook (`kohn_sham_matrix.py`)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmstyle import A_COLOR, H_COLOR, INK, K_COLOR, KIN_COLOR, OMEGA_COLOR, PHASE_COLOR  # noqa: E402

PIECE_TITLES = {
    "T": "T  (kinetic)", "V": "V  (nuclear attraction)", "J": "J  (Coulomb)",
    "Vxc": "Vxc  (exchange-correlation)", "mK": "−K/2  (exact exchange, weighted)", "F": "F  (sum of all terms)",
}


def solve_generalized(F, S):
    """Solve F C = S C eps with numpy only: symmetric orthogonalisation with S^(-1/2).

    Step (a): S = U s U^T (eigenvalues s, eigenvectors U). (b): X = S^(-1/2) = U s^(-1/2) U^T, so X S X = 1.
    (c): F' = X F X is an ordinary symmetric matrix; solve F' C' = C' eps. (d): C = X C' solves the original problem.
    """
    s, U = np.linalg.eigh(S)
    X = U @ np.diag(s**-0.5) @ U.T
    eps, Cp = np.linalg.eigh(X @ F @ X)
    return eps, X @ Cp


def _norm(m):
    top = np.abs(m).max()
    return SymLogNorm(linthresh=max(top * 1e-3, 1e-12), vmin=-top, vmax=top)


def short_label(label):
    """'0 O 2px' -> 'O 2px' (the leading number is the atom index)."""
    return " ".join(label.split()[1:])


def matrices_figure(ks):
    S, F = np.array(ks["S"]), np.array(ks["F"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, m, title in zip(axes, (S, F), ("S  (overlap)", "F  (Kohn-Sham, hartree)")):
        im = ax.imshow(m, cmap="RdBu_r", norm=_norm(m))
        ax.set_title(title)
        ax.set_xlabel("basis function ν")
        ax.set_ylabel("basis function μ")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Water, 58 basis functions: both matrices are symmetric 58 × 58 (colour on a signed log scale)")
    fig.tight_layout()
    return fig


def pieces_figure(ks):
    pieces = {k: np.array(v) for k, v in ks["pieces"].items()}
    pieces["Vxc"] = pieces["Vxc"] + pieces["Vnlc"]
    del pieces["Vnlc"]
    pieces["F"] = np.array(ks["F"])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
    for ax, key in zip(axes.ravel(), ("T", "V", "J", "Vxc", "mK", "F")):
        m = pieces[key]
        im = ax.imshow(m, cmap="RdBu_r", norm=_norm(m))
        ax.set_title(PIECE_TITLES[key], fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.text(0.03, 0.04, f"largest |element| {np.abs(m).max():.2f}", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.suptitle("The terms of F at the converged density (hartree; each panel has its own signed log colour scale)")
    fig.tight_layout()
    return fig


def scf_loop_figure(ks):
    hist = ks["hand_scf"]
    cycles = [h["cycle"] for h in hist]
    energy = np.array([h["energy_Ha"] for h in hist])
    change = np.array([h["max_change_D"] for h in hist])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].semilogy(cycles, change, "o-", color=A_COLOR)
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("largest change of any element of D")
    axes[0].set_title("The density stops changing")
    axes[1].semilogy(cycles[:-1], np.abs(energy[:-1] - energy[-1]), "o-", color=H_COLOR)
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("|E − E(last cycle)| (hartree)")
    axes[1].set_title("and the energy settles")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def levels_figure(ks):
    eps = np.array(ks["eps_Ha"]) * 27.211386
    n_occ = ks["n_occ"]
    fig, ax = plt.subplots(figsize=(4.6, 5))
    for i, e in enumerate(eps[: n_occ + 6]):
        occupied = i < n_occ
        ax.hlines(e, 0, 1, color=(H_COLOR if occupied else PHASE_COLOR), lw=2)
        ax.text(1.05, e, f"{e:.2f} eV", va="center", fontsize=8)
    ax.set_ylim(-35, 12)
    ax.set_xlim(0, 1.7)
    ax.set_xticks([])
    ax.set_ylabel("orbital energy ε (eV)")
    ax.set_title("Eigenvalues of F (occupied blue, empty purple)", fontsize=9)
    fig.tight_layout()
    return fig
