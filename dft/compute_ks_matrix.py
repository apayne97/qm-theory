"""Run water at wB97M-V/def2-TZVPD with PySCF and write `data_ks.json` for `kohn_sham_matrix.py`.

The notebook only reads that file, so it needs no PySCF. Needs `pyscf` (2.14 tested); run from this folder with
`pixi run -e compute python compute_ks_matrix.py` (about 15 s).

What is saved: the matrices S, T, V, J, Vxc, Vnlc, -K/2 and their sum F at the converged density, the orbital
coefficients C and energies eps, and the history of a plain self-consistent-field loop written out by hand (build F,
solve the generalized eigenproblem, mix the density), so that the notebook can show every step of the loop.
"""
import json
import time
from pathlib import Path

import click
import numpy as np
import scipy.linalg as sl
from pyscf import dft, gto, scf

HERE = Path(__file__).parent
WATER = """O 0 0 0.117790
H 0 0.755453 -0.471161
H 0 -0.755453 -0.471161"""
GRID = (99, 590)


def make_mf():
    mol = gto.M(atom=WATER, unit="Angstrom", basis="def2-tzvpd", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "wb97m_v"
    mf.nlc = "VV10"
    mf.grids.atom_grid = GRID
    mf.conv_tol = 1e-10
    return mol, mf


def terms(mol, mf, D):
    """The pieces of the Kohn-Sham matrix for the total density matrix D (= 2 P)."""
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
    k_full = mf.get_k(mol, D)
    k_long = mf.get_k(mol, D, omega=omega)
    k_eff = hyb * k_full + (alpha - hyb) * k_long          # short range hyb, long range alpha
    _, e_xc, v_xc = ni.nr_rks(mol, mf.grids, mf.xc, D)
    _, e_nlc, v_nlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, D)
    terms.energies = {"xc": float(e_xc), "nlc": float(e_nlc), "exact_exchange": float(-0.25 * np.einsum("ij,ji", k_eff, D))}
    return {
        "T": mol.intor("int1e_kin"),
        "V": mol.intor("int1e_nuc"),
        "J": mf.get_j(mol, D),
        "Vxc": v_xc,
        "Vnlc": v_nlc,
        "mK": -0.5 * k_eff,
    }


def hand_scf(mol, mf, S, nocc, mixing=0.5, tol=1e-6, max_cycle=80):
    """Roothaan loop with linear density mixing, using PySCF only to build F from a density."""
    h1e = mf.get_hcore()
    D = mf.get_init_guess(mol, "minao")
    history = []
    for cycle in range(1, max_cycle + 1):
        vhf = mf.get_veff(mol, D)
        F = h1e + vhf
        energy = float(mf.energy_tot(D, h1e, vhf))
        eps, C = sl.eigh(F, S)
        D_out = 2 * C[:, :nocc] @ C[:, :nocc].T
        change = float(np.abs(D_out - D).max())
        history.append({"cycle": cycle, "energy_Ha": energy, "max_change_D": change,
                        "homo_Ha": float(eps[nocc - 1])})
        print(history[-1], flush=True)
        if change < tol:
            break
        D = (1 - mixing) * D + mixing * D_out
    return history


@click.command()
def main():
    t0 = time.time()
    mol, mf = make_mf()
    mf.kernel()
    nocc = mol.nelectron // 2
    C, eps, S = mf.mo_coeff, mf.mo_energy, mf.get_ovlp()
    D = mf.make_rdm1()
    F = mf.get_fock(dm=D)
    pieces = terms(mol, mf, D)
    energies = dict(terms.energies)     # xc, nlc and exact-exchange energies at the converged D

    D0 = mf.get_init_guess(mol, "minao")
    pieces0 = terms(mol, mf, D0)
    shift = {k: float(np.linalg.norm(pieces0[k] - pieces[k])) for k in pieces}

    # Hartree-Fock for comparison: F = h + J - K/2, no exchange-correlation term.
    hf = scf.RHF(mol)
    hf.kernel()
    D_hf = hf.make_rdm1()
    f_hf_rebuilt = hf.get_hcore() + hf.get_j(mol, D_hf) - 0.5 * hf.get_k(mol, D_hf)

    data = {
        "hf": {"e_tot_Ha": float(hf.e_tot), "rebuild_max_diff": float(np.abs(hf.get_fock() - f_hf_rebuilt).max())},
        "e_tot_Ha": float(mf.e_tot), "n_basis": mol.nao, "n_occ": nocc,
        "ao_labels": [" ".join(x.split()) for x in mol.ao_labels()],
        "S": S.tolist(), "F": F.tolist(), "C": C.tolist(), "eps_Ha": eps.tolist(), "D": D.tolist(),
        "pieces": {k: v.tolist() for k, v in pieces.items()},
        "shift_from_guess": shift,
        "energy_parts_Ha": {
            "kinetic": float(np.einsum("ij,ji", pieces["T"], D)),
            "nuclear_attraction": float(np.einsum("ij,ji", pieces["V"], D)),
            "coulomb": float(0.5 * np.einsum("ij,ji", pieces["J"], D)),
            "nuclear_repulsion": float(mol.energy_nuc()),
            **{k + "_energy": v for k, v in energies.items()},
        },
        "hand_scf": hand_scf(mol, mf, S, nocc),
        "seconds": time.time() - t0,
    }
    (HERE / "data_ks.json").write_text(json.dumps(data))
    print(f"wrote data_ks.json in {data['seconds']:.0f} s")


if __name__ == "__main__":
    main()
