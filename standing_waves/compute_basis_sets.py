"""PySCF numbers behind `basis_sets_from_base_states.py`; writes data_basis_sets.json next to this file.

The notebook only reads the JSON, so it needs no PySCF. Needs pyscf (2.14 tested) and numpy.
Run from this folder:  pixi run -e compute python compute_basis_sets.py
Contents:
  hydrogen   : for several basis sets, the UHF energy of the hydrogen atom (exact: -0.5 hartree; one electron, so UHF is
               exact within the basis), its dipole polarizability from a finite field (exact: 4.5 bohr^3) and the density
               along a line out from the nucleus (exact: exp(-2 r)/pi).
  h2plus     : total energy of H2+ in a very large basis, a curve in the distance R, to stand in for the exact answer.
  sto_ng     : exponents of the STO-3G and STO-6G hydrogen 1s function, scaled to a Slater exponent of 1 (they are tabulated
               for 1.24), to test the fit in basis_sets.py.
"""
import json
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf

HERE = Path(__file__).parent
FIELD = 0.002


def h_basis(name, drop_l=()):
    """Basis of hydrogen with the shells of angular momentum in drop_l removed."""
    shells = gto.basis.load(name, "H")
    return [sh for sh in shells if sh[0] not in drop_l]


def hydrogen_atom(basis, field=0.0):
    mol = gto.M(atom="H 0 0 0", spin=1, basis={"H": basis}, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    h0 = mf.get_hcore()
    if field:
        z = mol.intor("int1e_r")[2]
        mf.get_hcore = lambda *a, **k: h0 + field * z
    mf.kernel()
    assert mf.converged
    return mol, mf


def one_basis(label, basis):
    mol, mf = hydrogen_atom(basis)
    e0 = float(mf.e_tot)
    ep = float(hydrogen_atom(basis, FIELD)[1].e_tot)
    em = float(hydrogen_atom(basis, -FIELD)[1].e_tot)
    alpha = -(ep + em - 2 * e0) / FIELD**2
    r = np.linspace(0.0, 12.0, 241)
    pts = np.zeros((r.size, 3))
    pts[:, 2] = r
    ao = mol.eval_gto("GTOval", pts)
    dm = mf.make_rdm1()
    rho = np.einsum("pi,ij,pj->p", ao, dm[0] + dm[1], ao)
    return {"label": label, "n_basis": mol.nao, "energy": e0, "polarizability": float(alpha),
            "r": r.tolist(), "density": rho.tolist()}


def main():
    t0 = time.time()
    out = {"hydrogen": [], "h2plus": {}, "sto_ng": {}}
    cases = [
        ("STO-3G", h_basis("sto-3g")),
        ("def2-SVP", h_basis("def2-svp")),
        ("def2-TZVP, without its p shell", h_basis("def2-tzvp", drop_l=(1,))),
        ("def2-TZVP", h_basis("def2-tzvp")),
        ("def2-TZVPD", h_basis("def2-tzvpd")),
        ("def2-TZVPPD", h_basis("def2-tzvppd")),
        ("def2-QZVP", h_basis("def2-qzvp")),
    ]
    for label, basis in cases:
        out["hydrogen"].append(one_basis(label, basis))
        d = out["hydrogen"][-1]
        print(label, d["n_basis"], d["energy"], d["polarizability"], flush=True)
    rs = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    out["h2plus"]["R"] = rs
    out["h2plus"]["basis"] = "aug-cc-pV5Z"
    out["h2plus"]["energy"] = []
    for rr in rs:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {rr * 0.529177210903}", charge=1, spin=1, basis="aug-cc-pv5z", verbose=0)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        out["h2plus"]["energy"].append(float(mf.e_tot))
        print("H2+", rr, mf.e_tot, flush=True)
    for name in ("sto-3g", "sto-6g"):
        sh = gto.basis.load(name, "H")[0]
        out["sto_ng"][name] = {"exponents": [row[0] / 1.24**2 for row in sh[1:]], "coefficients": [row[1] for row in sh[1:]]}
    out["seconds"] = time.time() - t0
    (HERE / "data_basis_sets.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
