"""PySCF numbers behind the orbital-size figure in `from_atoms_to_molecules.py`; writes data_atom_sizes.json next to this file.

For each neutral atom H to Ar (ground-state spin, UHF, def2-TZVP) the mean distance <r> of the highest occupied spin orbital
from the nucleus, in bohr, by numerical integration of |phi|^2 r on PySCF's atomic grid. The notebook only reads the JSON.
Run from this folder:  pixi run -e compute python compute_atom_sizes.py
"""
import json
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, scf

HERE = Path(__file__).parent
SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar".split()


def homo_radius(z):
    mol = gto.M(atom=f"{SYMBOLS[z - 1]} 0 0 0", basis="def2-tzvp", spin=HUND[z], verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged, SYMBOLS[z - 1]
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()
    ao = mol.eval_gto("GTOval", grids.coords)
    r = np.linalg.norm(grids.coords, axis=1)
    occ = mf.mo_occ[0] > 0
    homo = np.where(occ)[0][-1]
    phi = ao @ mf.mo_coeff[0][:, homo]
    norm = np.sum(grids.weights * phi**2)
    return float(np.sum(grids.weights * phi**2 * r) / norm), float(mf.mo_energy[0][homo])


# number of unpaired electrons in the ground state (Hund's rule)
HUND = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0}


def main():
    t0 = time.time()
    out = {"symbols": SYMBOLS, "z": list(range(1, 19)), "mean_radius_bohr": [], "homo_energy_hartree": []}
    for z in out["z"]:
        r, e = homo_radius(z)
        out["mean_radius_bohr"].append(r)
        out["homo_energy_hartree"].append(e)
        print(SYMBOLS[z - 1], round(r, 3), round(e, 3), flush=True)
    out["seconds"] = time.time() - t0
    (HERE / "data_atom_sizes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
