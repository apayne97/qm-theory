"""Run the PySCF calculations behind `coupled_cluster.py`.

Writes, next to this file,
  data_cc.json            energies from PySCF (HF, MP2, CISD, CCSD, CCSD(T), FCI), counts and timings
  data_cc_integrals.npz   molecular-orbital integrals of the small systems, so that the notebook and the tests can
                          run the pure-numpy coupled-cluster code of `cc_numerics.py` without PySCF.

The notebook only reads these files. Needs `pyscf` (2.14 tested), numpy and click:

    pixi run -e compute python dft/compute_cc_data.py            # everything (a few minutes)
    pixi run -e compute python dft/compute_cc_data.py --part small
"""

import json
import time
from math import comb
from pathlib import Path

import sys

import click
import numpy as np
from pyscf import ao2mo, cc, ci, fci, gto, mp, scf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
BOHR = 0.529177210903

# Same water as compute_dft_data.py: O-H 0.9572 A, H-O-H 104.5 degrees (written as coordinates).
WATER_ATOMS = [("O", (0.0, 0.0, 0.117790)), ("H", (0.0, 0.755453, -0.471161)), ("H", (0.0, -0.755453, -0.471161))]
H2_BOND = 1.0  # angstrom: a little stretched, so that there is something to correlate


def water(stretch=1.0):
    """Water with both O-H bonds multiplied by `stretch` (the angle is kept)."""
    (_, o), (_, h1), (_, h2) = WATER_ATOMS
    o = np.array(o)
    atoms = [("O", o)]
    for _, h in ((None, h1), (None, h2)):
        atoms.append(("H", o + stretch * (np.array(h) - o)))
    return "\n".join(f"{e} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" for e, x in atoms)


def h2_pair(separation_angstrom=None):
    """One H2 along z at x=0, or two H2 with a second one displaced along x."""
    lines = [f"H 0 0 0", f"H 0 0 {H2_BOND}"]
    if separation_angstrom is not None:
        lines += [f"H {separation_angstrom} 0 0", f"H {separation_angstrom} 0 {H2_BOND}"]
    return "\n".join(lines)


def run_scf(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mol, mf


def energies(mol, mf, with_fci=False, with_triples=True, with_cisd=False):
    """Correlation energies (hartree) of the usual ladder, all electrons correlated."""
    out = {"n_basis": int(mol.nao), "n_electrons": int(mol.nelectron), "e_hf": float(mf.e_tot)}
    t = time.time()
    m = mp.MP2(mf).run()
    out["e_mp2_corr"] = float(m.e_corr)
    out["t_mp2_s"] = time.time() - t
    t = time.time()
    c = cc.CCSD(mf)
    c.conv_tol = 1e-11
    c.conv_tol_normt = 1e-9
    c.max_cycle = 300
    c.run()
    out["ccsd_converged"] = bool(c.converged)
    out["e_ccsd_corr"] = float(c.e_corr)
    out["t_ccsd_s"] = time.time() - t
    if with_triples:
        t = time.time()
        out["e_t"] = float(c.ccsd_t())
        out["t_t_s"] = time.time() - t
    if with_cisd:
        cs = ci.CISD(mf).run()
        out["e_cisd_corr"] = float(cs.e_corr)
    if with_fci:
        t = time.time()
        n_orb = mol.nao
        solver = fci.FCI(mf)
        solver.conv_tol = 1e-10
        e, _ = solver.kernel()
        out["e_fci_corr"] = float(e - mf.e_tot)
        out["t_fci_s"] = time.time() - t
        na = mol.nelectron // 2
        out["n_det_fci"] = comb(n_orb, na) ** 2
    return out


def mo_integrals(mol, mf):
    """h (n,n), (pq|rs) (n,n,n,n), occupied count, nuclear repulsion, orbital energies."""
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    return {"h": h, "eri": eri, "n_occ": np.array(mol.nelectron // 2), "e_nuc": np.array(mol.energy_nuc()),
            "mo_energy": mf.mo_energy, "e_hf": np.array(mf.e_tot)}


def part_small(store):
    """Systems small enough for the numpy code in cc_numerics.py: integrals plus PySCF reference energies."""
    systems = {}
    cases = {
        "h2_631g": (h2_pair(), "6-31g", True),
        "water_sto3g": (water(), "sto-3g", True),
        "water_631g": (water(), "6-31g", False),  # FCI for 6-31G is done in part_series
    }
    for name, (atom, basis, with_fci) in cases.items():
        mol, mf = run_scf(atom, basis)
        e = energies(mol, mf, with_fci=with_fci, with_cisd=True)
        e.update(basis=basis, geometry=atom)
        systems[name] = e
        for key, arr in mo_integrals(mol, mf).items():
            store[f"{name}__{key}"] = arr
        print(name, e, flush=True)
    return systems


def part_dimer():
    """Two H2 molecules 50 A apart, computed as one supermolecule (the far-apart check of size consistency)."""
    out = {}
    for label, atom in (("monomer", h2_pair()), ("dimer_50A", h2_pair(50.0))):
        mol, mf = run_scf(atom, "6-31g")
        e = energies(mol, mf, with_triples=False, with_cisd=True, with_fci=(label == "monomer"))
        out[label] = e
        print(label, e, flush=True)
    # also a series of separations, to show the size-consistency error appearing as the molecules separate
    scan = {}
    for r in (1.5, 2.0, 3.0, 5.0, 10.0, 50.0):
        mol, mf = run_scf(h2_pair(r), "6-31g")
        c = ci.CISD(mf).run()
        cc_ = cc.CCSD(mf).run()
        scan[str(r)] = {"e_hf": float(mf.e_tot), "e_cisd_corr": float(c.e_corr), "e_ccsd_corr": float(cc_.e_corr)}
        print("scan", r, scan[str(r)], flush=True)
    out["scan"] = scan
    return out


def part_series(with_stretch=True):
    """Water: the correlation-energy ladder up to FCI (6-31G), and with the O-H bonds stretched."""
    out = {"equilibrium": {}, "stretched": {}}
    for basis in ("sto-3g", "6-31g"):
        mol, mf = run_scf(water(), basis)
        out["equilibrium"][basis] = energies(mol, mf, with_fci=True)
        print("series", basis, out["equilibrium"][basis], flush=True)
    if with_stretch:
        for s in (1.0, 1.5, 2.0):
            mol, mf = run_scf(water(s), "6-31g")
            out["stretched"][str(s)] = energies(mol, mf, with_fci=True)
            print("stretch", s, out["stretched"][str(s)], flush=True)
    return out


def part_timing():
    """Time one CCSD amplitude update and one (T) evaluation of cc_numerics.py as the number of virtual orbitals grows.

    The integrals are random numbers with the right shape (the cost of the contractions does not depend on the values),
    with n_occ = 8 spin orbitals. Times are wall-clock seconds on this machine.
    """
    import cc_numerics as cn

    rng = np.random.default_rng(0)
    o = 8
    rows = []
    for v in (8, 12, 16, 24, 32, 40):
        n = o + v
        g = rng.standard_normal((n, n, n, n)) * 0.01
        g = g - g.transpose(1, 0, 2, 3)
        g = g - g.transpose(0, 1, 3, 2)
        f = np.diag(np.concatenate([-np.linspace(1.5, 0.5, o), np.linspace(0.5, 3.0, v)]))
        sys_ = {"n_occ": o, "n_so": n, "f": f, "g": g}
        t1, t2 = np.zeros((o, v)), g[:o, :o, o:, o:] * 0.1
        cn.ccsd_update(sys_, t1, t2)  # warm up
        t0 = time.time()
        cn.ccsd_update(sys_, t1, t2)
        row = {"n_occ_so": o, "n_virt_so": v, "t_update_s": time.time() - t0}
        if v <= 24:
            t0 = time.time()
            cn.triples_correction(sys_, t1, t2)
            row["t_triples_s"] = time.time() - t0
        rows.append(row)
        print("timing", row, flush=True)
    return rows


def part_counts():
    """Determinant counts of full CI for water at equilibrium in several basis sets (5 alpha and 5 beta electrons)."""
    rows = []
    for basis in ("sto-3g", "6-31g", "cc-pvdz", "cc-pvtz"):
        mol = gto.M(atom=water(), basis=basis, verbose=0)
        rows.append({"basis": basis, "n_basis": int(mol.nao), "n_det": comb(mol.nao, 5) ** 2})
    return rows


@click.command()
@click.option("--part", type=click.Choice(["all", "small", "dimer", "series", "timing"]), default="all", show_default=True,
              help="Which part to compute; the others are kept from the existing data_cc.json.")
@click.option("--out", type=click.Path(path_type=Path), default=HERE / "data_cc.json", show_default=True)
def main(part, out):
    data = json.loads(out.read_text()) if out.exists() else {}
    npz_path = HERE / "data_cc_integrals.npz"
    store = dict(np.load(npz_path)) if npz_path.exists() else {}
    t0 = time.time()
    data["pyscf_note"] = "computed with PySCF 2.14 (RHF reference, all electrons correlated, hartree)"
    if part in ("all", "small"):
        data["systems"] = part_small(store)
        np.savez_compressed(npz_path, **store)
    if part in ("all", "dimer"):
        data["h2_dimer"] = part_dimer()
    if part in ("all", "series"):
        data["water_series"] = part_series()
    if part in ("all", "timing"):
        data["timing"] = part_timing()
        data["fci_counts"] = part_counts()
    data["seconds_" + part] = time.time() - t0
    out.write_text(json.dumps(data, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
