"""Run the PySCF calculations behind `reading_a_dft_label.py` and write one JSON file per part.

The notebook only reads these files, so it needs no PySCF. Needs `pyscf` (2.14 tested) and numpy.
Run from this folder:  python compute_dft_data.py PART      with PART one of
scf, basis, ladder, dispersion, opls5.  Each part writes data_PART.json.
"""
import json
import time
from pathlib import Path

import click
import numpy as np
from pyscf import dft, gto, scf
from pyscf.dft import numint

HERE = Path(__file__).parent
HA2KCAL = 627.5095
HA2EV = 27.211386
BOHR = 0.529177210903

# Same geometries as ../../qm-calculations (water, rigid water dimer O...O 2.91 A).
WATER = """O 0 0 0.117790
H 0 0.755453 -0.471161
H 0 -0.755453 -0.471161"""
DIMER = [
    ("O", (0.0, 0.0, 0.0)), ("H", (-0.336048, 0.479927, 0.756950)), ("H", (-0.336048, 0.479927, -0.756950)),
    ("O", (2.91, 0.0, 0.0)), ("H", (1.9528, 0.0, 0.0)), ("H", (3.149987, 0.0, 0.926627)),
]
GRID = (99, 590)


def fmt(atoms, ghost=()):
    return "\n".join(
        f"{'ghost-' if i in ghost else ''}{el} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" for i, (el, x) in enumerate(atoms)
    )


def make_mf(mol, method, grid=GRID):
    """method: 'HF', or any PySCF xc name; 'wb97m_v' gets VV10, 'wb97m_v_novv10' has it switched off."""
    if method == "HF":
        return scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.grids.atom_grid = grid
    if method == "wb97m_v_novv10":
        mf.xc = "wb97m_v"
        mf.nlc = False  # nlc = "" would leave VV10 on (see the methane-dimer notes)
    else:
        mf.xc = method
        if method == "wb97m_v":
            mf.nlc = "VV10" if not getattr(mf, "nlc", None) else mf.nlc
    return mf


def energy(atoms, method, basis, ghost=(), coupled_cluster=False):
    mol = gto.M(atom=fmt(atoms, ghost), basis=basis, verbose=0)
    mf = make_mf(mol, "HF" if coupled_cluster else method)
    mf.kernel()
    assert mf.converged, (method, basis)
    if not coupled_cluster:
        return mf.e_tot
    from pyscf import cc
    ccsd = cc.CCSD(mf).run()
    return mf.e_tot + ccsd.e_corr + ccsd.ccsd_t()


def interaction(atoms, n_a, method, basis, coupled_cluster=False):
    """Raw and counterpoise-corrected interaction energy in kcal/mol; monomer A = first n_a atoms."""
    idx_a, idx_b = set(range(n_a)), set(range(n_a, len(atoms)))
    kw = dict(coupled_cluster=coupled_cluster)
    e_ab = energy(atoms, method, basis, **kw)
    e_a = energy([a for i, a in enumerate(atoms) if i in idx_a], method, basis, **kw)
    e_b = energy([a for i, a in enumerate(atoms) if i in idx_b], method, basis, **kw)
    e_a_g = energy(atoms, method, basis, ghost=idx_b, **kw)
    e_b_g = energy(atoms, method, basis, ghost=idx_a, **kw)
    return {"raw": (e_ab - e_a - e_b) * HA2KCAL, "cp": (e_ab - e_a_g - e_b_g) * HA2KCAL}


def part_scf():
    mol = gto.M(atom=WATER, basis="def2-tzvpd", verbose=0)
    out = {"n_basis": mol.nao, "n_electrons": mol.nelectron}
    mf = make_mf(mol, "wb97m_v")
    history = []
    mf.callback = lambda env: history.append(float(env["e_tot"]))
    mf.kernel()
    out["scf_history_Ha"] = history
    out["e_tot_Ha"] = float(mf.e_tot)
    out["orbital_energies_Ha"] = [float(e) for e in mf.mo_energy[: mol.nelectron // 2 + 1]]
    # PySCF returns (omega, long-range fraction, short-range fraction) of exact exchange; checked below
    out["omega"], out["long_range_exchange"], out["short_range_exchange"] = [
        float(x) for x in mf._numint.rsh_and_hybrid_coeff("wb97m_v")
    ]
    # integration grid study (same functional and basis)
    grids = {}
    for g in [(20, 50), (35, 110), (50, 194), (75, 302), (99, 590), (200, 974)]:
        m = make_mf(gto.M(atom=WATER, basis="def2-tzvpd", verbose=0), "wb97m_v", grid=g)
        t = time.time()
        e = m.kernel()
        grids[f"{g[0]},{g[1]}"] = {"e_Ha": float(e), "n_points": int(len(m.grids.weights)), "seconds": time.time() - t}
    out["grid_study"] = grids
    # fields along the O-H axis, starting a little behind the oxygen
    o = np.array([0, 0, 0.117790]) / BOHR
    h = np.array([0, 0.755453, -0.471161]) / BOHR
    u = (h - o) / np.linalg.norm(h - o)
    s = np.linspace(-1.5, 4.0, 221)  # bohr along the O->H direction from O
    pts = o[None, :] + s[:, None] * u[None, :]
    ao = numint.eval_ao(mol, pts, deriv=1)
    dm = mf.make_rdm1()
    rho, gx, gy, gz, _lap, tau = numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=True)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    out["line"] = {
        "s_bohr": s.tolist(), "rho": rho.tolist(), "grad_rho": grad.tolist(), "tau": tau.tolist(),
        "oh_length_bohr": float(np.linalg.norm(h - o)),
        # reduced gradient s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3))
        "reduced_gradient": (grad / (2 * (3 * np.pi**2) ** (1 / 3) * np.maximum(rho, 1e-12) ** (4 / 3))).tolist(),
        # von Weizsaecker kinetic energy density |grad rho|^2 / (8 rho); tau >= tau_W
        "tau_w": (grad**2 / (8 * np.maximum(rho, 1e-12))).tolist(),
    }
    return out


def shells(mol, element):
    """Contracted shells of ONE atom of the element as {l: count}, e.g. {'s': 5, 'p': 3, 'd': 2, 'f': 1}."""
    counts = {}
    first = next(i for i in range(mol.natm) if mol.atom_symbol(i) == element)
    for ib in range(mol.nbas):
        if mol.bas_atom(ib) == first:
            letter = "spdfgh"[mol.bas_angular(ib)]
            counts[letter] = counts.get(letter, 0) + mol.bas_nctr(ib)
    return counts


def part_basis():
    out = {"basis": {}}
    for b in ["sto-3g", "def2-svp", "def2-tzvp", "def2-tzvpd", "def2-tzvpp", "def2-tzvppd", "def2-qzvp"]:
        mol = gto.M(atom=WATER, basis=b, verbose=0)
        mf = make_mf(mol, "wb97m_v")
        mf.kernel()
        out["basis"][b] = {
            "n_basis": mol.nao, "O_shells": shells(mol, "O"), "H_shells": shells(mol, "H"),
            "e_water_Ha": float(mf.e_tot), "homo_Ha": float(mf.mo_energy[mol.nelectron // 2 - 1]),
        }
        print(b, out["basis"][b], flush=True)
    out["dimer"] = {}
    for b in ["def2-svp", "def2-tzvp", "def2-tzvpd", "def2-tzvppd"]:
        out["dimer"][b] = interaction(DIMER, 3, "wb97m_v", b)
        print(b, out["dimer"][b], flush=True)
    # primitives and contraction of the oxygen 1s and valence shells in def2-TZVP, for the picture
    mol = gto.M(atom=WATER, basis="def2-tzvpd", verbose=0)
    prims = []
    for ib in range(mol.nbas):
        if mol.atom_symbol(mol.bas_atom(ib)) == "O":
            prims.append({"l": int(mol.bas_angular(ib)), "exps": mol.bas_exp(ib).tolist(),
                          "coeffs": mol.bas_ctr_coeff(ib).T.tolist()})
    out["O_def2_tzvpd_shells"] = prims
    return out


def part_ladder():
    out = {"functionals": {}}
    for name in ["HF", "svwn", "pbe", "r2scan", "b3lyp", "wb97x", "wb97m_v"]:
        mol = gto.M(atom=WATER, basis="def2-tzvpd", verbose=0)
        mf = make_mf(mol, name)
        mf.kernel()
        homo = float(mf.mo_energy[mol.nelectron // 2 - 1]) * HA2EV
        res = interaction(DIMER, 3, name, "def2-tzvpd")
        out["functionals"][name] = {"homo_eV": homo, **res}
        print(name, out["functionals"][name], flush=True)
    out["ccsdt"] = interaction(DIMER, 3, "HF", "def2-tzvpd", coupled_cluster=True)
    print("CCSD(T)", out["ccsdt"], flush=True)
    return out


def methane(r_cc):
    r_ch = 1.087
    apex = [("C", np.zeros(3)), ("H", np.array([0, 0, -r_ch]))]
    for k in range(3):
        a = np.radians(120 * k)
        rad = r_ch * np.sqrt(8) / 3
        apex.append(("H", np.array([rad * np.cos(a), rad * np.sin(a), r_ch / 3])))
    other = [(el, np.array([0, 0, r_cc]) - x) for el, x in apex]
    return apex + other


def part_dispersion():
    atoms = methane(3.8)
    out = {"r_cc": 3.8, "methods": {}}
    for name in ["HF", "b3lyp", "wb97m_v_novv10", "wb97m_v"]:
        out["methods"][name] = interaction(atoms, 5, name, "def2-tzvpd")
        print(name, out["methods"][name], flush=True)
    out["ccsdt"] = interaction(atoms, 5, "HF", "def2-tzvpd", coupled_cluster=True)
    print("CCSD(T)", out["ccsdt"], flush=True)
    return out


def benzene():
    """D6h benzene, r(CC) = 1.397 A, r(CH) = 1.087 A (a standard geometry, not the one used in the OPLS5 paper)."""
    atoms = []
    for k in range(6):
        a = np.radians(60 * k)
        atoms.append(("C", (1.397 * np.cos(a), 1.397 * np.sin(a), 0.0)))
        atoms.append(("H", ((1.397 + 1.087) * np.cos(a), (1.397 + 1.087) * np.sin(a), 0.0)))
    return atoms


def strip_f(basis_name, elements):
    """The 'def2-TZVPD(-f)' idea: the named basis with every shell of angular momentum >= 3 removed."""
    out = {}
    for el in elements:
        out[el] = [sh for sh in gto.basis.load(basis_name, el) if sh[0] < 3]
    return out


def count(atoms, basis, ecp=None, charge=0, spin=0):
    mol = gto.M(atom=fmt(atoms), basis=basis, ecp=ecp, charge=charge, spin=spin, verbose=0)
    return {"n_basis": int(mol.nao), "n_electrons": int(mol.nelectron),
            "n_electrons_all": int(sum(gto.charge(el) for el, _ in atoms) - charge)}


# cisplatin-like Pt(NH3)2Cl2 (square planar, illustrative geometry; only counts are used)
PT_COMPLEX = [("Pt", (0, 0, 0)), ("Cl", (2.33, 0, 0)), ("Cl", (0, 2.33, 0)), ("N", (-2.05, 0, 0)), ("N", (0, -2.05, 0)),
              ("H", (-2.45, 0.9, 0)), ("H", (-2.45, -0.45, 0.78)), ("H", (-2.45, -0.45, -0.78)),
              ("H", (0.9, -2.45, 0)), ("H", (-0.45, -2.45, 0.78)), ("H", (-0.45, -2.45, -0.78))]


def part_opls5():
    out = {}
    # (a) exchange fractions for the functionals named in the OPLS5 paper, as PySCF/libxc report them
    out["rsh"] = {}
    for label, name in [("wB97X", "HYB_GGA_XC_WB97X"), ("wB97X-D", "HYB_GGA_XC_WB97X_D"),
                       ("wB97X-D3", "HYB_GGA_XC_WB97X_D3"), ("wB97M-V", "wb97m_v")]:
        w, lr, sr_minus_lr = dft.libxc.rsh_coeff(name)
        out["rsh"][label] = {"omega": float(w), "long_range": float(lr), "short_range": float(lr + sr_minus_lr)}
    # (c) +1 point charge above the ring centre of benzene, wB97X-D functional, 6-311++G**, as in Table 1 of the paper.
    # The -D term is pair-wise between atoms, and a point charge has none, so it adds nothing to this interaction.
    mol = gto.M(atom=fmt(benzene()), basis="6-311++g**", verbose=0)
    def run(z=None):
        mf = dft.RKS(mol)
        mf.xc = "HYB_GGA_XC_WB97X_D"
        mf.grids.atom_grid = (75, 302)
        if z is not None:
            from pyscf import qmmm
            mf = qmmm.mm_charge(mf, [[0, 0, z]], [1.0], unit="Angstrom")
        mf.kernel()
        assert mf.converged
        return mf.e_tot
    e0 = run()
    out["probe"] = {"n_basis": int(mol.nao), "paper_QM": {"3.5": -13.6, "4.0": -9.5},
                    "computed": {f"{z}": (run(z) - e0) * HA2KCAL for z in (3.5, 4.0)}}
    print(out["probe"], flush=True)
    # basis function counts: benzene and a platinum complex
    bz = benzene()
    out["benzene"] = {
        "6-31+G*": count(bz, "6-31+g*"), "6-311++G**": count(bz, "6-311++g**"),
        "def2-TZVPD": count(bz, "def2-tzvpd"), "def2-TZVPD(-f)": count(bz, strip_f("def2-tzvpd", ["C", "H"])),
    }
    els = ["Cl", "N", "H"]
    light = {"Cl": "6-31+g*", "N": "6-31+g*", "H": "6-31+g*"}
    out["pt_complex"] = {
        "LANL2DZ on Pt, 6-31+G* elsewhere (ECP)": count(PT_COMPLEX, {"Pt": "lanl2dz", **light}, ecp={"Pt": "lanl2dz"}),
        "def2-TZVPD with def2 ECP on Pt": count(PT_COMPLEX, "def2-tzvpd", ecp={"Pt": "def2-tzvpd"}),
        "def2-TZVPD(-f) with def2 ECP on Pt": count(
            PT_COMPLEX, {**strip_f("def2-tzvpd", els), "Pt": "def2-tzvpd"}, ecp={"Pt": "def2-tzvpd"}),
    }
    print(out["benzene"], out["pt_complex"], flush=True)
    return out


PARTS = {"scf": part_scf, "basis": part_basis, "ladder": part_ladder, "dispersion": part_dispersion, "opls5": part_opls5}


@click.command()
@click.argument("part", type=click.Choice(sorted(PARTS)), default="scf")
def main(part):
    t = time.time()
    data = PARTS[part]()
    data["seconds"] = time.time() - t
    (HERE / f"data_{part}.json").write_text(json.dumps(data, indent=1))
    print(f"wrote data_{part}.json in {data['seconds']:.0f} s")


if __name__ == "__main__":
    main()
