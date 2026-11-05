"""pytest checks of cc_numerics.py against PySCF reference numbers stored by compute_cc_data.py.

The tests need no PySCF: the molecular-orbital integrals are in data_cc_integrals.npz and PySCF's energies in
data_cc.json. Run `pixi run -e compute python dft/compute_cc_data.py` to regenerate both.
"""

import json
from math import comb

import numpy as np
import pytest

import cc_numerics as cn
from dft_data import HERE

DATA = HERE / "data_cc.json"
NPZ = HERE / "data_cc_integrals.npz"
pytestmark = pytest.mark.skipif(not (DATA.exists() and NPZ.exists()), reason="run compute_cc_data.py first")


@pytest.fixture(scope="module")
def ref():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def systems():
    npz = np.load(NPZ)
    out = {}
    for name in ("h2_631g", "water_sto3g", "water_631g"):
        out[name] = cn.spin_orbital_system(npz[f"{name}__h"], npz[f"{name}__eri"], int(npz[f"{name}__n_occ"]),
                                           float(npz[f"{name}__e_nuc"]))
    return out


@pytest.fixture(scope="module")
def solved(systems):
    return {name: cn.ccsd(s) for name, s in systems.items()}


NAMES = ["h2_631g", "water_sto3g", "water_631g"]


@pytest.mark.parametrize("name", NAMES)
def test_hartree_fock_energy_from_spin_orbitals(name, systems, ref):
    assert systems[name]["e_hf"] == pytest.approx(ref["systems"][name]["e_hf"], abs=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_fock_matrix_is_diagonal_for_canonical_orbitals(name, systems):
    f = systems[name]["f"]
    assert np.abs(f - np.diag(np.diag(f))).max() < 1e-6


@pytest.mark.parametrize("name", NAMES)
def test_mp2_matches_pyscf(name, systems, ref):
    assert cn.mp2_energy(systems[name]) == pytest.approx(ref["systems"][name]["e_mp2_corr"], abs=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_ccsd_matches_pyscf_to_1e8(name, solved, ref):
    assert solved[name]["converged"]
    assert solved[name]["e_corr"] == pytest.approx(ref["systems"][name]["e_ccsd_corr"], abs=1e-8)


@pytest.mark.parametrize("name", ["water_sto3g", "water_631g"])
def test_triples_correction_matches_pyscf(name, systems, solved, ref):
    e_t = cn.triples_correction(systems[name], solved[name]["t1"], solved[name]["t2"])
    assert e_t == pytest.approx(ref["systems"][name]["e_t"], abs=1e-8)
    assert e_t < 0


def test_diis_does_not_change_the_answer(systems):
    s = systems["water_sto3g"]
    a, b = cn.ccsd(s, diis=True), cn.ccsd(s, diis=False, max_iter=300)
    assert b["converged"] and a["iterations"] < b["iterations"]
    assert a["e_corr"] == pytest.approx(b["e_corr"], abs=1e-9)


def test_first_iteration_energy_is_mp2(systems):
    r = cn.ccsd(systems["water_sto3g"])
    assert r["history"][0] == pytest.approx(cn.mp2_energy(systems["water_sto3g"]), abs=1e-12)


def test_expanded_ccd_equation_equals_the_intermediate_form(systems):
    s = systems["water_sto3g"]
    rng = np.random.default_rng(1)
    t2 = cn.mp2_amplitudes(s) + 0.02 * rng.standard_normal(cn.mp2_amplitudes(s).shape)
    t2 = t2 - t2.swapaxes(0, 1)  # keep the antisymmetry of real amplitudes
    t2 = t2 - t2.swapaxes(2, 3)
    t1 = np.zeros((s["n_occ"], s["n_so"] - s["n_occ"]))
    _, intermediate_form = cn.ccsd_update(s, t1, t2, singles=False)
    assert np.allclose(cn.ccd_update_expanded(s, t2), intermediate_form, atol=1e-12)


def test_ccd_is_close_to_ccsd_but_not_equal(systems, ref):
    r = cn.ccsd(systems["water_sto3g"], singles=False)
    assert r["converged"] and not np.any(r["t1"])
    diff = r["e_corr"] - ref["systems"]["water_sto3g"]["e_ccsd_corr"]
    assert 1e-8 < abs(diff) < 2e-3


# ---- configuration interaction --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2_631g", "water_sto3g"])
def test_full_ci_matches_pyscf(name, systems, ref):
    r = cn.ci_solve(systems[name])
    assert r["e_corr"] == pytest.approx(ref["systems"][name]["e_fci_corr"], abs=1e-8)


@pytest.mark.parametrize("name", ["h2_631g", "water_sto3g"])
def test_cisd_matches_pyscf(name, systems, ref):
    r = cn.ci_solve(systems[name], level=2)
    assert r["e_corr"] == pytest.approx(ref["systems"][name]["e_cisd_corr"], abs=1e-8)


def test_determinant_counts(systems, ref):
    assert len(cn.determinants(systems["water_sto3g"])) == cn.fci_dimension(7, 5, 5) == 441
    assert cn.fci_dimension(13, 5, 5) == ref["water_series"]["equilibrium"]["6-31g"]["n_det_fci"] == 1656369
    assert cn.fci_dimension(58, 5, 5) == comb(58, 5) ** 2


def test_ci_energies_decrease_as_the_space_grows(systems):
    s = systems["water_sto3g"]
    e = [cn.ci_solve(s, level=k)["e_corr"] for k in (1, 2, 3, 4)] + [cn.ci_solve(s)["e_corr"]]
    assert all(b <= a + 1e-12 for a, b in zip(e, e[1:]))
    assert e[0] == pytest.approx(0.0, abs=1e-9)  # singles do not mix with the Hartree-Fock determinant (Brillouin)


def test_doubles_dominate_the_water_wave_function(systems):
    s = systems["water_sto3g"]
    w = cn.excitation_level_weights(cn.ci_solve(s), 10)
    assert w[0] > 0.97 and w[2] > w[1] and w[2] > w[3] and sum(w.values()) == pytest.approx(1.0)


# ---- two electrons: CCSD is exact -------------------------------------------------------------------------------


def test_two_electron_system_ccsd_cisd_fci_all_agree(systems, solved):
    s = systems["h2_631g"]
    fci = cn.ci_solve(s)["e_corr"]
    assert fci < -0.03
    assert solved["h2_631g"]["e_corr"] == pytest.approx(fci, abs=1e-9)
    assert cn.ci_solve(s, level=2)["e_corr"] == pytest.approx(fci, abs=1e-10)
    assert cn.triples_correction(s, solved["h2_631g"]["t1"], solved["h2_631g"]["t2"]) == pytest.approx(0.0, abs=1e-12)
    assert cn.mp2_energy(s) > fci  # MP2 recovers only part of it


def test_ccsd_amplitude_equals_ci_coefficient_for_two_electrons(systems, solved):
    s, r = systems["h2_631g"], solved["h2_631g"]
    ci = cn.ci_solve(s)
    t1, t2 = r["t1"], r["t2"]
    # sigma_g^2 -> sigma_u^2 is the move of spin orbitals 0, 1 to 2, 3 (virtual labels 0 and 1).
    # CI coefficient of a double = t_ij^ab + t_i^a t_j^b - t_i^b t_j^a (the amplitude of e^T on that determinant).
    tau = t2[0, 1, 0, 1] + t1[0, 0] * t1[1, 1] - t1[0, 1] * t1[1, 0]
    assert abs(cn.coefficient(ci, [(0, 2), (1, 3)])) == pytest.approx(abs(tau), rel=1e-6)


# ---- size consistency ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(systems):
    return cn.combine_noninteracting(systems["h2_631g"], systems["h2_631g"])


def test_hartree_fock_is_additive(systems, pair):
    assert pair["e_hf"] == pytest.approx(2 * systems["h2_631g"]["e_hf"], abs=1e-10)


def test_ccsd_energy_is_additive(systems, pair, solved):
    e2 = cn.ccsd(pair)["e_corr"]
    assert e2 == pytest.approx(2 * solved["h2_631g"]["e_corr"], abs=1e-9)


def test_mp2_energy_is_additive(systems, pair):
    assert cn.mp2_energy(pair) == pytest.approx(2 * cn.mp2_energy(systems["h2_631g"]), abs=1e-12)


def test_cisd_energy_is_not_additive(systems, pair):
    one = cn.ci_solve(systems["h2_631g"], level=2)["e_corr"]
    two = cn.ci_solve(pair, level=2)["e_corr"]
    assert two - 2 * one > 1e-3  # CISD of the pair is higher than twice CISD of one: not size consistent


def test_ccsd_of_the_pair_is_exact(pair):
    assert cn.ccsd(pair)["e_corr"] == pytest.approx(cn.ci_solve(pair)["e_corr"], abs=1e-9)


def test_numpy_pair_agrees_with_pyscf_supermolecule_at_50_angstrom(pair, ref):
    d = ref["h2_dimer"]["dimer_50A"]
    assert cn.ci_solve(pair, level=2)["e_corr"] == pytest.approx(d["e_cisd_corr"], abs=1e-6)
    assert cn.ccsd(pair)["e_corr"] == pytest.approx(d["e_ccsd_corr"], abs=1e-6)


def test_quadruple_coefficient_is_the_product_of_two_doubles(pair):
    """In the exact wave function of two independent molecules the amplitude of 'both doubly excited' is c_A * c_B.

    Combined ordering: occupied of A = 0,1; occupied of B = 2,3; virtual of A start at 4; virtual of B at 10.
    """
    fci = cn.ci_solve(pair)
    c_a = cn.coefficient(fci, [(0, 4), (1, 5)])
    c_b = cn.coefficient(fci, [(2, 10), (3, 11)])
    c_ab = cn.coefficient(fci, [(0, 4), (1, 5), (2, 10), (3, 11)])
    assert abs(c_a) > 0.05
    assert abs(c_ab) == pytest.approx(abs(c_a * c_b), rel=1e-6)


def test_two_waters_ccsd_and_triples_are_additive(systems, solved):
    s = systems["water_sto3g"]
    pair = cn.combine_noninteracting(s, s)
    r = cn.ccsd(pair)
    assert r["e_corr"] == pytest.approx(2 * solved["water_sto3g"]["e_corr"], abs=1e-9)
    e_t = cn.triples_correction(s, solved["water_sto3g"]["t1"], solved["water_sto3g"]["t2"])
    assert cn.triples_correction(pair, r["t1"], r["t2"]) == pytest.approx(2 * e_t, abs=1e-9)


def test_pyscf_scan_shows_cisd_error_growing_to_a_constant_and_ccsd_error_vanishing(ref):
    mono = ref["h2_dimer"]["monomer"]["e_ccsd_corr"]
    mono_hf = ref["h2_dimer"]["monomer"]["e_hf"]
    scan = ref["h2_dimer"]["scan"]
    cisd_err = {float(r): v["e_cisd_corr"] - 2 * mono for r, v in scan.items()}
    ccsd_err = {float(r): v["e_ccsd_corr"] - 2 * mono for r, v in scan.items()}
    assert cisd_err[50.0] > 1e-3 and abs(cisd_err[50.0] - cisd_err[10.0]) < 1e-5
    assert abs(ccsd_err[50.0]) < 1e-7
    assert scan["50.0"]["e_hf"] == pytest.approx(2 * mono_hf, abs=1e-7)


# ---- the numbers used in the accuracy ladder --------------------------------------------------------------------


def test_water_ladder_orders_as_expected(ref):
    e = ref["water_series"]["equilibrium"]["6-31g"]
    assert e["e_fci_corr"] < e["e_ccsd_corr"] < e["e_mp2_corr"] < 0
    err_ccsd = abs(e["e_ccsd_corr"] - e["e_fci_corr"])
    err_t = abs(e["e_ccsd_corr"] + e["e_t"] - e["e_fci_corr"])
    assert err_t < err_ccsd / 2


def test_stretched_water_ccsd_t_overshoots_fci(ref):
    e = ref["water_series"]["stretched"]["2.0"]
    assert e["e_ccsd_corr"] + e["e_t"] < e["e_fci_corr"]  # (T) goes below the exact answer: not variational


def test_existing_dft_ladder_has_coupled_cluster_reference():
    from dft_data import load
    assert load("ladder")["ccsdt"]["cp"] < load("ladder")["functionals"]["HF"]["cp"]
    assert load("dispersion")["ccsdt"]["cp"] < 0 < load("dispersion")["methods"]["HF"]["cp"]


def test_operation_counts_grow_in_the_expected_order():
    a, b = cn.operation_counts(5, 20), cn.operation_counts(5, 40)
    assert b["ccsd"] / a["ccsd"] == 16 and b["ccsd(t)"] / a["ccsd(t)"] == 16
