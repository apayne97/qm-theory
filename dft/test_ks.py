"""Checks of the stored Kohn-Sham matrix data and of the numpy generalized-eigenproblem solver."""
import numpy as np

from dft_data import load
from ks_plots import solve_generalized


def arrays():
    ks = load("ks")
    return ks, np.array(ks["S"]), np.array(ks["F"]), np.array(ks["C"]), np.array(ks["eps_Ha"])


def test_shapes():
    ks, S, F, C, eps = arrays()
    n = ks["n_basis"]
    assert S.shape == F.shape == C.shape == (n, n) and len(eps) == n and ks["n_occ"] == 5


def test_f_is_sum_of_pieces():
    ks, _, F, _, _ = arrays()
    p = {k: np.array(v) for k, v in ks["pieces"].items()}
    assert np.abs(sum(p.values()) - F).max() < 1e-10


def test_generalized_eigenproblem_holds():
    _, S, F, C, eps = arrays()
    assert np.abs(F @ C - S @ C * eps).max() < 1e-6
    assert np.abs(C.T @ S @ C - np.eye(len(eps))).max() < 1e-10
    assert np.abs(C.T @ F @ C - np.diag(eps)).max() < 1e-6


def test_numpy_solver_reproduces_stored_eigenvalues():
    _, S, F, C, eps = arrays()
    e2, C2 = solve_generalized(F, S)
    assert np.abs(e2 - eps).max() < 1e-6
    assert np.abs(F @ C2 - S @ C2 * e2).max() < 1e-6


def test_density_from_occupied_orbitals():
    ks, S, _, C, _ = arrays()
    D = 2 * C[:, :5] @ C[:, :5].T
    assert np.abs(D - np.array(ks["D"])).max() < 1e-8
    assert abs(np.trace(D @ S) - 10) < 1e-8


def test_energy_matches_pyscf_total():
    ks, *_ = arrays()
    assert abs(ks["e_tot_Ha"] - (-76.4334)) < 1e-4
    assert ks["hand_scf"][-1]["energy_Ha"] - ks["e_tot_Ha"] < 1e-6


def test_toy_two_function_example():
    a, b, s = -0.5, -0.4, 0.6
    F, S = np.array([[a, b], [b, a]]), np.array([[1, s], [s, 1]])
    eps, _ = solve_generalized(F, S)
    assert np.allclose(sorted(eps), sorted([(a + b) / (1 + s), (a - b) / (1 - s)]))


def test_energy_parts_add_up_to_total():
    ks, *_ = arrays()
    assert abs(sum(ks["energy_parts_Ha"].values()) - ks["e_tot_Ha"]) < 1e-8
