import numpy as np
import pytest

from dft_data import (
    HERE,
    damped_dispersion,
    erf_split,
    exact_exchange_fraction,
    levels_of_basis_function_count,
    load,
    shell_label,
)


def test_erf_split_adds_up_to_one_over_r():
    r = np.linspace(0.2, 10, 50)
    short, long_ = erf_split(r, 0.3)
    assert np.allclose(short + long_, 1 / r)


def test_exchange_fraction_goes_from_short_to_long_range_value():
    r = np.array([1e-6, 100.0])
    f = exact_exchange_fraction(r, 0.3, 0.15, 1.0)
    assert f[0] == pytest.approx(0.15, abs=1e-4)
    assert f[1] == pytest.approx(1.0, abs=1e-6)


def test_exchange_fraction_is_monotonic_when_long_exceeds_short():
    r = np.linspace(0.01, 20, 200)
    assert np.all(np.diff(exact_exchange_fraction(r, 0.3, 0.15, 1.0)) >= 0)


def test_basis_function_count_matches_pyscf_numbers_for_def2_tzvp_water():
    o = {"s": 5, "p": 3, "d": 2, "f": 1}
    h = {"s": 3, "p": 1}
    assert levels_of_basis_function_count(o) + 2 * levels_of_basis_function_count(h) == 43


def test_shell_label():
    assert shell_label({"s": 5, "p": 3, "d": 2, "f": 1}) == "[5s3p2d1f]"


def test_damped_dispersion_is_finite_at_zero_and_matches_bare_far_away():
    assert np.isfinite(damped_dispersion(0.0, 1.0, 2.0))
    assert damped_dispersion(50.0, 1.0, 2.0) == pytest.approx(-1 / 50.0**6, rel=1e-6)


data_present = all((HERE / f"data_{p}.json").exists() for p in ["scf", "basis", "ladder", "dispersion"])


@pytest.mark.skipif(not data_present, reason="run compute_dft_data.py first")
def test_computed_numbers_are_consistent():
    basis, scf, ladder, disp = load("basis"), load("scf"), load("ladder"), load("dispersion")
    # shell counts reproduce the number of basis functions (water = O + 2 H)
    for name, v in basis["basis"].items():
        n = levels_of_basis_function_count(v["O_shells"]) + 2 * levels_of_basis_function_count(v["H_shells"])
        assert n == v["n_basis"], name
    # SCF energy equals the last history entry, and the grid study at (99,590) reproduces it
    assert scf["scf_history_Ha"][-1] == pytest.approx(scf["e_tot_Ha"], abs=1e-9)
    assert scf["grid_study"]["99,590"]["e_Ha"] == pytest.approx(scf["e_tot_Ha"], abs=1e-6)
    # known facts: water HOMO ~ -12 eV needs range separation; methane dimer binds only with dispersion
    assert -ladder["functionals"]["wb97m_v"]["homo_eV"] > -ladder["functionals"]["pbe"]["homo_eV"] + 3
    assert disp["methods"]["HF"]["cp"] > 0 > disp["methods"]["wb97m_v"]["cp"]
    # BSSE (raw - CP) is smaller with diffuse functions
    d = basis["dimer"]
    assert abs(d["def2-tzvpd"]["raw"] - d["def2-tzvpd"]["cp"]) < abs(d["def2-svp"]["raw"] - d["def2-svp"]["cp"])
    # density at the oxygen nucleus is the largest value on the line
    line = scf["line"]
    assert max(line["rho"]) == pytest.approx(line["rho"][int(np.argmin(np.abs(np.array(line["s_bohr"]))))], rel=0.3)
