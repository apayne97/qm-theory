import numpy as np
import pytest

from standing_waves import (
    box_energy_ev,
    box_wave,
    count_states_below,
    degeneracies,
    energy,
    energy_units,
    levels,
    nodes_per_direction,
    smooth_count,
)


def test_wave_vanishes_on_every_wall():
    Ls = (1.0, 1.7, 0.8)
    ns = (2, 3, 4)
    for axis, L in enumerate(Ls):
        pts = [np.array(0.37), np.array(0.9), np.array(0.4)]
        for wall in (0.0, L):
            p = list(pts)
            p[axis] = np.array(wall)
            assert abs(box_wave(ns, p, Ls)) < 1e-12


def test_states_are_orthonormal_in_2d():
    Ls = (1.0, 1.5)
    x = np.linspace(0, Ls[0], 801)[:, None]
    y = np.linspace(0, Ls[1], 1201)[None, :]
    dA = (x[1, 0] - x[0, 0]) * (y[0, 1] - y[0, 0])

    def inner(a, b):
        return np.sum(box_wave(a, (x, y), Ls) * box_wave(b, (x, y), Ls)) * dA

    assert inner((1, 2), (1, 2)) == pytest.approx(1, abs=1e-3)
    assert inner((2, 3), (2, 3)) == pytest.approx(1, abs=1e-3)
    assert inner((1, 2), (2, 1)) == pytest.approx(0, abs=1e-3)
    assert inner((1, 1), (1, 3)) == pytest.approx(0, abs=1e-3)


def test_schroedinger_equation_in_3d_by_finite_differences():
    Ls = (1.0, 1.3, 0.9)
    ns = (2, 1, 3)
    h = 1e-3
    p = [np.array(0.31), np.array(0.62), np.array(0.27)]
    lap = 0.0
    for i in range(3):
        up, dn = list(p), list(p)
        up[i] = p[i] + h
        dn[i] = p[i] - h
        lap += (box_wave(ns, up, Ls) - 2 * box_wave(ns, p, Ls) + box_wave(ns, dn, Ls)) / h**2
    assert -0.5 * lap == pytest.approx(energy(ns, Ls) * box_wave(ns, p, Ls), rel=1e-4)


def test_square_and_cube_degeneracies():
    sq = dict((round(e), g) for e, g in degeneracies((1.0, 1.0), 60))
    assert sq[2] == 1 and sq[5] == 2 and sq[8] == 1 and sq[10] == 2 and sq[50] == 3  # 50: (1,7),(5,5),(7,1)
    cube = [(round(e), g) for e, g in degeneracies((1.0, 1.0, 1.0), 15)]
    assert cube == [(3, 1), (6, 3), (9, 3), (11, 3), (12, 1), (14, 6)]


def test_unequal_sides_lift_the_degeneracy():
    ds = degeneracies((1.0, 1.1), 20)
    assert all(g == 1 for _, g in ds)
    assert energy_units((1, 2), (1.0, 1.1)) != pytest.approx(energy_units((2, 1), (1.0, 1.1)))


def test_levels_are_sorted_and_complete():
    lv = levels((1.0, 1.0), 30)
    es = [e for e, _ in lv]
    assert es == sorted(es)
    assert count_states_below((1.0, 1.0), 30) == sum(g for _, g in degeneracies((1.0, 1.0), 30))
    # brute force count
    brute = sum(1 for a in range(1, 10) for b in range(1, 10) if a * a + b * b <= 30)
    assert count_states_below((1.0, 1.0), 30) == brute


def test_energy_matches_ev_formula_for_1d_box():
    # E1 for 1 nm is 0.376 eV; energy_units gives n^2
    e1 = float(box_energy_ev(1, 1.0))
    assert e1 == pytest.approx(0.376, abs=0.001)
    assert float(box_energy_ev(3, 1.0)) == pytest.approx(9 * e1)
    assert energy_units((3,), (1.0,)) == 9


def test_nodes():
    assert nodes_per_direction((1, 3, 2)) == (0, 2, 1)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_smooth_count_approaches_exact_count(dim):
    Ls = (1.0,) * dim
    e = 3000.0 if dim < 3 else 800.0
    exact = count_states_below(Ls, e)
    # the smooth estimate misses a surface correction of lower order, so it is a little too high
    assert exact == pytest.approx(smooth_count(dim, e), rel=0.1)
    assert exact <= smooth_count(dim, e) + 1
