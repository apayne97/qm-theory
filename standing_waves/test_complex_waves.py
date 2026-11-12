import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from complex_waves import (
    angular_momentum_z,
    complex_vs_real_figure,
    complex_wave,
    cos_nodes,
    phase_winding,
    real_combination,
    real_wave,
)

PHI = np.linspace(0, 2 * np.pi, 400)


@pytest.mark.parametrize("m", [-3, -1, 0, 1, 2, 5])
def test_complex_wave_has_uniform_probability_for_any_overall_phase(m):
    for theta in (0.0, 0.7, 2.5):
        assert np.allclose(np.abs(complex_wave(m, PHI, theta)) ** 2, 1.0)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_overall_phase_does_not_change_probability_of_real_wave(m):
    p0 = np.abs(real_wave(m, PHI, 0.0)) ** 2
    assert np.allclose(np.abs(real_wave(m, PHI, 1.9)) ** 2, p0)


def test_complex_wave_is_single_valued_only_for_whole_m():
    assert np.isclose(complex_wave(3, 2 * np.pi), complex_wave(3, 0.0))
    assert not np.isclose(complex_wave(2.5, 2 * np.pi), complex_wave(2.5, 0.0))


@pytest.mark.parametrize("m", [-4, -1, 1, 3])
def test_phase_winds_m_times(m):
    assert phase_winding(m) == pytest.approx(m, abs=1e-6)


def test_overall_phase_moves_the_complex_real_part_but_not_the_real_wave_nodes():
    # Re of e^{i(m phi - theta)} = cos(m phi - theta): its zeros shift by theta/m. The real wave's zeros stay.
    m, theta = 2, 0.4
    assert not np.allclose(complex_wave(m, cos_nodes(m), theta).real, 0, atol=1e-3)
    assert np.allclose(complex_wave(m, cos_nodes(m) + theta / m, theta).real, 0, atol=1e-12)
    assert np.allclose(real_wave(m, cos_nodes(m), theta), 0, atol=1e-12)


@pytest.mark.parametrize("m", [1, 2, 3, 6])
def test_cos_nodes_count_and_spacing(m):
    nodes = cos_nodes(m)
    assert len(nodes) == 2 * m                      # 2|m| zeros round the ring = |m| planes
    assert np.allclose(np.diff(nodes), np.pi / m)   # equally spaced
    assert np.allclose(np.cos(m * nodes), 0, atol=1e-12)


def test_cos_nodes_empty_for_m_zero_and_symmetric_in_sign():
    assert len(cos_nodes(0)) == 0
    assert np.allclose(cos_nodes(-3), cos_nodes(3))


@pytest.mark.parametrize("m", [1, 2, 5])
def test_sum_of_plus_and_minus_m_is_the_real_cosine(m):
    assert np.allclose(real_combination(m, PHI), np.cos(m * PHI))
    assert np.allclose(real_combination(m, PHI).imag, 0)


def test_plus_and_minus_m_have_opposite_lz_and_the_cosine_is_a_mix():
    # -i d/dphi acting on e^{i m phi} gives m: check numerically with a finite difference
    m, h = 3, 1e-5
    f = complex_wave(m, 0.9)
    dfdphi = (complex_wave(m, 0.9 + h) - complex_wave(m, 0.9 - h)) / (2 * h)
    assert (-1j * dfdphi / f) == pytest.approx(m, abs=1e-6)
    assert angular_momentum_z(-m) == -angular_momentum_z(m)
    # for cos(m phi) the same ratio is not a constant, so it has no single value of L_z
    g = lambda p: np.cos(m * p)
    ratio = [(-1j * (g(p + h) - g(p - h)) / (2 * h) / g(p)) for p in (0.2, 0.5)]
    assert not np.isclose(ratio[0], ratio[1])


def test_figure_builds_for_several_settings():
    for m, theta in [(0, 0), (1, 45), (3, 200)]:
        fig = complex_vs_real_figure(m, theta)
        assert len(fig.axes) >= 4
        fig.canvas.draw()
        plt.close(fig)
