import numpy as np

from field_image import bessel_j0, field_height

G = np.linspace(-9, 9, 181)
XX, YY = np.meshgrid(G, G)


def test_the_excitation_is_at_the_origin_and_is_the_highest_point_without_background():
    z = field_height(XX, YY, ripple_amplitude=0.0)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    assert np.isclose(XX[i, j], 0) and np.isclose(YY[i, j], 0)
    assert np.isclose(z.max(), 1.0)


def test_the_bump_dies_away_far_from_the_origin():
    z = field_height(XX, YY, ripple_amplitude=0.0)
    far = np.hypot(XX, YY) > 8
    assert np.abs(z[far]).max() < 0.05


def test_bessel_j0_matches_known_values():
    assert np.isclose(bessel_j0(0.0), 1.0)
    assert abs(bessel_j0(2.404825557695773)) < 1e-9  # first zero
    assert abs(bessel_j0(5.520078110286311)) < 1e-9  # second zero
    assert np.isclose(bessel_j0(1.0), 0.7651976865579666, atol=1e-12)


def test_there_are_several_rings_around_the_peak():
    r = np.linspace(0, 8, 2000)
    z = field_height(r, 0.0 * r, ripple_amplitude=0.0)
    sign_changes = int(np.sum(np.sign(z[1:]) != np.sign(z[:-1])))
    assert sign_changes >= 4


def test_the_bump_is_circularly_symmetric():
    z = field_height(np.array([2.0, 0.0, -2.0, 0.0]), np.array([0.0, 2.0, 0.0, -2.0]), ripple_amplitude=0.0)
    assert np.allclose(z, z[0])


def test_the_background_is_small_compared_with_the_bump():
    z = field_height(XX, YY, ripple_amplitude=0.12)
    far = np.hypot(XX, YY) > 8
    assert np.abs(z[far]).max() < 0.15
