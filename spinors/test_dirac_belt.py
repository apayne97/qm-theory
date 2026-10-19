import numpy as np
import pytest

from dirac_belt import (
    belt_centerline,
    belt_frames,
    belt_quaternions,
    double_turn_quaternions,
    loop_quaternions,
    plate_quaternion,
    quaternion_to_matrix,
    ribbon_edges,
    rotation_vector,
    single_turn_quaternions,
    split_at_surface,
    surface_crossings,
)

S = np.linspace(0, 1, 41)
PHIS = np.linspace(0, 4 * np.pi, 33)


def matrices(phi):
    return quaternion_to_matrix(belt_quaternions(S, phi))


def rot_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@pytest.mark.parametrize("phi", PHIS)
def test_quaternions_are_unit_length(phi):
    assert np.allclose(np.linalg.norm(belt_quaternions(S, phi), axis=-1), 1)


@pytest.mark.parametrize("phi", PHIS)
def test_wall_and_plate_conditions(phi):
    q = belt_quaternions(S, phi)
    assert np.allclose(q[0], [1, 0, 0, 0])
    assert np.allclose(q[-1], plate_quaternion(phi))
    assert np.allclose(matrices(phi)[-1], rot_z(phi))


@pytest.mark.parametrize("phi", PHIS)
def test_matrices_are_rotations(phi):
    m = matrices(phi)
    assert np.allclose(m @ m.transpose(0, 2, 1), np.eye(3))
    assert np.allclose(np.linalg.det(m), 1)


def test_untwisted_at_zero_and_four_pi():
    for phi in (0.0, 4 * np.pi):
        assert np.allclose(matrices(phi), np.eye(3))


def test_belt_returns_after_four_pi_for_all_phi():
    for phi in PHIS[:8]:
        assert np.allclose(belt_quaternions(S, phi + 4 * np.pi), belt_quaternions(S, phi), atol=1e-9)


def test_belt_does_not_return_after_two_pi():
    """Plate looks the same after 2*pi but the belt differs, for every phi that is not 0 mod 4*pi... and 2*pi."""
    for phi in np.linspace(0.3, 1.9 * np.pi, 8):
        a = matrices(phi)
        b = matrices(phi + 2 * np.pi)
        assert np.allclose(a[-1], b[-1])  # plate orientation is the same
        assert np.max(np.abs(a - b)) > 0.1  # but the belt is not


def test_two_pi_belt_is_one_full_twist_about_y():
    m = matrices(2 * np.pi)
    expected = np.array([rot_y(2 * np.pi * s) for s in S])
    assert np.allclose(m, expected, atol=1e-9)


def test_family_is_continuous_in_phi():
    """No jumps in the belt as the plate turns, including through phi = 2*pi where the arc is not unique."""
    phis = np.linspace(0, 4 * np.pi, 2001)
    q = np.stack([belt_quaternions(S, p) for p in phis])
    step = np.max(np.abs(np.diff(q, axis=0)))
    assert step < 0.02


def test_centerline_straight_at_zero_two_pi_and_four_pi():
    for phi in (0.0, 2 * np.pi, 4 * np.pi):
        c = belt_centerline(belt_frames(phi, 200), length=1.0)
        assert np.allclose(c[-1], [0, 1, 0], atol=1e-3)
        assert np.allclose(c[:, 0], 0, atol=1e-6) and np.allclose(c[:, 2], 0, atol=1e-6)


def test_ribbon_edges_straddle_centerline():
    f = belt_frames(1.0, 50)
    c = belt_centerline(f)
    a, b = ribbon_edges(f, c, width=0.4)
    assert np.allclose((a + b) / 2, c)
    assert np.allclose(np.linalg.norm(a - b, axis=1), 0.4)


# --- the rotation ball --------------------------------------------------------


def test_rotation_vector_ignores_overall_sign():
    q = belt_quaternions(S, 1.3)
    assert np.allclose(rotation_vector(q), rotation_vector(-q))


def test_rotation_vector_of_z_rotation():
    q = plate_quaternion(np.array([0.5, 1.0, 3.0]))
    v = rotation_vector(q)
    assert np.allclose(v, [[0, 0, 0.5], [0, 0, 1.0], [0, 0, 3.0]])
    assert np.all(np.linalg.norm(rotation_vector(plate_quaternion(np.linspace(0, 8, 50))), axis=1) <= np.pi + 1e-9)


def test_single_turn_crosses_surface_once_and_double_turn_twice():
    assert surface_crossings(single_turn_quaternions()) == 1
    assert surface_crossings(double_turn_quaternions()) == 2


def test_loop_family_endpoints():
    great = loop_quaternions(np.pi / 2)
    assert np.allclose(great[0], [1, 0, 0, 0]) and np.allclose(great[-1], [1, 0, 0, 0], atol=1e-9)
    # kappa = pi/2 is the 4*pi rotation of the plate: its rotation vectors are along z
    assert np.allclose(rotation_vector(great)[:, :2], 0, atol=1e-9)
    tiny = loop_quaternions(1e-6)
    assert np.allclose(tiny, [1, 0, 0, 0], atol=1e-5)


@pytest.mark.parametrize("kappa", np.linspace(0.05, np.pi / 2, 8))
def test_loop_family_closed_unit_through_identity(kappa):
    q = loop_quaternions(kappa)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)
    assert np.allclose(q[0], [1, 0, 0, 0]) and np.allclose(q[-1], [1, 0, 0, 0], atol=1e-9)


def test_loop_family_is_a_loop_of_rotations_that_crosses_surface_an_even_number_of_times():
    for kappa in np.linspace(0.05, np.pi / 2, 12):
        assert surface_crossings(loop_quaternions(kappa)) % 2 == 0


def test_loop_family_changes_smoothly():
    kappas = np.linspace(0.0, np.pi / 2, 400)
    q = np.stack([loop_quaternions(k) for k in kappas])
    assert np.max(np.abs(np.diff(q, axis=0))) < 0.02


def test_split_at_surface_pieces_cover_path():
    q = double_turn_quaternions()
    v = rotation_vector(q)
    pieces = split_at_surface(v, q)
    assert len(pieces) == 3
    assert sum(len(p) for p in pieces) == len(v)


# --- the belt's quaternions are the 2x2 matrices of the notebook --------------


def quaternion_to_su2(q):
    """U = w - i (x sigma_x + y sigma_y + z sigma_z), the 2x2 matrix of the same rotation."""
    from spinors import spin_matrices

    sigma = 2 * spin_matrices(0.5)
    w, x, y, z = q
    return w * np.eye(2) - 1j * (x * sigma[0] + y * sigma[1] + z * sigma[2])


@pytest.mark.parametrize("phi", [0.4, 1.7, 3.0, 5.2, 8.0])
def test_belt_quaternions_are_the_su2_matrices(phi):
    from spinors import su2_to_so3

    q = belt_quaternions(np.array([0.0, 0.3, 0.7, 1.0]), phi)
    for qi in q:
        u = quaternion_to_su2(qi)
        assert np.allclose(u @ u.conj().T, np.eye(2))
        assert np.allclose(su2_to_so3(u), quaternion_to_matrix(qi), atol=1e-9)


def test_plate_end_of_belt_is_the_spin_half_z_matrix():
    """At the plate the belt's matrix is diag(exp(-i phi/2), exp(i phi/2)): +1 at 0, -1 at 2*pi, +1 at 4*pi."""
    for phi, sign in ((0.0, 1), (2 * np.pi, -1), (4 * np.pi, 1)):
        u = quaternion_to_su2(plate_quaternion(phi))
        assert np.allclose(u, sign * np.eye(2), atol=1e-12)
        u_belt = quaternion_to_su2(belt_quaternions(1.0, phi)) if phi < 4 * np.pi else quaternion_to_su2(plate_quaternion(phi))
        assert np.allclose(u_belt, sign * np.eye(2), atol=1e-12)
