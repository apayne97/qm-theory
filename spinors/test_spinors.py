import numpy as np
import pytest

from spinors import (
    bloch_vector,
    commutator_norms,
    expectation_vs_time,
    interference_intensity,
    phasor_pair,
    rotation_axis_angle,
    rotation,
    rotation_matrix_so3,
    spin_matrices,
    su2_to_so3,
)

AXES = [(0, 0, 1), (1, 0, 0), (1, 2, -1)]


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_commutation_relations(j):
    jx, jy, jz = spin_matrices(j)
    assert np.allclose(jx @ jy - jy @ jx, 1j * jz)
    assert np.allclose(jy @ jz - jz @ jy, 1j * jx)


@pytest.mark.parametrize("axis", AXES)
def test_spin_half_two_pi_is_minus_one_four_pi_is_plus_one(axis):
    assert np.allclose(rotation(0.5, axis, 2 * np.pi), -np.eye(2))
    assert np.allclose(rotation(0.5, axis, 4 * np.pi), np.eye(2))


@pytest.mark.parametrize("axis", AXES)
def test_spin_one_returns_after_two_pi(axis):
    assert np.allclose(rotation(1, axis, 2 * np.pi), np.eye(3))


def test_half_integer_spin_flips_sign_integer_spin_does_not():
    for j, sign in [(0.5, -1), (1, 1), (1.5, -1), (2, 1)]:
        dim = int(2 * j + 1)
        assert np.allclose(rotation(j, (0, 1, 0), 2 * np.pi), sign * np.eye(dim))


def test_spin_half_rotation_is_su2():
    u = rotation(0.5, (1, 2, 3), 1.234)
    assert np.allclose(u.conj().T @ u, np.eye(2))
    assert np.isclose(np.linalg.det(u), 1.0)


@pytest.mark.parametrize("axis", AXES)
def test_bloch_vector_follows_ordinary_rotation(axis):
    """SU(2) rotation of a spinor == SO(3) rotation of its Bloch vector."""
    psi = np.array([np.cos(0.3), np.sin(0.3) * np.exp(0.7j)])
    theta = 1.1
    rotated = bloch_vector(rotation(0.5, axis, theta) @ psi)
    expected = rotation_matrix_so3(axis, theta) @ bloch_vector(psi)
    assert np.allclose(rotated, expected)


def test_bloch_vector_returns_at_two_pi_but_spinor_does_not():
    psi = np.array([1, 1j]) / np.sqrt(2)
    rotated = rotation(0.5, (0, 0, 1), 2 * np.pi) @ psi
    assert np.allclose(bloch_vector(rotated), bloch_vector(psi))
    assert np.allclose(rotated, -psi)


def test_interference_has_period_four_pi_for_spin_half():
    up = np.array([1, 0], dtype=complex)
    for theta in [0.0, 0.7, np.pi, 2 * np.pi, 3.0]:
        assert np.isclose(
            interference_intensity(up, 0.5, (0, 0, 1), theta), np.cos(theta / 4) ** 2
        )
    # full destructive interference after a 2*pi rotation of one arm
    assert np.isclose(interference_intensity(up, 0.5, (0, 0, 1), 2 * np.pi), 0.0)
    assert np.isclose(interference_intensity(up, 0.5, (0, 0, 1), 4 * np.pi), 1.0)


def test_interference_has_period_two_pi_for_spin_one():
    m1 = np.array([1, 0, 0], dtype=complex)  # m = +1 along z
    assert np.isclose(interference_intensity(m1, 1, (0, 0, 1), 2 * np.pi), 1.0)


@pytest.mark.parametrize("axis", AXES)
def test_su2_to_so3_matches_rodrigues_and_double_covers(axis):
    u = rotation(0.5, axis, 0.9)
    assert np.allclose(su2_to_so3(u), rotation_matrix_so3(axis, 0.9))
    assert np.allclose(su2_to_so3(-u), su2_to_so3(u))  # +U and -U: same rotation


def test_two_pi_loop_in_so3_is_open_path_in_su2():
    assert np.allclose(su2_to_so3(rotation(0.5, (0, 0, 1), 2 * np.pi)), np.eye(3))
    assert not np.allclose(rotation(0.5, (0, 0, 1), 2 * np.pi), np.eye(2))


def test_field_along_z_conserves_only_jz():
    h = -2 * spin_matrices(0.5)[2]  # H = -B sigma_z
    assert np.allclose(commutator_norms(h, 0.5)[2], 0)
    assert commutator_norms(h, 0.5)[0] > 0
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    jexp = expectation_vs_time(h, psi, 0.5, np.linspace(0, 3, 20))
    assert np.allclose(jexp[:, 2], jexp[0, 2])
    assert not np.allclose(jexp[:, 0], jexp[0, 0])


def test_phasor_pair_identifications():
    gamma, phi = 0.4, 1.3
    pair = phasor_pair(gamma, phi)
    assert np.allclose(pair, phasor_pair(gamma + 2 * np.pi, phi))
    assert np.allclose(pair, phasor_pair(gamma + np.pi, phi + 2 * np.pi))
    assert np.allclose(phasor_pair(gamma, phi + 2 * np.pi), -pair)  # 2*pi in phi flips the sign
    assert np.allclose(phasor_pair(gamma, phi + 4 * np.pi), pair)


def test_phasor_pair_interference_depends_only_on_phi_with_period_two_pi():
    for gamma in [0.0, 0.9, 2.5]:
        for phi in [0.0, 1.0, np.pi]:
            assert np.isclose(abs(phasor_pair(gamma, phi).sum()) ** 2, 2 + 2 * np.cos(phi))
            assert np.isclose(abs(phasor_pair(gamma, phi + 2 * np.pi).sum()) ** 2, 2 + 2 * np.cos(phi))


def test_rotations_about_different_axes_do_not_commute():
    rx = rotation_matrix_so3((1, 0, 0), np.pi / 2)
    ry = rotation_matrix_so3((0, 1, 0), np.pi / 2)
    assert not np.allclose(rx @ ry, ry @ rx)
    rz1 = rotation_matrix_so3((0, 0, 1), 0.3)
    rz2 = rotation_matrix_so3((0, 0, 1), 1.1)
    assert np.allclose(rz1 @ rz2, rz2 @ rz1)  # same axis commutes


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
def test_small_commutator_loop_is_a_z_rotation_by_eps_squared(eps):
    """R_x(e) R_y(e) R_x(-e) R_y(-e) ~ R_z(e^2): the group version of [Jx, Jy] = i Jz."""
    x, y = (1, 0, 0), (0, 1, 0)
    loop = (
        rotation_matrix_so3(x, eps) @ rotation_matrix_so3(y, eps)
        @ rotation_matrix_so3(x, -eps) @ rotation_matrix_so3(y, -eps)
    )
    axis, angle = rotation_axis_angle(loop)
    assert np.allclose(axis, (0, 0, 1), atol=eps)
    assert np.isclose(angle, eps**2, rtol=2 * eps)


def test_rotation_axis_angle_round_trip():
    axis, angle = rotation_axis_angle(rotation_matrix_so3((1, 2, 2), 0.8))
    assert np.allclose(axis, np.array([1, 2, 2]) / 3)
    assert np.isclose(angle, 0.8)


from spinors import commutator_loop, raising_operator  # noqa: E402


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_commutator_loop_is_z_rotation_by_eps_squared_in_every_representation(j):
    eps = 0.05
    dim = int(2 * j + 1)
    loop = commutator_loop(j, eps)
    assert np.linalg.norm(loop - rotation(j, (0, 0, 1), eps**2)) < 20 * eps**3
    assert np.linalg.norm(loop - np.eye(dim)) < 5 * eps**2 * (j + 1)  # near +1, even for half-integer j


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_raising_operator_shifts_m_by_exactly_one(j):
    jz = spin_matrices(j)[2]
    jp = raising_operator(j)
    assert np.allclose(jz @ jp - jp @ jz, jp)  # [Jz, J+] = +J+
    m = np.diag(jz).real
    rows, cols = np.nonzero(np.abs(jp) > 1e-12)
    assert np.allclose(m[rows] - m[cols], 1)


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_stern_gerlach_sequence_is_binomial(j):
    from math import comb

    from spinors import stern_gerlach_sequence

    n = int(round(2 * j))
    expected = np.array([comb(n, k) for k in range(n + 1)]) / 2**n
    p_x, p_z = stern_gerlach_sequence(j)
    assert np.allclose(p_x, expected)
    assert np.allclose(p_z, expected)
    assert np.isclose(p_x.sum(), 1) and np.isclose(p_z.sum(), 1)
