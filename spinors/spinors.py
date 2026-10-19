"""Rotations of spin-j states: why spin-1/2 needs 4*pi to come back.

A rotation by angle theta about unit axis n acts on a spin-j state as

    U(n, theta) = exp(-i * theta * (n . J))

where J = (Jx, Jy, Jz) are the spin matrices of dimension 2j+1.
"""

import numpy as np

SQRT2 = np.sqrt(2.0)


def spin_matrices(j: float) -> np.ndarray:
    """Return (Jx, Jy, Jz) as an array of shape (3, 2j+1, 2j+1), with hbar = 1."""
    dim = int(round(2 * j + 1))
    m = j - np.arange(dim)  # j, j-1, ..., -j
    jz = np.diag(m).astype(complex)
    # <m+1| J+ |m> = sqrt(j(j+1) - m(m+1)); J+ sits on the superdiagonal
    # because the basis is ordered from m = +j downward.
    jp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        mk = m[k]
        jp[k - 1, k] = np.sqrt(j * (j + 1) - mk * (mk + 1))
    jx = (jp + jp.conj().T) / 2
    jy = (jp - jp.conj().T) / (2j)
    return np.array([jx, jy, jz])


def rotation(j: float, axis, theta: float) -> np.ndarray:
    """exp(-i theta n.J) for spin j, built by diagonalising the Hermitian n.J."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    n_dot_j = np.tensordot(axis, spin_matrices(j), axes=1)
    eigvals, eigvecs = np.linalg.eigh(n_dot_j)
    return (eigvecs * np.exp(-1j * theta * eigvals)) @ eigvecs.conj().T


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """Expectation values (<sx>, <sy>, <sz>) of a spin-1/2 state (Pauli matrices)."""
    sx, sy, sz = 2 * spin_matrices(0.5)
    return np.array([np.vdot(psi, s @ psi).real for s in (sx, sy, sz)])


def rotation_matrix_so3(axis, theta: float) -> np.ndarray:
    """Ordinary 3x3 rotation matrix (Rodrigues' formula), for comparison with SU(2)."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def interference_intensity(psi: np.ndarray, j: float, axis, theta: float) -> float:
    """Output intensity when a beam is split, one arm is rotated, and they recombine.

    I = |(psi + U psi) / 2|^2 = (1 + Re <psi|U|psi>) / 2.
    The global phase of U is invisible on its own but shows up here, as a
    relative phase between the two arms.
    """
    u = rotation(j, axis, theta)
    return float(np.linalg.norm((psi + u @ psi) / 2) ** 2)


def su2_to_so3(u: np.ndarray) -> np.ndarray:
    """The 3x3 rotation that a 2x2 SU(2) matrix implements: U sigma_j U^dag = sum_i R_ij sigma_i.

    R_ij = (1/2) Tr(sigma_i U sigma_j U^dag). Note R(U) == R(-U): two spinor
    matrices for every physical rotation (the double cover).
    """
    sigma = 2 * spin_matrices(0.5)
    return np.array(
        [[0.5 * np.trace(si @ u @ sj @ u.conj().T).real for sj in sigma] for si in sigma]
    )


def bloch_azimuth(psi: np.ndarray) -> float:
    """Azimuthal angle of a spin-1/2 state's direction: arg(<sx> + i<sy>)."""
    bx, by, _ = bloch_vector(psi)
    return float(np.arctan2(by, bx))


def commutator_norms(h: np.ndarray, j: float) -> np.ndarray:
    """||[H, J_k]|| for k = x, y, z. Zero means J_k is conserved (a symmetry of H)."""
    return np.array([np.linalg.norm(h @ jk - jk @ h) for jk in spin_matrices(j)])


def expectation_vs_time(h: np.ndarray, psi: np.ndarray, j: float, times) -> np.ndarray:
    """<J> at each time under exp(-i H t), shape (len(times), 3)."""
    energies, vecs = np.linalg.eigh(h)
    jk = spin_matrices(j)
    out = []
    for t in times:
        psi_t = (vecs * np.exp(-1j * energies * t)) @ (vecs.conj().T @ psi)
        out.append([np.vdot(psi_t, m @ psi_t).real for m in jk])
    return np.array(out)


def phasor_pair(gamma: float, phi: float) -> np.ndarray:
    """Toy model: two unit phasors with mean phase gamma and relative phase phi.

    (a1, a2) = (e^{i(gamma - phi/2)}, e^{i(gamma + phi/2)}). Interference
    |a1 + a2|^2 = 2 + 2 cos(phi) depends on phi only, but the pair itself is
    unchanged only if (gamma, phi) -> (gamma + 2 pi, phi) or (gamma + pi, phi + 2 pi).
    """
    return np.exp(1j * np.array([gamma - phi / 2, gamma + phi / 2]))


def rotation_axis_angle(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis (unit vector) and angle in [0, pi] of a 3x3 rotation matrix."""
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    if np.isclose(angle, 0):
        return np.array([0.0, 0.0, 1.0]), 0.0
    eigvals, eigvecs = np.linalg.eig(r)
    axis = eigvecs[:, np.argmin(abs(eigvals - 1))].real
    axis = axis / np.linalg.norm(axis)
    antisym = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if axis @ antisym < 0:
        axis = -axis
    return axis, angle


def commutator_loop(j: float, eps: float) -> np.ndarray:
    """R_x(eps) R_y(eps) R_x(-eps) R_y(-eps) in the spin-j representation.

    For small eps this is close to R_z(eps^2), the group version of [Jx, Jy] = i Jz,
    and it is close to +1 (not -1) for every j.
    """
    x, y = (1, 0, 0), (0, 1, 0)
    return rotation(j, x, eps) @ rotation(j, y, eps) @ rotation(j, x, -eps) @ rotation(j, y, -eps)


def raising_operator(j: float) -> np.ndarray:
    """J+ = Jx + i Jy. It raises m by exactly 1."""
    jx, jy, _ = spin_matrices(j)
    return jx + 1j * jy
