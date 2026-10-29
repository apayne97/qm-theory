"""Polarisation of light as a rotation problem: Jones vectors, the two circular states (m_s = +1 and -1),
wave plates, and twisted beams with orbital angular momentum m_l.

Conventions (used by the notebook and the figures)

- The wave travels along +z. At a fixed point the field is E(t) = Re[ J exp(-i w t) ] with the Jones vector
  J = (J_x, J_y) of two complex amplitudes. In J = (cos b, e^{i d} sin b) the phase difference d means that the
  y part lags the x part by d.
- Rotating the whole setup through alpha about z (counterclockwise seen with +z towards the viewer) acts as
  the matrix R(alpha) on J.
- m_s = +1 is e_plus = (1, i)/sqrt(2): R(alpha) e_plus = exp(-i alpha) e_plus, and the field turns
  counterclockwise (positive helicity). m_s = -1 is e_minus = (1, -i)/sqrt(2).
"""

import numpy as np

SQRT2 = np.sqrt(2.0)

E_X = np.array([1, 0], dtype=complex)
E_Y = np.array([0, 1], dtype=complex)
E_PLUS = np.array([1, 1j]) / SQRT2
E_MINUS = np.array([1, -1j]) / SQRT2

# J = i dR/d(alpha) at alpha = 0, as in section 2 of the spinor notebook.
J_MATRIX = np.array([[0, -1j], [1j, 0]])


def rotation_2d(alpha: float) -> np.ndarray:
    """Ordinary rotation of the xy-plane through alpha."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]], dtype=complex)


def linear(alpha: float) -> np.ndarray:
    """Linearly polarised light with the field along the direction at angle alpha from x."""
    return np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)


def jones(beta: float, delta: float) -> np.ndarray:
    """General state (cos b, e^{i d} sin b): x and y parts with a phase difference d (y lags x by d)."""
    return np.array([np.cos(beta), np.exp(1j * delta) * np.sin(beta)])


def helicity_amplitudes(j: np.ndarray) -> tuple[complex, complex]:
    """Amplitudes (a_plus, a_minus) of J on the states of m_s = +1 and -1."""
    return np.vdot(E_PLUS, j), np.vdot(E_MINUS, j)


def from_helicity(a_plus: complex, a_minus: complex) -> np.ndarray:
    """Jones vector with the given amplitudes on m_s = +1 and -1."""
    return a_plus * E_PLUS + a_minus * E_MINUS


def mean_m(j: np.ndarray) -> float:
    """Average m_s of a beam in state J: P(+1) - P(-1), between -1 (all minus) and +1 (all plus)."""
    a_plus, a_minus = helicity_amplitudes(j)
    p_plus, p_minus = abs(a_plus) ** 2, abs(a_minus) ** 2
    return float((p_plus - p_minus) / (p_plus + p_minus))


def field(j: np.ndarray, wt) -> np.ndarray:
    """Real field (E_x, E_y) at the times wt (radians). Shape (2,) for a number, (2, n) for an array."""
    wt = np.asarray(wt, dtype=float)
    out = np.real(j[:, None] * np.exp(-1j * np.atleast_1d(wt))[None, :])
    return out[:, 0] if wt.ndim == 0 else out


def ellipse_axes(j: np.ndarray) -> tuple[float, float, float]:
    """(semi-major, semi-minor, direction of the major axis) of the curve traced by the field tip.

    The field is the sum of two arrows turning in opposite senses, of lengths |a_plus|/sqrt2 and |a_minus|/sqrt2.
    They add to the major axis when they line up and subtract to the minor axis half a turn later, and the
    major axis points at (arg a_minus - arg a_plus)/2: the relative phase of the two amplitudes is twice the angle.
    """
    a_plus, a_minus = helicity_amplitudes(j)
    major = (abs(a_plus) + abs(a_minus)) / SQRT2
    minor = abs(abs(a_plus) - abs(a_minus)) / SQRT2
    angle = (np.angle(a_minus) - np.angle(a_plus)) / 2
    return float(major), float(minor), float(angle)


def counter_rotating_arrows(j: np.ndarray, wt: float) -> tuple[np.ndarray, np.ndarray]:
    """The m_s = +1 part (turns counterclockwise) and the m_s = -1 part (clockwise) of the real field at time wt."""
    a_plus, a_minus = helicity_amplitudes(j)
    part_plus = np.real(a_plus * E_PLUS * np.exp(-1j * wt))
    part_minus = np.real(a_minus * E_MINUS * np.exp(-1j * wt))
    return part_plus, part_minus


def retarder(gamma: float) -> np.ndarray:
    """Wave plate with its fast axis along x: the y part is delayed by the retardance gamma.

    gamma = pi/2 is a quarter-wave plate, gamma = pi a half-wave plate.
    """
    return np.diag([1, np.exp(1j * gamma)])


def plate_torque(j_in: np.ndarray, plate: np.ndarray) -> float:
    """Angular momentum (units of hbar, along the beam) handed to the plate per photon: m_in minus m_out."""
    return mean_m(j_in) - mean_m(plate @ j_in)


def phase_factor(m: float, alpha: float) -> complex:
    """Factor exp(-i m alpha) picked up by a state of definite m under a rotation through alpha."""
    return complex(np.exp(-1j * m * alpha))


def vortex_field(m_l: int, x, y, w: float = 1.0) -> np.ndarray:
    """Twisted beam: ring-shaped profile times exp(i m_l phi), with phi the angle round the axis."""
    r2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
    phi = np.arctan2(y, x)
    return (np.sqrt(r2) / w) ** abs(m_l) * np.exp(-r2 / w**2) * np.exp(1j * m_l * phi)


def ring_radius(m_l: int, w: float = 1.0) -> float:
    """Radius of the brightest ring of `vortex_field`: w sqrt(|m_l| / 2)."""
    return w * np.sqrt(abs(m_l) / 2)


def mean_orbital_m(psi_polar: np.ndarray) -> float:
    """Average of -i d/d(phi) for a wave sampled on a polar grid, array shape (n_radius, n_phi).

    Each integer frequency round the axis is a component of definite orbital m_l, so the average is the
    weighted mean of those integers.
    """
    spectrum = np.abs(np.fft.fft(psi_polar, axis=1)) ** 2
    freqs = np.fft.fftfreq(psi_polar.shape[1], d=1.0 / psi_polar.shape[1])
    return float((spectrum.sum(axis=0) * freqs).sum() / spectrum.sum())


def polar_grid(w: float = 1.0, n_radius: int = 200, n_phi: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Grid (r, phi) for `mean_orbital_m`, covering radius 0 to 5 w."""
    r = (np.arange(n_radius) + 0.5) / n_radius * 5 * w
    phi = np.arange(n_phi) / n_phi * 2 * np.pi
    return np.meshgrid(r, phi, indexing="ij")


def total_m(m_l: int, m_s: int) -> int:
    """Angular momentum along the beam of a twisted, polarised beam: orbital plus spin."""
    return m_l + m_s
