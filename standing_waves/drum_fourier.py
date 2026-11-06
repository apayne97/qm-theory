"""A pattern on the circular drum as a sum of drum modes (a Fourier-Bessel series). numpy only.

The modes of a drum of radius 1 with a fixed rim are
    psi_k(r, phi) = N * J_l(j_{l, n_r+1} r) * cos(l phi)   (m_l = +l, l >= 0)
    psi_k(r, phi) = N * J_l(j_{l, n_r+1} r) * sin(l phi)   (m_l = -l, l >= 1)
with l the number of angular nodes, n_r the number of nodal circles, and N chosen so that
    integral of psi_k^2 over the disc (weight r dr dphi) = 1.
They are ordered by frequency j_{l, n_r+1}. Two different modes are orthogonal, so the coefficient of a pattern f is
    c_k = integral of f psi_k over the disc,    and    f is approximately the sum of c_k psi_k over the first M modes.
"""

from functools import lru_cache

import numpy as np

from hydrogen import bessel_j, bessel_zeros

M_MAX = 200
L_MAX, NR_MAX = 32, 12  # candidate modes: l < L_MAX, n_r < NR_MAX (enough for the first M_MAX modes)
N_R_QUAD, N_PHI_QUAD = 128, 256
RES = 161  # pixels of the top-view images
RES_SLIDE = 241
TARGETS = {
    "poke": "an off-centre bump (a poke)",
    "pluck": "a pluck along a diameter",
    "two": "two bumps of opposite sign",
    "blob": "a smooth, non-symmetric blob",
    "rose": "a rose (rings of petals)",
}


@lru_cache(maxsize=None)
def _zeros_all(l):
    """The first NR_MAX zeros of J_l."""
    return tuple(bessel_zeros(l, NR_MAX))


def _zeros(l, count):
    return _zeros_all(l)[:count]


@lru_cache(maxsize=None)
def mode_list(m_max=M_MAX):
    """The first m_max modes, ordered by frequency: a tuple of (l, n_r, m_l, j) with m_l = +l (cos) or -l (sin)."""
    cands = []
    for l in range(0, L_MAX):
        for nr in range(0, NR_MAX):
            j = _zeros(l, nr + 1)[-1]
            cands.append((j, l, nr))
    cands.sort()
    modes = []
    for j, l, nr in cands:
        modes.append((l, nr, l, j))
        if l > 0:
            modes.append((l, nr, -l, j))
        if len(modes) >= m_max + 2:
            break
    # every mode left out of the candidate set (l >= L_MAX or n_r >= NR_MAX) has a larger frequency than the ones kept
    assert modes[m_max - 1][3] < min(_zeros_all(L_MAX)[0], _zeros_all(0)[-1]), "candidate set too small"
    return tuple(modes[:m_max])


def _norm(l, j):
    ang = 2 * np.pi if l == 0 else np.pi
    return 1.0 / np.sqrt(ang * 0.5 * float(bessel_j(l + 1, np.array([j]))[0]) ** 2)


@lru_cache(maxsize=None)
def _radial_table(l, nr):
    r = np.linspace(0, 1, 801)
    j = _zeros(l, nr + 1)[-1]
    return r, _norm(l, j) * bessel_j(l, j * r)


def mode_values(k, r, phi):
    """Mode number k (0-based, in order of frequency) at distance r and angle phi."""
    l, nr, m, _ = mode_list()[k]
    rt, prof = _radial_table(l, nr)
    ang = np.cos(l * phi) if m >= 0 else np.sin(l * phi)
    return np.interp(r, rt, prof) * ang


@lru_cache(maxsize=None)
def quadrature():
    """Gauss-Legendre in r, equal steps in phi: (R, PHI, W) with W the weight of each node (including r dr dphi)."""
    x, w = np.polynomial.legendre.leggauss(N_R_QUAD)
    r, wr = 0.5 * (x + 1), 0.5 * w
    phi = 2 * np.pi * np.arange(N_PHI_QUAD) / N_PHI_QUAD
    return r[:, None], phi[None, :], (wr * r)[:, None] * (2 * np.pi / N_PHI_QUAD) * np.ones((1, N_PHI_QUAD))


@lru_cache(maxsize=None)
def modes_on_quadrature(m_max=M_MAX):
    """Array (m_max, n_r, n_phi): the modes at the quadrature nodes (computed exactly, not interpolated)."""
    r, phi, _ = quadrature()
    out = np.empty((m_max, r.shape[0], phi.shape[1]))
    for k, (l, nr, m, j) in enumerate(mode_list(m_max)):
        ang = np.cos(l * phi) if m >= 0 else np.sin(l * phi)
        out[k] = _norm(l, j) * bessel_j(l, j * r[:, 0])[:, None] * ang
    return out


@lru_cache(maxsize=None)
def display_grid(res=RES):
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x)
    return X, Y, np.hypot(X, Y), np.arctan2(Y, X)


@lru_cache(maxsize=None)
def modes_on_display(m_max=M_MAX, res=RES):
    _, _, R, PHI = display_grid(res)
    rr = np.minimum(R, 1.0)
    return np.array([mode_values(k, rr, PHI) for k in range(m_max)])


# rings of petals: (number of petals, radius of their centres, half-length along the ring's radial direction,
# half-width across it, tilt in radians, angular offset). Outer rings are painted last, on top of the inner ones.
ROSE_RINGS = (
    (3, 0.12, 0.13, 0.13, 0.9, 0.3),
    (5, 0.37, 0.21, 0.23, 0.9, 0.0),
    (7, 0.69, 0.29, 0.34, 0.9, 0.2),
)
ROSE_GAP, ROSE_EDGE = 0.06, 0.018  # width of the white gap round each petal, and softness of a petal edge


def rose(x, y):
    """A rose seen from above: rings of rounded, slightly wavy petals around a small spiral bud. Each petal is a flat
    red (negative) shape; the next petal is painted over the earlier ones and leaves a thin white gap round its edge.

    A petal is a smooth indicator 1/2 (1 - tanh(d / edge)) of its (wavy) ellipse, d being roughly the distance from the
    outline. Painting a petal means: erase (set to white) everything within the gap of its outline, then fill the petal.
    The picture is multiplied by an envelope 1 - r^12 so that it vanishes at the rim of the drum.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    value = np.zeros_like(x)
    for ring, (n, c, a, b, tilt, off) in enumerate(ROSE_RINGS):
        for k in range(n):
            th = 2 * np.pi * k / n + off
            dx, dy = x - c * np.cos(th), y - c * np.sin(th)
            e = th + tilt
            rho, t = dx * np.cos(e) + dy * np.sin(e), -dx * np.sin(e) + dy * np.cos(e)
            alpha = np.arctan2(t / b, rho / a)
            wave = 1 + 0.05 * np.cos(3 * alpha + 1.3 * k)  # slightly wavy edge
            d = (np.hypot(rho / a, t / b) / wave - 1) * min(a, b)  # about the distance from the outline (negative inside)
            erase = 0.5 * (1 - np.tanh((d - ROSE_GAP) / ROSE_EDGE))
            fill = 0.5 * (1 - np.tanh(d / ROSE_EDGE))
            value = np.clip(value * (1 - erase) + fill, 0.0, 1.0)  # erase the gap, then fill the petal
    r2 = x**2 + y**2
    # the bud: a small disc with a spiral gap
    u = 2.5 * np.arctan2(y, x) - 5.0 * np.log(r2 + 0.002)
    bud = 0.5 * (1 - np.tanh((np.sqrt(r2) - 0.075) / 0.015)) * (1 - 0.9 * np.exp(-np.sin(u / 2) ** 2 / 0.12**2))
    value = value * (1 - 0.5 * (1 - np.tanh((np.sqrt(r2) - 0.085) / 0.012))) + bud
    return -np.clip(value, 0, 1) * (1 - r2**6)


def target_function(key, x, y):
    """The target patterns; each is multiplied by (1 - r^2) so that it vanishes at the rim."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    taper = 1 - (x**2 + y**2)
    g = lambda cx, cy, s: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))  # noqa: E731
    if key == "poke":
        return taper * g(0.40, 0.20, 0.17)
    if key == "pluck":
        return taper * np.exp(-(y**2) / (2 * 0.07**2))
    if key == "two":
        return taper * (g(-0.42, 0.10, 0.2) - g(0.42, -0.10, 0.2))
    if key == "blob":
        return taper * np.exp(-((x - 0.25) ** 2 / 0.25 + (y + 0.15) ** 2 / 0.08 - 0.5 * (x - 0.25) * (y + 0.15)))
    if key == "rose":
        return rose(x, y)
    raise KeyError(key)


def _project(f_on_quad, m_max=M_MAX):
    _, _, W = quadrature()
    modes = modes_on_quadrature(m_max)
    return np.tensordot(modes, f_on_quad * W, axes=([1, 2], [0, 1]))


@lru_cache(maxsize=None)
def series(key, m_max=M_MAX):
    """Coefficients c_k of a target, its squared size, and the relative error after M = 0 ... m_max modes.

    Returns (c, norm2, rel_error); rel_error[M] = sqrt(1 - sum_{k<M} c_k^2 / norm2) is the size of what is left after M
    modes compared with the size of the pattern (it uses orthonormality: the 'energy' identity of the series).
    """
    r, phi, W = quadrature()
    f = target_function(key, r * np.cos(phi), r * np.sin(phi))
    c = _project(f, m_max)
    norm2 = float(np.sum(f**2 * W))
    rem = np.maximum(1 - np.concatenate([[0.0], np.cumsum(c**2)]) / norm2, 0.0)
    return c, norm2, np.sqrt(rem)


def approximation(key, M, res=RES):
    """The sum of the first M modes on the display grid (NaN outside the drum)."""
    c, _, _ = series(key)
    _, _, R, _ = display_grid(res)
    out = np.tensordot(c[:M], modes_on_display(M_MAX, res)[:M], axes=(0, 0)) if M > 0 else np.zeros_like(R)
    return np.where(R <= 1, out, np.nan)


def target_on_display(key, res=RES):
    X, Y, R, _ = display_grid(res)
    return np.where(R <= 1, target_function(key, X, Y), np.nan)
