"""Spherical harmonics and the hydrogen atom, in numpy only (no scipy).

Atomic units: hbar = m_e = e = 4 pi eps0 = 1. Lengths are in bohr (a0 = 0.0529 nm), energies in hartree
(1 Eh = 27.211 eV), and the nucleus is a fixed point charge.

    wave function    psi_nlm(r, theta, phi) = R_nl(r) Y_lm(theta, phi)
    angular part     -(1/sin t) d/dt (sin t dY/dt) - (1/sin^2 t) d^2Y/dphi^2 = l (l + 1) Y
    radial part      u = r R:   -u''/2 + [ l (l + 1) / (2 r^2) - 1/r ] u = E u,     E_n = -1 / (2 n^2)

Conventions: P_l^m (m >= 0) is the associated Legendre function WITHOUT the Condon-Shortley sign, and
Y_lm = N_lm P_l^|m|(cos theta) exp(i m phi) for every m. Real harmonics use cos(m phi) for m > 0 and sin(|m| phi)
for m < 0. Both choices change only signs, never shapes, nodes or energies.
"""

from functools import lru_cache
from math import factorial

import numpy as np

HARTREE_EV = 27.211386
BOHR_NM = 0.0529177


# ---------------------------------------------------------------------------------------------------------------
# Angular part


def legendre_assoc(l, m, x):
    """Associated Legendre function P_l^m(x) for 0 <= m <= l (no Condon-Shortley phase), by upward recursion."""
    if not 0 <= m <= l:
        raise ValueError("need 0 <= m <= l")
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    if m > 0:
        pmm = np.prod(np.arange(1, 2 * m, 2)) * (1.0 - x * x) ** (m / 2)
    if l == m:
        return pmm
    pm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pm1
    for k in range(m + 2, l + 1):
        pmm, pm1 = pm1, ((2 * k - 1) * x * pm1 - (k + m - 1) * pmm) / (k - m)
    return pm1


def ylm_norm(l, m):
    """Normalisation constant N_lm of Y_lm, so that the integral of |Y_lm|^2 over the sphere is 1."""
    m = abs(m)
    return np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))


def ylm(l, m, theta, phi):
    """Complex spherical harmonic Y_lm(theta, phi)."""
    return ylm_norm(l, m) * legendre_assoc(l, abs(m), np.cos(theta)) * np.exp(1j * m * np.asarray(phi))


def real_ylm(l, m, theta, phi):
    """Real spherical harmonic: P cos(m phi) for m > 0, P for m = 0, P sin(|m| phi) for m < 0 (all normalised)."""
    p = ylm_norm(l, m) * legendre_assoc(l, abs(m), np.cos(theta))
    if m == 0:
        return p
    trig = np.cos(m * np.asarray(phi)) if m > 0 else np.sin(-m * np.asarray(phi))
    return np.sqrt(2.0) * p * trig


def sphere_quadrature(n_theta=80, n_phi=160):
    """Gauss-Legendre in cos(theta) and uniform in phi: returns (theta, phi, weights), each of shape (n_theta, n_phi).
    The weights sum to 4 pi and integrate any smooth function of low degree exactly."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    theta = np.arccos(x)[:, None] * np.ones((1, n_phi))
    weights = w[:, None] * (2 * np.pi / n_phi) * np.ones((1, n_phi))
    return theta, np.ones((n_theta, 1)) * phi[None, :], weights


def sphere_inner(f, g, n_theta=80, n_phi=160):
    """Integral over the sphere of conj(f) g, with f and g functions of (theta, phi)."""
    theta, phi, w = sphere_quadrature(n_theta, n_phi)
    return np.sum(np.conj(f(theta, phi)) * g(theta, phi) * w)


def angular_operator_fd(f, theta, phi, h=1e-3):
    """The angular operator  -(1/sin t) d/dt (sin t df/dt) - (1/sin^2 t) d^2f/dphi^2  of a function f(theta, phi),
    by central finite differences (step h). It is the quantity whose eigenvalue is l (l + 1)."""
    s_p, s_m = np.sin(theta + h / 2), np.sin(theta - h / 2)
    d_theta = (s_p * (f(theta + h, phi) - f(theta, phi)) / h - s_m * (f(theta, phi) - f(theta - h, phi)) / h) / h
    d_phi = (f(theta, phi + h) - 2 * f(theta, phi) + f(theta, phi - h)) / h**2
    return -(d_theta / np.sin(theta) + d_phi / np.sin(theta) ** 2)


def angular_degeneracy(l):
    """Number of angular states with the same l: m = -l, ..., l."""
    return len(range(-l, l + 1))


def nodal_structure(l, m):
    """(circles through the poles, circles of latitude): |m| and l - |m|. In total l nodal lines."""
    return abs(m), l - abs(m)


def latitude_nodes(l, m):
    """Polar angles (radians, increasing) of the circles of latitude where Y_lm = 0, from the zeros of P_l^|m|."""
    m = abs(m)
    x = np.linspace(-1 + 1e-9, 1 - 1e-9, 20000)  # even count: x = 0 is not a grid point
    p = legendre_assoc(l, m, x)
    idx = np.where(np.sign(p[:-1]) * np.sign(p[1:]) < 0)[0]
    roots = [x[i] - p[i] * (x[i + 1] - x[i]) / (p[i + 1] - p[i]) for i in idx]
    return np.sort(np.arccos(roots))


def latitude_nodes_predicted(l, m):
    """Polar angles (radians, increasing) of the circles of latitude, from the roots of a polynomial.

    P_l^|m|(x) = (1 - x^2)^(|m|/2) times the |m|-th derivative of the Legendre polynomial P_l(x). The first factor
    vanishes only at the poles, so the latitude nodes are the l - |m| roots of the derivative, with x = cos(theta).
    This route uses numpy's polynomial roots, not the recursion or the sign scan of `latitude_nodes`."""
    m = abs(m)
    coef = np.zeros(l + 1)
    coef[l] = 1.0
    roots = np.polynomial.legendre.Legendre(coef).deriv(m).roots() if l > m else np.array([])
    return np.sort(np.arccos(np.real(roots)))


def azimuth_nodes(m, kind="cos"):
    """Azimuths (radians, in [0, pi)) of the |m| planes through the z axis where the real harmonic vanishes.

    cos(m phi) = 0 at phi = pi/(2m) + k pi/m; sin(m phi) = 0 at phi = k pi/m, with k = 0, ..., |m| - 1. Each plane
    contains the two meridians phi and phi + pi, so the values repeat every pi."""
    m = abs(m)
    k = np.arange(m)
    return (np.pi / (2 * m) if kind == "cos" else 0.0) + k * np.pi / m if m else np.array([])


def count_sign_changes(values):
    """Number of sign changes in a sequence, ignoring exact zeros."""
    s = np.sign(np.asarray(values))
    s = s[s != 0]
    return int(np.sum(s[:-1] * s[1:] < 0))


# ---------------------------------------------------------------------------------------------------------------
# Why l is a whole number: the solution along the polar angle must stay finite on the axis


def polar_pole_coefficient(ell, parity, theta_end=1e-3, steps=4000):
    """For m = 0, solve  y'' + cot(t) y' + ell (ell + 1) y = 0  from the equator (t = pi/2) toward the pole.

    Start even (y = 1, y' = 0) or odd (y = 0, y' = 1) about the equator. Near the pole every solution looks like
    y = a + c ln t. The solution stays finite only if c = 0. Returns c = t y'(t) at t = theta_end (an array if `ell`
    is an array). Whole ell makes c vanish for one parity: even ell with the even start, odd ell with the odd start.
    The integration is in s = ln t, where the equation is smooth (RK4).
    """
    ell = np.asarray(ell, dtype=float)
    lam = ell * (ell + 1)
    s0, s1 = np.log(np.pi / 2), np.log(theta_end)
    y = np.zeros_like(lam) if parity == "odd" else np.ones_like(lam)
    w = np.full_like(lam, np.pi / 2 if parity == "odd" else 0.0)  # w = t y'

    def rhs(s, y, w):
        t = np.exp(s)
        return w, w - t / np.tan(t) * w - t * t * lam * y

    h = (s1 - s0) / steps
    s = s0
    for _ in range(steps):
        k1 = rhs(s, y, w)
        k2 = rhs(s + h / 2, y + h / 2 * k1[0], w + h / 2 * k1[1])
        k3 = rhs(s + h / 2, y + h / 2 * k2[0], w + h / 2 * k2[1])
        k4 = rhs(s + h, y + h * k3[0], w + h * k3[1])
        y = y + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        s += h
    return w


def polar_solution(ell, parity, theta_end=1e-3, steps=1500):
    """The same solution as in `polar_pole_coefficient` for one value of ell, returned as (theta, y) from the equator
    (theta = pi/2) down to theta_end, for plotting."""
    lam = float(ell) * (float(ell) + 1)
    s0, s1 = np.log(np.pi / 2), np.log(theta_end)
    y, w = (0.0, np.pi / 2) if parity == "odd" else (1.0, 0.0)

    def rhs(s, y, w):
        t = np.exp(s)
        return w, w - t / np.tan(t) * w - t * t * lam * y

    h = (s1 - s0) / steps
    s = s0
    ts, ys = [np.exp(s)], [y]
    for _ in range(steps):
        k1 = rhs(s, y, w)
        k2 = rhs(s + h / 2, y + h / 2 * k1[0], w + h / 2 * k1[1])
        k3 = rhs(s + h / 2, y + h / 2 * k2[0], w + h / 2 * k2[1])
        k4 = rhs(s + h, y + h * k3[0], w + h * k3[1])
        y += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        s += h
        ts.append(np.exp(s))
        ys.append(y)
    return np.array(ts), np.array(ys)


# ---------------------------------------------------------------------------------------------------------------
# Radial part


def hydrogen_energy(n):
    """E_n = -1 / (2 n^2) hartree."""
    return -0.5 / np.asarray(n, dtype=float) ** 2


def hydrogen_energy_ev(n):
    """E_n = -13.606 / n^2 eV."""
    return hydrogen_energy(n) * HARTREE_EV


def effective_potential(r, l):
    """-1/r + l (l + 1) / (2 r^2): attraction of the nucleus plus the 'centrifugal' cost of angular curvature."""
    r = np.asarray(r, dtype=float)
    return -1.0 / r + l * (l + 1) / (2.0 * r * r)


def laguerre(k, alpha, x):
    """Generalised Laguerre polynomial L_k^alpha(x), by the three-term recursion."""
    x = np.asarray(x, dtype=float)
    l0 = np.ones_like(x)
    if k == 0:
        return l0
    l1 = 1.0 + alpha - x
    for j in range(1, k):
        l0, l1 = l1, ((2 * j + 1 + alpha - x) * l1 - (j + alpha) * l0) / (j + 1)
    return l1


def radial_R(n, l, r):
    """Exact hydrogen radial function R_nl(r), normalised so that the integral of R^2 r^2 dr is 1."""
    if not 0 <= l < n:
        raise ValueError("need 0 <= l < n")
    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / n
    norm = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho**l * laguerre(n - l - 1, 2 * l + 1, rho)


def radial_u(n, l, r):
    """u = r R, the function the radial equation is written for; the integral of u^2 dr is 1."""
    return np.asarray(r, dtype=float) * radial_R(n, l, r)


def mean_radius(n, l):
    """<r> = (3 n^2 - l (l + 1)) / 2 bohr."""
    return 0.5 * (3 * n * n - l * (l + 1))


@lru_cache(maxsize=None)
def solve_radial(l, h=0.05, n_grid=1600, n_states=6):
    """Lowest radial states for angular number l by finite differences.

    Grid r_i = i h (i = 1 ... n_grid) with u = 0 at r = 0 and at the far end; the second derivative is the
    three-point difference. Returns (r, energies, u) with u[:, k] normalised (sum u^2 h = 1) and with a positive
    first lobe, k = 0 ... n_states - 1 (the state with k radial nodes, n = l + 1 + k).
    """
    r = h * np.arange(1, n_grid + 1)
    diag = 1.0 / h**2 + effective_potential(r, l)
    off = -0.5 / h**2 * np.ones(n_grid - 1)
    mat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    e, v = np.linalg.eigh(mat)
    e, v = e[:n_states], v[:, :n_states]
    v = v / np.sqrt(np.sum(v**2, axis=0) * h)
    first = np.array([col[np.argmax(np.abs(col) > 1e-3 * np.abs(col).max())] for col in v.T])
    v = v * np.where(first < 0, -1.0, 1.0)
    return r, e, v


def numerical_levels(l_max, n_max, **kw):
    """Dict {(n, l): finite-difference energy in hartree} for l <= l_max, n <= n_max."""
    out = {}
    for l in range(l_max + 1):
        _, e, _ = solve_radial(l, **kw)
        for k, ek in enumerate(e):
            n = l + 1 + k
            if n <= n_max:
                out[(n, l)] = float(ek)
    return out


def radial_probability(n, l, r):
    """Probability per unit distance of finding the electron at distance r: r^2 R^2 = u^2."""
    return radial_u(n, l, r) ** 2


# ---------------------------------------------------------------------------------------------------------------
# Whole orbitals


def psi_real(n, l, m, x, y, z):
    """Real orbital R_nl(r) * real Y_lm at Cartesian points (broadcastable arrays), in atomic units."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r == 0, 1e-12, r)
    theta = np.arccos(np.clip(z / r_safe, -1, 1))
    phi = np.arctan2(y, x)
    return radial_R(n, l, r) * real_ylm(l, m, theta, phi)


def psi_complex(n, l, m, r, theta, phi):
    """Complex orbital R_nl(r) Y_lm(theta, phi)."""
    return radial_R(n, l, r) * ylm(l, m, theta, phi)


def quantum_numbers(n):
    """All (l, m) allowed for a given n: l = 0 ... n - 1, m = -l ... l."""
    return [(l, m) for l in range(n) for m in range(-l, l + 1)]


def degeneracy(n):
    """Number of orbitals in level n (without spin): n^2."""
    return len(quantum_numbers(n))


def orbital_name(n, l, m=None):
    """'2p', '3d', ... with the real-orbital label (z, x, y, xy, ...) when m is given and l <= 2."""
    letter = "spdfgh"[l]
    base = f"{n}{letter}"
    if m is None:
        return base
    labels = {
        (0, 0): "", (1, 0): "z", (1, 1): "x", (1, -1): "y",
        (2, 0): "z²", (2, 1): "xz", (2, -1): "yz", (2, 2): "x²−y²", (2, -2): "xy",
    }
    return base + labels.get((l, m), f" m={m}")


def cross_section_plane(m, extent, resolution=400):
    """Grid in a plane containing the z axis, chosen so that a real orbital with this m does not vanish in it:
    phi0 = 0 for m >= 0 (the xz plane), phi0 = pi / (2 |m|) for m < 0. Returns (s, z, x, y) with s the signed distance
    from the axis along the in-plane direction (cos phi0, sin phi0)."""
    phi0 = 0.0 if m >= 0 else np.pi / (2 * abs(m))
    s = np.linspace(-extent, extent, resolution)[None, :]
    z = np.linspace(-extent, extent, resolution)[:, None]
    return s, z, s * np.cos(phi0) + 0 * z, s * np.sin(phi0) + 0 * z


def sample_density(n, l, m, count, extent, seed=0):
    """`count` points drawn from |psi|^2 of the real orbital by rejection sampling in a cube of half-side `extent`."""
    rng = np.random.default_rng(seed)
    probe = rng.uniform(-extent, extent, size=(200000, 3))
    top = np.max(psi_real(n, l, m, *probe.T) ** 2) * 1.2
    pts = []
    got = 0
    while got < count:
        c = rng.uniform(-extent, extent, size=(200000, 3))
        keep = rng.uniform(0, top, size=len(c)) < psi_real(n, l, m, *c.T) ** 2
        pts.append(c[keep])
        got += keep.sum()
    return np.concatenate(pts)[:count]


# ---------------------------------------------------------------------------------------------------------------
# The drumhead: a circular membrane with a fixed rim (radius 1), the two-dimensional counterpart of the atom


def bessel_j(m, x, n_points=400):
    """Bessel function J_m(x) of whole order m from its integral (1/pi) int_0^pi cos(m t - x sin t) dt, evaluated by
    the trapezoid rule on a periodic integrand (exponentially accurate)."""
    x = np.asarray(x, dtype=float)
    t = (np.arange(n_points) + 0.5) * np.pi / n_points
    return np.mean(np.cos(m * t - x[..., None] * np.sin(t)), axis=-1)


def bessel_zeros(m, count):
    """The first `count` positive zeros j_{m,1}, j_{m,2}, ... of J_m (scan for sign changes, then bisection)."""
    zeros = []
    x0, step = 0.5 + m * 0.5 if m > 0 else 0.5, 0.05
    while len(zeros) < count:
        a, b = x0, x0 + step
        fa, fb = bessel_j(m, a), bessel_j(m, b)
        if fa * fb < 0:
            for _ in range(60):
                c = 0.5 * (a + b)
                if bessel_j(m, c) * fa > 0:
                    a = c
                else:
                    b = c
            zeros.append(0.5 * (a + b))
        x0 += step
    return np.array(zeros)


def drum_mode(m, k, r, phi):
    """Real drum mode J_m(j_{m,k} r) cos(m phi) on the unit disc (zero on the rim r = 1)."""
    return bessel_j(m, bessel_zeros(m, k)[-1] * np.asarray(r, dtype=float)) * np.cos(m * np.asarray(phi, dtype=float))


def drum_frequency(m, k):
    """Frequency of the mode in units where the lowest (m = 0, k = 1) is 2.405: it equals j_{m,k}."""
    return float(bessel_zeros(m, k)[-1])


def drum_nodes(m, k):
    """(radii of the nodal circles, angles in [0, pi) of the nodal diameters): k - 1 circles and m diameters.
    The rim is held fixed and is not counted."""
    z = bessel_zeros(m, k)
    radii = z[:-1] / z[-1]
    angles = (2 * np.arange(m) + 1) * np.pi / (2 * m) if m > 0 else np.array([])
    return radii, angles


# ---------------------------------------------------------------------------------------------------------------
# Where the nodes of a hydrogen orbital are


def node_geometry(n, l, m):
    """Nodal surfaces of the real orbital (n, l, m): dict with
    'spheres'  radii of the n - l - 1 spherical shells (zeros of R_nl),
    'cones'    polar angles of the l - |m| cones (zeros of the Legendre function; pi/2 is the flat plane z = 0),
    'planes'   azimuth angles in [0, pi) of the |m| planes through the z axis.
    Together they are the n - 1 nodal surfaces."""
    rmax = 2.5 * n * n + 10
    r = np.linspace(1e-6, rmax, 400001)
    f = radial_R(n, l, r)
    idx = np.where(f[:-1] * f[1:] < 0)[0]
    spheres = np.array([r[i] - f[i] * (r[i + 1] - r[i]) / (f[i + 1] - f[i]) for i in idx])
    cones = latitude_nodes(l, m)
    am = abs(m)
    if am == 0:
        planes = np.array([])
    elif m > 0:
        planes = (2 * np.arange(am) + 1) * np.pi / (2 * am)   # cos(m phi) = 0
    else:
        planes = np.arange(am) * np.pi / am                   # sin(|m| phi) = 0
    return {"spheres": spheres, "cones": cones, "planes": planes}


# ---------------------------------------------------------------------------------------------------------------
# Fourier series on a circle, and the same idea on a sphere


def circle_function(phi):
    """A test shape on a circle: a square pulse centred at phi = pi (value 1 for |phi - pi| < 1, else 0)."""
    phi = np.asarray(phi, dtype=float)
    return (np.abs(phi - np.pi) < 1.0).astype(float)


def circle_coefficients(f, m_max, n=4096):
    """Real Fourier coefficients (a_m, b_m), m = 0 ... m_max, of f on the circle: the projections of f on cos(m phi)
    and sin(m phi). f(phi) is approximately a_0 + sum a_m cos(m phi) + b_m sin(m phi)."""
    phi = (np.arange(n) + 0.5) * 2 * np.pi / n
    y = f(phi)
    a = [np.mean(y)] + [2 * np.mean(y * np.cos(m * phi)) for m in range(1, m_max + 1)]
    b = [0.0] + [2 * np.mean(y * np.sin(m * phi)) for m in range(1, m_max + 1)]
    return np.array(a), np.array(b)


def circle_reconstruct(a, b, phi, m_max):
    """Sum of the first m_max harmonics."""
    phi = np.asarray(phi, dtype=float)
    out = a[0] * np.ones_like(phi)
    for m in range(1, m_max + 1):
        out = out + a[m] * np.cos(m * phi) + b[m] * np.sin(m * phi)
    return out


_BUMPS = [  # (polar angle, azimuth, sharpness, height): smooth bumps on the sphere, like a rough map of land
    (1.0, 0.8, 6.0, 1.0),
    (1.9, 3.6, 9.0, 0.8),
    (2.5, 1.9, 14.0, -0.6),
    (0.9, 4.9, 20.0, 0.7),
]


def sphere_function(theta, phi):
    """A test shape on the sphere: a sum of smooth bumps exp(s (cos(angle to centre) - 1))."""
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    out = np.zeros(np.broadcast(theta, phi).shape)
    for t0, p0, s, h in _BUMPS:
        cosang = np.cos(theta) * np.cos(t0) + np.sin(theta) * np.sin(t0) * np.cos(phi - p0)
        out = out + h * np.exp(s * (cosang - 1.0))
    return out


@lru_cache(maxsize=None)
def sphere_coefficients(l_max=20):
    """Coefficients a_lm = integral of f * (real Y_lm) over the sphere, for l <= l_max: dict {(l, m): a_lm}."""
    th, ph, w = sphere_quadrature(90, 180)
    f = sphere_function(th, ph)
    return {(l, m): float(np.sum(f * real_ylm(l, m, th, ph) * np.ones_like(th) * w))
            for l in range(l_max + 1) for m in range(-l, l + 1)}


def sphere_reconstruct(coeffs, l_max, theta, phi):
    """Sum of a_lm Y_lm for l <= l_max."""
    out = np.zeros(np.broadcast(np.asarray(theta), np.asarray(phi)).shape)
    for (l, m), a in coeffs.items():
        if l <= l_max:
            out = out + a * real_ylm(l, m, theta, phi)
    return out


def sphere_power(coeffs):
    """Sum of a_lm^2 over m for each l: how much of the shape lives at each 'frequency' l."""
    l_max = max(l for l, _ in coeffs)
    return np.array([sum(a * a for (l, m), a in coeffs.items() if l == ll) for ll in range(l_max + 1)])
