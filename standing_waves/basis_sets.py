"""Numerics for the basis-set notebook (`basis_sets_from_base_states.py`). Atomic units throughout.

Parts, in the order of the notebook:
  * a truncated expansion of the lowest states of a Morse well in oscillator functions (section 2),
  * the H2+ ion in a two-function LCAO basis of Slater 1s functions (section 3),
  * fits of the hydrogen 1s function by n Gaussians, and the product of two Gaussians (section 4),
  * shell counts, a split-valence illustration and the PySCF numbers read from JSON (section 5).
Everything here is plain numpy and is tested in `test_basis_sets.py`.
"""

import json
from functools import lru_cache
from math import factorial, pi, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DFT_DIR = HERE.parent / "dft"


def load_data():
    """The PySCF numbers of `compute_basis_sets.py` (data_basis_sets.json)."""
    return json.loads((HERE / "data_basis_sets.json").read_text())


def load_water():
    """Water, one row per def2 basis, from the DFT decoder notebook's data file (wB97M-V energies, shell counts)."""
    return json.loads((DFT_DIR / "data_basis.json").read_text())["basis"]


def trapz(y, x):
    """Trapezoid rule (numpy 2 removed np.trapz under that name in some builds, so it is written out)."""
    y = np.asarray(y)
    return float(np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1))


# ---------------------------------------------------------------------------------------------------------------
# Section 2: a Morse well, expanded in the functions of a harmonic oscillator

MORSE_D = 6.0  # well depth, hartree
MORSE_A = 0.5  # inverse width, 1/bohr
MORSE_LAMBDA = sqrt(2 * MORSE_D) / MORSE_A  # = sqrt(2 m D) / (a hbar) with m = hbar = 1
MORSE_OMEGA = MORSE_A * sqrt(2 * MORSE_D)  # small-vibration angular frequency at the bottom of the well
X_GRID = np.linspace(-8.0, 16.0, 4801)


def morse_potential(x):
    """V(x) = D (1 - exp(-a x))^2."""
    return MORSE_D * (1.0 - np.exp(-MORSE_A * np.asarray(x, dtype=float))) ** 2


def morse_energy(n):
    """Exact bound levels E_n = omega (n + 1/2) - omega^2 (n + 1/2)^2 / (4 D), for n < lambda - 1/2."""
    n = np.asarray(n, dtype=float)
    return MORSE_OMEGA * (n + 0.5) - MORSE_OMEGA**2 * (n + 0.5) ** 2 / (4 * MORSE_D)


def morse_bound_states():
    """Number of bound states, the whole numbers n with n < lambda - 1/2."""
    return int(np.ceil(MORSE_LAMBDA - 0.5))


def morse_psi0(x=None):
    """Exact ground state: psi_0 proportional to xi^(lambda - 1/2) exp(-xi/2), xi = 2 lambda exp(-a x).

    Normalised by a trapezoid sum on X_GRID, whatever x is, so that the integral of psi_0^2 is 1.
    """
    def raw(t):
        xi = 2 * MORSE_LAMBDA * np.exp(-MORSE_A * np.asarray(t, dtype=float))
        return np.exp((MORSE_LAMBDA - 0.5) * np.log(xi) - xi / 2)

    norm = sqrt(trapz(raw(X_GRID) ** 2, X_GRID))
    return raw(X_GRID if x is None else x) / norm


def ho_functions(n_max, x=None, omega=None):
    """Orthonormal oscillator functions phi_0 ... phi_(n_max-1) of frequency omega, as rows, by the three-term recursion."""
    x = X_GRID if x is None else np.asarray(x, dtype=float)
    omega = MORSE_OMEGA if omega is None else omega
    out = np.zeros((n_max, x.size))
    out[0] = (omega / pi) ** 0.25 * np.exp(-omega * x * x / 2)
    if n_max > 1:
        out[1] = sqrt(2 * omega) * x * out[0]
    for n in range(1, n_max - 1):
        out[n + 1] = sqrt(2 * omega / (n + 1)) * x * out[n] - sqrt(n / (n + 1)) * out[n - 1]
    return out


def oscillator_matrices(n_basis, omega=None):
    """Matrices H and S of the Morse Hamiltonian in the first n_basis oscillator functions.

    H = (kinetic + omega^2 x^2 / 2) + (V - omega^2 x^2 / 2). The first bracket is diagonal, omega (n + 1/2); the second is
    integrated on the grid. The functions are orthonormal, so S is the unit matrix (it is computed, not assumed).
    """
    omega = MORSE_OMEGA if omega is None else omega
    phi = ho_functions(n_basis, omega=omega)
    dv = morse_potential(X_GRID) - 0.5 * omega**2 * X_GRID**2
    h = np.array([[trapz(phi[i] * dv * phi[j], X_GRID) for j in range(n_basis)] for i in range(n_basis)])
    h += np.diag(omega * (np.arange(n_basis) + 0.5))
    s = np.array([[trapz(phi[i] * phi[j], X_GRID) for j in range(n_basis)] for i in range(n_basis)])
    return h, s


def monomial_matrices(n_basis, omega=None):
    """H and S in the non-orthogonal functions x^k exp(-omega x^2 / 2), k = 0 ... n_basis - 1 (same span as the oscillator set)."""
    omega = MORSE_OMEGA if omega is None else omega
    x = X_GRID
    g = np.exp(-omega * x * x / 2)
    f = np.array([x**k * g for k in range(n_basis)])
    df = np.array([(k * x ** (k - 1) if k else 0 * x) * g - omega * x ** (k + 1) * g for k in range(n_basis)])
    v = morse_potential(x)
    s = np.array([[trapz(f[i] * f[j], x) for j in range(n_basis)] for i in range(n_basis)])
    h = np.array([[trapz(0.5 * df[i] * df[j] + f[i] * v * f[j], x) for j in range(n_basis)] for i in range(n_basis)])
    return h, s


def solve_generalised(h, s, drop=1e-12):
    """Solve H c = E S c for symmetric H and positive S: S = U diag(w) U^T, X = U diag(w)^(-1/2), eigenproblem of X^T H X.

    Returns (energies ascending, C) with the coefficients as columns, normalised so that c^T S c = 1.
    """
    w, u = np.linalg.eigh(s)
    keep = w > drop * w.max()
    x = u[:, keep] / np.sqrt(w[keep])
    e, cp = np.linalg.eigh(x.T @ h @ x)
    return e, x @ cp


@lru_cache(maxsize=None)
def variational_levels(n_basis):
    """(energies, coefficient columns) of the Morse well in the first n_basis oscillator functions."""
    h, s = oscillator_matrices(n_basis)
    e, c = solve_generalised(h, s)
    for k in range(c.shape[1]):  # fix the arbitrary sign: the largest coefficient is positive
        if c[np.argmax(np.abs(c[:, k])), k] < 0:
            c[:, k] = -c[:, k]
    return e, c


def expansion_wavefunction(n_basis, state=0):
    """psi(x) = sum_n c_n phi_n(x) on X_GRID for the given state of the n_basis-function problem."""
    return variational_levels(n_basis)[1][:, state] @ ho_functions(n_basis)


def energy_errors(n_max, n_states=3):
    """Table E_k(N) - E_k(exact), N = 1 ... n_max (rows) and k = 0 ... n_states - 1 (columns); NaN where N <= k."""
    out = np.full((n_max, n_states), np.nan)
    for n in range(1, n_max + 1):
        e = variational_levels(n)[0]
        for k in range(min(n, n_states)):
            out[n - 1, k] = e[k] - morse_energy(k)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Section 3: H2+ with one Slater 1s function (zeta = 1) on each nucleus

def sto_overlap(r):
    """S(R) = <a|b> = exp(-R) (1 + R + R^2/3) for two 1s functions exp(-r)/sqrt(pi) a distance R apart."""
    r = np.asarray(r, dtype=float)
    return np.exp(-r) * (1 + r + r * r / 3)


def sto_coulomb(r):
    """J(R) = <a|1/r_b|a> = 1/R - exp(-2R) (1 + 1/R): attraction of a 1s cloud to the other nucleus."""
    r = np.asarray(r, dtype=float)
    return 1 / r - np.exp(-2 * r) * (1 + 1 / r)


def sto_exchange(r):
    """K(R) = <a|1/r_a|b> = <a|1/r_b|b> = exp(-R) (1 + R)."""
    r = np.asarray(r, dtype=float)
    return np.exp(-r) * (1 + r)


def h2plus_matrices(r):
    """2x2 matrices H and S (electron only, nuclei fixed) in the basis (1s on A, 1s on B)."""
    s_ab, j, k = float(sto_overlap(r)), float(sto_coulomb(r)), float(sto_exchange(r))
    h_aa = -0.5 - j
    h_ab = -0.5 * s_ab - k
    return np.array([[h_aa, h_ab], [h_ab, h_aa]]), np.array([[1.0, s_ab], [s_ab, 1.0]])


def h2plus_lcao(r):
    """Electronic energies and coefficients of the two LCAO states at distance r; total energy adds 1/R.

    By symmetry c = (1, +1)/sqrt(2 + 2S) (bonding) and (1, -1)/sqrt(2 - 2S) (antibonding).
    """
    h, s = h2plus_matrices(r)
    e, c = solve_generalised(h, s)
    return e, c


def h2plus_total(r, state=0):
    """Total energy (electron + nuclear repulsion 1/R) of the bonding (0) or antibonding (1) LCAO state."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.array([h2plus_lcao(x)[0][state] + 1 / x for x in r])
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------------------------------------------
# Section 4: Slater against Gaussian functions

R_GRID = np.linspace(0.0, 30.0, 12001)


def slater_1s(r, zeta=1.0):
    """Normalised Slater 1s function zeta^(3/2) exp(-zeta r) / sqrt(pi)."""
    return zeta**1.5 / sqrt(pi) * np.exp(-zeta * np.asarray(r, dtype=float))


def gauss_1s(r, alpha):
    """Normalised Gaussian s function (2 alpha / pi)^(3/4) exp(-alpha r^2); integral of |g|^2 over space is 1."""
    return (2 * alpha / pi) ** 0.75 * np.exp(-alpha * np.asarray(r, dtype=float) ** 2)


def gauss_overlap_matrix(alphas):
    """S_kl = <g_k|g_l> = (2 sqrt(a_k a_l) / (a_k + a_l))^(3/2) for normalised s Gaussians at one centre."""
    a = np.asarray(alphas, dtype=float)
    return (2 * np.sqrt(np.outer(a, a)) / (a[:, None] + a[None, :])) ** 1.5


def gauss_kinetic_matrix(alphas):
    """T_kl = 3 a_k a_l / (a_k + a_l) S_kl."""
    a = np.asarray(alphas, dtype=float)
    return 3 * np.outer(a, a) / (a[:, None] + a[None, :]) * gauss_overlap_matrix(a)


def gauss_attraction_matrix(alphas, z=1.0):
    """V_kl = -Z <g_k|1/r|g_l> = -Z 2 pi / (a_k + a_l) N_k N_l, N = (2a/pi)^(3/4)."""
    a = np.asarray(alphas, dtype=float)
    n = (2 * a / pi) ** 0.75
    return -z * 2 * pi / (a[:, None] + a[None, :]) * np.outer(n, n)


def slater_gauss_overlaps(alphas, zeta=1.0):
    """b_k = <slater 1s|g_k>, by the trapezoid rule on R_GRID (the integrand is 4 pi r^2 s(r) g(r))."""
    r = R_GRID
    s = slater_1s(r, zeta)
    return np.array([trapz(4 * pi * r * r * s * gauss_1s(r, a), r) for a in np.atleast_1d(alphas)])


def best_contraction(alphas, zeta=1.0):
    """For fixed exponents, the combination g = sum c_k g_k closest to the Slater function: c = S^(-1) b, scaled to <g|g> = 1.

    Returns (c, overlap squared). The overlap squared of the best normalised combination is b^T S^(-1) b.
    """
    s = gauss_overlap_matrix(alphas)
    b = slater_gauss_overlaps(alphas, zeta)
    c = np.linalg.solve(s, b)
    c = c / np.sqrt(c @ s @ c)
    return c, float((c @ b) ** 2)


def nelder_mead(f, x0, step=0.3, tol=1e-12, max_iter=4000):
    """Minimal Nelder-Mead simplex minimiser (scipy is not part of this environment)."""
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol and np.max(np.abs(pts[-1] - pts[0])) < 1e-9:
            break
        centre = np.mean(pts[:-1], axis=0)
        xr = centre + (centre - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centre + 2 * (centre - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centre + 0.5 * ((xr if fr < vals[-1] else pts[-1]) - centre)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


@lru_cache(maxsize=None)
def sto_ng_fit(n, zeta=1.0):
    """STO-nG: exponents (descending) and contraction coefficients of n Gaussians fitted to the Slater 1s function.

    The fit maximises the overlap with the Slater function (as Hehre, Stewart and Pople did). The exponents are found by
    Nelder-Mead on their logarithms from several starting ladders; the coefficients follow for each trial exponent set.
    Returns (alphas, coefficients, overlap squared); coefficients refer to normalised primitives and <g|g> = 1.
    """
    def loss(logs):
        a = np.exp(logs)
        n = a.size
        if np.any(a < 1e-3) or np.any(a > 1e4) or (n > 1 and np.min(np.abs(np.diff(np.sort(a)))) < 1e-6):
            return 1.0
        return 1.0 - best_contraction(a, zeta)[1]

    best = None
    for ratio in (2.5, 3.5, 5.0):
        centre = np.log(0.6 * zeta**2)
        start = centre + np.log(ratio) * (np.arange(n) - (n - 1) / 2)
        logs, val = nelder_mead(loss, start)
        logs, val = nelder_mead(loss, logs, step=0.05)  # restart to shake off a stalled simplex
        if best is None or val < best[1]:
            best = (logs, val)
    a = np.sort(np.exp(best[0]))[::-1]
    c, ov2 = best_contraction(a, zeta)
    return a, c, ov2


def contracted_radial(r, alphas, coeffs):
    """sum_k c_k g_k(r) for the contracted s function."""
    r = np.asarray(r, dtype=float)
    return sum(c * gauss_1s(r, a) for a, c in zip(alphas, coeffs))


def hydrogen_energy_of_contraction(alphas, coeffs):
    """<g|H|g> / <g|g> for the hydrogen atom, H = -nabla^2/2 - 1/r, with g = sum c_k g_k (analytic matrix elements)."""
    c = np.asarray(coeffs, dtype=float)
    h = gauss_kinetic_matrix(alphas) + gauss_attraction_matrix(alphas)
    s = gauss_overlap_matrix(alphas)
    return float(c @ h @ c / (c @ s @ c))


def best_energy_of_exponents(alphas):
    """Lowest hydrogen-atom energy obtainable from these fixed s Gaussians (coefficients free): lowest root of H c = E S c."""
    h = gauss_kinetic_matrix(alphas) + gauss_attraction_matrix(alphas)
    return float(solve_generalised(h, gauss_overlap_matrix(alphas))[0][0])


def gaussian_product(alpha, centre_a, beta, centre_b):
    """exp(-alpha |r-A|^2) exp(-beta |r-B|^2) = K exp(-(alpha+beta) |r-P|^2): returns (K, P, alpha+beta)."""
    a, b = np.asarray(centre_a, dtype=float), np.asarray(centre_b, dtype=float)
    p = (alpha * a + beta * b) / (alpha + beta)
    k = np.exp(-alpha * beta / (alpha + beta) * np.sum((a - b) ** 2))
    return float(k), p, alpha + beta


def cusp_ratio(psi, h=1e-5):
    """(d psi/dr) / psi at r = 0, one-sided finite difference. The Coulomb cusp condition requires -Z."""
    return float((psi(h) - psi(0.0)) / h / psi(0.0))


# ---------------------------------------------------------------------------------------------------------------
# Section 5: shells, zeta, polarization

SHELL_LETTERS = "spdfgh"


def shell_size(ell):
    """A shell of angular momentum ell holds 2 ell + 1 real harmonics: s 1, p 3, d 5, f 7, g 9."""
    return 2 * ell + 1


def function_count(shells):
    """Number of basis functions on one atom from shell counts such as {'s': 3, 'p': 2, 'd': 1}."""
    return sum(n * shell_size(SHELL_LETTERS.index(letter)) for letter, n in shells.items())


def molecule_function_count(composition):
    """composition: list of shell dictionaries, one per atom."""
    return sum(function_count(s) for s in composition)


def gaussian_shell(ell, alpha, x, z):
    """Cartesian-like shell function r^ell exp(-alpha r^2) Y_ell0(theta) in the xz plane (x, z arrays); Y is the real harmonic."""
    from hydrogen import real_ylm

    r = np.hypot(x, z)
    theta = np.arctan2(x, z)  # polar angle from the z axis
    return r**ell * np.exp(-alpha * r * r) * real_ylm(ell, 0, theta, 0.0)


def split_valence_errors(zetas, kind):
    """How well a fixed basis can follow a 1s function that has swollen or shrunk (exponent zeta in exp(-zeta r)).

    kind 'single': the one fixed STO-3G function of zeta = 1 (a single zeta, the weights are frozen).
    kind 'double': that function's three Gaussians split into a tight contracted pair and a wide single one,
                   whose weights the calculation may choose freely (a double zeta).
    kind 'triple': all three Gaussians free (three independent functions).
    Returns 1 - (best overlap with the target)^2, which is 0 for a perfect match.
    """
    a, c, _ = sto_ng_fit(3)
    out = []
    for z in zetas:
        b = slater_gauss_overlaps(a, z)
        s = gauss_overlap_matrix(a)
        t = 1.0  # target Slater function is normalised
        if kind == "single":
            out.append(1 - float((c @ b) ** 2 / (c @ s @ c)))
        elif kind == "double":
            u = np.array([[c[0], c[1], 0.0], [0.0, 0.0, 1.0]])  # tight pair (contracted), wide function
            sb = u @ s @ u.T
            bb = u @ b
            out.append(1 - float(bb @ np.linalg.solve(sb, bb)) / t)
        elif kind == "triple":
            out.append(1 - float(b @ np.linalg.solve(s, b)) / t)
        else:
            raise ValueError(kind)
    return np.array(out)


def density_slater(r, zeta=1.0):
    """Exact density of the hydrogen 1s electron, exp(-2 r) / pi."""
    return np.exp(-2 * zeta * np.asarray(r, dtype=float)) * zeta**3 / pi


def split_valence_coefficients(zeta, kind):
    """Coefficients on the three normalised primitives of the STO-3G fit for the best approximation to the 1s function of exponent zeta.

    Same three kinds as in split_valence_errors; the result is the orthogonal projection of the target onto the span.
    """
    a, c, _ = sto_ng_fit(3)
    b = slater_gauss_overlaps(a, zeta)
    s = gauss_overlap_matrix(a)
    if kind == "single":
        return c * float(c @ b) / float(c @ s @ c)
    if kind == "double":
        u = np.array([[c[0], c[1], 0.0], [0.0, 0.0, 1.0]])
        return u.T @ np.linalg.solve(u @ s @ u.T, u @ b)
    if kind == "triple":
        return np.linalg.solve(s, b)
    raise ValueError(kind)


def shell_string(shells):
    """{'s': 3, 'p': 2, 'd': 1} -> '3s2p1d' (in the order s, p, d, f, g)."""
    return "".join(f"{shells[letter]}{letter}" for letter in SHELL_LETTERS if letter in shells)
