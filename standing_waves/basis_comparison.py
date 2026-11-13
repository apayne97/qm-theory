"""Numerics for section 3b of `basis_sets_from_base_states.py`: why atom-centred functions? Atomic units throughout.

One electron, the hydrogen atom (exact energy -1/2 hartree) and H2+ (exact -0.602634 hartree at R = 2 bohr), solved
variationally in two kinds of basis:

  * box standing waves: products cos(k_x x) cos(k_y y) cos(k_z z) with k = (2m+1) pi / L in a cube of side L, walls at
    +-L/2, nucleus (nuclei) inside. Only the functions even in x, y and z are used (the 1s state is even), so the count N
    is of the functions of the right parity, not of every standing wave of the cube. The functions are taken in order of
    their kinetic energy, which is a plane-wave "cutoff". The Coulomb attraction is evaluated through
        1/r = (2/sqrt(pi)) * integral_0^inf exp(-s^2 r^2) ds,
    a sum of Gaussians, so that every matrix element is a sum of products of one-dimensional integrals.
  * atom-centred s Gaussians with free exponents (all exponents optimised for the hydrogen atom).

Plain numpy; tested in `test_basis_sets.py`.
"""

from functools import lru_cache
from math import erf, pi, sqrt

import numpy as np

from basis_sets import nelder_mead, solve_generalised, sto_ng_fit

E_H_EXACT = -0.5
E_H2PLUS_EXACT = -0.6026342144949  # H2+ at R = 2 bohr (1.06 Å); the notebook's large-basis value agrees to 3e-5
R_H2PLUS = 2.0
M_BOX = 11  # one-dimensional functions per axis in the largest box basis: 11**3 = 1331 functions


# ----------------------------------------------------------------------------------------------------------------
# Box standing waves

def _coulomb_nodes(h=0.35, s_min=1e-6, s_max=1e4):
    """Trapezoid rule in u = ln s for 1/r = (2/sqrt(pi)) int exp(-s^2 r^2) ds. Returns exponents s_t and weights w_t."""
    u = np.arange(np.log(s_min), np.log(s_max) + h, h)
    s = np.exp(u)
    return s, 2 / sqrt(pi) * h * s


def coulomb_gaussian_sum(r):
    """The sum of Gaussians that stands in for 1/r (used by the box matrices; tested against 1/r)."""
    s, w = _coulomb_nodes()
    r = np.asarray(r, dtype=float)
    return (w * np.exp(-(s**2) * r[..., None] ** 2)).sum(axis=-1)


def _wavenumbers(box, m):
    return (2 * np.arange(m) + 1) * pi / box


def _overlap_1d(box, m, s, centre):
    """A_ab = integral over [-L/2, L/2] of phi_a phi_b exp(-s^2 (x-c)^2), phi_a = sqrt(2/L) cos(k_a x)."""
    k = _wavenumbers(box, m)
    if s * (box / 2 - abs(centre)) > 7.0:  # the walls cut off nothing: use the integral over the whole line
        km, kp = k[:, None] - k[None, :], k[:, None] + k[None, :]

        def line(q):
            return sqrt(pi) / s * np.exp(-(q**2) / (4 * s**2)) * np.cos(q * centre)

        return (line(km) + line(kp)) / box
    x, w = np.polynomial.legendre.leggauss(600)
    x, w = x * box / 2, w * box / 2
    phi = np.sqrt(2 / box) * np.cos(k[:, None] * x[None, :])
    return (phi * (w * np.exp(-(s**2) * (x - centre) ** 2))) @ phi.T


@lru_cache(maxsize=1)
def _box_order(m):
    """Index triples (i, j, k) sorted by n_x^2 + n_y^2 + n_z^2 with n = 2i+1, and those sums."""
    i, j, k = np.meshgrid(*[np.arange(m)] * 3, indexing="ij")
    nsq = ((2 * i + 1) ** 2 + (2 * j + 1) ** 2 + (2 * k + 1) ** 2).ravel()
    order = np.argsort(nsq, kind="stable")
    return order, nsq[order]


@lru_cache(maxsize=4)
def _box_hamiltonian(box, r_bond, m):
    """Full hamiltonian (m^3 x m^3, functions in kinetic order) of the electron in the box. r_bond=None: one proton at the
    centre. Otherwise two protons at z = +-r_bond/2 (the electronic part only; add 1/R separately)."""
    s_nodes, w_nodes = _coulomb_nodes()
    centres = [0.0] if r_bond is None else [r_bond / 2]  # the two protons give equal matrices for even functions
    v = np.zeros((m**3, m**3))
    for s, w in zip(s_nodes, w_nodes):
        ax = _overlap_1d(box, m, s, 0.0)
        for c in centres:
            az = _overlap_1d(box, m, s, c)
            v -= w * np.einsum("ab,cd,ef->acebdf", ax, ax, az).reshape(m**3, m**3)
    if r_bond is not None:
        v *= 2
    order, nsq = _box_order(m)
    k0 = (pi / box) ** 2 / 2
    h = v[np.ix_(order, order)]
    h[np.diag_indices_from(h)] += k0 * nsq
    return h


def box_function_counts(m=M_BOX, per_decade=3):
    """Basis sizes for the error curves: roughly log-spaced, ending with all m^3 functions."""
    n = np.unique(np.round(10 ** np.arange(0, np.log10(m**3), 1 / per_decade)).astype(int))
    return tuple(int(x) for x in np.append(n, m**3))


@lru_cache(maxsize=32)
def box_energies(box, ns, r_bond=None, m=M_BOX):
    """Lowest variational energy (hartree) from the first N box functions, for each N in ns. For H2+ (r_bond given)
    the proton repulsion 1/R is included."""
    h = _box_hamiltonian(box, r_bond, m)
    shift = 0.0 if r_bond is None else 1 / r_bond
    return np.array([np.linalg.eigvalsh(h[:n, :n])[0] + shift for n in ns])


def box_ground_state(box, n, r_bond=None, m=M_BOX):
    """Coefficients of the lowest state in the first n box functions, and the one-dimensional indices of those functions."""
    h = _box_hamiltonian(box, r_bond, m)
    c = np.linalg.eigh(h[:n, :n])[1][:, 0]
    order, _ = _box_order(m)
    idx = np.stack(np.unravel_index(order[:n], (m, m, m)), axis=1)
    return c, idx


def box_psi_on_axis(box, n, z, r_bond=None, m=M_BOX):
    """Ground-state wave along the z axis (through the nucleus, or both protons), y = x = 0, normalised over space."""
    c, idx = box_ground_state(box, n, r_bond, m)
    k = _wavenumbers(box, m)
    phi0 = np.sqrt(2 / box) * np.cos(k * 0.0)
    phiz = np.sqrt(2 / box) * np.cos(k[:, None] * np.asarray(z)[None, :])
    psi = (c * phi0[idx[:, 0]] * phi0[idx[:, 1]])[:, None] * phiz[idx[:, 2]]
    psi = psi.sum(axis=0)
    return psi if psi[np.argmin(np.abs(np.asarray(z)))] > 0 else -psi  # fix the overall sign


def box_projection_of_gaussian(box, alpha, m=M_BOX):
    """Coefficients (kinetic order) of the normalised s Gaussian exp(-alpha r^2) at the origin in the box functions,
    and the norm captured. Used to test the matrices."""
    k = _wavenumbers(box, m)
    x, w = np.polynomial.legendre.leggauss(800)
    x, w = x * box / 2, w * box / 2
    phi = np.sqrt(2 / box) * np.cos(k[:, None] * x[None, :])
    p = (phi * (w * np.exp(-alpha * x**2))) @ np.ones_like(x)  # per-axis integral of phi g
    full = np.einsum("a,b,c->abc", p, p, p).ravel() * (2 * alpha / pi) ** 0.75
    order, _ = _box_order(m)
    return full[order]


def box_count_below_cutoff(box, e_cut):
    """Number of even box functions with kinetic energy (k^2/2) at most e_cut, for a large m (counts the lattice points)."""
    mm = int(np.ceil(box * sqrt(2 * e_cut) / pi / 2)) + 2
    _, nsq = _box_order(mm)
    return int(np.sum((pi / box) ** 2 / 2 * nsq <= e_cut))


# ----------------------------------------------------------------------------------------------------------------
# Atom-centred s Gaussians

@lru_cache(maxsize=None)
def gaussian_exponents(n):
    """Exponents (descending) of n s Gaussians that minimise the hydrogen-atom energy; the coefficients are free.

    Nelder-Mead on the logarithms of the exponents, from the n-term STO-nG exponents (n <= 6) and from a geometric ladder.
    For n = 1 the optimum is 8/(9 pi) with energy -4/(3 pi).
    """
    from basis_sets import best_energy_of_exponents

    def loss(logs):
        a = np.exp(logs)
        if np.any(a < 1e-3) or np.any(a > 1e5) or (a.size > 1 and np.min(np.abs(np.diff(np.sort(logs)))) < 0.05):
            return 1.0
        return best_energy_of_exponents(a) + 0.5

    if n == 1:
        return np.array([8 / (9 * pi)])
    starts = [np.log(0.3) + np.log(r) * (np.arange(n) - 0.0) for r in (2.5, 3.5, 5.0)]
    if n <= 6:
        starts.append(np.log(sto_ng_fit(n)[0][::-1]))
    best = None
    for st in starts:
        x, val = nelder_mead(loss, st, step=0.2, max_iter=6000)
        x, val = nelder_mead(loss, x, step=0.05, max_iter=6000)
        if best is None or val < best[1]:
            best = (x, val)
    return np.sort(np.exp(best[0]))[::-1]


def gaussian_atom_energy(n):
    """Hydrogen-atom energy from n optimised s Gaussians."""
    from basis_sets import best_energy_of_exponents

    return best_energy_of_exponents(gaussian_exponents(n))


def gaussian_atom_state(n):
    """Exponents and coefficients (normalised primitives, <psi|psi>=1) of the ground state in n Gaussians."""
    from basis_sets import gauss_attraction_matrix, gauss_kinetic_matrix, gauss_overlap_matrix

    a = gaussian_exponents(n)
    h = gauss_kinetic_matrix(a) + gauss_attraction_matrix(a)
    s = gauss_overlap_matrix(a)
    _, c = solve_generalised(h, s)
    return a, c[:, 0]


def gaussian_psi(r, n):
    """Radial ground-state wave in n Gaussians (sign chosen positive at the nucleus)."""
    from basis_sets import gauss_1s

    a, c = gaussian_atom_state(n)
    psi = sum(ck * gauss_1s(r, ak) for ak, ck in zip(a, c))
    return psi if np.ravel(sum(ck * gauss_1s(0.0, ak) for ak, ck in zip(a, c)))[0] > 0 else -psi


def _boys0(t):
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    big = t > 1e-10
    out[big] = 0.5 * np.sqrt(pi / t[big]) * np.array([erf(sqrt(x)) for x in t[big]])
    return out


def gaussian_h2plus_energy_s_analytic(n, r_bond=R_H2PLUS):
    """H2+ total energy (with 1/R) from n hydrogen-atom s Gaussians on each proton, with the textbook closed-form (Boys function)
    integrals. Only s functions; the general code below must reproduce it."""
    a = gaussian_exponents(n)
    norm = (2 * a / pi) ** 0.75
    pos = np.array([-r_bond / 2, r_bond / 2])
    al = np.concatenate([a, a])
    cen = np.repeat(pos, n)
    nm = np.concatenate([norm, norm])
    p = al[:, None] + al[None, :]
    mu = al[:, None] * al[None, :] / p
    d2 = (cen[:, None] - cen[None, :]) ** 2
    kk = np.exp(-mu * d2)
    s = (pi / p) ** 1.5 * kk
    t = mu * (3 - 2 * mu * d2) * s
    pc = (al[:, None] * cen[:, None] + al[None, :] * cen[None, :]) / p
    v = np.zeros_like(s)
    for z in pos:
        v += -(2 * pi / p) * kk * _boys0(p * (pc - z) ** 2)
    nn = np.outer(nm, nm)
    h, s = (t + v) * nn, s * nn
    return float(solve_generalised(h, s)[0][0]) + 1 / r_bond


def _gh(n=24):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w


def _z_integral(a1, z1, l1, a2, z2, l2, q, c, deriv=False):
    """1D integrals over z of f1 f2 exp(-q (z-c)^2) for f = (z-Z)^l exp(-a (z-Z)^2) (or of f1' f2'), by Gauss-Hermite.

    All arguments broadcast; l is 0 or 1. The integrand is a polynomial times one Gaussian, so the quadrature is exact."""
    e = a1 + a2 + q
    z0 = (a1 * z1 + a2 * z2 + q * c) / e
    pref = np.exp(-(a1 * (z0 - z1) ** 2 + a2 * (z0 - z2) ** 2 + q * (z0 - c) ** 2))
    xi, wi = _gh()
    z = z0[..., None] + xi / np.sqrt(e)[..., None]

    def f(a, zc, l):
        d = z - zc[..., None]
        if not deriv:
            return d ** l[..., None] if np.ndim(l) else d**l
        return (l[..., None] * (d ** np.maximum(l[..., None] - 1, 0)) if np.ndim(l) else l * d ** max(l - 1, 0)) - 2 * a[..., None] * d ** (l[..., None] + 1)

    return pref * ((f(a1, z1, l1) * f(a2, z2, l2)) * wi).sum(axis=-1) / np.sqrt(e)


def _gaussian_electronic_matrices(centres, exps, powers, protons):
    """S, T and V (attraction to unit charges at z = protons) for functions (z-Z)^l exp(-a r_Z^2), l in {0, 1}, normalised."""
    z, a, l = (np.asarray(v, dtype=float) for v in (centres, exps, powers))
    a1, a2 = a[:, None], a[None, :]
    z1, z2 = z[:, None], z[None, :]
    l1, l2 = l[:, None], l[None, :]
    p = a1 + a2
    sxy = pi / p  # the x and y integrals together
    sz = _z_integral(a1, z1, l1, a2, z2, l2, 0.0, 0.0)
    s = sxy * sz
    dxy = 4 * a1 * a2 * np.sqrt(pi) / (2 * p**1.5) * np.sqrt(pi / p)  # d/dx d/dx in one of x, y, times the other's overlap
    dz = _z_integral(a1, z1, l1, a2, z2, l2, 0.0, 0.0, deriv=True)
    t = 0.5 * (2 * dxy * sz + sxy * dz)
    s_nodes, w_nodes = _coulomb_nodes()
    v = np.zeros_like(s)
    for st, wt in zip(s_nodes, w_nodes):
        xy = pi / (p + st**2)
        for c in protons:
            v -= wt * xy * _z_integral(a1, z1, l1, a2, z2, l2, st**2, c)
    nrm = 1 / np.sqrt(np.diag(s))
    outer = np.outer(nrm, nrm)
    return s * outer, t * outer, v * outer


def gaussian_h2plus_general(s_exps, p_exps=(), r_bond=R_H2PLUS):
    """H2+ total energy (with 1/R): s Gaussians with the exponents s_exps and p_z Gaussians with p_exps on each proton."""
    pos = (-r_bond / 2, r_bond / 2)
    cen, ex, pw = [], [], []
    for zc in pos:
        for e in s_exps:
            cen.append(zc), ex.append(e), pw.append(0)
        for e in p_exps:
            cen.append(zc), ex.append(e), pw.append(1)
    s, t, v = _gaussian_electronic_matrices(cen, ex, pw, pos)
    return float(solve_generalised(t + v, s)[0][0]) + 1 / r_bond


@lru_cache(maxsize=None)
def gaussian_p_exponent(n_s, r_bond=R_H2PLUS):
    """The single p_z exponent (on each proton) that lowers the H2+ energy most, with the s exponents of the hydrogen atom."""
    a = tuple(gaussian_exponents(n_s))
    x, _ = nelder_mead(lambda lg: gaussian_h2plus_general(a, (float(np.exp(lg[0])),), r_bond), [np.log(0.5)], step=0.5)
    return float(np.exp(x[0]))
