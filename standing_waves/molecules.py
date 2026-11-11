"""Numerics for the bridge notebook `from_atoms_to_molecules.py`. Atomic units (hartree, bohr) unless a name says otherwise.

Parts, in the order of the notebook:
  * counting coordinates: why a molecule has no closed-form solution (section 1),
  * Born-Oppenheimer: the electronic energy of H2+ as a potential for the nuclei, and the nuclear levels in it (section 2),
  * the orbital approximation: helium with one shared 1s orbital, and a screened potential that splits 2s from 2p (section 3),
  * hybrid orbitals s + sqrt(n) p and the angle between them (section 4),
  * a symmetry toy for methane: 8 functions, an a1 level and a triply degenerate t2 level (section 5).
Everything is plain numpy and is tested in `test_molecules.py`. The H2+ energies are read from `data_basis_sets.json`.
"""

from math import sqrt

import numpy as np

import basis_sets as bs

HARTREE_EV = 27.211386
HARTREE_CM = 219474.63  # one hartree in wavenumbers (cm^-1)
PROTON_MASS = 1836.15267343  # in electron masses

# ---------------------------------------------------------------------------------------------------------------
# Section 1: how many coordinates?


def coordinate_count(n_nuclei, n_electrons):
    """Three position numbers for every nucleus and every electron."""
    return 3 * (n_nuclei + n_electrons)


def grid_values(n_coordinates, points_per_axis=10):
    """Numbers needed to tabulate a wave function of n coordinates on a grid of `points_per_axis` points per axis."""
    return points_per_axis**n_coordinates


# ---------------------------------------------------------------------------------------------------------------
# Section 2: Born-Oppenheimer for H2+

REDUCED_MASS_H2PLUS = PROTON_MASS / 2  # two protons


def h2plus_points():
    """(R, E_total) of H2+ in a very large basis (PySCF, stands in for the exact electronic energy at fixed nuclei)."""
    d = bs.load_data()["h2plus"]
    return np.array(d["R"]), np.array(d["energy"])


def h2plus_fit(r_lo=1.25, r_hi=4.0, degree=6):
    """Polynomial through the computed points in [r_lo, r_hi]: a smooth E(R) near the minimum. Returns a numpy poly1d."""
    r, e = h2plus_points()
    keep = (r >= r_lo) & (r <= r_hi)
    return np.poly1d(np.polyfit(r[keep], e[keep], degree))


def h2plus_minimum(fit=None):
    """(R_e, E_min) of the fitted curve."""
    fit = fit if fit is not None else h2plus_fit()
    r = np.linspace(1.5, 3.0, 30001)
    i = int(np.argmin(fit(r)))
    return float(r[i]), float(fit(r[i]))


def harmonic_wavenumber(fit=None, mass=REDUCED_MASS_H2PLUS):
    """hbar omega / (hc) in cm^-1 for small vibrations: omega = sqrt(E''(R_e) / mass)."""
    fit = fit if fit is not None else h2plus_fit()
    r_e, _ = h2plus_minimum(fit)
    k = float(fit.deriv(2)(r_e))
    return sqrt(k / mass) * HARTREE_CM


def nuclear_levels(n_levels=4, fit=None, mass=REDUCED_MASS_H2PLUS, r_lo=1.25, r_hi=4.0, n_grid=2000):
    """Lowest vibrational levels (hartree, absolute) and functions of the nuclear problem -1/(2M) chi'' + E(R) chi = E chi.

    Finite differences on [r_lo, r_hi] with chi = 0 at both ends; chi is normalised (sum chi^2 dR = 1) with a positive first lobe.
    """
    fit = fit if fit is not None else h2plus_fit(r_lo, r_hi)
    r = np.linspace(r_lo, r_hi, n_grid + 2)[1:-1]
    h = r[1] - r[0]
    diag = 1 / (mass * h**2) + fit(r)
    off = -0.5 / (mass * h**2) * np.ones(n_grid - 1)
    e, v = np.linalg.eigh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    v = v[:, :n_levels] / np.sqrt(h)
    first = np.array([col[np.argmax(np.abs(col) > 1e-3 * np.abs(col).max())] for col in v.T])
    return r, e[:n_levels], v * np.where(first < 0, -1.0, 1.0)


def vibrational_spacing_cm(fit=None):
    """Spacing of the two lowest nuclear levels, in cm^-1."""
    _, e, _ = nuclear_levels(2, fit)
    return float(e[1] - e[0]) * HARTREE_CM


def timescale_ratio(fit=None):
    """Nuclear vibration quantum divided by the binding energy of the electron: a small number means 'electrons are fast'."""
    fit = fit if fit is not None else h2plus_fit()
    _, e_min = h2plus_minimum(fit)
    binding = -0.5 - e_min  # H + proton is at -1/2 hartree
    return float(vibrational_spacing_cm(fit) / HARTREE_CM / binding)


# ---------------------------------------------------------------------------------------------------------------
# Section 3: the orbital approximation

HELIUM_EXACT = -2.90372  # hartree, non-relativistic ground state (Pekeris; literature value, not computed here)
HELIUM_HF_LIMIT = -2.86168  # hartree, Hartree-Fock limit (literature value)


def helium_energy(zeta, z=2.0):
    """Both electrons in the same hydrogen-like 1s orbital exp(-zeta r), nucleus of charge z:
    kinetic zeta^2 + attraction -2 z zeta + electron repulsion 5 zeta / 8 (hartree)."""
    zeta = np.asarray(zeta, dtype=float)
    return zeta**2 - 2 * z * zeta + 5 * zeta / 8


def helium_best_zeta(z=2.0):
    """Minimum of helium_energy: zeta = z - 5/16 (the nucleus as seen through the other electron)."""
    return z - 5 / 16


def helium_best_energy(z=2.0):
    """-(z - 5/16)^2."""
    return -((z - 5 / 16) ** 2)


def screened_potential(r, z=3.0, decay=0.5):
    """Toy for an atom with a filled 1s shell: charge z seen close in, z - 2 seen far out (V = -Zeff(r)/r).

    Zeff(r) = (z - 2) + 2 exp(-r / decay) : the two core electrons screen the nucleus outside `decay` bohr.
    """
    r = np.asarray(r, dtype=float)
    return -((z - 2.0) + 2.0 * np.exp(-r / decay)) / r


def radial_levels(l, potential, h=0.02, n_grid=3000, n_states=4):
    """Lowest radial energies (hartree) of -u''/2 + [potential(r) + l(l+1)/(2 r^2)] u = E u, finite differences."""
    r = h * np.arange(1, n_grid + 1)
    diag = 1 / h**2 + potential(r) + l * (l + 1) / (2 * r * r)
    off = -0.5 / h**2 * np.ones(n_grid - 1)
    e = np.linalg.eigvalsh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    return e[:n_states]


def screened_levels(decay=0.5, z=3.0):
    """{(n, l): energy} for the lowest levels of the screened potential, n = l + 1 + (number of radial nodes)."""
    out = {}
    for l in range(3):
        e = radial_levels(l, lambda r: screened_potential(r, z, decay), n_states=4)
        for k, ek in enumerate(e):
            out[(l + 1 + k, l)] = float(ek)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Section 4: hybrid orbitals

def hybrid_angle_degrees(n):
    """Angle between two hybrids (s + sqrt(n) p along a direction): they are orthogonal when 1 + n cos(angle) = 0."""
    return float(np.degrees(np.arccos(-1.0 / n)))


def hybrid_profile(n, gamma):
    """Angular amplitude of the hybrid (s + sqrt(n) p_gamma) / sqrt(1 + n) at angle gamma from its axis, times sqrt(4 pi).

    The s function is 1 / sqrt(4 pi) and the p function along the axis is sqrt(3) cos(gamma) / sqrt(4 pi).
    """
    gamma = np.asarray(gamma, dtype=float)
    return (1.0 + sqrt(3.0 * n) * np.cos(gamma)) / sqrt(1.0 + n)


TETRAHEDRON = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / sqrt(3.0)


def sp3_coefficients():
    """4x4 matrix: row i gives the hybrid i in the basis (s, px, py, pz): (s + sqrt(3) n_i . p) / 2."""
    return 0.5 * np.hstack([np.ones((4, 1)), sqrt(3.0) * TETRAHEDRON])


def tetrahedral_angle_degrees():
    """Angle between two tetrahedron corner directions."""
    return float(np.degrees(np.arccos(TETRAHEDRON[0] @ TETRAHEDRON[1])))


# ---------------------------------------------------------------------------------------------------------------
# Section 5: methane, a symmetry toy
# Basis: C 2s, C 2px, 2py, 2pz, then H1s on each corner of the tetrahedron. The numbers below are illustrative
# (valence-orbital energies of the right size, a coupling of a few eV); the pattern of levels does not depend on them.

TOY_ENERGIES_EV = {"C2s": -19.4, "C2p": -10.6, "H1s": -13.6}
TOY_COUPLING_EV = -5.0  # <C s|H> and the size of <C p|H> for a p orbital pointing at that H

# Real calculation for comparison, read from ~/science/qm-calculations/methane/methane_output.txt
# (wB97M-V / def2-TZVPD, Hartree): occupied canonical orbital energies and the C-H localised-orbital composition.
METHANE_REAL = {
    "orbital_energies": [-10.3191, -0.8128, -0.4932, -0.4932, -0.4932],  # C 1s, a1, t2 (x3)
    "bond_orbital_on_carbon": {"s": 0.201, "p": 0.758, "d": 0.041},
    "a1_share_of_localised": 0.25,
}


def methane_toy_matrix(energies=None, coupling=TOY_COUPLING_EV):
    """8x8 Hamiltonian (eV), orthonormal basis (no overlap): diagonal valence energies, C-H coupling coupling x (1 | n_j)."""
    e = energies if energies is not None else TOY_ENERGIES_EV
    h = np.zeros((8, 8))
    h[0, 0] = e["C2s"]
    for i in (1, 2, 3):
        h[i, i] = e["C2p"]
    for j in range(4):
        h[4 + j, 4 + j] = e["H1s"]
        h[0, 4 + j] = h[4 + j, 0] = coupling
        for i in range(3):
            h[1 + i, 4 + j] = h[4 + j, 1 + i] = coupling * sqrt(3.0) * TETRAHEDRON[j, i]
    return h


def methane_toy_levels(**kw):
    """Eigenvalues (eV) and eigenvectors (columns) of the toy, ascending."""
    return np.linalg.eigh(methane_toy_matrix(**kw))


def methane_toy_closed_form(energies=None, coupling=TOY_COUPLING_EV):
    """(a1 pair, t2 pair): eigenvalues of the 2x2 blocks. a1: C s with the sum of the four H, coupling 2c;
    t2: C p_i with the H combination sum_j n_ji H_j / sqrt(4/3); its coupling is sqrt(3) c * sqrt(4/3) = 2c as well."""
    e = energies if energies is not None else TOY_ENERGIES_EV
    a1 = np.linalg.eigvalsh(np.array([[e["C2s"], 2 * coupling], [2 * coupling, e["H1s"]]]))
    t2 = np.linalg.eigvalsh(np.array([[e["C2p"], 2 * coupling], [2 * coupling, e["H1s"]]]))
    return a1, t2


def localised_bond_orbitals(vectors=None):
    """The four occupied toy orbitals (lowest four) and four equivalent bond orbitals made from them.

    Returns (occupied, bonds), 8x4 arrays. The lowest orbital is a1 (sign chosen so that its C 2s part is positive). The three
    degenerate t2 orbitals come in an arbitrary rotation, so t2_i is rebuilt as the projection of the C 2p_i function onto the
    occupied t2 space. Then bond_j = (a1 + sqrt(3) sum_i n_ji t2_i) / 2 : s character 1/4, p character 3/4.
    """
    if vectors is None:
        _, vectors = methane_toy_levels()
    occ = vectors[:, :4]
    a1 = occ[:, 0] * np.sign(occ[0, 0])
    t2_space = occ[:, 1:4]
    t2 = np.column_stack([t2_space @ t2_space[1 + i] for i in range(3)])
    t2 = t2 / np.linalg.norm(t2, axis=0)
    bonds = np.column_stack([(a1 + sqrt(3.0) * (t2 @ TETRAHEDRON[j])) / 2 for j in range(4)])
    return occ, bonds
