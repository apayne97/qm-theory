"""Standing waves of a particle in a rectangular box in one, two and three dimensions.

Units: hbar = m = 1 unless stated. A box has side lengths `Ls` (a tuple), the wave function vanishes on the walls,
and the allowed states are labelled by one whole number n >= 1 per direction.

    psi_n(x) = prod_i sqrt(2/L_i) sin(n_i pi x_i / L_i),     E_n = (pi^2 / 2) sum_i (n_i / L_i)^2.

`energy_units` measures E in units of the one-dimensional ground-state energy of a box of length Ls[0], so that
E = sum_i (n_i Ls[0] / L_i)^2 and the numbers are small whole numbers for a square or cube.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantization"))
from quantization import box_energy_ev  # noqa: E402,F401  (re-exported: E = n^2 pi^2 hbar^2 / 2 m L^2 in eV)


def box_wave(ns, points, Ls):
    """Normalised standing wave of the box. `points` is a sequence of arrays, one per direction (broadcastable)."""
    psi = 1.0
    for n, x, L in zip(ns, points, Ls):
        psi = psi * np.sqrt(2.0 / L) * np.sin(n * np.pi * np.asarray(x, dtype=float) / L)
    return psi


def energy_units(ns, Ls):
    """Energy in units of the 1D ground-state energy of a box of length Ls[0]."""
    return sum((n * Ls[0] / L) ** 2 for n, L in zip(ns, Ls))


def energy(ns, Ls):
    """Energy with hbar = m = 1: (pi^2/2) sum (n_i / L_i)^2."""
    return 0.5 * np.pi**2 * sum((n / L) ** 2 for n, L in zip(ns, Ls))


def nodes_per_direction(ns):
    """A standing wave with n half-wavelengths across a side has n - 1 nodes inside it."""
    return tuple(n - 1 for n in ns)


def levels(Ls, e_max, tol=1e-9):
    """All levels with energy_units <= e_max, as a list of (energy, [states]) sorted by energy.

    Two states belong to the same level when their energies agree to `tol`.
    """
    ranges = [range(1, int(np.floor(np.sqrt(e_max) * L / Ls[0])) + 1) for L in Ls]
    states = [(energy_units(ns, Ls), ns) for ns in product(*ranges)]
    states = sorted((e, ns) for e, ns in states if e <= e_max + tol)
    out = []
    for e, ns in states:
        if out and abs(out[-1][0] - e) <= tol * max(1.0, e):
            out[-1][1].append(ns)
        else:
            out.append((e, [ns]))
    return out


def degeneracies(Ls, e_max):
    """List of (energy, number of states) for the levels up to e_max."""
    return [(e, len(s)) for e, s in levels(Ls, e_max)]


def count_states_below(Ls, e):
    """Number of box states with energy_units <= e."""
    return sum(len(s) for _, s in levels(Ls, e))


def smooth_count(dim, e):
    """Large-E estimate of the number of states below e in a cube/square/line: the part of a ball of radius sqrt(e)
    in the positive 'corner' (all n_i > 0): sqrt(e), pi e / 4, pi e^(3/2) / 6."""
    r = np.sqrt(e)
    return {1: r, 2: np.pi * r**2 / 4, 3: np.pi * r**3 / 6}[dim]
