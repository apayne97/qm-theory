"""Data loading and small analytic helpers for the DFT decoder notebook.

The numbers come from `compute_dft_data.py` (PySCF), which writes `data_*.json` next to this file; the notebook
itself needs no PySCF. The helpers below are plain functions so that they can be tested.
"""

import json
from math import erf, erfc
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HA2EV = 27.211386
HA2KCAL = 627.5095
EXPERIMENTAL_WATER_IP_EV = 12.62  # vertical first ionisation energy of water, about 12.6 eV


def load(part):
    """Contents of data_<part>.json (parts: scf, basis, ladder, dispersion)."""
    return json.loads((HERE / f"data_{part}.json").read_text())


def erf_split(r, omega):
    """Split 1/r into a short-range and a long-range part: 1/r = erfc(omega r)/r + erf(omega r)/r."""
    r = np.asarray(r, dtype=float)
    short = np.array([erfc(omega * x) for x in r]) / r
    long_ = np.array([erf(omega * x) for x in r]) / r
    return short, long_


def exact_exchange_fraction(r, omega, short_fraction, long_fraction):
    """Fraction of exact (Hartree-Fock) exchange used between two electrons a distance r apart.

    Exchange is split with the erf rule above; the short-range piece gets `short_fraction` of exact exchange
    and the long-range piece `long_fraction`. The weight of the short-range piece at distance r is erfc(omega r).
    """
    r = np.asarray(r, dtype=float)
    w_short = np.array([erfc(omega * x) for x in r])
    return short_fraction * w_short + long_fraction * (1 - w_short)


def levels_of_basis_function_count(shells):
    """Number of basis functions in a dict of shells such as {'s': 5, 'p': 3, 'd': 2, 'f': 1}.

    Real basis functions per shell: s 1, p 3, d 5, f 7 (spherical harmonics, 2l+1).
    """
    per_shell = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9}
    return sum(per_shell[letter] * count for letter, count in shells.items())


def shell_label(shells):
    """'[5s3p2d1f]' style label of a dict of shells."""
    return "[" + "".join(f"{shells[k]}{k}" for k in "spdfg" if k in shells) + "]"


def damped_dispersion(r, c6, r0):
    """Shape of a damped -C6/r^6 term: -C6 / (r^6 + R0^6) (Becke-Johnson style, finite at r = 0)."""
    r = np.asarray(r, dtype=float)
    return -c6 / (r**6 + r0**6)


def kcal(e_hartree):
    return e_hartree * HA2KCAL
