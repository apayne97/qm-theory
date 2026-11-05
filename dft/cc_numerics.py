"""A small, readable coupled-cluster toolkit in plain numpy (no PySCF), for `coupled_cluster.py`.

Everything works in the *spin-orbital* basis with antisymmetrised two-electron integrals

    g[p, q, r, s] = <pq||rs> = <pq|rs> - <pq|sr>,      <pq|rs> = (pr|qs)  (spin must match: p with r, q with s),

and a one-electron Hamiltonian h (kinetic energy plus attraction to the nuclei). Spin orbitals are ordered with the
occupied ones first, so the reference determinant occupies indices 0 .. n_occ-1. The labels i, j, k, l are occupied,
a, b, c, d are virtual (unoccupied), and p, q, r, s are either.

What is here:
  * `spin_orbital_system`   : build the spin-orbital arrays from restricted Hartree-Fock molecular-orbital integrals
  * `combine_noninteracting`: two fragments with no interaction, as one system (exactly the far-apart limit)
  * `mp2_energy`, `ccsd` (also CCD), `ccd_update_expanded`, `triples_correction`  : the perturbative and coupled-cluster energies
  * `ci_solve`              : configuration interaction in a space of determinants (full, or cut at an excitation level)
  * `fci_dimension`, `operation_counts` : counting
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

# --------------------------------------------------------------------------------------------------------------
# building the spin-orbital system
# --------------------------------------------------------------------------------------------------------------


def spin_orbital_system(h_mo, eri_mo, n_occ_spatial, e_nuc=0.0):
    """Spin-orbital arrays from spatial molecular-orbital integrals.

    h_mo   : (n, n) core Hamiltonian in the molecular orbitals.
    eri_mo : (n, n, n, n) integrals (pq|rs) in chemists' notation, for the same orbitals.
    Spin orbital 2p is the alpha spin of spatial orbital p and 2p+1 its beta spin, so the first 2*n_occ_spatial
    spin orbitals are the occupied ones.
    """
    n = h_mo.shape[0]
    n_so = 2 * n
    spatial = np.arange(n_so) // 2
    spin = np.arange(n_so) % 2
    h = h_mo[np.ix_(spatial, spatial)] * (spin[:, None] == spin[None, :])
    phys = eri_mo.transpose(0, 2, 1, 3)  # (pr|qs) stored as [p, q, r, s]
    same = (spin[:, None, None, None] == spin[None, None, :, None]) & (spin[None, :, None, None] == spin[None, None, None, :])
    pq_rs = phys[np.ix_(spatial, spatial, spatial, spatial)] * same
    g = pq_rs - pq_rs.transpose(0, 1, 3, 2)
    return _finish(h, g, 2 * n_occ_spatial, spin, e_nuc)


def _finish(h, g, n_occ, spin, e_nuc):
    f = h + np.einsum("pkqk->pq", g[:, :n_occ, :, :n_occ])
    e_elec = np.einsum("ii", h[:n_occ, :n_occ]) + 0.5 * np.einsum("ijij", g[:n_occ, :n_occ, :n_occ, :n_occ])
    return {
        "h": h, "g": g, "f": f, "n_occ": int(n_occ), "n_so": int(h.shape[0]), "spin": np.asarray(spin),
        "e_nuc": float(e_nuc), "e_hf": float(e_elec + e_nuc),
    }


def combine_noninteracting(a, b):
    """The two systems side by side with no integral coupling them (the limit of infinite separation).

    The orbitals are re-ordered so that all occupied orbitals (of a, then of b) come before all virtual ones.
    """
    na, nb = a["n_so"], b["n_so"]
    n = na + nb
    h = np.zeros((n, n))
    g = np.zeros((n, n, n, n))
    h[:na, :na], h[na:, na:] = a["h"], b["h"]
    g[:na, :na, :na, :na], g[na:, na:, na:, na:] = a["g"], b["g"]
    order = np.concatenate([
        np.arange(a["n_occ"]), na + np.arange(b["n_occ"]),
        np.arange(a["n_occ"], na), na + np.arange(b["n_occ"], nb),
    ])
    spin = np.concatenate([a["spin"], b["spin"]])[order]
    return _finish(h[np.ix_(order, order)], g[np.ix_(order, order, order, order)], a["n_occ"] + b["n_occ"], spin,
                   a["e_nuc"] + b["e_nuc"])


# --------------------------------------------------------------------------------------------------------------
# helpers for the antisymmetrisers
# --------------------------------------------------------------------------------------------------------------


def _p_ab(x):
    """P(ab) X_ijab = X_ijab - X_ijba."""
    return x - x.swapaxes(2, 3)


def _p_ij(x):
    """P(ij) X_ijab = X_ijab - X_jiab."""
    return x - x.swapaxes(0, 1)


def _blocks(sys_):
    no = sys_["n_occ"]
    o, v = slice(0, no), slice(no, sys_["n_so"])
    g, f = sys_["g"], sys_["f"]
    return o, v, g, f


def _denominators(sys_):
    o, v, _, f = _blocks(sys_)
    eps = np.diag(f)
    d1 = eps[o][:, None] - eps[v][None, :]
    d2 = eps[o][:, None, None, None] + eps[o][None, :, None, None] - eps[v][None, None, :, None] - eps[v][None, None, None, :]
    return d1, d2


# --------------------------------------------------------------------------------------------------------------
# MP2
# --------------------------------------------------------------------------------------------------------------


def mp2_amplitudes(sys_):
    o, v, g, _ = _blocks(sys_)
    return g[o, o, v, v] / _denominators(sys_)[1]


def mp2_energy(sys_):
    """Second-order perturbation correlation energy, (1/4) sum <ij||ab> t_ij^ab with the first-order t."""
    o, v, g, _ = _blocks(sys_)
    return 0.25 * np.einsum("ijab,ijab", g[o, o, v, v], mp2_amplitudes(sys_))


# --------------------------------------------------------------------------------------------------------------
# coupled cluster singles and doubles (and doubles only)
# --------------------------------------------------------------------------------------------------------------


def cc_energy(sys_, t1, t2):
    """E_corr = sum f_ia t_i^a + (1/4) sum <ij||ab> t_ij^ab + (1/2) sum <ij||ab> t_i^a t_j^b."""
    o, v, g, f = _blocks(sys_)
    return (np.einsum("ia,ia", f[o, v], t1) + 0.25 * np.einsum("ijab,ijab", g[o, o, v, v], t2)
            + 0.5 * np.einsum("ijab,ia,jb", g[o, o, v, v], t1, t1))


def ccsd_update(sys_, t1, t2, singles=True):
    """One Jacobi step: new amplitudes = (right-hand side evaluated with the old ones) / (orbital-energy difference).

    The right-hand sides are the CCSD equations written with the usual intermediates (the F and W arrays of the notebook).
    With singles=False the single-excitation amplitudes are kept at zero (that is CCD).
    """
    o, v, g, f = _blocks(sys_)
    es = np.einsum
    d1, d2 = _denominators(sys_)
    foo = f[o, o] - np.diag(np.diag(f)[o])  # off-diagonal Fock blocks (zero for canonical Hartree-Fock orbitals)
    fvv = f[v, v] - np.diag(np.diag(f)[v])
    fov = f[o, v]

    tt = es("ia,jb->ijab", t1, t1)
    tau_t = t2 + 0.5 * (tt - tt.swapaxes(2, 3))  # tau-tilde
    tau = t2 + tt - tt.swapaxes(2, 3)

    # intermediates
    f_ae = fvv - 0.5 * es("me,ma->ae", fov, t1) + es("mf,mafe->ae", t1, g[o, v, v, v]) - 0.5 * es("mnaf,mnef->ae", tau_t, g[o, o, v, v])
    f_mi = foo + 0.5 * es("me,ie->mi", fov, t1) + es("ne,mnie->mi", t1, g[o, o, o, v]) + 0.5 * es("inef,mnef->mi", tau_t, g[o, o, v, v])
    f_me = fov + es("nf,mnef->me", t1, g[o, o, v, v])
    x = es("je,mnie->mnij", t1, g[o, o, o, v])
    w_mnij = g[o, o, o, o] + (x - x.swapaxes(2, 3)) + 0.25 * es("ijef,mnef->mnij", tau, g[o, o, v, v])
    y = es("mb,amef->abef", t1, g[v, o, v, v])
    w_abef = g[v, v, v, v] - (y - y.swapaxes(0, 1)) + 0.25 * es("mnab,mnef->abef", tau, g[o, o, v, v])
    w_mbej = (g[o, v, v, o] + es("jf,mbef->mbej", t1, g[o, v, v, v]) - es("nb,mnej->mbej", t1, g[o, o, v, o])
              - es("jnfb,mnef->mbej", 0.5 * t2 + es("jf,nb->jnfb", t1, t1), g[o, o, v, v]))

    # singles
    r1 = (fov + es("ie,ae->ia", t1, f_ae) - es("ma,mi->ia", t1, f_mi) + es("imae,me->ia", t2, f_me)
          - es("nf,naif->ia", t1, g[o, v, o, v]) - 0.5 * es("imef,maef->ia", t2, g[o, v, v, v])
          - 0.5 * es("mnae,nmei->ia", t2, g[o, o, v, o]))

    # doubles
    f_be = f_ae - 0.5 * es("mb,me->be", t1, f_me)
    f_mj = f_mi + 0.5 * es("je,me->mj", t1, f_me)
    r2 = g[o, o, v, v].copy()
    r2 += _p_ab(es("ijae,be->ijab", t2, f_be))
    r2 -= _p_ij(es("imab,mj->ijab", t2, f_mj))
    r2 += 0.5 * es("mnab,mnij->ijab", tau, w_mnij) + 0.5 * es("ijef,abef->ijab", tau, w_abef)
    z = es("imae,mbej->ijab", t2, w_mbej) - es("ie,ma,mbej->ijab", t1, t1, g[o, v, v, o])
    r2 += _p_ij(_p_ab(z))
    r2 += _p_ij(es("ie,abej->ijab", t1, g[v, v, v, o]))
    r2 -= _p_ab(es("ma,mbij->ijab", t1, g[o, v, o, o]))

    new_t1 = r1 / d1 if singles else np.zeros_like(t1)
    return new_t1, r2 / d2


def ccsd(sys_, singles=True, max_iter=100, tol=1e-10, diis=True):
    """Solve the CCSD (or CCD) amplitude equations by iteration.

    Start from t1 = 0 and the first-order doubles (so the first energy is MP2), update with `ccsd_update`, and stop
    when the energy changes by less than `tol`. A short DIIS extrapolation (least-squares mixing of the last few
    amplitude sets) speeds up convergence without changing the answer.
    Returns a dict with e_corr, t1, t2, the energy history and the number of iterations.
    """
    o, v, _, _ = _blocks(sys_)
    t1 = np.zeros((sys_["n_occ"], sys_["n_so"] - sys_["n_occ"]))
    t2 = mp2_amplitudes(sys_)
    history = [cc_energy(sys_, t1, t2)]
    vecs, errs = [], []
    converged = False
    for it in range(1, max_iter + 1):
        n1, n2 = ccsd_update(sys_, t1, t2, singles)
        vecs.append(np.concatenate([n1.ravel(), n2.ravel()]))
        errs.append(vecs[-1] - np.concatenate([t1.ravel(), t2.ravel()]))
        if diis and len(vecs) >= 3:
            vecs, errs = vecs[-6:], errs[-6:]
            m = len(vecs)
            b = -np.ones((m + 1, m + 1))
            b[m, m] = 0.0
            for i in range(m):
                for j in range(m):
                    b[i, j] = errs[i] @ errs[j]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            c = np.linalg.solve(b, rhs)[:m]
            x = sum(ci * vi for ci, vi in zip(c, vecs))
            n1 = x[: n1.size].reshape(n1.shape)
            n2 = x[n1.size:].reshape(n2.shape)
        t1, t2 = n1, n2
        history.append(cc_energy(sys_, t1, t2))
        if abs(history[-1] - history[-2]) < tol and np.linalg.norm(errs[-1]) < 1e-7:
            converged = True
            break
    return {"e_corr": history[-1], "t1": t1, "t2": t2, "history": history, "iterations": it, "converged": converged}


def ccd_update_expanded(sys_, t2):
    """The same CCD step as `ccsd_update(..., singles=False)`, written as the fully expanded equation of the notebook.

    (e_i + e_j - e_a - e_b) t_ij^ab = <ij||ab>
        + 1/2 sum_cd <ab||cd> t_ij^cd + 1/2 sum_kl <kl||ij> t_kl^ab + P(ij)P(ab) sum_kc <kb||cj> t_ik^ac
        + 1/4 sum_klcd <kl||cd> t_ij^cd t_kl^ab
        - 1/2 P(ab) sum_klcd <kl||cd> t_ij^ac t_kl^bd
        - 1/2 P(ij) sum_klcd <kl||cd> t_ik^ab t_jl^cd
        - 1/2 P(ij)P(ab) sum_klcd <kl||cd> t_ik^ac t_jl^db
    """
    o, v, g, _ = _blocks(sys_)
    es = np.einsum
    goovv, gvvvv, goooo, govvo = g[o, o, v, v], g[v, v, v, v], g[o, o, o, o], g[o, v, v, o]
    r = goovv.copy()
    r += 0.5 * es("abcd,ijcd->ijab", gvvvv, t2) + 0.5 * es("klij,klab->ijab", goooo, t2)
    r += _p_ij(_p_ab(es("kbcj,ikac->ijab", govvo, t2)))
    r += 0.25 * es("klcd,ijcd,klab->ijab", goovv, t2, t2)
    r -= 0.5 * _p_ab(es("klcd,ijac,klbd->ijab", goovv, t2, t2))
    r -= 0.5 * _p_ij(es("klcd,ikab,jlcd->ijab", goovv, t2, t2))
    r -= 0.5 * _p_ij(_p_ab(es("klcd,ikac,jldb->ijab", goovv, t2, t2)))
    return r / _denominators(sys_)[1]


# --------------------------------------------------------------------------------------------------------------
# the perturbative triples correction (T)
# --------------------------------------------------------------------------------------------------------------


def _p_i_jk(x):
    """P(i/jk) X_ijk = X_ijk - X_jik - X_kji (index order i, j, k first)."""
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)


def _p_a_bc(x):
    """P(a/bc) X^abc = X^abc - X^bac - X^cba (virtual indices are the last three axes)."""
    return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)


def triples_correction(sys_, t1, t2):
    """The (T) energy, from converged CCSD amplitudes.

    D t_ijk^abc(connected)    = P(i/jk) P(a/bc) [ sum_d t_jk^ad <bc||di> - sum_l t_il^bc <la||jk> ]
    D t_ijk^abc(disconnected) = P(i/jk) P(a/bc) t_i^a <jk||bc>
    E(T) = (1/36) sum_ijkabc  t(connected) D [ t(connected) + t(disconnected) ]
    with D = e_i + e_j + e_k - e_a - e_b - e_c. Used once, after the CCSD amplitudes have converged.
    """
    o, v, g, f = _blocks(sys_)
    es = np.einsum
    eps = np.diag(f)
    no, nv = sys_["n_occ"], sys_["n_so"] - sys_["n_occ"]
    # raw[i,j,k,a,b,c]
    x = es("jkad,bcdi->ijkabc", t2, g[v, v, v, o]) - es("ilbc,lajk->ijkabc", t2, g[o, v, o, o])
    conn = _p_i_jk(_p_a_bc(x))
    y = es("ia,jkbc->ijkabc", t1, g[o, o, v, v])
    disc = _p_i_jk(_p_a_bc(y))
    eo, ev = eps[o], eps[v]
    d3 = (eo[:, None, None, None, None, None] + eo[None, :, None, None, None, None] + eo[None, None, :, None, None, None]
          - ev[None, None, None, :, None, None] - ev[None, None, None, None, :, None] - ev[None, None, None, None, None, :])
    t_conn = conn / d3
    return float(np.sum(t_conn * (conn + disc)) / 36.0)


# --------------------------------------------------------------------------------------------------------------
# configuration interaction in a space of determinants
# --------------------------------------------------------------------------------------------------------------


def _popcount(x):
    x = np.asarray(x, dtype=np.uint64)
    return np.bitwise_count(x).astype(int)


def determinants(sys_, n_electrons=None, level=None, sz_zero=True, max_dim=60000):
    """Bit-string determinants (integers whose set bits are the occupied spin orbitals) as a numpy array.

    level : keep only determinants that differ from the reference (the n_occ lowest spin orbitals occupied) by at
            most this many electrons moved (1 = singles, 2 = doubles, ...). None keeps every determinant (full CI).
    sz_zero : keep only those with as many alpha as beta electrons.
    """
    n = sys_["n_so"]
    ne = sys_["n_occ"] if n_electrons is None else n_electrons
    spin = sys_["spin"]
    ref = (1 << ne) - 1
    if comb(n, ne) > 5_000_000:
        raise ValueError("full determinant list too large for this toy code")
    out = []
    for occ in combinations(range(n), ne):
        if sz_zero and sum(spin[list(occ)]) * 2 != ne:
            continue
        d = sum(1 << p for p in occ)
        if level is not None and bin(d ^ ref).count("1") // 2 > level:
            continue
        out.append(d)
    if len(out) > max_dim:
        raise ValueError(f"{len(out)} determinants is too many for dense diagonalisation")
    return np.array(out, dtype=np.int64)


def _bits(d):
    return [p for p in range(d.bit_length()) if (d >> p) & 1]


def _phase_move(d, p, r):
    """Sign for moving an electron from spin orbital p to r in determinant d: (-1)^(number of occupied orbitals between)."""
    lo, hi = (p, r) if p < r else (r, p)
    between = (d >> (lo + 1)) & ((1 << (hi - lo - 1)) - 1)
    return -1.0 if bin(between).count("1") % 2 else 1.0


def ci_hamiltonian(sys_, dets):
    """Dense electronic CI Hamiltonian between determinants by the Slater-Condon rules (nuclear repulsion not included)."""
    h, g = sys_["h"], sys_["g"]
    n = len(dets)
    mat = np.zeros((n, n))
    pc = _popcount(dets[:, None] ^ dets[None, :])
    for a in range(n):
        da = int(dets[a])
        occ_a = _bits(da)
        mat[a, a] = sum(h[i, i] for i in occ_a) + 0.5 * sum(g[i, j, i, j] for i in occ_a for j in occ_a)
        for b in np.nonzero((pc[a] == 2) | (pc[a] == 4))[0]:
            if b <= a:
                continue
            db = int(dets[b])
            diff = da ^ db
            holes = _bits(da & diff)   # in a, not in b
            parts = _bits(db & diff)   # in b, not in a
            if len(holes) == 1:
                p, r = holes[0], parts[0]
                common = _bits(da & db)
                val = h[r, p] + sum(g[r, k, p, k] for k in common)
                val *= _phase_move(da, p, r)
            else:
                p, q = holes
                r, s = parts
                ph1 = _phase_move(da, p, r)
                d1 = (da & ~(1 << p)) | (1 << r)
                ph2 = _phase_move(d1, q, s)
                val = ph1 * ph2 * g[r, s, p, q]
            mat[a, b] = mat[b, a] = val
    return mat


def ci_solve(sys_, level=None, n_electrons=None):
    """Lowest CI state. Returns dict(e_corr, e_total, vector, dets, reference_index).

    level=None is full CI (all determinants of the right spin); level=2 is CISD.
    """
    dets = determinants(sys_, n_electrons, level)
    mat = ci_hamiltonian(sys_, dets)
    w, vec = np.linalg.eigh(mat)
    e_elec = w[0]
    ref = (1 << (sys_["n_occ"] if n_electrons is None else n_electrons)) - 1
    ref_idx = int(np.nonzero(dets == ref)[0][0])
    return {"e_total": e_elec + sys_["e_nuc"], "e_corr": e_elec + sys_["e_nuc"] - sys_["e_hf"],
            "vector": vec[:, 0], "dets": dets, "reference_index": ref_idx}


def excitation_level_weights(result, n_electrons):
    """Weight (squared coefficient) of the singles, doubles, triples ... in a CI vector, as a dict level -> weight."""
    ref = (1 << n_electrons) - 1
    weights = {}
    for d, c in zip(result["dets"], result["vector"]):
        lev = bin(int(d) ^ ref).count("1") // 2
        weights[lev] = weights.get(lev, 0.0) + float(c) ** 2
    return weights


def coefficient(result, occupied_to_virtual):
    """CI coefficient relative to the reference, for the determinant with spin orbitals `occupied_to_virtual` swapped.

    occupied_to_virtual : list of (i, a) pairs, electron moved from occupied i to virtual a. Sign follows the
    bit-string ordering, so compare magnitudes only.
    """
    ref_idx = result["reference_index"]
    d = int(result["dets"][ref_idx])
    for i, a in occupied_to_virtual:
        d = (d & ~(1 << i)) | (1 << a)
    idx = np.nonzero(result["dets"] == d)[0]
    return float(result["vector"][idx[0]] / result["vector"][ref_idx])


# --------------------------------------------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------------------------------------------


def fci_dimension(n_orbitals, n_alpha, n_beta):
    """Number of determinants in full CI: C(n, n_alpha) * C(n, n_beta)."""
    return comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)


def operation_counts(n_occ, n_virt):
    """Leading-term operation counts (up to a constant factor) per method, for n_occ occupied and n_virt virtual orbitals.

    mp2   : o N^4 with N = o + v, the integral transformation that dominates MP2 (the correlation step itself is o^2 v^2)
    ccsd  : o^2 v^4 per iteration (the particle-particle ladder term)
    ccsd(t): o^3 v^4, once (the (T) correction)
    """
    o, v = n_occ, n_virt
    return {"mp2": o * (o + v)**4, "ccsd": o**2 * v**4, "ccsd(t)": o**3 * v**4}


# --------------------------------------------------------------------------------------------------------------
# loading the stored data
# --------------------------------------------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402
import json as _json  # noqa: E402

_HERE = _Path(__file__).parent


def load_cc_json():
    """Contents of data_cc.json (PySCF energies, counts and timings; written by compute_cc_data.py)."""
    return _json.loads((_HERE / "data_cc.json").read_text())


def load_system(name):
    """Spin-orbital system for a stored molecule: 'h2_631g', 'water_sto3g' or 'water_631g'."""
    npz = np.load(_HERE / "data_cc_integrals.npz")
    return spin_orbital_system(npz[f"{name}__h"], npz[f"{name}__eri"], int(npz[f"{name}__n_occ"]),
                               float(npz[f"{name}__e_nuc"]))
