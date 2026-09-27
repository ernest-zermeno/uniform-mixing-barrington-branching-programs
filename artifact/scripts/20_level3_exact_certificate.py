#!/usr/bin/env python3
"""
20_level3_exact_certificate.py
==============================

EXACT RATIONAL CERTIFICATE for the level-3 contraction theorem (thm:level23):

    ||Dhat_3||_op <= lambda := 71/2000 = 0.0355

for EVERY valid level-3 configuration of the fixed S_5 transposition template
and EVERY non-trivial, non-sign irreducible representation of S_5, i.e. for
all 9,447,840 oriented pairs of ordered level-2 blocks (= 4,723,920 unordered
pairs, Scripts 17/18).  This replaces the float64 verification of Scripts
17/18 by exact integer arithmetic: ZERO floating-point operations occur in
the certification chain (floats appear only in clearly marked sanity checks).

METHOD
------
1.  ORBIT REDUCTION WITH EXACT ACCOUNTING.  The symmetry group
    H = S_5 x Z_2 (simultaneous conjugation of all eight transpositions,
    and the swap of the two level-2 blocks) preserves ||Dhat_3||_op:
      * conjugation by s:  Dhat_3(s.cfg) = rho(s) Dhat_3(cfg) rho(s)^{-1}
        with rho(s) G-orthogonal (similarity by a G-isometry), because
        P_{s t s^{-1}} = rho(s) P_t rho(s)^{-1};
      * swap:  Dhat_3(j,i) = Dhat_3(i,j)^*  (the G-adjoint), and a Hilbert
        space adjoint has the same operator norm.
    The 9,447,840 oriented pairs decompose into 39,591 H-orbits.  The sum of
    the orbit sizes is checked to reconstruct EXACTLY 9,447,840.

2.  RATIONAL CARRIERS.  Instead of irrational orthonormal bases, each irrep
    is certified inside an INTEGER permutation module carrying an invariant
    integer Gram matrix G:
      * M41 (dim 4):  sum-zero subspace of the natural 5-point permutation
        module, basis b_i = e_i - e_4, G = I + J.        Contains (4,1).
      * M9  (dim 9):  sum-zero subspace of the permutation module on the 10
        2-subsets of {0..4}, G = I + J.       Contains (4,1) + (3,2).
      * M9s (dim 9):  M9 twisted by sign (P'_t = (I - rho(t))/2, since
        sgn(t) = -1 for a transposition).  Contains (2,1,1,1) + (2,2,1).
      * M6  (dim 6):  wedge^2 of M41, G_wedge = wedge^2(G).  Equals (3,1,1).
    Dhat_3 is the image of a fixed element of the group algebra Q[S_5]
    (a word in the projections P_t = (e + t)/2 and their adjoints, and each
    P_t is G-self-adjoint so adjoints are reversed words), hence it preserves
    every isotypic component of a module.  Distinct isotypic components are
    G-orthogonal, so the G-operator norm on a module is the MAX of the norms
    on its irreducible constituents; by Schur's lemma every invariant inner
    product on an irrep is proportional, so those norms are basis-independent.
    Certifying M41, M9, M9s, M6 therefore certifies all five non-trivial,
    non-sign irreps (4,1), (3,2), (3,1,1), (2,2,1), (2,1,1,1).

3.  INTEGER GRAM-FORM CERTIFICATE (pattern of P2A s4_internal_gap_certify.py).
    In every module all P_t have entries in (1/2)Z, so
      D1 = (P_a P_b)^2            has denominator 2^4,
      D2 = D1_A D1_B D1_A* D1_B*  has denominator 2^16   (X* = reversed word),
      D3 = D2_i D2_j D2_i* D2_j*  has denominator 2^64.
    Writing D3 = N/2^64 with N an INTEGER matrix,
      ||D3||_G <= lambda   <=>   lambda^2 G - D3^T G D3  is PSD
                           <=>   C := 5041 * 2^128 * G - 4,000,000 * N^T G N
                                 is PSD  (an INTEGER matrix).
    C is certified POSITIVE DEFINITE via its leading principal minors
    (Sylvester), computed exactly by fraction-free Bareiss elimination.
    (A PSD fallback over all principal minors exists; it is never needed.)

4.  TIGHT BRACKET FOR THE MAXIMUM.  A second run on all four modules certifies
    the tight bound  ||Dhat_3|| <= 354002321/10^10 = 0.0354002321  for ALL
    configurations and all five non-trivial, non-sign irreps.  The bottleneck
    configuration's largest squared singular value is enclosed rigorously
    (integer characteristic polynomial det(y G - N^T G N) + exact rational
    bisection), giving the exact global bracket

        global max ||Dhat_3||  in  [0.035400232083798...,  0.0354002321]

    consistent with the float64 value 0.03540023208379892 of Scripts 17/18.

RUN
---
    python3 20_level3_exact_certificate.py          (~1-2 min, pure Python;
                                                     numpy used only for the
                                                     float sanity section)
"""

from __future__ import annotations

import random
import time
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np

T_START = time.time()

LAM_NUM, LAM_DEN = 71, 2000                  # lambda = 71/2000 = 0.0355
TIGHT_NUM, TIGHT_DEN = 354002321, 10**10     # tight all-irrep bound = 0.0354002321
FLOAT64_MAX_SCRIPT18 = 0.03540023208379892   # Scripts 17/18 reference value


def log(msg: str) -> None:
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


# ======================================================================
# PART 0: permutation utilities (conventions of Scripts 10/17/18)
# ======================================================================

ID = tuple(range(5))


def pmul(a, b):
    return tuple(a[b[i]] for i in range(5))


def pinv(p):
    r = [0] * 5
    for i, j in enumerate(p):
        r[j] = i
    return tuple(r)


def trans(a, b):
    p = list(range(5))
    p[a], p[b] = p[b], p[a]
    return tuple(p)


def comm(p, q):
    return pmul(pmul(p, q), pmul(pinv(p), pinv(q)))


def sgn(p):
    s = 1
    for i in range(5):
        for j in range(i + 1, 5):
            if p[i] > p[j]:
                s = -s
    return s


ALL_TRANS = [(a, b) for a in range(5) for b in range(a + 1, 5)]
S5 = [tuple(p) for p in permutations(range(5))]


def conj_trans(s, t):
    a, b = s[t[0]], s[t[1]]
    return (a, b) if a < b else (b, a)


# ======================================================================
# PART 1: level-3 structure (identical to Scripts 17/18) + orbit reduction
# ======================================================================

log("PART 1: structure and orbit reduction")

# --- 60 ordered adjacent level-1 pairs ---
adjacent = []
for t1 in ALL_TRANS:
    for t2 in ALL_TRANS:
        if t1 == t2 or len(set(t1) & set(t2)) != 1:
            continue
        if comm(trans(*t1), trans(*t2)) == ID:
            continue
        adjacent.append((t1, t2))
assert len(adjacent) == 60, len(adjacent)
adj_idx = {p: k for k, p in enumerate(adjacent)}
adj_rev = [adj_idx[(p[1], p[0])] for p in adjacent]
tgt1 = [comm(trans(*p[0]), trans(*p[1])) for p in adjacent]

# --- 3240 ordered valid level-2 blocks ---
blocks = []
tgt2 = []
for ia in range(60):
    for ib in range(60):
        if ia == ib:
            continue
        t = comm(tgt1[ia], tgt1[ib])
        if t == ID:
            continue
        blocks.append((ia, ib))
        tgt2.append(t)
assert len(blocks) == 3240, len(blocks)
NB = 3240
blk_idx = {b: k for k, b in enumerate(blocks)}
blk_rev = [blk_idx[(b[1], b[0])] for b in blocks]   # adjoint = reversed block

# --- conjugation action of S_5 on the 3240 blocks ---
tidx = {t: k for k, t in enumerate(ALL_TRANS)}
blk_perm = {}
for s in S5:
    tmap = [conj_trans(s, t) for t in ALL_TRANS]
    amap = [adj_idx[(tmap[tidx[p[0]]], tmap[tidx[p[1]]])] for p in adjacent]
    blk_perm[s] = [blk_idx[(amap[b[0]], amap[b[1]])] for b in blocks]

# exact sanity: the action is compatible with conjugation of targets
rng = random.Random(1)
for _ in range(50):
    s = rng.choice(S5)
    k = rng.randrange(NB)
    assert tgt2[blk_perm[s][k]] == pmul(pmul(s, tgt2[k]), pinv(s))

# --- orbits of S_5 on blocks ---
GENS = [trans(0, 1), (1, 2, 3, 4, 0)]
orbit_id = [-1] * NB
orbit_reps = []
for start in range(NB):
    if orbit_id[start] != -1:
        continue
    oid = len(orbit_reps)
    orbit_reps.append(start)
    orbit_id[start] = oid
    stack = [start]
    while stack:
        x = stack.pop()
        for g in GENS:
            y = blk_perm[g][x]
            if orbit_id[y] == -1:
                orbit_id[y] = oid
                stack.append(y)
orbit_size = [0] * len(orbit_reps)
for k in range(NB):
    orbit_size[orbit_id[k]] += 1
stabs = {r: [s for s in S5 if blk_perm[s][r] == r] for r in orbit_reps}
for r in orbit_reps:
    assert len(stabs[r]) * orbit_size[orbit_id[r]] == 120
log(f"  block orbits under S_5: {len(orbit_reps)} "
    f"(sizes {sorted(set(orbit_size))}, stabilizer orders "
    f"{sorted(set(len(stabs[r]) for r in orbit_reps))})")

# --- exact count of valid oriented level-3 pairs via degrees ---
deg = {}
total_oriented = 0
for r in orbit_reps:
    ti = tgt2[r]
    d = sum(1 for j in range(NB) if j != r and comm(ti, tgt2[j]) != ID)
    deg[r] = d
    total_oriented += orbit_size[orbit_id[r]] * d
assert total_oriented == 9_447_840, total_oriented
log(f"  valid oriented level-3 pairs (degree count over orbits): {total_oriented}")

# --- S_5-orbits on ordered pairs: Stab(i0)-orbits on the valid j's ---
pair_orbits = []          # (i0, j0, S5-orbit size)
acc = 0
for r in orbit_reps:
    st = stabs[r]
    m0 = orbit_size[orbit_id[r]]
    ti = tgt2[r]
    seen = set()
    for j in range(NB):
        if j in seen or j == r or comm(ti, tgt2[j]) == ID:
            continue
        orb = {j}
        stack = [j]
        while stack:
            x = stack.pop()
            for s in st:
                y = blk_perm[s][x]
                if y not in orb:
                    orb.add(y)
                    stack.append(y)
        seen |= orb
        size = m0 * len(orb)
        pair_orbits.append((r, min(orb), size))
        acc += size
assert acc == 9_447_840, acc
log(f"  S_5-orbits on ordered pairs: {len(pair_orbits)}; "
    f"EXACT ACCOUNTING: sum of orbit sizes = {acc}")

# --- swap pairing: orbits of H = S_5 x Z_2 ---
def canon_key(i, j):
    best = None
    for s in S5:
        bp = blk_perm[s]
        c = (bp[i], bp[j])
        if best is None or c < best:
            best = c
    return best


key_of = {}
for (i0, j0, size) in pair_orbits:
    key_of[canon_key(i0, j0)] = (i0, j0, size)
assert len(key_of) == len(pair_orbits)

kept = []                 # (i0, j0, H-orbit size)
acc2 = 0
n_self = 0
for key, (i0, j0, size) in key_of.items():
    pk = canon_key(j0, i0)
    if pk == key:                      # swap maps the S_5-orbit to itself
        kept.append((i0, j0, size))
        acc2 += size
        n_self += 1
    elif key < pk:                     # keep one of the two swapped orbits
        assert key_of[pk][2] == size
        kept.append((i0, j0, 2 * size))
        acc2 += 2 * size
assert acc2 == 9_447_840, acc2
log(f"  H = S_5 x Z_2 orbits: {len(kept)} representatives "
    f"({n_self} swap-self-paired)")
log(f"  EXACT ACCOUNTING: sum of H-orbit sizes = {acc2} = 9,447,840  [OK]")


# ======================================================================
# PART 2: integer permutation modules and exact D2 matrices
# ======================================================================

log("PART 2: integer modules (M41, M9, M9s, M6) and exact D2 matrices")


def imat(rows):
    return tuple(tuple(int(x) for x in r) for r in rows)


def mmul(A, B):
    Bt = tuple(zip(*B))
    return tuple(tuple(sum(a * b for a, b in zip(row, col)) for col in Bt)
                 for row in A)


def mT(A):
    return tuple(zip(*A))


def madd(A, B):
    return tuple(tuple(x + y for x, y in zip(r, s)) for r, s in zip(A, B))


def msub(A, B):
    return tuple(tuple(x - y for x, y in zip(r, s)) for r, s in zip(A, B))


def mscal(c, A):
    return tuple(tuple(c * x for x in r) for r in A)


def meye(n):
    return tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))


def sumzero_matrix(n, action):
    """Integer matrix of a permutation action of n points restricted to the
    sum-zero subspace, in the basis b_i = e_i - e_{n-1}, i = 0..n-2."""
    m = n - 1
    M = [[0] * m for _ in range(m)]
    last = action(n - 1)
    for col in range(m):
        p = action(col)
        if p < m:
            M[p][col] += 1
        if last < m:
            M[last][col] -= 1
    return imat(M)


def gram_sumzero(n):
    m = n - 1
    return imat([[2 if i == j else 1 for j in range(m)] for i in range(m)])


def rho41(perm):
    return sumzero_matrix(5, lambda i: perm[i])


PAIRS10 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
P10_IDX = {p: k for k, p in enumerate(PAIRS10)}


def rho9(perm):
    def act(k):
        i, j = PAIRS10[k]
        a, b = perm[i], perm[j]
        return P10_IDX[(a, b) if a < b else (b, a)]
    return sumzero_matrix(10, act)


WEDGE = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def wedge2(M):
    W = [[0] * 6 for _ in range(6)]
    for r, (i, j) in enumerate(WEDGE):
        for c, (k, l) in enumerate(WEDGE):
            W[r][c] = M[i][k] * M[j][l] - M[i][l] * M[j][k]
    return imat(W)


def rho6(perm):
    return wedge2(rho41(perm))


G41 = gram_sumzero(5)
G9 = gram_sumzero(10)
G6 = wedge2(G41)          # wedge^2 of a Gram matrix is the wedge Gram matrix

RHO = {"M41": rho41, "M9": rho9, "M9s": rho9, "M6": rho6}
GRAMS = {"M41": G41, "M9": G9, "M9s": G9, "M6": G6}
DIMS = {"M41": 4, "M9": 9, "M9s": 9, "M6": 6}
TWISTED = {"M41": False, "M9": False, "M9s": True, "M6": False}
MODULE_COVERS = {
    "M41": ["(4,1)"],
    "M9": ["(4,1)", "(3,2)"],
    "M9s": ["(2,1,1,1)", "(2,2,1)"],
    "M6": ["(3,1,1)"],
}
MODULE_NAMES = ["M41", "M9", "M9s", "M6"]

# numerators of the projections: P_t = NP_t / 2 (integer NP_t)
NPROJ = {}
for name in MODULE_NAMES:
    rho, G, dim, tw = RHO[name], GRAMS[name], DIMS[name], TWISTED[name]
    NP = {}
    for t in ALL_TRANS:
        R = rho(trans(*t))
        N = msub(meye(dim), R) if tw else madd(meye(dim), R)
        NP[t] = N
        # EXACT sanity: idempotence, G-self-adjointness, G-orthogonality
        assert mmul(N, N) == mscal(2, N), (name, t, "P not idempotent")
        assert mmul(mT(N), G) == mmul(G, N), (name, t, "P not G-self-adjoint")
        assert mmul(mmul(mT(R), G), R) == G, (name, t, "rho not G-orthogonal")
    NPROJ[name] = NP

# EXACT sanity: conjugation covariance rho(s) N_t = N_{s t s^-1} rho(s)
# (checked for the group generators; general s follows by multiplicativity,
#  and a random sample of full group elements is checked as well)
for name in MODULE_NAMES:
    rho, tw = RHO[name], TWISTED[name]
    checks = [(g, t) for g in GENS for t in ALL_TRANS]
    rng2 = random.Random(2)
    checks += [(rng2.choice(S5), rng2.choice(ALL_TRANS)) for _ in range(20)]
    for s, t in checks:
        R = rho(s)
        assert mmul(R, NPROJ[name][t]) == mmul(NPROJ[name][conj_trans(s, t)], R)
log("  exact module sanity: idempotence, self-adjointness, orthogonality, "
    "covariance  [OK]")

# exact D1 (den 2^4) and D2 (den 2^16); adjoints are reversed words
D2INT = {}
for name in MODULE_NAMES:
    NP = NPROJ[name]
    N1 = []
    for (ta, tb) in adjacent:
        M = mmul(NP[ta], NP[tb])
        N1.append(mmul(M, M))
    N2 = []
    for (ia, ib) in blocks:
        N2.append(mmul(mmul(N1[ia], N1[ib]),
                       mmul(N1[adj_rev[ia]], N1[adj_rev[ib]])))
    D2INT[name] = N2
log("  exact integer D2 for the 3240 blocks in 4 modules  [OK]")

DEN2 = 1 << 16            # D2 = N2 / 2^16
DEN3 = 1 << 64            # D3 = N3 / 2^64


def d3_int(name, i, j):
    N2 = D2INT[name]
    return mmul(mmul(N2[i], N2[j]), mmul(N2[blk_rev[i]], N2[blk_rev[j]]))


# ======================================================================
# PART 3: float64 sanity (NOT part of the certificate)
# ======================================================================

log("PART 3: float64 sanity cross-check against the Script-18 pipeline")

_B = np.array([[1, -1, 0, 0, 0], [1, 1, -2, 0, 0],
               [1, 1, 1, -3, 0], [1, 1, 1, 1, -4]], float)
_B = _B / np.linalg.norm(_B, axis=1, keepdims=True)


def std41f(perm):
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            fjp = np.array([_B[j][perm[k]] for k in range(5)])
            M[i, j] = _B[i] @ fjp
    return M


def _permpairs(perm):
    M = np.zeros((10, 10))
    for k, (i, j) in enumerate(PAIRS10):
        a, b = sorted((perm[i], perm[j]))
        M[P10_IDX[(a, b)], k] = 1
    return M


_W = np.zeros((10, 5))
for _k, (_i, _j) in enumerate(PAIRS10):
    _W[_k, _i] += 1
    _W[_k, _j] += 1
_U, _s, _ = np.linalg.svd(_W)
_C32 = _U[:, np.sum(_s > 1e-9):]


def rep32f(perm):
    return _C32.T @ _permpairs(perm) @ _C32


def rep311f(perm):
    M = std41f(perm)
    W = np.zeros((6, 6))
    for r, (i, j) in enumerate(WEDGE):
        for c, (k, l) in enumerate(WEDGE):
            W[r, c] = M[i, k] * M[j, l] - M[i, l] * M[j, k]
    return W


REPSF = {
    "(4,1)": (std41f, 4),
    "(3,2)": (rep32f, 5),
    "(3,1,1)": (rep311f, 6),
    "(2,2,1)": (lambda p: sgn(p) * rep32f(p), 5),
    "(2,1,1,1)": (lambda p: sgn(p) * std41f(p), 4),
}
D2F = {}
for rname, (rep, d) in REPSF.items():
    proj = {t: (rep(trans(*t)) + np.eye(d)) / 2 for t in ALL_TRANS}
    n1 = [np.linalg.matrix_power(proj[p[0]] @ proj[p[1]], 2) for p in adjacent]
    D2F[rname] = [n1[ia] @ n1[ib] @ n1[ia].T @ n1[ib].T for (ia, ib) in blocks]

CHOL = {name: np.linalg.cholesky(np.array(GRAMS[name], dtype=float))
        for name in MODULE_NAMES}
CHOL_INV_T = {name: np.linalg.inv(CHOL[name].T) for name in MODULE_NAMES}


def opnorm_G_float(name, N3):
    """Float G-operator norm of N3/2^64 (sanity only)."""
    L = CHOL[name]
    M = np.array(N3, dtype=float) / float(DEN3)
    return np.linalg.svd(L.T @ M @ CHOL_INV_T[name], compute_uv=False)[0]


# (a) exact integer chain vs Script-18 orthonormal float chain, random sample
rng = random.Random(3)
sample = rng.sample(kept, 60)
worst = 0.0
for (i, j, _size) in sample:
    for name in MODULE_NAMES:
        s_int = opnorm_G_float(name, d3_int(name, i, j))
        di = [D2F[r][i] for r in MODULE_COVERS[name]]
        dj = [D2F[r][j] for r in MODULE_COVERS[name]]
        s_flt = max(np.linalg.svd(a @ b @ a.T @ b.T, compute_uv=False)[0]
                    for a, b in zip(di, dj))
        worst = max(worst, abs(s_int - s_flt))
assert worst < 1e-9, worst
log(f"  exact chain vs Script-18 chain on 60 reps x 4 modules: "
    f"max |diff| = {worst:.2e}  [OK]")

# (b) float maxima per irrep over ALL orbit representatives
float_max = {rname: (0.0, None) for rname in REPSF}
for (i, j, _size) in kept:
    for rname in REPSF:
        a, b = D2F[rname][i], D2F[rname][j]
        v = np.linalg.svd(a @ b @ a.T @ b.T, compute_uv=False)[0]
        if v > float_max[rname][0]:
            float_max[rname] = (v, (i, j))
log("  float64 maxima per irrep over the orbit representatives:")
for rname, (v, arg) in float_max.items():
    log(f"    {rname:>10}: max ||D3||_op = {v:.17f}   at rep {arg}")
argmax_rep = float_max["(4,1)"][1]
diff_ref = abs(float_max["(4,1)"][0] - FLOAT64_MAX_SCRIPT18)
assert diff_ref < 1e-13, diff_ref
log(f"  (4,1) max vs Scripts 17/18 value {FLOAT64_MAX_SCRIPT18}: "
    f"|diff| = {diff_ref:.2e}  [OK]")


# ======================================================================
# PART 4: EXACT CERTIFICATION (integer arithmetic only)
# ======================================================================

log("PART 4: exact certification of lambda = 71/2000 on all orbit reps")


def leading_minors_bareiss(C):
    """All n leading principal minors of integer matrix C via fraction-free
    Bareiss elimination.  Returns None if an intermediate pivot is <= 0
    (then C is certainly not positive definite)."""
    n = len(C)
    A = [list(r) for r in C]
    prev = 1
    for k in range(n - 1):
        piv = A[k][k]
        if piv <= 0:
            return None
        for i in range(k + 1, n):
            Aik = A[i][k]
            Ai, Ak = A[i], A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * piv - Aik * Ak[j]) // prev
        prev = piv
    return [A[k][k] for k in range(n)]


def frac_det(sub):
    """Exact determinant via Fraction Gaussian elimination with pivoting."""
    m = len(sub)
    A = [[Fraction(x) for x in row] for row in sub]
    detv = Fraction(1)
    for k in range(m):
        piv = next((r2 for r2 in range(k, m) if A[r2][k] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            detv = -detv
        detv *= A[k][k]
        inv = 1 / A[k][k]
        for r2 in range(k + 1, m):
            f = A[r2][k] * inv
            if f:
                for c2 in range(k, m):
                    A[r2][c2] -= f * A[k][c2]
    return detv


def all_principal_minors_psd(C):
    """PSD fallback: every principal minor >= 0 (exact).  Never needed in
    practice; kept for completeness of the certificate logic."""
    n = len(C)
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            if frac_det([[C[i][j] for j in idx] for i in idx]) < 0:
                return False
    return True


def certify_module(name, bound_num, bound_den):
    """Certify ||D3||_G <= bound for every kept orbit representative in the
    given module.  Exact integers only.  Returns (ok, min_minor_fraction)."""
    N2 = D2INT[name]
    G = GRAMS[name]
    n = DIMS[name]
    b2num = bound_num * bound_num
    b2den = bound_den * bound_den
    scale_g = b2num * DEN3 * DEN3        # bound^2 * 2^128 (numerator side)
    unscale = b2den * DEN3 * DEN3        # minor_k(C) = minors[k] / unscale^k
    grows = [tuple(G[r]) for r in range(n)]
    # for fixed k all reps share the denominator unscale^k, so the per-k
    # minimum can be tracked with pure integer comparisons
    min_int = [None] * n
    n_fallback = 0
    for (i, j, _size) in kept:
        N3 = mmul(mmul(N2[i], N2[j]), mmul(N2[blk_rev[i]], N2[blk_rev[j]]))
        S = mmul(mmul(mT(N3), G), N3)
        C = tuple(tuple(scale_g * grows[r][c] - b2den * S[r][c]
                        for c in range(n)) for r in range(n))
        mins = leading_minors_bareiss(C)
        if mins is None or any(m <= 0 for m in mins):
            n_fallback += 1
            if not all_principal_minors_psd(C):
                return False, None, (i, j)
            continue
        for k, m in enumerate(mins):
            if min_int[k] is None or m < min_int[k]:
                min_int[k] = m
    min_frac = min(Fraction(m, unscale ** (k + 1))
                   for k, m in enumerate(min_int) if m is not None)
    return True, min_frac, n_fallback


results = {}
for name in MODULE_NAMES:
    t0 = time.time()
    ok, min_frac, extra = certify_module(name, LAM_NUM, LAM_DEN)
    dt = time.time() - t0
    if not ok:
        log(f"  {name}: FAILED at representative {extra}")
        raise SystemExit(1)
    results[name] = (min_frac, dt)
    log(f"  {name} ({'+'.join(MODULE_COVERS[name])}): CERTIFIED "
        f"||D3|| <= 71/2000 on {len(kept)} reps in {dt:.1f}s; "
        f"PSD fallbacks used: {extra}; "
        f"min leading-minor margin = {float(min_frac):.3e} (exact fraction kept)")

log("  => all five irreps (4,1),(3,2),(3,1,1),(2,2,1),(2,1,1,1) certified")

# --- tight global bracket on all five irreps via the four carrier modules ---
log(f"PART 4b: tight certification ||D3|| <= {TIGHT_NUM}/{TIGHT_DEN} "
    f"= {TIGHT_NUM/TIGHT_DEN} on all irreps")
tight_results = {}
for name in MODULE_NAMES:
    t0 = time.time()
    ok, min_frac, extra = certify_module(name, TIGHT_NUM, TIGHT_DEN)
    dt = time.time() - t0
    if not ok:
        log(f"  {name}: TIGHT BOUND FAILED at representative {extra}")
        raise SystemExit(1)
    tight_results[name] = (min_frac, dt)
    log(f"  {name} tight ({'+'.join(MODULE_COVERS[name])}): CERTIFIED on "
        f"{len(kept)} reps in {dt:.1f}s; PSD fallbacks: {extra}; "
        f"min leading-minor margin = {float(min_frac):.3e}")
log("  => tight bound certified on all five non-trivial, non-sign irreps")

# ======================================================================
# PART 5: direct certification of RANDOM RAW configurations
# (end-to-end validation of the orbit reduction; exact, no orbits used)
# ======================================================================

log("PART 5: direct exact certification of 200 random raw oriented pairs")
rng = random.Random(5)
n_raw = 0
while n_raw < 200:
    i = rng.randrange(NB)
    j = rng.randrange(NB)
    if i == j or comm(tgt2[i], tgt2[j]) == ID:
        continue
    n_raw += 1
    for name in MODULE_NAMES:
        N3 = d3_int(name, i, j)
        G = GRAMS[name]
        n = DIMS[name]
        S = mmul(mmul(mT(N3), G), N3)
        C = tuple(tuple(LAM_NUM * LAM_NUM * DEN3 * DEN3 * G[r][c]
                        - LAM_DEN * LAM_DEN * S[r][c]
                        for c in range(n)) for r in range(n))
        mins = leading_minors_bareiss(C)
        assert mins is not None and all(m > 0 for m in mins), (name, i, j)
log("  200 random raw configurations certified directly  [OK]")

# ======================================================================
# PART 6: exact rational enclosure of the bottleneck maximum
# ======================================================================

log("PART 6: exact enclosure of the bottleneck (4,1) maximum")
i_max, j_max = argmax_rep
N3 = d3_int("M41", i_max, j_max)
G = G41
S = mmul(mmul(mT(N3), G), N3)     # integer; D3^T G D3 = S / 2^128

# q(y) = det(y*G - S): integer polynomial whose roots are the squared
# G-singular values of D3 multiplied by 2^128.  Computed via exact expansion.
def det_poly(G, S):
    """Coefficients (ascending in y) of det(y*G - S) for small n, exactly."""
    n = len(G)
    # entries are linear polynomials a*y + b represented as (a, b) -> use
    # full symbolic expansion via permutations (n! = 24 terms for n = 4)
    coeffs = [0] * (n + 1)
    for perm in permutations(range(n)):
        sign = 1
        # permutation sign
        for a in range(n):
            for b in range(a + 1, n):
                if perm[a] > perm[b]:
                    sign = -sign
        # product of linear terms (G[i][perm_i]*y - S[i][perm_i])
        poly = [1]
        for i in range(n):
            a, b = G[i][perm[i]], -S[i][perm[i]]
            new = [0] * (len(poly) + 1)
            for d, c in enumerate(poly):
                new[d] += c * b
                new[d + 1] += c * a
            poly = new
        for d, c in enumerate(poly):
            coeffs[d] += sign * c
    return coeffs


q = det_poly(G, S)


def q_eval(fr):
    v = Fraction(0)
    for c in reversed(q):
        v = v * fr + c
    return v


Y_SCALE = DEN3 * DEN3     # 2^128; sigma^2 = y / 2^128

# bracket the largest root using the float value, then bisect exactly
y_float = (float_max["(4,1)"][0] ** 2) * float(Y_SCALE)
lo = Fraction(int(y_float * (1 - 1e-6)))
hi = Fraction(int(y_float * (1 + 1e-6)))
assert q_eval(lo) != 0 and q_eval(hi) != 0
s_lo, s_hi = q_eval(lo) > 0, q_eval(hi) > 0
assert s_lo != s_hi, "no sign change in initial bracket"
for _ in range(220):                      # width / 2^220: overkill precision
    mid = (lo + hi) / 2
    v = q_eval(mid)
    if v == 0:
        lo = hi = mid
        break
    if (v > 0) == s_lo:
        lo = mid
    else:
        hi = mid

# no further root above: exact check q has constant sign on [hi, lambda^2*2^128]
# (Descartes-style check via sign of q at endpoints and monotonicity is not
#  rigorous alone; instead verify with Sturm-free argument: the certificate of
#  Part 4b already proves ALL squared singular values <= (TIGHT bound)^2, so
#  the largest root of q for THIS configuration lies in [lo, hi] once we also
#  check q has no root in (hi, tight^2 * 2^128]: done by exact bisection
#  bracketing below.)
tight2 = Fraction(TIGHT_NUM * TIGHT_NUM, TIGHT_DEN * TIGHT_DEN) * Y_SCALE
# count sign changes of q on a fine exact grid is not a proof; use Sturm chain
def sturm_chain(p):
    def norm(p):
        while p and p[-1] == 0:
            p.pop()
        return p
    def deriv(p):
        return [Fraction(i) * c for i, c in enumerate(p)][1:]
    def rem(a, b):
        a = a[:]
        while len(a) >= len(b) > 0:
            f = a[-1] / b[-1]
            sh = len(a) - len(b)
            for k in range(len(b)):
                a[sh + k] -= f * b[k]
            a = norm(a)
            if not a:
                break
        return a
    chain = [norm([Fraction(c) for c in p])]
    chain.append(norm(deriv(chain[0])))
    while chain[-1]:
        r = rem(chain[-2], chain[-1])
        if not r:
            break
        chain.append([-c for c in r])
    return [c for c in chain if c]


def sturm_count(chain, x):
    signs = []
    for p in chain:
        v = Fraction(0)
        for c in reversed(p):
            v = v * x + c
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


assert lo < hi and q_eval(lo) != 0 and q_eval(hi) != 0 and q_eval(tight2) != 0
chain = sturm_chain(q)
roots_above = sturm_count(chain, hi) - sturm_count(chain, tight2)
assert roots_above == 0, f"unexpected roots in (hi, tight^2]: {roots_above}"
roots_in = sturm_count(chain, lo) - sturm_count(chain, hi)
assert roots_in == 1, roots_in
log("  Sturm: exactly one root of det(y*G - S) in [lo, hi], none above "
    "(up to the tight bound)  [OK]")

# exact rational sqrt bounds for sigma = sqrt(y / 2^128)
def sqrt_lower(fr, digits=25):
    from math import isqrt
    scale = 10 ** (2 * digits)
    n = fr.numerator * scale // fr.denominator
    return Fraction(isqrt(n), 10 ** digits)


def sqrt_upper(fr, digits=25):
    from math import isqrt
    scale = 10 ** (2 * digits)
    n = -((-fr.numerator * scale) // fr.denominator)   # ceil
    r = isqrt(n)
    if r * r < n:
        r += 1
    return Fraction(r, 10 ** digits)


sig_lo = sqrt_lower(lo / Y_SCALE)
sig_hi = sqrt_upper(hi / Y_SCALE)
assert sig_lo * sig_lo <= lo / Y_SCALE and sig_hi * sig_hi >= hi / Y_SCALE
log(f"  characteristic polynomial det(y*G - S) coefficients (ascending):")
for d, c in enumerate(q):
    log(f"    y^{d}: {c}")
log(f"  certified enclosure of the bottleneck sigma_max:")
log(f"    {sig_lo.numerator}/{sig_lo.denominator} <= sigma_max <= "
    f"{sig_hi.numerator}/{sig_hi.denominator}")
log(f"    float: {float(sig_lo):.17f} <= sigma_max <= {float(sig_hi):.17f}")
log(f"    Scripts 17/18 float64 value: {FLOAT64_MAX_SCRIPT18}")
# the float64 reference itself carries rounding error of a few ulps
assert abs(float((sig_lo + sig_hi) / 2) - FLOAT64_MAX_SCRIPT18) < 1e-14

lam2 = Fraction(LAM_NUM * LAM_NUM, LAM_DEN * LAM_DEN)
margin = lam2 - hi / Y_SCALE
assert margin > 0
log(f"  exact margin lambda^2 - sigma_max^2 >= {margin} "
    f"(~ {float(margin):.6e})")
global_margin = lam2 - Fraction(TIGHT_NUM * TIGHT_NUM, TIGHT_DEN * TIGHT_DEN)
assert global_margin > 0
log(f"  exact global margin from the all-irrep tight bound: "
    f"lambda^2 - ||D3||^2 >= {global_margin} (~ {float(global_margin):.6e})")

# ======================================================================
# SUMMARY
# ======================================================================

print()
print("=" * 72)
print("EXACT LEVEL-3 CERTIFICATE: SUMMARY")
print("=" * 72)
print(f"  oriented level-3 pairs:        9,447,840 "
      f"(= 2 x 4,723,920 unordered; exact orbit accounting verified)")
print(f"  H = S_5 x Z_2 orbit reps:      {len(kept)}")
print(f"  certified bound (all irreps):  ||Dhat_3||_op <= 71/2000 = 0.0355")
for name in MODULE_NAMES:
    mf, dt = results[name]
    print(f"    {name:4} ({'+'.join(MODULE_COVERS[name]):>19}): PASS   "
          f"min minor margin {float(mf):.3e}   [{dt:.0f}s]")
print(f"  tight bound (all irreps):       ||Dhat_3|| <= {TIGHT_NUM}/{TIGHT_DEN}"
      f" = {TIGHT_NUM/TIGHT_DEN}   PASS")
for name in MODULE_NAMES:
    mf, dt = tight_results[name]
    print(f"    {name:4} ({'+'.join(MODULE_COVERS[name]):>19}): PASS   "
          f"min minor margin {float(mf):.3e}   [{dt:.0f}s]")
print(f"  bottleneck maximum enclosure:  sigma_max in "
      f"[{float(sig_lo):.17f}, {float(sig_hi):.17f}]")
print(f"  global maximum bracket:        [{float(sig_lo):.17f}, "
      f"{TIGHT_NUM/TIGHT_DEN}]  (exact)")
print(f"  global squared-margin:          lambda^2 - ||Dhat_3||^2 >= "
      f"{global_margin}  (exact)")
print(f"  float64 reference (Scripts 17/18): {FLOAT64_MAX_SCRIPT18}  [consistent]")
print(f"  total time: {time.time()-T_START:.0f}s")
print()
print("LEVEL-3 EXACT RATIONAL CERTIFICATE: PASS")
