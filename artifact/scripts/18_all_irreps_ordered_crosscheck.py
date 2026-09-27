"""
18_all_irreps_ordered_crosscheck.py
===================================
Full-oriented level-3 cross-check over ALL FIVE non-trivial, non-sign irreducible representations of S_5.

Motivation (audit 2026-06-26): Script 10/12 enumerate a REDUCED orientation quotient (30 unordered
adjacent level-1 pairs -> 405 level-2 quadruples); Script 17 enumerates the ORDERED level-2 universe for the
bottleneck (4,1) only (60 ordered pairs -> 3240 ordered level-2 blocks -> 4,723,920 unordered pairs). This
script closes the remaining gap: it verifies the level-3 operator-norm contraction for EVERY non-trivial,
non-sign irrep over the full ordered universe.

Counts (permutation-level, rep-independent):
  - 60 ordered adjacent level-1 pairs;
  - 3240 ordered valid level-2 blocks;
  - 4,723,920 unordered pairs of ordered level-2 blocks (i<j);
  - swapping (A,B)->(B,A) gives the transpose, hence the SAME singular values, so this covers the
    9,447,840 oriented pairs.

Representations (S_5 has irreps of dim 1,1,4,4,5,5,6):
  (4,1)     dim 4  = standard
  (3,2)     dim 5  = component of the action on 2-subsets
  (3,1,1)   dim 6  = wedge^2 of the standard rep
  (2,2,1)   dim 5  = sign (x) (3,2)
  (2,1,1,1) dim 4  = sign (x) (4,1)

Expected maxima (independently reproduced by three audits):
  (4,1) 0.035400232084 ; (3,2) 0.000019292891 ; (3,1,1) 0.000010474590 ; (2,2,1)~0 ; (2,1,1,1)~0.
ZERO pairs with operator norm >= 1.  Conservative paper bound lambda = 0.0355 covers all.
"""
import numpy as np
import time


def trans(a, b):
    p = list(range(5)); p[a], p[b] = p[b], p[a]; return tuple(p)
def pmul(a, b): return tuple(a[b[i]] for i in range(5))
def pinv(p):
    r = [0] * 5
    for i, j in enumerate(p): r[j] = i
    return tuple(r)
def comm(p, q): return pmul(pmul(p, q), pmul(pinv(p), pinv(q)))
def sign(p):
    s = 1
    for i in range(5):
        for j in range(i + 1, 5):
            if p[i] > p[j]: s = -s
    return s


ID = tuple(range(5))
ALL = [(a, b) for a in range(5) for b in range(a + 1, 5)]

# (4,1) standard rep in an orthonormal basis
_B = np.array([[1, -1, 0, 0, 0], [1, 1, -2, 0, 0], [1, 1, 1, -3, 0], [1, 1, 1, 1, -4]], float)
_B = _B / np.linalg.norm(_B, axis=1, keepdims=True)
def std41(perm):
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            fjp = np.array([_B[j][perm[k]] for k in range(5)])
            M[i, j] = _B[i] @ fjp
    return M

# (3,2): 5-dim component of perm rep on the 10 unordered pairs
_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
_pidx = {p: k for k, p in enumerate(_pairs)}
def _permpairs(perm):
    M = np.zeros((10, 10))
    for k, (i, j) in enumerate(_pairs):
        a, b = sorted((perm[i], perm[j])); M[_pidx[(a, b)], k] = 1
    return M
_W = np.zeros((10, 5))
for _k, (_i, _j) in enumerate(_pairs):
    _W[_k, _i] += 1; _W[_k, _j] += 1
_U, _s, _ = np.linalg.svd(_W)
_C32 = _U[:, np.sum(_s > 1e-9):]      # 10x5 carrier of (3,2)
def rep32(perm):
    return _C32.T @ _permpairs(perm) @ _C32

# (3,1,1) = wedge^2 of standard (4,1), dim 6
_wedge = [(i, j) for i in range(4) for j in range(i + 1, 4)]   # 6
def rep311(perm):
    M = std41(perm)
    W = np.zeros((6, 6))
    for r, (i, j) in enumerate(_wedge):
        for c, (k, l) in enumerate(_wedge):
            W[r, c] = M[i, k] * M[j, l] - M[i, l] * M[j, k]
    return W

REPS = {
    "(4,1)":     (lambda p: std41(p), 4),
    "(3,2)":     (lambda p: rep32(p), 5),
    "(3,1,1)":   (lambda p: rep311(p), 6),
    "(2,2,1)":   (lambda p: sign(p) * rep32(p), 5),
    "(2,1,1,1)": (lambda p: sign(p) * std41(p), 4),
}


def main():
    t0 = time.time()
    # rep-independent structure: 60 ordered adjacent pairs, 3240 ordered level-2 blocks, targets
    adjacent = []
    for t1 in ALL:
        for t2 in ALL:
            if t1 == t2 or len(set(t1) & set(t2)) != 1:
                continue
            if comm(trans(*t1), trans(*t2)) == ID:
                continue
            adjacent.append((t1, t2))
    assert len(adjacent) == 60, len(adjacent)
    tgt1 = {p: comm(trans(*p[0]), trans(*p[1])) for p in adjacent}

    blocks = []          # (pairA, pairB)
    tgt2 = []
    for A in adjacent:
        for Bp in adjacent:
            if A == Bp:
                continue
            t = comm(tgt1[A], tgt1[Bp])
            if t == ID:
                continue
            blocks.append((A, Bp)); tgt2.append(t)
    assert len(blocks) == 3240, len(blocks)
    print(f"60 ordered level-1 pairs, {len(blocks)} ordered level-2 blocks", flush=True)

    results = {}
    for name, (rep, d) in REPS.items():
        proj = {t: (rep(trans(*t)) + np.eye(d)) / 2 for t in ALL}
        d1 = {p: np.linalg.matrix_power(proj[p[0]] @ proj[p[1]], 2) for p in adjacent}
        d2 = [d1[A] @ d1[Bp] @ d1[A].T @ d1[Bp].T for (A, Bp) in blocks]
        mx = 0.0; nval = 0
        n = len(d2)
        for i in range(n):
            di = d2[i]; ti = tgt2[i]
            for j in range(i + 1, n):
                if comm(ti, tgt2[j]) == ID:
                    continue
                nval += 1
                nv = np.linalg.norm(di @ d2[j] @ di.T @ d2[j].T, 2)
                if nv > mx:
                    mx = nv
        results[name] = (mx, nval)
        print(f"  {name:>10}: max ||D3||_op = {mx:.12f}   (valid unordered pairs = {nval})", flush=True)

    nval = results["(4,1)"][1]
    print(f"\nUnordered pairs of ordered level-2 blocks: {nval}")
    assert nval == 4723920, nval
    print(f"Oriented pairs covered (swap = transpose, same singular values): {2*nval}")
    overall = max(v[0] for v in results.values())
    print(f"Overall max over ALL irreps = {overall:.12f}  <  0.0355 : {overall < 0.0355}")
    print(f"Zero pairs with norm >= 1: {overall < 1.0}")
    print(f"ALL-IRREPS ORDERED CROSS-CHECK: {'PASS' if overall < 0.0355 else 'FAIL'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
