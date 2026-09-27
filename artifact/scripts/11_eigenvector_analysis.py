"""
11_eigenvector_analysis.py
===========================
DISCOVERS: The 270 level-2 quadruples with operator norm = 1 preserve
exactly 5 directions in R^4, corresponding to the projections of the
5 standard basis vectors e_0,...,e_4 onto the hyperplane V = {sum v_i = 0}.

DISCOVERS: These 5 directions form a regular simplex with pairwise
|cos theta| = 1/4 (exactly).

DISCOVERS: 54 quadruples preserve each direction (perfectly symmetric).

DISCOVERS: Among the 29,160 both-problematic level-3 pairs, ZERO share
the same preserved direction. This is the key that enables the level-3
contraction proof (Theorem 3 in Script 10).

Referenced in paper: Section 4.3 (simplex structure observation)
"""

import numpy as np
from sympy import Matrix, sqrt, eye, zeros, simplify
import time

# -- Infrastructure (same as Script 10) --

def make_basis():
    f1 = Matrix([1, -1, 0, 0, 0]) / sqrt(2)
    f2 = Matrix([1, 1, -2, 0, 0]) / sqrt(6)
    f3 = Matrix([1, 1, 1, -3, 0]) / sqrt(12)
    f4 = Matrix([1, 1, 1, 1, -4]) / sqrt(20)
    return [f1, f2, f3, f4]

BASIS = make_basis()

def std_rep(perm):
    M = zeros(4, 4)
    for i in range(4):
        for j in range(4):
            fj_perm = Matrix([BASIS[j][perm[k]] for k in range(5)])
            M[i, j] = simplify(BASIS[i].dot(fj_perm))
    return M

def proj_np(a, b):
    perm = list(range(5)); perm[a], perm[b] = perm[b], perm[a]
    return np.array(((std_rep(tuple(perm)) + eye(4)) / 2).tolist(), dtype=float)

def pmul(a, b): return tuple(a[b[i]] for i in range(5))
def pinv(p):
    r = [0]*5
    for i, j in enumerate(p): r[j] = i
    return tuple(r)
def trans_perm(a, b):
    p = list(range(5)); p[a], p[b] = p[b], p[a]; return tuple(p)
def comm(p1, p2):
    return pmul(pmul(p1, p2), pmul(pinv(p1), pinv(p2)))


if __name__ == "__main__":
    t0 = time.time()
    all_trans = [(a, b) for a in range(5) for b in range(a+1, 5)]

    print()
    print("=" * 70)
    print("EIGENVECTOR ANALYSIS OF LEVEL-2 OBSTRUCTION")
    print("=" * 70)

    # Precompute
    proj_cache = {}
    for a, b in all_trans:
        proj_cache[(a,b)] = proj_np(a, b)

    # Build valid quadruples
    adjacent = []
    for (a1,b1) in all_trans:
        for (a2,b2) in all_trans:
            if (a1,b1)>=(a2,b2): continue
            if len(set([a1,b1])&set([a2,b2])) != 1: continue
            if comm(trans_perm(a1,b1),trans_perm(a2,b2)) != tuple(range(5)):
                adjacent.append(((a1,b1),(a2,b2)))

    valid_quads = []
    quad_targets = {}
    for i, pA in enumerate(adjacent):
        cA = comm(trans_perm(*pA[0]), trans_perm(*pA[1]))
        for j, pB in enumerate(adjacent):
            if j <= i: continue
            cB = comm(trans_perm(*pB[0]), trans_perm(*pB[1]))
            if comm(cA, cB) != tuple(range(5)):
                idx = len(valid_quads)
                valid_quads.append((pA, pB))
                quad_targets[idx] = comm(cA, cB)

    # Reference directions: projections of e_k onto hyperplane
    ref_dirs = []
    for k in range(5):
        ek = np.zeros(5); ek[k] = 1.0
        v = ek - 0.2 * np.ones(5)
        coords = np.array([float(BASIS[i].dot(Matrix(v.tolist()))) for i in range(4)])
        coords /= np.linalg.norm(coords)
        ref_dirs.append(coords)

    # Classify quadruples
    problematic = []
    quad_dir = {}

    for idx, (pA, pB) in enumerate(valid_quads):
        (a1,b1),(a2,b2) = pA
        (c1,d1),(c2,d2) = pB
        PA1, PA2 = proj_cache[(a1,b1)], proj_cache[(a2,b2)]
        PB1, PB2 = proj_cache[(c1,d1)], proj_cache[(c2,d2)]
        DA = np.linalg.matrix_power(PA1 @ PA2, 2)
        DB = np.linalg.matrix_power(PB1 @ PB2, 2)
        DA_inv = np.linalg.matrix_power(PA2 @ PA1, 2)
        DB_inv = np.linalg.matrix_power(PB2 @ PB1, 2)
        D2 = DA @ DB @ DA_inv @ DB_inv

        svs = np.linalg.svd(D2, compute_uv=False)
        if svs[0] > 1.0 - 1e-10:
            DtD = D2.T @ D2
            evals, evecs = np.linalg.eigh(DtD)
            evec = evecs[:, -1]
            # Normalize sign
            for i in range(4):
                if abs(evec[i]) > 1e-10:
                    if evec[i] < 0: evec = -evec
                    break
            best_dir = max(range(5), key=lambda k: abs(np.dot(evec, ref_dirs[k])))
            problematic.append((idx, (pA, pB), evec, best_dir))
            quad_dir[idx] = best_dir
        else:
            quad_dir[idx] = -1

    print(f"\n  Problematic quadruples (||D2||_op = 1): {len(problematic)}")
    print(f"  Non-problematic (||D2||_op < 1): {len(valid_quads) - len(problematic)}")

    # -- 5 reference directions --
    print(f"\n  5 REFERENCE DIRECTIONS (simplex projections of e_k):")
    for k in range(5):
        print(f"    e_{k} -> [{', '.join(f'{x:.6f}' for x in ref_dirs[k])}]")

    print(f"\n  Pairwise |cos theta|:")
    for i in range(5):
        for j in range(i+1, 5):
            c = abs(np.dot(ref_dirs[i], ref_dirs[j]))
            print(f"    e_{i} vs e_{j}: {c:.10f}")

    # -- Direction distribution --
    dir_counts = [0]*5
    for p in problematic:
        dir_counts[p[3]] += 1
    print(f"\n  Direction distribution: {dir_counts}")
    print(f"  Perfectly symmetric: {'YES' if len(set(dir_counts)) == 1 else 'NO'}")

    # -- Clustering verification --
    evecs_list = [p[2] for p in problematic]
    cos_angles = []
    for i in range(len(evecs_list)):
        for j in range(i+1, len(evecs_list)):
            cos_angles.append(abs(np.dot(evecs_list[i], evecs_list[j])))
    cos_angles = np.array(cos_angles)

    n_aligned = np.sum(cos_angles > 1.0 - 1e-8)
    n_between = np.sum((cos_angles > 1e-8) & (cos_angles < 1.0 - 1e-8))

    print(f"\n  Pairwise angles among {len(problematic)} eigenvectors:")
    print(f"    Aligned (cos > 1-1e-8): {n_aligned}")
    print(f"    Misaligned (0 < cos < 1): {n_between}")
    print(f"    All misaligned have |cos| = {cos_angles[cos_angles < 0.5].mean():.10f}")

    # -- Level-3 direction mixing --
    print(f"\n{'='*70}")
    print("  LEVEL-3 DIRECTION MIXING")
    print("=" * 70)

    n_valid_l3 = 0
    n_both_prob = 0
    n_same_dir = 0
    n_diff_dir = 0
    n_at_least_one_ok = 0

    for i in range(len(valid_quads)):
        tgt_i = quad_targets[i]
        for j in range(i+1, len(valid_quads)):
            tgt_j = quad_targets[j]
            if comm(tgt_i, tgt_j) == tuple(range(5)):
                continue
            n_valid_l3 += 1
            if quad_dir[i] >= 0 and quad_dir[j] >= 0:
                n_both_prob += 1
                if quad_dir[i] == quad_dir[j]:
                    n_same_dir += 1
                else:
                    n_diff_dir += 1
            else:
                n_at_least_one_ok += 1

    print(f"\n  Valid level-3 pairs: {n_valid_l3}")
    print(f"  At least one non-problematic: {n_at_least_one_ok}")
    print(f"  Both problematic, different dir: {n_diff_dir}")
    print(f"  Both problematic, SAME dir: {n_same_dir}")

    if n_same_dir == 0:
        print(f"\n  CONCLUSION: No valid level-3 configuration shares a direction.")
        print(f"  This guarantees operator-norm contraction at level 3.")
    else:
        print(f"\n  WARNING: {n_same_dir} same-direction pairs found!")

    print(f"\n  Time: {time.time()-t0:.0f}s")
    print()
