"""
Numerical cross-check of all relevant S5 irreducibles on the reduced census.
Each representation is evaluated on 73,810 level-3 pairs from 405 level-2
blocks. Ordering conventions omit orientations, so these numerical maxima
cannot be used as global induction constants. Script 20 supplies the exact
full-universe certificate for 0.0355; Script 16 checks conditioning.

Representations of S_5:
  (5)       dim 1  Trivial        [excluded from D-S sum]
  (4,1)     dim 4  Standard       [verified here]
  (3,2)     dim 5                 [verified here]
  (3,1,1)   dim 6  Wedge^2(Std)   [verified here]
  (2,2,1)   dim 5  Sign x (3,2)  [verified here]
  (2,1,1,1) dim 4  Sign x Std    [verified here]
  (1^5)     dim 1  Sign           [vanishes: leaf D_hat = 0]

Construction methods:
  (4,1):     Standard rep on hyperplane V = {v in R^5 : sum=0}
  (2,1,1,1): sign(sigma) * rho_{(4,1)}(sigma)
  (3,1,1):   Exterior square of (4,1)
  (3,2):     Character projection from Sym^2((4,1))
  (2,2,1):   sign(sigma) * rho_{(3,2)}(sigma)
"""

import numpy as np
from itertools import permutations
import time

# ===================================================================
# S_5 Group Infrastructure
# ===================================================================

def compose(a, b):
    return tuple(a[b[i]] for i in range(5))

def inverse(p):
    r = [0]*5
    for i, j in enumerate(p): r[j] = i
    return tuple(r)

def sign_perm(p):
    visited = [False]*5; s = 1
    for i in range(5):
        if visited[i]: continue
        j = i; cl = 0
        while not visited[j]:
            visited[j] = True; j = p[j]; cl += 1
        s *= (-1)**(cl - 1)
    return s

def cycle_type(p):
    visited = [False]*5; cycles = []
    for i in range(5):
        if visited[i]: continue
        j = i; length = 0
        while not visited[j]:
            visited[j] = True; j = p[j]; length += 1
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))

def trans_perm(a, b):
    p = list(range(5)); p[a], p[b] = p[b], p[a]; return tuple(p)

def comm(p1, p2):
    return compose(compose(p1, p2), compose(inverse(p1), inverse(p2)))

ALL_S5 = [tuple(p) for p in permutations(range(5))]
ALL_TRANS = [(a, b) for a in range(5) for b in range(a+1, 5)]  # 10 transpositions

# Character table of S_5 (indexed by cycle type)
CHAR = {
    (5,):       {(1,1,1,1,1): 1, (2,1,1,1): 1,  (2,2,1): 1,  (3,1,1): 1,  (3,2): 1,  (4,1): 1,  (5,): 1},
    (4,1):      {(1,1,1,1,1): 4, (2,1,1,1): 2,  (2,2,1): 0,  (3,1,1): 1,  (3,2):-1,  (4,1): 0,  (5,):-1},
    (3,2):      {(1,1,1,1,1): 5, (2,1,1,1): 1,  (2,2,1): 1,  (3,1,1):-1,  (3,2): 1,  (4,1):-1,  (5,): 0},
    (3,1,1):    {(1,1,1,1,1): 6, (2,1,1,1): 0,  (2,2,1):-2,  (3,1,1): 0,  (3,2): 0,  (4,1): 0,  (5,): 1},
    (2,2,1):    {(1,1,1,1,1): 5, (2,1,1,1):-1,  (2,2,1): 1,  (3,1,1):-1,  (3,2):-1,  (4,1): 1,  (5,): 0},
    (2,1,1,1):  {(1,1,1,1,1): 4, (2,1,1,1):-2,  (2,2,1): 0,  (3,1,1): 1,  (3,2): 1,  (4,1): 0,  (5,):-1},
    (1,1,1,1,1):{(1,1,1,1,1): 1, (2,1,1,1):-1,  (2,2,1): 1,  (3,1,1): 1,  (3,2):-1,  (4,1):-1,  (5,): 1},
}

# ===================================================================
# Standard Representation (4,1) — dim 4
# ===================================================================

STD_BASIS = np.array([
    [1, -1, 0, 0, 0],
    [1, 1, -2, 0, 0],
    [1, 1, 1, -3, 0],
    [1, 1, 1, 1, -4],
], dtype=float)
STD_BASIS[0] /= np.sqrt(2)
STD_BASIS[1] /= np.sqrt(6)
STD_BASIS[2] /= np.sqrt(12)
STD_BASIS[3] /= np.sqrt(20)

def std_rep(perm):
    """4x4 standard representation matrix for a permutation."""
    P5 = np.zeros((5, 5))
    for i in range(5): P5[perm[i], i] = 1.0
    return STD_BASIS @ P5 @ STD_BASIS.T

# ===================================================================
# Build ALL Irreducible Representations (for transpositions)
# ===================================================================

def build_all_reps():
    """
    Build representation matrices for all 10 transpositions in all 5
    non-trivial irreps (excluding sign, which vanishes).

    Returns: dict {name: {(a,b): rho_matrix}}
    """
    reps = {}

    # --- (4,1): Standard ---
    reps['(4,1)'] = {t: std_rep(trans_perm(*t)) for t in ALL_TRANS}

    # --- (2,1,1,1): Sign x Standard ---
    # For transposition sigma: sign(sigma) = -1
    reps['(2,1,1,1)'] = {t: -reps['(4,1)'][t] for t in ALL_TRANS}

    # --- (3,1,1): Exterior square of Standard ---
    # Basis: e_i ^ e_j for 0 <= i < j <= 3 (6 elements)
    wedge_idx = [(i, j) for i in range(4) for j in range(i+1, 4)]
    reps['(3,1,1)'] = {}
    for t in ALL_TRANS:
        rho = reps['(4,1)'][t]  # 4x4
        M = np.zeros((6, 6))
        for ci, (i, j) in enumerate(wedge_idx):
            vi, vj = rho[:, i], rho[:, j]
            for ri, (k, l) in enumerate(wedge_idx):
                M[ri, ci] = vi[k]*vj[l] - vi[l]*vj[k]
        reps['(3,1,1)'][t] = M

    # --- (3,2): Character projection from Sym^2(Standard) ---
    # Step 1: Compute standard rep for all 120 elements of S_5
    std_all = {g: std_rep(g) for g in ALL_S5}

    # Step 2: Build symmetric subspace of V tensor V (dim 16 -> 10)
    d = 4
    swap = np.zeros((d*d, d*d))
    for i in range(d):
        for j in range(d):
            swap[i*d+j, j*d+i] = 1.0
    P_sym = (np.eye(d*d) + swap) / 2.0
    evals, evecs = np.linalg.eigh(P_sym)
    sym_basis = evecs[:, evals > 0.5]  # (16, 10)
    assert sym_basis.shape[1] == 10, f"S^2 dim should be 10, got {sym_basis.shape[1]}"

    # Step 3: Character projector for (3,2) within S^2(V)
    # P_{(3,2)} = (d_rho / |G|) * Sum_g chi_{(3,2)}(g) * S^2(rho(g))
    proj_32 = np.zeros((10, 10))
    s2_cache = {}
    for g in ALL_S5:
        rho_g = std_all[g]
        kron_g = np.kron(rho_g, rho_g)
        S2_g = sym_basis.T @ kron_g @ sym_basis  # 10x10
        s2_cache[g] = S2_g
        chi = CHAR[(3,2)][cycle_type(g)]
        proj_32 += chi * S2_g
    proj_32 *= 5.0 / 120.0

    # Step 4: Extract (3,2) basis (rank 5)
    evals_32, evecs_32 = np.linalg.eigh(proj_32)
    basis_32 = evecs_32[:, evals_32 > 0.5]  # (10, 5)
    assert basis_32.shape[1] == 5, f"(3,2) dim should be 5, got {basis_32.shape[1]}"

    # Step 5: Extract (3,2) rep matrices for transpositions
    reps['(3,2)'] = {}
    for t in ALL_TRANS:
        g = trans_perm(*t)
        M = basis_32.T @ s2_cache[g] @ basis_32  # 5x5
        reps['(3,2)'][t] = M

    # --- (2,2,1): Sign x (3,2) ---
    reps['(2,2,1)'] = {t: -reps['(3,2)'][t] for t in ALL_TRANS}

    return reps


def verify_reps(reps):
    """Verify representation matrices: traces, orthogonality, involution."""
    expected_traces = {
        '(4,1)': 2, '(3,2)': 1, '(3,1,1)': 0,
        '(2,2,1)': -1, '(2,1,1,1)': -2
    }
    dims = {'(4,1)': 4, '(3,2)': 5, '(3,1,1)': 6, '(2,2,1)': 5, '(2,1,1,1)': 4}

    for name in reps:
        d = dims[name]
        exp_tr = expected_traces[name]
        for t in ALL_TRANS:
            M = reps[name][t]
            assert M.shape == (d, d), f"{name} wrong shape at {t}"
            tr = np.trace(M)
            assert abs(tr - exp_tr) < 1e-10, f"{name} trace({t}) = {tr:.6f}, expected {exp_tr}"
            assert np.allclose(M @ M.T, np.eye(d), atol=1e-10), f"{name} not orthogonal at {t}"
            assert np.allclose(M @ M, np.eye(d), atol=1e-10), f"{name} not involution at {t}"
    return True


# ===================================================================
# Build Barrington Structure (pairs, quadruples, targets)
# ===================================================================

def build_barrington_structure():
    """Build adjacent pairs, valid quadruples, and level-2 targets."""
    # Adjacent pairs: transpositions with nontrivial commutator
    adjacent_pairs = []
    for i, (a1, b1) in enumerate(ALL_TRANS):
        for j, (a2, b2) in enumerate(ALL_TRANS):
            if j <= i: continue
            if len({a1, b1} & {a2, b2}) != 1: continue
            p1, p2 = trans_perm(a1, b1), trans_perm(a2, b2)
            if comm(p1, p2) != tuple(range(5)):
                adjacent_pairs.append(((a1, b1), (a2, b2)))

    # Valid quadruples: pairs of adjacent pairs with nontrivial outer commutator
    valid_quads = []
    quad_targets = {}
    for i, pA in enumerate(adjacent_pairs):
        cA = comm(trans_perm(*pA[0]), trans_perm(*pA[1]))
        for j, pB in enumerate(adjacent_pairs):
            if j <= i: continue
            cB = comm(trans_perm(*pB[0]), trans_perm(*pB[1]))
            if comm(cA, cB) != tuple(range(5)):
                idx = len(valid_quads)
                valid_quads.append((pA, pB))
                quad_targets[idx] = comm(cA, cB)

    return adjacent_pairs, valid_quads, quad_targets


# ===================================================================
# Contraction Analysis for One Representation
# ===================================================================

def contraction_analysis(rep_name, rho_trans, adjacent_pairs, valid_quads, quad_targets):
    """
    Full contraction analysis (levels 0-3) for one representation.

    rho_trans: dict {(a,b): matrix} — representation matrices for transpositions
    """
    t0 = time.time()
    some_t = ALL_TRANS[0]
    d = rho_trans[some_t].shape[0]

    # Leaf projections: P_sigma = (rho(sigma) + I) / 2
    proj = {t: (rho_trans[t] + np.eye(d)) / 2.0 for t in ALL_TRANS}

    # Check leaf projection properties
    leaf_rank = int(round(np.trace(proj[some_t])))
    leaf_frob = np.trace(proj[some_t])  # = rank for a projection

    # --- Level 1: (P_s P_t)^2 for all adjacent pairs ---
    l1_ops = []
    l1_frobs = []
    for (s, t) in adjacent_pairs:
        D1 = np.linalg.matrix_power(proj[s] @ proj[t], 2)
        l1_ops.append(np.linalg.svd(D1, compute_uv=False)[0])
        l1_frobs.append(np.sum(np.linalg.svd(D1, compute_uv=False)**2))
    max_l1_op = max(l1_ops)
    max_l1_frob = max(l1_frobs)

    # --- Level 2: D2 = DA . DB . DA_inv . DB_inv ---
    D2_cache = {}
    l2_ops = []
    l2_frobs = []
    for idx, (pA, pB) in enumerate(valid_quads):
        DA = np.linalg.matrix_power(proj[pA[0]] @ proj[pA[1]], 2)
        DB = np.linalg.matrix_power(proj[pB[0]] @ proj[pB[1]], 2)
        DA_inv = np.linalg.matrix_power(proj[pA[1]] @ proj[pA[0]], 2)
        DB_inv = np.linalg.matrix_power(proj[pB[1]] @ proj[pB[0]], 2)
        D2 = DA @ DB @ DA_inv @ DB_inv
        D2_cache[idx] = D2
        svs = np.linalg.svd(D2, compute_uv=False)
        l2_ops.append(svs[0])
        l2_frobs.append(np.sum(svs**2))
    max_l2_op = max(l2_ops)
    max_l2_frob = max(l2_frobs)
    n_l2_op1 = sum(1 for x in l2_ops if x > 1 - 1e-10)

    # --- Level 3: D3 = D2_A . D2_B . D2_A^T . D2_B^T ---
    max_l3_op = 0.0
    max_l3_frob = 0.0
    n_l3 = 0
    n_geq1 = 0
    for i in range(len(valid_quads)):
        for j in range(i+1, len(valid_quads)):
            if comm(quad_targets[i], quad_targets[j]) == tuple(range(5)):
                continue
            n_l3 += 1
            D3 = D2_cache[i] @ D2_cache[j] @ D2_cache[i].T @ D2_cache[j].T
            svs = np.linalg.svd(D3, compute_uv=False)
            op = svs[0]
            frob = np.sum(svs**2)
            max_l3_op = max(max_l3_op, op)
            max_l3_frob = max(max_l3_frob, frob)
            if op >= 1.0 - 1e-10:
                n_geq1 += 1

    elapsed = time.time() - t0
    return {
        'name': rep_name, 'dim': d, 'leaf_rank': leaf_rank,
        'leaf_frob': leaf_frob,
        'l1_max_op': max_l1_op, 'l1_max_frob': max_l1_frob,
        'l2_max_op': max_l2_op, 'l2_max_frob': max_l2_frob, 'l2_n_op1': n_l2_op1,
        'l3_configs': n_l3, 'l3_max_op': max_l3_op, 'l3_max_frob': max_l3_frob,
        'l3_n_geq1': n_geq1,
        'time': elapsed,
    }


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    T_START = time.time()

    print()
    print("=" * 72)
    print("ALL-REPRESENTATIONS REDUCED-CENSUS CROSS-CHECK")
    print("Numerical reduced census; the exact full-universe certificate is Script 20")
    print("=" * 72)

    # Build representations
    print("\n--- Building irreducible representations ---")
    reps = build_all_reps()

    print("--- Verifying representation matrices ---")
    verify_reps(reps)
    print("  All verifications passed.\n")

    # Build Barrington structure
    print("--- Building Barrington structure ---")
    adjacent_pairs, valid_quads, quad_targets = build_barrington_structure()
    print(f"  Adjacent pairs: {len(adjacent_pairs)}")
    print(f"  Valid quadruples: {len(valid_quads)}")
    print(f"  (Level-3 configs will be computed per representation)\n")

    # Sign representation: vanishes at leaf
    print("--- Sign representation (1^5): vanishes ---")
    print("  For transposition sigma: sign(sigma) = -1")
    print("  Leaf D_hat = (sign(sigma) + 1) / 2 = (-1 + 1) / 2 = 0")
    print("  Propagates to all levels by multiplicativity.")
    print("  Contribution to D-S bound: ZERO.\n")

    # Run contraction analysis for each representation
    rep_order = ['(4,1)', '(3,2)', '(3,1,1)', '(2,2,1)', '(2,1,1,1)']
    dims = {'(4,1)': 4, '(3,2)': 5, '(3,1,1)': 6, '(2,2,1)': 5, '(2,1,1,1)': 4}
    results = {}

    for rep_name in rep_order:
        print("=" * 72)
        print(f"  REPRESENTATION {rep_name}  (dim = {dims[rep_name]})")
        print("=" * 72)
        R = contraction_analysis(rep_name, reps[rep_name],
                                 adjacent_pairs, valid_quads, quad_targets)
        results[rep_name] = R

        print(f"  Leaf: rank-{R['leaf_rank']} projection, ||P||^2_F = {R['leaf_frob']:.0f}")
        print(f"  Level 1: max ||D1||_op = {R['l1_max_op']:.10f}, "
              f"max ||D1||^2_F = {R['l1_max_frob']:.10f}")
        print(f"  Level 2: max ||D2||_op = {R['l2_max_op']:.10f}, "
              f"max ||D2||^2_F = {R['l2_max_frob']:.10f}")
        print(f"           quadruples with ||D2||_op = 1: {R['l2_n_op1']}/{len(valid_quads)}")
        print(f"  Level 3: {R['l3_configs']} configs checked")
        print(f"           reduced max ||D3||_op = {R['l3_max_op']:.10f}")
        print(f"           max ||D3||^2_F = {R['l3_max_frob']:.10f}")
        print(f"           configs with ||D3||_op >= 1: {R['l3_n_geq1']}")
        status = "YES" if R['l3_n_geq1'] == 0 else "**NO** — CONTRACTION FAILS"
        print(f"           ALL STRICTLY < 1: {status}")
        print(f"  Time: {R['time']:.1f}s\n")

    print("\nREDUCED-CENSUS SUMMARY (NUMERICAL)")
    for rep_name in rep_order:
        R = results[rep_name]
        assert R['l3_configs'] == 73810
        assert R['l3_n_geq1'] == 0
        print(f"  {rep_name}: observed max = {R['l3_max_op']:.12f}")
    print("  These maxima apply only to the reduced census.")
    print("  Global induction uses 0.0355 from the separate exact Script 20 certificate.")
    print(f"  Total time: {time.time() - T_START:.1f}s")
