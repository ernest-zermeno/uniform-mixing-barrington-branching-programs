"""
Standard-representation checks for S5, S6 and S7.
The reduced census keeps one ordering at each node; its numerical maximum
must not be confused with a claim that every configuration has that norm.
Only S5 has one-dimensional level-2 norm-preserving spaces. For S6/S7 we
report their dimensions, not an arbitrary singular vector as a direction.
Exact integer examples certify both norm 1 and norm < 1 in S6/S7.
The full ordered S5 bound is certified separately by Script 20.
"""

import numpy as np
from itertools import combinations
import time
import sys


def verify_conjecture(n, verbose=True):
    """
    Check the S5 simplex and the reduced S5/S6/S7 operator-norm census.
    Returns a dict with all results.
    """
    T0 = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"DIMENSIONAL OBSTRUCTION VERIFICATION FOR S_{n}")
        print(f"Standard representation (n-1,1), dimension {n-1}")
        print(f"{'='*70}")

    # ===================================================================
    # Step 1: Build orthonormal basis for V = {v in R^n : sum v_i = 0}
    # ===================================================================
    # Gram-Schmidt on {e_0 - e_1, e_0 + e_1 - 2e_2, ...}
    dim = n - 1
    basis = np.zeros((dim, n))
    for k in range(dim):
        # v_k = e_0 + e_1 + ... + e_k - (k+1)*e_{k+1}
        for j in range(k + 1):
            basis[k, j] = 1.0
        basis[k, k + 1] = -(k + 1)
        basis[k] /= np.linalg.norm(basis[k])

    # Verify orthonormality
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(dim)), "Basis not orthonormal!"

    if verbose:
        print(f"\n  Orthonormal basis for V (dim {dim}): VERIFIED")

    # ===================================================================
    # Step 2: Standard representation matrices for transpositions
    # ===================================================================
    all_trans = list(combinations(range(n), 2))
    n_trans = len(all_trans)

    def std_rep_trans(a, b):
        """Standard rep matrix for transposition (a b)."""
        P = np.eye(n)
        P[[a, b]] = P[[b, a]]
        return basis @ P @ basis.T

    def projection(a, b):
        """Leaf Fourier transform P_{(a,b)} = (rho((a,b)) + I) / 2."""
        return (std_rep_trans(a, b) + np.eye(dim)) / 2

    # Precompute all projections
    proj_cache = {}
    for t in all_trans:
        proj_cache[t] = projection(*t)

    # Verify projection properties
    for t in all_trans:
        P = proj_cache[t]
        assert np.allclose(P @ P, P), f"P_{t} not idempotent!"
        assert np.allclose(P, P.T), f"P_{t} not symmetric!"
        assert abs(np.trace(P) - (dim - 1)) < 1e-10, f"P_{t} trace != {dim-1}!"
        assert np.linalg.matrix_rank(P, tol=1e-10) == dim - 1, f"P_{t} rank != {dim-1}!"

    if verbose:
        print(f"  {n_trans} transpositions, projections rank {dim-1}: VERIFIED")

    # ===================================================================
    # Step 3: Enumerate adjacent pairs and valid quadruples
    # ===================================================================
    def trans_perm(a, b):
        p = list(range(n))
        p[a], p[b] = p[b], p[a]
        return tuple(p)

    def compose(a, b):
        return tuple(a[b[i]] for i in range(n))

    def inverse(p):
        r = [0] * n
        for i, j in enumerate(p):
            r[j] = i
        return tuple(r)

    def comm(p1, p2):
        return compose(compose(p1, p2), compose(inverse(p1), inverse(p2)))

    ID = tuple(range(n))

    # Adjacent pairs: transpositions sharing exactly 1 element, nontrivial commutator
    adjacent_pairs = []
    for i, t1 in enumerate(all_trans):
        for j, t2 in enumerate(all_trans):
            if j <= i:
                continue
            if len(set(t1) & set(t2)) == 1:
                p1 = trans_perm(*t1)
                p2 = trans_perm(*t2)
                if comm(p1, p2) != ID:
                    adjacent_pairs.append((t1, t2))

    n_adj = len(adjacent_pairs)

    # Compute commutator target for each pair
    pair_targets = {}
    for idx, (t1, t2) in enumerate(adjacent_pairs):
        pair_targets[idx] = comm(trans_perm(*t1), trans_perm(*t2))

    # Valid quadruples: pairs of adjacent pairs with nontrivial outer commutator
    valid_quads = []
    quad_targets = {}
    for i in range(n_adj):
        for j in range(i + 1, n_adj):
            c = comm(pair_targets[i], pair_targets[j])
            if c != ID:
                idx = len(valid_quads)
                valid_quads.append((i, j))
                quad_targets[idx] = c

    n_quads = len(valid_quads)

    if verbose:
        print(f"  Adjacent pairs: {n_adj}")
        print(f"  Valid quadruples: {n_quads}")

    # ===================================================================
    # Step 4: Compute level-2 Fourier transforms and operator norms
    # ===================================================================
    if verbose:
        print(f"\n  Computing level-2 D_hat matrices...")

    D2_cache = {}
    op_norms = {}
    frob_norms = {}

    for idx, (i, j) in enumerate(valid_quads):
        t1_A, t2_A = adjacent_pairs[i]
        t1_B, t2_B = adjacent_pairs[j]

        PA1, PA2 = proj_cache[t1_A], proj_cache[t2_A]
        PB1, PB2 = proj_cache[t1_B], proj_cache[t2_B]

        DA = np.linalg.matrix_power(PA1 @ PA2, 2)
        DB = np.linalg.matrix_power(PB1 @ PB2, 2)
        DA_inv = np.linalg.matrix_power(PA2 @ PA1, 2)
        DB_inv = np.linalg.matrix_power(PB2 @ PB1, 2)

        D2 = DA @ DB @ DA_inv @ DB_inv
        D2_cache[idx] = D2

        svs = np.linalg.svd(D2, compute_uv=False)
        op_norms[idx] = svs[0]
        frob_norms[idx] = np.sum(svs ** 2)

        if verbose and (idx + 1) % 2000 == 0:
            print(f"    {idx+1}/{n_quads} computed...")

    if verbose:
        print(f"    All {n_quads} computed.")

    # ===================================================================
    # Step 5: Identify problematic quadruples (op-norm = 1)
    # ===================================================================
    problematic = [idx for idx, op in op_norms.items() if op > 1.0 - 1e-10]
    non_problematic = [idx for idx, op in op_norms.items() if op <= 1.0 - 1e-10]

    n_prob = len(problematic)
    n_nonprob = len(non_problematic)

    if verbose:
        print(f"\n  Operator norm = 1: {n_prob}/{n_quads}")
        print(f"  Operator norm < 1: {n_nonprob}/{n_quads}")
        if non_problematic:
            max_nonprob = max(op_norms[idx] for idx in non_problematic)
            print(f"  Max op-norm among non-problematic: {max_nonprob:.10f}")

    # ===================================================================
    # Step 6: Compute preserved directions for problematic quadruples
    # ===================================================================
    quad_dir = {}
    dir_counts = None
    all_match = None
    predicted_cos = None
    if n == 5:
        # Reference directions: projections of e_k onto V
        ref_dirs = []
        for k in range(n):
            ek = np.zeros(n)
            ek[k] = 1.0
            v = ek - np.ones(n) / n  # project onto V
            coords = basis @ v  # express in orthonormal basis
            coords /= np.linalg.norm(coords)
            ref_dirs.append(coords)

        # Verify reference directions form a regular simplex
        predicted_cos = 1.0 / (n - 1)
        if verbose:
            print(f"\n  Reference directions (projections of e_k onto V):")
            print(f"  Predicted pairwise |cos theta| = 1/{n-1} = {predicted_cos:.10f}")

        pairwise_cos = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_ij = abs(np.dot(ref_dirs[i], ref_dirs[j]))
                pairwise_cos.append(cos_ij)

        if verbose:
            print(f"  Actual pairwise |cos theta|: min={min(pairwise_cos):.10f}, "
                  f"max={max(pairwise_cos):.10f}")
            print(f"  All equal to 1/{n-1}? {all(abs(c - predicted_cos) < 1e-10 for c in pairwise_cos)}")

        # For each problematic quadruple, find preserved direction
        quad_dir = {}
        for idx in problematic:
            D2 = D2_cache[idx]
            DtD = D2.T @ D2
            evals, evecs = np.linalg.eigh(DtD)
            # Eigenvector with eigenvalue closest to 1
            evec = evecs[:, -1]
            # Match to reference direction
            best_k = max(range(n), key=lambda k: abs(np.dot(evec, ref_dirs[k])))
            match_cos = abs(np.dot(evec, ref_dirs[best_k]))
            quad_dir[idx] = (best_k, match_cos)

        # Verify all match a reference direction
        all_match = all(match_cos > 1.0 - 1e-6 for _, match_cos in quad_dir.values())
        dir_counts = [sum(1 for _, (k, _) in quad_dir.items() if k == d) for d in range(n)]

        if verbose:
            print(f"\n  All {n_prob} preserved directions match a reference e_k? "
                  f"{'YES' if all_match else 'NO'}")
            if not all_match:
                worst = min(quad_dir.values(), key=lambda x: x[1])
                print(f"    Worst match: direction {worst[0]}, cos = {worst[1]:.10f}")
            print(f"  Direction distribution: {dir_counts}")
            print(f"  Perfectly symmetric ({n_prob}/{n} = {n_prob//n} each)? "
                  f"{'YES' if len(set(dir_counts)) == 1 and dir_counts[0] == n_prob // n else 'NO'}")

        # ===================================================================
        # Step 7: Verify pairwise angles among preserved directions
        # ===================================================================
        if n_prob > 0:
            # Check that problematic pairs with different directions have |cos| = 1/(n-1)
            misaligned_cos = []
            aligned_count = 0
            for i_idx in problematic:
                for j_idx in problematic:
                    if j_idx <= i_idx:
                        continue
                    k_i = quad_dir[i_idx][0]
                    k_j = quad_dir[j_idx][0]
                    if k_i == k_j:
                        aligned_count += 1
                    else:
                        # Compute actual angle between preserved eigenvectors
                        D2_i = D2_cache[i_idx]
                        D2_j = D2_cache[j_idx]
                        _, evecs_i = np.linalg.eigh(D2_i.T @ D2_i)
                        _, evecs_j = np.linalg.eigh(D2_j.T @ D2_j)
                        cos_val = abs(np.dot(evecs_i[:, -1], evecs_j[:, -1]))
                        misaligned_cos.append(cos_val)

            if verbose:
                print(f"\n  Pairwise analysis among {n_prob} problematic quadruples:")
                print(f"    Aligned (same direction): {aligned_count}")
                print(f"    Misaligned (different direction): {len(misaligned_cos)}")
                if misaligned_cos:
                    print(f"    Misaligned |cos| range: [{min(misaligned_cos):.10f}, "
                          f"{max(misaligned_cos):.10f}]")
                    print(f"    All |cos| = 1/{n-1} = {predicted_cos:.10f}? "
                          f"{all(abs(c - predicted_cos) < 1e-6 for c in misaligned_cos)}")

    else:
        dimensions = {}
        for idx in problematic:
            vals = np.linalg.eigvalsh(D2_cache[idx].T @ D2_cache[idx])
            fixed_dim = int(np.sum(np.abs(vals - 1) < 1e-10))
            dimensions[fixed_dim] = dimensions.get(fixed_dim, 0) + 1
        if verbose:
            print("  Numerical level-2 preserved-space dimensions:", dimensions)
            print("  Single-direction simplex diagnostic: not applicable.")

    # ===================================================================
    # Step 8: Level-3 direction mixing check
    # ===================================================================
    if verbose:
        print("\n  Reduced level-3 norm census (float64, tolerance 1e-10)...")

    n_valid_l3 = 0
    n_both_prob = 0
    n_same_dir = 0
    max_op_l3 = 0.0
    n_contracting_l3 = 0
    n_unit_l3 = 0

    for i in range(n_quads):
        tgt_i = quad_targets[i]
        for j in range(i + 1, n_quads):
            tgt_j = quad_targets[j]
            if comm(tgt_i, tgt_j) == ID:
                continue
            n_valid_l3 += 1

            i_prob = op_norms[i] > 1.0 - 1e-10
            j_prob = op_norms[j] > 1.0 - 1e-10
            if i_prob and j_prob:
                n_both_prob += 1
                if n == 5 and quad_dir[i][0] == quad_dir[j][0]:
                    n_same_dir += 1

            # Compute level-3 operator norm
            D3 = D2_cache[i] @ D2_cache[j] @ D2_cache[i].T @ D2_cache[j].T
            op3 = np.linalg.svd(D3, compute_uv=False)[0]
            max_op_l3 = max(max_op_l3, op3)
            assert op3 <= 1.0 + 1e-10
            if op3 < 1.0 - 1e-10:
                n_contracting_l3 += 1
            else:
                n_unit_l3 += 1

        if verbose and (i + 1) % 500 == 0:
            print(f"    i={i+1}/{n_quads}, L3 pairs={n_valid_l3}, "
                  f"max_op={max_op_l3:.10f}")

    expected = {5: (73810, 73810), 6: (1242915, 800055),
                7: (10282335, 3131940)}
    assert (n_valid_l3, n_contracting_l3) == expected[n]
    assert n_contracting_l3 + n_unit_l3 == n_valid_l3
    if n == 5:
        assert all_match and dir_counts == [54] * 5 and n_same_dir == 0
    if verbose:
        print(f"\n  Reduced level-3 pairs: {n_valid_l3}")
        print(f"  Numerically strictly below 1: {n_contracting_l3}")
        print(f"  Numerically equal to 1 (tolerance 1e-10): {n_unit_l3}")
        print(f"  Maximum norm in this census: {max_op_l3:.12f}")
        if n == 5:
            print(f"  S5 valid same-direction pairs: {n_same_dir}")
        print(f"  Time: {time.time() - T0:.1f}s")
    return dict(n=n, n_quads=n_quads, n_valid_l3=n_valid_l3,
                n_contracting_l3=n_contracting_l3, n_unit_l3=n_unit_l3,
                max_op_l3=max_op_l3, dir_counts=dir_counts)


def exact_examples():
    """Independent integer-basis check; G is the Euclidean Gram matrix.

    In the basis e_i-e_(n-1), D=N/2^L. Positive leading principal minors
    of 2^(2L)G-N^T G N certify strict contraction by Sylvester's criterion.
    Unit-norm examples fix a nonzero vector and are products of projections.
    Every internal commutator is checked to be nonidentity.
    """
    from sympy import Matrix, eye, ones
    seed = [(0,1),(0,2),(0,1),(0,4),(0,1),(0,2),(0,1),(1,3)]
    contracting = {
        6: [(0,1),(0,4),(0,5),(2,5),(1,3),(1,5),(1,4),(3,4)],
        7: [(0,1),(0,5),(0,2),(0,3),(0,2),(0,6),(2,4),(2,5)],
    }
    for n in (6, 7):
        identity = tuple(range(n))
        def mul(a, b):
            return tuple(a[b[i]] for i in range(n))
        def inv(a):
            return tuple(a.index(i) for i in range(n))
        def node(ts):
            if len(ts) == 1:
                p = list(identity)
                i, j = ts[0]
                p[i], p[j] = p[j], p[i]
                R = Matrix(n-1, n-1, lambda i,j: int(i == p[j])-int(i == p[n-1]))
                N = eye(n-1) + R
                return tuple(p), N, N, 1
            a, A, As, la = node(ts[:len(ts)//2])
            b, B, Bs, lb = node(ts[len(ts)//2:])
            target = mul(mul(a,b), mul(inv(a),inv(b)))
            assert target != identity, ('invalid internal target', ts)
            return target, A*B*As*Bs, B*A*Bs*As, 2*(la+lb)
        G = eye(n-1) + ones(n-1)
        for kind, leaves in [('unit', seed), ('contractive', contracting[n])]:
            target, N, Nstar, L = node(leaves)
            assert L == 64 and Nstar.T*G == G*N
            if kind == 'unit':
                v = ones(n-1, 1)
                assert all(n-1 not in t for t in leaves)
                assert N*v == (2**L)*v
                print(f"S{n} exact norm 1: nonzero fixed vector, all internal targets valid: PASS")
            else:
                gap = (2**(2*L))*G - N.T*G*N
                assert gap == gap.T
                minors = [gap[:k,:k].det(method='domain-ge') for k in range(1,n)]
                assert all(x > 0 for x in minors)
                print(f"S{n} exact norm < 1: {n-1} positive integer Gram minors: PASS")
            print(f"  leaves={leaves}; target={target}")
    print("EXACT S6/S7 EXAMPLES: PASS")


if __name__ == "__main__":
    exact_examples()
    results = [verify_conjecture(n) for n in (5, 6, 7)]
    print("\nREDUCED CENSUS: NUMERICAL COUNTS, EXACT EXAMPLES ABOVE")
    print("group   valid pairs   below 1   equal to 1   maximum")
    for r in results:
        print(f"S{r['n']} {r['n_valid_l3']:12d} {r['n_contracting_l3']:10d} "
              f"{r['n_unit_l3']:10d} {r['max_op_l3']:.12f}")
    print("S6/S7 have norm-1 examples and strictly contractive examples.")
    print("REDUCED-CENSUS AND EXACT-EXAMPLE CHECKS: PASS")
