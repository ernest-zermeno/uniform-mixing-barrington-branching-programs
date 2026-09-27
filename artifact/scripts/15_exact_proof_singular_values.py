#!/usr/bin/env python3
"""
EXACT PROOF (symbolic algebraic arithmetic) that D_1 = (P_sigma P_tau)^2 has
singular values [1,1,...,1, 1/8, 0] for all adjacent transposition pairs in S_n.

Uses sympy with exact symbolic/algebraic arithmetic; no floating point anywhere.
(The orthonormal basis carries radicals sqrt(2), sqrt(6), sqrt(12), sqrt(20);
sympy keeps them exact, so intermediates are algebraic numbers, not rationals.
The certified singular values themselves are rational.)

For S_n acting on V = {v in R^n : sum v_i = 0}, dim = n-1:
  - Orthonormal basis f_k = (e_0+...+e_k - (k+1)*e_{k+1}) / sqrt(k*(k+1)),  k=1..n-1
    (We use 1-indexed: f_1..f_{n-1})
  - rho(sigma) permutes coordinates of R^n; project to V basis
  - P_sigma = (rho(sigma) + I) / 2
  - D_1 = (P_sigma * P_tau)^2
  - Adjacent means |{a,b} ∩ {c,d}| = 1 for sigma=(a,b), tau=(c,d)

We verify:
  S_5: SVs = [1, 1, 1/8, 0]        (2 ones)
  S_6: SVs = [1, 1, 1, 1/8, 0]     (3 ones)
  S_7: SVs = [1, 1, 1, 1, 1/8, 0]  (4 ones)
"""

from sympy import Matrix, Rational, sqrt, zeros, eye, Poly, Symbol, factor
from itertools import combinations
import sys
import time


def orthonormal_basis_vectors(n):
    """
    Return list of n-1 column vectors (as lists of Rationals) forming an
    orthonormal basis for V = {v in R^n : sum v_i = 0}.

    f_k has entries:
      f_k[i] = 1/sqrt(k*(k+1))          for i < k   (0-indexed, i = 0..k-1)
      f_k[k] = -k/sqrt(k*(k+1))
      f_k[i] = 0                         for i > k

    where k = 1, ..., n-1.
    """
    basis = []
    for k in range(1, n):
        v = [Rational(0)] * n
        norm_sq = k * (k + 1)
        # We'll store them as exact; the inner products will use sqrt
        # Actually let's store the unnormalized vector and the norm separately
        for i in range(k):
            v[i] = Rational(1, 1)
        v[k] = Rational(-k, 1)
        # v has norm^2 = k + k^2 = k(k+1)
        basis.append((v, norm_sq))
    return basis


def representation_matrix(sigma_pair, n):
    """
    Compute the (n-1) x (n-1) matrix of rho(sigma) in the orthonormal basis,
    where sigma = transposition (a, b).

    rho(sigma) permutes coordinates: (rho(sigma) v)_i = v_{sigma^{-1}(i)}.
    For a transposition, sigma^{-1} = sigma, so it swaps coordinates a and b.

    Matrix entry [i,j] = <f_i, rho(sigma) f_j>
    where f_j are the orthonormal basis vectors.

    We compute this exactly using the unnormalized vectors and their norms.
    """
    dim = n - 1
    basis = orthonormal_basis_vectors(n)

    a, b = sigma_pair

    M = zeros(dim, dim)
    for j in range(dim):
        # Apply sigma to basis vector j
        fj_vec, fj_normsq = basis[j]
        # sigma(fj): swap entries a and b
        sfj = list(fj_vec)
        sfj[a], sfj[b] = sfj[b], sfj[a]

        for i in range(dim):
            fi_vec, fi_normsq = basis[i]
            # Inner product <fi, sigma(fj)>
            # = (1/sqrt(fi_normsq)) * (1/sqrt(fj_normsq)) * sum(fi_vec[k] * sfj[k])
            dot = sum(fi_vec[k] * sfj[k] for k in range(n))
            # The entry is dot / sqrt(fi_normsq * fj_normsq)
            # Since fi_normsq and fj_normsq are k(k+1) type products,
            # dot^2 / (fi_normsq * fj_normsq) should be rational.
            # Let's check: we need exact sqrt. Let's use sqrt from sympy.
            M[i, j] = Rational(dot, 1) / sqrt(Rational(fi_normsq * fj_normsq, 1))

    return M


def representation_matrix_fast(sigma_pair, n):
    """
    Faster version: compute rho(sigma) matrix using rational square roots.

    Actually, let's compute M[i,j] = <f_i, rho(sigma) f_j> directly.
    The result may involve sqrt factors. But for D^T D eigenvalues we need
    everything rational. Let's think...

    Actually the matrix entries CAN be irrational (involving sqrt(k(k+1))),
    but D^T D will be rational. Let's just compute symbolically.

    For speed, let's precompute the permuted basis vectors.
    """
    dim = n - 1
    basis = orthonormal_basis_vectors(n)
    a, b = sigma_pair

    # Precompute sqrt of norm squares
    sqrt_norms = [sqrt(Rational(ns, 1)) for _, ns in basis]

    M = zeros(dim, dim)
    for j in range(dim):
        fj_vec, fj_normsq = basis[j]
        # Apply sigma: swap a,b
        sfj = list(fj_vec)
        sfj[a], sfj[b] = sfj[b], sfj[a]

        for i in range(dim):
            fi_vec, fi_normsq = basis[i]
            dot = sum(fi_vec[k] * sfj[k] for k in range(n))
            M[i, j] = dot / (sqrt_norms[i] * sqrt_norms[j])

    return M


def get_all_transpositions(n):
    """All transpositions (a,b) with 0 <= a < b < n."""
    return [(a, b) for a in range(n) for b in range(a+1, n)]


def are_adjacent(t1, t2):
    """Two transpositions are adjacent if they share exactly 1 element."""
    s1 = {t1[0], t1[1]}
    s2 = {t2[0], t2[1]}
    return len(s1 & s2) == 1


def get_adjacent_pairs(n):
    """All ordered pairs (sigma, tau) of adjacent transpositions."""
    transpositions = get_all_transpositions(n)
    pairs = []
    for i, t1 in enumerate(transpositions):
        for j, t2 in enumerate(transpositions):
            if i != j and are_adjacent(t1, t2):
                pairs.append((t1, t2))
    return pairs


def verify_sn(n, expected_ones):
    """
    Verify the singular value structure for S_n.

    Expected: (n-2) singular values equal to 1, one equal to 1/8, one equal to 0.
    """
    dim = n - 1
    print(f"\n{'='*70}")
    print(f"  EXACT VERIFICATION FOR S_{n}")
    print(f"  V = {{v in R^{n} : sum = 0}},  dim = {dim}")
    print(f"  Expected SVs: {expected_ones} ones, one 1/8, one 0")
    print(f"{'='*70}")

    pairs = get_adjacent_pairs(n)
    # We only need unordered pairs for the check
    unordered = set()
    for t1, t2 in pairs:
        key = (min(t1, t2), max(t1, t2))
        unordered.add(key)
    unordered = sorted(unordered)

    print(f"  Number of adjacent transposition pairs (unordered): {len(unordered)}")
    print(f"  Number of ordered pairs: {len(pairs)}")

    # Cache representation matrices
    print(f"  Computing representation matrices...")
    t0 = time.time()
    trans = get_all_transpositions(n)
    rho_cache = {}
    for t in trans:
        rho_cache[t] = representation_matrix_fast(t, n)
    t1 = time.time()
    print(f"  Computed {len(trans)} rho matrices in {t1-t0:.1f}s")

    # For each ordered pair, compute D_1 = (P_sigma P_tau)^2 and verify SVs
    all_pass = True
    count = 0

    # We check ALL ordered pairs (both orderings matter for D_1)
    expected_eigenvalues_of_DTD = {Rational(1): expected_ones,
                                    Rational(1, 64): 1,
                                    Rational(0): 1}

    t0 = time.time()
    for sigma, tau in pairs:
        count += 1

        # P_sigma = (rho(sigma) + I) / 2
        I = eye(dim)
        P_sigma = (rho_cache[sigma] + I) / 2
        P_tau = (rho_cache[tau] + I) / 2

        # D_1 = (P_sigma * P_tau)^2
        PsPt = P_sigma * P_tau
        D1 = PsPt * PsPt  # = (P_sigma P_tau)^2

        # D1^T D1
        D1T_D1 = D1.T * D1

        # Simplify entries
        D1T_D1 = D1T_D1.applyfunc(lambda x: x.simplify())

        # Characteristic polynomial
        lam = Symbol('lam')
        char_poly = (D1T_D1 - lam * eye(dim)).det()
        char_poly = char_poly.simplify()

        # Factor and find roots
        p = Poly(char_poly, lam)
        roots = p.all_roots()  # exact roots

        # Count eigenvalues
        eig_count = {}
        for r in roots:
            r_simplified = r.simplify()
            if r_simplified in eig_count:
                eig_count[r_simplified] += 1
            else:
                eig_count[r_simplified] = 1

        # Check against expected
        ok = (eig_count == expected_eigenvalues_of_DTD)

        if not ok:
            print(f"  FAIL pair #{count}: sigma={sigma}, tau={tau}")
            print(f"    Eigenvalues of D^T D: {eig_count}")
            print(f"    Expected: {expected_eigenvalues_of_DTD}")
            all_pass = False
        else:
            if count % 20 == 0 or count <= 3:
                elapsed = time.time() - t0
                print(f"  PASS pair #{count}/{len(pairs)}: sigma={sigma}, tau={tau}  "
                      f"eigenvalues={dict(eig_count)}  [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\n  Checked {count} ordered pairs in {elapsed:.1f}s")

    if all_pass:
        print(f"  *** ALL {count} PAIRS VERIFIED ***")
        print(f"  THEOREM (exact): For every adjacent transposition pair in S_{n},")
        print(f"  D_1 = (P_sigma P_tau)^2 has singular values:")
        svs = ['1'] * expected_ones + ['1/8', '0']
        print(f"    [{', '.join(svs)}]")
        print(f"  with exactly {expected_ones} singular values equal to 1.")
    else:
        print(f"  *** SOME PAIRS FAILED ***")

    return all_pass


def verify_sn_eigenvalue_method(n, expected_ones):
    """
    More robust version using eigenvals() which returns a dict of eigenvalue: multiplicity.
    """
    dim = n - 1
    print(f"\n{'='*70}")
    print(f"  EXACT VERIFICATION FOR S_{n}")
    print(f"  V = {{v in R^{n} : sum = 0}},  dim = {dim}")
    print(f"  Expected: {expected_ones} SVs = 1, one SV = 1/8, one SV = 0")
    print(f"{'='*70}")

    pairs = get_adjacent_pairs(n)

    print(f"  Total ordered adjacent pairs: {len(pairs)}")

    # Cache representation matrices
    print(f"  Computing representation matrices...")
    t0 = time.time()
    trans = get_all_transpositions(n)
    rho_cache = {}
    for t in trans:
        rho_cache[t] = representation_matrix_fast(t, n)
    t1 = time.time()
    print(f"  Computed {len(trans)} rho matrices in {t1-t0:.1f}s")

    all_pass = True
    count = 0

    expected_eigs = {Rational(1): expected_ones,
                     Rational(1, 64): 1,
                     Rational(0): 1}

    t0 = time.time()
    for sigma, tau in pairs:
        count += 1

        I_mat = eye(dim)
        P_sigma = (rho_cache[sigma] + I_mat) / 2
        P_tau = (rho_cache[tau] + I_mat) / 2

        # D_1 = (P_sigma * P_tau)^2
        PsPt = P_sigma * P_tau
        D1 = PsPt * PsPt

        # D1^T D1
        D1T_D1 = D1.T * D1

        # Simplify
        D1T_D1 = D1T_D1.applyfunc(lambda x: x.simplify())

        # Get eigenvalues with multiplicities
        eigs = D1T_D1.eigenvals()

        # Simplify eigenvalue keys
        eigs_simplified = {}
        for ev, mult in eigs.items():
            ev_s = ev.simplify()
            eigs_simplified[ev_s] = mult

        ok = (eigs_simplified == expected_eigs)

        if not ok:
            print(f"  FAIL #{count}: sigma={sigma}, tau={tau}")
            print(f"    Got eigenvalues: {eigs_simplified}")
            print(f"    Expected: {expected_eigs}")
            all_pass = False
        else:
            if count <= 5 or count % 20 == 0 or count == len(pairs):
                elapsed = time.time() - t0
                print(f"  PASS #{count}/{len(pairs)}: σ={sigma}, τ={tau}  "
                      f"eigs(D^T D)={dict(eigs_simplified)}  [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\n  Verified {count}/{len(pairs)} ordered pairs in {elapsed:.1f}s")

    if all_pass:
        svs = ['1'] * expected_ones + ['1/8', '0']
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  THEOREM PROVED (S_{n}):                                 ║")
        print(f"  ║  For ALL {count:>3} adjacent transposition pairs,            ║")
        print(f"  ║  D_1 = (P_σ P_τ)² has singular values:                 ║")
        print(f"  ║    [{', '.join(svs)}]{' '*(40 - len(', '.join(svs)))}║")
        print(f"  ║  Exactly {expected_ones} singular value(s) equal to 1.            ║")
        print(f"  ║  Proof: exact symbolic arithmetic, all {count} cases.     ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
    else:
        print(f"\n  *** VERIFICATION FAILED FOR SOME PAIRS ***")

    return all_pass


def quick_test():
    """Quick sanity check with one pair from S_5."""
    print("Quick sanity check: one pair from S_5")
    n = 5
    dim = 4
    sigma = (0, 1)
    tau = (1, 2)

    rho_s = representation_matrix_fast(sigma, n)
    rho_t = representation_matrix_fast(tau, n)

    print(f"  rho({sigma}) =")
    for i in range(dim):
        row = [rho_s[i,j].simplify() for j in range(dim)]
        print(f"    {row}")

    I_mat = eye(dim)
    P_s = (rho_s + I_mat) / 2
    P_t = (rho_t + I_mat) / 2

    PsPt = P_s * P_t
    D1 = PsPt * PsPt
    D1 = D1.applyfunc(lambda x: x.simplify())

    print(f"\n  D_1 = (P_σ P_τ)² =")
    for i in range(dim):
        row = [D1[i,j] for j in range(dim)]
        print(f"    {row}")

    D1TD1 = D1.T * D1
    D1TD1 = D1TD1.applyfunc(lambda x: x.simplify())

    print(f"\n  D_1^T D_1 =")
    for i in range(dim):
        row = [D1TD1[i,j] for j in range(dim)]
        print(f"    {row}")

    eigs = D1TD1.eigenvals()
    eigs_s = {k.simplify(): v for k, v in eigs.items()}
    print(f"\n  Eigenvalues of D^T D: {eigs_s}")

    # Singular values
    print(f"  Singular values: ", end="")
    for ev, mult in sorted(eigs_s.items(), reverse=True):
        sv = sqrt(ev).simplify()
        print(f"{sv} (mult {mult}), ", end="")
    print()


if __name__ == '__main__':
    print("EXACT SINGULAR VALUE PROOF FOR D_1 = (P_σ P_τ)²")
    print("Using sympy symbolic/algebraic arithmetic; zero numerical error")
    print()

    # Quick sanity check
    quick_test()

    # S_5: expect 2 ones (SVs: 1, 1, 1/8, 0)
    r5 = verify_sn_eigenvalue_method(5, expected_ones=2)

    # S_6: expect 3 ones (SVs: 1, 1, 1, 1/8, 0)
    r6 = verify_sn_eigenvalue_method(6, expected_ones=3)

    # S_7: expect 4 ones (SVs: 1, 1, 1, 1, 1/8, 0)
    r7 = verify_sn_eigenvalue_method(7, expected_ones=4)

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  S_5: {'PROVED' if r5 else 'FAILED'}")
    print(f"  S_6: {'PROVED' if r6 else 'FAILED'}")
    print(f"  S_7: {'PROVED' if r7 else 'FAILED'}")

    if r5 and r6 and r7:
        print(f"\n  MAIN RESULT: For S_n (n=5,6,7), the operator D_1 = (P_σ P_τ)²")
        print(f"  for ANY pair of adjacent transpositions has exactly (n-3)")
        print(f"  singular values equal to 1, one equal to 1/8, and one equal to 0.")
        print(f"  This is proved with mathematical certainty via exact arithmetic.")
