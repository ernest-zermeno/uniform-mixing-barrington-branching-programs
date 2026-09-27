"""
06_contraction_operator.py
===========================
GOAL: Compute the contraction factor of each irreducible representation
of S_5 through the Barrington commutator tree, level by level.

If ||D̂(ρ)||²_F contracts by factor λ_ρ < 1 at each level,
then d_TV → 0 exponentially, establishing the uniform-mixing property.

APPROACH:
  - At each tree level, compute the EXACT distribution of the output
    permutation when each step's bit is chosen uniformly at random
  - Track the Fourier coefficient ||D̂(ρ)||²_F at each level
  - The ratio between consecutive levels gives the contraction factor

The tree has depth log₂(ℓ):
  Level 0 (leaves): single EQ blocks, 1 instruction each
  Level 1: AND of 2 EQ blocks via commutator, 4 instructions
  Level 2: AND of 2 level-1 blocks, 16 instructions (= ℓ=4 tree)
  Level 3: AND of 2 level-2 blocks, 64 instructions (= ℓ=8 tree)

NOTE ON CONDITIONING:
  This script computes the distribution over ALL 2^L random bit strings,
  denoted P_T in the paper, NOT the distribution conditioned on inconsistency.
  The security-relevant distribution D_T is conditioned on inconsistent paths.

  The paper transfers the unconditional Fourier bound to D_T with the
  conditioning lemma:
    d_TV(D_T, U) ≤ d_TV(P_T, U) + p,
    p = Pr[consistent] = 2^ℓ / 2^L.
  For ℓ=4: p = 16/65536 = 0.024%, so the additive loss is small.
  For ℓ≥8: p ≤ 2^{-56}, negligible.

  This script does not claim that the level-by-level contraction factors for
  P_T are contraction factors for the conditioned distribution D_T; only the
  final total-variation bound is transferred by the lemma.

Referenced in paper: Section 5.5 (Path to formal proof)
"""

import numpy as np
from itertools import permutations
from collections import Counter
import time

IDENTITY = (0, 1, 2, 3, 4)
S5 = list(permutations(range(5)))
S5_idx = {p: i for i, p in enumerate(S5)}

def perm_inverse(p):
    inv = [0]*5
    for i, j in enumerate(p): inv[j] = i
    return tuple(inv)

def perm_mul(a, b):
    return tuple(a[b[i]] for i in range(5))

def perm_sign(p):
    vis = [False]*5; nc = 0
    for i in range(5):
        if not vis[i]:
            nc += 1; j = i
            while not vis[j]: vis[j] = True; j = p[j]
    return 1 if (5-nc) % 2 == 0 else -1

def cycle_type(p):
    vis = [False]*5; c = []
    for i in range(5):
        if not vis[i]:
            l, j = 0, i
            while not vis[j]: vis[j] = True; j = p[j]; l += 1
            c.append(l)
    return tuple(sorted(c, reverse=True))

def commutator(s, t):
    return perm_mul(perm_mul(s, t), perm_mul(perm_inverse(s), perm_inverse(t)))

# ── Build representations ────────────────────────────────────

def build_std(sigma):
    M = np.zeros((5, 5))
    for j, i in enumerate(sigma): M[i][j] = 1.0
    B = np.zeros((5, 4))
    for k in range(4): B[k, k] = 1.0; B[4, k] = -1.0
    return np.linalg.inv(B.T @ B) @ B.T @ M @ B

def build_ext2(R):
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    M = np.zeros((6, 6))
    for k, (i, j) in enumerate(pairs):
        for l, (a, b) in enumerate(pairs):
            M[l, k] = R[a, i]*R[b, j] - R[b, i]*R[a, j]
    return M

def build_pair(sigma):
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    pidx = {p: k for k, p in enumerate(pairs)}
    M = np.zeros((10, 10))
    for k, (i, j) in enumerate(pairs):
        si, sj = sigma[i], sigma[j]
        t = (min(si, sj), max(si, sj))
        M[pidx[t], k] = 1.0
    return M

def build_all_reps():
    std_m = [build_std(s) for s in S5]
    reps = {
        (4,1): std_m,
        (2,1,1,1): [float(perm_sign(S5[i])) * std_m[i] for i in range(120)],
        (3,1,1): [build_ext2(std_m[i]) for i in range(120)],
        (1,1,1,1,1): [np.array([[float(perm_sign(s))]]) for s in S5],
    }
    pair_m = [build_pair(s) for s in S5]
    P32 = np.zeros((10, 10))
    chi32 = {(1,1,1,1,1):5,(2,1,1,1):1,(2,2,1):1,(3,1,1):-1,(3,2):1,(4,1):-1,(5,):0}
    for i in range(120):
        P32 += chi32[cycle_type(S5[i])] * pair_m[i]
    P32 *= 5.0 / 120.0
    U, sv, _ = np.linalg.svd(P32)
    basis = U[:, [i for i, s in enumerate(sv) if s > 0.5]]
    reps[(3,2)] = [basis.T @ M @ basis for M in pair_m]
    reps[(2,2,1)] = [float(perm_sign(S5[i])) * reps[(3,2)][i] for i in range(120)]
    return reps

PARTS = [(4,1), (3,2), (3,1,1), (2,2,1), (2,1,1,1), (1,1,1,1,1)]


# ── Distribution arithmetic on S_5 ──────────────────────────
# A distribution is a dict {perm: probability}

def uniform_dist():
    return {g: 1.0/120 for g in S5}

def delta_dist(g):
    return {h: (1.0 if h == g else 0.0) for h in S5}

def mix_dist(d1, d2, w1=0.5, w2=0.5):
    """Mixture: w1*d1 + w2*d2"""
    return {g: w1 * d1.get(g, 0) + w2 * d2.get(g, 0) for g in S5}

def product_dist(d1, d2):
    """Distribution of g*h where g~d1, h~d2 independently."""
    result = {g: 0.0 for g in S5}
    for g1, p1 in d1.items():
        if p1 < 1e-15: continue
        for g2, p2 in d2.items():
            if p2 < 1e-15: continue
            prod = perm_mul(g1, g2)
            result[prod] += p1 * p2
    return result

def inverse_dist(d):
    """Distribution of g⁻¹ where g~d."""
    return {perm_inverse(g): p for g, p in d.items()}

def fourier_coefficient(dist, reps, part):
    """Compute D̂(ρ) = Σ_g D(g) · ρ(g)."""
    d = reps[part][0].shape[0]
    Dh = np.zeros((d, d))
    for i, g in enumerate(S5):
        Dh += dist.get(g, 0) * reps[part][i]
    return Dh

def fourier_norm_sq(dist, reps, part):
    """||D̂(ρ)||²_F"""
    Dh = fourier_coefficient(dist, reps, part)
    return np.sum(Dh ** 2)

def dtv_bound(dist, reps):
    """Upper bound on d_TV from Diaconis-Shahshahani."""
    total = 0.0
    for part in PARTS:
        d = reps[part][0].shape[0]
        total += d * fourier_norm_sq(dist, reps, part)
    return 0.5 * np.sqrt(total)


# ── Build distributions level by level ───────────────────────

def leaf_dist(perm):
    """Distribution of a single leaf EQ block output for random bit.
    With prob 1/2: output = perm (bit matches target)
    With prob 1/2: output = I (bit doesn't match)
    THIS IS THE SAME regardless of target_bit (T-independence at leaf!)
    """
    return mix_dist(delta_dist(perm), delta_dist(IDENTITY))


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("CONTRACTION OPERATOR: Level-by-Level Fourier Analysis")
    print("(Unconditional distribution — transferred to D_T by conditioning lemma)")
    print("=" * 70)
    
    reps = build_all_reps()
    t0 = time.time()
    
    # ── Level 0: single leaf ─────────────────────────────
    sigma = (1, 0, 2, 3, 4)  # (0 1)
    tau = (0, 2, 1, 3, 4)    # (1 2)
    
    d_leaf_sigma = leaf_dist(sigma)
    d_leaf_tau = leaf_dist(tau)
    
    print(f"\n  LEVEL 0 (single leaf):")
    print(f"    Distribution: {{σ: 1/2, I: 1/2}}")
    print(f"    d_TV bound: {dtv_bound(d_leaf_sigma, reps):.6e}")
    for part in PARTS:
        n = fourier_norm_sq(d_leaf_sigma, reps, part)
        if n > 1e-15:
            print(f"    ||D̂({part})||²_F = {n:.6e}")
    
    # ── Level 1: single AND block (commutator of 2 leaves) ──
    d_step0 = leaf_dist(sigma)
    d_step1 = leaf_dist(tau)
    d_step2 = leaf_dist(perm_inverse(sigma))
    d_step3 = leaf_dist(perm_inverse(tau))
    
    d_level1 = product_dist(product_dist(d_step0, d_step1),
                            product_dist(d_step2, d_step3))
    
    print(f"\n  LEVEL 1 (single AND block = 4 steps):")
    print(f"    Steps: σ · τ · σ⁻¹ · τ⁻¹ (each {{perm, I}} with p=1/2)")
    print(f"    Target: [σ,τ] = {commutator(sigma, tau)}")
    print(f"    d_TV bound: {dtv_bound(d_level1, reps):.6e}")
    norms_l1 = {}
    for part in PARTS:
        n = fourier_norm_sq(d_level1, reps, part)
        norms_l1[part] = n
        if n > 1e-15:
            print(f"    ||D̂({part})||²_F = {n:.6e}")
    
    # ── Level 2: AND of 2 level-1 blocks (= ℓ=4 tree, 16 steps) ──
    sigma_p = (0, 1, 3, 2, 4)  # (2 3)
    tau_p = (0, 1, 2, 4, 3)    # (3 4)
    
    d_B_step0 = leaf_dist(sigma_p)
    d_B_step1 = leaf_dist(tau_p)
    d_B_step2 = leaf_dist(perm_inverse(sigma_p))
    d_B_step3 = leaf_dist(perm_inverse(tau_p))
    d_level1_B = product_dist(product_dist(d_B_step0, d_B_step1),
                              product_dist(d_B_step2, d_B_step3))
    
    tgt_A = commutator(sigma, tau)
    tgt_B = commutator(sigma_p, tau_p)
    
    d_A_inv_step0 = leaf_dist(perm_inverse(sigma))
    d_A_inv_step1 = leaf_dist(perm_inverse(tau))
    d_A_inv_step2 = leaf_dist(sigma)
    d_A_inv_step3 = leaf_dist(tau)
    d_level1_A_inv = product_dist(product_dist(d_A_inv_step0, d_A_inv_step1),
                                  product_dist(d_A_inv_step2, d_A_inv_step3))
    
    d_B_inv_step0 = leaf_dist(perm_inverse(sigma_p))
    d_B_inv_step1 = leaf_dist(perm_inverse(tau_p))
    d_B_inv_step2 = leaf_dist(sigma_p)
    d_B_inv_step3 = leaf_dist(tau_p)
    d_level1_B_inv = product_dist(product_dist(d_B_inv_step0, d_B_inv_step1),
                                  product_dist(d_B_inv_step2, d_B_inv_step3))
    
    d_level2 = product_dist(product_dist(d_level1, d_level1_B),
                            product_dist(d_level1_A_inv, d_level1_B_inv))
    
    print(f"\n  LEVEL 2 (ℓ=4 tree = 16 steps):")
    print(f"    Target: [[σ,τ],[σ',τ']] = {commutator(tgt_A, tgt_B)}")
    print(f"    d_TV bound: {dtv_bound(d_level2, reps):.6e}")
    norms_l2 = {}
    for part in PARTS:
        n = fourier_norm_sq(d_level2, reps, part)
        norms_l2[part] = n
        if n > 1e-15:
            print(f"    ||D̂({part})||²_F = {n:.6e}")
    
    # ── Contraction factors ──────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  CONTRACTION FACTORS BY REPRESENTATION")
    print(f"  {'='*60}")
    print(f"\n  {'Rep':<16s} {'Level 0':>12s} {'Level 1':>12s} {'Level 2':>12s} {'L1/L0':>10s} {'L2/L1':>10s}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    
    for part in PARTS:
        n0 = fourier_norm_sq(d_leaf_sigma, reps, part)
        n1 = norms_l1.get(part, 0)
        n2 = norms_l2.get(part, 0)
        r1 = n1 / max(n0, 1e-30)
        r2 = n2 / max(n1, 1e-30)
        print(f"  {str(part):<16s} {n0:>12.4e} {n1:>12.4e} {n2:>12.4e} {r1:>10.6f} {r2:>10.6f}")
    
    # ── Conditioning correction ───────────────────────────
    print(f"\n  {'='*60}")
    print(f"  CONDITIONING CORRECTION")
    print(f"  {'='*60}")
    
    # For ℓ=4: Pr[consistent] = 2^4 / 2^16 = 1/4096
    pr_con_4 = 16 / 65536
    pr_incon_4 = 1 - pr_con_4
    correction_4 = pr_con_4
    print(f"\n  ℓ=4: Pr[consistent] = {pr_con_4:.6f}")
    print(f"        Pr[inconsistent] = {pr_incon_4:.6f}")
    print(f"        Max additive loss from conditioning: {correction_4:.6e}")
    print(f"        d_TV(unconditional) = {dtv_bound(d_level2, reps):.6e}")
    print(f"        d_TV(conditional) ≤ d_TV(unconditional) + {correction_4:.6e}")
    print(f"        Error is {correction_4/dtv_bound(d_level2, reps)*100:.2f}% of signal")
    
    # ── Spectral gap analysis ─────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  SPECTRAL GAP ANALYSIS")
    print(f"  {'='*60}")
    
    n41_l0 = fourier_norm_sq(d_leaf_sigma, reps, (4,1))
    n41_l1 = norms_l1[(4,1)]
    n41_l2 = norms_l2[(4,1)]
    
    print(f"\n  (4,1) representation (dominant):")
    print(f"    Level 0 → 1: contraction = {n41_l1/n41_l0:.6f}")
    print(f"    Level 1 → 2: contraction = {n41_l2/n41_l1:.6f}")
    
    geo_41 = np.sqrt((n41_l1/n41_l0) * (n41_l2/n41_l1))
    print(f"    Geometric mean: {geo_41:.6f}")
    print(f"    Implied spectral norm: ||T_(4,1)|| ≈ {np.sqrt(geo_41):.6f}")
    print(f"    Spectral gap: {1 - np.sqrt(geo_41):.6f}")
    
    if np.sqrt(geo_41) < 1.0:
        print(f"\n  ||T_(4,1)|| < 1 ✓")
        print(f"  This implies exponential convergence to uniform")
        print(f"  with rate λ ≈ {np.sqrt(geo_41):.4f} per level.")
        print(f"\n  NOTE: These rates are from the UNCONDITIONAL distribution P_T.")
        print(f"  The paper transfers only the final total-variation bound to D_T:")
        print(f"      d_TV(D_T, U) ≤ d_TV(P_T, U) + Pr[consistent].")
    
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print()
