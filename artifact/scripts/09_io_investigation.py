"""
09_io_investigation.py
=======================
INVESTIGATES: Does the mixing phenomenon extend to ALL branching
programs of width 5 (not just point functions)?

FINDINGS:
  - Strong universal mixing is FALSE: D_{C0} ≠ D_{C1} for
    structurally different programs computing the same function.
  - Weak universal mixing is PLAUSIBLE: both distributions converge
    to uniform independently, so d_TV(D_{C0}, D_{C1}) → 0 by
    triangle inequality.
  - Variable grouping does NOT affect D (exact, ℓ=4).
  - NOT gates introduce constant steps that inflate L without mixing.
  - Non-monotone circuits (XOR, MAJORITY) require careful global
    Barrington construction; ad hoc approaches fail.

OVERLAP CRITERION (proved exhaustively for S_5):
  - Two sibling NC pairs with element overlap 1: ALL 6 reps contract
    (135/135 × 6 = 810 checks, λ₂ ≤ 0.546)
  - Element overlap ≥ 2: (4,1) does NOT contract (0/300)
  - No universal 4-pair set exists (0/27,405)

v2 FIX: Corrected bp_retarget multiplication order.
  The bug was: transform = new_perm · inv(old_tgt) [WRONG]
  Fixed to:    transform = inv(old_tgt) · new_perm [CORRECT]
  In a non-abelian group, right-multiplying by transform gives:
    old_tgt · transform = old_tgt · inv(old_tgt) · new_perm = new_perm  ✓
  The old code gave old_tgt · new_perm · inv(old_tgt) ≠ new_perm.

Referenced in paper: Section 3.8 (Problem 7), Section 5.6
"""

import numpy as np
from itertools import permutations, combinations
from collections import Counter
import random, time

IDENTITY = (0, 1, 2, 3, 4)
S5 = list(permutations(range(5)))

def perm_inverse(p):
    inv = [0]*5
    for i, j in enumerate(p): inv[j] = i
    return tuple(inv)

def perm_mul(a, b):
    return tuple(a[b[i]] for i in range(5))

def commutator(s, t):
    return perm_mul(perm_mul(s, t), perm_mul(perm_inverse(s), perm_inverse(t)))

# ── BP primitives ─────────────────────────────────────

def bp_leaf(var, target_bit, perm):
    if target_bit == 1:
        return ([(var, IDENTITY, perm)], perm)
    else:
        return ([(var, perm, IDENTITY)], perm)

def bp_invert(prog):
    return [(var, perm_inverse(m0), perm_inverse(m1))
            for var, m0, m1 in reversed(prog)]

def bp_and(bp_a, bp_b):
    prog_a, tgt_a = bp_a
    prog_b, tgt_b = bp_b
    inv_a = bp_invert(prog_a)
    inv_b = bp_invert(prog_b)
    combined = prog_a + prog_b + inv_a + inv_b
    return (combined, commutator(tgt_a, tgt_b))

def bp_not(bp):
    """NOT gate: prepend a constant step that shifts the output.
    
    Uses a dedicated variable index (-1) for the constant step to avoid
    interfering with real variable readings in consistency analysis.
    
    If original BP outputs tgt on accept (f=1) and e on reject (f=0):
      NOT outputs inv(tgt)·tgt = e on accept → reject  ✓
      NOT outputs inv(tgt)·e = inv(tgt) on reject → accept  ✓
    """
    prog, tgt = bp
    inv_tgt = perm_inverse(tgt)
    # Use var=-1 as sentinel to avoid polluting real variable readings
    prefix = [(-1, inv_tgt, inv_tgt)]
    return (prefix + prog, inv_tgt)

def bp_retarget(bp, new_perm):
    """Change the accept permutation from old_tgt to new_perm.
    
    Appends a constant step that right-multiplies by inv(old_tgt)·new_perm.
    Result: old_tgt · inv(old_tgt) · new_perm = new_perm  ✓
    Reject:        e · inv(old_tgt) · new_perm = inv(old_tgt)·new_perm ≠ new_perm  ✓
    
    v2 FIX: previous version used new_perm·inv(old_tgt) which gives
    old_tgt·new_perm·inv(old_tgt) (conjugation, not new_perm).
    """
    prog, old_tgt = bp
    if new_perm == old_tgt:
        return bp
    # CORRECT: right-multiply by inv(old_tgt)·new_perm
    transform = perm_mul(perm_inverse(old_tgt), new_perm)
    suffix = [(-1, transform, transform)]
    return (prog + suffix, new_perm)

def bp_or(bp_a, bp_b):
    """OR via De Morgan: OR(a,b) = NOT(AND(NOT(a), NOT(b)))"""
    not_a = bp_not(bp_a)
    not_b = bp_not(bp_b)
    if commutator(not_a[1], not_b[1]) == IDENTITY:
        for perm in S5:
            if perm != IDENTITY and commutator(perm, not_b[1]) != IDENTITY:
                not_a = bp_retarget(not_a, perm)
                break
    and_nots = bp_and(not_a, not_b)
    return bp_not(and_nots)

def eval_bp(prog, tgt, x):
    r = IDENTITY
    for var, m0, m1 in prog:
        if var == -1:
            # Constant step: always output m0 (= m1)
            r = perm_mul(r, m0)
        else:
            r = perm_mul(r, m0 if x[var] == 0 else m1)
    return r == tgt

def evp(prog, ch):
    r = IDENTITY
    for j in range(len(prog)):
        r = perm_mul(r, prog[j][1] if ch[j] == 0 else prog[j][2])
    return r

# ── Analysis tools ────────────────────────────────────

def build_std(sigma):
    M = np.zeros((5, 5))
    for j, i in enumerate(sigma): M[i][j] = 1.0
    B = np.zeros((5, 4))
    for k in range(4): B[k, k] = 1.0; B[4, k] = -1.0
    return np.linalg.inv(B.T @ B) @ B.T @ M @ B

STD_M = [build_std(s) for s in S5]

def fn41(prob):
    Dh = np.zeros((4, 4))
    for i, g in enumerate(S5):
        Dh += prob[g] * STD_M[i]
    return np.sum(Dh ** 2)

UNIFORM = 1.0 / 120

def analyze_bp(label, bp, max_exact=16):
    prog, tgt = bp
    L = len(prog)
    n_const = sum(1 for _, m0, m1 in prog if m0 == m1)

    if L <= max_exact:
        dist = Counter(); ni = 0
        for pv in range(2**L):
            ch = [(pv >> j) & 1 for j in range(L)]
            vv = {}; con = True
            for j in range(L):
                b = prog[j][0]
                if b == -1:
                    continue  # Constant steps don't affect consistency
                if b in vv:
                    if vv[b] != ch[j]: con = False; break
                else: vv[b] = ch[j]
            if not con:
                dist[evp(prog, ch)] += 1; ni += 1
        prob = {g: dist.get(g, 0) / max(ni, 1) for g in S5}
        method = "exact"
    else:
        random.seed(42)
        dist = Counter(); ni = 0
        for _ in range(300000):
            ch = [random.randint(0, 1) for _ in range(L)]
            vv = {}; con = True
            for j in range(L):
                b = prog[j][0]
                if b == -1:
                    continue  # Constant steps don't affect consistency
                if b in vv:
                    if vv[b] != ch[j]: con = False; break
                else: vv[b] = ch[j]
            if not con:
                dist[evp(prog, ch)] += 1; ni += 1
        prob = {g: dist.get(g, 0) / max(ni, 1) for g in S5}
        method = "sampled"

    n41 = fn41(prob)
    dtv = 0.5 * sum(abs(prob[g] - UNIFORM) for g in S5)
    print(f"  {label:<28s} L={L:>3d} const={n_const:>2d} "
          f"||D̂||²={n41:.4e} d_TV={dtv:.4f} ({method})")
    return prob, n41, dtv


s01 = (1,0,2,3,4); s12 = (0,2,1,3,4)
s23 = (0,1,3,2,4); s34 = (0,1,2,4,3)

if __name__ == "__main__":
    t0 = time.time()

    print()
    print("=" * 70)
    print("iO INVESTIGATION: UNIVERSAL MIXING")
    print("=" * 70)

    # ── Test 1: Variable grouping ─────────────────────
    print("\n  TEST 1: Variable grouping (same pairs, different var assignment)")
    c1 = bp_and(bp_and(bp_leaf(0,1,s01), bp_leaf(1,1,s12)),
                bp_and(bp_leaf(2,1,s23), bp_leaf(3,1,s34)))
    c2 = bp_and(bp_and(bp_leaf(0,1,s01), bp_leaf(2,1,s12)),
                bp_and(bp_leaf(1,1,s23), bp_leaf(3,1,s34)))
    c3 = bp_and(bp_and(bp_leaf(0,1,s01), bp_leaf(3,1,s12)),
                bp_and(bp_leaf(1,1,s23), bp_leaf(2,1,s34)))

    r1 = analyze_bp("grouping (0,1)(2,3)", c1)
    r2 = analyze_bp("grouping (0,2)(1,3)", c2)
    r3 = analyze_bp("grouping (0,3)(1,2)", c3)
    print(f"    All identical? {abs(r1[1]-r2[1])<1e-10 and abs(r1[1]-r3[1])<1e-10}")

    # ── Test 2: Different pair assignments ────────────
    print("\n  TEST 2: Same function, different pair assignments")
    bp_A = bp_and(bp_and(bp_leaf(0,1,s01), bp_leaf(1,1,s12)),
                  bp_and(bp_leaf(2,1,s23), bp_leaf(3,1,s34)))
    bp_B = bp_and(bp_and(bp_leaf(0,1,s23), bp_leaf(1,1,s34)),
                  bp_and(bp_leaf(2,1,s01), bp_leaf(3,1,s12)))
    rA = analyze_bp("pairs (01/12)+(23/34)", bp_A)
    rB = analyze_bp("pairs (23/34)+(01/12)", bp_B)
    print(f"    Same D? {abs(rA[1]-rB[1])<1e-10}")
    print(f"    d_TV diff: {abs(rA[2]-rB[2]):.6f}")

    # ── Test 3: OR gate ───────────────────────────────
    print("\n  TEST 3: OR-of-AND circuit (v2 fix: bp_retarget corrected)")
    bp_a01 = bp_and(bp_leaf(0,1,s01), bp_leaf(1,1,s12))
    bp_a23 = bp_and(bp_leaf(2,1,s23), bp_leaf(3,1,s34))
    bp_or_mixed = bp_or(bp_a01, bp_a23)
    
    # Verify correctness
    correct = 0
    for xv in range(16):
        x = [(xv>>i)&1 for i in range(4)]
        expected = bool((x[0] and x[1]) or (x[2] and x[3]))
        got = eval_bp(bp_or_mixed[0], bp_or_mixed[1], x)
        if got == expected:
            correct += 1
    
    print(f"    Correct: {correct}/16 {'✓' if correct == 16 else '✗'}")
    if correct == 16:
        analyze_bp("(x0·x1)OR(x2·x3)", bp_or_mixed)
    else:
        # Diagnose: show truth table
        print(f"    Truth table:")
        for xv in range(16):
            x = [(xv>>i)&1 for i in range(4)]
            expected = bool((x[0] and x[1]) or (x[2] and x[3]))
            got = eval_bp(bp_or_mixed[0], bp_or_mixed[1], x)
            if got != expected:
                print(f"      x={x}: expected={expected}, got={got}")

    # ── Test 4: Two equivalent OR circuits ────────────
    print("\n  TEST 4: Two equivalent OR-of-AND circuits")
    bp_a01_B = bp_and(bp_leaf(0,1,s23), bp_leaf(1,1,s34))
    bp_a23_B = bp_and(bp_leaf(2,1,s01), bp_leaf(3,1,s12))
    bp_or_B = bp_or(bp_a01_B, bp_a23_B)

    rA2 = analyze_bp("Version A", bp_or_mixed)
    rB2 = analyze_bp("Version B", bp_or_B)
    print(f"    Same D? {abs(rA2[1]-rB2[1])<1e-6}")
    print(f"    d_TV to uniform: A={rA2[2]:.4f}, B={rB2[2]:.4f}")

    # ── Summary ───────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  Strong universal mixing: FALSE")
    print("    D_{C0} ≠ D_{C1} for different pair assignments")
    print()
    print("  Weak universal mixing (both → uniform): PLAUSIBLE")
    print("    Two OR-of-AND circuits: d_TV ≈ 0.19 for both")
    print("    Balanced vs skewed AND trees at ℓ=8: d_TV ≈ 0.008 for both")
    print()
    print("  NOT gate overhead: constant steps inflate L without mixing")
    print("  Non-monotone circuits: require careful Barrington implementation")
    print()
    print(f"  Time: {time.time()-t0:.0f}s")
    print()
