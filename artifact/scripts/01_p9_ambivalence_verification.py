"""
01_p9_ambivalence_verification.py
=================================
PROVES: The similarity attack is blind at ALL levels of the Barrington
tree, for ANY ℓ, ANY base permutations, ANY target T.

REASON: S_5 is ambivalent — every element is conjugate to its inverse.
Therefore π and π⁻¹ have identical cycle type, trace, characteristic
polynomial, and all similarity invariants.

Referenced in paper: Section 3.4.3 (Proposition: Universal Blindness),
                     Section 5.1
"""

from itertools import permutations
from collections import Counter
import time

IDENTITY = (0, 1, 2, 3, 4)

def perm_inverse(p):
    inv = [0]*5
    for i, j in enumerate(p): inv[j] = i
    return tuple(inv)

def perm_mul(a, b):
    return tuple(a[b[i]] for i in range(5))

def cycle_type(p):
    n, visited, cycles = len(p), [False]*len(p), []
    for i in range(n):
        if not visited[i]:
            l, j = 0, i
            while not visited[j]: visited[j] = True; j = p[j]; l += 1
            cycles.append(l)
    return tuple(sorted(cycles, reverse=True))

def trace_of(p):
    return sum(1 for i in range(len(p)) if p[i] == i)

def commutator(s, t):
    return perm_mul(perm_mul(s, t), perm_mul(perm_inverse(s), perm_inverse(t)))

def perm_to_matrix(perm):
    M = [[0]*5 for _ in range(5)]
    for j, i in enumerate(perm): M[i][j] = 1
    return M

# ── Barrington construction ──────────────────────────────────

class Leaf:
    def __init__(self, bit_idx, target_bit, perm):
        self.bit_idx = bit_idx; self.target_bit = target_bit; self.target = perm
    def flatten(self):
        if self.target_bit == 0: return [(self.bit_idx, self.target, IDENTITY)]
        else: return [(self.bit_idx, IDENTITY, self.target)]
    def with_target(self, new_perm):
        return Leaf(self.bit_idx, self.target_bit, new_perm)

class And:
    def __init__(self, left, right):
        self.left = left; self.right = right
        self.target = commutator(left.target, right.target)
    def flatten(self):
        li = self.left.with_target(perm_inverse(self.left.target))
        ri = self.right.with_target(perm_inverse(self.right.target))
        return self.left.flatten() + self.right.flatten() + li.flatten() + ri.flatten()
    def with_target(self, new_perm):
        if new_perm == perm_inverse(self.target): return And(self.right, self.left)
        elif new_perm == self.target: return And(self.left, self.right)
        else: raise ValueError()

def build_cc4(T):
    s01=(1,0,2,3,4); s12=(0,2,1,3,4); s23=(0,1,3,2,4); s34=(0,1,2,4,3)
    root = And(And(Leaf(0,T[0],s01), Leaf(1,T[1],s12)),
               And(Leaf(2,T[2],s23), Leaf(3,T[3],s34)))
    return root.flatten(), root.target


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("P9 VERIFICATION: Ambivalence of S_5")
    print("=" * 70)

    S5 = list(permutations(range(5)))
    t0 = time.time()

    # Test 1: cycle_type(g) = cycle_type(g⁻¹) for all g ∈ S_5
    fails_ct = sum(1 for g in S5 if cycle_type(g) != cycle_type(perm_inverse(g)))
    print(f"\n  cycle_type(g) = cycle_type(g⁻¹) for all g: {120-fails_ct}/120  {'✓' if fails_ct==0 else '✗'}")

    # Test 2: trace(g) = trace(g⁻¹) for all g ∈ S_5
    fails_tr = sum(1 for g in S5 if trace_of(g) != trace_of(perm_inverse(g)))
    print(f"  trace(g) = trace(g⁻¹) for all g:            {120-fails_tr}/120  {'✓' if fails_tr==0 else '✗'}")

    # Test 3: g conjugate to g⁻¹ for all g ∈ S_5
    fails_conj = 0
    for g in S5:
        gi = perm_inverse(g)
        if not any(perm_mul(P, perm_mul(g, perm_inverse(P))) == gi for P in S5):
            fails_conj += 1
    print(f"  g conjugate to g⁻¹ for all g:               {120-fails_conj}/120  {'✓' if fails_conj==0 else '✗'}")

    # Test 4: Blindness on actual Barrington program for all 16 targets, ℓ=4
    print(f"\n  Blindness on Barrington program (ℓ=4, all 16 targets):")
    all_blind = True
    for tv in range(16):
        T = [(tv >> i) & 1 for i in range(4)]
        prog, tgt = build_cc4(T)
        for j, (bidx, m0, m1) in enumerate(prog):
            if m0 != IDENTITY and m1 == IDENTITY: active = m0
            elif m0 == IDENTITY and m1 != IDENTITY: active = m1
            else: continue
            if cycle_type(active) != cycle_type(perm_inverse(active)):
                all_blind = False
    print(f"    All 16 targets × all 16 steps blind: {'✓' if all_blind else '✗'}")

    # Test 5: Conjugacy class table
    print(f"\n  Conjugacy classes of S_5:")
    print(f"    {'Cycle type':<16s} {'|C|':>5s} {'g=g⁻¹?':>8s} {'blind?':>7s}")
    seen = set()
    for g in S5:
        ct = cycle_type(g)
        if ct in seen: continue
        seen.add(ct)
        sz = sum(1 for p in S5 if cycle_type(p) == ct)
        print(f"    {str(ct):<16s} {sz:>5d} {'yes' if g==perm_inverse(g) else 'no':>8s} {'✓':>7s}")

    elapsed = time.time() - t0
    ok = fails_ct == 0 and fails_tr == 0 and fails_conj == 0 and all_blind
    print(f"\n  {'✓ ALL TESTS PASSED' if ok else '✗ FAILURES DETECTED'}  [{elapsed:.1f}s]")
    print()
