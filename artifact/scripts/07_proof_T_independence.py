"""
07_proof_T_independence.py
===========================
PROVES: D_T = D_{T'} exactly for all T, T' ∈ {0,1}^ℓ.

The proof is 5 lines. The rest of this file is verification.

════════════════════════════════════════════════════════════════
THEOREM (Exact T-Independence of D_T).

Let BP_T be the Barrington branching program for the point function
CC_T over {0,1}^ℓ. Let D_T be the distribution over S_5 induced by
evaluating BP_T on a uniformly random L-bit string b = (b_1,...,b_L),
conditioned on b being inconsistent (i.e., at least one variable
receives different values at different steps).

Then D_T = D_{T'} for all T, T' ∈ {0,1}^ℓ, for programs built from
the same instruction template: the same read sequence and the same
transposition at each leaf, with T only swapping which branch receives it.

PROOF.

Let S = {i : T[i] ≠ T'[i]} be the set of differing bits. Define
F_S: {0,1}^L → {0,1}^L by flipping b_j at every step j that reads
a variable in S:

    F_S(b)_j = 1 - b_j  if i_j ∈ S
    F_S(b)_j = b_j       otherwise

where i_j is the variable read at step j.

We verify three properties:

(1) F_S is an involution (F_S ∘ F_S = id), hence a bijection.

(2) F_S preserves inconsistency: b is inconsistent iff F_S(b) is
    inconsistent. This is because F_S flips ALL readings of each
    variable in S simultaneously. If variable x_i (with i ∈ S) had
    all readings equal (say, all 0), after flipping they are all 1
    — still consistent. If some readings were 0 and some were 1,
    after flipping some are 1 and some are 0 — still inconsistent.
    Variables not in S are unaffected. Therefore inconsistency of
    each variable is preserved individually.

(3) φ_{T'}(b) = φ_T(F_S(b)) for all b. This is because at each
    step j reading variable x_{i_j}:
    - If i_j ∉ S: T[i_j] = T'[i_j], and F_S doesn't flip b_j.
      Same input, same T-bit, same output.
    - If i_j ∈ S: T'[i_j] = 1 - T[i_j], and F_S flips b_j.
      For T[i_j]=0: output_T(b_j) = σ if b_j=0, I if b_j=1.
      For T'[i_j]=1: output_{T'}(b_j) = I if b_j=0, σ if b_j=1
                                        = σ if (1-b_j)=0, I if (1-b_j)=1
                                        = output_T(F_S(b)_j).
    So the per-step outputs, and hence the product, are equal.

Combining (1)-(3):

    D_{T'}(g) = Pr[φ_{T'}(b) = g | b inconsistent]
              = Pr[φ_T(F_S(b)) = g | b inconsistent]         by (3)
              = Pr[φ_T(b') = g | F_S⁻¹(b') inconsistent]     substituting b' = F_S(b)
              = Pr[φ_T(b') = g | b' inconsistent]             by (2), since F_S = F_S⁻¹ by (1)
              = D_T(g)

Therefore D_T = D_{T'} for all T, T'.  □

REMARK. The proof does not use any property of S_5 — it works for
branching programs of ANY width over ANY group, provided the programs
for different targets share the same instruction template and differ
only by branch swaps at target-changing literals.

REMARK. The same proof shows that D_T is identical for the
distribution over ALL paths (not just inconsistent ones), since
F_S also preserves the set of consistent paths. The conditioning
on inconsistency is not needed for T-independence — it is needed
only for the security argument (consistent paths reveal T).

REMARK. The proof extends to conjunctions with wildcards: if some
positions have T[i] = *, the corresponding leaf outputs σ for BOTH
bit values. Flipping b_j at such positions changes both branches
identically (both are σ), so the output is unchanged. The proof
goes through with S restricted to non-wildcard differing positions.
════════════════════════════════════════════════════════════════

Referenced in paper: Section 3 (Exact T-Independence),
                     Section 6.1 (Conjunctions with Wildcards)
"""

from itertools import permutations
from collections import Counter, defaultdict
import random

IDENTITY = (0, 1, 2, 3, 4)

def perm_inverse(p):
    inv = [0]*5
    for i, j in enumerate(p): inv[j] = i
    return tuple(inv)

def perm_mul(a, b):
    return tuple(a[b[i]] for i in range(5))

def commutator(s, t):
    return perm_mul(perm_mul(s, t), perm_mul(perm_inverse(s), perm_inverse(t)))

class Leaf:
    def __init__(self, bi, tb, pm):
        self.bit_idx = bi; self.target_bit = tb; self.target = pm
    def flatten(self):
        return [(self.bit_idx, self.target, IDENTITY)] if self.target_bit == 0 else [(self.bit_idx, IDENTITY, self.target)]
    def with_target(self, np_):
        return Leaf(self.bit_idx, self.target_bit, np_)

class And:
    def __init__(self, l, r):
        self.left = l; self.right = r
        self.target = commutator(l.target, r.target)
    def flatten(self):
        li = self.left.with_target(perm_inverse(self.left.target))
        ri = self.right.with_target(perm_inverse(self.right.target))
        return self.left.flatten() + self.right.flatten() + li.flatten() + ri.flatten()
    def with_target(self, np_):
        if np_ == perm_inverse(self.target): return And(self.right, self.left)
        elif np_ == self.target: return And(self.left, self.right)
        else: raise ValueError()

def evp(prog, ch):
    r = IDENTITY
    for j in range(len(prog)):
        r = perm_mul(r, prog[j][1] if ch[j] == 0 else prog[j][2])
    return r

def build_prog(T, perms):
    """Build Barrington program for arbitrary ℓ = len(T)."""
    ell = len(T)
    leaves = [Leaf(i, T[i], perms[i % len(perms)]) for i in range(ell)]
    nodes = leaves
    while len(nodes) > 1:
        new = []
        for k in range(0, len(nodes), 2):
            if k+1 < len(nodes):
                new.append(And(nodes[k], nodes[k+1]))
            else:
                new.append(nodes[k])
        nodes = new
    return nodes[0].flatten(), nodes[0].target

# ── Standard construction for verification ────────────────────

s01 = (1,0,2,3,4); s12 = (0,2,1,3,4)
s23 = (0,1,3,2,4); s34 = (0,1,2,4,3)

def build4(T):
    root = And(And(Leaf(0,T[0],s01), Leaf(1,T[1],s12)),
               And(Leaf(2,T[2],s23), Leaf(3,T[3],s34)))
    return root.flatten(), root.target

P1 = [((1,0,2,3,4),(0,2,1,3,4)),((0,1,3,2,4),(0,1,2,4,3)),
      ((2,1,0,3,4),(0,1,4,3,2)),((3,1,2,0,4),(0,3,2,1,4))]

def build8(T, off=0):
    leaves = [Leaf(off+2*k+j, T[2*k+j], P1[k][j]) for k in range(4) for j in range(2)]
    l1 = [And(leaves[2*k], leaves[2*k+1]) for k in range(4)]
    l2 = [And(l1[2*k], l1[2*k+1]) for k in range(2)]
    root = And(l2[0], l2[1])
    return root.flatten(), root.target


def verify_theorem(ell, builder, n_pairs=20):
    """Exhaustively verify the T-independence theorem for given ℓ."""
    random.seed(42)

    for trial in range(n_pairs):
        T  = [random.randint(0,1) for _ in range(ell)]
        Tp = [random.randint(0,1) for _ in range(ell)]
        S  = [i for i in range(ell) if T[i] != Tp[i]]

        prog_T, _ = builder(T)
        prog_Tp, _ = builder(Tp)
        L = len(prog_T)

        # Build variable reading pattern
        var_reads = defaultdict(list)
        for j in range(L):
            var_reads[prog_T[j][0]].append(j)

        # Define F_S
        flip_set = set()
        for v in S:
            for j in var_reads[v]:
                flip_set.add(j)

        def flip(b):
            return tuple(1-b[j] if j in flip_set else b[j] for j in range(L))

        def is_inconsistent(b):
            vv = {}
            for j in range(L):
                v = prog_T[j][0]
                if v in vv:
                    if vv[v] != b[j]: return True
                else: vv[v] = b[j]
            return False

        # Verify all three properties for ALL 2^L paths
        prop1_ok = True  # F_S is involution
        prop2_ok = True  # F_S preserves inconsistency
        prop3_ok = True  # φ_{T'}(b) = φ_T(F_S(b))

        for pv in range(2**L):
            b = tuple((pv >> j) & 1 for j in range(L))
            fb = flip(b)
            ffb = flip(fb)

            if ffb != b:
                prop1_ok = False

            if is_inconsistent(b) != is_inconsistent(fb):
                prop2_ok = False

            if evp(prog_Tp, b) != evp(prog_T, fb):
                prop3_ok = False

        ok = prop1_ok and prop2_ok and prop3_ok
        if not ok or trial < 3 or trial >= n_pairs - 2:
            print(f"    T={T}, T'={Tp}")
            print(f"      S={S}, |flip|={len(flip_set)}")
            print(f"      (1) involution: {'✓' if prop1_ok else '✗'}")
            print(f"      (2) preserves incon: {'✓' if prop2_ok else '✗'}")
            tag = "T'"
            print(f"      (3) phi_{tag} = phi_T o F_S: {'✓' if prop3_ok else '✗'}")
        elif trial == 3:
            print(f"    ... ({n_pairs - 5} more pairs verified)")

        if not ok:
            return False
    return True


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("THEOREM: Exact T-Independence of D_T")
    print("Proof verification by exhaustive computation")
    print("=" * 70)

    print(f"\n  ℓ=4 (L=16, exhaustive over all 65536 paths):")
    ok4 = verify_theorem(4, build4, n_pairs=20)
    print(f"  Result: {'THEOREM VERIFIED ✓' if ok4 else 'FAILURE ✗'}")

    print(f"\n  ℓ=8 (L=64, exhaustive over all 2^64... sampling instead):")
    # For ℓ=8, we can't enumerate 2^64 paths. Instead verify the
    # distributional consequence: D_T = D_{T'} by sampling.
    n_samples = 200_000
    random.seed(42)

    T8a = [1,0,1,1,0,1,0,1]
    T8b = [0,1,0,0,1,0,1,0]  # complement
    prog_a, tgt_a = build8(T8a)
    prog_b, tgt_b = build8(T8b)
    L = len(prog_a)

    var_reads = defaultdict(list)
    for j in range(L): var_reads[prog_a[j][0]].append(j)
    S = list(range(8))  # all bits differ
    flip_set = set()
    for v in S:
        for j in var_reads[v]: flip_set.add(j)

    # Sample and compare distributions
    dist_a = Counter()
    dist_b = Counter()
    n_check = 0
    n_match = 0

    for _ in range(n_samples):
        b = tuple(random.randint(0,1) for _ in range(L))
        vv = {}; con = True
        for j in range(L):
            v = prog_a[j][0]
            if v in vv:
                if vv[v] != b[j]: con = False; break
            else: vv[v] = b[j]
        if con: continue

        fb = tuple(1-b[j] if j in flip_set else b[j] for j in range(L))

        out_Tp_b = evp(prog_b, b)
        out_T_Fb = evp(prog_a, fb)

        dist_a[evp(prog_a, b)] += 1
        dist_b[evp(prog_b, b)] += 1

        n_check += 1
        if out_Tp_b == out_T_Fb:
            n_match += 1

    print(f"    Property (3) verified on {n_match}/{n_check} samples "
          f"({'ALL ✓' if n_match == n_check else 'FAILURE ✗'})")

    # Compare distributions
    S5 = list(permutations(range(5)))
    total_a = sum(dist_a.values())
    total_b = sum(dist_b.values())
    max_diff = max(abs(dist_a.get(g,0)/total_a - dist_b.get(g,0)/total_b) for g in S5)
    print(f"    max|D_T(g) - D_T'(g)| = {max_diff:.6e}")
    print(f"    Distributions {'statistically identical ✓' if max_diff < 0.005 else 'differ ✗'}")

    print()
    print("=" * 70)
    print("PROOF SUMMARY")
    print("=" * 70)
    print()
    print("  The T-independence of D_T follows from a 5-line argument:")
    print()
    print("  1. Define F_S: flip b_j at all steps reading variables where T≠T'")
    print("  2. F_S is an involution (flipping twice = identity)")
    print("  3. F_S preserves inconsistency (flips ALL reads of each variable)")
    print("  4. φ_{T'}(b) = φ_T(F_S(b)) (swapping T-bit and flipping b cancels)")
    print("  5. By (2)-(4): D_{T'} = D_T via change of variables.  □")
    print()
    print("  This proof requires NO property of S_5 — it works for any group.")
    print("  It applies to point-function programs built from the same")
    print("  instruction template, not to arbitrary independently chosen programs.")
    print()
    print("  CONSEQUENCE: the fixed-template uniform-mixing question reduces to:")
    print("    sup_T d_TV(D_T, Uniform(S_5)) ≤ negl(ℓ)")
    print("  = d_TV(D_T, Uniform(S_5)) ≤ negl(ℓ)  for ANY fixed T")
    print("  since D_T is the same for all T.")
    print()
