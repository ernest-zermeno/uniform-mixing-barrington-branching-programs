"""
02_p13_collision_exhaustive.py
==============================
DISCOVERS: 283 inconsistent paths produce π_accept for ℓ=4.
DISCOVERS: The collision count is EXACTLY 283 for ALL 16 targets.
DISCOVERS: The collision rate (0.43%) is below uniform (1/120 = 0.83%).

Referenced in paper: Section 3.5.7 (T-independence property),
                     Section 5.3
"""

from itertools import permutations
from collections import Counter, defaultdict
import time

IDENTITY = (0, 1, 2, 3, 4)

def perm_inverse(p):
    inv = [0]*5
    for i, j in enumerate(p): inv[j] = i
    return tuple(inv)

def perm_mul(a, b):
    return tuple(a[b[i]] for i in range(5))

def cycle_type(p):
    visited = [False]*5; cycles = []
    for i in range(5):
        if not visited[i]:
            l, j = 0, i
            while not visited[j]: visited[j] = True; j = p[j]; l += 1
            cycles.append(l)
    return tuple(sorted(cycles, reverse=True))

def commutator(s, t):
    return perm_mul(perm_mul(s, t), perm_mul(perm_inverse(s), perm_inverse(t)))

class Leaf:
    def __init__(self, bi, tb, pm):
        self.bit_idx=bi; self.target_bit=tb; self.target=pm
    def flatten(self):
        return [(self.bit_idx, self.target, IDENTITY)] if self.target_bit==0 else [(self.bit_idx, IDENTITY, self.target)]
    def with_target(self, np):
        return Leaf(self.bit_idx, self.target_bit, np)

class And:
    def __init__(self, l, r):
        self.left=l; self.right=r; self.target=commutator(l.target, r.target)
    def flatten(self):
        li=self.left.with_target(perm_inverse(self.left.target))
        ri=self.right.with_target(perm_inverse(self.right.target))
        return self.left.flatten()+self.right.flatten()+li.flatten()+ri.flatten()
    def with_target(self, np):
        if np==perm_inverse(self.target): return And(self.right, self.left)
        elif np==self.target: return And(self.left, self.right)
        else: raise ValueError()

def build_cc4(T):
    s01=(1,0,2,3,4); s12=(0,2,1,3,4); s23=(0,1,3,2,4); s34=(0,1,2,4,3)
    root = And(And(Leaf(0,T[0],s01), Leaf(1,T[1],s12)),
               And(Leaf(2,T[2],s23), Leaf(3,T[3],s34)))
    return root.flatten(), root.target

def eval_path(prog, choices):
    r = IDENTITY
    for j in range(len(prog)):
        r = perm_mul(r, prog[j][1] if choices[j]==0 else prog[j][2])
    return r


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("P13 EXHAUSTIVE: Collision Analysis for ℓ=4")
    print("=" * 70)

    S5 = list(permutations(range(5)))

    # ── All 16 targets ─────────────────────────────────
    print(f"\n  Exhaustive analysis for each of the 16 possible targets:")
    print(f"  {'T':<16s} {'N_exact':>8s} {'N_incon':>10s} {'rate':>10s} {'1/120':>10s}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    all_counts = []
    total_time = 0
    for tv in range(16):
        T = [(tv >> i) & 1 for i in range(4)]
        prog, tgt = build_cc4(T)
        L = len(prog)

        t0 = time.time()
        n_incon = 0; n_exact = 0
        output_dist = Counter()

        for pv in range(2**L):
            choices = [(pv >> j) & 1 for j in range(L)]
            var_vals = {}; consistent = True
            for j in range(L):
                bidx = prog[j][0]
                if bidx in var_vals:
                    if var_vals[bidx] != choices[j]: consistent = False; break
                else: var_vals[bidx] = choices[j]
            if not consistent:
                n_incon += 1
                output = eval_path(prog, choices)
                output_dist[output] += 1
                if output == tgt: n_exact += 1

        total_time += time.time() - t0
        rate = n_exact / n_incon
        all_counts.append(n_exact)
        print(f"  {str(T):<16s} {n_exact:>8d} {n_incon:>10d} {rate:>10.6f} {1/120:>10.6f}")

    print(f"\n  T-INDEPENDENCE CHECK:")
    print(f"    All collision counts: {set(all_counts)}")
    print(f"    All identical? {'YES ✓' if len(set(all_counts))==1 else 'NO ✗'}")
    print(f"    Collision rate: {all_counts[0]}/65520 = {all_counts[0]/65520:.6f}")
    print(f"    vs uniform 1/120 = {1/120:.6f}")
    print(f"    Ratio to uniform: {(all_counts[0]/65520)/(1/120):.4f}")

    # ── Distribution analysis for one target ───────────
    T = [1, 0, 1, 1]
    prog, tgt = build_cc4(T)
    L = len(prog)
    output_dist = Counter()
    n_incon = 0
    for pv in range(2**L):
        choices = [(pv >> j) & 1 for j in range(L)]
        var_vals = {}; consistent = True
        for j in range(L):
            bidx = prog[j][0]
            if bidx in var_vals:
                if var_vals[bidx] != choices[j]: consistent = False; break
            else: var_vals[bidx] = choices[j]
        if not consistent:
            n_incon += 1
            output_dist[eval_path(prog, choices)] += 1

    n_distinct = len(output_dist)
    expected = n_incon / 120
    counts = list(output_dist.values())
    chi2 = sum((c - expected)**2 / expected for c in counts) + (120 - n_distinct) * expected

    print(f"\n  Output distribution (T=[1,0,1,1]):")
    print(f"    Distinct permutations: {n_distinct}/120")
    print(f"    Expected per element (uniform): {expected:.1f}")
    print(f"    Actual range: [{min(counts)}, {max(counts)}]")
    print(f"    Chi-squared: {chi2:.1f} (df=119, critical@0.05=146.6)")
    print(f"    Distribution {'≈ uniform' if chi2 < 146.6 else '≠ uniform'}")

    # ── Anatomy of collision paths ─────────────────────
    collisions = []
    for pv in range(2**L):
        choices = [(pv >> j) & 1 for j in range(L)]
        var_vals = {}; consistent = True
        for j in range(L):
            bidx = prog[j][0]
            if bidx in var_vals:
                if var_vals[bidx] != choices[j]: consistent = False; break
            else: var_vals[bidx] = choices[j]
        if not consistent and eval_path(prog, choices) == tgt:
            collisions.append(choices)

    n_incon_vars = Counter()
    for choices in collisions:
        var_assignments = defaultdict(set)
        for j in range(L): var_assignments[prog[j][0]].add(choices[j])
        n_incon_vars[sum(1 for v in var_assignments.values() if len(v) > 1)] += 1

    print(f"\n  Anatomy of {len(collisions)} collision paths:")
    for n, count in sorted(n_incon_vars.items()):
        print(f"    {n} inconsistent variable(s): {count} paths")

    print(f"\n  Total time: {total_time:.1f}s")
    print()
