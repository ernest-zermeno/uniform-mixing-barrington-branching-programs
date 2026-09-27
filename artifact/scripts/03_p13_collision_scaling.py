"""
03_p13_collision_scaling.py
===========================
DISCOVERS: Collision rate converges from 0.38% (ℓ=4) to 0.83% ≈ 1/120 (ℓ=8).
CONFIRMS:  The distribution approaches uniform over S_5 as tree depth grows.
CONFIRMS:  Rate is T-independent across multiple targets for ℓ=8.

Referenced in paper: Section 3.5.7 (collision rate table),
                     Section 5.3
"""

from itertools import permutations
from collections import Counter
import random, time

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

def build_cc8(T, off=0):
    pairs=[((1,0,2,3,4),(0,2,1,3,4)),((0,1,3,2,4),(0,1,2,4,3)),
           ((2,1,0,3,4),(0,1,4,3,2)),((3,1,2,0,4),(0,3,2,1,4))]
    leaves = [Leaf(off+2*k+j, T[2*k+j], pairs[k][j]) for k in range(4) for j in range(2)]
    l1 = [And(leaves[2*k], leaves[2*k+1]) for k in range(4)]
    l2 = [And(l1[2*k], l1[2*k+1]) for k in range(2)]
    root = And(l2[0], l2[1])
    return root.flatten(), root.target

def eval_path(prog, choices):
    r = IDENTITY
    for j in range(len(prog)):
        r = perm_mul(r, prog[j][1] if choices[j]==0 else prog[j][2])
    return r

def eval_input(prog, x):
    r = IDENTITY
    for bidx, m0, m1 in prog:
        r = perm_mul(r, m0 if x[bidx]==0 else m1)
    return r

def sample_collision_rate(prog, tgt, n_samples):
    L = len(prog); n_incon = 0; n_coll = 0
    for _ in range(n_samples):
        ch = [random.randint(0,1) for _ in range(L)]
        vv = {}; con = True
        for j in range(L):
            b = prog[j][0]
            if b in vv:
                if vv[b] != ch[j]: con = False; break
            else: vv[b] = ch[j]
        if not con:
            n_incon += 1
            if eval_path(prog, ch) == tgt: n_coll += 1
    return n_coll, n_incon


if __name__ == "__main__":
    random.seed(42)
    print()
    print("=" * 70)
    print("P13 SCALING: Collision Rate from ℓ=4 to ℓ=8")
    print("=" * 70)

    # ℓ=4 exhaustive (reference)
    print(f"\n  ℓ=4 (exhaustive):")
    T4 = [1, 0, 1, 1]
    prog4, tgt4 = build_cc4(T4)
    L4 = len(prog4)
    n_incon4 = 0; n_coll4 = 0
    for pv in range(2**L4):
        ch = [(pv>>j)&1 for j in range(L4)]
        vv = {}; con = True
        for j in range(L4):
            b = prog4[j][0]
            if b in vv:
                if vv[b] != ch[j]: con = False; break
            else: vv[b] = ch[j]
        if not con:
            n_incon4 += 1
            if eval_path(prog4, ch) == tgt4: n_coll4 += 1
    rate4 = n_coll4 / n_incon4
    print(f"    N_exact={n_coll4}, rate={rate4:.6f}, 1/120={1/120:.6f}, ratio={rate4/(1/120):.4f}")

    # ℓ=8 sampled, multiple targets
    print(f"\n  ℓ=8 (sampled, 500K per target, 8 targets):")
    n_samples = 500_000
    t0 = time.time()
    rates_8 = []
    for trial in range(8):
        T8 = [random.randint(0,1) for _ in range(8)]
        prog8, tgt8 = build_cc8(T8)
        # Verify correctness
        c8 = sum(1 for xv in range(256) for x in [[(xv>>i)&1 for i in range(8)]]
                 if (eval_input(prog8, x) == tgt8) == (x == T8))
        n_coll, n_incon = sample_collision_rate(prog8, tgt8, n_samples)
        rate = n_coll / n_incon
        rates_8.append(rate)
        print(f"    T={T8}: correct={c8}/256, collisions={n_coll}/{n_incon}, "
              f"rate={rate:.6f}, ratio={rate/(1/120):.4f}")

    import numpy as np
    mean8 = np.mean(rates_8)
    std8 = np.std(rates_8)
    print(f"\n    Mean rate: {mean8:.6f} ± {std8:.6f}")
    print(f"    CV (spread/mean): {std8/mean8:.4f}")
    print(f"    T-independent? {'YES (CV < 0.1)' if std8/mean8 < 0.1 else 'INVESTIGATE'}")

    print(f"\n  SCALING SUMMARY:")
    print(f"    ℓ=4:  rate = {rate4:.6f}  (ratio to 1/120: {rate4/(1/120):.4f})")
    print(f"    ℓ=8:  rate = {mean8:.6f}  (ratio to 1/120: {mean8/(1/120):.4f})")
    print(f"    Convergence to 1/120: {'CONFIRMED' if abs(mean8 - 1/120) < 0.002 else 'PARTIAL'}")
    print(f"    Time: {time.time()-t0:.0f}s")
    print()
