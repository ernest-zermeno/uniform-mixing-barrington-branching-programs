"""
16_conditioning_lemma_l4_verification.py
========================================
PROVES (exhaustively, exact rational arithmetic): the conditioning lemma that
bridges the unconditional Fourier analysis to the security-relevant conditioned
distribution D_T, for the normalized Barrington point-function template at l = 4.
Other targets are obtained from the same instruction template by branch swaps,
as in the T-independence proof.

Let, for a fixed target T,
  P_T = output distribution under L independent uniform bits  (UNCONDITIONAL),
  C_T = P_T conditioned on CONSISTENT paths,
  D_T = P_T conditioned on INCONSISTENT paths,
  p   = Pr[consistent].
Then by the law of total probability
  P_T = (1 - p) D_T + p C_T,
and by the triangle inequality + d_TV <= 1,
  d_TV(D_T, U) <= d_TV(P_T, U) + d_TV(D_T, P_T)
              = d_TV(P_T, U) + p . d_TV(D_T, C_T)
              <= d_TV(P_T, U) + p.
The paper bounds d_TV(P_T, U) (its Fourier transform factorizes into the leaf
projections, Theorems 1-4); this lemma transfers that bound to D_T at an additive
cost of p. For the point-function program every variable is read at least once, so
consistent paths are in bijection with {0,1}^l; with L = l^2,
  p = 2^l / 2^L = 2^{-l(l-1)}.

This script builds the balanced commutator program
  [[x1 ^ x2], [x3 ^ x4]]   (16 steps over S_5),
enumerates ALL 2^16 bit-paths, and verifies every constant and inequality above
in exact rational arithmetic.

Backs the conditioning lemma (replaces former Remark 5) and the Main-Theorem
additive term d_TV(D_T, U) <= 6 . lambda^{4^{d-3}} + 2^{-l(l-1)}.

Referenced in paper: Section 4 (conditioning lemma),
                     Section 5 (Main Theorem).
"""

from itertools import product, permutations
from fractions import Fraction as F
import sys

# ---- permutations of {0,1,2,3,4} -----------------------------------------
I = (0, 1, 2, 3, 4)
def mul(a, b):            # a o b  (apply b, then a)
    return tuple(a[b[i]] for i in range(5))
def inv(p):
    r = [0] * 5
    for i, j in enumerate(p):
        r[j] = i
    return tuple(r)
def transp(a, b):
    p = list(range(5)); p[a], p[b] = p[b], p[a]; return tuple(p)

# one transposition per variable (same assignment as Script 06)
SIGMA = {1: transp(0, 1), 2: transp(1, 2), 3: transp(2, 3), 4: transp(3, 4)}

# ---- build the normalized Barrington commutator template -----------------
# An instruction is a pair (variable, gen): on bit 1 it outputs `gen`, on bit 0
# it outputs the identity. This is the all-ones normalized target; other targets
# only swap the two branches at the corresponding literals.
def leaf(var):
    return [(var, SIGMA[var])]
def inv_block(block):                       # inverse program: reverse + invert
    return [(var, inv(g)) for (var, g) in reversed(block)]
def commutator(A, B):                       # [A,B] = A . B . A^-1 . B^-1
    return A + B + inv_block(A) + inv_block(B)

blockA  = commutator(leaf(1), leaf(2))      # alpha-computes x1 ^ x2
blockB  = commutator(leaf(3), leaf(4))      # alpha-computes x3 ^ x4
PROGRAM = commutator(blockA, blockB)        # (x1 ^ x2) ^ (x3 ^ x4)

L = len(PROGRAM)                            # 16
l = 4
step_var = [v for (v, _) in PROGRAM]
var_steps = {v: [j for j in range(L) if step_var[j] == v] for v in (1, 2, 3, 4)}

def output(bits):
    acc = I
    for j in range(L):
        if bits[j]:
            acc = mul(acc, PROGRAM[j][1])
    return acc

def consistent(bits):                       # all reads of each variable agree
    return all(len({bits[j] for j in var_steps[v]}) == 1 for v in (1, 2, 3, 4))

# ---- enumerate all 2^L paths ---------------------------------------------
S5 = list(permutations(range(5)))
cntP, cntC, cntD = {}, {}, {}
nC = 0
for bits in product((0, 1), repeat=L):
    g = output(bits)
    cntP[g] = cntP.get(g, 0) + 1
    if consistent(bits):
        cntC[g] = cntC.get(g, 0) + 1
        nC += 1
    else:
        cntD[g] = cntD.get(g, 0) + 1
nP = 2 ** L
nD = nP - nC
p = F(nC, nP)

def dist(cnt, tot):
    return {g: F(cnt.get(g, 0), tot) for g in S5}
P, C, D = dist(cntP, nP), dist(cntC, nC), dist(cntD, nD)
U = {g: F(1, 120) for g in S5}
def dtv(a, b):
    return sum(abs(a[g] - b[g]) for g in S5) / 2

# ---- checks ---------------------------------------------------------------
print("=" * 70)
print("CONDITIONING LEMMA — EXHAUSTIVE VERIFICATION (l = 4, all 2^16 paths)")
print("=" * 70)
ok = True
def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

check("L = l^2", L == l * l, f"L={L}, l^2={l*l}")
check("every variable read >= 1", all(len(var_steps[v]) >= 1 for v in (1, 2, 3, 4)),
      f"reads/var = {[len(var_steps[v]) for v in (1,2,3,4)]}")
check("|consistent| = 2^l", nC == 2 ** l, f"|consistent|={nC}, 2^l={2**l}")
check("p = 2^{-l(l-1)}", p == F(1, 2 ** (l * (l - 1))),
      f"p={p} = 2^-{l*(l-1)}")
mix_err = max(abs(P[g] - ((1 - p) * D[g] + p * C[g])) for g in S5)
check("mixture identity P = (1-p)D + pC exact", mix_err == 0, f"max|err|={mix_err}")
dDU, dPU, dDC, dDP = dtv(D, U), dtv(P, U), dtv(D, C), dtv(D, P)
check("d_TV(D,P) = p . d_TV(D,C)", dDP == p * dDC, f"{float(dDP):.3e} == {float(p*dDC):.3e}")
check("bound d_TV(D,U) <= d_TV(P,U) + p", dDU <= dPU + p,
      f"{float(dDU):.8f} <= {float(dPU + p):.8f}")
# faithfulness: a correct program outputs pi_accept iff all-match, else e
pi_accept = output((1,) * L)
check("C_T concentrated on {e, pi_accept}",
      set(cntC) == {I, pi_accept} and cntC[pi_accept] == 1 and cntC[I] == 2 ** l - 1,
      f"support sizes {sorted(cntC.values(), reverse=True)}, pi_accept={pi_accept}")

print("-" * 70)
print(f"  d_TV(D_T, U)   = {float(dDU):.8f}")
print(f"  d_TV(P_T, U)   = {float(dPU):.8f}   (Fourier-bounded in the paper)")
print(f"  d_TV(D_T, C_T) = {float(dDC):.8f}")
print(f"  p              = {p} = {float(p):.3e}")
print(f"  additive loss  d_TV(D,U) - d_TV(P,U) bounded by p = {float(p):.3e}")
print("=" * 70)
print(f"  RESULT: {'ALL CHECKS PASS' if ok else 'FAILURE'}")
print("=" * 70)
sys.exit(0 if ok else 1)
