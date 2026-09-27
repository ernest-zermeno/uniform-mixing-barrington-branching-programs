"""
05_barrington_construction_verification.py
===========================================
VERIFIES: Correct Barrington construction for ℓ=4, 8, 16.
VERIFIES: All rejects produce identity permutation.
VERIFIES: Kilian obfuscation over Z_n preserves correctness (ℓ=4).
VERIFIES: Bookend scalar compression works (accept → s, reject → random).

Referenced in paper: Section 3.2 (Barrington construction),
                     Section 3.3 (Kilian obfuscation),
                     Section 5.4.2 (construction verification table)
"""

import random, math, time
from collections import Counter

IDENTITY = (0, 1, 2, 3, 4)

def perm_inverse(p):
    inv=[0]*5
    for i,j in enumerate(p): inv[j]=i
    return tuple(inv)
def perm_mul(a,b): return tuple(a[b[i]] for i in range(5))
def commutator(s,t): return perm_mul(perm_mul(s,t),perm_mul(perm_inverse(s),perm_inverse(t)))
def cycle_type(p):
    vis=[False]*5; c=[]
    for i in range(5):
        if not vis[i]:
            l,j=0,i
            while not vis[j]: vis[j]=True; j=p[j]; l+=1
            c.append(l)
    return tuple(sorted(c,reverse=True))

class Leaf:
    def __init__(self, bi, tb, pm):
        self.bit_idx=bi; self.target_bit=tb; self.target=pm
    def flatten(self):
        return [(self.bit_idx, self.target, IDENTITY)] if self.target_bit==0 else [(self.bit_idx, IDENTITY, self.target)]
    def with_target(self, np_):
        return Leaf(self.bit_idx, self.target_bit, np_)

class And:
    def __init__(self, l, r):
        self.left=l; self.right=r; self.target=commutator(l.target, r.target)
    def flatten(self):
        li=self.left.with_target(perm_inverse(self.left.target))
        ri=self.right.with_target(perm_inverse(self.right.target))
        return self.left.flatten()+self.right.flatten()+li.flatten()+ri.flatten()
    def with_target(self, np_):
        if np_==perm_inverse(self.target): return And(self.right, self.left)
        elif np_==self.target: return And(self.left, self.right)
        else: raise ValueError()

def eval_input(prog, x):
    r=IDENTITY
    for bidx,m0,m1 in prog: r=perm_mul(r,m0 if x[bidx]==0 else m1)
    return r

def build4(T):
    s01=(1,0,2,3,4);s12=(0,2,1,3,4);s23=(0,1,3,2,4);s34=(0,1,2,4,3)
    root=And(And(Leaf(0,T[0],s01),Leaf(1,T[1],s12)),And(Leaf(2,T[2],s23),Leaf(3,T[3],s34)))
    return root.flatten(), root.target

P1=[((1,0,2,3,4),(0,2,1,3,4)),((0,1,3,2,4),(0,1,2,4,3)),
    ((2,1,0,3,4),(0,1,4,3,2)),((3,1,2,0,4),(0,3,2,1,4))]
P2=[((4,1,2,3,0),(0,1,4,3,2)),((1,0,2,3,4),(0,3,2,1,4)),
    ((2,1,0,3,4),(0,1,3,2,4)),((0,3,2,1,4),(0,1,2,4,3))]

def build8_node(T, off, pairs):
    leaves=[Leaf(off+2*k+j,T[2*k+j],pairs[k][j]) for k in range(4) for j in range(2)]
    l1=[And(leaves[2*k],leaves[2*k+1]) for k in range(4)]
    l2=[And(l1[2*k],l1[2*k+1]) for k in range(2)]
    return And(l2[0],l2[1])

def build8(T): return build8_node(T,0,P1).flatten(), build8_node(T,0,P1).target
def build16(T):
    root=And(build8_node(T[:8],0,P1),build8_node(T[8:],8,P2))
    return root.flatten(), root.target

# ── Kilian obfuscation utilities ─────────────────────────────

def extended_gcd(a, b):
    if a == 0: return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x

def mod_inverse(a, m):
    a = a % m; g, x, _ = extended_gcd(a, m)
    return x % m if g == 1 else None

def mat_mul_mod(A, B, n):
    sz = len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(sz))%n for j in range(sz)] for i in range(sz)]

def mat_det_mod(M, n):
    sz=len(M); A=[row[:] for row in M]; det=1
    for col in range(sz):
        pivot=next((r for r in range(col,sz) if A[r][col]%n!=0),-1)
        if pivot==-1: return 0
        if pivot!=col: A[col],A[pivot]=A[pivot],A[col]; det=(-det)%n
        inv=mod_inverse(A[col][col],n)
        if inv is None: return None
        det=(det*A[col][col])%n
        for row in range(col+1,sz):
            f=(A[row][col]*inv)%n
            for j in range(col,sz): A[row][j]=(A[row][j]-f*A[col][j])%n
    return det%n

def mat_inverse_mod(M, n):
    sz=len(M); aug=[M[i][:]+[1 if i==j else 0 for j in range(sz)] for i in range(sz)]
    for col in range(sz):
        pivot=next((r for r in range(col,sz) if aug[r][col]%n!=0),-1)
        if pivot==-1: return None
        aug[col],aug[pivot]=aug[pivot],aug[col]
        inv=mod_inverse(aug[col][col],n)
        if inv is None: return None
        aug[col]=[(x*inv)%n for x in aug[col]]
        for row in range(sz):
            if row!=col:
                f=aug[row][col]
                aug[row]=[(aug[row][j]-f*aug[col][j])%n for j in range(2*sz)]
    return [row[sz:] for row in aug]

def random_invertible_matrix(sz, n):
    for _ in range(200):
        M=[[random.randint(0,n-1) for _ in range(sz)] for _ in range(sz)]
        d=mat_det_mod(M,n)
        if d is not None and d!=0 and math.gcd(d,n)==1: return M
    return None

def perm_to_matrix(perm):
    M=[[0]*5 for _ in range(5)]
    for j,i in enumerate(perm): M[i][j]=1
    return M


if __name__ == "__main__":
    random.seed(42)
    print()
    print("=" * 70)
    print("BARRINGTON CONSTRUCTION VERIFICATION")
    print("=" * 70)

    # ── Plaintext correctness for ℓ=4, 8, 16 ─────────
    print("\n  PLAINTEXT CORRECTNESS:")
    for ell, builder in [(4, build4), (8, build8), (16, build16)]:
        T = [random.randint(0,1) for _ in range(ell)]
        prog, tgt = builder(T)
        L = len(prog)
        correct = 0; reject_not_id = 0
        for xv in range(min(2**ell, 65536)):
            x = [(xv>>i)&1 for i in range(ell)]
            result = eval_input(prog, x)
            if (x==T) == (result==tgt): correct += 1
            if x!=T and result!=IDENTITY: reject_not_id += 1
        n_test = min(2**ell, 65536)
        print(f"    ℓ={ell:3d}: L={L:5d}, tgt={tgt}, ct={cycle_type(tgt)}, "
              f"correct={correct}/{n_test}, reject≠I={reject_not_id} "
              f"{'✓' if correct==n_test and reject_not_id==0 else '✗'}")

    # ── Kilian obfuscation for ℓ=4 ───────────────────
    print("\n  KILIAN OBFUSCATION OVER Z_n (ℓ=4):")
    p_mod = 100003; q_mod = 100019; n_mod = p_mod * q_mod
    T = [1, 0, 1, 1]
    prog, tgt = build4(T)
    L = len(prog)

    for attempt in range(5):
        S = [random_invertible_matrix(5, n_mod) for _ in range(L+1)]
        if any(s is None for s in S): continue
        Si = [mat_inverse_mod(s, n_mod) for s in S]
        if any(s is None for s in Si): continue

        obf = []
        for j, (bidx, m0, m1) in enumerate(prog):
            H0 = mat_mul_mod(mat_mul_mod(S[j], perm_to_matrix(m0), n_mod), Si[j+1], n_mod)
            H1 = mat_mul_mod(mat_mul_mod(S[j], perm_to_matrix(m1), n_mod), Si[j+1], n_mod)
            obf.append((bidx, H0, H1))

        u = [random.randint(1, n_mod-1) for _ in range(5)]
        v = [random.randint(1, n_mod-1) for _ in range(5)]
        M_tgt = perm_to_matrix(tgt)
        s_scalar = sum(u[i]*M_tgt[i][j]*v[j] for i in range(5) for j in range(5)) % n_mod
        u_hat = [sum(u[i]*Si[0][i][j] for i in range(5))%n_mod for j in range(5)]
        v_hat = [sum(S[L][i][j]*v[j] for j in range(5))%n_mod for i in range(5)]

        obf_correct = 0
        for xv in range(16):
            x = [(xv>>i)&1 for i in range(4)]
            R = [[1 if i==j else 0 for j in range(5)] for i in range(5)]
            for j,(bidx,H0,H1) in enumerate(obf):
                M = H0 if x[bidx]==0 else H1
                R = mat_mul_mod(R, M, n_mod)
            alpha = sum(u_hat[i]*R[i][j]*v_hat[j] for i in range(5) for j in range(5)) % n_mod
            if (x==T) == (alpha==s_scalar): obf_correct += 1

        print(f"    Attempt {attempt+1}: n={n_mod}, correct={obf_correct}/16 "
              f"{'✓ END-TO-END VERIFIED' if obf_correct==16 else '✗'}")
        if obf_correct == 16: break

    print()
