# Exact rational certificate for the level-3 contraction (Theorem `thm:level23`)

**Script:** [`scripts/20_level3_exact_certificate.py`](scripts/20_level3_exact_certificate.py)
**Status:** PASS — all five non-trivial, non-sign irreps certified in exact integer arithmetic.
**Closes:** pre-resubmission check #2 in the repository `README.md` ("Level-3 certificate hardening").

## What is certified

For **every** valid level-3 configuration of the fixed $S_5$ transposition template
— the full ordered universe of Scripts 17/18: $60$ ordered adjacent level-1 pairs
$\to 3{,}240$ ordered level-2 blocks $\to 9{,}447{,}840$ oriented pairs
($= 2 \times 4{,}723{,}920$ unordered pairs) — and **every** non-trivial, non-sign
irreducible representation $\rho \in \{(4,1),\,(3,2),\,(3,1,1),\,(2,2,1),\,(2,1,1,1)\}$ of $S_5$:

$$\left\lVert \hat D_3(\rho) \right\rVert_{\mathrm{op}} \;\le\; \lambda \;=\; \frac{71}{2000} \;=\; 0.0355 ,$$

with **zero floating-point operations in the certification chain** (floats appear only in
clearly marked sanity cross-checks against the Script-17/18 pipeline). This upgrades the
decisive level-3 bound of Theorem `thm:level23` from a `float64` verification (Scripts 17/18)
to an exact rational certificate, following the Gram-rational + principal-minors pattern of
the companion P2A certificate (`s4_internal_gap_certify.py`).

In addition, a second exact run on all four carrier modules certifies the **tight** bound
$\lVert \hat D_3(\rho)\rVert \le 354002321/10^{10} = 0.0354002321$ for every relevant
$\rho$ and every configuration. The bottleneck configuration's largest singular value is
enclosed rigorously, giving the exact bracket for the global maximum (see below).

## Method

### 1. Orbit reduction with exact accounting

The symmetry group $H = S_5 \times \mathbb{Z}_2$ acts on oriented level-3 configurations and
preserves the operator norm:

* **Simultaneous conjugation** by $s \in S_5$ maps each template transposition $t \mapsto sts^{-1}$;
  since $P_{sts^{-1}} = \rho(s) P_t \rho(s)^{-1}$, it maps
  $\hat D_3 \mapsto \rho(s)\,\hat D_3\,\rho(s)^{-1}$ with $\rho(s)$ a $G$-isometry — same $G$-operator norm.
* **Swap** of the two level-2 blocks maps $\hat D_3 \mapsto \hat D_3^{*}$ (the $G$-adjoint),
  which has the same operator norm.

Orbit census (computed exactly, all sums verified by the script):

| object | count |
|---|---|
| $S_5$-orbits on the $3{,}240$ level-2 blocks | $27$, all of size $120$ (trivial stabilizers) |
| $S_5$-orbits on valid **ordered** level-3 pairs | $78{,}732$, all of size $120$ |
| $H$-orbits ($S_5 \times$ swap) | $\mathbf{39{,}591}$ representatives ($450$ swap-self-paired) |

**Exact accounting** (asserted by the script, two independent ways — degree sums over block
orbits, and the sum of pair-orbit sizes):

$$78{,}732 \times 120 \;=\; 450 \times 120 + 39{,}141 \times 240 \;=\; 9{,}447{,}840 . \checkmark$$

Each of the $39{,}591$ representatives is certified; by the two invariances above this
certifies all $9{,}447{,}840$ oriented configurations. As an end-to-end check of the
reduction, the script additionally certifies $200$ random **raw** (non-reduced) oriented
configurations directly, with the same exact arithmetic.

### 2. Integer carriers for the five irreps

Instead of irrational orthonormal bases, each irrep is certified inside an **integer**
permutation module with an invariant **integer** Gram matrix $G$:

| module | dim | construction | Gram $G$ | contains |
|---|---|---|---|---|
| `M41` | 4 | sum-zero subspace of the 5-point permutation module, basis $b_i = e_i - e_4$ | $I+J$ | $(4,1)$ |
| `M9`  | 9 | sum-zero subspace of the module on the ten 2-subsets of $\{0..4\}$ | $I+J$ | $(4,1) \oplus (3,2)$ |
| `M9s` | 9 | `M9` twisted by sign: $P'_t = (I - \rho(t))/2$ (as $\mathrm{sgn}(t) = -1$) | $I+J$ | $(2,1,1,1) \oplus (2,2,1)$ |
| `M6`  | 6 | $\Lambda^2$ of `M41` | $\Lambda^2(I+J)$ | $(3,1,1)$ |

$\hat D_3$ is the image of a fixed element of the group algebra $\mathbb{Q}[S_5]$ (a word in
the projections $P_t = \tfrac12(e + t)$; each $P_t$ is $G$-self-adjoint, so all adjoints are
*reversed words* — no $G^{-1}$ ever enters the computation). Hence $\hat D_3$ preserves every
isotypic component of a module, distinct isotypic components are $G$-orthogonal, and the
$G$-operator norm on a module equals the **max** of the norms on its irreducible constituents.
By Schur's lemma the invariant inner product on an irrep is unique up to a positive scalar, so
these norms are basis-independent. Certifying the four modules therefore certifies all five
irreps. (Both decompositions here are multiplicity-free.)

The script verifies **exactly** (integer identities): idempotence $P_t^2 = P_t$,
$G$-self-adjointness $P_t^{\mathsf T} G = G P_t$, $G$-orthogonality
$\rho^{\mathsf T} G \rho = G$, and the conjugation covariance
$\rho(s) P_t = P_{sts^{-1}} \rho(s)$.

### 3. Integer Gram-form certificate (P2A pattern)

In every module all $P_t$ have entries in $\tfrac12\mathbb{Z}$, so
$D_1 = (P_a P_b)^2$ has denominator $2^4$, $D_2 = D_{1,A} D_{1,B} D_{1,A}^{*} D_{1,B}^{*}$
has denominator $2^{16}$, and $D_3 = D_{2,i} D_{2,j} D_{2,i}^{*} D_{2,j}^{*}$ has denominator
$2^{64}$. Writing $D_3 = N/2^{64}$ with $N$ an **integer** matrix:

$$\lVert D_3 \rVert_G \le \frac{p}{q} \iff \frac{p^2}{q^2} G - D_3^{\mathsf T} G D_3 \succeq 0 \iff C := p^2\, 2^{128}\, G \;-\; q^2\, N^{\mathsf T} G N \succeq 0,$$

where $C$ is an **integer** matrix. $C$ is certified **positive definite** via Sylvester's
criterion: all leading principal minors, computed exactly by fraction-free Bareiss
elimination over $\mathbb{Z}$, are strictly positive. (A PSD fallback over all principal
minors is implemented; it was never triggered — every certificate matrix was strictly PD.)

## Results

All runs on the $39{,}591$ orbit representatives. "Min minor margin" is the smallest leading
principal minor of $\frac{p^2}{q^2}G - D_3^{\mathsf T} G D_3$ encountered (exact fraction in
the script; float shown here). All margins are strictly positive **exact** rationals.

| module | irreps covered | bound | result | min minor margin | time |
|---|---|---|---|---|---|
| `M41` | $(4,1)$ | $71/2000$ | **PASS** | $7.079 \times 10^{-14}$ | 2.4 s |
| `M9`  | $(4,1) \oplus (3,2)$ | $71/2000$ | **PASS** | $4.501 \times 10^{-28}$ | 26.7 s |
| `M9s` | $(2,1,1,1) \oplus (2,2,1)$ | $71/2000$ | **PASS** | $8.019 \times 10^{-26}$ | 24.4 s |
| `M6`  | $(3,1,1)$ | $71/2000$ | **PASS** | $5.008 \times 10^{-16}$ | 6.6 s |
| `M41` (tight) | $(4,1)$ | $354002321/10^{10}$ | **PASS** | $1.129 \times 10^{-20}$ | 2.5 s |
| `M9` (tight) | $(4,1) \oplus (3,2)$ | $354002321/10^{10}$ | **PASS** | $6.977 \times 10^{-35}$ | 33.4 s |
| `M9s` (tight) | $(2,1,1,1) \oplus (2,2,1)$ | $354002321/10^{10}$ | **PASS** | $7.623 \times 10^{-26}$ | 32.2 s |
| `M6` (tight) | $(3,1,1)$ | $354002321/10^{10}$ | **PASS** | $4.842 \times 10^{-16}$ | 7.7 s |

### Exact enclosure of the bottleneck maximum

For the bottleneck representative (the configuration attaining the float64 maximum), with
$S = N^{\mathsf T} G N$ in `M41`, the integer characteristic polynomial of the pencil is

$$\det(y\,G - S) \;=\; 5y^4 \;-\; 2132170043845660805587982652288532480\, y^3 \;+\; 354919251610376337755610189956763641856298691075034341496061952000\, y^2$$

(the $y^0, y^1$ coefficients vanish exactly: this $\hat D_3$ has rank $2$). A Sturm chain
proves the polynomial has exactly one root in the bisection bracket and none above it up to
the tight bound; exact rational bisection then gives the certified enclosure
($\sigma_{\max}^2 = y_{\max}/2^{128}$):

$$\frac{88500580209497127671039}{2.5 \times 10^{24}} \;\le\; \sigma_{\max} \;\le\; \frac{354002320837988510684157}{10^{25}},$$

i.e. $\sigma_{\max} = 0.0354002320837988510684156\ldots$ (enclosure width $10^{-25}$).

Combined with the tight all-irrep certificate ($\le 0.0354002321$ for *all* configurations), the
**global** level-3 maximum over all $9{,}447{,}840$ oriented configurations and all five
irreps lies in the exact bracket

$$\boxed{\;0.035400232083798851\ldots \;\le\; \max \lVert \hat D_3 \rVert_{\mathrm{op}} \;\le\; 0.0354002321\;}$$

and the certified paper bound $\lambda = 71/2000 = 0.0355$ holds globally with exact squared margin

$$\lambda^2 - \max\lVert\hat D_3\rVert_{\mathrm{op}}^2
\;\ge\; \frac{707356726612959}{10^{20}} \;\approx\; 7.0736 \times 10^{-6}.$$

### Consistency with the float64 scans (Scripts 17/18)

* Float64 maxima per irrep recomputed over the orbit representatives (sanity section):
  $(4,1)\;0.0354002320837989$, $(3,2)\;1.9293 \times 10^{-5}$, $(3,1,1)\;1.0475 \times 10^{-5}$,
  $(2,2,1) \approx 4 \times 10^{-16}$, $(2,1,1,1) \approx 10^{-17}$ — matching Script 18.
* The exact integer chain agrees with the Script-18 orthonormal float chain to
  $6.3 \times 10^{-17}$ on a random sample of representatives and modules.
* **Precision note:** the certified bottleneck configuration has norm
  $0.03540023208379885106\ldots$; the float64 value $0.03540023208379892$ printed by
  Scripts 17/18 carries $\sim 10$ ulps of accumulated SVD rounding and is correct to 15
  significant digits. The exact all-irrep certificate, not those rounded digits, proves the
  paper's conservative $\lambda = 0.0355$ bound.

## Reproduce

```bash
cd completo-iacr/scripts
python3 20_level3_exact_certificate.py
```

Runtime: **~145 s** single-threaded (Python 3.9, standard library + `numpy`; `numpy` is used
*only* in the float sanity section — the certification chain is pure exact integer
arithmetic). Expected final line: `LEVEL-3 EXACT RATIONAL CERTIFICATE: PASS`.

## Scope and honest caveats

1. **What is exact:** the orbit reduction and its accounting, all module sanity identities,
   the PD certificates (both $\lambda = 71/2000$ and the tight bound on all five irreps),
   the characteristic polynomial, the Sturm root isolation, and the enclosure/margins. These
   involve no floating point.
2. **What floats are used for:** *finding* the bottleneck candidate and cross-checking
   against Scripts 17/18. The identification of the specific representative as the global
   argmax is float-guided; the *exact* statements are the two-sided bracket above (a lower
   bound from one explicitly certified configuration, an upper bound certified for **all**
   configurations), which pins the global maximum to within $1.6 \times 10^{-11}$.
3. The trivial representation (norm $1$, the stationary component) and the sign
   representation ($P_t = 0$, so $\hat D_3 = 0$) are excluded exactly as in
   Theorem `thm:level23` and Script 18.
4. Scripts 17/18 remain in the artifact as the independent float64 cross-checks; this
   certificate supersedes them as the *proof-grade* verification of Theorem `thm:level23`.
