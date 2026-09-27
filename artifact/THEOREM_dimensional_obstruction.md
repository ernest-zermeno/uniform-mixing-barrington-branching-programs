# Theorem: Dimensional Obstruction to Spectral Contraction in S_n for n >= 6

## Statement

**Theorem.** Let $\rho_{(n-1,1)}$ be the standard representation of $S_n$ on the
hyperplane $V = \{v \in \mathbb{R}^n : \sum v_i = 0\}$ (dimension $n-1$). For any
pair of adjacent transpositions $\sigma = (a\;b)$ and $\tau = (b\;c)$ in $S_n$,
define the level-1 Fourier transform

$$D_1(\sigma, \tau) = (P_\sigma P_\tau)^2, \qquad P_\sigma = \frac{\rho(\sigma) + I}{2}.$$

Then $D_1$ has singular values

$$\underbrace{1, \ldots, 1}_{n-3}, \quad \frac{1}{8}, \quad 0.$$

**Corollary.** For $n \geq 6$, the level-2 commutator $D_2$ has operator norm $1$
for *every* valid quadruple of transpositions, so the simplex/direction-mixing
proof strategy of the main paper is specific to $S_5$ and does not extend to
$S_n$ for $n \geq 6$. At level 3 the maximum is 1 in $S_6,S_7$, but
some configurations contract. For $n \geq 10$, every level-3 configuration
has norm 1 by the dimensional argument. The analysis here does not decide
$S_8,S_9$ or uniform contraction at greater depths.


## Proof

### Step 1: Null vectors of the projections

The transposition $\sigma = (a\;b)$ acts on $V$ with eigenvalues $\{+1, -1\}$.
The $(-1)$-eigenspace is spanned by

$$v_\sigma = \frac{e_a - e_b}{\sqrt{2}} \in V.$$

Since $P_\sigma = \frac{1}{2}(\rho(\sigma) + I)$ projects onto the $(+1)$-eigenspace:

$$P_\sigma = I_V - v_\sigma v_\sigma^T, \qquad \text{rank}(P_\sigma) = n-2.$$

Similarly for $\tau = (b\;c)$:

$$v_\tau = \frac{e_b - e_c}{\sqrt{2}}, \qquad P_\tau = I_V - v_\tau v_\tau^T.$$

### Step 2: The inner product is universal

$$\langle v_\sigma, v_\tau \rangle = \frac{1}{2}\langle e_a - e_b, e_b - e_c \rangle = \frac{1}{2}(0 - 1 - 0 + 0) = -\frac{1}{2}.$$

This value is **independent of $n$** and depends only on the adjacency structure
(sharing exactly one element).

### Step 3: Block decomposition $V = W \oplus W^\perp$

Define $W = \text{span}(v_\sigma, v_\tau)$, which is 2-dimensional since
$|\langle v_\sigma, v_\tau \rangle| = 1/2 \neq 1$.

- **On $W^\perp$ (dimension $n-3$):** Both $P_\sigma$ and $P_\tau$ act as the
  identity (since $\langle w, v_\sigma \rangle = \langle w, v_\tau \rangle = 0$
  for $w \in W^\perp$). Therefore $D_1 = (P_\sigma P_\tau)^2 = I$ on $W^\perp$,
  contributing **$n-3$ singular values equal to $1$**.

- **On $W$ (dimension 2):** Both projections map $W$ into $W$. The restriction
  $D_1|_W$ is a $2 \times 2$ problem.

### Step 4: The 2x2 block (independent of n)

Orthonormalize: let $f_1 = v_\sigma$ and

$$f_2 = \frac{v_\tau + \frac{1}{2}v_\sigma}{\|v_\tau + \frac{1}{2}v_\sigma\|} = \frac{v_\tau + \frac{1}{2}v_\sigma}{\sqrt{3}/2}.$$

In the basis $(f_1, f_2)$:

$$P_\sigma|_W = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}, \qquad
P_\tau|_W = \begin{pmatrix} 3/4 & \sqrt{3}/4 \\ \sqrt{3}/4 & 1/4 \end{pmatrix}.$$

Computing the product and its square:

$$D_1|_W = (P_\sigma P_\tau)^2|_W = \begin{pmatrix} 0 & 0 \\ \sqrt{3}/16 & 1/16 \end{pmatrix}.$$

The matrix $D_1|_W^T D_1|_W$:

$$D_1|_W^T D_1|_W = \frac{1}{256}\begin{pmatrix} 3 & \sqrt{3} \\ \sqrt{3} & 1 \end{pmatrix}.$$

This has trace $= 4/256 = 1/64$ and determinant $= (3 - 3)/256^2 = 0$.

**Eigenvalues: $1/64$ and $0$. Singular values: $1/8$ and $0$.** $\square$

### Step 5: Dimensional obstruction for n >= 6

$D_1(\sigma, \tau)$ preserves $W^\perp(\sigma, \tau)$ isometrically (all singular
values $= 1$ there). This subspace has dimension $n-3$.

For a level-2 commutator using two level-1 blocks $A$ and $B$:

$$\dim(W_A^\perp \cap W_B^\perp) \geq (n-3) + (n-3) - (n-1) = n - 5.$$

| $n$ | $\dim W^\perp$ | Min intersection | Level-2 contraction possible? |
|-----|:-:|:-:|:-:|
| 5 | 2 | **0** | Yes (and proven in the paper) |
| 6 | 3 | **1** | No |
| 7 | 4 | **2** | No |
| $n$ | $n-3$ | $n-5$ | No for $n \geq 6$ |

For $n \geq 6$, any unit vector in $W_A^\perp \cap W_B^\perp$ is preserved
isometrically by both $D_{1,A}$ and $D_{1,B}$. The level-2 commutator
$D_2 = D_{1,A} D_{1,B} D_{1,A}^T D_{1,B}^T$ therefore has operator norm $1$ for
every valid quadruple. $\blacksquare$

### Step 6: The level-3 distinction

The level-2 fixed spaces have dimension at least $n-5$. Their intersection
has dimension at least $2(n-5)-(n-1)=n-9$, so every level-3 operator has norm 1
when $n \geq 10$. For $S_6,S_7$, the depth-3 seed already displayed in the
manuscript remains valid and fixes the last point. Thus each leaf projection
fixes the nonzero sum-zero vector $(1,\ldots,1,-(n-1))$, proving norm 1 for
this example. This proves failure of a uniform strict bound in these two groups.

A maximum of 1 does not imply that every configuration has norm 1. Script 14
also includes a strictly contractive example in each of $S_6,S_7$. In the
integer basis $e_i-e_{n-1}$, let $G=I+J$ be the Gram matrix and write
$D=N/2^{64}$. Positive leading principal minors of
$2^{128}G-N^TGN$ certify strict contraction exactly. The script checks
nonidentity of every internal target for both kinds of example.

## Reproduction and scope

- Script 14: standard-representation numerical census for $S_5,S_6,S_7$,
  exact norm-1 witnesses and exact contractive witnesses for $S_6,S_7$.
- Script 15: exact singular-value calculation for adjacent pairs.
- Script 20: exact full ordered $S_5$ certificate, unchanged.

| Reduced census | $S_5$ | $S_6$ | $S_7$ |
|---|---:|---:|---:|
| Level-2 blocks | 405 | 1,620 | 4,725 |
| Level-2 blocks with norm 1 | 270 | 1,620 | 4,725 |
| Valid level-3 pairs | 73,810 | 1,242,915 | 10,282,335 |
| Numerically strictly contractive | 73,810 | 800,055 | 3,131,940 |
| Numerically norm 1 | 0 | 442,860 | 7,150,395 |
| Observed maximum | 0.0353047532 | 1 | 1 |

Counts of norms use float64 with tolerance $10^{-10}$. Only the explicit
witnesses in Script 14 are certified by integer arithmetic. The ordering
conventions of this reduced census omit orientations; its $S_5$ maximum is
not a uniform induction constant. The full ordered universe has 9,447,840
pairs, and Script 20 certifies the paper's uniform constant 0.0355.

The one-dimensional simplex diagnostic applies to $S_5$. For $S_6,S_7$,
Script 14 reports dimensions of preserved spaces; an arbitrary top singular
vector cannot describe a multidimensional space. The main $S_5$ theorem,
conditioning argument and terminal-sample corollary are unchanged.
