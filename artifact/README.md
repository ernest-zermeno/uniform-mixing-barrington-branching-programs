# Verification artifact: Uniform Mixing in Barrington Branching Programs

Final-version reproducibility artifact for the paper (Section 7). It contains every
computational verification cited in the paper.

## Contents

- `scripts/01_...19_*.py`: the original verification suite covering
  T-independence, collision counts, Fourier and spectral analysis, levels 0-2,
  exhaustive level-3 cross-checks, the dimensional obstruction, the
  conditioning lemma, and conjunctions with wildcards.
- `scripts/20_level3_exact_certificate.py`: proof-grade exact certificate for
  Theorem `thm:level23`. It reduces 9,447,840 oriented level-3 configurations
  to 39,591 orbits under `S5 x Z2` and certifies the operator-norm bound on all
  five non-trivial, non-sign irreducible representations using integer Gram
  forms and fraction-free Bareiss principal minors.
- `CERTIFICATE_LEVEL3.md`: statement, method, exact results, and caveats for
  Script 20.
- `THEOREM_dimensional_obstruction.md`: analytic proof notes for the
  dimensional obstruction.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT license for the verification code.

## How to run

Use Python 3.9 or later. The final checks were run with Python 3.13,
numpy 2.4 and sympy 1.14:

```bash
pip install -r requirements.txt
cd scripts
python3 01_p9_ambivalence_verification.py
# Continue through Script 19 as needed.
python3 20_level3_exact_certificate.py
```

Script 20 takes about 2.5 minutes on the audited laptop. Its expected final
line is:

```text
LEVEL-3 EXACT RATIONAL CERTIFICATE: PASS
```

The complete 20-script suite takes about 33 minutes sequentially. Scripts
17/18 are independent `float64` cross-checks; Script 20 is the authoritative
exact certificate for the paper's level-3 bound.

## Scope of the reduced censuses

Scripts 12 and 13 numerically check 73,810 reduced level-3 pairs. Their maxima
are not global induction constants. Script 14 uses the same reduced ordering
conventions in S5/S6/S7 and reports separately the maximum and the numbers of
contractive and norm-1 configurations (float64, tolerance 1e-10). Its simplex
direction check is restricted to S5; higher-dimensional fixed spaces in S6/S7
are reported by dimension. It additionally verifies explicit examples of both
norm 1 and strict contraction in S6/S7 with exact integer arithmetic.

The authoritative full ordered bound remains 0.0355 from Script 20.

To reproduce the final-revision checks:

```bash
python3 scripts/12_level3_exhaustive_verification.py
python3 scripts/13_all_representations_contraction.py
python3 scripts/14_dimensional_obstruction_verification.py
python3 scripts/20_level3_exact_certificate.py
```
