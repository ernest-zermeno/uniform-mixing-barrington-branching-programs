#!/usr/bin/env python3
"""
19_ordered_level2_exact.py
==========================

Exact rational cross-check for the Level-2 Contraction theorem (Section 4.3)
in the ordered-orientation universe.

The reduced scan in Script 10 enumerates 30 unordered adjacent level-1 pairs
and 405 valid level-2 quadruples.  The theorem statement is naturally ordered:
there are 60 ordered adjacent level-1 pairs and 3240 ordered valid level-2
blocks.  This script verifies, in exact sympy arithmetic, that the ordered
universe has the same Frobenius-norm maximum

    67586345 / 67108864,

although it has one additional non-maximal norm value (17 distinct values
instead of 16 in the reduced quotient).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT10 = HERE / "10_spectral_contraction_proof.py"

spec = importlib.util.spec_from_file_location("spectral_contraction_proof", SCRIPT10)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


ID = tuple(range(5))
ALL_TRANS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def target_pair(pair):
    return base.comm_perm(base.trans_perm(*pair[0]), base.trans_perm(*pair[1]))


def main() -> None:
    proj = {t: base.projection_sym(*t) for t in ALL_TRANS}

    adjacent = []
    for t1 in ALL_TRANS:
        for t2 in ALL_TRANS:
            if t1 == t2:
                continue
            if len(set(t1) & set(t2)) != 1:
                continue
            if target_pair((t1, t2)) != ID:
                adjacent.append((t1, t2))
    assert len(adjacent) == 60, len(adjacent)

    d1 = {}
    target1 = {}
    for pair in adjacent:
        d1[pair] = (proj[pair[0]] * proj[pair[1]]) ** 2
        target1[pair] = target_pair(pair)

    values = []
    for pair_a in adjacent:
        target_a = target1[pair_a]
        for pair_b in adjacent:
            if pair_a == pair_b:
                continue
            target_b = target1[pair_b]
            if base.comm_perm(target_a, target_b) == ID:
                continue
            da = d1[pair_a]
            db = d1[pair_b]
            d2 = da * db * da.T * db.T
            values.append(base.frob_sq(d2))

    assert len(values) == 3240, len(values)
    distinct = sorted(set(values), key=float)
    maximum = max(values, key=float)
    expected = base.Rational(67586345, 67108864)
    extra = base.Rational(67143833, 67108864)

    print(f"Ordered adjacent pairs: {len(adjacent)}")
    print(f"Ordered valid level-2 blocks: {len(values)}")
    print(f"Distinct Frobenius norms: {len(distinct)}")
    print(f"Maximum: {maximum} = {float(maximum):.12f}")
    print(f"Additional ordered-only value present: {extra in distinct}")

    assert len(distinct) == 17, len(distinct)
    assert maximum == expected, maximum
    assert extra in distinct
    print("ORDERED LEVEL-2 EXACT CROSS-CHECK: PASS")


if __name__ == "__main__":
    main()
