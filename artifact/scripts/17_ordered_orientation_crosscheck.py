#!/usr/bin/env python3
"""
17_ordered_orientation_crosscheck.py
====================================

Cross-checks the level-3 contraction constant in the ordered-orientation
interpretation of the fixed S_5 transposition template.

Script 10 works on a reduced orientation quotient and obtains max
||D_3||_op ~= 0.0353047532 over 73,810 configurations.  This script keeps
the natural order of adjacent level-1 pairs and level-2 commutator blocks:

  * 60 ordered adjacent level-1 pairs;
  * 3,240 ordered valid level-2 blocks;
  * 4,723,920 valid unordered pairs of ordered level-2 blocks at level 3.

The maximum in this larger ordered universe is
0.03540023208379892, so the conservative paper constant lambda = 0.0355
covers both scans.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np


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
    start = time.time()
    proj = {t: base.projection_np(*t) for t in ALL_TRANS}

    adjacent_pairs = []
    for t1 in ALL_TRANS:
        for t2 in ALL_TRANS:
            if t1 == t2:
                continue
            if len(set(t1) & set(t2)) != 1:
                continue
            if base.comm_perm(base.trans_perm(*t1), base.trans_perm(*t2)) != ID:
                adjacent_pairs.append((t1, t2))

    print(f"Ordered adjacent pairs: {len(adjacent_pairs)}")
    assert len(adjacent_pairs) == 60

    d1_cache = {}
    target1 = {}
    for pair in adjacent_pairs:
        d1_cache[pair] = np.linalg.matrix_power(proj[pair[0]] @ proj[pair[1]], 2)
        target1[pair] = target_pair(pair)

    d2_cache = []
    target2 = []
    for pair_a in adjacent_pairs:
        target_a = target1[pair_a]
        for pair_b in adjacent_pairs:
            if pair_a == pair_b:
                continue
            target_b = target1[pair_b]
            target = base.comm_perm(target_a, target_b)
            if target == ID:
                continue
            d_a = d1_cache[pair_a]
            d_b = d1_cache[pair_b]
            d2_cache.append(d_a @ d_b @ d_a.T @ d_b.T)
            target2.append(target)

    print(f"Ordered valid level-2 blocks: {len(d2_cache)}")
    assert len(d2_cache) == 3240

    n_valid_l3 = 0
    max_op = 0.0
    max_pair = None
    n = len(d2_cache)

    for i in range(n):
        d_i = d2_cache[i]
        target_i = target2[i]
        for j in range(i + 1, n):
            if base.comm_perm(target_i, target2[j]) == ID:
                continue
            n_valid_l3 += 1
            d3 = d_i @ d2_cache[j] @ d_i.T @ d2_cache[j].T
            op = np.linalg.svd(d3, compute_uv=False)[0]
            if op > max_op:
                max_op = op
                max_pair = (i, j)

        if (i + 1) % 500 == 0:
            print(
                f"  {i+1:4d}/{n}: valid={n_valid_l3}, "
                f"max={max_op:.17f}, elapsed={time.time() - start:.1f}s"
            )

    print(f"Valid ordered level-3 pairs: {n_valid_l3}")
    print(f"Max ||D3||_op: {max_op:.17f}")
    print(f"Max pair indices: {max_pair}")

    assert n_valid_l3 == 4_723_920
    assert max_op < 0.0355
    assert abs(max_op - 0.03540023208379892) < 1e-14

    print("ORDERED ORIENTATION CROSS-CHECK: PASS")


if __name__ == "__main__":
    main()
