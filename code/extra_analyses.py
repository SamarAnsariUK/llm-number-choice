#!/usr/bin/env python3
"""
extra_analyses.py
=================

Reproduces the numbers added to the paper in v3 to v6 that analyze.py does
not print:

  1. Sampler effect at N=30, temperature 1.0 and 0.7 (Section 5)
  2. Odd vs even with the top 1, 2 and 3 integers removed, with IQRs and
     counts above 1 (Section 6.3)
  3. Midpoint percentiles (Section 6.4)
  4. Ceiling-normalised median TV by range (Section 7)
  5. Lowest retained valid_mass per range (Appendix A.2)
  6. Two-sided sign tests quoted in the paper

Usage
-----
    python3 extra_analyses.py --jsonl results/rtx.jsonl results/mac24.jsonl

The first file is treated as the CUDA stack and is preferred where a
condition ran on both stacks, which is the rule the paper's tables use.
Needs only numpy.
"""

from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np

PRIOR = "T1_p1_k0"
MIN_VALID_PRIMARY = 1e-4
MIN_VALID_SET = 0.10
RANGES = (5, 10, 20, 30, 100)


def condition(r):
    if r.get("prompt_style") == "completion":
        return "base"
    return "prefill" if r.get("prefill") else "chat"


def load(paths):
    rows = []
    for pri, path in enumerate(paths):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_pri"] = pri
            r["_cond"] = condition(r)
            r["_vm"] = r.get("valid_mass") or 0.0
            rows.append(r)
    return rows


def ok(r, thr):
    tv = r.get("tv_distance")
    return r["_vm"] >= thr and tv is not None and tv == tv


def select(rows, decode, thr=MIN_VALID_SET):
    """One row per (model, condition, N): first stack listed wins."""
    out = {}
    for r in sorted(rows, key=lambda x: x["_pri"]):
        if r["decode"] != decode or not ok(r, thr):
            continue
        out.setdefault((r["model"], r["_cond"], r["n_max"]), r)
    return out


def upper_median(v):
    v = sorted(v)
    return v[len(v) // 2]


def sign_test_two_sided(k, n):
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def is_prime(k):
    return k > 1 and all(k % i for i in range(2, int(k ** 0.5) + 1))


def kind(cond):
    return "base" if cond == "base" else "instruct"


# -----------------------------------------------------------------------------

def sampler_effect(rows):
    print("\n1. SAMPLER EFFECT at N=30 (top-p 0.95 vs untruncated, same T)")
    for T in ("T1", "T0.7"):
        u = {(r["_pri"], r["model"], r["_cond"]): r for r in rows
             if r["decode"] == f"{T}_p1_k0" and r["n_max"] == 30
             and ok(r, MIN_VALID_SET)}
        t = {(r["_pri"], r["model"], r["_cond"]): r for r in rows
             if r["decode"] == f"{T}_p0.95_k0" and r["n_max"] == 30
             and ok(r, MIN_VALID_PRIMARY)}
        keys = [k for k in u if k in t]
        add = np.array([t[k]["tv_distance"] - u[k]["tv_distance"] for k in keys])
        rat = np.array([t[k]["top1_prob"] / u[k]["top1_prob"] for k in keys
                        if u[k]["top1_prob"]])
        conds = {(k[1], k[2]) for k in keys}
        print(f"   {T}: {len(keys)} stack rows, {len(conds)} conditions; "
              f"median TV added {np.median(add):.4f} "
              f"(IQR {np.percentile(add, 25):.4f} to {np.percentile(add, 75):.4f}), "
              f"max {add.max():.3f}; median p(mode) ratio {np.median(rat):.3f}")


def odd_even(sel):
    print("\n2. ODD vs EVEN, endpoints and multiples of 5 removed, top k removed")
    for drop in (1, 2, 3):
        acc = {}
        for (model, cond, n), r in sel.items():
            p = np.array(r["p_conditional"], float)
            top = set((np.argsort(-p)[:drop] + 1).tolist())
            keep = [v for v in range(2, n) if v not in top]
            if p[[v - 1 for v in keep]].sum() <= 0 or len(keep) < 8:
                continue
            non5 = [v for v in keep if v % 5]
            a = [v for v in non5 if v % 2]
            b = [v for v in non5 if v % 2 == 0]
            if len(a) < 3 or len(b) < 3:
                continue
            ra = p[[v - 1 for v in a]].mean() / p[[v - 1 for v in b]].mean()
            acc.setdefault((kind(cond), n), []).append(ra)
        print(f"   top {drop} removed:")
        for k in ("base", "instruct"):
            for n in (20, 30, 100):
                v = acc.get((k, n))
                if not v:
                    continue
                above = sum(x > 1 for x in v)
                print(f"      {k:8s} N={n:3d}: median {upper_median(v):.2f} "
                      f"IQR {np.percentile(v, 25):.2f} to {np.percentile(v, 75):.2f}, "
                      f"{above}/{len(v)} above 1, sign test p="
                      f"{sign_test_two_sided(above, len(v)):.2g}")


def midpoint(sel):
    print("\n3. MIDPOINT percentile among interior non-multiples of 5")
    mids = {10: 6, 20: 11, 30: 16, 100: 51}
    for k in ("base", "instruct"):
        cells = []
        for n, m in mids.items():
            pct = []
            for (model, cond, nn), r in sel.items():
                if nn != n or kind(cond) != k:
                    continue
                p = r["p_conditional"]
                others = [v for v in range(2, n) if v % 5 and v != m]
                pct.append(np.mean([p[m - 1] > p[v - 1] for v in others]))
            cells.append(f"N={n}: {np.median(pct):.2f}")
        print(f"   {k:8s} " + "  ".join(cells))
    for k in ("base", "instruct"):
        rs = [r for (mo, c, n), r in sel.items() if n == 5 and kind(c) == k]
        print(f"   N=5 {k}: midpoint 3 is the mode in "
              f"{sum(r['top1_value'] == 3 for r in rs)} of {len(rs)}")


def normalised_tv(sel):
    print("\n4. MEDIAN TV / (1 - 1/N), from upper medians as in Table 6")
    for k in ("instruct", "base"):
        cells = []
        for n in RANGES:
            v = [r["tv_distance"] for (m, c, nn), r in sel.items()
                 if nn == n and kind(c) == k]
            med = round(upper_median(v), 3)
            cells.append(f"N={n}: {med:.3f} -> {med / (1 - 1 / n) + 1e-9:.3f}")
        print(f"   {k:8s} " + "  ".join(cells))


def lowest_valid(rows):
    print("\n5. LOWEST valid_mass at the model prior, per range")
    sel = select(rows, PRIOR, thr=MIN_VALID_PRIMARY)
    for n in RANGES:
        v = sorted(((r["_vm"], m, c) for (m, c, nn), r in sel.items() if nn == n))
        print(f"   N={n:3d}: " + ", ".join(f"{m.split('/')[-1]} [{c}] {vm:.3f}"
                                         for vm, m, c in v[:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="CUDA file first, then Metal file")
    args = ap.parse_args()
    rows = load(args.jsonl)
    sel = select(rows, PRIOR)
    sampler_effect(rows)
    odd_even(sel)
    midpoint(sel)
    normalised_tv(sel)
    lowest_valid(rows)
    print("\n6. SIGN TESTS (two-sided)")
    for k, n in ((15, 15), (16, 16), (21, 33), (27, 33), (10, 15)):
        print(f"   {k}/{n}: p = {sign_test_two_sided(k, n):.3g}")


if __name__ == "__main__":
    main()
