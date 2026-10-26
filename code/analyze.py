#!/usr/bin/env python3
"""
analyze.py
==========

Reads the result files from both machines and answers the questions the study
was designed around. Pure standard library, so it runs anywhere the data
lands. Plots are optional and skipped if matplotlib is missing.

Usage
-----
    python analyze.py --input rtx=results/rtx.jsonl mac24=results/mac24.jsonl

    List the CUDA file (rtx) FIRST. Where a condition ran on both stacks,
    the first file wins, and the paper's tables use the CUDA value.

    python analyze.py --input rtx=... mac24=... --plots

Outputs
-------
Printed report, plus CSV files in analysis/ that you can open in a spreadsheet
or attach to the paper.

The four questions
------------------
  1. REPLICATION. Do the two machines agree? They ran different hardware AND
     different numeric precision (bfloat16 on CUDA, float16 on MPS), so this
     is both a reproducibility check and a free precision experiment.
  2. HEADLINE. How far is each model from uniform, and which integer wins?
  3. SAMPLER vs MODEL. How much of the bias is the model's own distribution,
     and how much is nucleus truncation adding on top? This is the novel one.
  4. POST-TRAINING. Do base checkpoints differ from their instruct versions?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

OUTDIR = "analysis"

# Decode keys, as written by DecodeConfig.key() in rng_exact.py.
PRIOR = "T1_p1_k0"              # the model's own distribution, untruncated
STACK = "T0.7_p0.95_k0"         # what a typical local tool actually ships
T07 = "T0.7_p1_k0"              # 0.7 with no truncation
GREEDY = "greedy_p1_k0"

# Rows below this are ratios of near-zero numbers. Excluding them is a
# pre-specified rule, not a judgement call made after seeing the data.
MIN_VALID_MASS = 1e-4

PRIMARY_RANGE = 30


# =============================================================================
# Loading
# =============================================================================

def load(specs: List[str]) -> List[dict]:
    """specs look like 'rtx=results/rtx.jsonl'."""
    rows: List[dict] = []
    for spec in specs:
        if "=" not in spec:
            print(f"bad --input '{spec}'. Use machine=path.")
            sys.exit(1)
        machine, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"missing file: {path}")
            sys.exit(1)
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                r["machine"] = machine
                rows.append(r)
                n += 1
        print(f"loaded {n:>6} rows from {path}  (machine={machine})")
    return rows


def condition(r: dict) -> str:
    """A short label for the prompt condition."""
    if r.get("prompt_style") == "completion":
        return "base"
    return "prefill" if r.get("prefill") else "chat"


def usable(r: dict) -> bool:
    vm = r.get("valid_mass", 0.0)
    tv = r.get("tv_distance")
    return (vm is not None and vm >= MIN_VALID_MASS
            and tv is not None and tv == tv)


def tv_between(p: List[float], q: List[float]) -> float:
    """Total variation distance between two distributions."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


# =============================================================================
# 1. Replication across machines
# =============================================================================

def report_replication(rows: List[dict]) -> List[dict]:
    print("\n" + "=" * 74)
    print("1. REPLICATION ACROSS MACHINES")
    print("=" * 74)
    print("Two machines ran overlapping models on different hardware AND")
    print("different numeric precision: bfloat16 on the CUDA box, float16 on")
    print("Apple Silicon. So a disagreement here is a precision result, not")
    print("only a reproducibility check.\n")

    key = defaultdict(dict)
    for r in rows:
        if not usable(r):
            continue
        k = (r["model"], condition(r), r["n_max"], r["decode"])
        key[k][r["machine"]] = r

    machines = sorted({r["machine"] for r in rows})
    if len(machines) < 2:
        print("Only one machine in the data. Nothing to compare.")
        return []

    a, b = machines[0], machines[1]
    out = []
    for k, d in key.items():
        if a in d and b in d:
            pa = d[a]["p_conditional"]
            pb = d[b]["p_conditional"]
            out.append({
                "model": k[0], "condition": k[1], "n_max": k[2],
                "decode": k[3],
                "tv_between_machines": tv_between(pa, pb),
                f"top1_{a}": d[a]["top1_value"],
                f"top1_{b}": d[b]["top1_value"],
                f"tv_{a}": d[a]["tv_distance"],
                f"tv_{b}": d[b]["tv_distance"],
            })

    if not out:
        print("No overlapping conditions found.")
        return []

    # Greedy is a point mass. If two tokens are nearly tied, fp16 versus
    # bf16 rounding can flip which one wins, and TV jumps straight from 0 to
    # 1.0. That is a knife-edge, not a distributional disagreement, so it has
    # to be reported separately or it swamps the real comparison.
    greedy = [o for o in out if o["decode"].startswith("greedy")]
    stoch = [o for o in out if not o["decode"].startswith("greedy")]

    def summarise(name: str, group: List[dict]) -> Optional[float]:
        if not group:
            print(f"{name}: none")
            return None
        d = sorted(o["tv_between_machines"] for o in group)
        agree = sum(1 for o in group if o[f"top1_{a}"] == o[f"top1_{b}"])
        med = d[len(d) // 2]
        print(f"{name}")
        print(f"   conditions            : {len(group)}")
        print(f"   median TV difference  : {med:.5f}")
        print(f"   90th percentile TV    : {d[int(0.9*len(d))]:.5f}")
        print(f"   max TV difference     : {d[-1]:.5f}")
        print(f"   same modal integer    : {agree}/{len(group)} "
              f"({100*agree/len(group):.1f}%)")
        return med

    med_stoch = summarise("STOCHASTIC decoding (the primary comparison)",
                          stoch)
    print()
    summarise("GREEDY decoding (reported separately, see note above)", greedy)

    print("\nBiggest STOCHASTIC disagreements:")
    for o in sorted(stoch, key=lambda x: -x["tv_between_machines"])[:8]:
        print(f"   TV={o['tv_between_machines']:.4f}  {o['model']:<34} "
              f"{o['condition']:<8} N={o['n_max']:<4} {o['decode']:<16} "
              f"top1 {o[f'top1_{a}']} vs {o[f'top1_{b}']}")

    # Temperature is the variable that explains the spread. Dividing logits
    # by a small T multiplies every difference between them, including the
    # rounding difference between fp16 and bf16. So disagreement should grow
    # as temperature falls, and greedy is the limiting case.
    print("\nDisagreement by temperature:")
    by_t: Dict[str, List[float]] = defaultdict(list)
    by_t_mode: Dict[str, List[bool]] = defaultdict(list)
    for o in out:
        d = o["decode"]
        t = "greedy" if d.startswith("greedy") else d.split("_")[0]
        by_t[t].append(o["tv_between_machines"])
        by_t_mode[t].append(o[f"top1_{a}"] == o[f"top1_{b}"])

    def tsort(k: str) -> float:
        return -1.0 if k == "greedy" else float(k[1:])

    print(f"   {'temp':>8} {'n':>6} {'median TV':>11} {'p90 TV':>9} "
          f"{'max TV':>8} {'same mode':>10}")
    for t in sorted(by_t, key=tsort):
        d = sorted(by_t[t])
        agree = sum(by_t_mode[t]) / len(by_t_mode[t])
        print(f"   {t:>8} {len(d):>6} {d[len(d)//2]:>11.5f} "
              f"{d[int(0.9*len(d))]:>9.5f} {d[-1]:>8.5f} {agree:>9.1%}")
    print("   Read the p90 and same-mode columns, NOT the median. At low")
    print("   temperature most conditions are a near point mass, so the two")
    print("   machines agree exactly and the median is 0. The failures are")
    print("   rare but total: when the top two tokens are nearly tied,")
    print("   precision decides the winner and TV jumps to its maximum.")
    print("   Low temperature is therefore bimodal, not uniformly worse.")
    print("   The honest summary statistic is the mode-agreement rate.")

    print()
    if med_stoch is not None and med_stoch < 0.02:
        print("VERDICT: the two stacks agree on stochastic decoding. Report")
        print("this as a successful implementation replication across")
        print("hardware and across fp16 versus bf16.")
        if greedy:
            g_agree = sum(1 for o in greedy
                          if o[f"top1_{a}"] == o[f"top1_{b}"])
            print(f"\nGreedy is a separate story: the two machines pick the")
            print(f"same integer in only {g_agree}/{len(greedy)} cases. When")
            print("the top two tokens are nearly tied, numeric precision")
            print("decides the winner. Greedy 'random' number generation is")
            print("therefore not reproducible across inference stacks, which")
            print("is worth stating as a finding in its own right.")
    else:
        print("VERDICT: the stacks disagree on stochastic decoding too.")
        print("Investigate before writing anything up.")
    return out


# =============================================================================
# 2. Headline table
# =============================================================================

def report_headline(rows: List[dict], n_max: int = PRIMARY_RANGE
                    ) -> List[dict]:
    print("\n" + "=" * 74)
    print(f"2. HEADLINE: distance from uniform at N=1..{n_max}")
    print("=" * 74)
    print("Model prior means temperature 1.0 with no truncation. That IS the")
    print("model's own distribution; everything else is the sampler acting")
    print("on it. TV of 0 is perfectly uniform.\n")

    best: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        if r["n_max"] != n_max or r["decode"] != PRIOR or not usable(r):
            continue
        k = (r["model"], condition(r))
        # The first file passed to --input wins where both stacks ran the
        # condition. Pass the CUDA file (rtx) first: that is the rule the
        # paper's tables use.
        best.setdefault(k, r)

    out = []
    for (model, cond), r in sorted(best.items(),
                                   key=lambda kv: kv[1]["tv_distance"]):
        out.append({
            "model": model, "condition": cond, "n_max": n_max,
            "tv_distance": r["tv_distance"],
            "top1_value": r["top1_value"],
            "top1_prob": r["top1_prob"],
            "valid_mass": r["valid_mass"],
            "support_size": r.get("support_size"),
            "entropy_deficit_nats": r.get("entropy_deficit_nats"),
        })

    print(f"{'TV':>7} {'top1':>5} {'p(top1)':>8} {'valid':>7} "
          f"{'supp':>5}  model / condition")
    thin = []
    for o in out:
        mark = ""
        if o["valid_mass"] < 0.10:
            mark = "  <-- THIN"
            thin.append(o)
        print(f"{o['tv_distance']:7.4f} {o['top1_value']:5d} "
              f"{o['top1_prob']:8.3f} {o['valid_mass']:7.3f} "
              f"{o['support_size']:5d}  {o['model']} [{o['condition']}]{mark}")

    if thin:
        print(f"\n{len(thin)} rows marked THIN have valid_mass below 0.10.")
        print("The model emits a bare integer under 10% of the time, so the")
        print("conditional distribution is conditioned on a rare event. The")
        print("numbers are arithmetically correct but they describe a corner")
        print("of the model's behaviour, not its typical output. Either drop")
        print("these rows or report them separately with the caveat stated.")

    if out:
        modes = defaultdict(int)
        inst_modes = defaultdict(int)
        base_modes = defaultdict(int)
        for o in out:
            modes[o["top1_value"]] += 1
            if o["condition"] == "base":
                base_modes[o["top1_value"]] += 1
            else:
                inst_modes[o["top1_value"]] += 1

        print("\nFavourite integer, ALL model-conditions:")
        for v, c in sorted(modes.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {v:>4} : {c}")
        print("\nFavourite integer, BASE checkpoints only:")
        for v, c in sorted(base_modes.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {v:>4} : {c}")
        print("\nFavourite integer, INSTRUCT checkpoints only:")
        for v, c in sorted(inst_modes.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {v:>4} : {c}")
        print("\nWhat this does and does not show. Post-training changes")
        print("WHICH integer wins, but not toward any single shared target.")
        print("Check section 6 before claiming a 17 effect: if P(17) rises")
        print("in some families and collapses in others, the reliable")
        print("finding is concentration, not the identity of the winner.")
    return out


# =============================================================================
# 3. Sampler versus model. The novel part.
# =============================================================================

def report_sampler_effect(rows: List[dict]) -> List[dict]:
    print("\n" + "=" * 74)
    print("3. SAMPLER versus MODEL")
    print("=" * 74)
    print("Nucleus truncation cannot make a distribution more uniform; it can")
    print("only delete tail mass. The question is how much bias it ADDS on")
    print("top of the model's own. Local tools ship with top_p at 0.9 or 0.95")
    print("by default, so this is the gap between what the model believes and")
    print("what the user experiences.\n")

    idx: Dict[Tuple, dict] = {}
    for r in rows:
        if usable(r):
            idx[(r["machine"], r["model"], condition(r), r["n_max"],
                 r["decode"])] = r

    out = []
    for k, r in idx.items():
        if k[4] != T07:
            continue
        stack = idx.get((k[0], k[1], k[2], k[3], STACK))
        if not stack:
            continue
        out.append({
            "machine": k[0], "model": k[1], "condition": k[2], "n_max": k[3],
            "tv_untruncated": r["tv_distance"],
            "tv_top_p_0.95": stack["tv_distance"],
            "tv_added_by_sampler": stack["tv_distance"] - r["tv_distance"],
            "p_top1_untruncated": r["top1_prob"],
            "p_top1_top_p_0.95": stack["top1_prob"],
            "support_untruncated": r.get("support_size"),
            "support_top_p_0.95": stack.get("support_size"),
        })

    if not out:
        print("No matching pairs found.")
        return []

    prim = [o for o in out if o["n_max"] == PRIMARY_RANGE]
    if prim:
        adds = sorted(o["tv_added_by_sampler"] for o in prim)
        ratios = [o["p_top1_top_p_0.95"] / o["p_top1_untruncated"]
                  for o in prim if o["p_top1_untruncated"] > 0]
        print(f"at N=1..{PRIMARY_RANGE}, {len(prim)} model-conditions:")
        print(f"   median TV added by top_p=0.95 : {adds[len(adds)//2]:+.4f}")
        print(f"   largest TV added              : {adds[-1]:+.4f}")
        if ratios:
            ratios.sort()
            print(f"   median increase in p(mode)    : "
                  f"{ratios[len(ratios)//2]:.2f}x")

        print("\n   Largest sampler effects:")
        for o in sorted(prim, key=lambda x: -x["tv_added_by_sampler"])[:8]:
            print(f"      +{o['tv_added_by_sampler']:.4f}  "
                  f"{o['model']:<34} [{o['condition']}]  "
                  f"p(mode) {o['p_top1_untruncated']:.3f} -> "
                  f"{o['p_top1_top_p_0.95']:.3f}")
    return out


# =============================================================================
# 4. Base versus instruct
# =============================================================================

BASE_TO_INSTRUCT = {
    "Qwen/Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b": "google/gemma-2-2b-it",
    "google/gemma-2-9b": "google/gemma-2-9b-it",
    "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
    "google/gemma-3-4b-pt": "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def report_base_vs_instruct(rows: List[dict]) -> List[dict]:
    print("\n" + "=" * 74)
    print("4. BASE versus INSTRUCT")
    print("=" * 74)
    print("Same tokeniser, same architecture, same pretraining. Only the")
    print("post-training differs. Compare the base run against the instruct")
    print("PREFILL run, because the base prompt already forces the answer")
    print("position and comparing against no-prefill would confound")
    print("post-training with answer-position forcing.\n")

    # The paper applies the valid_mass >= 0.10 rule to every model-prior
    # analysis, including this one. Without it the gemma-2-2b pair (instruct
    # valid_mass 0.026) enters and the count reads 16/16 instead of 15/15.
    idx: Dict[Tuple, dict] = {}
    for r in rows:
        if (usable(r) and r["n_max"] == PRIMARY_RANGE and r["decode"] == PRIOR
                and r.get("valid_mass", 0) >= 0.10):
            idx.setdefault((r["model"], condition(r)), r)

    out = []
    for base_repo, inst_repo in BASE_TO_INSTRUCT.items():
        b = idx.get((base_repo, "base"))
        i = idx.get((inst_repo, "prefill"))
        if not b or not i:
            continue
        out.append({
            "family": inst_repo,
            "tv_base": b["tv_distance"],
            "tv_instruct": i["tv_distance"],
            "tv_change": i["tv_distance"] - b["tv_distance"],
            "top1_base": b["top1_value"],
            "top1_instruct": i["top1_value"],
            "p_top1_base": b["top1_prob"],
            "p_top1_instruct": i["top1_prob"],
        })

    if not out:
        print("No complete base/instruct pairs found.")
        return []

    print(f"{'TV base':>8} {'TV inst':>8} {'change':>8} "
          f"{'mode b':>7} {'mode i':>7}  family")
    for o in sorted(out, key=lambda x: -x["tv_change"]):
        print(f"{o['tv_base']:8.4f} {o['tv_instruct']:8.4f} "
              f"{o['tv_change']:+8.4f} {o['top1_base']:7d} "
              f"{o['top1_instruct']:7d}  {o['family']}")

    worse = sum(1 for o in out if o["tv_change"] > 0)
    print(f"\ninstruct less uniform than base: {worse}/{len(out)} pairs")
    print("Existing work (arXiv:2505.00047) reports base models are closer")
    print("to uniform. If your pairs agree, you have replicated it with exact")
    print("measurement instead of sampling. If they disagree, say so.")
    return out


# =============================================================================
# CSV output and plots
# =============================================================================

def report_by_range(rows: List[dict]) -> List[dict]:
    """How the favourite integer moves as the range changes.

    You collected N = 5, 10, 20, 30 and 100 but only N=30 has been looked at.
    This is where the anecdote gets properly tested. The classic human claim
    is 17 out of 1 to 20. Coronado-Blazquez reports 3 and 4 for 1 to 5, 5 and
    7 for 1 to 10, and 37, 47 and 73 for 1 to 100. Those are directly
    comparable to what you have.
    """
    print("\n" + "=" * 74)
    print("5. HOW THE FAVOURITE INTEGER MOVES WITH THE RANGE")
    print("=" * 74)

    out = []
    seen = set()
    for r in rows:
        if r["decode"] != PRIOR or not usable(r):
            continue
        if r.get("valid_mass", 0) < 0.10:
            continue
        k = (r["model"], condition(r), r["n_max"])
        if k in seen:
            continue
        seen.add(k)
        out.append({"model": r["model"], "condition": condition(r),
                    "n_max": r["n_max"], "top1_value": r["top1_value"],
                    "top1_prob": r["top1_prob"],
                    "tv_distance": r["tv_distance"]})

    for kind in ("base", "instruct"):
        print(f"\n{kind.upper()} checkpoints, favourite integer by range:")
        for n in (5, 10, 20, 30, 100):
            sel = [o for o in out if o["n_max"] == n
                   and (o["condition"] == "base") == (kind == "base")]
            if not sel:
                continue
            modes: Dict[int, int] = defaultdict(int)
            for o in sel:
                modes[o["top1_value"]] += 1
            top = sorted(modes.items(), key=lambda kv: -kv[1])[:5]
            tv = sorted(o["tv_distance"] for o in sel)
            summary = ", ".join(f"{v}({c})" for v, c in top)
            print(f"   N=1..{n:<4} n={len(sel):<3} median TV="
                  f"{tv[len(tv)//2]:.3f}   modes: {summary}")

    print("\nWhat to check against the literature:")
    print("   N=1..5    Coronado-Blazquez 2025 reports 3 and 4")
    print("   N=1..10   reports 5 and 7")
    print("   N=1..100  reports 37, 47 and 73")
    print("   N=1..20   this is the classic human '17' range; no LLM study")
    print("             has published it, so it is genuinely yours")
    return out


def report_seventeen(rows: List[dict]) -> List[dict]:
    """Track P(17) specifically, from base checkpoint to instruct.

    The right question is not 'is 17 the mode' but 'how much probability
    does 17 hold relative to uniform, and does post-training raise it'.
    A ratio of 1.0 means no preference at all.
    """
    print("\n" + "=" * 74)
    print("6. THE 17 PREFERENCE, MEASURED DIRECTLY")
    print("=" * 74)

    idx: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        if (r["n_max"] == 30 and r["decode"] == PRIOR and usable(r)
                and r.get("valid_mass", 0) >= 0.10):
            idx.setdefault((r["model"], condition(r)), r)

    out = []
    for base_repo, inst_repo in BASE_TO_INSTRUCT.items():
        b = idx.get((base_repo, "base"))
        i = idx.get((inst_repo, "prefill"))
        if not b or not i:
            continue
        pb = b["p_conditional"][16]          # index 16 is the integer 17
        pi = i["p_conditional"][16]
        out.append({
            "family": inst_repo,
            "p17_base": pb, "p17_instruct": pi,
            "ratio_base_vs_uniform": pb * 30,
            "ratio_instruct_vs_uniform": pi * 30,
            "amplification": (pi / pb) if pb > 0 else float("inf"),
        })

    if not out:
        print("No usable pairs.")
        return []

    print("Uniform would give P(17) = 0.0333. The ratio column is how many")
    print("times more likely than uniform 17 actually is.\n")
    print(f"{'P17 base':>9} {'ratio':>7} {'P17 inst':>9} {'ratio':>7} "
          f"{'amplif':>7}  family")
    for o in sorted(out, key=lambda x: -x["ratio_instruct_vs_uniform"]):
        print(f"{o['p17_base']:9.4f} {o['ratio_base_vs_uniform']:7.2f} "
              f"{o['p17_instruct']:9.4f} "
              f"{o['ratio_instruct_vs_uniform']:7.2f} "
              f"{o['amplification']:7.2f}  {o['family']}")

    rb = sorted(o["ratio_base_vs_uniform"] for o in out)
    ri = sorted(o["ratio_instruct_vs_uniform"] for o in out)
    up = sum(1 for o in out if o["amplification"] > 1)
    down = len(out) - up
    print(f"\nmedian ratio, base     : {rb[len(rb)//2]:.2f}x uniform")
    print(f"median ratio, instruct : {ri[len(ri)//2]:.2f}x uniform")
    print(f"P(17) rose in {up}/{len(out)} families, FELL in {down}")
    print("\nIf the count is split and the two medians are close, there is")
    print("no reliable 17 effect from post-training. Post-training")
    print("concentrates the distribution hard, but the target it")
    print("concentrates on is model-specific. Some families land on 17 and")
    print("some actively move away from it. Report the split honestly; a")
    print("bidirectional result is still a result, and claiming a uniform")
    print("amplification the data does not support is how papers get")
    print("retracted.")
    return out


def report_structure(rows: List[dict]) -> List[dict]:
    """Test structural hypotheses about WHICH integers get the mass.

    Looking at a table of modal integers and spotting a pattern is exactly
    how I got the last six calls wrong in this project. So this measures the
    whole distribution rather than the mode, across every model and range,
    and reports ratios against uniform. A ratio of 1.00 means no preference.

    Hypotheses tested, all pre-specifiable before looking:
      endpoints      1 and N are avoided
      round numbers  multiples of 10 and of 5 are avoided
      primes         primes are preferred
      odd            odd numbers are preferred
      midpoint half  the middle third of the range is preferred
    """
    print("\n" + "=" * 74)
    print("7. WHICH INTEGERS GET THE MASS")
    print("=" * 74)
    print("Ratios against uniform, averaged over models. 1.00 = no")
    print("preference. Computed on the whole distribution, not the mode.\n")

    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        i = 2
        while i * i <= k:
            if k % i == 0:
                return False
            i += 1
        return True

    def sets_for(n: int) -> Dict[str, List[int]]:
        allv = list(range(1, n + 1))

        # Endpoints are removed from every set below. They are measured on
        # their own in the endpoints row and in the controlled contrasts.
        # Leaving them in double-counts the endpoint effect: at N=10 the
        # only multiple of 10 is 10 itself, so that cell would report
        # endpoint avoidance under a round-number label.
        inner = [v for v in allv if v not in (1, n)]

        # The two round-number sets are made disjoint. Every multiple of 10
        # is a multiple of 5, so comparing them as originally written set a
        # set against its own superset.
        m10 = [v for v in inner if v % 10 == 0]
        m5_not_10 = [v for v in inner if v % 5 == 0 and v % 10 != 0]

        return {
            "endpoints (1, N)": [1, n],
            "multiples of 10": m10,
            "multiples of 5, not of 10": m5_not_10,
            "primes": [v for v in inner if is_prime(v)],
            "odd numbers": [v for v in inner if v % 2 == 1],
        }

    acc: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    seen = set()
    for r in rows:
        if r["decode"] != PRIOR or not usable(r):
            continue
        if r.get("valid_mass", 0) < 0.10:
            continue
        kind = "base" if condition(r) == "base" else "instruct"
        k = (r["model"], condition(r), r["n_max"])
        if k in seen:
            continue
        seen.add(k)
        n = r["n_max"]
        p = r["p_conditional"]
        for name, members in sets_for(n).items():
            if not members:
                continue
            # Observed mass on the set, divided by the mass uniform would
            # put there. Size-invariant, so ranges are comparable.
            obs = sum(p[v - 1] for v in members)
            exp = len(members) / n
            acc[(kind, name, n)].append(obs / exp)

    out = []
    names = list(sets_for(30).keys())
    for kind in ("base", "instruct"):
        print(f"{kind.upper()} checkpoints:")
        print(f"   {'set':<26} " + "".join(f"{'N='+str(n):>9}"
                                           for n in (5, 10, 20, 30, 100)))
        for name in names:
            cells = []
            for n in (5, 10, 20, 30, 100):
                vals = acc.get((kind, name, n))
                if not vals:
                    cells.append(f"{'-':>9}")
                    continue
                vals = sorted(vals)
                med = vals[len(vals) // 2]
                cells.append(f"{med:>9.2f}")
                out.append({"kind": kind, "set": name, "n_max": n,
                            "median_ratio_vs_uniform": med,
                            "n_models": len(vals)})
            print(f"   {name:<26} " + "".join(cells))
        print()

    print("How to read this. A row that sits below 1.00 across every range")
    print("is an avoided class; above 1.00 is a preferred class. Compare the")
    print("base block against the instruct block: if a preference is already")
    print("present in base checkpoints, pretraining put it there. If it only")
    print("appears in the instruct block, post-training did.")
    print("\nDo not go hunting for more patterns after seeing this table.")
    print("These five were specified in advance. Anything you notice now and")
    print("test on the same data is a post-hoc finding and has to be")
    print("labelled as one.")
    return out


def report_disentangle(rows: List[dict]) -> List[dict]:
    """Contrasts that control for the two things that broke the first pass.

    FLAW 1, endpoint contamination. The integer 1 is a non-prime at every
    range, and instruct models give 1 a probability of essentially zero. At
    N=5 the entire non-prime side IS {1}, so a prime-vs-non-prime ratio of
    28,000 was just the endpoint effect wearing a different label. Fix:
    remove 1 and N from every contrast set. The endpoint effect is measured
    on its own in contrast C, where it belongs.

    FLAW 2, point-mass domination. When p(mode) reaches 0.98, any set-based
    contrast reports only whether the mode happens to fall in the set. The
    'odd preference' of 7.32 at N=5 was the fact that the modal answer, 3,
    is odd. Fix: remove the modal integer and renormalise. The question then
    becomes the useful one: given the model did not give its favourite
    answer, what does it prefer among the rest?

    Both fixes are pre-specified corrections to a broken measure, not
    choices made to get a nicer number. Say that in the methods section.
    """
    print("\n" + "=" * 74)
    print("8. SEPARATING THE OVERLAPPING EFFECTS")
    print("=" * 74)
    print("Endpoints and the modal integer are excluded from contrasts A")
    print("and B, and the remainder renormalised. Without that, both")
    print("contrasts just re-measure the endpoint effect and the mode.")
    print("A ratio of 1.00 means no preference within the controlled set.\n")

    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        i = 2
        while i * i <= k:
            if k % i == 0:
                return False
            i += 1
        return True

    def ratio(p: List[float], inset: List[int], outset: List[int],
              min_members: int = 3) -> Optional[float]:
        """Mass per member inside vs per member outside."""
        if len(inset) < min_members or len(outset) < min_members:
            return None
        a = sum(p[v - 1] for v in inset) / len(inset)
        b = sum(p[v - 1] for v in outset) / len(outset)
        return (a / b) if b > 0 else None

    accA: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    accB: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    accE: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)

    seen = set()
    for r in rows:
        if r["decode"] != PRIOR or not usable(r):
            continue
        if r.get("valid_mass", 0) < 0.10:
            continue
        k = (r["model"], condition(r), r["n_max"])
        if k in seen:
            continue
        seen.add(k)

        kind = "base" if condition(r) == "base" else "instruct"
        n, p = r["n_max"], list(r["p_conditional"])
        allv = list(range(1, n + 1))

        # Contrast C first, on the untouched distribution.
        interior = [v for v in allv if v not in (1, n)]
        base_rate = sum(p[v - 1] for v in interior) / len(interior)
        if base_rate > 0:
            accE[(kind, n, "P(1)")].append(p[0] / base_rate)
            accE[(kind, n, "P(N)")].append(p[n - 1] / base_rate)

        # Now strip endpoints and the mode, then renormalise.
        mode_v = r["top1_value"]
        keep = [v for v in allv if v not in (1, n) and v != mode_v]
        rest = sum(p[v - 1] for v in keep)
        if rest <= 0 or len(keep) < 8:
            continue
        q = [0.0] * n
        for v in keep:
            q[v - 1] = p[v - 1] / rest

        non5 = [v for v in keep if v % 5 != 0]
        odd5 = [v for v in non5 if v % 2 == 1]
        even5 = [v for v in non5 if v % 2 == 0]
        ra = ratio(q, odd5, even5)
        if ra is not None:
            accA[(kind, n)].append(ra)

        pr = [v for v in odd5 if is_prime(v)]
        npr = [v for v in odd5 if not is_prime(v)]
        rb = ratio(q, pr, npr)
        if rb is not None:
            accB[(kind, n)].append(rb)

    out = []
    ranges = (5, 10, 20, 30, 100)

    def block(title: str, acc, label: str) -> None:
        print(title)
        print(f"   {'':<20} " + "".join(f"{'N='+str(n):>9}" for n in ranges))
        for kind in ("base", "instruct"):
            cells = []
            for n in ranges:
                vals = acc.get((kind, n))
                if not vals:
                    cells.append(f"{'-':>9}")
                    continue
                vals = sorted(vals)
                med = vals[len(vals) // 2]
                cells.append(f"{med:>9.2f}")
                out.append({"contrast": label, "kind": kind, "n_max": n,
                            "median_ratio": med, "n_models": len(vals)})
            print(f"   {kind:<20} " + "".join(cells))
        # Show how many models survived the filters, so a cell built on two
        # models is not read as if it were built on thirty.
        for kind in ("base", "instruct"):
            counts = [len(acc.get((kind, n), [])) for n in ranges]
            print(f"   {'  n models, ' + kind:<20} "
                  + "".join(f"{c:>9d}" for c in counts))
        print("   A dash means the contrast set was too small at that range.")
        print()

    block("A. odd vs even, excluding multiples of 5, endpoints and the mode",
          accA, "odd_vs_even")
    block("B. prime vs non-prime, among odd non-multiples of 5, "
          "excluding\n   endpoints and the mode", accB, "prime_vs_nonprime")

    print("C. endpoints, each relative to the average interior integer")
    print(f"   {'':<20} " + "".join(f"{'N='+str(n):>9}" for n in ranges))
    for kind in ("base", "instruct"):
        for which in ("P(1)", "P(N)"):
            cells = []
            for n in ranges:
                vals = accE.get((kind, n, which))
                if not vals:
                    cells.append(f"{'-':>9}")
                    continue
                vals = sorted(vals)
                med = vals[len(vals) // 2]
                cells.append(f"{med:>9.3f}")
                out.append({"contrast": f"endpoint_{which}", "kind": kind,
                            "n_max": n, "median_ratio": med,
                            "n_models": len(vals)})
            print(f"   {kind + ' ' + which:<20} " + "".join(cells))
    print("\nContrast C is the robust one. It uses no set membership, so")
    print("neither flaw applies to it.")

    return out


def write_csv(name: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def make_plots(rows: List[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping plots. "
              "pip install matplotlib")
        return

    os.makedirs(OUTDIR, exist_ok=True)

    # The figure that carries the whole paper: observed vs uniform for the
    # models with the strongest and weakest bias.
    cands = [r for r in rows
             if r["n_max"] == PRIMARY_RANGE and r["decode"] == PRIOR
             and usable(r) and condition(r) == "prefill"]
    if not cands:
        print("no data for plots")
        return
    cands.sort(key=lambda r: r["tv_distance"])
    picks = [cands[0], cands[len(cands) // 2], cands[-1]]

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3 * len(picks)),
                             sharex=True)
    for ax, r in zip(axes, picks):
        p = r["p_conditional"]
        ax.bar(range(1, len(p) + 1), p, color="#4a6fa5")
        ax.axhline(1.0 / len(p), color="crimson", ls="--", lw=1,
                   label="uniform")
        ax.set_title(f"{r['model']}   TV={r['tv_distance']:.3f}   "
                     f"mode={r['top1_value']}", fontsize=10)
        ax.set_ylabel("P(n)")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel(f"integer, 1..{PRIMARY_RANGE}")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "distributions.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


# =============================================================================

def main() -> int:
    global PRIMARY_RANGE

    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--input", nargs="+", required=True,
                   metavar="MACHINE=PATH",
                   help="e.g. rtx=results/rtx.jsonl mac24=results/mac24.jsonl")
    p.add_argument("--range", type=int, default=PRIMARY_RANGE)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    PRIMARY_RANGE = args.range

    rows = load(args.input)
    total = len(rows)
    good = sum(1 for r in rows if usable(r))
    print(f"\n{total} rows total, {good} usable "
          f"({100*good/total:.1f}%), {total-good} excluded for "
          f"valid_mass < {MIN_VALID_MASS} or NaN.")
    print("Excluding those is a pre-specified rule. State it in the methods "
          "section and report the count.")

    write_csv("replication.csv", report_replication(rows))
    write_csv("headline.csv", report_headline(rows, PRIMARY_RANGE))
    write_csv("sampler_effect.csv", report_sampler_effect(rows))
    write_csv("base_vs_instruct.csv", report_base_vs_instruct(rows))
    write_csv("by_range.csv", report_by_range(rows))
    write_csv("seventeen.csv", report_seventeen(rows))
    write_csv("structure.csv", report_structure(rows))
    write_csv("disentangle.csv", report_disentangle(rows))

    if args.plots:
        make_plots(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
