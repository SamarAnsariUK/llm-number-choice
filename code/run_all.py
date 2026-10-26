#!/usr/bin/env python3
"""
run_all.py
==========

Runs rng_exact.py across the whole model list, in a sensible order, on
whichever machine you are sitting at.

Written in Python rather than PowerShell or bash so that one script works on
the Windows box and both Macs without maintaining three versions.

What it does for you
--------------------
  * Picks the models that fit the machine's memory. You do not have to keep a
    separate list per machine.
  * Runs smallest first, so a broken setup fails in two minutes instead of
    two hours.
  * SKIPS WORK ALREADY DONE. It reads results/exact.jsonl, works out which
    runs are already complete, and does not repeat them. Safe to stop with
    Ctrl-C and restart later. This is the important one, because model
    downloads fail often.
  * Keeps going when one model fails, and lists the failures at the end.
  * Applies the right flags per model type, so you cannot forget --no-think
    on a reasoning model or --prompt-style completion on a base model.

Usage
-----
    python run_all.py --machine rtx --dry-run     # see the plan first
    python run_all.py --machine rtx               # do it
    python run_all.py --machine mac24 --tier 2    # the bigger models
    python run_all.py --check                     # audit the results file
    python run_all.py --dedupe                    # remove duplicate rows

Always start with --dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Configuration
# =============================================================================

RANGES = ["5", "10", "20", "30", "100"]

# 29 decoding configurations come out of --full-grid. Used to tell a finished
# run from a half-finished one. If you change DEFAULT_DECODE_GRID in
# rng_exact.py, change this too or the resume logic will be wrong.
CONFIGS_PER_RANGE = 29

RESULTS = os.path.join("results", "exact.jsonl")
LOGDIR = "logs"

# Usable memory in GB, leaving headroom for the OS and activations.
MACHINES = {
    "rtx":   11.0,   # RTX 4070 Ti, 12 GB VRAM
    "mac24": 20.0,   # Mac Mini, 24 GB unified
    "mac16":  8.0,   # Mac Mini, 16 GB unified. Small models only.
    "cpu":   64.0,   # no limit worth enforcing, but it will be slow
}


@dataclass
class Model:
    repo: str
    params_b: float          # billions of parameters
    kind: str                # "instruct" or "base"
    reasoning: bool = False  # hybrid reasoning model with a <think> block
    tier: int = 1            # see the blocks below
    gated: bool = False      # needs a licence click on Hugging Face
    verified: bool = True    # False = repo id is from memory, check it
    note: str = ""

    @property
    def approx_gb(self) -> float:
        # BF16 weights are 2 bytes per parameter. Add a fixed allowance for
        # activations, the KV cache and the logits tensor over a 150k vocab.
        return self.params_b * 2.0 + 1.5


# -----------------------------------------------------------------------------
# The model list, in four blocks.
#
#   tier 1  Qwen3 size ladder. The primary within-family manipulation.
#   tier 2  Qwen2.5 size ladder. Same lab, no reasoning block. The control
#           that separates "Qwen" from "reasoning model".
#   tier 3  Cross-family. Different tokenisers and pretraining corpora.
#   tier 4  Repo ids I am less sure about. Verify before running.
#
# Memory filtering is separate from tier. A tier 3 model that does not fit
# your GPU is skipped automatically and reported as skipped.
#
# EVERY REPO ID HERE CAME FROM MEMORY, and my knowledge stops around May 2026.
# Open each model page and confirm the id and the commit hash before a final
# publishable run. Ones I am least confident about are marked verified=False.
# -----------------------------------------------------------------------------
MODELS: List[Model] = [
    # ---- TIER 1. Qwen3 size ladder, base and instruct at every size. -------
    # The only place where parameter count varies while the tokeniser,
    # architecture and training recipe stay roughly fixed.
    Model("Qwen/Qwen3-0.6B",      0.6, "instruct", reasoning=True,  tier=1),
    Model("Qwen/Qwen3-0.6B-Base", 0.6, "base",                      tier=1),
    Model("Qwen/Qwen3-1.7B",      1.7, "instruct", reasoning=True,  tier=1),
    Model("Qwen/Qwen3-1.7B-Base", 1.7, "base",                      tier=1),
    Model("Qwen/Qwen3-4B",        4.0, "instruct", reasoning=True,  tier=1),
    Model("Qwen/Qwen3-4B-Base",   4.0, "base",                      tier=1),
    Model("Qwen/Qwen3-8B",        8.0, "instruct", reasoning=True,  tier=1),
    Model("Qwen/Qwen3-8B-Base",   8.0, "base",                      tier=1),

    # ---- TIER 2. Qwen2.5 size ladder. -------------------------------------
    # Same lab, same tokeniser family, NO reasoning block. Without this you
    # cannot tell whether a Qwen3 result is about Qwen or about reasoning
    # models. It is cheap and it fits the 12 GB card up to 3B.
    Model("Qwen/Qwen2.5-0.5B-Instruct", 0.5, "instruct", tier=2),
    Model("Qwen/Qwen2.5-0.5B",          0.5, "base",     tier=2),
    Model("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "instruct", tier=2),
    Model("Qwen/Qwen2.5-1.5B",          1.5, "base",     tier=2),
    Model("Qwen/Qwen2.5-3B-Instruct",   3.1, "instruct", tier=2),
    Model("Qwen/Qwen2.5-3B",            3.1, "base",     tier=2),
    Model("Qwen/Qwen2.5-7B-Instruct",   7.6, "instruct", tier=2),
    Model("Qwen/Qwen2.5-7B",            7.6, "base",     tier=2),

    # ---- TIER 3. Cross-family. --------------------------------------------
    # Small enough for the 12 GB card:
    Model("google/gemma-2-2b-it",       2.6, "instruct", tier=3),
    Model("google/gemma-2-2b",          2.6, "base",     tier=3),
    Model("google/gemma-3-1b-it",       1.0, "instruct", tier=3),
    Model("google/gemma-3-1b-pt",       1.0, "base",     tier=3),
    Model("google/gemma-3-4b-it",       4.3, "instruct", tier=3),
    Model("google/gemma-3-4b-pt",       4.3, "base",     tier=3),
    Model("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, "instruct", tier=3),
    Model("HuggingFaceTB/SmolLM2-1.7B",          1.7, "base",     tier=3),
    Model("microsoft/Phi-4-mini-instruct", 3.8, "instruct", tier=3,
          note="no base counterpart released"),
    Model("meta-llama/Llama-3.2-1B-Instruct", 1.2, "instruct", tier=3,
          gated=True),
    Model("meta-llama/Llama-3.2-1B",          1.2, "base",     tier=3,
          gated=True),
    Model("meta-llama/Llama-3.2-3B-Instruct", 3.2, "instruct", tier=3,
          gated=True),
    Model("meta-llama/Llama-3.2-3B",          3.2, "base",     tier=3,
          gated=True),

    # Needs the 24 GB Mac in BF16:
    Model("mistralai/Mistral-7B-Instruct-v0.3", 7.2, "instruct", tier=3),
    Model("mistralai/Mistral-7B-v0.3",          7.2, "base",     tier=3),
    Model("meta-llama/Llama-3.1-8B-Instruct",   8.0, "instruct", tier=3,
          gated=True, note="click the licence on the model page first"),
    Model("meta-llama/Llama-3.1-8B",            8.0, "base",     tier=3,
          gated=True),
    Model("google/gemma-2-9b-it",               9.2, "instruct", tier=3),
    Model("google/gemma-2-9b",                  9.2, "base",     tier=3),
    Model("allenai/OLMo-2-1124-7B-Instruct",    7.3, "instruct", tier=3,
          note="training corpus is public, the only checkable one"),
    Model("allenai/OLMo-2-1124-7B",             7.3, "base",     tier=3),

    # ---- TIER 4. Repo ids I am less sure of. Verify each one first. --------
    # These are real model families but I am reciting the exact repository
    # strings from memory. Open the page, confirm, then move them to tier 3.
    Model("nvidia/Nemotron-Mini-4B-Instruct", 4.2, "instruct", tier=4,
          verified=False),
    Model("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", 8.0, "instruct", tier=4,
          verified=False, note="Llama-derived, so not an independent family"),
    Model("ibm-granite/granite-3.1-8b-instruct", 8.1, "instruct", tier=4,
          verified=False),
    Model("ibm-granite/granite-3.1-8b-base",     8.1, "base",     tier=4,
          verified=False),
    Model("tiiuae/Falcon3-7B-Instruct", 7.5, "instruct", tier=4,
          verified=False),
    Model("tiiuae/Falcon3-7B-Base",     7.5, "base",     tier=4,
          verified=False),
]


# =============================================================================
# Building the run plan
# =============================================================================

@dataclass(frozen=True)
class Run:
    """One invocation of rng_exact.py."""
    repo: str
    prompt_style: str
    prefill: str
    enable_thinking: Optional[bool]
    label: str
    device: Optional[str] = None

    def args(self) -> List[str]:
        a = [sys.executable, "rng_exact.py",
             "--model", self.repo,
             "--ranges", *RANGES,
             "--prompt-style", self.prompt_style,
             "--full-grid",
             "--out", RESULTS]
        if self.prefill:
            a += ["--prefill", self.prefill]
        if self.enable_thinking is False:
            a += ["--no-think"]
        elif self.enable_thinking is True:
            a += ["--think"]
        if self.device:
            a += ["--device", self.device]
        return a

    def signature(self) -> Tuple:
        """Must match the fields written into each result row."""
        return (self.repo, self.prompt_style, self.prefill,
                self.enable_thinking)


def runs_for(m: Model, with_thinking_on: bool = False,
             device: Optional[str] = None) -> List[Run]:
    """The set of conditions to measure for one model.

    INSTRUCT MODELS get two runs.
      no prefill  : measures instruction following AND number choice together
      prefill     : forces the answer position, isolating number choice

    The gap between the two is the instruction-following effect, and it will
    change a lot across the size ladder. That gap is a result in itself.

    BASE MODELS get one run. The completion prompt already ends in "A:", so
    the answer position is forced by construction.

    IMPORTANT ASYMMETRY, write this into the paper. The base condition is
    prefilled by design. So when comparing base against instruct, compare
    base against the instruct run WITH prefill. Comparing against the
    no-prefill run would confound post-training with answer-position forcing.
    """
    if m.kind == "base":
        return [Run(m.repo, "completion", "", None,
                    f"{m.repo} [completion]", device)]

    think = False if m.reasoning else None
    out = [
        Run(m.repo, "chat", "", think, f"{m.repo} [no prefill]", device),
        Run(m.repo, "chat", "Answer: ", think, f"{m.repo} [prefill]", device),
    ]
    if with_thinking_on and m.reasoning:
        # Secondary condition. Never pool this with the thinking-off runs.
        out.append(Run(m.repo, "chat", "", True,
                       f"{m.repo} [thinking on]", device))
    return out


def build_plan(machine: str, tiers: List[int], include_gated: bool,
               with_thinking_on: bool, include_unverified: bool,
               device: Optional[str] = None,
               ) -> Tuple[List[Run], List[Tuple[Model, str]]]:
    """Return the runs to do, and the models excluded with the reason why.

    Silent exclusion is how you end up wondering where Gemma went. Every
    dropped model is reported.
    """
    budget = MACHINES[machine]
    if device == "cpu":
        budget = MACHINES["cpu"]

    plan: List[Run] = []
    excluded: List[Tuple[Model, str]] = []
    chosen: List[Model] = []

    for m in MODELS:
        if m.tier not in tiers:
            excluded.append((m, f"tier {m.tier} not requested"))
        elif m.gated and not include_gated:
            excluded.append((m, "gated, pass --gated after accepting the "
                                "licence"))
        elif not m.verified and not include_unverified:
            excluded.append((m, "repo id unverified, pass --unverified"))
        elif m.approx_gb > budget:
            excluded.append((m, f"needs ~{m.approx_gb:.0f} GB, this machine "
                                f"has {budget:.0f} GB"))
        else:
            chosen.append(m)

    # Smallest first, so a broken setup fails fast.
    chosen.sort(key=lambda m: (m.approx_gb, m.repo))
    for m in chosen:
        plan.extend(runs_for(m, with_thinking_on, device))
    return plan, excluded


# =============================================================================
# Resume support
# =============================================================================

def completed_runs(path: str = RESULTS) -> Set[Tuple]:
    """Signatures of runs that are already fully present in the results file.

    A run counts as complete only if it has every range at every decoding
    configuration. A half-finished run is reported as incomplete so it gets
    redone, because a partial run in the dataset is worse than no run.
    """
    if not os.path.exists(path):
        return set()

    counts: Counter = Counter()
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            counts[(r.get("model"), r.get("prompt_style"),
                    r.get("prefill", ""), r.get("enable_thinking"))] += 1

    if bad:
        print(f"note: {bad} unparseable lines in {path}, ignored")

    expected = len(RANGES) * CONFIGS_PER_RANGE
    done = set()
    for sig, n in counts.items():
        if n == expected:
            done.add(sig)
        elif n > expected:
            print(f"warning: {sig[0]} {sig[1]} has {n} rows, expected "
                  f"{expected}. Duplicates. Run --dedupe.")
            done.add(sig)
        else:
            print(f"note: {sig[0]} {sig[1]} has {n}/{expected} rows, "
                  f"incomplete. It will be run again.")
    return done


def dedupe(path: str = RESULTS) -> None:
    """Remove exactly duplicated lines, keeping the first of each."""
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return
    seen: Set[str] = set()
    kept: List[str] = []
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            total += 1
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                kept.append(s)
    backup = path + ".bak"
    os.replace(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + "\n")
    print(f"kept {len(kept)} of {total} lines. Original saved as {backup}")


def check(path: str = RESULTS) -> None:
    """Audit the results file. Run this before any analysis."""
    if not os.path.exists(path):
        print(f"{path} does not exist yet")
        return
    expected = len(RANGES) * CONFIGS_PER_RANGE
    counts: Counter = Counter()
    low_mass = 0
    nan_rows = 0
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            counts[(r.get("model"), r.get("prompt_style"),
                    r.get("prefill", ""), r.get("enable_thinking"))] += 1
            vm = r.get("valid_mass", 0.0)
            if vm < 1e-4:
                low_mass += 1
            tv = r.get("tv_distance")
            if tv is None or tv != tv:      # NaN check
                nan_rows += 1

    print(f"{total} rows in {path}\n")
    for sig, n in sorted(counts.items()):
        flag = "ok " if n == expected else "BAD"
        think = {None: "-", True: "think", False: "nothink"}[sig[3]]
        pre = "prefill" if sig[2] else "-"
        print(f"  {flag} {n:>4}/{expected}  {sig[0]:<38} {sig[1]:<11} "
              f"{pre:<8} {think}")

    print(f"\nrows with valid_mass below 1e-4 : {low_mass}")
    print("   These are cases where the model does not emit a bare integer. "
          "Their conditional distributions are ratios of near-zero numbers "
          "and must be excluded from analysis, not averaged in.")
    print(f"rows with NaN tv_distance       : {nan_rows}")
    print("   Expected under greedy decoding when the most likely token is "
          "not a digit. That is a real result, not a bug.")


# =============================================================================
# Execution
# =============================================================================

def execute(plan: List[Run], done: Set[Tuple], timeout: int) -> None:
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS) or ".", exist_ok=True)

    todo = [r for r in plan if r.signature() not in done]
    skipped = len(plan) - len(todo)

    print(f"\n{len(plan)} runs in the plan, {skipped} already done, "
          f"{len(todo)} to do\n")
    if not todo:
        print("Nothing to do.")
        return

    failures: List[Tuple[str, str]] = []
    started = time.time()

    for i, run in enumerate(todo, 1):
        tag = (run.repo.replace("/", "_")
               + ("_prefill" if run.prefill else "")
               + ("_think" if run.enable_thinking is True else "")
               + ("_" + run.prompt_style))
        logpath = os.path.join(LOGDIR, tag + ".log")

        print(f"[{i}/{len(todo)}] {run.label}")
        print(f"          log: {logpath}")
        t0 = time.time()

        env = dict(os.environ)
        env["TQDM_DISABLE"] = "1"          # stop progress bars mangling output

        try:
            with open(logpath, "w", encoding="utf-8") as log:
                proc = subprocess.run(run.args(), stdout=log,
                                      stderr=subprocess.STDOUT,
                                      timeout=timeout, env=env)
            if proc.returncode == 0:
                print(f"          done in {time.time() - t0:.0f}s\n")
            else:
                print(f"          FAILED, exit code {proc.returncode}. "
                      f"See the log.\n")
                failures.append((run.label, f"exit {proc.returncode}"))
        except subprocess.TimeoutExpired:
            print(f"          TIMED OUT after {timeout}s\n")
            failures.append((run.label, "timeout"))
        except KeyboardInterrupt:
            print("\nStopped. Progress is saved. Re-run to continue where "
                  "you left off.")
            return
        except Exception as e:                              # noqa: BLE001
            print(f"          ERROR: {e}\n")
            failures.append((run.label, str(e)[:80]))

    mins = (time.time() - started) / 60
    print(f"finished in {mins:.1f} minutes")
    if failures:
        print(f"\n{len(failures)} failed:")
        for label, why in failures:
            print(f"   {label}  ({why})")
        print("\nCommon causes: a gated model you have not requested access "
              "to, no disk space left, or an out-of-memory error. Read the "
              "log for the specific run.")
    else:
        print("all runs succeeded")
    print("\nNow run:  python run_all.py --check")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--machine", choices=list(MACHINES), default="rtx",
                   help="Which box you are on. Sets the memory budget.")
    p.add_argument("--tier", type=int, nargs="+", default=[1, 2, 3],
                   help="1 = Qwen3 ladder, 2 = Qwen2.5 ladder, "
                        "3 = cross-family, 4 = unverified repo ids")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Force a device. Use cpu to run models that do not "
                        "fit the GPU. Slow but it works, and the numbers are "
                        "identical because the maths is the same.")
    p.add_argument("--gated", action="store_true",
                   help="Include Llama. Request access on Hugging Face first.")
    p.add_argument("--unverified", action="store_true",
                   help="Include tier 4, whose repo ids I could not verify.")
    p.add_argument("--thinking-on", action="store_true",
                   help="Add the secondary thinking-on condition for "
                        "reasoning models.")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Seconds per run before giving up. Default 2 hours.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and stop. Always do this first.")
    p.add_argument("--check", action="store_true",
                   help="Audit the results file and exit.")
    p.add_argument("--dedupe", action="store_true",
                   help="Remove duplicate rows and exit.")
    args = p.parse_args()

    if args.check:
        check()
        return 0
    if args.dedupe:
        dedupe()
        return 0

    if not os.path.exists("rng_exact.py"):
        print("rng_exact.py is not in this folder. Run from the project "
              "directory.")
        return 1

    plan, excluded = build_plan(args.machine, args.tier, args.gated,
                                args.thinking_on, args.unverified,
                                args.device)

    budget = (MACHINES["cpu"] if args.device == "cpu"
              else MACHINES[args.machine])
    print(f"machine: {args.machine}  ({budget:.0f} GB usable"
          + (f", forced to {args.device}" if args.device else "") + ")")
    print(f"tiers:   {args.tier}\n")

    if not plan:
        print("Nothing selected. See the exclusions below.")

    if plan:
        print("PLAN:")
        for r in plan:
            print(f"   {r.label}")

    # Never drop a model silently.
    if excluded:
        print("\nSKIPPED:")
        for m, why in sorted(excluded, key=lambda x: x[0].repo):
            print(f"   {m.repo:<42} {why}")

    if args.dry_run:
        print(f"\n{len(plan)} runs. Roughly 3 to 8 minutes each after the "
              f"model has downloaded.")
        print("Remove --dry-run to start.")
        return 0

    if not plan:
        return 1

    done = completed_runs()
    execute(plan, done, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
