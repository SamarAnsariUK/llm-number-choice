#!/usr/bin/env python3
"""
make_figures.py
===============

Replaces the three-panel plot in analyze.py, which showed the most uniform,
the median and the least uniform model-condition and nothing else. That was
my choice and it was the wrong one: it throws away 49 of the 52
model-conditions and, worse, it cannot show the paired base-to-instruct
movement, which is the paper's central claim.

Four figures, in the order they matter:

  fig1_pairs        Slope chart. TV base to TV instruct for all 16 paired
                    families, one line each. This is the paper's headline
                    result in a single image and should be Figure 1.
  fig2_grid         Small multiples. Every model-condition at N=30 as its
                    own panel with the uniform reference line. Shows the
                    full spread rather than three hand-picked cases.
  fig3_heuristics   The three heuristics by range, base against instruct:
                    endpoint avoidance, round-number avoidance, controlled
                    odd preference.
  fig4_range        Modal integer against range, base and instruct.

Usage
-----
    pip install matplotlib
    python3 make_figures.py \
        --jsonl results/rtx.jsonl results/mac24.jsonl \
        --analysis analysis \
        --out figures

Then in the paper: \\includegraphics[width=\\linewidth]{fig1_pairs.pdf}

PDF is the default output because LaTeX embeds vector PDF cleanly and a PNG
of 52 panels looks poor in print. Pass --png if you also want raster copies
for a blog post.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

PRIOR = "T1_p1_k0"
MIN_VALID = 0.10
PRIMARY_N = 30

# Consistent colours across every figure.
C_BASE = "#4a6fa5"
C_INST = "#b3452e"
C_REF = "#888888"


def condition(r: dict) -> str:
    if r.get("prompt_style") == "completion":
        return "base"
    return "prefill" if r.get("prefill") else "chat"


def load_jsonl(paths: List[str]) -> List[dict]:
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"   missing {p}, skipping")
            continue
        machine = os.path.splitext(os.path.basename(p))[0]
        n = 0
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["machine"] = machine
                rows.append(r)
                n += 1
        print(f"   loaded {n:,} rows from {p}")
    return rows


def load_csv(analysis_dir: str, name: str) -> List[dict]:
    path = os.path.join(analysis_dir, name)
    if not os.path.exists(path):
        print(f"   missing {path}")
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(d: dict, k: str, default=float("nan")) -> float:
    try:
        return float(d.get(k, ""))
    except (TypeError, ValueError):
        return default


def short(name: str) -> str:
    """Trim the org prefix so 16 labels fit on one axis."""
    return name.split("/")[-1]


# =============================================================================

def fig1_pairs(plt, analysis: str, out: str, exts) -> None:
    """Slope chart: every paired family's move from base to instruct.

    One line per family. If all 16 slope upward the reader sees the result
    without reading a number, which no table achieves.
    """
    rows = load_csv(analysis, "base_vs_instruct.csv")
    if not rows:
        return
    rows.sort(key=lambda r: fnum(r, "tv_change"))

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for r in rows:
        b, i = fnum(r, "tv_base"), fnum(r, "tv_instruct")
        ax.plot([0, 1], [b, i], "-", color=C_INST, alpha=0.55, lw=1.4,
                zorder=2)
        ax.plot([0], [b], "o", color=C_BASE, ms=5, zorder=3)
        ax.plot([1], [i], "o", color=C_INST, ms=5, zorder=3)

    # Several families land within 0.02 of each other near the top, so the
    # labels overlap and become unreadable. Push them apart vertically and
    # draw a leader line back to the real value.
    lab = sorted(((fnum(r, "tv_instruct"), short(r["family"])) for r in rows))
    min_gap = 0.030
    placed = []
    for y, name in lab:
        ty = y if not placed else max(y, placed[-1][0] + min_gap)
        placed.append((ty, y, name))
    for ty, y, name in placed:
        ax.annotate(name, (1.06, ty), fontsize=7, va="center", ha="left")
        if abs(ty - y) > 1e-9:
            ax.plot([1.005, 1.05], [y, ty], "-", color=C_REF, lw=0.5,
                    alpha=0.8, zorder=1)

    ax.set_xlim(-0.08, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base checkpoint", "instruction tuned"])
    ax.set_ylabel("total variation distance from uniform")
    ax.set_ylim(0, 1.0)
    ax.axhline(1 - 1 / PRIMARY_N, color=C_REF, ls=":", lw=1)
    ax.annotate("maximum possible, $1-1/N$", (0.02, 1 - 1 / PRIMARY_N - 0.03),
                fontsize=7, color=C_REF)
    ax.set_title(f"All {len(rows)} paired families move away from uniform "
                 f"after instruction tuning ($N=30$)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig1_pairs", exts)


def grid_shape(n: int, forced: Optional[int] = None) -> Tuple[int, int]:
    """Choose columns so the last row is not almost empty.

    A fixed six columns leaves 49 panels as eight full rows plus a single
    orphan, which looks broken. Search a sensible range of column counts and
    take the one that wastes the fewest slots, breaking ties toward the
    shape closest to a 3:2 page.
    """
    if forced:
        return forced, math.ceil(n / forced)
    best = None
    for ncol in range(4, 11):
        nrow = math.ceil(n / ncol)
        waste = ncol * nrow - n
        # panels are about 2.1 by 1.55, so this is the rendered aspect ratio
        aspect = (ncol * 2.1) / (nrow * 1.55)
        # Minimising waste alone is wrong: 52 panels would give 13 rows of
        # 4, which is zero waste and unusable on a page. Trade the two off.
        score = 0.6 * waste + 4.0 * abs(aspect - 1.45)
        if best is None or score < best[0]:
            best = (score, ncol, nrow)
    return best[1], best[2]


def fig2_grid(plt, rows: List[dict], out: str, exts,
              ncol_override: Optional[int] = None) -> None:
    """Small multiples: every model-condition at N=30, not three of them."""
    seen, picks = set(), []
    for r in rows:
        if (r.get("n_max") != PRIMARY_N or r.get("decode") != PRIOR
                or r.get("valid_mass", 0) < MIN_VALID):
            continue
        tv = r.get("tv_distance")
        if tv is None or tv != tv:
            continue
        k = (r["model"], condition(r))
        if k in seen:
            continue
        seen.add(k)
        picks.append(r)
    if not picks:
        print("   no usable rows for the grid")
        return
    picks.sort(key=lambda r: r["tv_distance"])

    ncol, nrow = grid_shape(len(picks), forced=ncol_override)
    print(f"   {len(picks)} panels laid out {nrow} x {ncol} "
          f"({ncol * nrow - len(picks)} empty)")
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.1, nrow * 1.55),
                             sharex=True, sharey=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, r in zip(axes, picks):
        p = r["p_conditional"]
        cond = condition(r)
        col = C_BASE if cond == "base" else C_INST
        ax.bar(range(1, len(p) + 1), p, color=col, width=0.9, linewidth=0)
        ax.axhline(1.0 / len(p), color=C_REF, ls="--", lw=0.7)
        ax.set_title(f"{short(r['model'])}\n[{cond}]  TV={r['tv_distance']:.2f}",
                     fontsize=6, pad=2)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(f"Exact answer distributions over $1..{PRIMARY_N}$, model "
                 f"prior, all {len(picks)} retained model-conditions. "
                 f"Blue base, red instruction tuned. Dashed line is uniform.",
                 fontsize=9, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out, "fig2_grid", exts)


def fig3_heuristics(plt, analysis: str, out: str, exts) -> None:
    """The three heuristics side by side, base against instruct."""
    struct = load_csv(analysis, "structure.csv")
    dis = load_csv(analysis, "disentangle.csv")
    if not struct and not dis:
        return
    ranges = [5, 10, 20, 30, 100]
    x = range(len(ranges))

    panels = [
        ("endpoint avoidance", "$P(1)$ relative to interior", dis,
         lambda r: r.get("contrast") == "endpoint_P(1)", "median_ratio"),
        ("round-number avoidance", "multiples of 10 vs uniform", struct,
         lambda r: r.get("set") == "multiples of 10",
         "median_ratio_vs_uniform"),
        ("odd preference, controlled", "odd vs even per member", dis,
         lambda r: r.get("contrast") == "odd_vs_even", "median_ratio"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, (title, ylab, src, filt, valcol) in zip(axes, panels):
        present = set()
        for kind, col in (("base", C_BASE), ("instruct", C_INST)):
            ys = []
            for n in ranges:
                hit = [r for r in src if filt(r) and r.get("kind") == kind
                       and str(r.get("n_max")) == str(n)]
                if hit:
                    present.add(n)
                ys.append(fnum(hit[0], valcol) if hit else float("nan"))
            ax.plot(list(x), ys, "o-", color=col, label=kind, ms=5)
        ax.axhline(1.0, color=C_REF, ls="--", lw=0.8)

        # Every panel starts at 0 and shares an upper bound, so a 2x effect
        # is not drawn to look like a 100x one. Starting panel 3 at 1.00,
        # as the earlier version did, put a base value of 1.01 flat on the
        # axis beside two panels that ran to zero.
        ax.set_ylim(0, 2.3)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

        # A blank column means the set is empty at that range once endpoints
        # are removed, not that the measurement is missing. Say so.
        for j, n in enumerate(ranges):
            if n not in present:
                ax.axvspan(j - 0.45, j + 0.45, color="#000000", alpha=0.04,
                           lw=0)
                ax.text(j, 1.9, "set\nempty", ha="center", va="center",
                        fontsize=6, color=C_REF, style="italic")

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"$N$={n}" for n in ranges], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")

    # No strapline. The LaTeX caption carries the interpretation, and the
    # old one ("tuning drives round numbers to zero") is no longer what the
    # middle panel shows: base checkpoints now sit at parity there.
    fig.tight_layout()
    save(fig, out, "fig3_heuristics", exts)


def fig4_range(plt, analysis: str, out: str, exts) -> None:
    """Where the mode sits as the range grows."""
    rows = load_csv(analysis, "by_range.csv")
    if not rows:
        return
    ranges = [5, 10, 20, 30, 100]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for kind, col, off in (("base", C_BASE, -0.10), ("instruct", C_INST, 0.10)):
        for j, n in enumerate(ranges):
            vals = [int(r["top1_value"]) / n for r in rows
                    if int(r["n_max"]) == n
                    and (r["condition"] == "base") == (kind == "base")]
            if not vals:
                continue
            ax.scatter([j + off] * len(vals), vals, s=14, alpha=0.45,
                       color=col, edgecolors="none",
                       label=kind if j == 0 else None)
    ax.axhline(0.5, color=C_REF, ls="--", lw=0.8)
    ax.annotate("midpoint of the range", (4.15, 0.51), fontsize=7,
                color=C_REF)
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels([f"$N$={n}" for n in ranges])
    ax.set_ylabel("modal integer divided by $N$")
    ax.set_ylim(0, 1.05)
    ax.set_title("The mode is not a fixed fraction of the range", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig4_range", exts)


def save(fig, out: str, stem: str, exts) -> None:
    os.makedirs(out, exist_ok=True)
    for ext in exts:
        path = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"   wrote {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--jsonl", nargs="+",
                   default=["results/rtx.jsonl", "results/mac24.jsonl"])
    p.add_argument("--analysis", default="analysis")
    p.add_argument("--out", default="figures")
    p.add_argument("--ncol", type=int, default=None,
                   help="force the number of columns in figure 2. By "
                        "default the shape is chosen to leave the fewest "
                        "empty panels.")
    p.add_argument("--png", action="store_true",
                   help="also write PNG copies, for a blog post")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return 1

    plt.rcParams.update({"font.family": "serif", "axes.grid": False})
    exts = ["pdf"] + (["png"] if args.png else [])

    print("figure 1: paired slope chart")
    fig1_pairs(plt, args.analysis, args.out, exts)

    print("figure 2: small multiples over every model-condition")
    rows = load_jsonl(args.jsonl)
    fig2_grid(plt, rows, args.out, exts, args.ncol)

    print("figure 3: the three heuristics by range")
    fig3_heuristics(plt, args.analysis, args.out, exts)

    print("figure 4: modal integer against range")
    fig4_range(plt, args.analysis, args.out, exts)

    print(f"\nCopy the PDFs next to your .tex files, then uncomment the "
          f"figure blocks.")
    print("Figure 1 is the one to lead with. It carries the headline claim "
          "without a single number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
