# llm-number-choice

Code and data for the paper **"3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models"** (Samar Ansari, 2026).

<!-- TODO: add the arXiv link and the Zenodo DOI badge here once you have them. -->

## What this study does

When a language model is asked for "a random integer between 1 and N", it defines one probability for every integer. This repository computes those probabilities directly, by summing over the token sequences that spell each integer, instead of estimating them by sampling. It does this for 18 open-weight model lines (15 matched base and instruction-tuned pairs), at N = 5, 10, 20, 30 and 100, under 29 decoding settings, on two hardware stacks.

## Hardware used

| Stack | Machine | Framework and precision | Runs |
|---|---|---|---|
| CUDA | RTX 4070 Ti, Windows, Python 3.12 | PyTorch CUDA, bfloat16 | 32 |
| Metal | Mac Mini, 24 GB, Python 3.14 | PyTorch MPS, float16 | 53 |

Each run covers one checkpoint in one prompt condition, at 5 ranges and 29 decoding settings, so it writes 145 rows. 85 runs give 12,325 rows.

## Folder layout

```
code/
  rng_exact.py         the measurement: enumeration, stop set, 29 decoding settings
  run_all.py           runs rng_exact.py over every model and condition that fits a machine
  analyze.py           main analysis: writes the CSVs in analysis/
  extra_analyses.py    the numbers analyze.py does not print (see the table below)
  make_figures_v6.py   Figures 1 to 4
data/
  rtx.jsonl            CUDA stack results, 4,640 rows
  mac24.jsonl          Metal stack results, 7,685 rows
analysis/              CSVs and printed reports, as used in the paper
logs/
  rtx/                 one log per run on the CUDA machine, plus runner.log
  mac24/               one log per run on the Metal machine, plus runner.log
figures/               fig1_pairs.pdf to fig4_range.pdf
paper/                 LaTeX source of the paper
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your hardware first (see the note in `requirements.txt`). The analysis and figure scripts need only numpy and matplotlib.

## Reproduce the paper from the released data

You do not need a GPU for this. Run the commands from the repository root.

**List the CUDA file (`rtx`) first in every command.** Where a condition ran on both stacks, the first file wins, and the paper's tables use the CUDA value.

```bash
# 1. Main analysis: writes the 8 CSVs into analysis/
python3 code/analyze.py --input rtx=data/rtx.jsonl mac24=data/mac24.jsonl > analysis/analysis_report.txt

# 2. The extra numbers: sampler effect at T=1.0, robustness checks, midpoints, sign tests
python3 code/extra_analyses.py --jsonl data/rtx.jsonl data/mac24.jsonl > analysis/extra_report.txt

# 3. Figures (--ncol 7 gives the 7 x 7 portrait layout of Figure 2)
python3 code/make_figures_v6.py --jsonl data/rtx.jsonl data/mac24.jsonl --analysis analysis --out figures --ncol 7
```

Quick check after step 1: `analysis_report.txt` should say "instruct less uniform than base: 15/15 pairs" and "P(17) rose in 10/15 families, FELL in 5".

## Where each result in the paper comes from

| Paper | Script | Output |
|---|---|---|
| Table 1 (agreement between stacks) | `analyze.py`, section 1 | `analysis/replication.csv` |
| Table 2 (15 base and instruct pairs) | `analyze.py`, section 4; entropy deficits from section 2 | `analysis/base_vs_instruct.csv`, `analysis/headline.csv` |
| Table 3 (most and least uniform at N = 30) | `analyze.py`, section 2 | `analysis/headline.csv` |
| Table 4 (sets of integers) | `analyze.py`, section 7 | `analysis/structure.csv` |
| Table 5 (controlled contrasts, endpoints) | `analyze.py`, section 8 | `analysis/disentangle.csv` |
| Table 6 (modal integer by range) | `analyze.py`, section 5 | `analysis/by_range.csv` |
| Table 7 (P(17)) | `analyze.py`, section 6 | `analysis/seventeen.csv` |
| Table 8 (excluded conditions) | `analyze.py`, section 2 (rows marked THIN) | `analysis/headline.csv` |
| Table 9 (models and stacks) | `logs/` and the two data files | |
| Section 4 and 5, sampler effect at T = 0.7 | `analyze.py`, section 3 | `analysis/sampler_effect.csv` |
| Section 5, sampler effect at T = 1.0 | `extra_analyses.py`, part 1 | `analysis/extra_report.txt` |
| Section 6.3, top two and top three removed, IQRs, counts | `extra_analyses.py`, part 2 | `analysis/extra_report.txt` |
| Section 6.4, midpoints | `extra_analyses.py`, part 3 | `analysis/extra_report.txt` |
| Section 7, TV divided by its maximum | `extra_analyses.py`, part 4 | `analysis/extra_report.txt` |
| Appendix A, lowest valid_mass per range | `extra_analyses.py`, part 5 | `analysis/extra_report.txt` |
| Sign tests | `extra_analyses.py`, part 6 | `analysis/extra_report.txt` |
| Figures 1 to 4 | `make_figures_v6.py` | `figures/` |

## Re-run the measurement (needs a GPU or Apple Silicon)

```bash
cd code
python3 run_all.py --machine rtx --dry-run      # show the plan first
python3 run_all.py --machine rtx                # CUDA machine
python3 run_all.py --machine mac24              # Metal machine
python3 run_all.py --check                      # audit the results file
```

`run_all.py` writes to `results/exact.jsonl` and skips runs that are already complete, so it is safe to stop and restart. After a full run, rename the file from each machine (to `rtx.jsonl` and `mac24.jsonl`) before the analysis step. Some models are gated on Hugging Face; request access to them first and log in with `hf auth login` (older versions: `huggingface-cli login`). Bit-for-bit agreement with the released data is not expected: the paper reports how much the two stacks differ (Table 1).

## Known gaps

- `google/gemma-2-9b` and `google/gemma-2-9b-it` did not run: the disk was full (see `logs/mac24/`).
- On the Metal stack, `gemma-3-4b-pt`, `gemma-3-4b-it` and `gemma-3-1b-pt` returned no valid probabilities; those conditions use the CUDA values only.
- `mistralai/Mistral-7B-Instruct-v0.3` with prefill has valid_mass below 10^-4 at every range, so it forms no pair.

## Licences

- Code: MIT (see `LICENSE`).
- Data, analysis outputs and logs: CC BY 4.0 (see `LICENSE-DATA`).
- The models themselves are not included and remain under their own licences.

## Citation

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.
