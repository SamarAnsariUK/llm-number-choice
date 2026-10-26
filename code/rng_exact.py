#!/usr/bin/env python3
"""
rng_exact.py
============

Exact measurement of the probability distribution a language model assigns to
integers 1..N when asked for a "random" number, plus a decomposition of that
distribution into its causes (model prior, temperature, nucleus/top-k
truncation, quantisation, base-vs-instruct post-training).

Why this is not sampling
------------------------
For a fixed prompt with no chat history, the model defines ONE fixed
probability vector. Prior work (Coronado-Blazquez 2025, arXiv:2502.19965)
estimates it with ~100 API calls per configuration. This script computes it
exactly by enumerating the token sequences for every candidate integer and
multiplying conditional probabilities. There is zero sampling error.

Quantity computed
-----------------
For each integer n in 1..N:

    P(n) = sum over tokenisation variants v of n:
             [ prod_i P(v_i | prompt, v_<i) ] * P(terminator | prompt, v)

The final terminator factor matters. Without it, "1" and "17" are not
mutually exclusive events, because "1" can continue into "17", and the
numbers will not be comparable.

Two things are then reported:

  valid_mass  = sum_n P(n)
                the probability the model emits a bare valid integer and stops.
                Nobody in this literature reports this. Small models will have
                very low valid_mass because they add preamble text.

  P_cond(n)   = P(n) / valid_mass
                the distribution over integers, conditional on a valid answer.
                This is the thing to compare against uniform.

Temperature and top-p cost nothing extra
----------------------------------------
Temperature scales logits at EVERY position, so P_T(n) is NOT
softmax(log P(n) / T). It has to be recomputed per position. This script does
that inside the same forward pass, so the whole temperature x top_p grid is
free. There is no extra inference.

Hardware notes for this project
-------------------------------
Tested targets: RTX 4070 Ti (12 GB VRAM, CUDA) and Mac Mini M-series
(24 GB unified memory, MPS). Rough BF16 memory: 2 GB per 1B parameters.
So BF16 fits up to ~4B on the 4070 Ti and up to ~10B on the Mac.
Anything larger goes through the llama.cpp backend as GGUF.

I have not been able to execute this script against a real model. Run the
self-test first (--selftest), then validate against sampling (--validate)
before trusting any number it produces.


==============================================================================
MODEL LIST
==============================================================================

Check Hugging Face for newer releases before you start. My knowledge of model
releases ends around May 2026 and some of the IDs below may have been
superseded.

------------------------------------------------------------------------------
BLOCK A. Size ladder, one family, base and instruct at every size.
------------------------------------------------------------------------------
This is the only place in the whole design where parameter count is varied
while tokeniser, architecture and training recipe are held roughly constant.
It is the closest thing to a controlled manipulation of model size that
exists. Everything runs in BF16 except the last two.

    Qwen/Qwen3-0.6B-Base            ~1.2 GB   BF16, both machines
    Qwen/Qwen3-0.6B                 ~1.2 GB
    Qwen/Qwen3-1.7B-Base            ~3.4 GB   BF16, both machines
    Qwen/Qwen3-1.7B                 ~3.4 GB
    Qwen/Qwen3-4B-Base              ~8.0 GB   BF16, both machines
    Qwen/Qwen3-4B                   ~8.0 GB
    Qwen/Qwen3-8B-Base             ~16.4 GB   BF16 Mac only; Q8_0 on the GPU
    Qwen/Qwen3-8B                  ~16.4 GB
    Qwen/Qwen3-14B                 ~29.0 GB   GGUF Q8_0 (~15 GB) on the Mac

The 0.6B and 1.7B models answer your "should I look at very tiny models"
question directly. Yes. They are nearly free to run, they extend the size
ladder by more than an order of magnitude, and they are where instruction
following breaks down, which makes valid_mass an interesting dependent
variable rather than a footnote.

------------------------------------------------------------------------------
BLOCK B. Cross-family check at the ~7-9B scale, base and instruct paired.
------------------------------------------------------------------------------
Different tokenisers, different pretraining corpora. Use this to show the
effect is not a Qwen artefact.

    meta-llama/Llama-3.1-8B                    gated, request access first
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-7B-v0.3
    mistralai/Mistral-7B-Instruct-v0.3
    google/gemma-2-9b                          Mac in BF16, GPU needs Q4
    google/gemma-2-9b-it
    allenai/OLMo-2-1124-7B                     fully open training data
    allenai/OLMo-2-1124-7B-Instruct
    microsoft/Phi-4-mini-instruct              no base counterpart released

OLMo-2 earns its place because the training corpus is public. If you want to
say anything at all about training data as a cause, it is the only model in
this list where that claim is checkable rather than speculative.

------------------------------------------------------------------------------
BLOCK C. Post-training quantisation ladder on ONE checkpoint.
------------------------------------------------------------------------------
This is the clean quantisation experiment. Same weights, same tokeniser, only
the numeric precision changes. Run on Qwen/Qwen3-8B via llama.cpp GGUF:

    BF16      ~16.4 GB   Mac only
    Q8_0       ~8.7 GB
    Q6_K       ~6.7 GB
    Q4_K_M     ~5.0 GB
    Q3_K_M     ~4.0 GB
    Q2_K       ~3.3 GB
    IQ1_M      ~2.2 GB   expect this one to be badly degraded

Prediction worth pre-registering: these distributions are heavily concentrated
on one or two modes, and small logit perturbations will not move the top mass.
So Q8 through Q4 will probably look nearly identical and the result will be
null. The interesting region is Q2_K and IQ1_M. Say this in advance so a null
at Q4 is a confirmed prediction rather than a disappointment.

------------------------------------------------------------------------------
BLOCK D. Extreme low-bit models. Answering your Bonsai question.
------------------------------------------------------------------------------
    prism-ml/Bonsai-8B-mlx-1bit        1.28 GB, MLX, Apple Silicon only
    prism-ml/Bonsai-8B-unpacked        BF16 safetensors of the same weights
    prism-ml/Bonsai-27B-mlx-1bit       5.13 GB, MLX
    prism-ml/Bonsai-27B-unpacked       BF16 safetensors, ~54 GB, too big
    lmstudio-community/Bonsai-27B-GGUF Q1_0 build, ~4.7 GB, runs on both boxes
    microsoft/bitnet-b1.58-2B-4T       native 1.58-bit, trained from scratch

Include these, but NOT in Block C, and here is the reason.

Bonsai is not post-training quantisation. PrismML's own documentation gives
Qwen3.6-27B as the base model, 1.125 effective bits per weight at g128, and
retention of 94.6 percent for the ternary build and 89.5 percent for the 1-bit
build against the FP16 baseline. Those retention numbers only come from a
quantisation-aware training or distillation pipeline, and PrismML has not
released that pipeline. So a Bonsai-versus-original comparison confounds
"fewer bits" with "extra training". It is not the controlled comparison that
Block C is.

That does not make it uninteresting. It makes it a different question:
does a QAT low-bit model inherit the parent's number preferences, or does the
retraining reshape them? If Bonsai-27B's modes differ from Qwen3.6-27B's
modes, that is a real finding about what QAT preserves. Treat it as its own
subsection with its own framing. The same applies to BitNet b1.58, which was
trained at 1.58 bits from scratch and has no FP16 parent at all.

Practical note: the 27B FP16 parent is about 54 GB and will not fit on either
of your machines, so you cannot run the paired 27B comparison locally. The 8B
pair is feasible on the 24 GB Mac. Do the pairing at 8B and treat 27B as
unpaired. Check the Bonsai 8B whitepaper for its stated base model before you
claim what it is paired against.

A note on the model you named
-----------------------------
You wrote "Qwen3.8-27B (3-bit quant)". I cannot find a Qwen3.8-27B. The model
PrismML lists as Bonsai's parent is Qwen3.6-27B. LM Studio display names and
Hugging Face repository IDs often diverge. Before you write any of this up,
open the model card in LM Studio and copy the exact repo ID and revision hash.
Getting a model name wrong in a paper about model behaviour is the kind of
error a reviewer will not forgive.

------------------------------------------------------------------------------
BLOCK E. Controls.
------------------------------------------------------------------------------
    secrets.randbelow(N) + 1        cryptographic reference distribution
    numpy Generator.integers        seeded, for pipeline validation

Both go through the identical metric code. They should return a TV distance
from uniform of approximately zero. If they do not, the bug is in your
analysis, not in the models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class DecodeConfig:
    """A decoding configuration. temperature <= 0 means greedy."""
    temperature: float
    top_p: float = 1.0
    top_k: int = 0          # 0 disables top-k

    def key(self) -> str:
        t = "greedy" if self.temperature <= 0 else f"T{self.temperature:g}"
        return f"{t}_p{self.top_p:g}_k{self.top_k}"


# The temperature grid deliberately includes the six values used by
# Coronado-Blazquez 2025 so the results are directly comparable, plus 0.7,
# which is the LM Studio and Ollama default that most users actually run.
DEFAULT_DECODE_GRID: List[DecodeConfig] = [
    DecodeConfig(temperature=t, top_p=p, top_k=k)
    for t in (0.1, 0.3, 0.5, 0.7, 0.8, 1.0, 2.0)
    for (p, k) in ((1.0, 0), (0.95, 0), (0.9, 0), (1.0, 40))
] + [DecodeConfig(temperature=0.0)]

DEFAULT_RANGES: List[int] = [5, 10, 20, 30, 100]
# 5, 10 and 100 reproduce Coronado-Blazquez exactly. 20 and 30 cover the
# original "17" anecdote. Do not build the framing around 20 and 30: testing
# an untested range is not a contribution.


# =============================================================================
# Prompts
# =============================================================================

CHAT_PROMPTS = {
    # Primary. Pre-register this one and only this one as primary.
    "primary": "Give me a random integer between 1 and {N}, inclusive. "
               "Reply with only the number.",
    # Robustness variants. Report separately, never pool with the primary.
    "pick":    "Pick a random integer between 1 and {N}, inclusive. "
               "Reply with only the number.",
    "choose":  "Choose one integer at random between 1 and {N}, inclusive. "
               "Reply with only the number.",
    "bare":    "Random integer between 1 and {N}:",
}

# Few-shot completion format for BASE checkpoints, which will not follow an
# instruction and will just continue the text.
#
# LIMITATION, state it in the paper: the examples below name concrete numbers
# (4 and 9), and Kubovy & Psotka (1976) showed that merely mentioning a number
# in the instructions changes how often people produce it. The same risk
# applies here. Mitigations, all imperfect: the example ranges (1-7, 1-12)
# never overlap the test ranges you care about most; run a second few-shot
# variant with different example numbers and check the modes do not move.
COMPLETION_PROMPT = (
    "Q: Give me a random integer between 1 and 7, inclusive.\nA: 4\n\n"
    "Q: Give me a random integer between 1 and 12, inclusive.\nA: 9\n\n"
    "Q: Give me a random integer between 1 and {N}, inclusive.\nA:"
)


def build_prompt_ids(tokenizer, n_max: int, style: str, variant: str,
                     assistant_prefill: str = "",
                     enable_thinking: Optional[bool] = None) -> List[int]:
    """Return the token ids the model conditions on.

    style="chat"       -> apply the model's chat template
    style="completion" -> raw few-shot text, for base checkpoints

    enable_thinking:
        None   leave the template at its own default
        False  ask the template to close the reasoning block immediately

    WHY THIS EXISTS. Reasoning models such as Qwen3 put a <think> block at the
    very start of the assistant turn. Their chat template turns it on by
    default. That makes the first generated token <think> with probability
    close to 1, so the probability of a bare digit at that position is
    effectively zero, and every number this script produces is meaningless.
    Passing enable_thinking=False makes the template emit an already-closed
    empty reasoning block, so the answer really does start at the position we
    are measuring.

    This is a real experimental condition, not a workaround. Report thinking
    on and thinking off separately. Never pool them.
    """
    if style == "completion":
        text = COMPLETION_PROMPT.format(N=n_max)
        return _as_int_list(
            tokenizer(text, add_special_tokens=True)["input_ids"])

    if style != "chat":
        raise ValueError(f"unknown prompt style: {style}")

    user_text = CHAT_PROMPTS[variant].format(N=n_max)

    # Render to TEXT first, then tokenise separately.
    #
    # Do not use tokenize=True here. Across transformers versions that call
    # has returned a plain list of ids, a BatchEncoding, and a dict. Iterating
    # a BatchEncoding yields its KEY NAMES, which is how you end up with
    # ValueError: too many dimensions 'str'. Rendering to text is stable
    # across every version.
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
    except TypeError:
        # Template does not accept enable_thinking. Not a reasoning model.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    if not isinstance(text, str):
        raise RuntimeError(
            "apply_chat_template did not return a string. Your transformers "
            f"version returned {type(text)}. Inspect it before continuing.")

    if assistant_prefill:
        # Forcing the answer position. Chatty models otherwise open with
        # "Sure, here's a random number:" and the digit is not at position 1.
        # Report prefill and no-prefill as separate conditions; prefill
        # inflates valid_mass by construction.
        text = text + assistant_prefill

    # add_special_tokens=False because the chat template already contains
    # whatever BOS the model expects. Adding another one shifts the whole
    # distribution.
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return _as_int_list(ids)


def _as_int_list(ids) -> List[int]:
    """Coerce whatever the tokeniser returned into a flat list of ints.

    Tokenisers variously return a list, a nested list, a BatchEncoding, or a
    tensor depending on version and call style. Fail loudly rather than
    letting a wrong shape reach the model.
    """
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, dict):
        ids = ids.get("input_ids", ids)
    while (isinstance(ids, (list, tuple)) and len(ids) == 1
           and isinstance(ids[0], (list, tuple))):
        ids = ids[0]
    out = list(ids)
    if not all(isinstance(x, int) for x in out):
        raise RuntimeError(
            f"Expected a list of token ids, got {out[:5]!r}. This is a "
            "tokeniser API mismatch, not a model result.")
    return out


# =============================================================================
# Candidate enumeration
# =============================================================================

def candidate_token_sequences(tokenizer, n: int,
                              with_leading_space: bool = True
                              ) -> List[Tuple[int, ...]]:
    """All tokenisations of the integer n that we will score.

    THE LEADING SPACE WILL BREAK YOUR RESULTS IF YOU IGNORE IT.
    In most BPE tokenisers "17" and " 17" are different token ids with very
    different probabilities. Which one the model wants depends on whether the
    prompt ends mid-sentence or at a turn boundary. Score both and sum.
    Nothing will look obviously wrong if you get this wrong.
    """
    forms = [str(n)]
    if with_leading_space:
        forms.append(" " + str(n))
    seqs = []
    for form in forms:
        ids = tuple(tokenizer(form, add_special_tokens=False)["input_ids"])
        if ids and ids not in seqs:
            seqs.append(ids)
    return seqs


def terminator_ids(tokenizer) -> List[int]:
    """Token ids that count as 'the model stopped after the number'."""
    out = set()

    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        out.add(eos)
    elif isinstance(eos, (list, tuple)):
        out.update(int(x) for x in eos)

    # Chat-template end markers, which are often NOT the tokeniser's eos.
    for tok_str in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>",
                    "<end_of_turn>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(
                tokenizer, "unk_token_id", -1):
            out.add(tid)

    # A bare newline also ends the answer in completion-style prompting.
    for tok_str in ("\n", "\n\n"):
        ids = tokenizer(tok_str, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out.add(int(ids[0]))

    return sorted(out)


# =============================================================================
# Decoding transforms applied to a single logit row
# =============================================================================

def adjusted_logprobs(logits: torch.Tensor, cfg: DecodeConfig) -> torch.Tensor:
    """Log-probabilities over the vocabulary after temperature, top-k, top-p.

    logits: 1-D float tensor of shape [vocab].

    This is the whole reason the temperature grid is free: the transform is
    applied at every position from stored logits, not re-run through the model.
    """
    logits = logits.float()

    if cfg.temperature <= 0.0:
        out = torch.full_like(logits, float("-inf"))
        out[int(torch.argmax(logits))] = 0.0
        return out

    scaled = logits / cfg.temperature

    if cfg.top_k and cfg.top_k > 0 and cfg.top_k < scaled.numel():
        kth = torch.topk(scaled, cfg.top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))

    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep the smallest set whose cumulative probability reaches top_p.
        # The token that crosses the threshold is kept, matching the standard
        # HF implementation.
        keep = (cum - probs) < cfg.top_p
        drop = sorted_idx[~keep]
        if drop.numel():
            scaled = scaled.index_fill(0, drop, float("-inf"))

    return torch.log_softmax(scaled, dim=-1)


# =============================================================================
# Backend interface
# =============================================================================

class Backend:
    """Minimal interface: tokenise, and return logits for a forced sequence."""

    tokenizer = None

    def logits_for(self, prompt_ids: Sequence[int],
                   continuation_ids: Sequence[int]) -> torch.Tensor:
        """Return logits of shape [len(continuation_ids) + 1, vocab].

        Row i is the model's distribution over the token that comes AFTER
        prompt_ids + continuation_ids[:i]. So row 0 predicts
        continuation_ids[0], and the final row predicts the terminator.
        """
        raise NotImplementedError


class HFBackend(Backend):
    """Hugging Face transformers. Primary, fully implemented backend.

    Use this for safetensors checkpoints in BF16 or FP16, which covers
    Blocks A, B and the *-unpacked Bonsai builds.
    """

    def __init__(self, model_id: str, device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 revision: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            # bfloat16 support on MPS is inconsistent across torch versions.
            dtype = torch.bfloat16 if device == "cuda" else torch.float16
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision)

        # transformers renamed torch_dtype to dtype. Try the new name first
        # and fall back, so this works on both sides of the change.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=dtype)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def logits_for(self, prompt_ids: Sequence[int],
                   continuation_ids: Sequence[int]) -> torch.Tensor:
        """One forward pass over prompt + candidate. No KV cache reuse.

        The earlier version cached the prompt and cropped the cache between
        candidates. That is faster, but if the crop ever fails quietly then
        candidate k is conditioned on candidate k-1 and the output looks
        completely normal while being wrong. The prompt here is about 30
        tokens, so re-running it costs almost nothing. Take the safe version.
        """
        ids = torch.tensor([list(prompt_ids) + list(continuation_ids)],
                           dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=False)

        # Row i predicts the token at position i+1. We want the row that
        # predicts the first candidate token, through to the row that predicts
        # whatever comes after the last one.
        start = len(prompt_ids) - 1
        end = start + len(continuation_ids) + 1
        return out.logits[0, start:end, :].detach().float().cpu()


class LlamaCppBackend(Backend):
    """GGUF via llama-cpp-python. For Block C and the 27B builds.

    WARNING, VERIFY THIS AGAINST YOUR INSTALLED VERSION.
    The llama-cpp-python API for retrieving full logits has changed several
    times (logits_all, eval_logits, scores). This adapter tries the variants
    it knows about. Run --selftest with this backend before using it, and
    cross-check one configuration against HFBackend on a model you can load
    both ways, for example Qwen3-1.7B in BF16 and as Q8_0 GGUF. If those two
    do not agree closely you have an API problem, not a quantisation finding.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048,
                 n_gpu_layers: int = -1):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, logits_all=True,
                         verbose=False)
        self.tokenizer = _LlamaCppTokenizerShim(self.llm)

    def _all_scores(self, n_tokens: int) -> torch.Tensor:
        for attr in ("scores", "_scores", "eval_logits"):
            obj = getattr(self.llm, attr, None)
            if obj is None:
                continue
            arr = torch.as_tensor(obj, dtype=torch.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, self.llm.n_vocab())
            return arr[-n_tokens:] if arr.shape[0] >= n_tokens else arr
        raise RuntimeError(
            "Could not read full logits from llama_cpp. Check the API for "
            "your installed version before going further.")

    def logits_for(self, prompt_ids: Sequence[int],
                   continuation_ids: Sequence[int]) -> torch.Tensor:
        self.llm.reset()
        full = list(prompt_ids) + list(continuation_ids)
        self.llm.eval(full)
        scores = self._all_scores(len(full))
        start = len(prompt_ids) - 1
        return scores[start:start + len(continuation_ids) + 1]


class _LlamaCppTokenizerShim:
    """Just enough tokenizer surface for the enumeration code."""

    def __init__(self, llm):
        self.llm = llm
        self.eos_token_id = llm.token_eos()
        self.unk_token_id = -1

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = self.llm.tokenize(text.encode("utf-8"),
                                add_bos=add_special_tokens,
                                special=False)
        return {"input_ids": list(ids)}

    def convert_tokens_to_ids(self, token: str) -> int:
        ids = self.llm.tokenize(token.encode("utf-8"), add_bos=False,
                                special=True)
        return int(ids[0]) if len(ids) == 1 else -1

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True):
        raise NotImplementedError(
            "llama.cpp chat templates are not wired up. Either pass "
            "--prompt-style completion, or build the chat string yourself "
            "with the HF tokenizer for the same model and feed it as raw "
            "text. Do not guess the template: getting it wrong changes the "
            "distribution.")


# =============================================================================
# The measurement
# =============================================================================

def enumerate_distribution(backend: Backend,
                           prompt_ids: Sequence[int],
                           n_max: int,
                           decode_grid: Sequence[DecodeConfig],
                           ) -> Dict[str, Dict]:
    """Exact P(n) for n in 1..n_max, for every decode configuration.

    Cost: one forward pass over the prompt, plus one short forward pass per
    candidate tokenisation. For n_max=100 that is roughly 150 to 200 short
    passes, not 100 sampled generations.

    Optimisation left undone on purpose: the candidates share token prefixes
    ("1" starts 1, 10..19, 100), so a trie walk would cut the work by roughly
    half. It also makes cache handling much easier to get wrong. Do the simple
    correct thing first.
    """
    tok = backend.tokenizer
    terms = terminator_ids(tok)
    if not terms:
        raise RuntimeError("No terminator tokens found. Without these, P(1) "
                           "and P(17) are not mutually exclusive and every "
                           "number in the output is meaningless.")
    term_idx = torch.tensor(terms, dtype=torch.long)

    # accum[cfg_key][n] = probability mass for integer n
    accum: Dict[str, torch.Tensor] = {
        cfg.key(): torch.zeros(n_max + 1, dtype=torch.float64)
        for cfg in decode_grid
    }

    for n in range(1, n_max + 1):
        for seq in candidate_token_sequences(tok, n):
            logits = backend.logits_for(prompt_ids, seq)   # [len(seq)+1, V]
            for cfg in decode_grid:
                lp = 0.0
                ok = True
                for i, token_id in enumerate(seq):
                    row = adjusted_logprobs(logits[i], cfg)
                    v = float(row[token_id])
                    if not math.isfinite(v):
                        ok = False       # truncated away by top-p or top-k
                        break
                    lp += v
                if not ok:
                    continue
                term_row = adjusted_logprobs(logits[len(seq)], cfg)
                term_lp = float(torch.logsumexp(term_row[term_idx], dim=0))
                if not math.isfinite(term_lp):
                    continue
                accum[cfg.key()][n] += math.exp(lp + term_lp)

    results = {}
    for cfg in decode_grid:
        p = accum[cfg.key()][1:]                 # drop the unused index 0
        valid_mass = float(p.sum())
        cond = (p / valid_mass) if valid_mass > 0 else torch.zeros_like(p)
        results[cfg.key()] = {
            "config": asdict(cfg),
            "valid_mass": valid_mass,
            "p_joint": [float(x) for x in p],
            "p_conditional": [float(x) for x in cond],
            **distribution_metrics(cond.numpy() if hasattr(cond, "numpy")
                                   else cond, n_max),
        }
    return results


def diagnose_position(backend: Backend, prompt_ids: Sequence[int],
                      top: int = 15) -> None:
    """Print what the model actually wants to say at the answer position.

    Run this whenever valid_mass comes out near zero. It answers the only
    question that matters at that point: is the model refusing to produce a
    bare digit, and if so, what is it producing instead?

    Typical culprits you will see here:
        '<think>'   a reasoning model opening its scratchpad
        'Sure'      a chatty model adding preamble
        '**'        a model that formats its answer in markdown
        '\\n'        a stray leading newline from the chat template
    """
    import torch as _t
    logits = backend.logits_for(prompt_ids, [])      # just the first position
    lp = _t.log_softmax(logits[0].float(), dim=-1)
    vals, idx = _t.topk(lp, top)
    print(f"\n  prompt ends with: "
          f"{backend.tokenizer.decode(list(prompt_ids)[-24:])!r}")
    print(f"  top {top} tokens at the answer position:")
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = backend.tokenizer.decode([i])
        print(f"    {tok!r:>18}  p={math.exp(v):.5f}")


def distribution_metrics(p, n_max: int) -> Dict[str, float]:
    """Metrics on a conditional distribution over 1..n_max.

    Total variation distance is the headline number because it reads in plain
    English: the share of probability mass sitting on the wrong integers.
    TV = 0 is perfectly uniform, TV = 1 - 1/N is a point mass.
    """
    p = [float(x) for x in p]
    u = 1.0 / n_max
    total = sum(p)
    if total <= 0:
        return {"tv_distance": float("nan"), "kl_from_uniform": float("nan"),
                "entropy_deficit_nats": float("nan"), "top1_value": -1,
                "top1_prob": float("nan"), "support_size": 0}

    tv = 0.5 * sum(abs(pi - u) for pi in p)
    kl = sum(pi * math.log(pi / u) for pi in p if pi > 0)
    ent = -sum(pi * math.log(pi) for pi in p if pi > 0)
    top_i = max(range(n_max), key=lambda i: p[i])
    support = sum(1 for pi in p if pi > 1e-12)

    return {
        "tv_distance": tv,
        "kl_from_uniform": kl,
        "entropy_deficit_nats": math.log(n_max) - ent,
        "top1_value": top_i + 1,
        "top1_prob": p[top_i],
        "support_size": support,
    }


# =============================================================================
# Controls and validation
# =============================================================================

def control_distribution(n_max: int, n_draws: int = 100_000,
                         kind: str = "secrets") -> Dict[str, float]:
    """Block E control. Must come back with TV near zero.

    If it does not, the bug is in distribution_metrics, not in the models.
    Run this before you run anything else.
    """
    counts = [0] * n_max
    if kind == "secrets":
        for _ in range(n_draws):
            counts[secrets.randbelow(n_max)] += 1
    elif kind == "numpy":
        import numpy as np
        rng = np.random.default_rng(20260905)
        for v in rng.integers(0, n_max, size=n_draws):
            counts[int(v)] += 1
    else:
        raise ValueError(kind)
    p = [c / n_draws for c in counts]
    return distribution_metrics(p, n_max)


@torch.no_grad()
def validate_by_sampling(backend: Backend, prompt_ids: Sequence[int],
                         n_max: int, cfg: DecodeConfig,
                         n_samples: int = 5000) -> Dict:
    """Draw real generations and check they match the enumerated distribution.

    This is the only thing standing between you and a silent tokenisation bug.
    Do it for at least three configurations per model. The empirical
    frequencies should fall inside the Monte Carlo error band of the
    enumerated P_conditional. A systematic gap means the enumeration is wrong,
    most likely the leading-space variant or the terminator set.
    """
    if not isinstance(backend, HFBackend):
        raise NotImplementedError("Sampling validation is wired up for the "
                                  "HF backend only.")
    tok = backend.tokenizer
    ids = torch.tensor([list(prompt_ids)], device=backend.device)
    counts = [0] * n_max
    invalid = 0
    for _ in range(n_samples):
        out = backend.model.generate(
            ids,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-6),
            top_p=cfg.top_p,
            top_k=cfg.top_k if cfg.top_k > 0 else 0,
            max_new_tokens=6,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        text = text.strip()
        if text.isdigit() and 1 <= int(text) <= n_max:
            counts[int(text) - 1] += 1
        else:
            invalid += 1
    valid = n_samples - invalid
    p = [c / valid for c in counts] if valid else [0.0] * n_max
    return {
        "n_samples": n_samples,
        "empirical_valid_rate": valid / n_samples,
        "empirical_counts": counts,
        **distribution_metrics(p, n_max),
    }


def selftest() -> None:
    """Checks that need no model. Run this first."""
    print("control, secrets, N=30 :", control_distribution(30, 100_000))
    print("control, numpy,   N=100:",
          control_distribution(100, 100_000, kind="numpy"))

    # A point mass must give TV = 1 - 1/N.
    n = 30
    point = [0.0] * n
    point[16] = 1.0                       # all mass on the integer 17
    m = distribution_metrics(point, n)
    expected = 1.0 - 1.0 / n
    assert abs(m["tv_distance"] - expected) < 1e-9, m
    assert m["top1_value"] == 17, m
    print("point mass at 17, N=30 :", m)

    # Exact uniform must give TV = 0.
    m = distribution_metrics([1.0 / n] * n, n)
    assert abs(m["tv_distance"]) < 1e-12, m
    print("exact uniform,    N=30 :", m)

    # Decoding transforms.
    logits = torch.tensor([2.0, 1.0, 0.0, -5.0])
    lp = adjusted_logprobs(logits, DecodeConfig(temperature=1.0))
    assert abs(float(torch.logsumexp(lp, 0))) < 1e-5
    lp = adjusted_logprobs(logits, DecodeConfig(temperature=0.0))
    assert float(lp[0]) == 0.0 and float(lp[1]) == float("-inf")
    lp = adjusted_logprobs(logits, DecodeConfig(temperature=1.0, top_p=0.5))
    assert float(lp[3]) == float("-inf")
    lp = adjusted_logprobs(logits, DecodeConfig(temperature=1.0, top_k=2))
    assert float(lp[2]) == float("-inf")
    print("decoding transforms    : ok")
    print("\nselftest passed")


# =============================================================================
# Runner
# =============================================================================

def run(args: argparse.Namespace) -> None:
    if args.gguf:
        backend: Backend = LlamaCppBackend(args.gguf,
                                           n_gpu_layers=args.n_gpu_layers)
        model_label = os.path.basename(args.gguf)
    else:
        backend = HFBackend(args.model, device=args.device,
                            revision=args.revision)
        model_label = args.model

    grid = DEFAULT_DECODE_GRID if args.full_grid else [
        DecodeConfig(temperature=0.0),
        DecodeConfig(temperature=0.7, top_p=1.0),    # model prior, untruncated
        DecodeConfig(temperature=0.7, top_p=0.95),   # typical default sampler
        DecodeConfig(temperature=1.0, top_p=1.0),
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for n_max in args.ranges:
            for variant in args.prompt_variants:
                prompt_ids = build_prompt_ids(
                    backend.tokenizer, n_max, args.prompt_style, variant,
                    assistant_prefill=args.prefill,
                    enable_thinking=args.enable_thinking)

                if args.diagnose:
                    diagnose_position(backend, prompt_ids)

                res = enumerate_distribution(backend, prompt_ids, n_max, grid)

                # A tiny valid_mass means the model almost never emits a bare
                # digit at this position, so the conditional distribution is a
                # ratio of near-zero numbers. It is numerically meaningless.
                # Warn loudly rather than let it into the dataset unremarked.
                worst = max(p["valid_mass"] for p in res.values())
                if worst < 1e-4:
                    print(f"\n  WARNING: valid_mass is {worst:.2e}. The model "
                          f"is not producing a bare integer here, so every "
                          f"number below is a ratio of near-zero values and "
                          f"means nothing.")
                    print("  Re-run with --diagnose to see what it emits "
                          "instead, and with --no-think if this is a "
                          "reasoning model such as Qwen3.")
                    if not args.diagnose:
                        diagnose_position(backend, prompt_ids)

                for cfg_key, payload in res.items():
                    record = {
                        "model": model_label,
                        "revision": args.revision,
                        "backend": "gguf" if args.gguf else "hf",
                        "prompt_style": args.prompt_style,
                        "prompt_variant": variant,
                        "prefill": args.prefill,
                        "enable_thinking": args.enable_thinking,
                        "n_max": n_max,
                        "decode": cfg_key,
                        **payload,
                    }
                    fh.write(json.dumps(record) + "\n")
                    print(f"{model_label} N={n_max} {variant} {cfg_key} "
                          f"TV={payload['tv_distance']:.4f} "
                          f"top1={payload['top1_value']} "
                          f"({payload['top1_prob']:.3f}) "
                          f"valid_mass={payload['valid_mass']:.3e}")

    if args.validate:
        cfg = DecodeConfig(temperature=0.7, top_p=1.0)
        prompt_ids = build_prompt_ids(backend.tokenizer, args.ranges[0],
                                      args.prompt_style,
                                      args.prompt_variants[0],
                                      assistant_prefill=args.prefill,
                                      enable_thinking=args.enable_thinking)
        v = validate_by_sampling(backend, prompt_ids, args.ranges[0], cfg,
                                 n_samples=args.validate)
        print("\nsampling validation:", json.dumps(v)[:400])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--model", default=None,
                   help="HF repo id, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default=None,
                   help="commit hash. Record it. 'main' is not reproducible.")
    p.add_argument("--gguf", default=None, help="path to a .gguf file")
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--device", default=None, choices=[None, "cuda", "mps",
                                                      "cpu"])
    p.add_argument("--ranges", type=int, nargs="+", default=DEFAULT_RANGES)
    p.add_argument("--prompt-style", default="chat",
                   choices=["chat", "completion"])
    p.add_argument("--prompt-variants", nargs="+", default=["primary"],
                   choices=list(CHAT_PROMPTS))
    p.add_argument("--prefill", default="",
                   help='e.g. "Answer: " to force the answer position')
    p.add_argument("--no-think", dest="enable_thinking", action="store_false",
                   default=None,
                   help="For reasoning models such as Qwen3. Closes the "
                        "<think> block in the chat template so the answer "
                        "starts at the position being measured. Without this "
                        "the first token is <think> and valid_mass is ~0.")
    p.add_argument("--think", dest="enable_thinking", action="store_true",
                   help="Force thinking ON. Report as a separate condition.")
    p.add_argument("--diagnose", action="store_true",
                   help="Print the model's top next tokens at the answer "
                        "position before measuring.")
    p.add_argument("--full-grid", action="store_true",
                   help="all 29 decode configs instead of 4. Costs nothing "
                        "extra in inference.")
    p.add_argument("--validate", type=int, default=0, metavar="N_SAMPLES")
    p.add_argument("--out", default="results/exact.jsonl")
    p.add_argument("--selftest", action="store_true")

    args = p.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if not args.model and not args.gguf:
        p.error("pass --model or --gguf, or --selftest")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
