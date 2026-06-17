#!/usr/bin/env python3
"""Tokenizer-mismatch analysis for CTPD journal extension.

For each teacher->student pair, tokenize UltraFeedback responses with BOTH
tokenizers, compute aligned spans via the repo's own find_parent_token, and
report mismatch statistics:
  - vocab overlap (Jaccard over token strings, %)
  - mean teacher / student tokens per aligned span
  - corpus-level teacher:student token ratio
  - span-type distribution (1:1, 1:n, n:1, n:m)
  - mean aligned-span length in characters

No GPU required: this is purely a tokenizer/offset-map computation.
Run inside the `train` conda env from the CTPD-2 repo root.
"""
import argparse
import json
import os
import sys

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Reuse the exact aligned-span algorithm used by the training pipeline.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "prefkd"))
from utils.parent_token_dict import find_parent_token  # noqa: E402

# teacher -> student pairs from the journal draft (Section: Experimental Setup).
PAIRS = [
    ("Qwen-2.5-14B->Llama-3.1-8B",   "Qwen/Qwen2.5-14B",        "meta-llama/Llama-3.1-8B"),
    ("Qwen-2.5-7B->Llama-3.2-1B",    "Qwen/Qwen2.5-7B",         "meta-llama/Llama-3.2-1B"),
    ("Gemma-2-9B->Llama-3.2-1B",     "google/gemma-2-9b",       "meta-llama/Llama-3.2-1B"),
    ("Qwen-2.5-7B->Gemma-2-2B",      "Qwen/Qwen2.5-7B",         "google/gemma-2-2b"),
    ("Mistral-7B->Llama-3.2-1B",     "mistralai/Mistral-7B-v0.3","meta-llama/Llama-3.2-1B"),
    ("Llama-3.1-8B->Qwen-2.5-1.5B",  "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-1.5B"),
    ("Llama-3.1-8B->Llama-3.2-1B",   "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B"),
]


def classify_span(n_teacher, n_student):
    if n_teacher == 1 and n_student == 1:
        return "1:1"
    if n_teacher == 1 and n_student > 1:
        return "1:n"
    if n_teacher > 1 and n_student == 1:
        return "n:1"
    return "n:m"


def vocab_overlap_jaccard(tok_t, tok_s):
    vt = set(tok_t.get_vocab().keys())
    vs = set(tok_s.get_vocab().keys())
    inter = len(vt & vs)
    union = len(vt | vs)
    return 100.0 * inter / union if union else 0.0


def analyze_text(text, tok_t, tok_s, acc):
    """Tokenize one response with both tokenizers, accumulate span stats into acc."""
    if not text or not text.strip():
        return
    enc_t = tok_t(text, add_special_tokens=False, return_offsets_mapping=True)
    enc_s = tok_s(text, add_special_tokens=False, return_offsets_mapping=True)
    off_t = enc_t["offset_mapping"]
    off_s = enc_s["offset_mapping"]
    if len(off_t) == 0 or len(off_s) == 0:
        return
    # find_parent_token signature: (student_offset_map, teacher_offset_map)
    student_parent, teacher_parent = find_parent_token(off_s, off_t)

    acc["n_teacher_tokens"] += len(off_t)
    acc["n_student_tokens"] += len(off_s)

    for span in student_parent:
        n_s = len(student_parent[span])
        n_t = len(teacher_parent.get(span, []))
        if n_s == 0 and n_t == 0:
            continue
        acc["span_count"] += 1
        acc["sum_t_per_span"] += n_t
        acc["sum_s_per_span"] += n_s
        acc["sum_span_chars"] += (span[1] - span[0])
        acc["types"][classify_span(n_t, n_s)] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=2000,
                    help="number of preference examples to sample")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--only_pair", type=str, default=None,
                    help="run a single pair by name substring")
    args = ap.parse_args()

    print(f"Loading UltraFeedback ({args.split}) from exlaw/tis-dpo-data ...", flush=True)
    ds = load_dataset(
        "exlaw/tis-dpo-data",
        data_files={"train": "ultra-feedback/train.jsonl",
                    "test": "ultra-feedback/test.jsonl"},
        split=args.split,
    )
    if args.n_samples and args.n_samples < len(ds):
        ds = ds.select(range(args.n_samples))
    print(f"Using {len(ds)} examples (chosen+rejected -> {2*len(ds)} responses).", flush=True)

    tok_cache = {}

    def get_tok(name):
        if name not in tok_cache:
            tok_cache[name] = AutoTokenizer.from_pretrained(name)
        return tok_cache[name]

    results = {}
    for pair_name, teacher_id, student_id in PAIRS:
        if args.only_pair and args.only_pair not in pair_name:
            continue
        print(f"\n=== {pair_name} ===", flush=True)
        print(f"  teacher={teacher_id}  student={student_id}", flush=True)
        tok_t = get_tok(teacher_id)
        tok_s = get_tok(student_id)

        acc = {
            "n_teacher_tokens": 0, "n_student_tokens": 0,
            "span_count": 0, "sum_t_per_span": 0, "sum_s_per_span": 0,
            "sum_span_chars": 0,
            "types": {"1:1": 0, "1:n": 0, "n:1": 0, "n:m": 0},
        }
        for ex in ds:
            for fld in ("chosen", "rejected"):
                analyze_text(ex.get(fld, ""), tok_t, tok_s, acc)

        sc = max(acc["span_count"], 1)
        overlap = vocab_overlap_jaccard(tok_t, tok_s)
        total_types = sum(acc["types"].values()) or 1
        row = {
            "pair": pair_name,
            "teacher": teacher_id,
            "student": student_id,
            "vocab_overlap_jaccard_pct": round(overlap, 1),
            "vocab_size_teacher": tok_t.vocab_size,
            "vocab_size_student": tok_s.vocab_size,
            "tok_per_span_teacher": round(acc["sum_t_per_span"] / sc, 3),
            "tok_per_span_student": round(acc["sum_s_per_span"] / sc, 3),
            "ts_ratio": round(acc["n_teacher_tokens"] / max(acc["n_student_tokens"], 1), 3),
            "mean_span_chars": round(acc["sum_span_chars"] / sc, 3),
            "n_spans": acc["span_count"],
            "pct_1to1": round(100.0 * acc["types"]["1:1"] / total_types, 2),
            "pct_1ton": round(100.0 * acc["types"]["1:n"] / total_types, 2),
            "pct_nto1": round(100.0 * acc["types"]["n:1"] / total_types, 2),
            "pct_ntom": round(100.0 * acc["types"]["n:m"] / total_types, 2),
        }
        results[pair_name] = row
        print(json.dumps(row, indent=2), flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (written to {args.out})", flush=True)

    print("\nALL DONE. Summary:", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
