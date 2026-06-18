#!/usr/bin/env python3
"""Build CTPD dataset with label-noise variants for robustness experiments.

Produces datasets where N% of preference labels are flipped, e.g.:
  train_5pct.jsonl  -> 5% flipped
  train_10pct.jsonl -> 10% flipped
  train_20pct.jsonl -> 20% flipped
  train_30pct.jsonl -> 30% flipped
  train_clean.jsonl -> 0% flipped (equal to the standard build)
"""
import argparse
import hashlib
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "src", "prefkd", "utils"))

from datasets import load_dataset
from parent_token_dict import batch_find_parent_token
from tokenize_func import tokenize_batch
from transformers import AutoTokenizer

STUDENT_RENAME = {
    "chosen_input_ids": "chosen_student_input_ids",
    "chosen_attention_mask": "chosen_student_attention_mask",
    "chosen_labels": "chosen_student_labels",
    "chosen_offset_mapping": "chosen_student_offset_mapping",
    "rejected_input_ids": "rejected_student_input_ids",
    "rejected_attention_mask": "rejected_student_attention_mask",
    "rejected_labels": "rejected_student_labels",
    "rejected_offset_mapping": "rejected_student_offset_mapping",
    "prompt_input_ids": "prompt_student_input_ids",
    "prompt_attention_mask": "prompt_student_attention_mask",
}
TEACHER_RENAME = {k: v.replace("student", "teacher") for k, v in STUDENT_RENAME.items()}


def _tok(ds, tokenizer, max_length, max_prompt_length):
    return ds.map(
        lambda ex: tokenize_batch(
            prompts=ex["prompt"], chosens=ex["chosen"], rejecteds=ex["rejected"],
            truncation_mode="keep_start", tokenizer=tokenizer,
            max_length=max_length, max_prompt_length=max_prompt_length,
        ),
        batched=True, batch_size=32, num_proc=8,
    )


def _align_and_listify(examples, mode):
    examples = batch_find_parent_token(examples, mode)
    for side in ("student", "teacher"):
        dcol = f"{mode}_{side}_parent_dict"
        lcol = f"{mode}_{side}_parent_list"
        examples[lcol] = [list(d.values()) for d in examples[dcol]]
        del examples[dcol]
    return examples


def _add_dummy_weight(example):
    example["chosen_weight"] = [1] * len(example["chosen_student_parent_list"])
    example["rejected_weight"] = [1] * len(example["rejected_student_parent_list"])
    return example


def build_split(raw, student_tok, teacher_tok, max_length, max_prompt_length):
    ds = _tok(raw, student_tok, max_length, max_prompt_length).rename_columns(STUDENT_RENAME)
    ds = ds.filter(lambda x: "</s>" not in x["prompt"])
    ds = _tok(ds, teacher_tok, max_length, max_prompt_length).rename_columns(TEACHER_RENAME)
    ds = ds.filter(lambda x: x["rejected"] != "")
    for side in ("chosen", "rejected"):
        ds = ds.filter(
            lambda x, s=side: len(x[f"{s}_student_offset_mapping"]) > 0
            and len(x[f"{s}_teacher_offset_mapping"]) > 0
        )
    ds = ds.map(lambda ex: _align_and_listify(ex, "chosen"),
                batched=True, batch_size=32, num_proc=8)
    ds = ds.map(lambda ex: _align_and_listify(ex, "rejected"),
                batched=True, batch_size=32, num_proc=8)
    ds = ds.map(_add_dummy_weight, batched=False, num_proc=8)
    return ds


def flip_labels(dataset, noise_pct, seed=42):
    """Flip chosen <-> rejected for noise_pct% of examples deterministically."""
    import random
    rng = random.Random(seed)
    n = len(dataset)
    n_flip = int(n * noise_pct / 100)

    # Deterministic selection based on prompt hash
    indices = list(range(n))
    rng.shuffle(indices)
    flip_set = set(indices[:n_flip])

    def _flip(ex, idx):
        if idx in flip_set:
            ex["chosen"], ex["rejected"] = ex["rejected"], ex["chosen"]
        return ex

    return dataset.map(_flip, with_indices=True, batched=False, num_proc=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hf_dataset", default="exlaw/tis-dpo-data")
    ap.add_argument("--train_file", default="ultra-feedback/train.jsonl")
    ap.add_argument("--noise_levels", default="0,5,10,20,30",
                    help="Comma-separated noise percentages")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=128)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    student_tok = AutoTokenizer.from_pretrained(args.student_model)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)

    raw = load_dataset(
        args.hf_dataset,
        data_files={"train": args.train_file},
    )
    src = raw["train"]
    if args.limit_train > 0:
        src = src.select(range(min(args.limit_train, len(src))))

    print(f"Loaded {len(src)} raw examples", flush=True)

    for level_str in args.noise_levels.split(","):
        noise = float(level_str.strip())
        label = "clean" if noise == 0 else f"{int(noise)}pct"

        # Flip labels on the RAW text (before tokenization)
        ds = flip_labels(src, noise, seed=args.seed) if noise > 0 else src
        print(f"[noise={label}] building {len(ds)} examples", flush=True)

        out = build_split(ds, student_tok, teacher_tok,
                         args.max_length, args.max_prompt_length)

        jsonl_path = os.path.join(args.output_dir, f"train_{label}.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        out.to_json(jsonl_path)
        print(f"[noise={label}] saved -> {jsonl_path} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
