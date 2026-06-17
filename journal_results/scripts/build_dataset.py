#!/usr/bin/env python3
"""Build the aligned-span CTPD dataset for an arbitrary teacher/student pair.

Consolidates the 4 manual notebook stages (processed-preference-datasets.ipynb +
gen-parent-and-dummy-weight.ipynb) into one parameterized, runnable script:

  1. tokenize_batch on the STUDENT tokenizer  -> chosen/rejected_student_*
  2. tokenize_batch on the TEACHER tokenizer  -> chosen/rejected_teacher_*
  3. batch_find_parent_token (chosen + rejected) -> *_parent_dict  (aligned spans)
  4. dict->list conversion + dummy unit weights -> *_parent_list, *_weight

Saves to disk (save_to_disk) instead of push_to_hub so new pairs need no Hub
write access. Output feeds weight.py / weight_lite.py / CTPD.sh directly.
"""
import argparse
import json
import os

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


def _convert_tolist(examples):
    for k in list(examples):
        if "parent" in k and "dict" in k:
            d = [json.loads(i) for i in examples[k]]
            examples[k.replace("dict", "list")] = [list(dd.values()) for dd in d]
    return examples


def _add_dummy_weight(example):
    example["chosen_weight"] = [1] * len(example["chosen_student_parent_list"])
    example["rejected_weight"] = [1] * len(example["rejected_student_parent_list"])
    return example


def build_split(raw, student_tok, teacher_tok, max_length, max_prompt_length):
    ds = _tok(raw, student_tok, max_length, max_prompt_length).rename_columns(STUDENT_RENAME)
    # teacher EOS </s> can appear literally in prompts and breaks alignment
    ds = ds.filter(lambda x: "</s>" not in x["prompt"])
    ds = _tok(ds, teacher_tok, max_length, max_prompt_length).rename_columns(TEACHER_RENAME)
    ds = ds.filter(lambda x: x["rejected"] != "")
    ds = ds.map(lambda ex: batch_find_parent_token(ex, mode="chosen"),
                batched=True, batch_size=32, num_proc=8)
    ds = ds.map(lambda ex: batch_find_parent_token(ex, mode="rejected"),
                batched=True, batch_size=32, num_proc=8)
    ds = ds.map(_convert_tolist, batched=True, batch_size=32, num_proc=8)
    ds = ds.map(_add_dummy_weight, batched=False, num_proc=8)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hf_dataset", default="exlaw/tis-dpo-data")
    ap.add_argument("--train_file", default="ultra-feedback/train.jsonl")
    ap.add_argument("--test_file", default="ultra-feedback/test.jsonl")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=128)
    ap.add_argument("--limit_train", type=int, default=0, help="0 = all")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    student_tok = AutoTokenizer.from_pretrained(args.student_model)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)

    raw = load_dataset(
        args.hf_dataset,
        data_files={"train": args.train_file, "test": args.test_file},
    )
    for split in args.splits.split(","):
        split = split.strip()
        src = raw[split]
        if split == "train" and args.limit_train > 0:
            src = src.select(range(min(args.limit_train, len(src))))
        print(f"[{split}] building {len(src)} examples", flush=True)
        out = build_split(src, student_tok, teacher_tok, args.max_length, args.max_prompt_length)
        dest = os.path.join(args.output_dir, split)
        os.makedirs(dest, exist_ok=True)
        out.save_to_disk(dest)
        print(f"[{split}] saved -> {dest} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
