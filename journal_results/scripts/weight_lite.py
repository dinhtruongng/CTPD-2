#!/usr/bin/env python3
"""CTPD-Lite weight generation: single-teacher cached span weights.

Full CTPD (weight.py) trains a positive teacher (DPO) and a negative teacher
(reverse-DPO), then scores each aligned span with
    w_i = exp(mu * clamp(log pi_T+(p_i) - log pi_T-(p_i), L, U)).
That needs TWO extra fine-tuned teachers and 2N teacher forward passes.

CTPD-Lite replaces the contrastive teacher pair with a SINGLE SFT teacher and a
chosen-vs-rejected margin (journal plan, Section "CTPD-Lite"):
    s_i      = log pi_T(p_i^w | x, p_{<i}^w) - log pi_T(p_i^l | x, p_{<i}^l)
    w_i^w    = exp(+mu * clamp(s_i, L, U))      # chosen span weight
    w_i^l    = exp(-mu * clamp(s_i, L, U))      # rejected span weight
Spans are paired by index; spans beyond min(len_w, len_l) get neutral weight 1.

This needs NO teacher DPO training and only N teacher forward passes per side.
Output columns (chosen_true_weight / rejected_true_weight) match weight.py, so
the resulting dataset feeds the unchanged CTPD.sh training stage directly --
the cache makes the weight source invisible to the optimizer.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

# Ensure CTPD-2/src/prefkd is on sys.path for `from loss.loss import ...`
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKGDIR = os.path.join(_SCRIPT_DIR, "..", "..", "src", "prefkd")
if _PKGDIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKGDIR))

import torch
from datasets import concatenate_datasets, load_from_disk
from loss.loss import get_ptoken_logps, prompt_remove
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.preference_datasets import get_collate_fn


def _span_logps(model, inputs, labels, parent_list, average_log_prob, device):
    """Per-span summed log-probs under a single model. Shape: (batch, n_spans)."""
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.log_softmax(logits, dim=-1)
    no_prompt_logits, no_prompt_labels = prompt_remove(logits, labels, inputs["input_ids"])
    return get_ptoken_logps(
        no_prompt_logits, no_prompt_labels, parent_list, average_log_prob=average_log_prob
    )


def lite_span_weights(
    teacher_model,
    tokenizer,
    samples,
    average_log_prob,
    student_all,
    batch_size=8,
    mu=1.0,
    k=1.0,
    L=-0.5,
    U=1.5,
    device=None,
    process_id=None,
):
    """Compute CTPD-Lite chosen/rejected span weights from one teacher.

    Returns (chosen_weights, rejected_weights), each a list of per-span lists.
    """
    if device is None:
        device = next(teacher_model.parameters()).device
    collate_fn = get_collate_fn(tokenizer)
    dataloader = DataLoader(
        samples, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    dataloader = tqdm(
        dataloader, total=len(dataloader), desc=f"GPU-{process_id or 0} [lite]",
        unit="batch", position=process_id or 0, leave=False,
        disable=not torch.cuda.is_available(),
    )

    # ALWAYS feed teacher-tokenized inputs to the teacher model — student
    # token IDs may be OOB when teacher vocab < student vocab (e.g. Mistral 32K
    # vs Llama 128K). The parent_list span structure is identical regardless.
    model_side = "teacher"

    chosen_all, rejected_all = [], []
    for batch in dataloader:
        c_inputs = {
            "input_ids": batch[f"chosen_{model_side}_input_ids"].to(device),
            "attention_mask": batch[f"chosen_{model_side}_attention_mask"].to(device),
        }
        r_inputs = {
            "input_ids": batch[f"rejected_{model_side}_input_ids"].to(device),
            "attention_mask": batch[f"rejected_{model_side}_attention_mask"].to(device),
        }
        c_labels = batch[f"chosen_{model_side}_labels"].to(device)
        r_labels = batch[f"rejected_{model_side}_labels"].to(device)
        c_parent = batch[f"chosen_{model_side}_parent_list"].to(device)
        r_parent = batch[f"rejected_{model_side}_parent_list"].to(device)

        c_logps = _span_logps(teacher_model, c_inputs, c_labels, c_parent, average_log_prob, device)
        r_logps = _span_logps(teacher_model, r_inputs, r_labels, r_parent, average_log_prob, device)

        # pair spans by index; truncate to the shorter side for the margin
        n_c = c_logps.shape[1]
        n_r = r_logps.shape[1]
        n = min(n_c, n_r)
        s = torch.clamp(c_logps[:, :n] - r_logps[:, :n], L, U)  # (batch, n)

        c_w = k * torch.exp(mu * s)    # chosen span weight
        r_w = k * torch.exp(-mu * s)   # rejected span weight (reciprocal of chosen)
        c_w = torch.round(c_w * 100) / 100
        r_w = torch.round(r_w * 100) / 100

        # spans beyond the paired region get neutral weight 1.0
        for b in range(c_logps.shape[0]):
            cw = c_w[b].tolist() + [1.0] * (n_c - n)
            rw = r_w[b].tolist() + [1.0] * (n_r - n)
            chosen_all.append(cw)
            rejected_all.append(rw)

    return chosen_all, rejected_all


def process_dataset_shard(
    gpu_id, teacher_model_name, student_model, data_shard,
    average_log_prob, student_all, batch_size=8, mu=1.0,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}", flush=True)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenizer = {"student": student_tokenizer, "teacher": teacher_tokenizer}

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype="float16", low_cpu_mem_usage=True,
    ).to(device)

    print(f"GPU {gpu_id}: Processing {len(data_shard)} examples", flush=True)
    chosen_weights, rejected_weights = lite_span_weights(
        model, tokenizer, data_shard,
        average_log_prob=average_log_prob, student_all=student_all,
        batch_size=batch_size, device=device, process_id=gpu_id, mu=mu,
    )

    def add_weight_col(example, index):
        example["chosen_true_weight"] = chosen_weights[index]
        example["rejected_true_weight"] = rejected_weights[index]
        return example

    data_shard = data_shard.map(
        lambda ex, idx: add_weight_col(ex, idx), with_indices=True, batched=False, num_proc=8,
    )
    del model
    torch.cuda.empty_cache()
    return data_shard


def parallel_process_file(args):
    data = load_from_disk(args.data_path)
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")
    print(f"Using {num_gpus} GPUs (available: {available_gpus})", flush=True)

    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus
    for i in range(0, len(data), shard_size):
        shards.append(data.select(range(i, min(i + shard_size, len(data)))))
    shards = shards[:num_gpus]
    print(f"Split data into {len(shards)} shards", flush=True)

    if args.force_sequential or len(shards) == 1:
        results = [
            process_dataset_shard(
                i % available_gpus, args.teacher_model, args.student_model,
                shards[i], args.average_log_prob, args.student_all, args.batch_size, args.mu,
            )
            for i in range(len(shards))
        ]
    else:
        with mp.Pool(num_gpus) as pool:
            async_res = [
                pool.apply_async(
                    process_dataset_shard,
                    args=(i % available_gpus, args.teacher_model, args.student_model,
                          shards[i], args.average_log_prob, args.student_all,
                          args.batch_size, args.mu),
                )
                for i in range(len(shards))
            ]
            results = [r.get() for r in async_res]

    processed = concatenate_datasets(results)
    out_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)
    print(f"Saved CTPD-Lite weighted data to {out_dir}", flush=True)
    return out_dir


def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="CTPD-Lite single-teacher span weights.")
    ap.add_argument("--teacher_model", type=str, required=True,
                    help="Single SFT teacher (no DPO teachers needed).")
    ap.add_argument("--student_model", type=str, required=True,
                    help="Student model, for its tokenizer.")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--average_log_prob", type=int, default=0)
    ap.add_argument("--student_all", type=int, default=1)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--force_sequential", action="store_true")
    args = ap.parse_args()

    if torch.cuda.device_count() == 0:
        raise RuntimeError("No GPU devices available")

    start = time.time()
    out = parallel_process_file(args)
    print(f"Finished in {time.time() - start:.2f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
