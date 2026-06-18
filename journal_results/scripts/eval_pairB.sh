#!/bin/bash
# Phase 4: Evaluate all Pair B checkpoints.
# Uses lm-evaluation-harness for Open LLM Leaderboard + MT-Bench.
set -euo pipefail

ROOT=~/CTPD-2
JR=$ROOT/journal_results
EVAL_DIR=$JR/efficiency
mkdir -p $EVAL_DIR
cd $ROOT/src/prefkd

MODELS=(
  # baseline student before any training
  "$JR/models/meta-llama_Llama-3.2-1B-Instruct:pairB_student_sft"
  # DPO baseline (Phase 1b output)
  "output/dpo_meta-llama_Llama-3.2-1B-Instruct_pairB_*/checkpoint-3541:pairB_dpo_baseline"  # need concrete path
  # CTPD-Lite (Phase 3 output)
  "output/KD_tisdpo_meta-llama_Llama-3.2-1B-Instruct_pairB_CTPD_lite_*/checkpoint-7082:pairB_ctpd_lite"
  # Full CTPD (Phase 3 output)
  "output/KD_tisdpo_meta-llama_Llama-3.2-1B-Instruct_pairB_CTPD_full_*/checkpoint-7082:pairB_ctpd_full"
)

LLM_TASKS="arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande,mmlu"

for entry in "${MODELS[@]}"; do
  model_path="${entry%%:*}"
  label="${entry##*:}"

  # Expand wildcard if checkpoint dir not yet known
  for mp in $model_path; do
    if [ -d "$mp" ]; then
      echo "=== Evaluating $label at $mp ==="

      # Open LLM Leaderboard
      lm_eval --model hf \
        --model_args pretrained=$mp,dtype=float16 \
        --tasks $LLM_TASKS \
        --batch_size auto \
        --output_path $EVAL_DIR/${label}_openllm \
        --log_samples 2>&1 | tee $EVAL_DIR/${label}_openllm.log

      # MT-Bench (single-turn for speed)
      python3 -m fastchat.llm_judge.gen_model_answer \
        --model-path $mp \
        --model-id $label \
        --bench-name mt_bench \
        2>&1 | tee $EVAL_DIR/${label}_mtbench.log
    fi
  done
done

echo "Phase 4 evaluation complete. Results in $EVAL_DIR/"
