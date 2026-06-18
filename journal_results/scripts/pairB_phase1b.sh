#!/bin/bash
# Pair B — Phase 1b: dpo_baseline + weight_lite
# Launch after tea_pos finishes (tea_neg can still run on its GPUs).
#
# GPU allocation:
#   dpo_baseline: GPU 4 (or 4,5 if both free — Llama-1B fits on 1 GPU)
#   weight_lite:   GPU 5 (inference only — SFT Mistral teacher)
#   tea_neg:       keeps GPU 6,7 (do NOT kill)
#
# Run: CUDA_VISIBLE_DEVICES=X bash pairB_phase1b.sh dpo_baseline
#      CUDA_VISIBLE_DEVICES=Y bash pairB_phase1b.sh weight_lite
set -euo pipefail

ROOT=~/CTPD-2
JR=$ROOT/journal_results
TEACHER=$JR/models/mistralai_Mistral-7B-Instruct-v0.3
STUDENT=$JR/models/meta-llama_Llama-3.2-1B-Instruct
DATA=$JR/data/pairB
mkdir -p $JR/weights $JR/logs
cd $ROOT/src/prefkd

STAGE=${1:-help}

case "$STAGE" in

# ==========================================================================
# DPO baseline — student (Llama-3.2-1B) trained with standard DPO
# Fits on 1 GPU (~10 GB VRAM)
# ==========================================================================
dpo_baseline)
  python3 -u train.py model=dpo \
    model.policy_name_or_path=$STUDENT \
    model.reference_name_or_path=$STUDENT \
    model.teacher_tokenizer_name_or_path=$STUDENT \
    model.student_tokenizer_name_or_path=$STUDENT \
    model.original_policy_name=$STUDENT \
    model.policy_block_name=LlamaDecoderLayer \
    model.reference_block_name=LlamaDecoderLayer \
    loss=dpo log_dir=pairB_dpo_baseline \
    policy_mode=student reference_mode=student \
    loss.beta=0.1 loss.label_smoothing=0 n_epochs=1 max_grad_norm=1.0 \
    gradient_accumulation_steps=1 batch_size=16 eval_batch_size=16 \
    total_steps=3541 warmup_steps=177 eval_every=566 \
    lr=1e-6 scheduler=cosine save_checkpoint=true \
    trainer=FSDPTrainer sample_during_eval=false \
    datasets=$DATA ;;

# ==========================================================================
# CTPD-Lite span weights — single SFT teacher (no DPO training needed)
# Inference only, 1 GPU
# ==========================================================================
weight_lite)
  python3 -u $JR/scripts/weight_lite.py \
    --teacher_model $TEACHER --student_model $STUDENT \
    --data_path $DATA --split train \
    --output_dir $JR/weights/pairB_lite \
    --batch_size 8 --num_gpus 1 \
    --average_log_prob 0 --student_all 1 --mu 1.0 ;;

*)
  echo "usage: $0 {dpo_baseline|weight_lite}" ;;
esac
