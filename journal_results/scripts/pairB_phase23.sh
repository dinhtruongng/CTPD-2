#!/bin/bash
# Pair B — Phase 2+3: weight_full + train_ctpd + train_lite
# Requires tea_pos AND tea_neg checkpoints from Phase 1a.
#
# LAUNCH:
#   Phase 2: CUDA_VISIBLE_DEVICES=4,5,6,7 bash pairB_phase23.sh weight_full POS_CKPT NEG_CKPT
#   Phase 3a: CUDA_VISIBLE_DEVICES=4,5,6,7 bash pairB_phase23.sh train_ctpd FULL_WEIGHT_DIR
#   Phase 3b: bash pairB_phase23.sh train_lite LITE_WEIGHT_DIR
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
# Phase 2 — full CTPD span weights (pos vs neg teacher contrastive)
# 4 GPUs, 2 teacher models fit in ~280 GB across 4 H200s
# ==========================================================================
weight_full)
  POS=${2:?pass positive ckpt dir}
  NEG=${3:?pass negative ckpt dir}
  python3 -u weight.py \
    --positive_model_name $POS --negative_model_name $NEG \
    --student_model $STUDENT \
    --data_path $DATA --output_dir $JR/weights/pairB_full \
    --split train --batch_size 8 --num_gpus 4 \
    --average_log_prob 0 --teacher_all 0 --student_all 1 ;;

# ==========================================================================
# Phase 3a — train CTPD (full) student
# 4 GPUs FSDP
# ==========================================================================
train_ctpd)
  WDATA=${2:?pass weighted dataset dir}
  python3 -u train.py model=KD_tisdpo \
    model.policy_name_or_path=$STUDENT \
    model.reference_name_or_path=$TEACHER \
    model.teacher_tokenizer_name_or_path=$TEACHER \
    model.student_tokenizer_name_or_path=$STUDENT \
    model.teacher_name_or_path=$TEACHER \
    model.student_name_or_path=$STUDENT \
    model.original_policy_name=$STUDENT \
    model.policy_block_name=LlamaDecoderLayer \
    model.reference_block_name=MistralDecoderLayer \
    loss=KD_tisdpo log_dir=pairB_CTPD_full \
    policy_mode=student reference_mode=teacher \
    loss.beta=0.1 loss.label_smoothing=0 loss.average_log_prob=false \
    n_epochs=1 max_grad_norm=1.0 \
    gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
    total_steps=7082 warmup_steps=708 eval_every=566 \
    lr=1e-6 scheduler=cosine transform.method=origin \
    trainer=FSDPTrainer sample_during_eval=false \
    datasets=$WDATA ;;

# ==========================================================================
# Phase 3b — train CTPD-Lite student
# 4 GPUs (but can run on fewer since Llama-1B is small)
# ==========================================================================
train_lite)
  WDATA=${2:?pass weighted dataset dir}
  python3 -u train.py model=KD_tisdpo \
    model.policy_name_or_path=$STUDENT \
    model.reference_name_or_path=$TEACHER \
    model.teacher_tokenizer_name_or_path=$TEACHER \
    model.student_tokenizer_name_or_path=$STUDENT \
    model.teacher_name_or_path=$TEACHER \
    model.student_name_or_path=$STUDENT \
    model.original_policy_name=$STUDENT \
    model.policy_block_name=LlamaDecoderLayer \
    model.reference_block_name=MistralDecoderLayer \
    loss=KD_tisdpo log_dir=pairB_CTPD_lite \
    policy_mode=student reference_mode=teacher \
    loss.beta=0.1 loss.label_smoothing=0 loss.average_log_prob=false \
    n_epochs=1 max_grad_norm=1.0 \
    gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
    total_steps=7082 warmup_steps=708 eval_every=566 \
    lr=1e-6 scheduler=cosine transform.method=origin \
    trainer=FSDPTrainer sample_during_eval=false \
    datasets=$WDATA ;;

*)
  echo "usage: $0 {weight_full POS NEG|train_ctpd WDATA|train_lite WDATA}" ;;
esac
