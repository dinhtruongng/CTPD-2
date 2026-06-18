#!/bin/bash
# Pair B = Mistral-7B-Instruct-v0.3 (teacher) -> Llama-3.2-1B-Instruct (student)
# Mistral-7B DPO won't fit on 1 H200 → tea_pos + tea_neg each use 2 GPUs.
#
# LAUNCH SCHEDULE:
#   Phase 1a: tea_pos (GPU 4,5) + tea_neg (GPU 6,7) — parallel, ~4h
#   Phase 1b: dpo_baseline (GPU 4) + weight_lite (GPU 7) — parallel, ~2h
#   Phase 2:  weight_full (GPU 4,5,6,7) — after 1a
#   Phase 3:  train_ctpd (GPU 4-7) + train_lite (simultaneous on CPU)
#   Phase 4:  eval
#
# Run:  CUDA_VISIBLE_DEVICES=X,Y bash pairB_pipeline.sh STAGE
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
# Phase 1a — teacher DPO (Mistral-7B, 2 GPUs each)
# ==========================================================================

tea_pos)
  python3 -u train.py model=dpo \
    model.policy_name_or_path=$TEACHER \
    model.reference_name_or_path=$TEACHER \
    model.teacher_tokenizer_name_or_path=$TEACHER \
    model.student_tokenizer_name_or_path=$TEACHER \
    model.original_policy_name=$TEACHER \
    model.policy_block_name=MistralDecoderLayer \
    model.reference_block_name=MistralDecoderLayer \
    loss=dpo log_dir=pairB_tea_positive \
    policy_mode=teacher reference_mode=teacher \
    loss.beta=0.5 loss.label_smoothing=0 n_epochs=1 max_grad_norm=3.0 \
    gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
    total_steps=3541 warmup_steps=354 eval_every=566 \
    lr=5e-7 scheduler=cosine save_checkpoint=true \
    trainer=FSDPTrainer sample_during_eval=false \
    fsdp_port=12355 \
    datasets=$DATA ;;

tea_neg)
  python3 -u train.py model=dpo \
    model.policy_name_or_path=$TEACHER \
    model.reference_name_or_path=$TEACHER \
    model.teacher_tokenizer_name_or_path=$TEACHER \
    model.student_tokenizer_name_or_path=$TEACHER \
    model.original_policy_name=$TEACHER \
    model.policy_block_name=MistralDecoderLayer \
    model.reference_block_name=MistralDecoderLayer \
    loss=dpo log_dir=pairB_tea_negative \
    policy_mode=teacher reference_mode=teacher \
    loss.beta=0.02 loss.label_smoothing=0.0 n_epochs=3 max_grad_norm=1.0 \
    gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
    total_steps=14164 warmup_steps=708 eval_every=1112 \
    lr=1e-5 scheduler=cosine save_checkpoint=true \
    trainer=FSDPTrainer sample_during_eval=false \
    fsdp_port=12356 \
    reverse_dataset=true \
    datasets=$DATA ;;

# ==========================================================================
# Phase 1b — baselines + lite weights (Llama-1B = 1 GPU, inference = 1 GPU)
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

weight_lite)
  python3 -u $JR/scripts/weight_lite.py \
    --teacher_model $TEACHER --student_model $STUDENT \
    --data_path $DATA --split train \
    --output_dir $JR/weights/pairB_lite \
    --batch_size 8 --num_gpus 1 \
    --average_log_prob 0 --student_all 1 --mu 1.0 ;;

# ==========================================================================
# Phase 2 — span-weight generation (after teacher DPO finishes)
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
# Phase 3 — student training
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
  echo "usage: $0 {tea_pos|tea_neg|dpo_baseline|weight_lite|weight_full POS NEG|train_ctpd WDATA|train_lite WDATA}" ;;
esac
