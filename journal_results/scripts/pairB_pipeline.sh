#!/bin/bash
# Pair B = Mistral-7B-Instruct-v0.3 (teacher) -> Llama-3.2-1B-Instruct (student)
# All models = public instruct checkpoints (already SFT'd) — no extra SFT step.
#
# LAUNCH ORDER (serial within each column, parallel across columns):
#   1. tea_pos         2. tea_neg         3. dpo_baseline  (parallel)
#   4. weight_full     5. weight_lite                       (parallel, AFTER tea_pos & tea_neg)
#   6. train_ctpd      7. train_lite                        (parallel, AFTER weights)
#   8. eval
#
# Run detached: `nohup bash pairB_pipeline.sh STAGE > logs/pairB_STAGE.log 2>&1 &`
set -e

ROOT=~/CTPD-2
JR=$ROOT/journal_results
TEACHER=$JR/models/mistralai_Mistral-7B-Instruct-v0.3
STUDENT=$JR/models/meta-llama_Llama-3.2-1B-Instruct
DATA=$JR/data/pairB
# Do NOT export CUDA_VISIBLE_DEVICES here — the caller sets one GPU per job
# so FSDP spawns 1 process, not N processes fighting for the same NCCL port.

cd $ROOT/src/prefkd
mkdir -p $JR/weights $JR/logs

STAGE=${1:-help}

case "$STAGE" in

# ---- 1. positive teacher: Mistral + DPO ------------------------------------
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
    gradient_accumulation_steps=2 batch_size=8 eval_batch_size=8 \
    total_steps=3541 warmup_steps=354 eval_every=566 \
    lr=5e-7 scheduler=cosine save_checkpoint=true \
    trainer=FSDPTrainer sample_during_eval=false \
    datasets=$DATA ;;

# ---- 2. negative teacher: Mistral + reverse DPO ----------------------------
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
    gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
    total_steps=14164 warmup_steps=708 eval_every=1112 \
    lr=1e-5 scheduler=cosine save_checkpoint=true \
    activation_checkpointing=true \
    trainer=FSDPTrainer sample_during_eval=false \
    reverse_dataset=true \
    datasets=$DATA ;;

# ---- 3. full-CTPD weight generation (pos vs neg teacher) --------------------
weight_full)
  POS=${2:?pass positive ckpt dir}
  NEG=${3:?pass negative ckpt dir}
  python3 weight.py \
    --positive_model_name $POS --negative_model_name $NEG \
    --student_model $STUDENT \
    --data_path $DATA/train --output_dir $JR/weights/pairB_full \
    --split train --batch_size 8 --num_gpus 4 \
    --average_log_prob 0 --teacher_all 0 --student_all 1 ;;

# ---- 4. CTPD-Lite weight generation (single teacher, no DPO needed) ---------
weight_lite)
  python3 $JR/scripts/weight_lite.py \
    --teacher_model $TEACHER --student_model $STUDENT \
    --data_path $DATA/train --output_dir $JR/weights/pairB_lite \
    --split train --batch_size 8 --num_gpus 4 \
    --average_log_prob 0 --student_all 1 --mu 1.0 ;;

# ---- 5a. train student: full CTPD ------------------------------------------
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

# ---- 5b. train student: CTPD-Lite ------------------------------------------
train_lite)
  WDATA=${2:?pass lite weighted dataset dir}
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

# ---- DPO baseline (student only) -------------------------------------------
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
    gradient_accumulation_steps=2 batch_size=16 eval_batch_size=16 \
    total_steps=3541 warmup_steps=177 eval_every=566 \
    lr=1e-6 scheduler=cosine save_checkpoint=true \
    trainer=FSDPTrainer sample_during_eval=false \
    datasets=$DATA ;;

*)
  echo "usage: $0 {tea_pos|tea_neg|dpo_baseline|weight_full POS NEG|weight_lite|train_ctpd WDATA|train_lite WDATA}" ;;
esac
