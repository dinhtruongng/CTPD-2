#!/bin/bash
# Self-contained Phase 1b launcher — runs directly on server.
# Launches dpo_baseline + weight_lite on whichever GPUs are free.
set -euo pipefail

ROOT=~/CTPD-2
JR=$ROOT/journal_results
TEACHER=$JR/models/mistralai_Mistral-7B-Instruct-v0.3
STUDENT=$JR/models/meta-llama_Llama-3.2-1B-Instruct
DATA=$JR/data/pairB
mkdir -p $JR/weights $JR/logs

cd $ROOT/src/prefkd

echo "=== Phase 1b launching ==="
echo "dpo_baseline using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"
echo "weight_lite using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"

STAGE=${1:-both}

if [ "$STAGE" = "dpo_baseline" ] || [ "$STAGE" = "both" ]; then
    echo "Launching DPO baseline..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -u train.py model=dpo \
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
        datasets=$DATA &
    PID_BASELINE=$!
    echo "dpo_baseline PID=$PID_BASELINE"
fi

if [ "$STAGE" = "weight_lite" ] || [ "$STAGE" = "both" ]; then
    echo "Launching weight_lite..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -u $JR/scripts/weight_lite.py \
        --teacher_model $TEACHER --student_model $STUDENT \
        --data_path $DATA --split train \
        --output_dir $JR/weights/pairB_lite \
        --batch_size 8 --num_gpus 1 \
        --average_log_prob 0 --student_all 1 --mu 1.0 &
    PID_LITE=$!
    echo "weight_lite PID=$PID_LITE"
fi

wait
echo "Phase 1b complete."
