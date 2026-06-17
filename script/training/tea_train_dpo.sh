#!/bin/bash
set -e


cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=dpo \
  model.policy_name_or_path= \
  model.reference_name_or_path= \
  model.teacher_tokenizer_name_or_path= \
  model.student_tokenizer_name_or_path= \
  model.original_policy_name= \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=dpo \
  log_dir=tea_V1positive \
  policy_mode=teacher \
  reference_mode=teacher \
  loss.beta=0.5 \
  loss.label_smoothing=0 \
  n_epochs=1 \
  max_grad_norm=3.0 \
  gradient_accumulation_steps=1 batch_size=16 eval_batch_size=16 \
  total_steps=3541 warmup_steps=354 eval_every=566 \
  lr=5e-7 scheduler=cosine \
  save_checkpoint=false \
  trainer=FSDPTrainer sample_during_eval=false \





