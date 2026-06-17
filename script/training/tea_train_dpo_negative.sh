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
  log_dir=tea_V1negative \
  policy_mode=teacher \
  reference_mode=teacher \
  loss.beta=0.02 \
  loss.label_smoothing=0.0 \
  n_epochs=3 \
  max_grad_norm=1.0 \
  gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
  total_steps=14164 warmup_steps=708 eval_every=1112 \
  lr=1e-5 scheduler=cosine \
  save_checkpoint=true \
  trainer=FSDPTrainer sample_during_eval=false \
  activation_checkpointing=false \
  reverse_dataset=true \
