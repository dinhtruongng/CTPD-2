#!/bin/bash
set -e



cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=sft \
  model.policy_name_or_path= \
  model.reference_name_or_path= \
  model.teacher_tokenizer_name_or_path= \
  model.student_tokenizer_name_or_path= \
  model.original_policy_name= \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=sft \
  log_dir=tea_sft_v1negative \
  policy_mode=teacher \
  n_epochs=1 \
  max_grad_norm=10.0 \
  gradient_accumulation_steps=2 batch_size=16 eval_batch_size=16 \
  total_steps=3541 warmup_steps=354 eval_every=566 \
  lr=3e-6 scheduler=cosine \
  save_checkpoint=false\
  trainer=FSDPTrainer sample_during_eval=false \
  reverse_dataset=true \

