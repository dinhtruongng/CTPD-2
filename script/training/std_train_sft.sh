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
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=LlamaDecoderLayer \
  loss=sft \
  log_dir=llama8B_sft \
  policy_mode=student \
  n_epochs=1 \
  max_grad_norm=1.0 \
  gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 \
  total_steps=442 warmup_steps=13 eval_every=5632 \
  trainer=FSDPTrainer sample_during_eval=false \
