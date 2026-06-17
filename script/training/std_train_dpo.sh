#!/bin/bash
set -e



cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=dpo \
  model.policy_name_or_path=\
  model.reference_name_or_path=\
  model.teacher_tokenizer_name_or_path=\
  model.student_tokenizer_name_or_path=\
  model.original_policy_name= \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=LlamaDecoderLayer \
  loss=dpo \
  log_dir=std_dpo_V1positive \
  policy_mode=student \
  reference_mode=student \
  loss.beta=0.1 \
  loss.label_smoothing=0 \
  n_epochs=1 \
  max_grad_norm=1.0 \
  gradient_accumulation_steps=2 batch_size=16 eval_batch_size=16 \
  total_steps=3541 warmup_steps=177 eval_every=566 \
  lr=1e-6 scheduler=cosine \
  trainer=FSDPTrainer sample_during_eval=false \
  save_checkpoint=false \

