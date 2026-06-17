#!/bin/bash
set -e


cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

python3 -u train.py \
  model=KD_tisdpo \
  model.policy_name_or_path=models/Llama-3.1-8B/sft/v6/Llama8b \
  model.reference_name_or_path=models/Qwen-2.5-14B \
  model.teacher_tokenizer_name_or_path=models/Qwen-2.5-14B \
  model.student_tokenizer_name_or_path=models/Llama-3.1-8B/sft/v6/Llama8b \
  model.teacher_name_or_path=models/Qwen-2.5-14B \
  model.student_name_or_path=models/Llama-3.1-8B/sft/v6/Llama8b \
  model.original_policy_name=models/Llama-3.1-8B \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=KD_tisdpo \
  log_dir=KDPO_stdAllV1 \
  policy_mode=student \
  reference_mode=teacher \
  loss.beta=0.1 \
  loss.label_smoothing=0 \
  loss.average_log_prob=false \
  n_epochs=1 \
  max_grad_norm=1.0 \
  gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
  total_steps=7082 warmup_steps=708 eval_every=566 \
  lr=1e-6 scheduler=cosine \
  transform.method=origin \
  trainer=FSDPTrainer sample_during_eval=false \
  datasets=data/ultra-feedback/weight14B/student_all/full \
