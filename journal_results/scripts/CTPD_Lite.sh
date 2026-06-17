#!/bin/bash
# CTPD-Lite training launch: identical to CTPD.sh EXCEPT the weighted dataset is
# produced by weight_lite.py (single SFT teacher, no positive/negative DPO teachers).
# The training stage itself is unchanged -- caching makes the weight source
# invisible to the optimizer. Fill in the paths before running.
set -e

cd "$(dirname "$0")/../.."
cd src/prefkd
echo "Current directory: $(pwd)"

# --- paths to fill in -------------------------------------------------------
STUDENT_SFT="models/Llama-3.1-8B/sft/v6/Llama8b"   # student SFT checkpoint
TEACHER="models/Qwen-2.5-14B"                       # single SFT teacher (also reference)
LITE_DATA="journal_results/lite_weights/full"        # output of weight_lite.py
# ---------------------------------------------------------------------------

python3 -u train.py \
  model=KD_tisdpo \
  model.policy_name_or_path=$STUDENT_SFT \
  model.reference_name_or_path=$TEACHER \
  model.teacher_tokenizer_name_or_path=$TEACHER \
  model.student_tokenizer_name_or_path=$STUDENT_SFT \
  model.teacher_name_or_path=$TEACHER \
  model.student_name_or_path=$STUDENT_SFT \
  model.original_policy_name=models/Llama-3.1-8B \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=KD_tisdpo \
  log_dir=CTPD_Lite_V1 \
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
  datasets=$LITE_DATA
