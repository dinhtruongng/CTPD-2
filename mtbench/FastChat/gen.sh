#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd fastchat/llm_judge
echo "Current directory: $(pwd)"


# Your model
# python gen_model_answer.py \
#        --model-path /mnt/nfs-nlp/models/Llama-3.1-8B/KD_tisdpo/v19/Llama8b \
#        --model-id   ctpd \
#        --num-gpus-total 8

# Baseline model
python gen_model_answer.py \
       --model-path /mnt/nfs-nlp/models/Llama-3.1-8B/dpo/V1negative/Llama8b \
       --model-id   llama8b_dpo_v2 \
       --num-gpus-total 8

