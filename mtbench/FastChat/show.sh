#!/bin/bash
set -e

cd fastchat/llm_judge
echo "Current directory: $(pwd)"

python show_result.py \
       --mode pairwise-baseline \
       --judge-model gpt-4 \
       --baseline-model llama8b_dpo_v2
