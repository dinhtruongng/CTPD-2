#!/bin/bash
set -e

export OPENAI_API_KEY=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


cd fastchat/llm_judge
echo "Current directory: $(pwd)"

python gen_judgment.py \
       --mode pairwise-baseline \
       --judge-model gpt-4 \
       --baseline-model llama8b_dpo_v2 \
       --model-list  ctpd \
       --parallel 2        # concurrent API calls
