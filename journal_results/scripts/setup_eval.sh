#!/bin/bash
# Fix eval env: pin transformers 4.52.4 compatible with lm_eval 0.4.8
# Uses --model hf (no vLLM). Installs WITHOUT the vllm extra to avoid
# transformers>=5.0 dependency conflict.
set -euo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Fresh eval env
conda remove -n eval --all -y 2>/dev/null || true
conda create -n eval python=3.11.11 -y
conda activate eval

pip install transformers==4.52.4 tokenizers==0.21.4 \
  huggingface-hub==0.36.2 accelerate==1.3.0 \
  datasets==3.5.0 torch==2.5.1

pip install 'lm-eval==0.4.8' sentencepiece==0.2.0 langdetect==1.0.9

echo "=== Verify ==="
python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
echo "Eval env ready."
