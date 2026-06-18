#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate eval
python -c "from lm_eval import simple_evaluate; print('lm_eval OK')"
echo "Eval env functional."
