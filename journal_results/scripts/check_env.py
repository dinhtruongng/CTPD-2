#!/usr/bin/env python3
"""Check whether a given conda env can run lm-eval."""
import subprocess, sys

env = sys.argv[1] if len(sys.argv) > 1 else "train"
cmd = f"""
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate {env}
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import huggingface_hub; print('hub:', huggingface_hub.__version__)"
python -c "import lm_eval; print('lm_eval:', lm_eval.__version__)"
"""
subprocess.run(["bash", "-c", cmd])
