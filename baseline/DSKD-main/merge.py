from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/ngannt61/PrefKD/baseline/DSKD-main/model_hub/llama/llama1B")
base = AutoModelForCausalLM.from_pretrained("/home/ngannt61/PrefKD/baseline/DSKD-main/model_hub/llama/llama1B")

merged = PeftModel.from_pretrained(base, "/home/ngannt61/PrefKD/baseline/DSKD-main/outputs/llama/llama1B/dual_space_kd_with_cma/criterion=dual_space_kd_with_cma__forward_kl-fp16__teacher=Qwen2.5-7B__kd^rate=0.5__kd^temp=2.0__epoch=1__bsz=4x2x4=32__lr=0.0005__proj^lr=0.001/epoch1_step1771_loss1.3571")
merged = merged.merge_and_unload()
merged.save_pretrained("llama3-1b-dskd")
tokenizer.save_pretrained("llama3-1b-dskd")  # same dir you just saved the model
