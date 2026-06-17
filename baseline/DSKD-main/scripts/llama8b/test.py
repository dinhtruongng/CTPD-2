import torch, platform, os
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
