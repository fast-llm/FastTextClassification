import torch
print(torch.version)
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func