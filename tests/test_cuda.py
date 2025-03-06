import torch
print(f"torch.version:{torch.version}")
print(f"torch.__version__:{torch.__version__}")
if torch.cuda.is_available():
    print(f"torch.cuda.is_available():{torch.cuda.is_available()}")
    print(f"torch.cuda.device_count():{torch.cuda.device_count()}")
    print(f"torch.version.cuda:{torch.version.cuda}")
    print(f"torch.backends.cudnn.version():{torch.backends.cudnn.version()}")
    

# 添加详细信息
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"CUDA device {i}: {device_name}, Capability: {device_capability}")