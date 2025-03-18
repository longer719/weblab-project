import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    # 简单测试
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    print("执行GPU测试...")
    z = torch.matmul(x, y)
    print("GPU测试成功!")