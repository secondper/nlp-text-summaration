import torch
try:
    import bert4torch
    bert4torch_installed = True
except ImportError:
    bert4torch_installed = False

print("-" * 30)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")
else:
    print("no GPU detected, use CPU to train(slow)")

print("-" * 30)

if bert4torch_installed:
    print("bert4torch:      Installed Successfully")
else:
    print("bert4torch:      Not Found")
print("-" * 30)