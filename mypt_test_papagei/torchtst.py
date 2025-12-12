import torch

print("===== PyTorch & CUDA Test =====")

# Test PyTorch version
print("PyTorch version:", torch.__version__)

# Test CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    # Try running a tensor on GPU
    try:
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x + y
        print("GPU tensor computation success! Result:")
        print(z)
    except Exception as e:
        print("Error running tensor on GPU:", e)
else:
    print("CUDA is NOT available. Running a CPU test...")

# Test CPU tensor creation
try:
    x_cpu = torch.rand(3, 3)
    print("CPU tensor computation success!")
    print(x_cpu)
except Exception as e:
    print("CPU tensor error:", e)

print("===== Test Completed =====")
