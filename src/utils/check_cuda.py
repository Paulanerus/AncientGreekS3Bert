import torch


def check_cuda_and_gpus():
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**2)
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} MiB\n")
    else:
        print("No CUDA-enabled GPU found.\n")

    return "cuda" if cuda_available else "cpu"
