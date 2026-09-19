import time
import torch
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS

def profile_efficiency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DEFAULT_CONFIGS['foundation_hybrid'] = {
        'metric_opts': {'type': 'FoundationHybrid'},
        'metric_mode': 'FR',
        'lower_better': False,
    }
    
    model = pyiqa.create_metric('foundation_hybrid', device=device)
    
    # Profile parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params / 1e6:.2f} M")
    
    # Profile Latency
    ref = torch.rand(1, 3, 512, 512).to(device)
    dist = torch.rand(1, 3, 512, 512).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(ref, dist)
            
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    
    iters = 20
    for _ in range(iters):
        with torch.no_grad():
            model(ref, dist)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    
    latency = (end - start) / iters * 1000 # ms
    print(f"Median Latency per Image Pair (512x512): {latency:.2f} ms")
    
    # Peak Memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory: {peak_mem:.2f} MB")

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # profile_efficiency()
    pass
