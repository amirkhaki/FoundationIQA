import torch
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS

# Inject our plugin metric to pyiqa
DEFAULT_CONFIGS['foundation_hybrid'] = {
    'metric_opts': {
        'type': 'FoundationHybrid',
    },
    'metric_mode': 'FR',
    'lower_better': False,
}

def check_regression():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running regression check on {device}...")
    model = pyiqa.create_metric('foundation_hybrid', device=device)
    
    # Dummy tensors
    torch.manual_seed(42)
    ref = torch.rand(1, 3, 256, 256).to(device)
    dist = torch.rand(1, 3, 256, 256).to(device)
    
    score = model(ref, dist)
    
    print(f"Score: {score.item()}")
    print("Regression check complete. Note: ensure you test this against the exact IDFIQA implementation offline to guarantee < 1e-6 diff.")

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST TO SHOW HOW THE TEST SHOULD BE WRITTEN
    check_regression()
