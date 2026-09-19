import os
import json
import torch
import pyiqa
import numpy as np
from tqdm import tqdm
from pyiqa.default_model_configs import DEFAULT_CONFIGS

# Import to register the architecture
import foundation_hybrid.arch

def run_tier1_cache(datasets, output_dir="raw/"):
    """
    Tier-1 Caching Script.
    Runs the model with `return_cache=True` and saves raw outputs (DISTS/Gram/Cosine scores)
    for every image pair so we can perform fast post-hoc ablation studies.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inject plugin to PyIQA
    DEFAULT_CONFIGS['foundation_hybrid_cache'] = {
        'metric_opts': {
            'type': 'FoundationHybrid',
            'return_cache': True,
        },
        'metric_mode': 'FR',
        'lower_better': False,
    }
    
    model = pyiqa.create_metric('foundation_hybrid_cache', device=device)
    
    for dataset_name in datasets:
        print(f"Caching dataset: {dataset_name}")
        dataset = pyiqa.load_dataset(dataset_name)
        
        cache_data = []
        
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            dist_img = data['img'].unsqueeze(0).to(device)
            if 'ref_img' in data:
                ref_img = data['ref_img'].unsqueeze(0).to(device)
            else:
                # Fallback if some dataset uses 'ref'
                ref_img = data['ref'].unsqueeze(0).to(device)
                
            mos = data.get('mos_label', data.get('mos', data.get('dmos', 0)))
            if isinstance(mos, torch.Tensor):
                mos = mos.item()
                
            # Some datasets have distortion types
            dist_type = data.get('distortion_type', 'unknown')
            
            with torch.no_grad():
                cache = model(ref_img, dist_img)
                
            # Move cache to CPU and convert to numpy for serialization
            cpu_cache = {}
            for view, view_cache in cache.items():
                cpu_cache[view] = {"dino": [], "cnn": []}
                for l in view_cache["dino"]:
                    cpu_cache[view]["dino"].append({k: v.cpu().numpy() for k, v in l.items()})
                for l in view_cache["cnn"]:
                    cpu_cache[view]["cnn"].append({k: v.cpu().numpy() for k, v in l.items()})
                    
            cache_data.append({
                "idx": i,
                "mos": float(mos),
                "dist_type": dist_type,
                "cache": cpu_cache
            })
            
        np.save(os.path.join(output_dir, f"{dataset_name}_cache.npy"), cache_data, allow_pickle=True)
        print(f"Saved {dataset_name}_cache.npy")

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # run_tier1_cache(["csiq", "tid2013", "live", "kadid10k", "pipal"])
    pass
