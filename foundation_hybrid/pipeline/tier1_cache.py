import os
import json
import torch
import pyiqa
import numpy as np
from tqdm import tqdm
from pyiqa.default_model_configs import DEFAULT_CONFIGS

# Import to register the architectures
import foundation_hybrid.arch
import foundation_hybrid.arch_dino

# arch name -> registered pyiqa arch type
ARCHS = {
    'hybrid': 'FoundationHybrid',
    'dino': 'FoundationDino',
}

def run_tier1_cache(datasets, output_dir="raw/", data_root=None, num_samples=None, seed=42, quiet=False,
                    arch="hybrid"):
    """
    Tier-1 Caching Script.
    Runs the model with `return_cache=True` and saves raw outputs (DISTS/Gram/Cosine scores)
    for every image pair so we can perform fast post-hoc ablation studies.

    arch: 'hybrid' (DINO + CNN experts, cache has per-view {'dino', 'cnn'}) or
          'dino' (DINO expert only, cache has per-view {'dino'}).

    data_root: where pyiqa looks for (and, if missing, downloads) the datasets.
    Falls back to $IQA_DATA_ROOT, then pyiqa's own default "./datasets" (relative to
    the current directory). Point it at a pre-populated, read-only location to avoid
    re-downloading the datasets on every run.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_root = data_root or os.environ.get("IQA_DATA_ROOT") or "./datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inject plugin to PyIQA
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {sorted(ARCHS)}")
    metric_name = f'foundation_{arch}_cache'
    DEFAULT_CONFIGS[metric_name] = {
        'metric_opts': {
            'type': ARCHS[arch],
            'return_cache': True,
        },
        'metric_mode': 'FR',
        'lower_better': False,
    }
    
    model = pyiqa.create_metric(metric_name, device=device)
    
    for dataset_name in datasets:
        print(f"Caching dataset: {dataset_name}")
        dataset = pyiqa.load_dataset(dataset_name, data_root=data_root)
        
        if num_samples is not None and num_samples < len(dataset):
            print(f"Subsampling {num_samples} images from {dataset_name} (seed={seed})")
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(dataset), num_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        # LIVE has variable image sizes, so it needs batch_size=1 unless custom collate is used.
        bs = 1 if dataset_name.lower() == 'live' else 8
        
        # DataLoader for fast parallel loading and batching
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True
        )
        
        cache_data = []
        idx_counter = 0
        
        for data in tqdm(dataloader, disable=quiet):
            dist_img = data['img'].to(device)
            if 'ref_img' in data:
                ref_img = data['ref_img'].to(device)
            else:
                ref_img = data['ref'].to(device)
                
            mos_batch = data.get('mos_label', data.get('mos', data.get('dmos', torch.zeros(dist_img.shape[0]))))
            dist_type_batch = data.get('distortion_type', ['unknown'] * dist_img.shape[0])
            
            with torch.no_grad():
                cache = model(ref_img, dist_img)
                
            B = dist_img.shape[0]
            
            # Split the batch back into individual items
            for b in range(B):
                cpu_cache = {}
                for view, view_cache in cache.items():
                    # one list of per-layer dicts per expert ('dino' and, for hybrid, 'cnn')
                    cpu_cache[view] = {
                        expert: [{k: v[b].cpu().numpy() for k, v in l.items()} for l in layers]
                        for expert, layers in view_cache.items()
                    }
                        
                mos_val = mos_batch[b].item() if isinstance(mos_batch, torch.Tensor) else mos_batch[b]
                dtype_val = dist_type_batch[b] if isinstance(dist_type_batch, list) or isinstance(dist_type_batch, tuple) else dist_type_batch
                
                cache_data.append({
                    "idx": idx_counter,
                    "mos": float(mos_val),
                    "dist_type": dtype_val,
                    "cache": cpu_cache
                })
                idx_counter += 1
                
        np.save(os.path.join(output_dir, f"{dataset_name}_cache.npy"), cache_data, allow_pickle=True)
        print(f"Saved {dataset_name}_cache.npy")

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # run_tier1_cache(["csiq", "tid2013", "live", "kadid10k", "pipal"])
    pass
