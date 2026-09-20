import os
import numpy as np
import pandas as pd
from .metrics import compute_metrics, bootstrap_ci

def compute_from_cache(cache, config):
    """
    Tier-2 Evaluation: Computes the final score based on a given ablation config
    without re-running the neural network.
    
    config can contain:
    - dino_weights: list of weights for DINO layers
    - cnn_weights: list of weights for CNN layers
    - worst_k_ratio: patch cosine worst-k
    - view_weights: (global, center, texture)
    - gating: (w_d, w_g, w_p, use_gate)
    - ...
    """
    scores = []
    
    # Example logic demonstrating how post-hoc reweighting works
    for item in cache:
        v_scores = {}
        for view in ['global', 'center', 'texture']:
            view_cache = item['cache'][view]
            
            # Recompute DINO
            dino_layer_scores = []
            for i, ldino in enumerate(view_cache['dino']):
                k_val = max(1, int(len(ldino['cos_flat']) * config.get('worst_k_ratio', 0.10)))
                worst_cos = np.mean(np.sort(ldino['cos_flat'])[:k_val])
                patch_cos = 0.5 * ldino['mean_cos'] + 0.5 * worst_cos
                ls = 0.45 * ldino['dists_score'] + 0.45 * patch_cos + 0.10 * ldino['cls_score']
                dino_layer_scores.append(ls * config.get('dino_weights', [0.25, 0.30, 0.25, 0.12, 0.08])[i])
            s_dino = np.sum(dino_layer_scores) / np.sum(config.get('dino_weights', [0.25, 0.30, 0.25, 0.12, 0.08]))
            
            # Recompute CNN
            dists_scores, gram_scores, nus = [], [], []
            for lcnn in view_cache['cnn']:
                dists_scores.append(lcnn['dists_score'])
                gram_scores.append(lcnn['gram_score'])
                nus.append(lcnn['std_diff'] / lcnn['mean_diff'])
            s_dists = np.mean(dists_scores)
            s_gram = np.mean(gram_scores)
            nu = np.mean(nus)
            
            # Gating
            if config.get('use_gate', True):
                gate_dino = 1 / (1 + np.exp(-4.0 * (nu - 0.70)))
                w_d = 0.35 + 0.30 * gate_dino
                w_g = 0.50 - 0.30 * gate_dino
                w_p = 0.15
            else:
                w_d, w_g, w_p = config.get('fixed_weights', (1/3, 1/3, 1/3))
                
            v_scores[view] = w_d * s_dino + w_g * s_gram + w_p * s_dists
            
        final_score = (
            config.get('view_weights', (0.60, 0.25, 0.15))[0] * v_scores['global'] +
            config.get('view_weights', (0.60, 0.25, 0.15))[1] * v_scores['center'] +
            config.get('view_weights', (0.60, 0.25, 0.15))[2] * v_scores['texture']
        )
        scores.append((item['mos'], final_score, item['dist_type']))
        
    return scores

def run_tier2_evaluations(cache_dir="raw/", out_csv=None, num_samples=None, seed=None):
    results = []
    configs = [
        # Group A: Components
        {"id": "A1_full", "use_gate": True},
        {"id": "A2_no_gate", "use_gate": False, "fixed_weights": (1/3, 1/3, 1/3)},
        {"id": "A3_dino_only", "use_gate": False, "fixed_weights": (1.0, 0.0, 0.0)},
        {"id": "A4_gram_only", "use_gate": False, "fixed_weights": (0.0, 1.0, 0.0)},
        {"id": "A5_dists_only", "use_gate": False, "fixed_weights": (0.0, 0.0, 1.0)},
        {"id": "A6_dino_gram", "use_gate": False, "fixed_weights": (0.5, 0.5, 0.0)},
        {"id": "A7_dino_dists", "use_gate": False, "fixed_weights": (0.5, 0.0, 0.5)},
        
        # Group B: DINO Worst-K Patch Selection
        {"id": "B1_worst_01", "use_gate": True, "worst_k_ratio": 0.01},
        {"id": "B1_worst_05", "use_gate": True, "worst_k_ratio": 0.05},
        {"id": "B1_worst_10_base", "use_gate": True, "worst_k_ratio": 0.10},
        {"id": "B1_worst_25", "use_gate": True, "worst_k_ratio": 0.25},
        {"id": "B1_worst_100", "use_gate": True, "worst_k_ratio": 1.00},
        
        # Group C: DINO Layers
        {"id": "C1_last_layer", "use_gate": True, "dino_weights": [0, 0, 0, 0, 1]},
        {"id": "C2_equal", "use_gate": True, "dino_weights": [0.2, 0.2, 0.2, 0.2, 0.2]},
        {"id": "C3_early_heavy", "use_gate": True, "dino_weights": [0.4, 0.3, 0.2, 0.1, 0.0]},
        {"id": "C4_baseline", "use_gate": True, "dino_weights": [0.25, 0.30, 0.25, 0.12, 0.08]},
        
        # Group E: Multi-Scale Views
        {"id": "E1_global_only", "use_gate": True, "view_weights": (1.0, 0.0, 0.0)},
        {"id": "E2_center_only", "use_gate": True, "view_weights": (0.0, 1.0, 0.0)},
        {"id": "E3_texture_only", "use_gate": True, "view_weights": (0.0, 0.0, 1.0)},
        {"id": "E4_global_center", "use_gate": True, "view_weights": (0.5, 0.5, 0.0)},
        {"id": "E5_baseline", "use_gate": True, "view_weights": (0.60, 0.25, 0.15)},
    ]
    
    for ds_file in os.listdir(cache_dir):
        if not ds_file.endswith("_cache.npy"): continue
        dataset_name = ds_file.replace("_cache.npy", "")
        cache = np.load(os.path.join(cache_dir, ds_file), allow_pickle=True)
        
        for cfg in configs:
            scores = compute_from_cache(cache, cfg)
            y_true = np.array([s[0] for s in scores])
            y_pred = np.array([s[1] for s in scores])
            
            metrics = compute_metrics(y_pred, y_true)
            results.append({
                "study_id": cfg["id"],
                "dataset": dataset_name,
                **metrics
            })
            
    df = pd.DataFrame(results)
    
    if out_csv is not None:
        out_file = out_csv
    else:
        if num_samples is not None:
            datasets_processed = list(df['dataset'].unique())
            prefix = datasets_processed[0] if len(datasets_processed) == 1 else "combined"
            out_file = f"{prefix}_{num_samples}_seed_{seed}.csv"
        else:
            out_file = "master_results.csv"
            
    df.to_csv(out_file, index=False)
    print(f"Tier-2 evaluations complete. Saved to {out_file}")
    print(df.to_string())

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # run_tier2_evaluations()
    pass
