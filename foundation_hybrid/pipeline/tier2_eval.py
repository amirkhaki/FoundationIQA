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
                steepness = config.get('gate_steepness', 4.0)
                threshold = config.get('gate_threshold', 0.70)
                gate_dino = 1 / (1 + np.exp(-steepness * (nu - threshold)))
                gate_w_d_range = config.get('gate_w_d_range', (0.35, 0.65))
                gate_w_g_range = config.get('gate_w_g_range', (0.50, 0.20))
                w_p = config.get('gate_w_p', 0.15)
                
                w_d = gate_w_d_range[0] + (gate_w_d_range[1] - gate_w_d_range[0]) * gate_dino
                w_g = gate_w_g_range[0] + (gate_w_g_range[1] - gate_w_g_range[0]) * gate_dino
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
        {"id": "B1_worst_002", "use_gate": True, "worst_k_ratio": 0.002},
        {"id": "B1_worst_005", "use_gate": True, "worst_k_ratio": 0.005},
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
        {"id": "C5_blend_1", "use_gate": True, "dino_weights": [0.3, 0.3, 0.2, 0.1, 0.1]},
        {"id": "C6_first_two", "use_gate": True, "dino_weights": [0.5, 0.5, 0.0, 0.0, 0.0]},
        
        # Group D: Gate Hyperparameters
        {"id": "D1_gate_thresh_04", "use_gate": True, "gate_threshold": 0.40},
        {"id": "D2_gate_thresh_05", "use_gate": True, "gate_threshold": 0.50},
        {"id": "D3_gate_thresh_06", "use_gate": True, "gate_threshold": 0.60},
        {"id": "D4_gate_steep_2", "use_gate": True, "gate_steepness": 2.0},
        {"id": "D5_gate_steep_8", "use_gate": True, "gate_steepness": 8.0},
        
        # Group F: DISTS Boost Gating
        {"id": "F1_dists_boost_20", "use_gate": True, "gate_w_p": 0.20, "gate_w_d_range": (0.3294, 0.6118), "gate_w_g_range": (0.4706, 0.1882)},
        {"id": "F2_dists_boost_30", "use_gate": True, "gate_w_p": 0.30, "gate_w_d_range": (0.2882, 0.5353), "gate_w_g_range": (0.4118, 0.1647)},
        {"id": "F3_dists_boost_40", "use_gate": True, "gate_w_p": 0.40, "gate_w_d_range": (0.2471, 0.4588), "gate_w_g_range": (0.3529, 0.1412)},
        
        # Group E: Multi-Scale Views
        {"id": "E1_global_only", "use_gate": True, "view_weights": (1.0, 0.0, 0.0)},
        {"id": "E2_center_only", "use_gate": True, "view_weights": (0.0, 1.0, 0.0)},
        {"id": "E3_texture_only", "use_gate": True, "view_weights": (0.0, 0.0, 1.0)},
        {"id": "E4_global_center", "use_gate": True, "view_weights": (0.5, 0.5, 0.0)},
        {"id": "E5_baseline", "use_gate": True, "view_weights": (0.60, 0.25, 0.15)},

        # Group J: Joint Optimizations
        {"id": "J1_joint_best", "use_gate": True, "worst_k_ratio": 0.01, "view_weights": (0.5, 0.5, 0.0), "dino_weights": [0.4, 0.3, 0.2, 0.1, 0.0]},
        {"id": "J2_joint_equal_dino", "use_gate": True, "worst_k_ratio": 0.01, "view_weights": (0.5, 0.5, 0.0), "dino_weights": [0.2, 0.2, 0.2, 0.2, 0.2]}
    ]
    
    diagnostics = {}
    
    for ds_file in os.listdir(cache_dir):
        if not ds_file.endswith("_cache.npy"): continue
        dataset_name = ds_file.replace("_cache.npy", "")
        cache = np.load(os.path.join(cache_dir, ds_file), allow_pickle=True)
        
        # Diagnostics
        nu_list = []
        for item in cache:
            view_cache = item['cache'].get('global', next(iter(item['cache'].values())))
            nus = [lcnn['std_diff'] / lcnn['mean_diff'] for lcnn in view_cache['cnn']]
            nu_list.append(np.mean(nus))
        diagnostics[dataset_name] = nu_list
        
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
    
    diag_file = out_file.replace(".csv", "_diagnostics.txt")
    with open(diag_file, "w") as f:
        f.write("Gate Diagnostics (nu distribution):\n")
        for ds, nus in diagnostics.items():
            arr = np.array(nus)
            if len(arr) == 0: continue
            f.write(f"\n{ds}:\n")
            f.write(f"  Count: {len(arr)}\n")
            f.write(f"  Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}, Mean: {np.mean(arr):.4f}\n")
            f.write(f"  % above 0.70 threshold: {np.mean(arr > 0.70) * 100:.2f}%\n")
            f.write(f"  % above 0.60 threshold: {np.mean(arr > 0.60) * 100:.2f}%\n")
            f.write(f"  % above 0.50 threshold: {np.mean(arr > 0.50) * 100:.2f}%\n")
            f.write(f"  % above 0.40 threshold: {np.mean(arr > 0.40) * 100:.2f}%\n")
            
    print(f"Tier-2 evaluations complete. Saved to {out_file} and {diag_file}")
    print(df.to_string())

if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # run_tier2_evaluations()
    pass
