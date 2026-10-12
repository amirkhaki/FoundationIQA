"""Tier-2 evaluation for the DINO-only architecture (``FoundationDino``).

Post-hoc re-scoring of a Tier-1 cache: no model forward pass. Only the ``dino`` part of each
view cache is used, so this also runs on a *hybrid* Tier-1 cache (whose ``dino`` entries are
identical), which makes the baseline directly comparable to ``A3_dino_only`` of the hybrid sweep.
"""
import os
import numpy as np
import pandas as pd
from .metrics import compute_metrics

VIEWS = ('global', 'center', 'texture')

DEFAULT_DINO_WEIGHTS = [0.25, 0.30, 0.25, 0.12, 0.08]
DEFAULT_TERM_WEIGHTS = (0.45, 0.45, 0.10)  # (dists, patch_cos, cls)
DEFAULT_VIEW_WEIGHTS = (0.60, 0.25, 0.15)  # (global, center, texture)


def _patch_cos(ldino, mode, k_ratio, alpha):
    cos_flat = ldino['cos_flat']
    mean_cos = ldino['mean_cos']
    if mode == 'mean':
        return mean_cos
    k_val = max(1, int(len(cos_flat) * k_ratio))
    worst_cos = np.mean(np.sort(cos_flat)[:k_val])
    if mode == 'worst':
        return worst_cos
    return alpha * mean_cos + (1 - alpha) * worst_cos


def dino_view_score(view_cache, config):
    """Score of one view from its cached per-layer DINO terms."""
    layer_w = config.get('dino_weights', DEFAULT_DINO_WEIGHTS)
    w_dists, w_patch, w_cls = config.get('term_weights', DEFAULT_TERM_WEIGHTS)
    mode = config.get('patch_cos_mode', 'both')
    k_ratio = config.get('worst_k_ratio', 0.10)
    alpha = config.get('patch_cos_alpha', 0.5)

    total = 0.0
    for ldino, lw in zip(view_cache['dino'], layer_w):
        patch_cos = _patch_cos(ldino, mode, k_ratio, alpha) if w_patch != 0 else 0.0
        ls = w_dists * ldino['dists_score'] + w_patch * patch_cos + w_cls * ldino['cls_score']
        total += ls * lw
    return total / np.sum(layer_w)


def compute_from_cache_dino(cache, config):
    """Returns [(mos, score, dist_type), ...]."""
    view_w = dict(zip(VIEWS, config.get('view_weights', DEFAULT_VIEW_WEIGHTS)))
    scores = []
    for item in cache:
        final = 0.0
        for view, w in view_w.items():
            if w == 0:
                continue
            if view not in item['cache']:
                raise KeyError(f"View '{view}' (weight {w}) is missing from the cache; "
                               f"cached views: {sorted(item['cache'])}")
            final += w * dino_view_score(item['cache'][view], config)
        scores.append((item['mos'], final, item['dist_type']))
    return scores


# Sanity set only - the real ablations get defined later. Add entries here.
DINO_CONFIGS = [
    {"id": "DINO_baseline"},  # == hybrid A3_dino_only
    {"id": "DINO_global_only", "view_weights": (1.0, 0.0, 0.0)},
    {"id": "DINO_center_only", "view_weights": (0.0, 1.0, 0.0)},
    {"id": "DINO_texture_only", "view_weights": (0.0, 0.0, 1.0)},
    {"id": "DINO_last_layer", "dino_weights": [0, 0, 0, 0, 1]},
]


def run_tier2_dino(cache_dir="raw/", out_csv=None, num_samples=None, seed=None, configs=None):
    configs = DINO_CONFIGS if configs is None else configs
    results = []

    for ds_file in sorted(os.listdir(cache_dir)):
        if not ds_file.endswith("_cache.npy"):
            continue
        dataset_name = ds_file.replace("_cache.npy", "")
        cache = np.load(os.path.join(cache_dir, ds_file), allow_pickle=True)

        for cfg in configs:
            scores = compute_from_cache_dino(cache, cfg)
            y_true = np.array([s[0] for s in scores])
            y_pred = np.array([s[1] for s in scores])
            results.append({"study_id": cfg["id"], "dataset": dataset_name, **compute_metrics(y_pred, y_true)})

    df = pd.DataFrame(results)

    if out_csv is not None:
        out_file = out_csv
    elif num_samples is not None:
        datasets_processed = list(df['dataset'].unique())
        prefix = datasets_processed[0] if len(datasets_processed) == 1 else "combined"
        out_file = f"dino_{prefix}_{num_samples}_seed_{seed}.csv"
    else:
        out_file = "master_results_dino.csv"

    df.to_csv(out_file, index=False)
    print(f"DINO Tier-2 evaluations complete. Saved to {out_file}")
    print(df.to_string())
    return df


if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    pass
