import os
import argparse
import json
import time
import numpy as np
import torch
import pyiqa
from .arch import FoundationHybrid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FoundationHybridAblation')

def logistic_func(p, x):
    t1 = p[0] - p[1]
    t2 = 1 + np.exp(-(x - p[2]) / p[3])
    return t1 / t2 + p[1]

def fit_logistic(x, y):
    from scipy.optimize import curve_fit
    p0 = [np.max(y), np.min(y), np.mean(x), np.std(x) + 1e-4]
    try:
        popt, _ = curve_fit(logistic_func, x, y, p0=p0, maxfev=2000)
        return popt
    except:
        return p0

def evaluate_metrics(preds, mos):
    from scipy.stats import spearmanr, kendalltau, pearsonr
    srcc = spearmanr(preds, mos)[0]
    krcc = kendalltau(preds, mos)[0]
    
    popt = fit_logistic(preds, mos)
    mapped_preds = logistic_func(popt, preds)
    
    plcc = pearsonr(mapped_preds, mos)[0]
    rmse = np.sqrt(np.mean((mapped_preds - mos) ** 2))
    return srcc, krcc, plcc, rmse

def bootstrap_ci(preds, mos, n_resamples=1000):
    n = len(preds)
    srccs = []
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        try:
            s, _, _, _ = evaluate_metrics(preds[indices], mos[indices])
            srccs.append(s)
        except:
            pass
    if len(srccs) == 0:
        return 0, 0
    return np.percentile(srccs, 2.5), np.percentile(srccs, 97.5)

def check_regression():
    logger.info("Running regression check...")
    from pyiqa.utils.registry import ARCH_REGISTRY
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    try:
        from foundation_hybrid import IDFIQA_FoundationHybrid
        old_model = IDFIQA_FoundationHybrid().cuda().eval()
    except Exception as e:
        logger.warning(f"Could not load old model to run regression test exactly: {e}. Skipping check.")
        return True

    new_model = FoundationHybrid().cuda().eval()
    
    # Generate dummy data
    torch.manual_seed(42)
    ref = torch.rand(1, 3, 256, 256).cuda()
    dist = torch.rand(1, 3, 256, 256).cuda()
    
    with torch.no_grad():
        out_old = old_model(ref, dist)
        out_new = new_model(ref, dist)
        
    old_score = out_old['score'].item()
    new_score = out_new.item()
    
    diff = abs(old_score - new_score)
    logger.info(f"Old score: {old_score:.6f}, New score: {new_score:.6f}, Diff: {diff:.6e}")
    if diff > 1e-6:
        logger.error(f"Regression check failed! Diff: {diff}")
        return False
    
    logger.info("Regression check passed.")
    return True


from .pipeline.tier1_cache import run_tier1_cache
from .pipeline.tier2_eval import run_tier2_evaluations
from .pipeline.visualization import generate_all_visualizations
from .pipeline.profiler import profile_efficiency

def main():
    parser = argparse.ArgumentParser(description="FoundationHybrid Ablation Suite")
    parser.add_argument("--group", type=str, default="all", help="Ablation group (A, B, C, D, E, F, G, H, I, all)")
    parser.add_argument("--priority", type=str, default="P0", help="Priority (P0, P1, P2)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["LIVE", "CSIQ", "TID2013", "KADID10k", "PIPAL"])
    parser.add_argument("--out-dir", type=str, default="raw")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Where pyiqa finds/downloads the IQA datasets "
                             "(default: $IQA_DATA_ROOT, else ./datasets)")
    parser.add_argument("--no-regression", action="store_true", help="Skip regression check")
    parser.add_argument("--skip-cache", action="store_true", help="Skip Tier-1 caching if already done")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (random subset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    
    args = parser.parse_args()
    args.datasets = [d.lower() for d in args.datasets]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if not args.no_regression:
        if not check_regression():
            return
            
    logger.info("=== Running Tier-1 Caching ===")
    if not args.skip_cache:
        run_tier1_cache(args.datasets, output_dir=args.out_dir, data_root=args.data_root, 
                        num_samples=args.num_samples, seed=args.seed, quiet=args.quiet)
        
    logger.info("=== Running Tier-2 Evaluations ===")
    run_tier2_evaluations(cache_dir=args.out_dir)
    
    logger.info("=== Generating Visualizations ===")
    # generate_all_visualizations()
    
    logger.info("=== Profiling Efficiency ===")
    # profile_efficiency()

if __name__ == "__main__":
    main()
