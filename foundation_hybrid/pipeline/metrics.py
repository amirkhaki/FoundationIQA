import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    try:
        popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    except:
        popt = beta
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic

def compute_metrics(y_pred, y_true):
    sq = np.reshape(np.asarray(y_true), (-1,))
    q = np.reshape(np.asarray(y_pred), (-1,))
    
    srcc = stats.spearmanr(sq, q)[0]
    krcc = stats.kendalltau(sq, q)[0]
    
    # Logistic fit for PLCC & RMSE
    q_mapped = fit_function(sq, q)
    plcc = stats.pearsonr(sq, q_mapped)[0]
    rmse = np.sqrt(((sq - q_mapped) ** 2).mean())
    
    return {
        "SRCC": float(srcc),
        "KRCC": float(krcc),
        "PLCC": float(plcc),
        "RMSE": float(rmse)
    }

def bootstrap_ci(y_pred, y_true, ref_ids, n_resamples=1000, alpha=0.05):
    """
    Bootstrap 95% CI resampling over REFERENCE images (ref_ids).
    """
    unique_refs = np.unique(ref_ids)
    metrics_dist = {"SRCC": [], "KRCC": [], "PLCC": [], "RMSE": []}
    
    for _ in range(n_resamples):
        sampled_refs = np.random.choice(unique_refs, size=len(unique_refs), replace=True)
        idx = np.concatenate([np.where(ref_ids == r)[0] for r in sampled_refs])
        
        m = compute_metrics(y_pred[idx], y_true[idx])
        for k, v in m.items():
            metrics_dist[k].append(v)
            
    ci = {}
    for k in metrics_dist:
        lower = np.percentile(metrics_dist[k], 100 * (alpha / 2))
        upper = np.percentile(metrics_dist[k], 100 * (1 - alpha / 2))
        ci[k] = (float(lower), float(upper))
        
    return ci
