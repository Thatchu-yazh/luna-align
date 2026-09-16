import numpy as np

def split_train_validation_spatially(src_pts: np.ndarray, dst_pts: np.ndarray, val_ratio=0.20):
    n = len(src_pts)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    boundary = int(n * (1.0 - val_ratio))
    return (src_pts[idx[:boundary]], dst_pts[idx[:boundary]]), (src_pts[idx[boundary:]], dst_pts[idx[boundary:]])

def compute_residual_metrics(residuals: np.ndarray):
    if len(residuals) == 0:
        return {k: 0.0 for k in ["rmse", "median", "p90", "max", "std"]}
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "median": float(np.median(residuals)),
        "p90": float(np.percentile(residuals, 90)),
        "max": float(np.max(residuals)),
        "std": float(np.std(residuals))
    }