import numpy as np

def evaluate_4x4_spatial_uniformity(points: np.ndarray, img_shape: tuple):
    """
    Audits tie-point dispersion across a 4x4 matrix, returning occupancy & Shannon spatial entropy.
    """
    h, w = img_shape[:2]
    grid = np.zeros((4, 4), dtype=int)
    
    if len(points) == 0:
        return grid, 0.0, 0.0
        
    x_bins = np.linspace(0, w, 5)
    y_bins = np.linspace(0, h, 5)
    
    x_idx = np.clip(np.digitize(points[:, 0], x_bins) - 1, 0, 3)
    y_idx = np.clip(np.digitize(points[:, 1], y_bins) - 1, 0, 3)
    
    for y, x in zip(y_idx, x_idx):
        grid[y, x] += 1
        
    occupied = np.count_nonzero(grid)
    coverage = (occupied / 16.0) * 100.0
    
    probs = grid.flatten() / len(points)
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log2(probs))
    uniformity = (entropy / np.log2(16.0)) * 100.0
    
    return grid, coverage, uniformity