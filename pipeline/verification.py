import cv2
import numpy as np
from scipy.spatial import cKDTree

def adaptive_non_maximal_suppression(kpts0, kpts1, radius=2.0):
    if len(kpts0) < 50: 
        return kpts0, kpts1
        
    tree = cKDTree(kpts1)
    retained = []
    suppressed = np.zeros(len(kpts1), dtype=bool)
    
    for i in range(len(kpts1)):
        if not suppressed[i]:
            retained.append(i)
            neighbors = tree.query_ball_point(kpts1[i], r=radius)
            suppressed[neighbors] = True
            
    return kpts0[retained], kpts1[retained]

def verify_with_magsac(kpts0: np.ndarray, kpts1: np.ndarray, conf: np.ndarray, threshold=2.0):
    if len(kpts0) < 8:
        return None, np.zeros(len(kpts0), dtype=bool), kpts0, kpts1
        
    # 1. MAGSAC++ First (To get accurate inlier ratio)
    H, inliers = cv2.findHomography(kpts0, kpts1, cv2.USAC_MAGSAC, threshold, 0.999, 10000)
    is_inlier = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(kpts0), dtype=bool)
    
    valid_src = kpts0[is_inlier]
    valid_ref = kpts1[is_inlier]
    
    # 2. Gentle ANMS on verified points only
    final_src, final_ref = adaptive_non_maximal_suppression(valid_src, valid_ref, radius=2.0)
    
    return H, is_inlier, final_src, final_ref