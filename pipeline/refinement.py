import numpy as np

def refine_keypoints_parabolic(gray0: np.ndarray, gray1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, patch_radius=4):
    refined_pts1 = pts1.copy()
    h1, w1 = gray1.shape
    h0, w0 = gray0.shape
    for i in range(len(pts0)):
        x0, y0 = int(round(pts0[i, 0])), int(round(pts0[i, 1]))
        x1, y1 = int(round(pts1[i, 0])), int(round(pts1[i, 1]))
        if (x0 - patch_radius < 0 or x0 + patch_radius >= w0 or 
            y0 - patch_radius < 0 or y0 + patch_radius >= h0 or
            x1 - patch_radius - 1 < 0 or x1 + patch_radius + 1 >= w1 or 
            y1 - patch_radius - 1 < 0 or y1 + patch_radius + 1 >= h1):
            continue
        patch0 = gray0[y0-patch_radius:y0+patch_radius+1, x0-patch_radius:x0+patch_radius+1].astype(np.float32)
        patch0 -= np.mean(patch0)
        norm0 = np.linalg.norm(patch0)
        if norm0 == 0:
            continue
        patch0 /= norm0
        corrs = np.zeros((3, 3), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                px, py = x1 + dx, y1 + dy
                target = gray1[py-patch_radius:py+patch_radius+1, px-patch_radius:px+patch_radius+1].astype(np.float32)
                target -= np.mean(target)
                norm1 = np.linalg.norm(target)
                if norm1 > 0:
                    corrs[dy+1, dx+1] = np.sum(patch0 * (target / norm1))
        denom_x = 2 * (2 * corrs[1, 1] - corrs[1, 2] - corrs[1, 0])
        denom_y = 2 * (2 * corrs[1, 1] - corrs[2, 1] - corrs[0, 1])
        dx_sub = np.clip((corrs[1, 2] - corrs[1, 0]) / denom_x if abs(denom_x) > 1e-5 else 0.0, -0.5, 0.5)
        dy_sub = np.clip((corrs[2, 1] - corrs[0, 1]) / denom_y if abs(denom_y) > 1e-5 else 0.0, -0.5, 0.5)
        refined_pts1[i, 0] += dx_sub
        refined_pts1[i, 1] += dy_sub
    return refined_pts1