import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

class TransformationBenchmark:
    def __init__(self, src_pts, dst_pts, src_val=None, dst_val=None):
        self.src_train = src_pts
        self.dst_train = dst_pts
        self.src_val = src_val if src_val is not None else src_pts
        self.dst_val = dst_val if dst_val is not None else dst_pts

    def eval_affine(self):
        try:
            M, _ = cv2.estimateAffine2D(self.src_train, self.dst_train)
            if M is None: return None
            pred_train = (M @ np.hstack([self.src_train, np.ones((len(self.src_train), 1))]).T).T
            pred_val = (M @ np.hstack([self.src_val, np.ones((len(self.src_val), 1))]).T).T
            val_residuals = np.linalg.norm(pred_val - self.dst_val, axis=1)
            return {
                "name": "Affine (Rigid/Shear)",
                "train_rmse": float(np.sqrt(np.mean(np.linalg.norm(pred_train - self.dst_train, axis=1)**2))),
                "val_rmse": float(np.sqrt(np.mean(val_residuals**2))),
                "residuals": val_residuals,
                "warp_fn": lambda img, dsize: cv2.warpAffine(img, M, dsize)
            }
        except Exception: return None

    def eval_homography(self):
        try:
            H, _ = cv2.findHomography(self.src_train, self.dst_train, 0)
            if H is None: return None
            def xform(pts):
                p = (H @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
                return p[:, :2] / p[:, 2:3]
            train_r = np.linalg.norm(xform(self.src_train) - self.dst_train, axis=1)
            val_r = np.linalg.norm(xform(self.src_val) - self.dst_val, axis=1)
            return {
                "name": "Homography (Planar Perspective)",
                "train_rmse": float(np.sqrt(np.mean(train_r**2))),
                "val_rmse": float(np.sqrt(np.mean(val_r**2))),
                "residuals": val_r,
                "warp_fn": lambda img, dsize: cv2.warpPerspective(img, H, dsize)
            }
        except Exception: return None

    def eval_tps(self):
        try:
            indices = np.random.choice(len(self.src_train), min(250, len(self.src_train)), replace=False)
            c_src, c_dst = self.src_train[indices], self.dst_train[indices]
            rbf_x = RBFInterpolator(c_src, c_dst[:, 0], kernel='thin_plate_spline', smoothing=1.0)
            rbf_y = RBFInterpolator(c_src, c_dst[:, 1], kernel='thin_plate_spline', smoothing=1.0)
            pred_v_x, pred_v_y = rbf_x(self.src_val), rbf_y(self.src_val)
            val_r = np.sqrt((pred_v_x - self.dst_val[:, 0])**2 + (pred_v_y - self.dst_val[:, 1])**2)
            
            tps_cv = cv2.createThinPlateSplineShapeTransformer()
            tps_cv.estimateTransformation(
                c_dst.reshape(1, -1, 2).astype(np.float32), 
                c_src.reshape(1, -1, 2).astype(np.float32), 
                [cv2.DMatch(i, i, 0) for i in range(len(c_src))]
            )
            return {
                "name": "Thin-Plate Spline (3D Elastic Relief)",
                "train_rmse": float(np.sqrt(np.mean((rbf_x(self.src_train) - self.dst_train[:, 0])**2 + (rbf_y(self.src_train) - self.dst_train[:, 1])**2))),
                "val_rmse": float(np.sqrt(np.mean(val_r**2))),
                "residuals": val_r,
                "warp_fn": lambda img, dsize: tps_cv.warpImage(img)
            }
        except Exception: return None

    def benchmark_all(self):
        models = {}
        for fn in [self.eval_affine, self.eval_homography, self.eval_tps]:
            res = fn()
            if res: models[res['name']] = res
        return models