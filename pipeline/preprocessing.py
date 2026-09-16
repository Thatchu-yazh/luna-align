import cv2
import numpy as np

def apply_illumination_normalization(img_rgb: np.ndarray, method="CLAHE"):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)
    
    if method == "Phase_Congruency":
        # Gentle Blend: 85% Texture + 15% Structural Phase
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(l_chan)
        
        img_float = np.float32(l_chan) / 255.0
        gx = cv2.Scharr(img_float, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img_float, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        pc_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        norm_l = cv2.addWeighted(base, 0.85, np.uint8(pc_norm), 0.15, 0)
        
    elif method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        norm_l = clahe.apply(l_chan)
    else:
        norm_l = l_chan
        
    merged = cv2.merge([norm_l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB), norm_l