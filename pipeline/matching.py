import torch
import numpy as np
import kornia as K
import kornia.feature as KF
import streamlit as st
import cv2

@st.cache_resource
def load_loftr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()
    return matcher, device

def extract_dense_correspondences(gray0: np.ndarray, gray1: np.ndarray, target_dim=(640, 640)):
    matcher, device = load_loftr()
    orig_h0, orig_w0 = gray0.shape
    orig_h1, orig_w1 = gray1.shape
    
    t0 = torch.from_numpy(gray0).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    t1 = torch.from_numpy(gray1).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    t0_r = K.geometry.resize(t0, target_dim, antialias=True)
    t1_r = K.geometry.resize(t1, target_dim, antialias=True)
    
    with torch.no_grad():
        out = matcher({"image0": t0_r, "image1": t1_r})
        
    kpts0 = out['keypoints0'].cpu().numpy()
    kpts1 = out['keypoints1'].cpu().numpy()
    conf = out['confidence'].cpu().numpy()
    
    if len(kpts0) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
        
    # --- NEW: TOP-K CONFIDENCE FILTER ---
    # Sort matches by AI confidence and keep only the top 1000.
    # This prevents the AI from hallucinating matches on repetitive synthetic craters.
    if len(conf) > 1000:
        top_idx = np.argsort(conf)[::-1][:1000]
        kpts0 = kpts0[top_idx]
        kpts1 = kpts1[top_idx]
        conf = conf[top_idx]
        
    kpts0[:, 0] *= (orig_w0 / target_dim[1])
    kpts0[:, 1] *= (orig_h0 / target_dim[0])
    kpts1[:, 0] *= (orig_w1 / target_dim[1])
    kpts1[:, 1] *= (orig_h1 / target_dim[0])
    
    return kpts0, kpts1, conf