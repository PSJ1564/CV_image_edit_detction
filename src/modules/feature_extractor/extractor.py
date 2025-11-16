import cv2
import numpy as np
from modules.edge.edge_module import analyze_edge_continuity
from modules.ela.ela_module import ela_features_final

def extract_features_from_image(bgr_image: np.ndarray) -> np.ndarray:
    edge_vec, _ = analyze_edge_continuity(bgr_image)
    ela_vec = ela_features_final(bgr_image)
    
    # 추가 4D feature: Laplacian variance, HSV mean
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.mean(axis=(0, 1)).astype(np.float32)  # H, S, V
    
    feature_vec = np.concatenate([edge_vec, ela_vec, [lap_var], hsv_mean]).astype(np.float32)
    return feature_vec  # 10D
