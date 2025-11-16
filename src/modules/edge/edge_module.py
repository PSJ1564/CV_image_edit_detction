# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Tuple, Optional

def _to_gray_and_hsv(bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"[EDGE_ERROR] Color conversion failed: {e}")
        return None, None
    return gray_image, hsv_image

def calculate_edge_intensity(gray_image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    edges = laplacian_abs[laplacian_abs > 0]
    return float(np.mean(edges)) if edges.size > 0 else 0.0

def calculate_discontinuity_ratio(gray_image: np.ndarray) -> Tuple[float, np.ndarray]:
    v = np.median(gray_image)
    sigma = 0.33
    T_low = int(max(0, (1.0 - sigma) * v))
    T_high = int(min(255, (1.0 + sigma) * v))
    canny_map = cv2.Canny(gray_image, T_low, T_high)
    total = np.sum(canny_map == 255)
    if total == 0:
        return 0.0, canny_map

    norm_edges = canny_map / 255.0
    kernel = np.ones((3, 3), np.float32)
    neighbor_count = cv2.filter2D(norm_edges, -1, kernel)
    endpoints = (neighbor_count <= 2) & (norm_edges == 1)
    ratio = np.sum(endpoints) / total
    return float(ratio), canny_map

def calculate_hsv_change_mean(hsv_image: np.ndarray, canny_map: np.ndarray) -> float:
    if np.sum(canny_map == 255) == 0:
        return 0.0
    h, s, v = cv2.split(hsv_image)
    gradient_sum = np.zeros_like(h, dtype=np.float64)
    for c in [h, s, v]:
        gx = cv2.Sobel(c, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(c, cv2.CV_64F, 0, 1)
        gradient_sum += np.sqrt(gx**2 + gy**2)
    return float(np.mean(gradient_sum[canny_map == 255]))

def analyze_edge_continuity(bgr_image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    gray, hsv = _to_gray_and_hsv(bgr_image)
    if gray is None: return None
    edge_intensity = calculate_edge_intensity(gray)
    discontinuity_ratio, canny_map = calculate_discontinuity_ratio(gray)
    hsv_change = calculate_hsv_change_mean(hsv, canny_map)
    vector = np.array([discontinuity_ratio, edge_intensity, hsv_change], dtype=np.float32)
    return vector, canny_map
