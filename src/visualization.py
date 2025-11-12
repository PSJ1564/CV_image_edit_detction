# -*- coding: utf-8 -*-
"""
시각화 모듈:
- 각 특징 모듈에서 반환된 의심 영역 마스크들을 가중 합산
- 최종 히트맵 이미지 생성 및 Score와 결합
"""

import cv2
import numpy as np

# 각 모듈별 최종 마스크에 적용할 가중치 (NN 학습 후 결정 필요)
WEIGHTS = {
    "sift_mask": 0.4,
    "edge_mask": 0.3,
    "ela_mask": 0.3
}

def create_heatmap(image: np.ndarray, masks: dict, score: float) -> np.ndarray:
    """
    3가지 마스크를 합산하여 최종 히트맵 이미지를 생성합니다.

    Args:
        image: 원본 BGR 이미지
        masks: {mask_name: mask_np_array} 형태의 딕셔너리
        score: NN 모델이 예측한 최종 조작 확률 (0~1)

    Returns:
        np.ndarray (BGR): 원본 이미지 위에 히트맵이 오버레이된 이미지
    """
    
    h, w, _ = image.shape
    
    # 1. 마스크 정규화 및 가중 합산 (마스크는 0~1 범위여야 함)
    # 마스크가 0~255라면 255로 나눠 0~1로 정규화하는 과정이 필요합니다.
    
    weighted_sum_mask = np.zeros((h, w), dtype=np.float32)
    
    for key, mask in masks.items():
        if key in WEIGHTS:
            # 마스크 크기를 이미지와 동일하게 조정
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # 마스크 값 정규화 (0~255 -> 0~1)
            mask_normalized = mask_resized.astype(np.float32) / 255.0 if mask_resized.max() > 1 else mask_resized
            
            weighted_sum_mask += mask_normalized * WEIGHTS[key]
            
    # 최종 마스크 정규화 (최대값을 1로)
    weighted_sum_mask = np.clip(weighted_sum_mask, 0, 1)

    # 2. 컬러맵 적용 (예: JET)
    # 마스크를 0~255로 스케일링
    mask_u8 = (weighted_sum_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)

    # 3. 원본 이미지와 히트맵 오버레이
    alpha = 0.5 + (score * 0.4) # 조작 확률이 높을수록 히트맵을 더 선명하게 (0.5 ~ 0.9)
    overlay_img = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    
    return overlay_img

if __name__ == "__main__":
    print("Visualization 모듈입니다. 다른 파일에서 import하여 사용하세요.")