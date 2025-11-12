# -*- coding: utf-8 -*-
"""
전처리 모듈:
- 이미지 로드, Resize, Grayscale 변환, 노이즈 제거 등
- 모든 특징 추출 모듈이 일관된 입력을 받도록 준비
"""

import cv2
import numpy as np

# 모든 모듈이 입력받을 공통 이미지 크기 (선택 사항, NN 학습 시 결정)
TARGET_SIZE = (512, 512) 

def load_image_and_preprocess(image_path: str, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    BGR 이미지를 로드하고, Resize 및 가벼운 노이즈 제거를 수행하여 반환

    Args:
        image_path: 이미지 파일 경로
        target_size: 이미지를 리사이즈할 크기 (None이면 원본 크기 유지)

    Returns:
        np.ndarray (BGR 포맷): 전처리된 이미지 배열, 실패 시 None 반환
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. Resize (선택적으로 크기 통일)
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 2. 노이즈 제거 (Edge 모듈의 전처리 로직과 일관성 유지)
    processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return processed_img

def get_grayscale_and_hsv(bgr_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    BGR 이미지로부터 Grayscale과 HSV 이미지를 반환 (Edge 모듈의 중복 방지)
    """
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return gray_image, hsv_image

if __name__ == "__main__":
    print("Preprocessor 모듈입니다. 다른 파일에서 import하여 사용하세요.")