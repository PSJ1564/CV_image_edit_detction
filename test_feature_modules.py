import os
import glob
import cv2
import numpy as np

# 경로 설정 (프로젝트 루트 기준)
from src.modules.edge.edge_module import analyze_edge_continuity
from src.modules.ela.ela_module import ela_features_final

TEST_IMAGE_FOLDER = "test_images"
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")

image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(TEST_IMAGE_FOLDER, ext)))

if not image_paths:
    print("테스트할 이미지가 없습니다.")
else:
    for path in image_paths:
        print(f"\n--- Testing {path} ---")
        bgr_image = cv2.imread(path)
        if bgr_image is None:
            print("이미지 로드 실패")
            continue

        # Edge Feature
        edge_vec, canny_map = analyze_edge_continuity(bgr_image)
        print("Edge Feature (3D):", edge_vec)
        print("Canny Map shape:", canny_map.shape)

        # ELA Feature
        ela_vec = ela_features_final(bgr_image)
        print("ELA Feature (3D):", ela_vec)
