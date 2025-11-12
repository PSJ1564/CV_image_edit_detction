# -*- coding: utf-8 -*-
"""
SIFT Matching Engine:
- 입력: NumPy 배열 (bgr_image)
- 기능: SIFT 검출, KNN 매칭, Ratio Test 및 거리 필터링 수행
- 출력: kp (키포인트), filtered_matches (정제된 매칭 목록)
"""
import cv2
import numpy as np

def find_sift_matches(bgr_image: np.ndarray, min_match_count=10, min_dist=25):
    """
    SIFT 검출, (KNN + Ratio Test) 매칭, (min_dist) 필터링 수행

    Args:
        bgr_image (np.ndarray): 전처리된 BGR 이미지 (NumPy 배열).
        min_dist (int): 복사-붙여넣기로 인정할 최소 픽셀 거리 (가까운 텍스처 매칭 제거용).

    Returns:
        kp, filtered_matches, error
    """
    
    # 1. 그레이스케일 변환 (전처리된 BGR 이미지 사용)
    try:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        return None, None, f"BGR -> GRAY 변환 오류: {e}"

    # 2. SIFT 검출 및 계산
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
    except cv2.error as e:
        # SIFT는 opencv-contrib-python에 포함될 수 있어 오류 메시지를 남김
        return None, None, f"SIFT 검출 오류: {e}. 'opencv-contrib-python'이 필요할 수 있습니다."

    if des is None or len(des) < 2:
        # 매칭이 불가능한 경우 (키포인트 2개 미만)
        return kp, [], None

    # 3. KNN 매칭 (Ratio Test 준비)
    bf = cv2.BFMatcher()
    # 이미지 자기 자신과의 매칭을 시도 (내부 매칭)
    matches = bf.knnMatch(des, des, k=2)

    # 4. Ratio Test + 자기 자신 매칭 제거
    good_matches = []
    for m, n in matches:
        # 자기 자신과의 매칭 (m.queryIdx == m.trainIdx) 및 Ratio Test
        if m.queryIdx != m.trainIdx and m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 5. min_dist 필터링 (가까운 거리의 매칭 제거)
    filtered_matches = []
    if len(good_matches) > 0:
        pts1 = np.float32([ kp[m.queryIdx].pt for m in good_matches ])
        pts2 = np.float32([ kp[m.trainIdx].pt for m in good_matches ])
        
        # 유클리드 거리 계산
        distances = np.linalg.norm(pts1 - pts2, axis=1)
        
        for i, m in enumerate(good_matches):
            if distances[i] > min_dist:
                filtered_matches.append(m)

    # print(f"[INFO] SIFT 엔진: Ratio Test {len(good_matches)}개 -> min_dist 필터 {len(filtered_matches)}개")
    
    # RANSAC 수행을 위해 키포인트와 필터링된 매칭 목록만 반환
    return kp, filtered_matches, None

# 테스트용 main 블록은 생략합니다.