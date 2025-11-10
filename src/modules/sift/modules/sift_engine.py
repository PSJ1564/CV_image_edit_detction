import cv2
import numpy as np

def find_sift_matches(image_path, min_match_count=10, min_dist=25):
    """
    Person A (엔진)
    SIFT 검출, (KNN + Ratio Test) 매칭, (min_dist) 필터링 수행
    
    Args:
        min_dist (int): 복사-붙여넣기로 인정할 최소 픽셀 거리.
                        (가까운 텍스처 매칭 제거용)
    
    Returns:
        img, kp, filtered_matches, error
    """
    
    # 1. 이미지 로드 및 그레이스케일
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, f"이미지를 로드할 수 없습니다: {image_path}"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 검출 및 계산
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
    except cv2.error as e:
        return None, None, None, "'opencv-contrib-python'이 필요합니다."

    if des is None or len(des) < 2:
        return img, kp, [], None

    # 3. KNN 매칭 (Ratio Test 준비)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, des, k=2)

    # 4. Ratio Test + 자기 자신 매칭 제거
    good_matches = []
    for m, n in matches:
        if m.queryIdx != m.trainIdx and m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 5. [신규] min_dist 필터링 (자기 중복 필터링)
    #    (매칭된 두 점 사이의 거리가 min_dist 픽셀 이상이어야 함)
    filtered_matches = []
    if len(good_matches) > 0:
        pts1 = np.float32([ kp[m.queryIdx].pt for m in good_matches ])
        pts2 = np.float32([ kp[m.trainIdx].pt for m in good_matches ])
        
        # 유클리드 거리 계산
        distances = np.linalg.norm(pts1 - pts2, axis=1)
        
        for i, m in enumerate(good_matches):
            if distances[i] > min_dist:
                filtered_matches.append(m)

    print(f"[Person A] SIFT 엔진: {len(matches)}개 매칭 -> Ratio Test {len(good_matches)}개 -> min_dist 필터 {len(filtered_matches)}개 찾음.")
    
    # Person B가 클러스터링에 사용할 수 있도록 데이터 반환
    return img, kp, filtered_matches, None