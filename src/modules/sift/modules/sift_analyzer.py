import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_matches_and_analyze(img, kp, filtered_matches, eps=25, min_samples=5):
    """
    Person B (분석)
    A가 전달한 매칭을 DBSCAN으로 클러스터링하고 3가지 변수 계산
    
    Args:
        eps (int): DBSCAN의 최대 탐색 거리 (이동 벡터의 유사성)
        min_samples (int): 클러스터로 인정할 최소 샘플 수
        
    Returns:
        metrics (dict): 3가지 변수가 담긴 사전
        mask (image): 조작 의심 영역 마스크 (0 또는 255)
        largest_cluster_matches (list): 가장 큰 클러스터의 매칭
    """
    
    if len(filtered_matches) < min_samples:
        print("[Person B] 클러스터링할 매칭 수가 부족합니다.")
        metrics = {"match_count": 0, "match_region_ratio": 0.0, "deformation_ratio": 0.0}
        mask = np.zeros(img.shape[:2], dtype="uint8")
        return metrics, mask, []

    # 1. 이동 벡터(Shift Vectors) 계산
    # (x1, y1) -> (x2, y2) 로 이동했을 때의 (dx, dy)
    vectors = []
    for m in filtered_matches:
        pt1 = kp[m.queryIdx].pt
        pt2 = kp[m.trainIdx].pt
        vectors.append([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    
    if not vectors:
        print("[Person B] 유효한 이동 벡터가 없습니다.")
        metrics = {"match_count": 0, "match_region_ratio": 0.0, "deformation_ratio": 0.0}
        mask = np.zeros(img.shape[:2], dtype="uint8")
        return metrics, mask, []

    # 2. [신규] DBSCAN 클러스터링
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(vectors)
    except ImportError:
        print("[오류] 'scikit-learn' 라이브러리가 필요합니다. 'pip install scikit-learn'을 실행하세요.")
        return None, None, None

    # 3. 가장 큰 클러스터 찾기 (노이즈 -1 제외)
    unique_labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
    
    largest_cluster_matches = []
    largest_cluster_id = -1
    
    if len(counts) > 0:
        largest_cluster_index = np.argmax(counts)
        largest_cluster_id = unique_labels[largest_cluster_index]
        
        # 가장 큰 클러스터에 속하는 매칭들만 추출
        for i, m in enumerate(filtered_matches):
            if clusters[i] == largest_cluster_id:
                largest_cluster_matches.append(m)
    
    print(f"[Person B] DBSCAN: 총 {len(unique_labels)}개 클러스터 발견. 최대 클러스터({largest_cluster_id}번) 크기: {len(largest_cluster_matches)}")

    # 4. [신규] 3가지 변수 계산 (클러스터 기준)
    match_count = len(largest_cluster_matches)
    deformation_ratio = 0.0
    match_region_ratio = 0.0
    mask = np.zeros(img.shape[:2], dtype="uint8")

    if match_count > 0:
        # deformation_ratio: 클러스터 매칭의 평균 거리
        distances = [m.distance for m in largest_cluster_matches]
        deformation_ratio = np.mean(distances)

        # match_region_ratio: 매칭 영역 비율 (Convex Hull)
        img_h, img_w = img.shape[:2]
        total_image_area = float(img_h * img_w)
        
        # 원본(src)과 대상(dst) 영역 모두 계산
        src_pts = np.float32([ kp[m.queryIdx].pt for m in largest_cluster_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in largest_cluster_matches ]).reshape(-1, 1, 2)
        
        if len(dst_pts) >= 3:
            hull_src = cv2.convexHull(src_pts)
            hull_dst = cv2.convexHull(dst_pts)
            
            # [신규] 마스크 생성: 두 영역을 흰색(255)으로 채움
            cv2.drawContours(mask, [np.int32(hull_src)], -1, 255, -1)
            cv2.drawContours(mask, [np.int32(hull_dst)], -1, 255, -1)
            
            # 비율은 마스크의 픽셀 수로 계산 (더 정확함)
            matched_area = np.sum(mask > 0)
            match_region_ratio = matched_area / total_image_area

    metrics = {
        "match_count": match_count,
        "match_region_ratio": match_region_ratio,
        "deformation_ratio": deformation_ratio
    }
            
    return metrics, mask, largest_cluster_matches


def generate_visualizations(img, kp, largest_cluster_matches, mask):
    """
    Person B (시각화)
    마스크를 기반으로 히트맵 생성 및 매칭 라인 시각화
    """
    
    # 1. 히트맵 생성
    # 마스크를 부드럽게 블러 처리
    blurred_mask = cv2.GaussianBlur(mask, (31, 31), 0)
    # 컬러맵 적용
    heatmap = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)
    heatmap[mask == 0] = [0, 0, 0] # 조작 아닌 영역은 검게
    
    # 원본과 히트맵 합성
    heatmap_viz = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # 2. 매칭 라인 시각화 (가장 큰 클러스터만)
    matches_viz = cv2.drawMatches(
        img, kp, img, kp, largest_cluster_matches, None,
        matchColor=(0, 255, 0), # 초록색
        singlePointColor=None,
        matchesMask=None, 
        flags=2
    )
    
    return matches_viz, heatmap_viz