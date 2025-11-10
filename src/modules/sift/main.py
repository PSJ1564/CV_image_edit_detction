import cv2
import sys
from modules.sift_engine import find_sift_matches
from modules.sift_analyzer import cluster_matches_and_analyze, generate_visualizations

# --- 설정 ---
IMAGE_PATH = 'test_images/example_copy.jpg' 
MIN_DIST_FILTER = 25 # 최소 25픽셀 떨어진 복사만 탐지
DBSCAN_EPS = 20       # 이동 벡터(x, y)가 20픽셀 내외로 비슷하면 같은 그룹
DBSCAN_MIN_SAMPLES = 5 # 최소 5개 이상의 매칭이 모여야 클러스터로 인정
# ----------------

def run_sift_module(image_path):
    print(f"=== SIFT 모듈(DBSCAN 기반) 분석 시작: {image_path} ===")
    
    # 1. Person A: SIFT 매칭 및 필터링
    img, kp, filtered_matches, error = find_sift_matches(
        image_path, 
        min_dist=MIN_DIST_FILTER
    )
    
    if error:
        print(f"[오류] {error}", file=sys.stderr)
        return

    # 2. Person B (분석): DBSCAN 클러스터링 및 변수 계산
    analysis_result = cluster_matches_and_analyze(
        img, kp, filtered_matches, 
        eps=DBSCAN_EPS, 
        min_samples=DBSCAN_MIN_SAMPLES
    )
    
    if analysis_result is None: # sklearn 임포트 오류
        return
        
    metrics, mask, cluster_matches = analysis_result
    
    print("\n--- 최종 계산된 변수 ---")
    print(metrics)
    
    # 3. Person B (시각화): 히트맵 및 매칭 라인 생성
    matches_viz, heatmap_viz = generate_visualizations(
        img, kp, cluster_matches, mask
    )
    
    # 4. 결과 보여주기
    cv2.imshow("Original", img)
    cv2.imshow("SIFT Matches (Largest Cluster)", matches_viz)
    cv2.imshow("Suspicious Region Mask", mask)
    cv2.imshow("Suspicious Region Heatmap", heatmap_viz)
    
    print("\n창을 닫으려면 아무 키나 누르세요...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 스크립트 실행 ---
if __name__ == "__main__":
    run_sift_module(IMAGE_PATH)