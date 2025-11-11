import cv2
import numpy as np
import time
import glob  
import os    

# --- 1단계: 전처리 함수 ---
def preprocess_images(bgr_image):
    """
    1단계: 전처리 함수
    입력된 BGR 이미지를 받아 노이즈 제거 후,
    분석에 필요한 grayscale과 HSV 이미지를 반환합니다.
    """
    try:
        blurred_bgr = cv2.GaussianBlur(bgr_image, (5, 5), 0)
    except cv2.error as e:
        print(f"Error during GaussianBlur: {e}")
        return None, None

    gray_image = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

    return gray_image, hsv_image

# --- 2단계: edge 평균 강도 계산 함수 ---
def calculate_edge_intensity(gray_image):
    """
    2단계: edge 평균 강도 계산 함수
    grayscale image에 라플라시안 필터를 적용하여,
    0이 아닌 edge 픽셀들의 평균 강도를 반환합니다.
    """
    laplacian_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian_map)
    edge_pixels = laplacian_abs[laplacian_abs > 0]

    if edge_pixels.size == 0:
        edge_intensity_mean = 0.0
    else:
        edge_intensity_mean = np.mean(edge_pixels)

    return edge_intensity_mean

# --- 3단계: 엣지 불연속 비율 계산 함수 ---
def calculate_discontinuity_ratio(gray_image):
    """
    3단계: edge 불연속 비율 계산 함수
    Canny edge를 검출하고, 전체 엣지 픽셀 중 끝점(endpoint) 픽셀의
    비율을 계산하여 반환합니다.
    """
    v = np.median(gray_image)
    sigma = 0.33
    T_low = int(max(0, (1.0 - sigma) * v))
    T_high = int(min(255, (1.0 + sigma) * v))
    canny_map = cv2.Canny(gray_image, T_low, T_high)

    total_edge_pixels = np.sum(canny_map == 255)

    if total_edge_pixels == 0:
        return 0.0, canny_map

    normalized_edges = canny_map / 255.0
    kernel = np.ones((3, 3), dtype=np.float32)
    neighbor_count_map = cv2.filter2D(normalized_edges, -1, kernel)

    endpoint_map = (neighbor_count_map == 2) & (normalized_edges == 1)
    endpoint_pixels = np.sum(endpoint_map)

    discontinuity_ratio = endpoint_pixels / total_edge_pixels

    return discontinuity_ratio, canny_map

# --- 4단계: HSV 색 변화 이상치 계산 함수 ---
def calculate_hsv_change_mean(hsv_image, canny_map):
    """
    4단계: HSV 색 변화 이상치 계산 함수
    Canny edge가 검출된 위치에서 H, S, V 값의
    gradient 총합의 평균을 계산합니다.
    """
    # 1. 엣지 픽셀이 있는지 확인
    total_edge_pixels = np.sum(canny_map == 255)
    if total_edge_pixels == 0:
        return 0.0

    # 2. H, S, V 채널 분리
    h, s, v = cv2.split(hsv_image)
    channels = [h, s, v]

    total_hsv_gradient = np.zeros_like(h, dtype=np.float64)

    # 3. 각 채널(H, S, V)에 대해 gradient 계산
    for channel in channels:
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        total_hsv_gradient += gradient_magnitude

    # 4. 엣지 위치의 변화량만 추출
    hsv_changes_at_edges = total_hsv_gradient[canny_map == 255]

    # 5. 평균 계산
    hsv_change_mean = np.mean(hsv_changes_at_edges)

    return hsv_change_mean

# --- 모듈 최종 통합 함수 (Public) ---
def analyze_edge_continuity(bgr_image):
    """
    A님의 'Edge Continuity' 모듈 최종 통합 함수
    """
    # Step 1: 전처리
    gray_img, hsv_img = preprocess_images(bgr_image)
    if gray_img is None:
        print("Error in Step 1: Preprocessing failed.")
        return None

    # Step 2: 엣지 평균 강도
    edge_intensity_mean = calculate_edge_intensity(gray_img)

    # Step 3: 엣지 불연속 비율 및 Canny 맵 확보
    discontinuity_ratio, canny_map = calculate_discontinuity_ratio(gray_img)

    # Step 4: HSV 색 변화 이상치
    hsv_change_mean = calculate_hsv_change_mean(hsv_img, canny_map)

    # Step 5: 1차원 배열로 취합하여 반환
    output_vector = np.array([
        discontinuity_ratio,
        edge_intensity_mean,
        hsv_change_mean
    ], dtype=np.float32)

    return output_vector

if __name__ == "__main__":
    
    # 1. 테스트할 이미지 폴더 경로 설정
    TEST_IMAGE_FOLDER = "test_images" 
    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")

    print(f"OpenCV/NumPy 모듈 로드 완료.")
    print(f"'{TEST_IMAGE_FOLDER}' 폴더에서 이미지 자동 탐색 시작...")

    # 2. 폴더 내 모든 이미지 파일 경로 탐색
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        search_path = os.path.join(TEST_IMAGE_FOLDER, ext)
        image_paths.extend(glob.glob(search_path))

    if not image_paths:
        print(f"'{TEST_IMAGE_FOLDER}' 폴더에 분석할 이미지가 없습니다.")
        print(f"(참고: {os.path.abspath(TEST_IMAGE_FOLDER)} 경로를 확인하세요)")
    else:
        print(f"총 {len(image_paths)}개의 이미지를 찾았습니다.")

    # 3. 찾은 이미지를 하나씩 반복 처리
    for image_path in image_paths:
        
        print(f"\n---  processing: {image_path} ---")
        
        test_image = cv2.imread(image_path)

        # 이미지 로드 성공 여부 확인
        if test_image is None:
            print(f"'{image_path}' 이미지를 불러오는 데 실패했습니다.")
        else:
            print(f"'{image_path}' 이미지 로드 완료. Shape: {test_image.shape}")

            # 최종 모듈 함수 실행
            print("모듈 (analyze_edge_continuity) 실행")
            feature_vector = analyze_edge_continuity(test_image)

            if feature_vector is not None:
                print("\n 최종 모듈 반환 값 (1D Array)")
                print(feature_vector)

                print("\n 세부 항목 ")
                print(f"1. Discontinuity Ratio: {feature_vector[0]:.4f}")
                print(f"2. Edge Intensity Mean: {feature_vector[1]:.4f}")
                print(f"3. HSV Change Mean:   {feature_vector[2]:.4f}")
            else:
                print("모듈 실행 중 오류가 발생했습니다.")

