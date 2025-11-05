import argparse
import cv2
import numpy as np

# (아직 구현 전이므로 임시 import 형태)
from src.modules.sift_module import extract_sift_features
from src.modules.edge_module import extract_edge_features
from src.modules.ela_module import extract_ela_features
from src.inference.model_loader import load_model, predict


def main():
    parser = argparse.ArgumentParser(description="Image Forgery Detection")
    parser.add_argument("--input", type=str, required=True, help="Input image or video file")
    args = parser.parse_args()

    # 1) 입력 불러오기
    img = cv2.imread(args.input)
    if img is None:
        print("Error: Unable to load input file.")
        return

    # 2) Feature 추출
    sift_feat = extract_sift_features(img)
    edge_feat = extract_edge_features(img)
    ela_feat = extract_ela_features(img)

    # 3) Feature 결합
    feature_vector = np.concatenate([sift_feat, edge_feat, ela_feat], axis=0)

    # 4) 모델 불러오기 & 예측
    model = load_model()  # 내부에서 자동으로 models/final_model.onnx 읽도록
    prob = predict(model, feature_vector)

    # 5) 결과 출력
    print(f"Manipulation Probability: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
