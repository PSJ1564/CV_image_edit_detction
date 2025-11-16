import os
import onnxruntime as ort
import numpy as np
from modules.feature_extractor.extractor import extract_features_from_image

# =====================================================
# 프로젝트 경로 및 모델 경로 설정
# =====================================================
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.onnx")

# =====================================================
# ONNX 모델 로딩 (글로벌 세션)
# =====================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =====================================================
# 이미지 예측 함수
# =====================================================
def predict_image(img):
    """
    img: OpenCV BGR 이미지
    return: 조작 확률 (label=1 기준)
    """
    # 1. 이미지 → 10D feature vector
    vec10 = extract_features_from_image(img)
    vec10 = np.array(vec10, dtype=np.float32).reshape(1, -1)  # (1, 10)

    # 2. ONNX inference
    output = session.run([output_name], {input_name: vec10})[0]  # shape (1,2)

    # 3. Softmax 확률 계산
    exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)

    # 4. 조작일 확률 반환
    return 1.0-float(probs[0, 1])
