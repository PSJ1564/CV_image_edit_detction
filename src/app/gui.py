import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QFont

# 기존 프로젝트 구조에 맞게 run_inference import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from modules.inference.run_inference import predict_image
except Exception as e:
    print("[FATAL] modules.inference.run_inference import 실패")
    print("[CAUSE]", e)
    sys.exit(1)

# -----------------------
# Utility Functions
# -----------------------
def prob_to_color(prob):
    """확률 0~1에 따라 초록->노랑->빨강 색상 반환"""
    r = int(255 * prob)
    g = int(255 * (1 - prob))
    b = 0
    return f"rgb({r},{g},{b})"

def cv2_to_qpixmap(img, max_size=(400, 400)):
    """OpenCV BGR 이미지를 QPixmap으로 변환하고 모듈별 특징점 상위 10% 오버레이"""
    overlay = img.copy()

    # ==========================
    # Edge module overlay (상위 10% endpoints)
    # ==========================
    from modules.edge.edge_module import analyze_edge_continuity
    edge_result = analyze_edge_continuity(img)
    if edge_result is not None:
        vector, canny_map = edge_result
        # endpoints
        norm_edges = canny_map / 255.0
        kernel = np.ones((3,3), np.float32)
        neighbor_count = cv2.filter2D(norm_edges, -1, kernel)
        endpoints = ((neighbor_count <= 2) & (norm_edges==1))
        ys, xs = np.where(endpoints)

        if len(xs) > 0:
            # 강도 기준 상위 10%
            vals = canny_map[ys, xs]
            threshold = np.percentile(vals, 90)
            strong_idx = vals >= threshold
            for x, y in zip(xs[strong_idx], ys[strong_idx]):
                cv2.circle(overlay, (x, y), 1, (0,0,255), 1)  # 빨간색

    # ==========================
    # ELA module overlay (상위 10%)
    # ==========================
    from modules.ela.ela_module import _ela_gray_inmem
    from PIL import Image
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    ela_gray = _ela_gray_inmem(img_pil)  # float32 배열

    if ela_gray is not None:
        threshold = np.percentile(ela_gray, 90)
        ys, xs = np.where(ela_gray >= threshold)
        overlay[ys, xs] = [255, 0, 0]  # 파란색

    # ==========================
    # Resize & QPixmap 변환
    # ==========================
    h, w, c = overlay.shape
    scale = min(max_size[0]/w, max_size[1]/h, 1.0)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(overlay, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, new_w, new_h, 3*new_w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# -----------------------
# GUI Class
# -----------------------
class ForgeryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forgery Detection")
        self.resize(600, 600)

        layout = QVBoxLayout()
        self.label = QLabel("파일을 선택하세요.")
        self.label.setFont(QFont("Arial", 16))
        layout.addWidget(self.label)

        self.btn_img = QPushButton("이미지 선택")
        self.btn_vid = QPushButton("동영상 선택")
        layout.addWidget(self.btn_img)
        layout.addWidget(self.btn_vid)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(400, 400)
        layout.addWidget(self.thumb_label)

        self.btn_img.clicked.connect(self.open_image)
        self.btn_vid.clicked.connect(self.open_video)

        self.setLayout(layout)

    def update_result(self, img, prob):
        """썸네일 표시 + 확률 텍스트 업데이트"""
        color = prob_to_color(prob)
        self.label.setText(f"조작 확률: {prob:.3f}")
        self.label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 18pt;")
        self.thumb_label.setPixmap(cv2_to_qpixmap(img))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is None:
                self.label.setText("이미지 로드 실패")
                return
            try:
                prob = predict_image(img)
                # GUI에서 반환되는 조작 확률 기준으로 1-prob 필요 시 predict_image에서 처리
                self.update_result(img, prob)
            except Exception as e:
                self.label.setText(f"오류 발생: {str(e)}")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "동영상 선택", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            cap = cv2.VideoCapture(path)
            frame_probs = []
            last_frame = None  # 마지막 프레임을 썸네일로 사용

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame.copy()
                prob = predict_image(frame)
                frame_probs.append(prob)

            cap.release()

            if frame_probs:
                avg_prob = sum(frame_probs) / len(frame_probs)
            else:
                avg_prob = 0.0
                last_frame = np.zeros((224, 224, 3), dtype=np.uint8)  # 빈 이미지

            # 이미지 처리와 동일하게 썸네일 + 색상 표시
            self.update_result(last_frame, avg_prob)



# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ForgeryApp()
    window.show()
    sys.exit(app.exec_())
