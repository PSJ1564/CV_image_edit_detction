import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/final_model.onnx")
MODEL_PATH = os.path.abspath(MODEL_PATH)
from src.modules.sift_module import extract_sift_features
from src.modules.edge_module import extract_edge_features
from src.modules.ela_module import extract_ela_features
from src.inference.model_loader import load_model, predict


class ForgeryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Forgery Detection")
        self.model = load_model()

        self.label = tk.Label(root, text="Select an image file to analyze")
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.btn_open = tk.Button(root, text="Open Image", command=self.open_file)
        self.btn_open.pack(pady=5)

        self.btn_analyze = tk.Button(root, text="Analyze", command=self.analyze, state=tk.DISABLED)
        self.btn_analyze.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.img = None

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if not path:
            return

        img = Image.open(path)
        img = img.resize((400, 300))
        self.img = cv2.imread(path)

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

        self.btn_analyze.config(state=tk.NORMAL)

    def analyze(self):
        sift_feat = extract_sift_features(self.img)
        edge_feat = extract_edge_features(self.img)
        ela_feat = extract_ela_features(self.img)

        feature_vector = np.concatenate([sift_feat, edge_feat, ela_feat], axis=0)

        prob = predict(self.model, feature_vector)

        self.result_label.config(text=f"Manipulation Probability: {prob * 100:.2f}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryGUI(root)
    root.mainloop()
