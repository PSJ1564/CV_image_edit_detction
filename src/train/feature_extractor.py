import cv2
import numpy as np
import os
from src.modules.sift_module import extract_sift_features
from src.modules.edge_module import extract_edge_features
from src.modules.ela_module import extract_ela_features

def extract_features_from_folder(folder_path):
    X = []
    y = []

    for label_name in ["real", "fake"]:
        label = 0 if label_name == "real" else 1
        subfolder = os.path.join(folder_path, label_name)

        for file in os.listdir(subfolder):
            path = os.path.join(subfolder, file)
            img = cv2.imread(path)
            if img is None:
                continue

            sift_feat = extract_sift_features(img)
            edge_feat = extract_edge_features(img)
            ela_feat = extract_ela_features(img)

            feature = np.concatenate([sift_feat, edge_feat, ela_feat], axis=0)

            X.append(feature)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y
