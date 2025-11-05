import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

from src.train.feature_extractor import extract_features_from_folder
from src.train.dataset import FeatureDataset


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train():
    # 1) feature dataset 생성
    print("[INFO] Extracting features...")
    X, y = extract_features_from_folder("data/train")  # data/train/real, data/train/fake 필요

    # 2) train/val 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # 3) dataloader 준비
    train_ds = FeatureDataset(X_train, y_train)
    val_ds = FeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 4) 모델 생성
    model = SimpleNN(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # 5) 학습 루프
    print("[INFO] Training model...")
    for epoch in range(20):
        model.train()
        for batch_x, batch_y in train_loader:
            pred = model(batch_x).squeeze()
            loss = loss_fn(pred, batch_y.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.4f}")

    # 6) .pth 저장
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model.pth")
    print("[INFO] Saved PyTorch model: models/final_model.pth")

    # 7) ONNX 변환
    print("[INFO] Exporting ONNX model...")
    dummy = torch.randn(1, X.shape[1])
    torch.onnx.export(
        model,
        dummy,
        "models/final_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    print("[INFO] Saved ONNX model: models/final_model.onnx")


if __name__ == "__main__":
    train()
