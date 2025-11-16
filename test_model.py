import torch
import torch.nn as nn
import torch.onnx
import os

# --------------------------
# Dummy Model Definition
# --------------------------
class DummyBinaryClassifier(nn.Module):
    def __init__(self):
        super(DummyBinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# --------------------------
# Initialize dummy model
# --------------------------
model = DummyBinaryClassifier()
model.eval()

# --------------------------
# ONNX Export Path
# 프로젝트 규칙에 따라 ForgeryDetection/models/final_model.onnx 로 저장
# --------------------------
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "final_model.onnx")

# --------------------------
# Example input (shape: 1x10)
# --------------------------
dummy_input = torch.randn(1, 10, dtype=torch.float32)

# --------------------------
# Export to ONNX
# --------------------------
torch.onnx.export(
    model,
    dummy_input,
    save_path,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"[SUCCESS] Dummy ONNX model saved at: {save_path}")
print("You can now run GUI and receive mock predictions.")
