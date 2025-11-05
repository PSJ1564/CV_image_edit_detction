import onnxruntime as ort
import numpy as np

def load_model(model_path="models/final_model.onnx"):
    return ort.InferenceSession(model_path)

def predict(model, feature_vector):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    feature_vector = feature_vector.astype(np.float32).reshape(1, -1)
    pred = model.run([output_name], {input_name: feature_vector})[0]
    return float(pred[0][0])
