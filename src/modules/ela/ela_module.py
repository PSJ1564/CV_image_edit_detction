import io, numpy as np
from PIL import Image, ImageChops
import cv2

def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    m = x.max()
    if m > 0:
        x = (x / m) * 255
    return x.astype(np.uint8)

def _ela_gray_inmem(img: Image.Image, quality: int = 90) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    reimg = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img, reimg)
    return np.asarray(diff, dtype=np.float32).max(axis=2)

def ela_features_final(bgr_image: np.ndarray, quality: int = 90) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    ela_gray = _ela_gray_inmem(img, quality=quality)
    ela_u8 = _to_uint8(ela_gray)
    mean = float(ela_gray.mean())
    std = float(ela_gray.std())
    high_ratio = float((ela_u8 >= mean + 2 * std).mean())
    return np.array([std, high_ratio, mean], dtype=np.float32)
