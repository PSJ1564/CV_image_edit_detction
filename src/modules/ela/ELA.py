# -*- coding: utf-8 -*-
"""
ELA Final Integrated Module (모듈형 + Batch 실행형)
--------------------------------------------------
- import해서 단일 이미지 feature 추출 가능 (함수형 모듈)
- python 파일을 직접 실행하면 → 폴더 내 이미지 전체 자동 분석
- 결과: [ELA_std, high_intensity_ratio, ELA_mean] → CSV 저장
"""

import io, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, List
from PIL import Image, ImageChops
import cv2

# ===========================================
# 0) 공통 타입 정의
# ===========================================
# → 코드 가독성 향상용: high_mode 파라미터 타입 지정
HighMode = Literal["otsu", "mean+2std", "percentile"]

# → 지원 이미지 확장자 (JPEG, PNG, BMP, TIFF 모두 가능)
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ===========================================
# 1) 기본 유틸 함수들
# ===========================================
def _to_uint8(x: np.ndarray) -> np.ndarray:
    """실수형 배열을 0~255 범위의 8bit 정수형으로 변환"""
    x = x.astype(np.float32)
    x -= x.min()
    m = x.max()
    if m > 0:
        x = (x / m) * 255.0
    return x.astype(np.uint8)

def _ela_gray_inmem(img_pil: Image.Image, quality: int = 90) -> np.ndarray:
    """
    ELA 핵심 단계:
    - 이미지를 메모리에서 JPEG(quality)로 재압축
    - 원본과 재압축본의 차이(|A-B|)를 구함
    - 채널 중 가장 큰 차이를 픽셀 단위로 사용 (그레이 ELA 맵)
    """
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # 메모리상에서 JPEG로 재저장
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    reimg = Image.open(buf).convert("RGB")

    # 원본과 재압축본의 절대차 계산
    diff = ImageChops.difference(img_pil, reimg)
    diff_np = np.asarray(diff, dtype=np.float32)
    return diff_np.max(axis=2)  # [H,W] float → 단일 채널 반환

def _auto_threshold(ela_u8: np.ndarray,
                    mode: HighMode = "otsu",
                    mean_val: float = None,
                    std_val: float = None,
                    percentile: float = 98.0) -> float:
    """
    고강도(밝은 영역) 임계값 계산 함수
    ---------------------------------
    - mode="otsu": 자동 이진화 (기본)
      → 단색/저대비 이미지일 경우 percentile 값으로 대체
    - mode="mean+2std": 평균 + 2표준편차 기준
    - mode="percentile": 상위 백분위수(percentile) 기준
    """
    if mode == "otsu":
        retval, _ = cv2.threshold(ela_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(retval)
        # Otsu가 실패할 경우 대비 (거의 단색 이미지)
        if thr <= 1 or thr >= 254:
            thr = float(np.percentile(ela_u8, percentile))
        return thr
    elif mode == "mean+2std":
        if mean_val is None or std_val is None:
            mean_val = float(ela_u8.mean()); std_val = float(ela_u8.std())
        return float(np.clip(mean_val + 2.0 * std_val, 0, 255))
    elif mode == "percentile":
        return float(np.percentile(ela_u8, percentile))
    else:
        raise ValueError("high_mode must be one of {'otsu','mean+2std','percentile'}")

# ===========================================
# 2) 핵심 ELA feature 추출 함수 (팀 모듈에서 import할 부분)
# ===========================================
def ela_features_final(image_path: str,
                       quality: int = 90,
                       high_mode: HighMode = "otsu",
                       percentile: float = 98.0) -> np.ndarray:
    """
    ELA 기반 3지표 산출 함수
    ----------------------------
    입력:
        image_path : 이미지 경로
        quality : JPEG 재압축 품질(85~95 권장)
        high_mode : 임계 계산 방식 ('otsu', 'mean+2std', 'percentile')
    반환:
        np.array([ELA_std, high_intensity_ratio, ELA_mean], dtype=float)
    """

    # PIL로 이미지 열기 (PNG, BMP, TIFF 모두 지원)
    img = Image.open(image_path)
    ela_gray = _ela_gray_inmem(img, quality=quality)

    # 평균 / 표준편차 계산 (전역 통계)
    ELA_mean = float(ela_gray.mean())
    ELA_std  = float(ela_gray.std())

    # 밝은 영역 비율 계산
    ela_u8 = _to_uint8(ela_gray)
    thr = _auto_threshold(ela_u8, mode=high_mode,
                          mean_val=ELA_mean, std_val=ELA_std,
                          percentile=percentile)
    high_ratio = float((ela_u8 >= thr).mean())

    return np.array([ELA_std, high_ratio, ELA_mean], dtype=float)

# ===========================================
# 3) Batch 실행 유틸 (폴더 내 이미지 자동 수집용)
# ===========================================
def script_dir() -> Path:
    """현재 실행 중인 .py 파일이 위치한 폴더 경로 반환"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()

def gather_images_same_dir() -> List[Path]:
    """
    스크립트와 같은 폴더에 존재하는 이미지 파일 목록 수집
    (jpg, png, bmp, tiff 등 확장자 기준)
    """
    root = script_dir()
    files: List[Path] = []
    for ext in IMG_EXTS:
        files += list(root.glob(f"*{ext}"))
        files += list(root.glob(f"*{ext.upper()}"))
    # 중복 제거 (dict 이용)
    return list(dict.fromkeys(files))

# ===========================================
# 4) Main: 폴더 내 모든 이미지 처리 + CSV 저장
# ===========================================
def main():
    """
    Batch 실행 함수
    -----------------
    1. 같은 폴더의 이미지 파일 자동 탐색
    2. 각 이미지별 ELA feature 계산
    3. 결과를 CSV(Ela_features_batch.csv)로 저장
    """
    quality = 90
    high_mode = "otsu"
    percentile = 98.0
    out_csv = script_dir() / "ELA_features_batch.csv"

    # 1. 이미지 수집
    paths = gather_images_same_dir()
    print(f"[INFO] Target folder: {script_dir()}")
    print(f"[INFO] Found {len(paths)} image files.\n")

    rows = []
    total = len(paths)

    # 2. 각 이미지별 feature 계산
    for i, p in enumerate(paths, 1):
        try:
            vals = ela_features_final(str(p), quality=quality,
                                      high_mode=high_mode, percentile=percentile)
            rows.append({
                "file": p.name,
                "ELA_std": float(vals[0]),
                "high_intensity_ratio": float(vals[1]),
                "ELA_mean": float(vals[2]),
            })
            print(f"[OK] {i}/{total} - {p.name}")  # 진행률 출력
        except Exception as e:
            print(f"[ERROR] {i}/{total} - {p.name}: {e}")  # 에러 발생해도 다음 파일 계속 진행

    # 3. 결과 저장
    if rows:
        df = pd.DataFrame(rows, columns=["file", "ELA_std", "high_intensity_ratio", "ELA_mean"])
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n[DONE] Saved CSV → {out_csv} ({len(df)} files processed)")
    else:
        print("[WARN] No valid images found. Please check the folder.")

# ===========================================
# 5) Entry Point
# ===========================================
if __name__ == "__main__":
    # 파일을 직접 실행했을 때만 batch 모드로 작동
    # (다른 코드에서 import 하면 main()은 실행되지 않음)
    main()
