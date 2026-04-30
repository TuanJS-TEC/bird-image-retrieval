import os
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .gpu_backend import load_cupy

_CP = load_cupy()


def extract_hsv_histogram_48(img: Image.Image, bins: int = 16) -> np.ndarray:
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    if _CP is None:
        channels = []
        for idx in range(3):
            hist, _ = np.histogram(hsv[:, :, idx], bins=bins, range=(0, 256))
            channels.append(hist.astype(np.float32))
        feat = np.concatenate(channels, axis=0)
    else:
        hsv_gpu = _CP.asarray(hsv)
        channels_gpu = []
        for idx in range(3):
            hist_gpu, _ = _CP.histogram(hsv_gpu[:, :, idx], bins=bins, range=(0, 256))
            channels_gpu.append(hist_gpu.astype(_CP.float32))
        feat = _CP.asnumpy(_CP.concatenate(channels_gpu, axis=0)).astype(np.float32, copy=False)
    return feat / (float(feat.sum()) + 1e-8)


def extract_lbp_texture_64(img: Image.Image, bins: int = 64) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    if _CP is None:
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for bit_idx, (dy, dx) in enumerate(shifts):
            shifted = np.roll(gray, shift=(dy, dx), axis=(0, 1))
            lbp |= ((shifted >= gray).astype(np.uint8) << bit_idx)
        hist, _ = np.histogram(lbp, bins=bins, range=(0, 256))
        feat = hist.astype(np.float32)
    else:
        gray_gpu = _CP.asarray(gray)
        lbp_gpu = _CP.zeros_like(gray_gpu, dtype=_CP.uint8)
        for bit_idx, (dy, dx) in enumerate(shifts):
            shifted_gpu = _CP.roll(gray_gpu, shift=(dy, dx), axis=(0, 1))
            lbp_gpu |= ((_CP.asarray(shifted_gpu >= gray_gpu, dtype=_CP.uint8)) << _CP.uint8(bit_idx))
        hist_gpu, _ = _CP.histogram(lbp_gpu, bins=bins, range=(0, 256))
        feat = _CP.asnumpy(hist_gpu).astype(np.float32, copy=False)
    return feat / (float(feat.sum()) + 1e-8)


def extract_hog_shape(img: Image.Image, grid_size: int = 4, n_bins: int = 9) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    if _CP is None:
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.degrees(np.arctan2(gy, gx)) % 180.0
    else:
        gray_gpu = _CP.asarray(gray, dtype=_CP.float32)
        gx = _CP.zeros_like(gray_gpu, dtype=_CP.float32)
        gy = _CP.zeros_like(gray_gpu, dtype=_CP.float32)
        gx[:, 1:-1] = gray_gpu[:, 2:] - gray_gpu[:, :-2]
        gy[1:-1, :] = gray_gpu[2:, :] - gray_gpu[:-2, :]
        magnitude = _CP.asnumpy(_CP.sqrt(gx**2 + gy**2))
        orientation = _CP.asnumpy(_CP.degrees(_CP.arctan2(gy, gx)) % 180.0)
    h, w = gray.shape
    cell_h = max(h // grid_size, 1)
    cell_w = max(w // grid_size, 1)
    descriptor = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * cell_h, min((r + 1) * cell_h, h)
            x0, x1 = c * cell_w, min((c + 1) * cell_w, w)
            cell_mag = magnitude[y0:y1, x0:x1].reshape(-1)
            cell_ori = orientation[y0:y1, x0:x1].reshape(-1)
            hist, _ = np.histogram(cell_ori, bins=n_bins, range=(0, 180), weights=cell_mag)
            descriptor.append(hist.astype(np.float32))
    feat = np.concatenate(descriptor, axis=0)
    return feat / (np.linalg.norm(feat) + 1e-8)


def extract_handcrafted_features(metadata_df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    feature_rows: list[dict[str, Any]] = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 3 - Handcrafted"):
        filename = str(row["filename"])
        img_id = int(row["img_id"])
        img_path = os.path.join(images_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            vector = np.concatenate([extract_hsv_histogram_48(img), extract_lbp_texture_64(img), extract_hog_shape(img)], axis=0)
            record: dict[str, Any] = {"img_id": img_id, "filename": filename}
            for idx, value in enumerate(vector):
                record[f"hc_{idx:03d}"] = float(value)
            feature_rows.append(record)
        except Exception as ex:
            print(f"  [WARN] Bo qua handcrafted {filename}: {ex}")
    return pd.DataFrame(feature_rows)
