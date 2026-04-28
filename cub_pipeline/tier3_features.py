import os
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def extract_hsv_histogram_48(img: Image.Image, bins: int = 16) -> np.ndarray:
    hsv = np.array(img.convert("HSV"))
    channels = []
    for idx in range(3):
        hist, _ = np.histogram(hsv[:, :, idx], bins=bins, range=(0, 256))
        channels.append(hist.astype(np.float32))
    feat = np.concatenate(channels, axis=0)
    return feat / (float(feat.sum()) + 1e-8)


def extract_lbp_texture_64(img: Image.Image, bins: int = 64) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for bit_idx, (dy, dx) in enumerate(shifts):
        shifted = np.roll(gray, shift=(dy, dx), axis=(0, 1))
        lbp |= ((shifted >= gray).astype(np.uint8) << bit_idx)
    hist, _ = np.histogram(lbp, bins=bins, range=(0, 256))
    feat = hist.astype(np.float32)
    return feat / (float(feat.sum()) + 1e-8)


def extract_hog_shape(img: Image.Image, grid_size: int = 4, n_bins: int = 9) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0
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
