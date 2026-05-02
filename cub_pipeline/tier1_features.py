import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from .common import parallel_tier_extraction_workers
from .config import ENABLE_GREEN_BINARY, GREEN_RATIO_BINARY_THRESHOLD
from .tier3_features import extract_lbp_texture_64

def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "attr"

def _similarity_attribute_spec() -> dict[str, dict[str, Any]]:
    return {
        "global_color": {
            "keywords": ["bill_color", "belly_color", "breast_color", "back_color"],
            "display_attributes": ["bill_color", "belly_color", "breast_color", "back_color"],
            "reason": "Mau la dac trung noi bat nhat giua cac loai chim.",
        },
        "shape": {
            "keywords": ["bill_shape", "wing_shape", "tail_shape"],
            "display_attributes": ["bill_shape", "wing_shape", "tail_shape"],
            "reason": "Hinh dang ho tro phan biet ho chim (se, dai bang, vi).",
        },
        "pattern": {
            "keywords": ["breast_pattern", "back_pattern"],
            "display_attributes": ["breast_pattern", "back_pattern"],
            "reason": "Hoa van tach cac loai gan nhau trong cung mot ho.",
        },
        "size": {
            "keywords": ["size"],
            "display_attributes": ["size (small/medium/large)"],
            "reason": "Kich thuoc tuong dong thuong di kem tuong dong loai.",
        },
    }


def _safe_entropy(hist: np.ndarray) -> float:
    prob = hist.astype(np.float64)
    prob = prob / (prob.sum() + 1e-12)
    return float(-(prob * np.log2(prob + 1e-12)).sum())


def _discretize_series(values: pd.Series, bins: int = 8) -> np.ndarray:
    arr = values.astype(float).to_numpy()
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=np.int32)
    v = arr[finite_mask]
    unique = np.unique(v)
    if len(unique) <= 1:
        out = np.zeros_like(arr, dtype=np.int32)
        return out
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(v, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        out = np.zeros_like(arr, dtype=np.int32)
        out[finite_mask] = (v > float(np.median(v))).astype(np.int32)
        return out
    out = np.zeros_like(arr, dtype=np.int32)
    out[finite_mask] = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    return out


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    if x.size == 0 or y.size == 0:
        return 0.0
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return 0.0
    # Ensure writable array: pandas may return a read-only view on some builds.
    joint = np.array(pd.crosstab(x, y).to_numpy(dtype=np.float64), copy=True)
    joint /= float(joint.sum() + 1e-12)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / (expected[mask] + 1e-12))).sum())


def _fisher_score(feature: np.ndarray, labels: np.ndarray) -> float:
    x = feature.astype(np.float64)
    y = labels.astype(np.int32)
    if len(np.unique(y)) <= 1:
        return 0.0
    mean_all = float(np.mean(x))
    num = 0.0
    den = 0.0
    for cls in np.unique(y):
        cls_vals = x[y == cls]
        if cls_vals.size == 0:
            continue
        cls_mean = float(np.mean(cls_vals))
        num += cls_vals.size * ((cls_mean - mean_all) ** 2)
        den += float(np.sum((cls_vals - cls_mean) ** 2))
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def _binary_mask_hu_moments(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.float64)
    m00 = binary.sum() + 1e-12
    ys, xs = np.indices(binary.shape)
    x_bar = float((xs * binary).sum() / m00)
    y_bar = float((ys * binary).sum() / m00)

    x = xs - x_bar
    y = ys - y_bar

    def mu(p: int, q: int) -> float:
        return float(((x**p) * (y**q) * binary).sum())

    def eta(p: int, q: int) -> float:
        gamma = 1.0 + (p + q) / 2.0
        return mu(p, q) / (m00**gamma + 1e-12)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
    hu1 = n20 + n02
    hu2 = (n20 - n02) ** 2 + 4.0 * (n11**2)
    hu3 = (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2
    return np.array([hu1, hu2, hu3], dtype=np.float64)


def _refine_fg_mask(raw_mask: np.ndarray) -> np.ndarray:
    mask = raw_mask.astype(bool)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    return mask


def extract_similarity_tier1_vector(img: Image.Image) -> np.ndarray:
    """
    Query-time Tier-1 vector aligned with training-time feature logic.
    Order matches similarity_cols in cub_pipeline/features.py.
    """
    rgb = img.convert("RGB")
    hsv = np.array(rgb.convert("HSV"), dtype=np.float32)
    h_deg = (hsv[:, :, 0] * (360.0 / 255.0)) % 360.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    chromatic_mask = (s >= 0.20) & (v >= 0.15)

    color_red = float((((h_deg >= 345.0) | (h_deg < 15.0)) & chromatic_mask).mean())
    color_yellow = float(((h_deg >= 45.0) & (h_deg < 75.0) & chromatic_mask).mean())
    color_blue = float(((h_deg >= 195.0) & (h_deg < 255.0) & chromatic_mask).mean())

    gray = np.array(rgb.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad_x_energy = float(np.mean(np.abs(gx)))
    grad_y_energy = float(np.mean(np.abs(gy)))
    grad_anisotropy = float(abs(grad_x_energy - grad_y_energy) / (grad_x_energy + grad_y_energy + 1e-8))

    gray_hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=32, range=(0, 256))
    texture_entropy = _safe_entropy(gray_hist)

    fg_mask = _refine_fg_mask(chromatic_mask | (v < 0.20))
    ys, xs = np.where(fg_mask)
    fg_area_ratio = float(fg_mask.mean())
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        fg_aspect_ratio = float(max(1, x1 - x0 + 1) / max(1, y1 - y0 + 1))
    else:
        fg_aspect_ratio = 1.0

    left = gray[:, : gray.shape[1] // 2]
    right = np.fliplr(gray[:, gray.shape[1] - left.shape[1] :])
    symmetry_lr = float(1.0 - np.mean(np.abs(left - right)))
    hu = _binary_mask_hu_moments(fg_mask.astype(np.uint8))

    return np.array(
        [
            color_red,
            color_yellow,
            color_blue,
            float(s.mean()),
            texture_entropy,
            fg_area_ratio,
            fg_aspect_ratio,
            symmetry_lr,
            grad_anisotropy,
            float(hu[0]),
            float(hu[1]),
        ],
        dtype=np.float32,
    )


def _custom_attr_record_from_image(
    img: Image.Image, img_id: int, filename: str, bbox_ratio: float
) -> dict[str, Any]:
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    h_deg = (hsv[:, :, 0] * (360.0 / 255.0)) % 360.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    chromatic_mask = (s >= 0.20) & (v >= 0.15)

    color_masks = {
        "red": (((h_deg >= 345.0) | (h_deg < 15.0)) & chromatic_mask),
        "orange": ((h_deg >= 15.0) & (h_deg < 45.0) & chromatic_mask),
        "yellow": ((h_deg >= 45.0) & (h_deg < 75.0) & chromatic_mask),
        "green": ((h_deg >= 75.0) & (h_deg < 165.0) & chromatic_mask),
        "cyan": ((h_deg >= 165.0) & (h_deg < 195.0) & chromatic_mask),
        "blue": ((h_deg >= 195.0) & (h_deg < 255.0) & chromatic_mask),
        "magenta": ((h_deg >= 255.0) & (h_deg < 345.0) & chromatic_mask),
        "white": ((s < 0.20) & (v >= 0.80)),
        "gray": ((s < 0.20) & (v >= 0.20) & (v < 0.80)),
        "black": (v < 0.20),
    }

    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.sqrt(gx**2 + gy**2)
    edge_thr = float(np.quantile(grad, 0.80))
    edge_density = float((grad > edge_thr).mean())
    grad_x_energy = float(np.mean(np.abs(gx)))
    grad_y_energy = float(np.mean(np.abs(gy)))
    grad_anisotropy = float(abs(grad_x_energy - grad_y_energy) / (grad_x_energy + grad_y_energy + 1e-8))

    gray_hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=32, range=(0, 256))
    texture_entropy = _safe_entropy(gray_hist)

    lbp_hist = extract_lbp_texture_64(img)
    lbp_energy = float(np.sum(lbp_hist**2))
    lbp_entropy = _safe_entropy(lbp_hist)

    fg_mask = _refine_fg_mask(chromatic_mask | (v < 0.20))
    ys, xs = np.where(fg_mask)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        fg_w = max(1, x1 - x0 + 1)
        fg_h = max(1, y1 - y0 + 1)
        fg_area_ratio = float(fg_mask.mean())
        fg_aspect_ratio = float(fg_w / fg_h)
        centroid_x = float(xs.mean() / max(gray.shape[1] - 1, 1))
        centroid_y = float(ys.mean() / max(gray.shape[0] - 1, 1))
    else:
        fg_area_ratio = 0.0
        fg_aspect_ratio = 1.0
        centroid_x = 0.5
        centroid_y = 0.5

    left = gray[:, : gray.shape[1] // 2]
    right = np.fliplr(gray[:, gray.shape[1] - left.shape[1] :])
    symmetry_lr = float(1.0 - np.mean(np.abs(left - right)))
    hu = _binary_mask_hu_moments(fg_mask.astype(np.uint8))

    upper_mass = float(fg_mask[: fg_mask.shape[0] // 2, :].mean())
    lower_mass = float(fg_mask[fg_mask.shape[0] // 2 :, :].mean())
    upper_lower_mass_ratio = float((upper_mass + 1e-8) / (lower_mass + 1e-8))

    record: dict[str, Any] = {
        "img_id": img_id,
        "filename": filename,
        "cattr_sat_mean": float(s.mean()),
        "cattr_val_mean": float(v.mean()),
        "cattr_edge_density": edge_density,
        "cattr_texture_entropy": texture_entropy,
        "cattr_lbp_energy": lbp_energy,
        "cattr_lbp_entropy": lbp_entropy,
        "cattr_fg_area_ratio": fg_area_ratio,
        "cattr_fg_aspect_ratio": fg_aspect_ratio,
        "cattr_centroid_x": centroid_x,
        "cattr_centroid_y": centroid_y,
        "cattr_symmetry_lr": symmetry_lr,
        "cattr_grad_anisotropy": grad_anisotropy,
        "cattr_upper_lower_mass_ratio": upper_lower_mass_ratio,
        "cattr_hu1": float(hu[0]),
        "cattr_hu2": float(hu[1]),
        "cattr_hu3": float(hu[2]),
        "cattr_bbox_ratio": float(bbox_ratio),
    }
    for color_name, mask in color_masks.items():
        record[f"cattr_color_{color_name}_ratio"] = float(mask.mean())
    if ENABLE_GREEN_BINARY and "cattr_color_green_ratio" in record:
        record["cattr_color_green_present"] = float(
            record["cattr_color_green_ratio"] >= GREEN_RATIO_BINARY_THRESHOLD
        )
    return record


def _tier1_custom_worker(task: tuple[int, str, str, float]) -> dict[str, Any] | None:
    img_id, filename, images_dir, bbox_ratio = task
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    path = os.path.join(images_dir, filename)
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    try:
        return _custom_attr_record_from_image(img, img_id, filename, bbox_ratio)
    except Exception as ex:
        print(f"  [WARN] Bo qua custom attrs {filename}: {ex}")
        return None


def extract_custom_visual_attributes(
    metadata_df: pd.DataFrame, images_dir: str, max_workers: int | None = None
) -> pd.DataFrame:
    tasks: list[tuple[int, str, str, float]] = []
    for _, row in metadata_df.iterrows():
        filename = str(row["filename"])
        img_id = int(row["img_id"])
        try:
            br = float(row["bbox_ratio"]) if "bbox_ratio" in row.index and pd.notna(row["bbox_ratio"]) else 0.0
        except Exception:
            br = 0.0
        tasks.append((img_id, filename, images_dir, br))

    workers = int(max_workers) if max_workers is not None else parallel_tier_extraction_workers()
    if workers <= 1 or len(tasks) < 6:
        rows: list[dict[str, Any]] = []
        for img_id, filename, idir, br in tqdm(tasks, total=len(tasks), desc="  Tang 1 - Custom attrs"):
            r = _tier1_custom_worker((img_id, filename, idir, br))
            if r is not None:
                rows.append(r)
        out = pd.DataFrame(rows)
        return out.sort_values("img_id").reset_index(drop=True) if not out.empty else out

    n_workers = max(1, min(workers, len(tasks)))
    mp_ctx = mp.get_context("spawn")
    rows_parallel: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as ex:
        futures = [ex.submit(_tier1_custom_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Tang 1 - Custom attrs"):
            r = fut.result()
            if r is not None:
                rows_parallel.append(r)
    out = pd.DataFrame(rows_parallel)
    return out.sort_values("img_id").reset_index(drop=True) if not out.empty else out


def build_attribute_information_value_table(custom_attrs_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = custom_attrs_df.merge(metadata_df[["img_id", "class_id"]], on="img_id", how="left")
    labels = merged["class_id"].astype(int).to_numpy()
    attr_cols = [c for c in merged.columns if c.startswith("cattr_")]
    rows: list[dict[str, Any]] = []
    for col in attr_cols:
        values = merged[col].astype(float).to_numpy()
        fisher = _fisher_score(values, labels)
        discretized = _discretize_series(merged[col], bins=8)
        mi = _mutual_info_discrete(discretized, labels)
        rows.append(
            {
                "attribute": col,
                "fisher_score": float(fisher),
                "mutual_information_bits": float(mi),
            }
        )
    info_df = pd.DataFrame(rows)
    if info_df.empty:
        return info_df
    for col in ["fisher_score", "mutual_information_bits"]:
        v = info_df[col].to_numpy(dtype=np.float64)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi - lo < 1e-12:
            info_df[f"{col}_norm"] = 0.0
        else:
            info_df[f"{col}_norm"] = (v - lo) / (hi - lo)
    info_df["combined_information_value"] = 0.55 * info_df["fisher_score_norm"] + 0.45 * info_df["mutual_information_bits_norm"]
    info_df = info_df.sort_values("combined_information_value", ascending=False).reset_index(drop=True)
    return info_df

def build_cub_312_binary_attributes(
    metadata_df: pd.DataFrame,
    attr_labels_df: pd.DataFrame,
    attr_names_df: pd.DataFrame,
    certainty_threshold: int = 3,
) -> tuple[pd.DataFrame, dict[int, str]]:
    filtered_ids = metadata_df["img_id"].astype(int).tolist()
    all_attr_ids = attr_names_df["attr_id"].astype(int).tolist()
    attr_id_to_col = {}
    for attr_id, attr_name in zip(attr_names_df["attr_id"].tolist(), attr_names_df["attr_name"].tolist()):
        aid = int(attr_id)
        attr_id_to_col[aid] = f"attr_{aid:03d}__{_slugify(attr_name)}"
    subset = attr_labels_df[attr_labels_df["img_id"].isin(filtered_ids)].copy()
    subset = subset[subset["certainty_id"] >= certainty_threshold]
    subset["value"] = (subset["is_present"] == 1).astype(np.uint8)
    pivot = subset.pivot_table(index="img_id", columns="attr_id", values="value", aggfunc="max", fill_value=0)
    pivot = pivot.reindex(columns=all_attr_ids, fill_value=0)
    pivot = pivot.reindex(filtered_ids, fill_value=0)
    pivot = pivot.astype(np.uint8)
    pivot.columns = [attr_id_to_col[int(attr_id)] for attr_id in pivot.columns]
    pivot.index.name = "img_id"
    return pivot, attr_id_to_col


def select_similarity_attribute_ids(attr_names_df: pd.DataFrame) -> dict[str, list[int]]:
    groups = _similarity_attribute_spec()
    lower_names = attr_names_df.copy()
    lower_names["attr_name_norm"] = lower_names["attr_name"].apply(lambda v: str(v).lower())
    selected: dict[str, list[int]] = {}
    for group_name, group_info in groups.items():
        keywords = group_info["keywords"]
        regex = "|".join(re.escape(k) for k in keywords)
        matches = lower_names[lower_names["attr_name_norm"].apply(lambda v: bool(re.search(regex, str(v))))]
        selected[group_name] = sorted(matches["attr_id"].astype(int).unique().tolist())
    return selected
