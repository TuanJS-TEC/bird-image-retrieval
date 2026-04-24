import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from tqdm import tqdm

from .algorithmic_tier2 import extract_algorithmic_embedding, fit_algorithmic_tier2, load_algorithmic_artifacts
from .config import (
    ENABLE_GREEN_BINARY,
    ENABLE_HU_BOOST,
    ENABLE_TIER_NORMALIZATION,
    GREEN_RATIO_BINARY_THRESHOLD,
    HU_MOMENT_WEIGHT,
    RETRIEVAL_EXCLUDED_TIER1_ATTRS,
    TIER1_WEIGHT,
    TIER2_WEIGHT,
    TIER2_PCA_DIM,
    TIER3_WEIGHT,
)


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


def _zscore_per_column(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    mat = matrix.astype(np.float32, copy=True)
    mean = np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((mat - mean) / std).astype(np.float32)


def _apply_tier2_pca(matrix: np.ndarray, target_dim: int | None) -> tuple[np.ndarray, dict[str, float]]:
    if matrix.size == 0 or target_dim is None or int(target_dim) <= 0:
        return matrix.astype(np.float32, copy=False), {"enabled": 0.0, "input_dim": float(matrix.shape[1] if matrix.ndim == 2 else 0), "output_dim": float(matrix.shape[1] if matrix.ndim == 2 else 0), "explained_variance_ratio": 1.0}
    n, d = matrix.shape
    n_comp = max(1, min(int(target_dim), n, d))
    if n_comp >= d:
        return matrix.astype(np.float32, copy=False), {"enabled": 0.0, "input_dim": float(d), "output_dim": float(d), "explained_variance_ratio": 1.0}
    pca = PCA(n_components=n_comp, random_state=42)
    transformed = pca.fit_transform(matrix.astype(np.float32)).astype(np.float32)
    meta = {
        "enabled": 1.0,
        "input_dim": float(d),
        "output_dim": float(n_comp),
        "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
    }
    return transformed, meta


def _apply_hu_boost(custom_matrix: np.ndarray, attr_cols: list[str], weight: float) -> np.ndarray:
    if custom_matrix.size == 0 or weight <= 1.0:
        return custom_matrix.astype(np.float32, copy=False)
    out = custom_matrix.astype(np.float32, copy=True)
    hu_cols = {"cattr_hu1", "cattr_hu2", "cattr_hu3"}
    hu_indices = [idx for idx, col in enumerate(attr_cols) if col in hu_cols]
    if hu_indices:
        out[:, hu_indices] *= float(weight)
    return out


def extract_custom_visual_attributes(metadata_df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 1 - Custom attrs"):
        filename = str(row["filename"])
        img_id = int(row["img_id"])
        img_path = os.path.join(images_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            hsv = np.array(img.convert("HSV"), dtype=np.float32)
            # PIL encodes Hue in [0, 255]; convert to degrees for robust color ranges.
            h_deg = (hsv[:, :, 0] * (360.0 / 255.0)) % 360.0
            s = hsv[:, :, 1] / 255.0
            v = hsv[:, :, 2] / 255.0
            chromatic_mask = (s >= 0.20) & (v >= 0.15)

            # Continuous hue partitions (no gaps) on 0..360 circle.
            color_masks = {
                "red": (((h_deg >= 345.0) | (h_deg < 15.0)) & chromatic_mask),
                "orange": ((h_deg >= 15.0) & (h_deg < 45.0) & chromatic_mask),
                "yellow": ((h_deg >= 45.0) & (h_deg < 75.0) & chromatic_mask),
                "green": ((h_deg >= 75.0) & (h_deg < 165.0) & chromatic_mask),
                "cyan": ((h_deg >= 165.0) & (h_deg < 195.0) & chromatic_mask),
                "blue": ((h_deg >= 195.0) & (h_deg < 255.0) & chromatic_mask),
                "magenta": ((h_deg >= 255.0) & (h_deg < 345.0) & chromatic_mask),
                # Neutral colors are defined by low saturation and brightness level.
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
                "cattr_bbox_ratio": float(row.get("bbox_ratio", 0.0)),
            }
            for color_name, mask in color_masks.items():
                record[f"cattr_color_{color_name}_ratio"] = float(mask.mean())
            if ENABLE_GREEN_BINARY and "cattr_color_green_ratio" in record:
                record["cattr_color_green_present"] = float(
                    record["cattr_color_green_ratio"] >= GREEN_RATIO_BINARY_THRESHOLD
                )
            rows.append(record)
        except Exception as ex:
            print(f"  [WARN] Bo qua custom attrs {filename}: {ex}")
    return pd.DataFrame(rows)


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


def write_attribute_diagnostic_reports(
    reports_dir: str,
    custom_attrs_df: pd.DataFrame,
    info_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cnn_aligned: np.ndarray,
) -> None:
    merged = custom_attrs_df.merge(metadata_df[["img_id", "class_id", "class_name"]], on="img_id", how="left")

    if "cattr_color_green_ratio" in merged.columns:
        green = merged["cattr_color_green_ratio"].astype(float)
        green_stats = {
            "count": int(green.shape[0]),
            "mean": float(green.mean()),
            "std": float(green.std()),
            "q10": float(green.quantile(0.10)),
            "q50": float(green.quantile(0.50)),
            "q90": float(green.quantile(0.90)),
            "pct_below_001": float((green < 0.01).mean()),
            "pct_below_005": float((green < 0.05).mean()),
        }
        if "cattr_color_green_present" in merged.columns:
            green_stats["pct_green_present"] = float(merged["cattr_color_green_present"].astype(float).mean())
        with open(os.path.join(reports_dir, "green_ratio_distribution.json"), "w", encoding="utf-8") as f:
            json.dump(green_stats, f, indent=2, ensure_ascii=False)

    if "cattr_centroid_y" in merged.columns:
        cty = (
            merged.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                centroid_y_mean=("cattr_centroid_y", "mean"),
                centroid_y_std=("cattr_centroid_y", "std"),
                sample_count=("cattr_centroid_y", "count"),
            )
            .reset_index(drop=True)
        )
        cty = cty.sort_values("centroid_y_mean", ascending=False)
        cty.to_csv(os.path.join(reports_dir, "centroid_y_by_class.csv"), index=False, encoding="utf-8")

    attr_cols = [c for c in custom_attrs_df.columns if c.startswith("cattr_")]
    if attr_cols and cnn_aligned.size:
        x = cnn_aligned.astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        n_comp = min(16, x.shape[0], x.shape[1])
        if n_comp > 0:
            u, s, _ = np.linalg.svd(x, full_matrices=False)
            pcs = u[:, :n_comp] * s[:n_comp]
            rows: list[dict[str, Any]] = []
            for attr in attr_cols:
                av = custom_attrs_df[attr].astype(float).to_numpy()
                av = av - av.mean()
                denom_a = float(np.sqrt((av**2).sum()) + 1e-12)
                corr_vals = []
                for i in range(pcs.shape[1]):
                    pc = pcs[:, i]
                    pc_centered = pc - pc.mean()
                    denom_pc = float(np.sqrt((pc_centered**2).sum()) + 1e-12)
                    corr = float((av * pc_centered).sum() / (denom_a * denom_pc))
                    corr_vals.append(abs(corr))
                rows.append(
                    {
                        "attribute": attr,
                        "max_abs_corr_top16_pca": float(np.max(corr_vals)),
                        "mean_abs_corr_top16_pca": float(np.mean(corr_vals)),
                    }
                )
            corr_df = pd.DataFrame(rows).sort_values("max_abs_corr_top16_pca", ascending=False)
            corr_df.to_csv(
                os.path.join(reports_dir, "custom_vs_resnet_correlation.csv"),
                index=False,
                encoding="utf-8",
            )

    if not info_df.empty:
        info_df.to_csv(os.path.join(reports_dir, "attribute_info_scatter_data.csv"), index=False, encoding="utf-8")


def write_requirement2_report(
    reports_dir: str,
    similarity_cols: list[str],
    difference_cols: list[str],
    info_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    reason_map = {
        "cattr_color_red_ratio": ("Ti le sac do", "Mau long/ma chim nhieu loai co tong mau dac trung, on dinh theo dieu kien chup."),
        "cattr_color_yellow_ratio": ("Ti le sac vang", "Giup gom nhom cac loai co bung/nguc vang, co y nghia cho truy hoi tuong dong."),
        "cattr_color_green_ratio": ("Ti le sac xanh la", "Tach nhom chim co tong the xanh, bo tro cho nhom se/rung."),
        "cattr_color_green_present": ("Co sac xanh la (nhi phan)", "Giam nhieu cho dac trung xanh thua thot bang cach dung nhan co/khong."),
        "cattr_color_cyan_ratio": ("Ti le sac cyan", "Nham bat cac vung long xanh ngoc, thuong xuat hien cuc bo."),
        "cattr_color_blue_ratio": ("Ti le sac xanh duong", "Mau noi bat de phan biet cac loai co canh/lung xanh."),
        "cattr_color_magenta_ratio": ("Ti le sac tim-hong", "Bo sung thong tin mau hiem de tao khoang cach giua cac loai gan nhau."),
        "cattr_sat_mean": ("Do bao hoa trung binh", "Do ruc long chim, on dinh va hieu qua cho tim anh cung tone."),
        "cattr_val_mean": ("Do sang trung binh", "Lam ro dieu kien do sang tong the, ho tro chuan hoa cam nhan mau."),
        "cattr_edge_density": ("Mat do bien", "Do phuc tap hinh dang/duong vien, tach chim tron min va chim hoa tiet manh."),
        "cattr_texture_entropy": ("Entropy ket cau xam", "Do da dang texture toan cuc, huu ich de phan biet bo long tron vs van."),
        "cattr_lbp_energy": ("Nang luong LBP", "Do do dong nhat micro-pattern, cao khi bo long it bien thien."),
        "cattr_lbp_entropy": ("Entropy LBP", "Do phong phu micro-texture, cao khi long co nhieu chi tiet khac nhau."),
        "cattr_fg_area_ratio": ("Ti le vung tien canh", "Phan anh kich thuoc doi tuong trong khung, ho tro nhom hinh thai."),
        "cattr_fg_aspect_ratio": ("Ty le ngang/doc tien canh", "Mo ta dang than-canh, dong gop manh cho phan biet hinh dang."),
        "cattr_centroid_x": ("Tam khoi theo truc X", "Do vi tri doi tuong theo ngang, bo tro xu ly bo cuc tuong dong."),
        "cattr_centroid_y": ("Tam khoi theo truc Y", "Do vi tri doi tuong theo doc, phan anh tu the/chieu chup."),
        "cattr_symmetry_lr": ("Doi xung trai-phai", "Nhiet ke hinh hoc: loai canh xoe/canh gap se co profile doi xung khac nhau."),
        "cattr_grad_anisotropy": ("Bat huong gradient", "Do huong cau truc noi bat theo ngang/doc, bo sung thong tin hinh dang tong quan."),
        "cattr_upper_lower_mass_ratio": ("Ty le khoi luong nua tren/duoi", "Proxy hinh thai than chim, giam phu thuoc mau sac."),
        "cattr_hu1": ("Hu moment 1", "Mo ta hinh dang tong the theo moment bat bien voi scale/rotation."),
        "cattr_hu2": ("Hu moment 2", "Bat bien hinh dang cap cao hon, bo tro cho phan biet silhouette."),
        "cattr_hu3": ("Hu moment 3", "Bat bien hinh dang phi tuyen, huu ich khi tu the thay doi nhe."),
        "cattr_bbox_ratio": ("Ty le bbox goc", "Thong tin hinh hoc goc truoc resize, bo tro kha nang tach dang than."),
    }
    info_map = info_df.set_index("attribute").to_dict(orient="index") if not info_df.empty else {}
    lines: list[str] = []
    lines.append("# Yeu cau 2 - Bo thuoc tinh tu xay dung cho nhan dien chim")
    lines.append("")
    lines.append("He thong KHONG su dung annotation attributes co san cua CUB-200-2011.")
    lines.append("Thay vao do, bo thuoc tinh duoc trich xuat truc tiep tu anh da xu ly (`dataset_processed/images`).")
    lines.append("")
    lines.append("## 1) Nguyen tac thiet ke")
    lines.append("")
    lines.append("- Thuoc tinh tuong dong: uu tien dac trung on dinh toan cuc (tone mau, texture tong quan, hinh hoc tong the).")
    lines.append("- Thuoc tinh khac biet: uu tien dac trung co gia tri phan lop cao theo thong ke tren tap du lieu.")
    lines.append("- Gia tri thong tin duoc dinh luong bang 2 chi so: Fisher score va Mutual Information.")
    lines.append("- Cong thuc Combined: `0.55 * Fisher_norm + 0.45 * MI_norm`.")
    lines.append("")
    lines.append("## 2) Nhom thuoc tinh cho tim kiem tuong dong")
    lines.append("")
    lines.append("| Attribute | Y nghia | Ly do chon |")
    lines.append("| --- | --- | --- |")
    for attr in similarity_cols:
        vn_name, reason = reason_map.get(attr, (attr, "Thuoc tinh trich xuat truc tiep tu anh va co tinh on dinh."))
        lines.append(f"| `{attr}` | {vn_name} | {reason} |")
    lines.append("")
    lines.append("## 3) Nhom thuoc tinh cho phan biet/khac nhau")
    lines.append("")
    lines.append("Nhom nay gom cac thuoc tinh co `combined_information_value` cao nhat.")
    lines.append("")
    lines.append("| Attribute | Fisher | MI (bit) | Combined | Dien giai gia tri thong tin |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for attr in difference_cols:
        metrics = info_map.get(attr, {})
        fisher = float(metrics.get("fisher_score", 0.0))
        mi = float(metrics.get("mutual_information_bits", 0.0))
        cv = float(metrics.get("combined_information_value", 0.0))
        _, reason = reason_map.get(attr, (attr, "Gia tri thong tin cao trong viec tach cac lop chim."))
        lines.append(f"| `{attr}` | {fisher:.4f} | {mi:.4f} | {cv:.4f} | {reason} |")
    lines.append("")
    lines.append("## 4) Tong hop kich thuoc vector")
    lines.append("")
    lines.append(f"- Tang 1 (Custom attributes): {manifest['tier1_custom_dim']}")
    lines.append(f"- Tang 1 (Similarity subset): {manifest['tier1_similarity_dim']}")
    lines.append(f"- Tang 1 (Difference subset): {manifest['tier1_difference_dim']}")
    lines.append(f"- Tang 2 (Algorithmic fusion): {manifest['tier2_algorithmic_dim']}")
    lines.append(f"- Tang 3 (Handcrafted): {manifest['tier3_handcrafted_dim']}")
    lines.append(f"- Final vector: {manifest['final_dim']}")
    lines.append("")
    lines.append("Trong cau hinh hien tai, Tang 3 duoc cau thanh boi:")
    lines.append("- HSV histogram: 48 chieu (`3 kenh x 16 bins`)")
    lines.append("- LBP texture: 64 chieu")
    lines.append("- HOG shape: 144 chieu (`grid_size=4`, `n_bins=9`)")
    lines.append("- Tong: `48 + 64 + 144 = 256` chieu")
    lines.append("")
    lines.append("## 5) Phan tich bias va trung lap thong tin")
    lines.append("")
    lines.append("- `green_ratio_distribution.json`: thong ke phan bo `cattr_color_green_ratio` tren toan dataset.")
    lines.append("- `centroid_y_by_class.csv`: trung binh/dao dong `cattr_centroid_y` theo loai de phat hien spurious correlation.")
    lines.append("- `custom_vs_resnet_correlation.csv`: tuong quan custom attributes voi top-16 PCA components cua Tier-2 vector.")
    lines.append("- `attribute_info_scatter_data.csv`: du lieu ve Fisher/MI/Combined de ve scatter plot trong bao cao.")
    lines.append("")
    lines.append("## 6) Tep ket qua")
    lines.append("")
    lines.append("- `tier1_custom_attributes.csv`: toan bo bo thuoc tinh tu xay dung")
    lines.append("- `tier1_similarity_attributes.csv`: nhom thuoc tinh uu tien cho tim anh tuong dong")
    lines.append("- `tier1_difference_attributes.csv`: nhom thuoc tinh uu tien cho phan biet")
    lines.append("- `tier1_attribute_information_value.csv`: bang chi so gia tri thong tin")
    lines.append("- `recognition_features_all.npz`: vector hop nhat cho retrieval/phan loai")
    lines.append("")
    report_path = os.path.join(reports_dir, "requirement2_attributes.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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


def extract_efficientnetv2_embeddings(
    metadata_df: pd.DataFrame,
    images_dir: str,
    require_torch: bool = True,
    cub_root: str | None = None,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    # Backward-compatible function name: now computes algorithm-only Tier-2 embedding.
    output_dir = os.path.dirname(images_dir.rstrip("\\/"))
    features_dir = os.path.join(output_dir, "features")
    artifacts_path = os.path.join(features_dir, "tier2_algorithmic_artifacts.pkl")
    if not os.path.exists(artifacts_path):
        if not cub_root:
            raise RuntimeError("Thieu cub_root de fit AG-SFP anatomy priors va cac model algorithmic.")
        print("  [INFO] Dang fit bo model algorithmic Tier 2 (AG-SFP + ACV + HFVE + PPD)...")
        fit_algorithmic_tier2(
            metadata_df=metadata_df,
            images_dir=images_dir,
            cub_root=cub_root,
            features_dir=features_dir,
        )
    artifacts = load_algorithmic_artifacts(features_dir)
    vectors, img_ids, filenames, anatomy_blocks = [], [], [], []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 2 - Algorithmic 512D"):
        filename = str(row["filename"])
        img_id = int(row["img_id"])
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue
        try:
            vec, anatomy = extract_algorithmic_embedding(img_path, artifacts)
            vectors.append(vec)
            anatomy_blocks.append(anatomy)
            img_ids.append(img_id)
            filenames.append(filename)
        except Exception as ex:
            print(f"  [WARN] Bo qua algorithmic embedding {filename}: {ex}")
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 512), dtype=np.float32)
    anatomy_matrix = np.stack(anatomy_blocks, axis=0) if anatomy_blocks else np.zeros((0, 5, 184), dtype=np.float32)
    return matrix, img_ids, filenames, anatomy_matrix


def build_recognition_feature_package(
    metadata_df: pd.DataFrame,
    attr_labels_df: pd.DataFrame,
    attr_names_df: pd.DataFrame,
    output_dir: str,
    certainty_threshold: int = 3,
    require_torch_for_cnn: bool = True,
    cub_root: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("BUOC 6: Xay dung bo thuoc tinh nhan dien (Yeu cau 2)")
    print("=" * 60)
    features_dir = os.path.join(output_dir, "features")
    reports_dir = os.path.join(output_dir, "reports")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    custom_attrs_df = extract_custom_visual_attributes(metadata_df, images_dir)
    custom_path = os.path.join(features_dir, "tier1_custom_attributes.csv")
    custom_attrs_df.to_csv(custom_path, index=False, encoding="utf-8")

    info_df = build_attribute_information_value_table(custom_attrs_df, metadata_df)
    info_csv_path = os.path.join(features_dir, "tier1_attribute_information_value.csv")
    info_df.to_csv(info_csv_path, index=False, encoding="utf-8")

    all_attr_cols = [c for c in custom_attrs_df.columns if c.startswith("cattr_")]
    similarity_cols = [
        c
        for c in [
            "cattr_color_red_ratio",
            "cattr_color_yellow_ratio",
            "cattr_color_blue_ratio",
            "cattr_sat_mean",
            "cattr_texture_entropy",
            "cattr_fg_area_ratio",
            "cattr_fg_aspect_ratio",
            "cattr_symmetry_lr",
            "cattr_grad_anisotropy",
            "cattr_hu1",
            "cattr_hu2",
        ]
        if c in all_attr_cols
    ]
    ranked_cols = info_df["attribute"].tolist() if not info_df.empty else all_attr_cols
    excluded_for_difference = {"cattr_centroid_y", "cattr_centroid_x", "cattr_val_mean"}
    difference_cols = [c for c in ranked_cols if c in all_attr_cols and c not in excluded_for_difference][:8]

    custom_attr_map = custom_attrs_df.set_index("img_id")
    ordered_img_ids = metadata_df["img_id"].astype(int).tolist()
    custom_matrix = (
        custom_attr_map.reindex(ordered_img_ids).fillna(0.0)[all_attr_cols].to_numpy(dtype=np.float32)
        if all_attr_cols
        else np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    )
    retrieval_attr_cols = [c for c in all_attr_cols if c not in set(RETRIEVAL_EXCLUDED_TIER1_ATTRS)]
    retrieval_custom_matrix = (
        custom_attr_map.reindex(ordered_img_ids).fillna(0.0)[retrieval_attr_cols].to_numpy(dtype=np.float32)
        if retrieval_attr_cols
        else np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    )

    similarity_df = (
        custom_attrs_df[["img_id"] + similarity_cols].copy() if similarity_cols else custom_attrs_df[["img_id"]].copy()
    )
    similarity_path = os.path.join(features_dir, "tier1_similarity_attributes.csv")
    similarity_df.to_csv(similarity_path, index=False, encoding="utf-8")

    difference_df = (
        custom_attrs_df[["img_id"] + difference_cols].copy() if difference_cols else custom_attrs_df[["img_id"]].copy()
    )
    difference_path = os.path.join(features_dir, "tier1_difference_attributes.csv")
    difference_df.to_csv(difference_path, index=False, encoding="utf-8")

    info_json_path = os.path.join(reports_dir, "tier1_attribute_information_value.json")
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(info_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    # Giu ten file cu de khong vo luong tai DB/retrieval phia sau.
    cub_312_path = os.path.join(features_dir, "tier1_cub312_binary.csv")
    custom_attrs_df[["img_id"] + all_attr_cols].to_csv(cub_312_path, index=False, encoding="utf-8")

    cnn_matrix, cnn_img_ids, cnn_filenames, anatomy_blocks = extract_efficientnetv2_embeddings(
        metadata_df,
        images_dir,
        require_torch_for_cnn,
        cub_root=cub_root,
    )
    cnn_path = os.path.join(features_dir, "tier2_algorithmic_embeddings.npy")
    np.save(cnn_path, cnn_matrix)
    cnn_index_path = os.path.join(features_dir, "tier2_algorithmic_index.csv")
    pd.DataFrame({"img_id": cnn_img_ids, "filename": cnn_filenames}).to_csv(cnn_index_path, index=False)
    np.savez_compressed(
        os.path.join(features_dir, "tier2_algorithmic_anatomy_blocks.npz"),
        img_ids=np.array(cnn_img_ids, dtype=np.int32),
        anatomy_blocks=anatomy_blocks.astype(np.float32),
    )

    handcrafted_df = extract_handcrafted_features(metadata_df, images_dir)
    handcrafted_path = os.path.join(features_dir, "tier3_handcrafted_features.csv")
    handcrafted_df.to_csv(handcrafted_path, index=False, encoding="utf-8")

    hc_feature_cols = [col for col in handcrafted_df.columns if col.startswith("hc_")]
    handcrafted_matrix = handcrafted_df.set_index("img_id")[hc_feature_cols].reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32) if hc_feature_cols else np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    if cnn_matrix.size:
        cnn_aligned = pd.DataFrame(cnn_matrix, index=cnn_img_ids).reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32)
    else:
        cnn_aligned = np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    write_attribute_diagnostic_reports(
        reports_dir=reports_dir,
        custom_attrs_df=custom_attrs_df,
        info_df=info_df,
        metadata_df=metadata_df,
        cnn_aligned=cnn_aligned,
    )
    tier2_compacted, tier2_pca_meta = _apply_tier2_pca(cnn_aligned, TIER2_PCA_DIM)
    custom_for_fusion = _zscore_per_column(retrieval_custom_matrix) if ENABLE_TIER_NORMALIZATION else retrieval_custom_matrix
    if ENABLE_HU_BOOST:
        custom_for_fusion = _apply_hu_boost(
            custom_for_fusion,
            attr_cols=retrieval_attr_cols,
            weight=float(HU_MOMENT_WEIGHT),
        )
    tier2_for_fusion = _zscore_per_column(tier2_compacted) if ENABLE_TIER_NORMALIZATION else tier2_compacted
    tier3_for_fusion = _zscore_per_column(handcrafted_matrix) if ENABLE_TIER_NORMALIZATION else handcrafted_matrix
    custom_weighted = custom_for_fusion * float(TIER1_WEIGHT)
    tier2_weighted = tier2_for_fusion * float(TIER2_WEIGHT)
    tier3_weighted = tier3_for_fusion * float(TIER3_WEIGHT)
    final_matrix = np.concatenate([custom_weighted, tier2_weighted, tier3_weighted], axis=1)
    final_path = os.path.join(features_dir, "recognition_features_all.npz")
    np.savez_compressed(
        final_path,
        img_ids=np.array(ordered_img_ids, dtype=np.int32),
        class_ids=metadata_df["class_id"].astype(int).to_numpy(dtype=np.int32),
        custom_attributes=custom_matrix,
        custom_attributes_retrieval=retrieval_custom_matrix,
        custom_attr_names_retrieval=np.array(retrieval_attr_cols),
        algorithmic_tier2=cnn_aligned,
        algorithmic_tier2_compacted=tier2_compacted,
        handcrafted=handcrafted_matrix,
        custom_weighted=custom_weighted,
        algorithmic_tier2_weighted=tier2_weighted,
        handcrafted_weighted=tier3_weighted,
        all_features=final_matrix,
    )
    manifest = {
        "total_images": int(len(metadata_df)),
        "tier1_custom_dim": int(custom_matrix.shape[1]),
        "tier1_retrieval_dim": int(retrieval_custom_matrix.shape[1]),
        "tier1_similarity_dim": int(len(similarity_cols)),
        "tier1_difference_dim": int(len(difference_cols)),
        "tier2_algorithmic_dim": int(cnn_aligned.shape[1]),
        "tier2_compacted_dim": int(tier2_compacted.shape[1]),
        "tier2_pca_enabled": bool(tier2_pca_meta["enabled"] > 0.5),
        "tier2_pca_explained_variance_ratio": float(tier2_pca_meta["explained_variance_ratio"]),
        "tier3_handcrafted_dim": int(handcrafted_matrix.shape[1]),
        "final_dim": int(final_matrix.shape[1]),
        "certainty_threshold": int(certainty_threshold),
        "tier_normalization_enabled": bool(ENABLE_TIER_NORMALIZATION),
        "tier_weights": {
            "tier1_custom": float(TIER1_WEIGHT),
            "tier2_algorithmic": float(TIER2_WEIGHT),
            "tier3_handcrafted": float(TIER3_WEIGHT),
        },
        "green_binary_enabled": bool(ENABLE_GREEN_BINARY),
        "green_ratio_binary_threshold": float(GREEN_RATIO_BINARY_THRESHOLD),
        "hu_boost_enabled": bool(ENABLE_HU_BOOST),
        "hu_moment_weight": float(HU_MOMENT_WEIGHT),
        "retrieval_excluded_tier1_attrs": list(RETRIEVAL_EXCLUDED_TIER1_ATTRS),
    }
    with open(os.path.join(reports_dir, "recognition_features_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_requirement2_report(reports_dir, similarity_cols, difference_cols, info_df, manifest)
