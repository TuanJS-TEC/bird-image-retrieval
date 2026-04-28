import json
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from tqdm import tqdm


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    return vec / (np.linalg.norm(vec) + 1e-8)


def _clip_box(x0: int, y0: int, x1: int, y1: int, width: int = 224, height: int = 224) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, int(x0)))
    y0 = max(0, min(height - 1, int(y0)))
    x1 = max(x0 + 1, min(width, int(x1)))
    y1 = max(y0 + 1, min(height, int(y1)))
    return x0, y0, x1, y1


def _crop_rgb(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


@dataclass
class AlgorithmicArtifacts:
    anatomy_regions: dict[str, list[int]]
    color_indices: list[int]
    lbp_indices_r1: list[int]
    lbp_indices_r2: list[int]
    lbp_indices_r3: list[int]
    moments_min: list[float]
    moments_max: list[float]
    agsfp_pca: Any
    acv_vocab: np.ndarray
    acv_idf: np.ndarray
    acv_pca: Any
    hfve_gmm: Any
    hfve_pca: Any
    final_pca: Any


def _tier2_speed_profile() -> dict[str, int]:
    """
    Tier2 speed/quality trade-off profile controlled by ALGO_TIER2_SPEED_LEVEL.
    Supported levels: quality | balanced | fast.
    """
    level = os.environ.get("ALGO_TIER2_SPEED_LEVEL", "balanced").strip().lower()
    if level == "quality":
        return {
            "level": 2,
            "dominant_color_sample": 0,
            "dominant_color_max_iter": 20,
            "acv_fg_sample": 0,
            "hfve_patch_stride": 8,
        }
    if level == "fast":
        return {
            "level": 0,
            "dominant_color_sample": 1200,
            "dominant_color_max_iter": 12,
            "acv_fg_sample": 2500,
            "hfve_patch_stride": 12,
        }
    return {
        "level": 1,
        "dominant_color_sample": 1800,
        "dominant_color_max_iter": 16,
        "acv_fg_sample": 4000,
        "hfve_patch_stride": 10,
    }


def _subsample_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or arr.shape[0] <= max_rows:
        return arr
    # Deterministic subsampling keeps runs reproducible and stable.
    idx = np.linspace(0, arr.shape[0] - 1, num=max_rows, dtype=np.int32)
    return arr[idx]


def _read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def _hsv_hist_256(region_rgb: np.ndarray) -> np.ndarray:
    hsv = np.array(Image.fromarray(region_rgb).convert("HSV"), dtype=np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hist, _ = np.histogramdd(
        np.stack([h, s, v], axis=-1).reshape(-1, 3),
        bins=(16, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.reshape(-1).astype(np.float32)
    return hist / (float(hist.sum()) + 1e-8)


def _dominant_color_24(region_rgb: np.ndarray) -> np.ndarray:
    prof = _tier2_speed_profile()
    out: list[float] = []
    if int(prof["level"]) >= 2:
        pixels_rgb = region_rgb.reshape(-1, 3).astype(np.float32)
        pixels_rgb = _subsample_rows(pixels_rgb, int(prof["dominant_color_sample"]))
        if len(pixels_rgb) < 3:
            pixels_rgb = np.repeat(pixels_rgb, 3, axis=0)
        km = MiniBatchKMeans(
            n_clusters=3,
            batch_size=2048,
            n_init="auto",
            max_iter=int(prof["dominant_color_max_iter"]),
            random_state=42,
        )
        labels = km.fit_predict(pixels_rgb)
        centers_rgb = np.clip(km.cluster_centers_, 0.0, 255.0).astype(np.uint8)
        counts = np.bincount(labels, minlength=3).astype(np.float32)
        ratios = counts / (float(counts.sum()) + 1e-8)
    else:
        q = (region_rgb.astype(np.uint16) >> 5).reshape(-1, 3)  # 8 bins/channel
        bin_id = (q[:, 0] << 6) + (q[:, 1] << 3) + q[:, 2]
        hist = np.bincount(bin_id, minlength=512).astype(np.float32)
        top = np.argsort(-hist)[:3]
        centers = []
        ratios = []
        total = float(hist.sum()) + 1e-8
        for b in top:
            r_bin = (b >> 6) & 7
            g_bin = (b >> 3) & 7
            b_bin = b & 7
            centers.append(
                [
                    int(r_bin * 32 + 16),
                    int(g_bin * 32 + 16),
                    int(b_bin * 32 + 16),
                ]
            )
            ratios.append(float(hist[b] / total))
        centers_rgb = np.array(centers, dtype=np.uint8)
        ratios = np.array(ratios, dtype=np.float32)
    order = np.argsort(-ratios)

    centers_strip = centers_rgb.reshape(1, 3, 3)
    centers_hsv = np.array(Image.fromarray(centers_strip, mode="RGB").convert("HSV"), dtype=np.float32).reshape(3, 3)
    centers_lab = np.array(Image.fromarray(centers_strip, mode="RGB").convert("LAB"), dtype=np.float32).reshape(3, 3)

    for centers in (centers_hsv, centers_lab):
        for idx in order:
            c = centers[idx]
            r = ratios[idx]
            out.extend([float(c[0] / 255.0), float(c[1] / 255.0), float(c[2] / 255.0), float(r)])
    return np.array(out, dtype=np.float32)


def _lbp_hist_256(gray: np.ndarray, radius: int, points: int) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
    lbp_scaled = np.clip((lbp / (lbp.max() + 1e-8)) * 255.0, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(lbp_scaled, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    return hist / (float(hist.sum()) + 1e-8)


def _oriented_gradient_36(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32) / 255.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ori = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
    hist, _ = np.histogram(ori.reshape(-1), bins=36, range=(0, 360), weights=mag.reshape(-1))
    hist = hist.astype(np.float32)
    return hist / (float(hist.sum()) + 1e-8)


def _moments_lab_12(region_rgb: np.ndarray) -> np.ndarray:
    lab = rgb2lab(region_rgb.astype(np.float32) / 255.0)
    feats: list[float] = []
    for i in range(3):
        c = lab[:, :, i].reshape(-1)
        mean_val = float(np.mean(c))
        std_val = float(np.std(c))
        if std_val < 1e-6:
            # Nearly uniform region — skew and kurtosis are undefined/meaningless.
            # Return 0.0 directly to avoid scipy RuntimeWarning about precision loss.
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = float(stats.skew(c, bias=False, nan_policy="omit"))
            kurt_val = float(stats.kurtosis(c, fisher=True, bias=False, nan_policy="omit"))
        feats.extend([mean_val, std_val, skew_val, kurt_val])
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)



def _fit_anatomy_regions(metadata_df: pd.DataFrame, cub_root: str) -> dict[str, list[int]]:
    part_path = os.path.join(cub_root, "parts", "part_locs.txt")
    parts = pd.read_csv(part_path, sep=" ", names=["img_id", "part_id", "x", "y", "visible"])
    parts = parts[parts["visible"] == 1].copy()
    cols = ["img_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    meta = metadata_df[cols].copy()
    merged = parts.merge(meta, on="img_id", how="inner")
    merged = merged[(merged["bbox_w"] > 1e-6) & (merged["bbox_h"] > 1e-6)]
    merged["x_new"] = (merged["x"] - merged["bbox_x"]) / merged["bbox_w"] * 224.0
    merged["y_new"] = (merged["y"] - merged["bbox_y"]) / merged["bbox_h"] * 224.0

    groups = {
        "head": [1, 2, 3, 4, 5, 6],
        "breast": [7, 8, 9],
        "back_wing": [10, 11, 12],
        "tail": [13, 14],
        "leg": [15],
    }
    default_boxes = {
        "head": [8, 8, 96, 92],
        "breast": [68, 70, 168, 190],
        "back_wing": [52, 30, 196, 146],
        "tail": [150, 68, 224, 188],
        "leg": [74, 160, 156, 224],
    }
    out: dict[str, list[int]] = {}
    for name, part_ids in groups.items():
        sub = merged[merged["part_id"].isin(part_ids)]
        if sub.empty:
            out[name] = default_boxes[name]
            continue
        mx, my = float(sub["x_new"].mean()), float(sub["y_new"].mean())
        sx, sy = float(sub["x_new"].std(ddof=0)), float(sub["y_new"].std(ddof=0))
        pad_x = max(16.0, 1.8 * sx)
        pad_y = max(16.0, 1.8 * sy)
        x0, y0, x1, y1 = _clip_box(mx - pad_x, my - pad_y, mx + pad_x, my + pad_y)
        out[name] = [x0, y0, x1, y1]
    return out


def _fg_mask(region_rgb: np.ndarray) -> np.ndarray:
    hsv = np.array(Image.fromarray(region_rgb).convert("HSV"), dtype=np.uint8)
    sat = hsv[:, :, 1]
    thr = int(threshold_otsu(sat))
    mask = sat >= thr
    if mask.mean() < 0.05:
        mask = sat >= max(8, int(np.quantile(sat, 0.5)))
    return mask


def _extract_region_184(region_rgb: np.ndarray, artifacts: AlgorithmicArtifacts) -> np.ndarray:
    color = _hsv_hist_256(region_rgb)[artifacts.color_indices]
    color = color / (float(color.sum()) + 1e-8)
    dcolor = _dominant_color_24(region_rgb)

    gray = np.array(Image.fromarray(region_rgb).convert("L"), dtype=np.uint8)
    lbp1 = _lbp_hist_256(gray, radius=1, points=8)[artifacts.lbp_indices_r1]
    lbp2 = _lbp_hist_256(gray, radius=2, points=16)[artifacts.lbp_indices_r2]
    lbp3 = _lbp_hist_256(gray, radius=3, points=24)[artifacts.lbp_indices_r3]
    lbp = np.concatenate([lbp1, lbp2, lbp3], axis=0)
    lbp = lbp / (float(lbp.sum()) + 1e-8)

    grad = _oriented_gradient_36(gray)
    moments = _moments_lab_12(region_rgb)
    mn = np.array(artifacts.moments_min, dtype=np.float32)
    mx = np.array(artifacts.moments_max, dtype=np.float32)
    moments = np.clip((moments - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    for idx in [2, 3, 6, 7, 10, 11]:
        moments[idx] = np.clip((moments[idx] * 6.0) - 3.0, -3.0, 3.0) / 3.0
    return np.concatenate([color, dcolor, lbp, grad, moments], axis=0).astype(np.float32)


def _agsfp_2760(image_rgb: np.ndarray, artifacts: AlgorithmicArtifacts) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    anatomy_blocks: list[np.ndarray] = []
    blocks.append(_extract_region_184(image_rgb, artifacts))
    step = 224 // 3
    for r in range(3):
        for c in range(3):
            x0, y0 = c * step, r * step
            x1 = 224 if c == 2 else (c + 1) * step
            y1 = 224 if r == 2 else (r + 1) * step
            blocks.append(_extract_region_184(_crop_rgb(image_rgb, (x0, y0, x1, y1)), artifacts))
    for key in ["head", "breast", "back_wing", "tail", "leg"]:
        box = tuple(artifacts.anatomy_regions[key])
        vec = _extract_region_184(_crop_rgb(image_rgb, box), artifacts)
        blocks.append(vec)
        anatomy_blocks.append(vec)
    return np.concatenate(blocks, axis=0), np.stack(anatomy_blocks, axis=0)


def _ppd_152(image_rgb: np.ndarray) -> np.ndarray:
    gray = np.array(Image.fromarray(image_rgb).convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ori = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    
    fdi = []
    hs = np.array_split(np.arange(224), 6)
    for ys in hs:
        h, _ = np.histogram(ori[ys, :].reshape(-1), bins=6, range=(0, 180), weights=mag[ys, :].reshape(-1))
        h = h.astype(np.float32)
        fdi.append(h / (float(h.sum()) + 1e-8))
    fdi_vec = np.concatenate(fdi, axis=0)

    lab = rgb2lab(image_rgb.astype(np.float32) / 255.0)
    cbs = []
    for ch in [1, 2]:
        channel = lab[:, :, ch]
        cx = np.zeros_like(channel, dtype=np.float32)
        cy = np.zeros_like(channel, dtype=np.float32)
        cx[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
        cy[1:-1, :] = channel[2:, :] - channel[:-2, :]
        cmag = np.sqrt(cx**2 + cy**2)
        for ys in np.array_split(np.arange(224), 10):
            cbs.append(float(cmag[ys, :].mean()))
    cbs_vec = np.array(cbs, dtype=np.float32)
    cbs_vec = (cbs_vec - cbs_vec.min()) / (cbs_vec.max() - cbs_vec.min() + 1e-8)

    left = image_rgb[:, :112, :]
    right = np.fliplr(image_rgb[:, 112:, :])
    diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).astype(np.uint8)
    hsv_diff = np.array(Image.fromarray(diff).convert("HSV"), dtype=np.uint8)
    lab_diff = np.array(Image.fromarray(diff).convert("LAB"), dtype=np.uint8)
    bsp = []
    for ch in [hsv_diff[:, :, 0], hsv_diff[:, :, 1], hsv_diff[:, :, 2], lab_diff[:, :, 0]]:
        hist, _ = np.histogram(ch.reshape(-1), bins=8, range=(0, 256))
        hist = hist.astype(np.float32)
        bsp.append(hist / (float(hist.sum()) + 1e-8))
    bsp_vec = np.concatenate(bsp, axis=0)

    hsv = np.array(Image.fromarray(image_rgb).convert("HSV"), dtype=np.float32)
    strips = np.array_split(np.arange(224), 8)
    hs: list[tuple[float, float]] = []
    for ys in strips:
        section = hsv[ys, :, :].reshape(-1, 3)
        hs.append((float(section[:, 0].mean() / 255.0), float(section[:, 1].mean() / 255.0)))
    d_h, d_s = [], []
    for i in range(len(hs) - 1):
        dh = hs[i + 1][0] - hs[i][0]
        dh = ((dh + 0.5) % 1.0) - 0.5
        ds = hs[i + 1][1] - hs[i][1]
        d_h.append(dh)
        d_s.append(ds)
    ctm_vals: list[float] = []
    for dh, ds in zip(d_h, d_s):
        hbin = np.histogram([dh], bins=4, range=(-0.5, 0.5))[0].astype(np.float32)
        sbin = np.histogram([ds], bins=4, range=(-1.0, 1.0))[0].astype(np.float32)
        ctm_vals.extend((hbin / (hbin.sum() + 1e-8)).tolist())
        ctm_vals.extend((sbin / (sbin.sum() + 1e-8)).tolist())
    ctm_stats = [float(np.mean(d_h)), float(np.std(d_h)), float(np.mean(d_s)), float(np.std(d_s))]
    ctm_stats = ctm_stats * 2
    ctm = np.array(ctm_vals + ctm_stats, dtype=np.float32)
    if ctm.shape[0] < 64:
        ctm = np.pad(ctm, (0, 64 - ctm.shape[0]))
    else:
        ctm = ctm[:64]
    return _l2_normalize(np.concatenate([fdi_vec, cbs_vec, bsp_vec, ctm], axis=0))


def _dense_root_patch_descriptor_128(gray: np.ndarray, patch_size: int = 32, stride: int = 8) -> np.ndarray:
    h, w = gray.shape
    descs: list[np.ndarray] = []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            patch = gray[y : y + patch_size, x : x + patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            gx = np.zeros_like(patch, dtype=np.float32)
            gy = np.zeros_like(patch, dtype=np.float32)
            gx[:, 1:-1] = patch[:, 2:] - patch[:, :-2]
            gy[1:-1, :] = patch[2:, :] - patch[:-2, :]
            mag = np.sqrt(gx**2 + gy**2)
            ori = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
            feat = []
            for r in range(4):
                for c in range(4):
                    ys = slice(r * 8, (r + 1) * 8)
                    xs = slice(c * 8, (c + 1) * 8)
                    hst, _ = np.histogram(ori[ys, xs].reshape(-1), bins=8, range=(0, 360), weights=mag[ys, xs].reshape(-1))
                    feat.append(hst.astype(np.float32))
            vec = np.concatenate(feat, axis=0)
            vec = vec / (float(np.sum(np.abs(vec))) + 1e-8)
            vec = np.sqrt(vec)
            descs.append(vec.astype(np.float32))
    if not descs:
        return np.zeros((1, 128), dtype=np.float32)
    return np.stack(descs, axis=0)


def _fisher_encode(descriptors: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    descriptors = descriptors.astype(np.float32)
    q = gmm.predict_proba(descriptors)  # N x K
    means = gmm.means_.astype(np.float32)  # K x D
    cov = gmm.covariances_.astype(np.float32)  # K x D
    sigma = np.sqrt(cov + 1e-8)
    n = float(descriptors.shape[0])
    diff = descriptors[:, None, :] - means[None, :, :]
    u = (q[:, :, None] * (diff / sigma[None, :, :])).sum(axis=0) / n
    v = (q[:, :, None] * (((diff**2) / (cov[None, :, :] + 1e-8)) - 1.0) / np.sqrt(2.0)).sum(axis=0) / n
    fv = np.concatenate([u.reshape(-1), v.reshape(-1)], axis=0).astype(np.float32)
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-8)
    return _l2_normalize(fv)


def fit_algorithmic_tier2(
    metadata_df: pd.DataFrame,
    images_dir: str,
    cub_root: str,
    features_dir: str,
    fit_metadata_df: pd.DataFrame | None = None,
    fit_images_dir: str | None = None,
) -> AlgorithmicArtifacts:
    """Fit Tier-2 algorithmic artifacts and encode retrieval vectors.

    Args:
        metadata_df: DataFrame of the 511 filtered images used for retrieval.
        images_dir: Directory containing the 511 processed/filtered images.
        cub_root: CUB_200_2011 root path (needed for anatomy part_locs).
        features_dir: Output directory for saved artifacts.
        fit_metadata_df: Optional larger pool DataFrame (e.g. 7 486 images from
            the original CUB dataset for the 126 filtered species).  When
            provided, unsupervised fitting (PCA, GMM, KMeans, anatomy priors)
            uses this pool instead of the 511 filtered images.
        fit_images_dir: Directory of images for fit_metadata_df.  Must be set
            when fit_metadata_df is provided.
    """
    os.makedirs(features_dir, exist_ok=True)
    prof = _tier2_speed_profile()
    fast_mode = os.environ.get("ALGO_TIER2_FAST_MODE", "1") == "1"

    # ── Decide which image pool to use for fitting ───────────────────────────
    use_extended_fit = fit_metadata_df is not None and fit_images_dir is not None
    if use_extended_fit:
        fit_img_rows = fit_metadata_df[["img_id", "filepath"]].copy()
        fit_paths = [
            os.path.join(fit_images_dir, fp)
            for fp in fit_img_rows["filepath"].tolist()
            if os.path.exists(os.path.join(fit_images_dir, fp))
        ]
        print(f"  [INFO] Fit pool: EXTENDED ({len(fit_paths)} anh goc cua 126 loai)")
    else:
        img_rows = metadata_df[["img_id", "filename"]].copy()
        fit_paths = [
            os.path.join(images_dir, fn)
            for fn in img_rows["filename"].tolist()
            if os.path.exists(os.path.join(images_dir, fn))
        ]
        print(f"  [INFO] Fit pool: FILTERED only ({len(fit_paths)} anh sau loc)")

    # For anatomy priors, always use metadata_df (511 filtered) which has bbox
    # coords aligned to the cropped/resized images already saved to images_dir.
    # The extended pool uses raw CUB images, whose bboxes differ, so anatomy
    # fitting still leverages metadata_df for coordinate-based region mapping.
    anatomy_meta = metadata_df

    sampled_limit = 240 if fast_mode else 600
    sampled_paths = fit_paths[: min(sampled_limit, len(fit_paths))]
    sampled_imgs = [_read_rgb(p) for p in sampled_paths]
    print(f"  [INFO] Tier2 fit mode={'FAST' if fast_mode else 'FULL'} | sampled_images={len(sampled_imgs)}")

    anatomy_regions = _fit_anatomy_regions(anatomy_meta, cub_root)


    color_bank = np.stack([_hsv_hist_256(img) for img in sampled_imgs], axis=0) if sampled_imgs else np.zeros((1, 256), dtype=np.float32)
    color_var = color_bank.var(axis=0)
    color_indices = np.argsort(-color_var)[:48].astype(int).tolist()

    lbp_specs = [(1, 8, 21), (2, 16, 21), (3, 24, 22)]
    lbp_idxs: list[list[int]] = []
    for radius, points, keep in lbp_specs:
        bank = []
        for img in sampled_imgs:
            gray = np.array(Image.fromarray(img).convert("L"), dtype=np.uint8)
            bank.append(_lbp_hist_256(gray, radius, points))
        mat = np.stack(bank, axis=0) if bank else np.zeros((1, 256), dtype=np.float32)
        lbp_idxs.append(np.argsort(-mat.var(axis=0))[:keep].astype(int).tolist())

    moments_bank = np.stack([_moments_lab_12(img) for img in sampled_imgs], axis=0) if sampled_imgs else np.zeros((1, 12), dtype=np.float32)
    moments_min = moments_bank.min(axis=0).astype(np.float32).tolist()
    moments_max = moments_bank.max(axis=0).astype(np.float32).tolist()

    dummy = PCA(n_components=1, random_state=42)
    artifacts = AlgorithmicArtifacts(
        anatomy_regions=anatomy_regions,
        color_indices=color_indices,
        lbp_indices_r1=lbp_idxs[0],
        lbp_indices_r2=lbp_idxs[1],
        lbp_indices_r3=lbp_idxs[2],
        moments_min=moments_min,
        moments_max=moments_max,
        agsfp_pca=dummy,
        acv_vocab=np.zeros((256, 3), dtype=np.float32),
        acv_idf=np.ones(256, dtype=np.float32),
        acv_pca=dummy,
        hfve_gmm=None,
        hfve_pca=dummy,
        final_pca=dummy,
    )

    # ── Encoding paths: always the 511 filtered images ────────────────────────
    retrieval_img_rows = metadata_df[["img_id", "filename"]].copy()
    encode_paths = [
        os.path.join(images_dir, fn)
        for fn in retrieval_img_rows["filename"].tolist()
        if os.path.exists(os.path.join(images_dir, fn))
    ]
    print(f"  [INFO] Encode pool: {len(encode_paths)} anh (511 anh da loc cho FAISS/SQLite)")

    all_px = []
    for img in tqdm(sampled_imgs, desc="    ACV collect foreground", leave=False):
        mask = _fg_mask(img)
        lab = rgb2lab(img.astype(np.float32) / 255.0).reshape(-1, 3)
        fg = lab[mask.reshape(-1)]
        if len(fg):
            all_px.append(fg)
    if all_px:
        px = np.concatenate(all_px, axis=0)
        if len(px) > 500_000:
            idx = np.random.default_rng(42).choice(len(px), size=500_000, replace=False)
            px = px[idx]
        km = MiniBatchKMeans(n_clusters=256, batch_size=8000, n_init=3 if fast_mode else 5, random_state=42)
        km.fit(px)
        vocab = km.cluster_centers_.astype(np.float32)
    else:
        vocab = np.zeros((256, 3), dtype=np.float32)
    artifacts.acv_vocab = vocab
    acv_tree = KDTree(artifacts.acv_vocab.astype(np.float32))

    desc_pool = []
    gmm_sample_cap = 160 if fast_mode else 300
    for img in tqdm(sampled_imgs[: min(gmm_sample_cap, len(sampled_imgs))], desc="    HFVE collect descriptors", leave=False):
        gray = np.array(Image.fromarray(img).convert("L"), dtype=np.float32) / 255.0
        d = _dense_root_patch_descriptor_128(gray, stride=int(prof["hfve_patch_stride"]))
        local_pick = 70 if fast_mode else 100
        if d.shape[0] > local_pick:
            sel = np.random.default_rng(42).choice(d.shape[0], size=local_pick, replace=False)
            d = d[sel]
        desc_pool.append(d)
    desc_train = np.concatenate(desc_pool, axis=0) if desc_pool else np.zeros((100, 128), dtype=np.float32)
    gmm_k = 48 if fast_mode else 64
    gmm = GaussianMixture(n_components=gmm_k, covariance_type="diag", max_iter=60 if fast_mode else 100, n_init=2, random_state=42)
    gmm.fit(desc_train)
    artifacts.hfve_gmm = gmm

    agsfp_vectors, acv_vectors, hfve_vectors, ppd_vectors = [], [], [], []
    acv_df_count = np.zeros(256, dtype=np.float32)
    anatomy_bank = []
    img_ids = []
    for path in tqdm(encode_paths, desc="    Encode Tier2 algorithmic", leave=False):
        img = _read_rgb(path)
        ag_raw, anatomy_blocks = _agsfp_2760(img, artifacts)
        agsfp_vectors.append(ag_raw)
        anatomy_bank.append(anatomy_blocks.astype(np.float32))

        acv_parts = []
        present_words = np.zeros(256, dtype=bool)
        for key in ["head", "breast", "back_wing", "tail", "leg"]:
            region = _crop_rgb(img, tuple(artifacts.anatomy_regions[key]))
            mask = _fg_mask(region)
            lab = rgb2lab(region.astype(np.float32) / 255.0).reshape(-1, 3)
            fg = lab[mask.reshape(-1)]
            if len(fg) == 0:
                fg = lab
            fg = _subsample_rows(fg, int(prof["acv_fg_sample"]))
            word = acv_tree.query(fg, k=1, workers=-1)[1]
            hist = np.bincount(word, minlength=256).astype(np.float32)
            present_words |= hist > 0
            acv_parts.append(hist)
        acv_vec = np.concatenate(acv_parts, axis=0).astype(np.float32)
        acv_vectors.append(acv_vec)
        acv_df_count += present_words.astype(np.float32)

        gray = np.array(Image.fromarray(img).convert("L"), dtype=np.float32) / 255.0
        desc = _dense_root_patch_descriptor_128(gray, stride=int(prof["hfve_patch_stride"]))
        hfve_vectors.append(_fisher_encode(desc, artifacts.hfve_gmm))
        ppd_vectors.append(_ppd_152(img))

    n_img = max(1, len(encode_paths))
    idf = np.log((n_img + 1.0) / (acv_df_count + 1.0)).astype(np.float32)
    artifacts.acv_idf = idf
    acv_weighted = [(_l2_normalize(v.reshape(5, 256) * idf.reshape(1, 256)).reshape(-1)) for v in acv_vectors]

    agsfp_mat = np.stack(agsfp_vectors, axis=0).astype(np.float32) if agsfp_vectors else np.zeros((1, 2760), dtype=np.float32)
    acv_mat = np.stack(acv_weighted, axis=0).astype(np.float32) if acv_weighted else np.zeros((1, 1280), dtype=np.float32)
    hfve_mat = np.stack(hfve_vectors, axis=0).astype(np.float32) if hfve_vectors else np.zeros((1, 16384), dtype=np.float32)
    ppd_mat = np.stack(ppd_vectors, axis=0).astype(np.float32) if ppd_vectors else np.zeros((1, 152), dtype=np.float32)

    def _fit_pca(mat: np.ndarray, max_dim: int) -> PCA:
        n_comp = int(min(max_dim, mat.shape[0], mat.shape[1]))
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(mat)
        return pca

    artifacts.agsfp_pca = _fit_pca(agsfp_mat, 512)
    artifacts.acv_pca = _fit_pca(acv_mat, 256)
    artifacts.hfve_pca = _fit_pca(hfve_mat, 512)

    agsfp_512 = artifacts.agsfp_pca.transform(agsfp_mat).astype(np.float32)
    acv_256 = artifacts.acv_pca.transform(acv_mat).astype(np.float32)
    hfve_512 = artifacts.hfve_pca.transform(hfve_mat).astype(np.float32)
    fused = np.concatenate([agsfp_512, acv_256, hfve_512, ppd_mat], axis=1)
    artifacts.final_pca = _fit_pca(fused, 512)

    with open(os.path.join(features_dir, "tier2_algorithmic_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    with open(os.path.join(features_dir, "anatomy_regions.json"), "w", encoding="utf-8") as f:
        json.dump(anatomy_regions, f, indent=2)
    return artifacts


def load_algorithmic_artifacts(features_dir: str) -> AlgorithmicArtifacts:
    path = os.path.join(features_dir, "tier2_algorithmic_artifacts.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_algorithmic_embedding(
    image_path: str,
    artifacts: AlgorithmicArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    prof = _tier2_speed_profile()
    img = _read_rgb(image_path)
    ag_raw, anatomy_blocks = _agsfp_2760(img, artifacts)
    agsfp_512 = artifacts.agsfp_pca.transform(ag_raw.reshape(1, -1)).astype(np.float32)
    acv_tree = KDTree(artifacts.acv_vocab.astype(np.float32))

    acv_parts = []
    for key in ["head", "breast", "back_wing", "tail", "leg"]:
        region = _crop_rgb(img, tuple(artifacts.anatomy_regions[key]))
        mask = _fg_mask(region)
        lab = rgb2lab(region.astype(np.float32) / 255.0).reshape(-1, 3)
        fg = lab[mask.reshape(-1)]
        if len(fg) == 0:
            fg = lab
        fg = _subsample_rows(fg, int(prof["acv_fg_sample"]))
        word = acv_tree.query(fg, k=1, workers=-1)[1]
        hist = np.bincount(word, minlength=256).astype(np.float32)
        acv_parts.append(hist)
    acv_raw = np.concatenate(acv_parts, axis=0).reshape(5, 256) * artifacts.acv_idf.reshape(1, 256)
    acv_raw = _l2_normalize(acv_raw.reshape(-1)).reshape(1, -1).astype(np.float32)
    acv_256 = artifacts.acv_pca.transform(acv_raw).astype(np.float32)

    gray = np.array(Image.open(image_path).convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    desc = _dense_root_patch_descriptor_128(gray, stride=int(prof["hfve_patch_stride"]))
    hfve_raw = _fisher_encode(desc, artifacts.hfve_gmm).reshape(1, -1)
    hfve_512 = artifacts.hfve_pca.transform(hfve_raw).astype(np.float32)

    ppd_152 = _ppd_152(img).reshape(1, -1).astype(np.float32)
    fused_1432 = np.concatenate([agsfp_512, acv_256, hfve_512, ppd_152], axis=1).astype(np.float32)
    final = artifacts.final_pca.transform(fused_1432).reshape(-1).astype(np.float32)
    return _l2_normalize(final), anatomy_blocks.astype(np.float32)

