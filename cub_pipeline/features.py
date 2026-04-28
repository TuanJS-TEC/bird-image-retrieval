import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .algorithmic_tier2 import extract_algorithmic_embedding, load_algorithmic_artifacts
from .config import (
    ENABLE_GREEN_BINARY,
    ENABLE_HU_BOOST,
    ENABLE_TIER_NORMALIZATION,
    GREEN_RATIO_BINARY_THRESHOLD,
    HU_MOMENT_WEIGHT,
    RETRIEVAL_EXCLUDED_TIER1_ATTRS,
    TIER1_WEIGHT,
    TIER2_PCA_DIM,
    TIER2_WEIGHT,
    TIER3_WEIGHT,
)
from .tier1_features import (
    build_attribute_information_value_table as _tier1_build_attribute_information_value_table,
    build_cub_312_binary_attributes as _tier1_build_cub_312_binary_attributes,
    extract_custom_visual_attributes as _tier1_extract_custom_visual_attributes,
    select_similarity_attribute_ids as _tier1_select_similarity_attribute_ids,
)
from .tier2_features import extract_efficientnetv2_embeddings as _tier2_extract_efficientnetv2_embeddings
from .tier3_features import (
    extract_handcrafted_features as _tier3_extract_handcrafted_features,
    extract_hog_shape as _tier3_extract_hog_shape,
    extract_hsv_histogram_48 as _tier3_extract_hsv_histogram_48,
    extract_lbp_texture_64 as _tier3_extract_lbp_texture_64,
)


def _zscore_per_column(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    mat = matrix.astype(np.float32, copy=True)
    mean = np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((mat - mean) / std).astype(np.float32)


def _zscore_with_reference(target_matrix: np.ndarray, reference_matrix: np.ndarray | None) -> np.ndarray:
    if target_matrix.size == 0:
        return target_matrix.astype(np.float32, copy=False)
    if reference_matrix is None or reference_matrix.size == 0:
        return _zscore_per_column(target_matrix)
    ref = reference_matrix.astype(np.float32, copy=False)
    tgt = target_matrix.astype(np.float32, copy=True)
    mean = np.mean(ref, axis=0, keepdims=True)
    std = np.std(ref, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((tgt - mean) / std).astype(np.float32)


def _apply_tier2_pca(
    matrix: np.ndarray,
    target_dim: int | None,
    fit_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    if matrix.size == 0 or target_dim is None or int(target_dim) <= 0:
        dim = float(matrix.shape[1] if matrix.ndim == 2 else 0)
        return (
            matrix.astype(np.float32, copy=False),
            fit_matrix.astype(np.float32, copy=False) if fit_matrix is not None else None,
            {"enabled": 0.0, "input_dim": dim, "output_dim": dim, "explained_variance_ratio": 1.0},
        )
    train = fit_matrix if fit_matrix is not None and fit_matrix.size > 0 else matrix
    n, d = train.shape
    n_comp = max(1, min(int(target_dim), n, d))
    if n_comp >= d:
        return (
            matrix.astype(np.float32, copy=False),
            fit_matrix.astype(np.float32, copy=False) if fit_matrix is not None else None,
            {"enabled": 0.0, "input_dim": float(d), "output_dim": float(d), "explained_variance_ratio": 1.0},
        )
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(train.astype(np.float32))
    transformed = pca.transform(matrix.astype(np.float32)).astype(np.float32)
    transformed_fit = pca.transform(train.astype(np.float32)).astype(np.float32)
    meta = {
        "enabled": 1.0,
        "input_dim": float(d),
        "output_dim": float(n_comp),
        "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
    }
    return transformed, transformed_fit, meta


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
    return _tier1_extract_custom_visual_attributes(metadata_df, images_dir)


def build_attribute_information_value_table(custom_attrs_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    return _tier1_build_attribute_information_value_table(custom_attrs_df, metadata_df)


def build_cub_312_binary_attributes(
    metadata_df: pd.DataFrame,
    attr_labels_df: pd.DataFrame,
    attr_names_df: pd.DataFrame,
    certainty_threshold: int = 3,
) -> tuple[pd.DataFrame, dict[int, str]]:
    return _tier1_build_cub_312_binary_attributes(metadata_df, attr_labels_df, attr_names_df, certainty_threshold)


def select_similarity_attribute_ids(attr_names_df: pd.DataFrame) -> dict[str, list[int]]:
    return _tier1_select_similarity_attribute_ids(attr_names_df)


def extract_hsv_histogram_48(img, bins: int = 16) -> np.ndarray:
    return _tier3_extract_hsv_histogram_48(img, bins=bins)


def extract_lbp_texture_64(img, bins: int = 64) -> np.ndarray:
    return _tier3_extract_lbp_texture_64(img, bins=bins)


def extract_hog_shape(img, grid_size: int = 4, n_bins: int = 9) -> np.ndarray:
    return _tier3_extract_hog_shape(img, grid_size=grid_size, n_bins=n_bins)


def extract_handcrafted_features(metadata_df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    return _tier3_extract_handcrafted_features(metadata_df, images_dir)


def extract_efficientnetv2_embeddings(
    metadata_df: pd.DataFrame,
    images_dir: str,
    require_torch: bool = True,
    cub_root: str | None = None,
    fit_metadata_df: pd.DataFrame | None = None,
    fit_images_dir: str | None = None,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    # Backward-compatible API for callers importing from cub_pipeline.features.
    return _tier2_extract_efficientnetv2_embeddings(
        metadata_df=metadata_df,
        images_dir=images_dir,
        require_torch=require_torch,
        cub_root=cub_root,
        fit_metadata_df=fit_metadata_df,
        fit_images_dir=fit_images_dir,
    )


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


def build_recognition_feature_package(
    metadata_df: pd.DataFrame,
    attr_labels_df: pd.DataFrame,
    attr_names_df: pd.DataFrame,
    output_dir: str,
    certainty_threshold: int = 3,
    require_torch_for_cnn: bool = True,
    cub_root: str | None = None,
    fit_metadata_df: pd.DataFrame | None = None,
    fit_images_dir: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("BUOC 6: Xay dung bo thuoc tinh nhan dien (Yeu cau 2)")
    print("=" * 60)
    features_dir = os.path.join(output_dir, "features")
    reports_dir = os.path.join(output_dir, "reports")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    use_extended_fit_pool = fit_metadata_df is not None and fit_images_dir is not None
    fit_metadata_for_extract = None
    if use_extended_fit_pool:
        fit_metadata_for_extract = fit_metadata_df.copy()
        if "filepath" in fit_metadata_for_extract.columns:
            fit_metadata_for_extract["filename"] = fit_metadata_for_extract["filepath"].astype(str)
        print(
            "  [INFO] Fit statistics pool: EXTENDED "
            f"({len(fit_metadata_for_extract)} anh goc / {fit_metadata_for_extract['class_id'].nunique()} loai)"
        )
    else:
        print(f"  [INFO] Fit statistics pool: FILTERED ({len(metadata_df)} anh)")

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

    # Keep old file naming for downstream SQLite/retrieval compatibility.
    cub_312_path = os.path.join(features_dir, "tier1_cub312_binary.csv")
    custom_attrs_df[["img_id"] + all_attr_cols].to_csv(cub_312_path, index=False, encoding="utf-8")

    cnn_matrix, cnn_img_ids, cnn_filenames, anatomy_blocks = extract_efficientnetv2_embeddings(
        metadata_df,
        images_dir,
        require_torch_for_cnn,
        cub_root=cub_root,
        fit_metadata_df=fit_metadata_df,
        fit_images_dir=fit_images_dir,
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
    handcrafted_matrix = (
        handcrafted_df.set_index("img_id")[hc_feature_cols].reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32)
        if hc_feature_cols
        else np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    )
    if cnn_matrix.size:
        cnn_aligned = pd.DataFrame(cnn_matrix, index=cnn_img_ids).reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32)
    else:
        cnn_aligned = np.zeros((len(ordered_img_ids), 0), dtype=np.float32)

    fit_retrieval_custom_matrix = None
    fit_handcrafted_matrix = None
    fit_tier2_matrix = None
    if fit_metadata_for_extract is not None and fit_images_dir is not None:
        fit_custom_attrs_df = extract_custom_visual_attributes(fit_metadata_for_extract, fit_images_dir)
        if len(fit_custom_attrs_df) > 0 and retrieval_attr_cols:
            fit_custom_map = fit_custom_attrs_df.set_index("img_id")
            fit_ids = fit_metadata_for_extract["img_id"].astype(int).tolist()
            fit_retrieval_custom_matrix = (
                fit_custom_map.reindex(fit_ids).fillna(0.0)[retrieval_attr_cols].to_numpy(dtype=np.float32)
            )

        fit_handcrafted_df = extract_handcrafted_features(fit_metadata_for_extract, fit_images_dir)
        if len(fit_handcrafted_df) > 0 and hc_feature_cols:
            fit_hc_map = fit_handcrafted_df.set_index("img_id")
            fit_ids = fit_metadata_for_extract["img_id"].astype(int).tolist()
            fit_handcrafted_matrix = fit_hc_map.reindex(fit_ids).fillna(0.0)[hc_feature_cols].to_numpy(dtype=np.float32)

        # IMPORTANT:
        # Do not call extract_efficientnetv2_embeddings() recursively on fit pool.
        # That path may trigger re-fit with metadata schema mismatch (bbox_x/y/w/h).
        # We only need transformed vectors using artifacts already fit above.
        artifacts = load_algorithmic_artifacts(features_dir)
        fit_tier2_rows: list[np.ndarray] = []
        fit_tier2_ids: list[int] = []
        for _, rr in fit_metadata_for_extract.iterrows():
            img_id = int(rr["img_id"])
            filename_or_path = str(rr["filename"])
            img_path = os.path.join(fit_images_dir, filename_or_path)
            if not os.path.exists(img_path):
                continue
            try:
                vec, _ = extract_algorithmic_embedding(img_path, artifacts)
            except Exception:
                continue
            fit_tier2_rows.append(vec)
            fit_tier2_ids.append(img_id)
        if fit_tier2_rows:
            fit_cnn_matrix = np.vstack(fit_tier2_rows).astype(np.float32)
            fit_ids = fit_metadata_for_extract["img_id"].astype(int).tolist()
            fit_tier2_matrix = pd.DataFrame(fit_cnn_matrix, index=fit_tier2_ids).reindex(fit_ids).fillna(0).to_numpy(dtype=np.float32)
    write_attribute_diagnostic_reports(
        reports_dir=reports_dir,
        custom_attrs_df=custom_attrs_df,
        info_df=info_df,
        metadata_df=metadata_df,
        cnn_aligned=cnn_aligned,
    )
    tier2_compacted, fit_tier2_compacted, tier2_pca_meta = _apply_tier2_pca(
        cnn_aligned,
        TIER2_PCA_DIM,
        fit_matrix=fit_tier2_matrix,
    )
    custom_for_fusion = (
        _zscore_with_reference(retrieval_custom_matrix, fit_retrieval_custom_matrix)
        if ENABLE_TIER_NORMALIZATION
        else retrieval_custom_matrix
    )
    if ENABLE_HU_BOOST:
        custom_for_fusion = _apply_hu_boost(
            custom_for_fusion,
            attr_cols=retrieval_attr_cols,
            weight=float(HU_MOMENT_WEIGHT),
        )
    tier2_for_fusion = _zscore_with_reference(tier2_compacted, fit_tier2_compacted) if ENABLE_TIER_NORMALIZATION else tier2_compacted
    tier3_for_fusion = (
        _zscore_with_reference(handcrafted_matrix, fit_handcrafted_matrix)
        if ENABLE_TIER_NORMALIZATION
        else handcrafted_matrix
    )
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
        "fit_pool_mode": "extended_original_images_126_species" if use_extended_fit_pool else "filtered_511_only",
        "fit_pool_total_images": int(len(fit_metadata_for_extract)) if fit_metadata_for_extract is not None else int(len(metadata_df)),
    }
    with open(os.path.join(reports_dir, "recognition_features_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_requirement2_report(reports_dir, similarity_cols, difference_cols, info_df, manifest)
