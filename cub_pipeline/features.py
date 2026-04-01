import importlib
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


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


def extract_resnet50_embeddings(metadata_df: pd.DataFrame, images_dir: str, require_torch: bool = True) -> tuple[np.ndarray, list[int], list[str]]:
    try:
        torch = importlib.import_module("torch")
        models = importlib.import_module("torchvision.models")
        transforms = importlib.import_module("torchvision.transforms")
    except Exception as ex:
        if require_torch:
            raise RuntimeError("Khong the import torch/torchvision. Can cai dat de tao ResNet50 embeddings.") from ex
        print(f"  [WARN] Khong co torch/torchvision -> bo qua Tang 2. ({ex})")
        return np.zeros((0, 0), dtype=np.float32), [], []
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    vectors, img_ids, filenames = [], [], []
    with torch.no_grad():
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 2 - ResNet50"):
            filename = str(row["filename"])
            img_id = int(row["img_id"])
            img_path = os.path.join(images_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                feat = backbone(tensor).flatten(1).squeeze(0).cpu().numpy().astype(np.float32)
                vectors.append(feat)
                img_ids.append(img_id)
                filenames.append(filename)
            except Exception as ex:
                print(f"  [WARN] Bo qua embedding {filename}: {ex}")
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 2048), dtype=np.float32)
    return matrix, img_ids, filenames


def write_requirement2_report(
    reports_dir: str,
    selected_groups: dict[str, list[int]],
    attr_names_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    spec = _similarity_attribute_spec()
    lines: list[str] = []
    lines.append("# Yeu cau 2 - Bo thuoc tinh (3 tang)")
    lines.append("")
    lines.append("Dataset CUB-200-2011 co 312 thuoc tinh nhi phan, duoc to chuc theo 28 nhom thuoc tinh thi giac.")
    lines.append("He thong trich xuat cho bai toan nhan dang duoc trinh bay theo 3 tang sau:")
    lines.append("")
    lines.append("## Tang 1 - Thuoc tinh the hien su tuong dong (CUB attributes)")
    lines.append("")
    lines.append("| Nhom | Thuoc tinh cu the | Ly do chon |")
    lines.append("| --- | --- | --- |")
    for group_name, info in spec.items():
        attrs = ", ".join(info["display_attributes"])
        reason = info["reason"]
        lines.append(f"| {group_name} | {attrs} | {reason} |")
    lines.append("")
    lines.append("Ket qua tu dong map tren CUB (theo keyword) duoc luu trong `tier1_similarity_groups.json`.")
    lines.append("")
    for group_name, attr_ids in selected_groups.items():
        names = attr_names_df[attr_names_df["attr_id"].isin(attr_ids)]["attr_name"].tolist()
        lines.append(f"- `{group_name}`: {len(attr_ids)} attributes")
        if names:
            preview = ", ".join(names[:8])
            suffix = " ..." if len(names) > 8 else ""
            lines.append(f"  - Vi du: {preview}{suffix}")
    lines.append("")
    lines.append("## Tang 2 - Deep features (CNN Embedding)")
    lines.append("")
    lines.append("Su dung ResNet50 pre-trained (ImageNet), bo classifier cuoi de lay embedding 2048 chieu:")
    lines.append("")
    lines.append("```python")
    lines.append("model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)")
    lines.append("backbone = torch.nn.Sequential(*list(model.children())[:-1])  # -> 2048-dim")
    lines.append("backbone.eval()")
    lines.append("")
    lines.append("transform = transforms.Compose([")
    lines.append("    transforms.Resize((224, 224)),")
    lines.append("    transforms.ToTensor(),")
    lines.append("    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),")
    lines.append("])")
    lines.append("```")
    lines.append("")
    lines.append("Embedding duoc luu trong `tier2_resnet50_embeddings.npy` va `tier2_resnet50_index.csv`.")
    lines.append("")
    lines.append("## Tang 3 - Thuoc tinh thu cong (handcrafted)")
    lines.append("")
    lines.append("- Color Histogram HSV: `3 x 16 bins = 48` chieu")
    lines.append("- LBP Texture: `64` chieu")
    lines.append("- HOG Shape: tuy cau hinh (hien tai `grid_size=4`, `n_bins=9` -> `144` chieu)")
    lines.append("")
    lines.append("Tat ca dac trung duoc ghep vao `recognition_features_all.npz`.")
    lines.append("")
    lines.append("## Tong hop kich thuoc vector")
    lines.append("")
    lines.append(f"- Tang 1 (CUB-312): {manifest['tier1_cub312_dim']}")
    lines.append(f"- Tang 1 (Similarity subset): {manifest['tier1_similarity_dim']}")
    lines.append(f"- Tang 2 (ResNet50): {manifest['tier2_resnet50_dim']}")
    lines.append(f"- Tang 3 (Handcrafted): {manifest['tier3_handcrafted_dim']}")
    lines.append(f"- Final vector: {manifest['final_dim']}")
    lines.append("")
    lines.append("Nguong certainty cho CUB attributes: " + str(manifest["certainty_threshold"]))
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
) -> None:
    print("\n" + "=" * 60)
    print("BUOC 6: Xay dung bo thuoc tinh nhan dien (Yeu cau 2)")
    print("=" * 60)
    features_dir = os.path.join(output_dir, "features")
    reports_dir = os.path.join(output_dir, "reports")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    cub_312_df, attr_id_to_col = build_cub_312_binary_attributes(metadata_df, attr_labels_df, attr_names_df, certainty_threshold)
    cub_312_path = os.path.join(features_dir, "tier1_cub312_binary.csv")
    cub_312_df.reset_index().to_csv(cub_312_path, index=False, encoding="utf-8")

    selected_groups = select_similarity_attribute_ids(attr_names_df)
    selected_attr_ids = sorted({attr_id for ids in selected_groups.values() for attr_id in ids})
    selected_cols = [attr_id_to_col[attr_id] for attr_id in selected_attr_ids if attr_id in attr_id_to_col]
    similarity_df = cub_312_df[selected_cols].copy() if selected_cols else pd.DataFrame(index=cub_312_df.index)
    similarity_path = os.path.join(features_dir, "tier1_similarity_attributes.csv")
    similarity_df.reset_index().to_csv(similarity_path, index=False, encoding="utf-8")

    group_report = {
        group_name: {
            "count": len(attr_ids),
            "attr_ids": attr_ids,
            "target_attributes": _similarity_attribute_spec()[group_name]["display_attributes"],
            "why_selected": _similarity_attribute_spec()[group_name]["reason"],
            "attr_names": attr_names_df[attr_names_df["attr_id"].isin(attr_ids)]["attr_name"].tolist(),
        }
        for group_name, attr_ids in selected_groups.items()
    }
    group_report_path = os.path.join(reports_dir, "tier1_similarity_groups.json")
    with open(group_report_path, "w", encoding="utf-8") as f:
        json.dump(group_report, f, indent=2, ensure_ascii=False)

    cnn_matrix, cnn_img_ids, cnn_filenames = extract_resnet50_embeddings(metadata_df, images_dir, require_torch_for_cnn)
    cnn_path = os.path.join(features_dir, "tier2_resnet50_embeddings.npy")
    np.save(cnn_path, cnn_matrix)
    cnn_index_path = os.path.join(features_dir, "tier2_resnet50_index.csv")
    pd.DataFrame({"img_id": cnn_img_ids, "filename": cnn_filenames}).to_csv(cnn_index_path, index=False)

    handcrafted_df = extract_handcrafted_features(metadata_df, images_dir)
    handcrafted_path = os.path.join(features_dir, "tier3_handcrafted_features.csv")
    handcrafted_df.to_csv(handcrafted_path, index=False, encoding="utf-8")

    ordered_img_ids = metadata_df["img_id"].astype(int).tolist()
    cub_matrix = cub_312_df.reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32)
    hc_feature_cols = [col for col in handcrafted_df.columns if col.startswith("hc_")]
    handcrafted_matrix = handcrafted_df.set_index("img_id")[hc_feature_cols].reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32) if hc_feature_cols else np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    if cnn_matrix.size:
        cnn_aligned = pd.DataFrame(cnn_matrix, index=cnn_img_ids).reindex(ordered_img_ids).fillna(0).to_numpy(dtype=np.float32)
    else:
        cnn_aligned = np.zeros((len(ordered_img_ids), 0), dtype=np.float32)
    final_matrix = np.concatenate([cub_matrix, cnn_aligned, handcrafted_matrix], axis=1)
    final_path = os.path.join(features_dir, "recognition_features_all.npz")
    np.savez_compressed(
        final_path,
        img_ids=np.array(ordered_img_ids, dtype=np.int32),
        class_ids=metadata_df["class_id"].astype(int).to_numpy(dtype=np.int32),
        cub_312=cub_matrix,
        resnet50=cnn_aligned,
        handcrafted=handcrafted_matrix,
        all_features=final_matrix,
    )
    manifest = {
        "total_images": int(len(metadata_df)),
        "tier1_cub312_dim": int(cub_matrix.shape[1]),
        "tier1_similarity_dim": int(len(selected_cols)),
        "tier2_resnet50_dim": int(cnn_aligned.shape[1]),
        "tier3_handcrafted_dim": int(handcrafted_matrix.shape[1]),
        "final_dim": int(final_matrix.shape[1]),
        "certainty_threshold": int(certainty_threshold),
    }
    with open(os.path.join(reports_dir, "recognition_features_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_requirement2_report(reports_dir, selected_groups, attr_names_df, manifest)
