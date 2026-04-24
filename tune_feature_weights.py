import argparse
import json
import os
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


def _zscore_per_column(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32, copy=False)
    out = mat.astype(np.float32, copy=True)
    mu = np.mean(out, axis=0, keepdims=True)
    sigma = np.std(out, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return ((out - mu) / sigma).astype(np.float32)


def _apply_pca(mat: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
    if mat.size == 0:
        return mat.astype(np.float32, copy=False), 1.0
    n, d = mat.shape
    n_comp = max(1, min(int(n_components), n, d))
    if n_comp >= d:
        return mat.astype(np.float32, copy=False), 1.0
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(mat).astype(np.float32)
    return reduced, float(np.sum(pca.explained_variance_ratio_))


def _apply_hu_boost(custom_mat: np.ndarray, attr_names: np.ndarray | None, hu_weight: float) -> np.ndarray:
    if custom_mat.size == 0 or hu_weight <= 1.0:
        return custom_mat.astype(np.float32, copy=False)
    if attr_names is None:
        return custom_mat.astype(np.float32, copy=False)
    names = [str(x) for x in attr_names.tolist()]
    hu_cols = {"cattr_hu1", "cattr_hu2", "cattr_hu3"}
    hu_idx = [idx for idx, name in enumerate(names) if name in hu_cols]
    if not hu_idx:
        return custom_mat.astype(np.float32, copy=False)
    out = custom_mat.astype(np.float32, copy=True)
    out[:, hu_idx] *= float(hu_weight)
    return out


def _average_precision_at_all(sorted_labels: np.ndarray, query_label: int) -> float | None:
    relevant = sorted_labels == query_label
    total_rel = int(relevant.sum())
    if total_rel <= 0:
        return None
    hit_count = 0
    precision_sum = 0.0
    for idx, is_rel in enumerate(relevant, start=1):
        if not bool(is_rel):
            continue
        hit_count += 1
        precision_sum += hit_count / float(idx)
    return precision_sum / float(total_rel)


def evaluate_retrieval(features: np.ndarray, class_ids: np.ndarray, ks: Iterable[int]) -> dict[str, float]:
    n = features.shape[0]
    sims = features @ features.T
    np.fill_diagonal(sims, -1.0)
    metrics: dict[str, float] = {}
    ks = sorted(set(int(k) for k in ks))
    recall_hits = {k: 0 for k in ks}
    ap_vals: list[float] = []
    for i in range(n):
        order = np.argsort(-sims[i])
        ranked_labels = class_ids[order]
        for k in ks:
            if np.any(ranked_labels[:k] == class_ids[i]):
                recall_hits[k] += 1
        ap = _average_precision_at_all(ranked_labels, int(class_ids[i]))
        if ap is not None:
            ap_vals.append(float(ap))
    for k in ks:
        metrics[f"recall@{k}"] = float(recall_hits[k] / max(n, 1))
    metrics["mAP"] = float(np.mean(ap_vals)) if ap_vals else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune retrieval compaction (Tier-1 weight + Tier-2 PCA) by Recall@K and mAP.")
    parser.add_argument(
        "--npz-path",
        default="./dataset_processed/features/recognition_features_all.npz",
        help="Path to recognition_features_all.npz",
    )
    parser.add_argument("--tier1-min", type=float, default=2.0, help="Min Tier-1 weight.")
    parser.add_argument("--tier1-max", type=float, default=4.0, help="Max Tier-1 weight.")
    parser.add_argument("--tier1-step", type=float, default=0.5, help="Tier-1 step.")
    parser.add_argument("--tier2-weight", type=float, default=1.0, help="Fixed Tier-2 weight.")
    parser.add_argument("--tier3-weight", type=float, default=1.0, help="Fixed Tier-3 weight.")
    parser.add_argument(
        "--tier2-pca-candidates",
        default="64,96,128",
        help="Comma-separated Tier-2 PCA dimensions to compare.",
    )
    parser.add_argument(
        "--hu-weight-candidates",
        default="1.0,1.25,1.5,1.75",
        help="Comma-separated Hu moment weights for cattr_hu1/2/3.",
    )
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated Recall@K list.")
    parser.add_argument(
        "--output-json",
        default="./dataset_processed/reports/feature_weight_tuning.json",
        help="Output report path.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"Missing file: {args.npz_path}")
    bundle = np.load(args.npz_path)
    class_ids = bundle["class_ids"].astype(np.int32)
    custom = (
        bundle["custom_attributes_retrieval"].astype(np.float32)
        if "custom_attributes_retrieval" in bundle
        else bundle["custom_attributes"].astype(np.float32)
    )
    tier2 = bundle["algorithmic_tier2"].astype(np.float32)
    tier3 = bundle["handcrafted"].astype(np.float32)
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    pca_dims = [int(x.strip()) for x in args.tier2_pca_candidates.split(",") if x.strip()]
    hu_weights = [float(x.strip()) for x in args.hu_weight_candidates.split(",") if x.strip()]
    attr_names = bundle["custom_attr_names_retrieval"] if "custom_attr_names_retrieval" in bundle else None
    if attr_names is None:
        csv_path = os.path.join(os.path.dirname(args.npz_path), "tier1_custom_attributes.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                raw_cols = [c for c in header if c.startswith("cattr_")]
                excluded = {"cattr_fg_aspect_ratio", "cattr_centroid_x", "cattr_centroid_y"}
                filtered = [c for c in raw_cols if c not in excluded]
                attr_names = np.array(filtered, dtype=object)
            except Exception:
                attr_names = None

    custom_n = _zscore_per_column(custom)
    tier2_n = _zscore_per_column(tier2)
    tier3_n = _zscore_per_column(tier3)

    baseline = _l2_normalize_rows(np.concatenate([custom_n, tier2_n, tier3_n], axis=1))
    baseline_metrics = evaluate_retrieval(baseline, class_ids, ks)

    runs: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for pca_dim in pca_dims:
        tier2_reduced, explained_ratio = _apply_pca(tier2_n, pca_dim)
        for hu_w in hu_weights:
            custom_hu = _apply_hu_boost(custom_n, attr_names, hu_w)
            w = args.tier1_min
            while w <= args.tier1_max + 1e-9:
                fused = np.concatenate(
                    [
                        custom_hu * float(w),
                        tier2_reduced * float(args.tier2_weight),
                        tier3_n * float(args.tier3_weight),
                    ],
                    axis=1,
                )
                fused = _l2_normalize_rows(fused)
                metrics = evaluate_retrieval(fused, class_ids, ks)
                run = {
                    "tier1_weight": float(round(w, 4)),
                    "tier2_weight": float(args.tier2_weight),
                    "tier3_weight": float(args.tier3_weight),
                    "tier2_pca_dim": int(pca_dim),
                    "tier2_pca_explained_variance_ratio": float(explained_ratio),
                    "hu_moment_weight": float(hu_w),
                    **metrics,
                }
                runs.append(run)
                if best is None or run["mAP"] > best["mAP"]:
                    best = run
                w += args.tier1_step

    report = {
        "npz_path": args.npz_path,
        "num_samples": int(class_ids.shape[0]),
        "baseline_equal_weights": {
            "tier1_weight": 1.0,
            "tier2_weight": 1.0,
            "tier3_weight": 1.0,
            **baseline_metrics,
        },
        "search_space": {
            "tier1_min": float(args.tier1_min),
            "tier1_max": float(args.tier1_max),
            "tier1_step": float(args.tier1_step),
            "tier2_weight": float(args.tier2_weight),
            "tier3_weight": float(args.tier3_weight),
            "tier2_pca_candidates": pca_dims,
            "hu_weight_candidates": hu_weights,
        },
        "best_by_map": best,
        "all_runs": runs,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[OK] Saved tuning report:", args.output_json)
    print("[INFO] Baseline:", json.dumps(report["baseline_equal_weights"], ensure_ascii=False))
    print("[INFO] Best:", json.dumps(report["best_by_map"], ensure_ascii=False))


if __name__ == "__main__":
    main()
