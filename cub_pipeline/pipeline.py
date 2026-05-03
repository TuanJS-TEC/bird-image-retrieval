import os

import pandas as pd

from .config import (
    ALLOW_RELAX_FALLBACK,
    ALLOW_RERUN_FILTERING,
    BUILD_METADATA_DB,
    CUB_ATTR_CERTAINTY_THRESHOLD,
    CUB_ROOT,
    EXTRACT_RECOGNITION_FEATURES,
    FAISS_INDEX_PATH,
    FAISS_USE_COSINE,
    FORCE_REBUILD_TIER1,
    FORCE_REBUILD_TIER2,
    FORCE_REBUILD_TIER3,
    FORCE_REBUILD_FUSION,
    MIN_IMAGES,
    OUTPUT_DIR,
    REQUIRE_TORCH_FOR_CNN,
    REUSE_EXISTING_DB,
    REUSE_EXISTING_FEATURES,
    REUSE_EXISTING_FILTERED_DATASET,
    SQLITE_DB_PATH,
    TARGET_SIZE,
    USE_FULL_SPECIES_FOR_FIT,
)
from .features import build_recognition_feature_package
from .filtering import compute_perching_score, find_perching_attribute_ids, visualize_bbox_distribution
from .gpu_backend import configure_cuda_library_path
from .metadata import load_all_metadata, load_full_species_for_fit
from .processing import process_and_save_dataset
from .retrieval_db import build_metadata_sqlite_and_faiss
from .reports import generate_dataset_report, print_final_summary, verify_output_sample


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _try_load_existing_filtered_dataset(output_dir: str) -> pd.DataFrame | None:
    """Neu metadata.csv + images/ hop le, tra ve DataFrame; nguoc lai None de chay loc lai."""
    meta_path = os.path.join(output_dir, "metadata.csv")
    images_dir = os.path.join(output_dir, "images")
    if not os.path.isfile(meta_path) or not os.path.isdir(images_dir):
        return None
    try:
        df = pd.read_csv(meta_path)
    except Exception:
        return None
    if df.empty or "filename" not in df.columns:
        return None
    n = len(df)
    check_idx = list(range(min(30, n)))
    if n > 30:
        step = max(1, n // 20)
        check_idx = sorted(set(check_idx + list(range(0, n, step))[:20]))
    missing = 0
    for i in check_idx:
        fn = str(df.iloc[i]["filename"])
        if not os.path.isfile(os.path.join(images_dir, fn)):
            missing += 1
    if missing > max(2, len(check_idx) // 5):
        print(
            f"  [WARN] Reuse bo loc: {missing}/{len(check_idx)} file kiem tra khong ton tai trong images/. "
            "Se chay lai buoc loc."
        )
        return None
    return df


def main() -> None:
    configure_cuda_library_path()
    print("\n" + "BIRD " * 10)
    print("  CHUAN BI DATASET CUB-200-2011 - YEU CAU 1")
    print("BIRD " * 10 + "\n")

    if not os.path.exists(CUB_ROOT):
        print(f"[LOI] Khong tim thay thu muc dataset: {CUB_ROOT}")
        print(
            """
                Huong dan:
                1. Tai CUB_200_2011 va giai nen
                2. Dat thu muc CUB_200_2011/ canh file script nay
                3. Chay lai script
            """
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reuse_filtered = _env_flag("BIRD_REUSE_FILTERED_DATASET", bool(REUSE_EXISTING_FILTERED_DATASET))

    data = load_all_metadata(CUB_ROOT)
    master_df = data["master"]
    attr_labels = data["attr_labels"]
    attr_names = data["attr_names"]
    part_locs = data["part_locs"]

    metadata_df: pd.DataFrame | None = None
    if reuse_filtered:
        loaded = _try_load_existing_filtered_dataset(OUTPUT_DIR)
        if loaded is not None:
            print("\n" + "=" * 60)
            print("BO QUA LOC ANH (dung metadata.csv + images/ tai OUTPUT_DIR)")
            print("=" * 60)
            print(f"  [INFO] BIRD_REUSE_FILTERED_DATASET / REUSE_EXISTING_FILTERED_DATASET -> tai {len(loaded)} dong.")
            metadata_df = loaded
        else:
            allow_rerun = _env_flag("BIRD_ALLOW_RERUN_FILTERING", bool(ALLOW_RERUN_FILTERING))
            if not allow_rerun:
                print("\n" + "=" * 60)
                print("[LOI] Khong tai duoc bo loc san tai OUTPUT_DIR (metadata.csv + images/).")
                print("=" * 60)
                print("  Pipeline da dung de KHONG tu dong chay lai buoc loc.")
                print("  Hay kiem tra duong dan OUTPUT_DIR, file anh con day du, hoac:")
                print("    - Trong config: ALLOW_RERUN_FILTERING = True")
                print("    - Hoac env: export BIRD_ALLOW_RERUN_FILTERING=1")
                print("    - Hoac tat reuse: REUSE_EXISTING_FILTERED_DATASET = False (hoac BIRD_REUSE_FILTERED_DATASET=0)")
                return
            print("\n  [INFO] Reuse khong hop le nhung ALLOW_RERUN_FILTERING -> chay day du buoc loc anh.")

    if metadata_df is None:
        perching_attr_ids = find_perching_attribute_ids(attr_names)
        master_df = compute_perching_score(master_df, attr_labels, perching_attr_ids, part_locs)
        visualize_bbox_distribution(master_df, OUTPUT_DIR)

        metadata_df = process_and_save_dataset(
            master_df,
            CUB_ROOT,
            OUTPUT_DIR,
            target_size=TARGET_SIZE,
            min_images=MIN_IMAGES,
            part_locs_df=part_locs,
            allow_relax_fallback=ALLOW_RELAX_FALLBACK,
        )

    # ── Feature fitting pool ─────────────────────────────────────────────────
    # Load all original CUB images belonging to the 126 filtered species.
    # These are used to fit unsupervised models (PCA, GMM, KMeans, anatomy
    # priors) so they generalise better across the species.
    # The 511 filtered images remain the only ones stored in FAISS + SQLite.
    fit_metadata_df = None
    fit_images_dir = None
    if USE_FULL_SPECIES_FOR_FIT:
        filtered_class_ids = set(metadata_df["class_id"].unique())
        fit_metadata_df = load_full_species_for_fit(CUB_ROOT, filtered_class_ids)
        fit_images_dir = os.path.join(CUB_ROOT, "images")

    if EXTRACT_RECOGNITION_FEATURES:
        # In trang thai checkpoint de nguoi dung biet buoc nao se chay lai.
        if REUSE_EXISTING_FEATURES:
            _skips = []
            _rebuilds = []
            for name, force in [
                ("Tier1", FORCE_REBUILD_TIER1),
                ("Tier2", FORCE_REBUILD_TIER2),
                ("Tier3", FORCE_REBUILD_TIER3),
                ("Fusion", FORCE_REBUILD_FUSION),
            ]:
                (_rebuilds if force else _skips).append(name)
            print("\n  [CHECKPOINT] REUSE_EXISTING_FEATURES=True")
            if _skips:
                print(f"    -> Co the bo qua (neu file da co): {', '.join(_skips)}")
            if _rebuilds:
                print(f"    -> Bat buoc tinh lai (FORCE_REBUILD=True): {', '.join(_rebuilds)}")
            if REUSE_EXISTING_DB:
                print("    -> SQLite/FAISS: giu lai neu features khong thay doi.")
        else:
            print("\n  [CHECKPOINT] REUSE_EXISTING_FEATURES=False -> tinh lai toan bo features.")

        features_rebuilt = build_recognition_feature_package(
            metadata_df=metadata_df,
            attr_labels_df=attr_labels,
            attr_names_df=attr_names,
            output_dir=OUTPUT_DIR,
            certainty_threshold=CUB_ATTR_CERTAINTY_THRESHOLD,
            require_torch_for_cnn=REQUIRE_TORCH_FOR_CNN,
            cub_root=CUB_ROOT,
            fit_metadata_df=fit_metadata_df,
            fit_images_dir=fit_images_dir,
        )
        if BUILD_METADATA_DB:
            print("\n" + "=" * 60)
            print("BUOC 7: Tao he CSDL sieu du lieu (SQLite + FAISS)")
            print("=" * 60)
            summary = build_metadata_sqlite_and_faiss(
                output_dir=OUTPUT_DIR,
                sqlite_path=SQLITE_DB_PATH,
                faiss_path=FAISS_INDEX_PATH,
                use_cosine=FAISS_USE_COSINE,
                features_rebuilt=features_rebuilt,
            )
            print(f"  [OK] SQLite DB: {summary['sqlite_path']}")
            print(f"  [OK] FAISS index: {summary['faiss_path']}")
            print(f"  [OK] Tong vectors: {summary['total_vectors']} | dim={summary['dimension']}")
            print(f"  [OK] So dong metadata: {summary['inserted_rows']} | metric={summary['metric']}")

    generate_dataset_report(metadata_df, OUTPUT_DIR)
    verify_output_sample(metadata_df, OUTPUT_DIR)
    print_final_summary(metadata_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
