import os

from .config import (
    ALLOW_RELAX_FALLBACK,
    BUILD_METADATA_DB,
    CUB_ATTR_CERTAINTY_THRESHOLD,
    CUB_ROOT,
    EXTRACT_RECOGNITION_FEATURES,
    FAISS_INDEX_PATH,
    FAISS_USE_COSINE,
    MIN_IMAGES,
    OUTPUT_DIR,
    REQUIRE_TORCH_FOR_CNN,
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
    data = load_all_metadata(CUB_ROOT)
    master_df = data["master"]
    attr_labels = data["attr_labels"]
    attr_names = data["attr_names"]
    part_locs = data["part_locs"]

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
        build_recognition_feature_package(
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
