import os

import pandas as pd


def load_all_metadata(cub_root: str) -> dict:
    print("=" * 60)
    print("BUOC 1: Doc metadata files")
    print("=" * 60)

    images_df = pd.read_csv(
        os.path.join(cub_root, "images.txt"),
        sep=" ",
        names=["img_id", "filepath"],
    )
    print(f"  Tong so anh: {len(images_df)}")

    labels_df = pd.read_csv(
        os.path.join(cub_root, "image_class_labels.txt"),
        sep=" ",
        names=["img_id", "class_id"],
    )

    classes_df = pd.read_csv(
        os.path.join(cub_root, "classes.txt"),
        sep=" ",
        names=["class_id", "class_name"],
    )
    print(f"  Tong so loai: {len(classes_df)}")

    bbox_df = pd.read_csv(
        os.path.join(cub_root, "bounding_boxes.txt"),
        sep=" ",
        names=["img_id", "x", "y", "width", "height"],
    )

    split_df = pd.read_csv(
        os.path.join(cub_root, "train_test_split.txt"),
        sep=" ",
        names=["img_id", "is_train"],
    )

    attr_file_candidates = [
        os.path.join(cub_root, "attributes", "attributes.txt"),
        os.path.join(cub_root, "attributes.txt"),
        "attributes.txt",
    ]
    attr_file_path = None
    for candidate in attr_file_candidates:
        if os.path.exists(candidate):
            attr_file_path = candidate
            break
    if attr_file_path is None:
        raise FileNotFoundError(
            "Khong tim thay attributes.txt. Da tim tai: " + ", ".join(attr_file_candidates)
        )

    attr_names_df = pd.read_csv(
        attr_file_path,
        sep=" ",
        names=["attr_id", "attr_name"],
    )
    print(f"  Tong so attributes: {len(attr_names_df)}")

    print("  Dang doc image_attribute_labels.txt (file lon, cho chut)...")
    attr_labels_df = pd.read_csv(
        os.path.join(cub_root, "attributes", "image_attribute_labels.txt"),
        sep=" ",
        names=["img_id", "attr_id", "is_present", "certainty_id", "time"],
        usecols=["img_id", "attr_id", "is_present", "certainty_id"],
    )
    print(f"  Tong so dong attribute labels: {len(attr_labels_df):,}")

    part_locs_df = pd.read_csv(
        os.path.join(cub_root, "parts", "part_locs.txt"),
        sep=" ",
        names=["img_id", "part_id", "x", "y", "visible"],
    )
    print(f"  Tong so dong part_locs: {len(part_locs_df):,}")

    master_df = (
        images_df.merge(labels_df, on="img_id")
        .merge(classes_df, on="class_id")
        .merge(bbox_df, on="img_id")
        .merge(split_df, on="img_id")
    )
    print(f"\n  Master DataFrame: {len(master_df)} anh")

    return {
        "master": master_df,
        "attr_labels": attr_labels_df,
        "attr_names": attr_names_df,
        "classes": classes_df,
        "part_locs": part_locs_df,
    }


def load_full_species_for_fit(
    cub_root: str,
    filtered_class_ids: set,
) -> pd.DataFrame:
    """Load all original CUB images for the 126 filtered species.

    Used exclusively to provide a larger image pool for fitting unsupervised
    models (PCA, GMM, KMeans, anatomy priors) in Tier-2.  The returned
    DataFrame is *not* written to FAISS or SQLite — only the 511 filtered
    images are indexed for retrieval.

    Args:
        cub_root: Path to the CUB_200_2011 root directory.
        filtered_class_ids: Set of class_ids that survived the perching filter.

    Returns:
        DataFrame with columns [img_id, filename, filepath, class_id,
        class_name, is_train, x, y, width, height] for every original CUB
        image belonging to filtered_class_ids.
    """
    print("\n" + "=" * 60)
    print("BUOC FIT POOL: Doc anh goc cua 126 loai tu CUB_200_2011")
    print("=" * 60)

    images_df = pd.read_csv(
        os.path.join(cub_root, "images.txt"),
        sep=" ",
        names=["img_id", "filepath"],
    )
    labels_df = pd.read_csv(
        os.path.join(cub_root, "image_class_labels.txt"),
        sep=" ",
        names=["img_id", "class_id"],
    )
    classes_df = pd.read_csv(
        os.path.join(cub_root, "classes.txt"),
        sep=" ",
        names=["class_id", "class_name"],
    )
    bbox_df = pd.read_csv(
        os.path.join(cub_root, "bounding_boxes.txt"),
        sep=" ",
        names=["img_id", "x", "y", "width", "height"],
    )
    split_df = pd.read_csv(
        os.path.join(cub_root, "train_test_split.txt"),
        sep=" ",
        names=["img_id", "is_train"],
    )

    merged = (
        images_df
        .merge(labels_df, on="img_id")
        .merge(classes_df, on="class_id")
        .merge(bbox_df, on="img_id")
        .merge(split_df, on="img_id")
    )

    # Keep only images belonging to the 126 filtered species.
    fit_df = merged[merged["class_id"].isin(filtered_class_ids)].copy().reset_index(drop=True)

    # Add a 'filename' column (basename of filepath) so callers that expect
    # metadata_df-style access can use fit_df directly.
    fit_df["filename"] = fit_df["filepath"].apply(os.path.basename)

    print(f"  [OK] Fit pool: {len(fit_df)} anh | {fit_df['class_id'].nunique()} loai")
    return fit_df

