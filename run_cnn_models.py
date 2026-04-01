import argparse
import os

import numpy as np
import pandas as pd

from cub_pipeline.features import extract_resnet50_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Khoi tao va chay ResNet50 embedding tren dataset da xu ly.")
    parser.add_argument("--output-dir", default="./dataset_processed", help="Thu muc output cua pipeline.")
    parser.add_argument("--limit", type=int, default=0, help="So anh toi da de chay test nhanh (0 = tat ca).")
    parser.add_argument(
        "--allow-missing-torch",
        action="store_true",
        help="Cho phep bo qua neu thieu torch/torchvision thay vi dung chuong trinh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    metadata_path = os.path.join(output_dir, "metadata.csv")
    images_dir = os.path.join(output_dir, "images")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Khong tim thay metadata: {metadata_path}. Hay chay build_cub_perched_side_dataset.py truoc.")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {images_dir}.")

    metadata_df = pd.read_csv(metadata_path)
    if args.limit and args.limit > 0:
        metadata_df = metadata_df.head(args.limit).copy()

    print(f"[INFO] So anh dua vao embedding: {len(metadata_df)}")
    cnn_matrix, cnn_img_ids, cnn_filenames = extract_resnet50_embeddings(
        metadata_df=metadata_df,
        images_dir=images_dir,
        require_torch=not args.allow_missing_torch,
    )

    embedding_path = os.path.join(features_dir, "tier2_resnet50_embeddings.npy")
    index_path = os.path.join(features_dir, "tier2_resnet50_index.csv")
    np.save(embedding_path, cnn_matrix)
    pd.DataFrame({"img_id": cnn_img_ids, "filename": cnn_filenames}).to_csv(index_path, index=False, encoding="utf-8")

    print(f"[OK] Da luu embedding: {embedding_path}")
    print(f"[OK] Da luu index: {index_path}")
    if cnn_matrix.size:
        print(f"[OK] Kich thuoc embedding: {cnn_matrix.shape}")
    else:
        print("[WARN] Khong tao duoc embedding nao.")


if __name__ == "__main__":
    main()
