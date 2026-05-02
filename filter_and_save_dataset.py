#!/usr/bin/env python3
"""
Loc anh (tu the dau / perching), crop bbox, chuan hoa huong nhin, luu JPEG + metadata.csv.
Cung luong xu ly voi buoc dau cua cub_pipeline.pipeline.main(), khong chay feature / FAISS.

Vi du:
  python3 filter_and_save_dataset.py \\
    --cub-root ./CUB_200_2011/CUB_200_2011 \\
    --output-dir ./dataset_process
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Loc anh CUB (perching), luu anh da xu ly va metadata.csv vao output-dir."
    )
    p.add_argument(
        "--cub-root",
        default=os.environ.get("CUB_ROOT", "./CUB_200_2011/CUB_200_2011"),
        help="Thu muc goc CUB_200_2011 (chua images/, attributes/, ...). Mac dinh: ./CUB_200_2011/CUB_200_2011 hoac bien CUB_ROOT.",
    )
    p.add_argument(
        "--output-dir",
        default="./dataset_process",
        help="Thu muc output: images/ + metadata.csv. Mac dinh: ./dataset_process",
    )
    p.add_argument("--target-size", type=int, default=224, help="Canh crop resize (vuong). Mac dinh: 224")
    p.add_argument(
        "--min-images",
        type=int,
        default=500,
        help="So anh toi thieu sau loc; neu it hon va --allow-relax-fallback thi pipeline noi long.",
    )
    p.add_argument(
        "--allow-relax-fallback",
        action="store_true",
        help="Neu so anh sau loc < min-images, noi long theo bbox_ratio (giong ALLOW_RELAX_FALLBACK trong config).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cub_root = os.path.abspath(args.cub_root)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(cub_root):
        print(f"[LOI] Khong tim thay CUB root: {cub_root}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    from cub_pipeline.filtering import compute_perching_score, find_perching_attribute_ids, visualize_bbox_distribution
    from cub_pipeline.metadata import load_all_metadata
    from cub_pipeline.processing import process_and_save_dataset

    data = load_all_metadata(cub_root)
    master_df = data["master"]
    attr_labels = data["attr_labels"]
    attr_names = data["attr_names"]
    part_locs = data["part_locs"]

    perching_attr_ids = find_perching_attribute_ids(attr_names)
    master_df = compute_perching_score(master_df, attr_labels, perching_attr_ids, part_locs)
    visualize_bbox_distribution(master_df, output_dir)

    target_size = (int(args.target_size), int(args.target_size))
    metadata_df = process_and_save_dataset(
        master_df,
        cub_root,
        output_dir,
        target_size=target_size,
        min_images=int(args.min_images),
        part_locs_df=part_locs,
        allow_relax_fallback=bool(args.allow_relax_fallback),
    )

    print("\n[OK] Hoan tat loc va luu.")
    print(f"  - Anh: {os.path.join(output_dir, 'images')}")
    print(f"  - Metadata: {os.path.join(output_dir, 'metadata.csv')}")
    print(f"  - So dong: {len(metadata_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
