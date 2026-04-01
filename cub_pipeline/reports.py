import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def generate_dataset_report(metadata_df: pd.DataFrame, output_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("BUOC 4: Sinh bao cao dataset")
    print("=" * 60)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stats = {
        "total_images": len(metadata_df),
        "total_classes": int(metadata_df["class_id"].nunique()),
        "train_images": int(metadata_df["is_train"].sum()),
        "test_images": int((~metadata_df["is_train"].astype(bool)).sum()),
        "img_per_class_mean": round(float(metadata_df.groupby("class_id").size().mean()), 2),
        "img_per_class_min": int(metadata_df.groupby("class_id").size().min()),
        "img_per_class_max": int(metadata_df.groupby("class_id").size().max()),
        "target_size": "224x224",
        "format": "JPEG",
    }
    with open(os.path.join(reports_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    imgs_per_class = metadata_df.groupby("class_name").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(imgs_per_class)), imgs_per_class.values, color="steelblue", alpha=0.8)
    ax.axhline(y=imgs_per_class.mean(), color="red", linestyle="--", label=f"Trung binh = {imgs_per_class.mean():.1f} anh/loai")
    ax.set_xlabel("Loai (sap xep giam dan theo so anh)", fontsize=12)
    ax.set_ylabel("So luong anh", fontsize=12)
    ax.set_title(f"Phan phoi so anh theo loai trong dataset da xu ly\nTong: {len(metadata_df)} anh | {metadata_df['class_id'].nunique()} loai", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "distribution_per_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return stats


def verify_output_sample(metadata_df: pd.DataFrame, output_dir: str, n_samples: int = 5) -> None:
    print("\n" + "=" * 60)
    print("BUOC 5: Kiem tra xac nhan anh output")
    print("=" * 60)
    images_dir = os.path.join(output_dir, "images")
    sample = metadata_df.sample(min(n_samples, len(metadata_df)), random_state=42)
    fig, axes = plt.subplots(1, len(sample), figsize=(3 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sample.iterrows()):
        img_path = os.path.join(images_dir, row["filename"])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(row["class_name"].replace("_", " ").split(".")[-1], fontsize=9)
        ax.axis("off")
    plt.suptitle("Anh mau sau xu ly (224x224 JPEG)", fontsize=12, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "reports", "sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_final_summary(metadata_df: pd.DataFrame, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("HOAN THANH YEU CAU 1")
    print("=" * 60)
    print(f"  Thu muc output: {output_dir}/")
    print(f"  Tong so anh: {len(metadata_df):,}")
    print(f"  So loai: {metadata_df['class_id'].nunique()}")
