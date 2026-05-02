import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common import _debug_log


def crop_and_resize(img: Image.Image, bbox: tuple, target_size: tuple = (224, 224)) -> Image.Image:
    x, y, w, h = bbox
    img_w, img_h = img.size
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(img_w, int(x + w))
    bottom = min(img_h, int(y + h))
    if right <= left or bottom <= top:
        return img.resize(target_size, Image.Resampling.LANCZOS)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def process_and_save_dataset(
    master_df: pd.DataFrame,
    cub_root: str,
    output_dir: str,
    target_size: tuple = (224, 224),
    min_images: int = 500,
    part_locs_df: pd.DataFrame | None = None,
    allow_relax_fallback: bool = False,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("BUOC 3: Crop, resize va luu anh")
    print("=" * 60)
    filtered_df = master_df[master_df["likely_perching"]].copy().reset_index(drop=True)
    print(f"  So anh sau loc tu the: {len(filtered_df)}")

    filtered_classes = int(filtered_df["class_id"].nunique()) if len(filtered_df) else 0
    avg_per_class = float(len(filtered_df) / max(filtered_classes, 1))

    if len(filtered_df) < min_images:
        print(f"  [!] Canh bao: {len(filtered_df)} anh < yeu cau {min_images}.")
        print(f"      -> Coverage classes={filtered_classes}, avg/class={avg_per_class:.2f}")
        if allow_relax_fallback:
            print("      -> Tu dong noi long threshold de tang do phu retrieval...")
            filtered_df = master_df.nsmallest(max(min_images, 8000), "bbox_ratio").copy().reset_index(drop=True)
            print(f"      -> Da chon {len(filtered_df)} anh")
        else:
            print("      -> Giu bo loc chat, KHONG noi long de tranh lot anh khong dat.")

    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)
    expected_filenames = {
        f"{int(img_id):05d}_{os.path.basename(filepath)}"
        for img_id, filepath in zip(filtered_df["img_id"], filtered_df["filepath"])
    }
    existing_jpg = [
        name
        for name in os.listdir(images_out_dir)
        if os.path.isfile(os.path.join(images_out_dir, name)) and name.lower().endswith((".jpg", ".jpeg"))
    ]
    stale_filenames = [name for name in existing_jpg if name not in expected_filenames]
    for stale_name in stale_filenames:
        stale_path = os.path.join(images_out_dir, stale_name)
        try:
            os.remove(stale_path)
        except Exception:
            pass
    _debug_log(run_id="debug", hypothesis_id="H", location="cub_pipeline/processing.py", message="sync_done", data={})

    # Standardize bird heading direction using beak/tail landmarks.
    # Convention: output image should face RIGHT (beak_x > tail_x).
    beak_x_map: dict[int, float] = {}
    tail_x_map: dict[int, float] = {}
    class_face_right_majority: dict[int, bool] = {}
    if part_locs_df is not None and len(part_locs_df) > 0:
        beaks = part_locs_df[(part_locs_df["part_id"] == 2) & (part_locs_df["visible"] == 1)][["img_id", "x"]].copy()
        tails = part_locs_df[(part_locs_df["part_id"] == 14) & (part_locs_df["visible"] == 1)][["img_id", "x"]].copy()
        beak_x_map = {int(r["img_id"]): float(r["x"]) for _, r in beaks.iterrows()}
        tail_x_map = {int(r["img_id"]): float(r["x"]) for _, r in tails.iterrows()}

        orient_rows = []
        for _, rr in master_df[["img_id", "class_id"]].drop_duplicates().iterrows():
            img_id = int(rr["img_id"])
            if img_id in beak_x_map and img_id in tail_x_map:
                orient_rows.append(
                    {
                        "class_id": int(rr["class_id"]),
                        "face_right": float(beak_x_map[img_id] > tail_x_map[img_id]),
                    }
                )
        if orient_rows:
            orient_df = pd.DataFrame(orient_rows)
            class_face_right_majority = (
                orient_df.groupby("class_id")["face_right"].mean().map(lambda v: bool(v >= 0.5)).to_dict()
            )

    metadata_records = []
    errors = []
    flipped_count = 0
    inferred_by_landmark = 0
    inferred_by_class_majority = 0
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="  Dang xu ly anh"):
        src_path = os.path.join(cub_root, "images", row["filepath"])
        if not os.path.exists(src_path):
            errors.append(f"Khong tim thay file: {src_path}")
            continue
        try:
            img = Image.open(src_path).convert("RGB")
            original_w, original_h = img.size
            processed_img = crop_and_resize(img, (row["x"], row["y"], row["width"], row["height"]), target_size)

            img_id = int(row["img_id"])
            class_id = int(row["class_id"])
            should_flip = False
            if img_id in beak_x_map and img_id in tail_x_map:
                inferred_by_landmark += 1
                should_flip = not bool(beak_x_map[img_id] > tail_x_map[img_id])
            elif class_id in class_face_right_majority:
                inferred_by_class_majority += 1
                should_flip = not bool(class_face_right_majority[class_id])
            if should_flip:
                processed_img = processed_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                flipped_count += 1

            original_filename = os.path.basename(row["filepath"])
            out_filename = f"{int(row['img_id']):05d}_{original_filename}"
            out_path = os.path.join(images_out_dir, out_filename)
            processed_img.save(out_path, "JPEG", quality=95)
            metadata_records.append(
                {
                    "img_id": int(row["img_id"]),
                    "filename": out_filename,
                    "original_path": row["filepath"],
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "is_train": int(row["is_train"]),
                    "orig_width": original_w,
                    "orig_height": original_h,
                    "bbox_x": row["x"],
                    "bbox_y": row["y"],
                    "bbox_w": row["width"],
                    "bbox_h": row["height"],
                    "bbox_ratio": round(float(row["bbox_ratio"]), 4),
                    "target_width": target_size[0],
                    "target_height": target_size[1],
                    "hflip_applied": int(should_flip),
                }
            )
        except Exception as e:
            errors.append(f"Loi xu ly {src_path}: {str(e)}")

    if errors:
        print(f"\n  [!] {len(errors)} loi trong qua trinh xu ly:")
        for err in errors[:10]:
            print(f"      {err}")

    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Da xu ly thanh cong: {len(metadata_df)} anh")
    print(
        "  [INFO] Chuan hoa huong nhin: "
        f"flipped={flipped_count}, by_landmark={inferred_by_landmark}, by_class_majority={inferred_by_class_majority}"
    )
    print(f"  [OK] Metadata CSV: {metadata_path}")
    return metadata_df
