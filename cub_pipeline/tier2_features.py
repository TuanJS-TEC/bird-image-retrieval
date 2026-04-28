import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .algorithmic_tier2 import extract_algorithmic_embedding, fit_algorithmic_tier2, load_algorithmic_artifacts


def extract_efficientnetv2_embeddings(
    metadata_df: pd.DataFrame,
    images_dir: str,
    require_torch: bool = True,
    cub_root: str | None = None,
    fit_metadata_df: pd.DataFrame | None = None,
    fit_images_dir: str | None = None,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    # Backward-compatible function name: now computes algorithm-only Tier-2 embedding.
    output_dir = os.path.dirname(images_dir.rstrip("\\/"))
    features_dir = os.path.join(output_dir, "features")
    artifacts_path = os.path.join(features_dir, "tier2_algorithmic_artifacts.pkl")
    if not os.path.exists(artifacts_path):
        if not cub_root:
            raise RuntimeError("Thieu cub_root de fit AG-SFP anatomy priors va cac model algorithmic.")
        print("  [INFO] Dang fit bo model algorithmic Tier 2 (AG-SFP + ACV + HFVE + PPD)...")
        fit_algorithmic_tier2(
            metadata_df=metadata_df,
            images_dir=images_dir,
            cub_root=cub_root,
            features_dir=features_dir,
            fit_metadata_df=fit_metadata_df,
            fit_images_dir=fit_images_dir,
        )
    artifacts = load_algorithmic_artifacts(features_dir)
    vectors, img_ids, filenames, anatomy_blocks = [], [], [], []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 2 - Algorithmic 512D"):
        filename = str(row["filename"])
        img_id = int(row["img_id"])
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue
        try:
            vec, anatomy = extract_algorithmic_embedding(img_path, artifacts)
            vectors.append(vec)
            anatomy_blocks.append(anatomy)
            img_ids.append(img_id)
            filenames.append(filename)
        except Exception as ex:
            print(f"  [WARN] Bo qua algorithmic embedding {filename}: {ex}")
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 512), dtype=np.float32)
    anatomy_matrix = np.stack(anatomy_blocks, axis=0) if anatomy_blocks else np.zeros((0, 5, 184), dtype=np.float32)
    return matrix, img_ids, filenames, anatomy_matrix

