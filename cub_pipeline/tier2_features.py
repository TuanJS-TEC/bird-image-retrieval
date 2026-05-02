import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from .algorithmic_tier2 import extract_algorithmic_embedding, fit_algorithmic_tier2, load_algorithmic_artifacts
from .common import parallel_tier_extraction_workers

_MAIN_T2_ART: Any = None
_MAIN_T2_TREE: Any = None


def _main_tier2_worker_init(features_dir: str) -> None:
    global _MAIN_T2_ART, _MAIN_T2_TREE
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    _MAIN_T2_ART = load_algorithmic_artifacts(features_dir)
    _MAIN_T2_TREE = KDTree(_MAIN_T2_ART.acv_vocab.astype(np.float32))


def _main_tier2_worker_task(payload: tuple[int, str, str]) -> tuple[int, str, np.ndarray | None, np.ndarray | None]:
    img_id, filename, images_dir = payload
    path = os.path.join(images_dir, filename)
    if _MAIN_T2_ART is None or _MAIN_T2_TREE is None or not os.path.exists(path):
        return img_id, filename, None, None
    try:
        vec, anatomy = extract_algorithmic_embedding(path, _MAIN_T2_ART, acv_tree=_MAIN_T2_TREE)
        return img_id, filename, vec, anatomy
    except Exception:
        return img_id, filename, None, None


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
    acv_tree = KDTree(artifacts.acv_vocab.astype(np.float32))
    tasks = [(int(row["img_id"]), str(row["filename"]), images_dir) for _, row in metadata_df.iterrows()]
    workers = parallel_tier_extraction_workers()

    if workers <= 1 or len(tasks) < 6:
        vectors, img_ids, filenames, anatomy_blocks = [], [], [], []
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="  Tang 2 - Algorithmic 512D"):
            filename = str(row["filename"])
            img_id = int(row["img_id"])
            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                continue
            try:
                vec, anatomy = extract_algorithmic_embedding(img_path, artifacts, acv_tree=acv_tree)
                vectors.append(vec)
                anatomy_blocks.append(anatomy)
                img_ids.append(img_id)
                filenames.append(filename)
            except Exception as ex:
                print(f"  [WARN] Bo qua algorithmic embedding {filename}: {ex}")
        matrix = np.vstack(vectors) if vectors else np.zeros((0, 512), dtype=np.float32)
        anatomy_matrix = np.stack(anatomy_blocks, axis=0) if anatomy_blocks else np.zeros((0, 5, 184), dtype=np.float32)
        return matrix, img_ids, filenames, anatomy_matrix

    n_workers = max(1, min(workers, len(tasks)))
    mp_ctx = mp.get_context("spawn")
    by_id: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp_ctx, initializer=_main_tier2_worker_init, initargs=(features_dir,)
    ) as ex:
        futures = [ex.submit(_main_tier2_worker_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Tang 2 - Algorithmic 512D"):
            img_id, filename, vec, anatomy = fut.result()
            if vec is not None and anatomy is not None:
                by_id[int(img_id)] = (str(filename), vec, anatomy)

    img_ids: list[int] = []
    filenames: list[str] = []
    vectors: list[np.ndarray] = []
    anatomy_blocks: list[np.ndarray] = []
    for _, row in metadata_df.iterrows():
        iid = int(row["img_id"])
        if iid not in by_id:
            continue
        fn, vec, anatomy = by_id[iid]
        img_ids.append(iid)
        filenames.append(fn)
        vectors.append(vec)
        anatomy_blocks.append(anatomy)
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 512), dtype=np.float32)
    anatomy_matrix = np.stack(anatomy_blocks, axis=0) if anatomy_blocks else np.zeros((0, 5, 184), dtype=np.float32)
    return matrix, img_ids, filenames, anatomy_matrix

