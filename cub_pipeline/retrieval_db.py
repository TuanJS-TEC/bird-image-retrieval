import json
import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd


def _load_faiss():
    try:
        import faiss  # type: ignore
    except Exception as ex:
        raise RuntimeError(
            "Khong the import faiss. Hay cai dat bang lenh: pip install faiss-cpu"
        ) from ex
    return faiss


def _normalize_if_needed(matrix: np.ndarray, use_cosine: bool, faiss_module: Any) -> np.ndarray:
    vectors = matrix.astype(np.float32, copy=True)
    if use_cosine:
        faiss_module.normalize_L2(vectors)
    return vectors


def build_metadata_sqlite_and_faiss(
    output_dir: str,
    sqlite_path: str = "birds.db",
    faiss_path: str = "birds.faiss",
    use_cosine: bool = True,
) -> dict[str, Any]:
    faiss = _load_faiss()
    metadata_path = os.path.join(output_dir, "metadata.csv")
    cub_attr_path = os.path.join(output_dir, "features", "tier1_cub312_binary.csv")
    emb_path = os.path.join(output_dir, "features", "tier2_resnet50_embeddings.npy")
    emb_index_path = os.path.join(output_dir, "features", "tier2_resnet50_index.csv")
    vectors_dir = os.path.join(output_dir, "features", "resnet50_vectors")
    os.makedirs(vectors_dir, exist_ok=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Khong tim thay file metadata: {metadata_path}")
    if not os.path.exists(cub_attr_path):
        raise FileNotFoundError(f"Khong tim thay file attributes: {cub_attr_path}")
    if not os.path.exists(emb_path) or not os.path.exists(emb_index_path):
        raise FileNotFoundError("Khong tim thay tier2 embeddings. Hay chay pipeline feature truoc.")

    metadata_df = pd.read_csv(metadata_path)
    cub_attr_df = pd.read_csv(cub_attr_path)
    emb_matrix = np.load(emb_path).astype(np.float32)
    emb_index_df = pd.read_csv(emb_index_path)

    attr_cols = [c for c in cub_attr_df.columns if c != "img_id"]
    if len(attr_cols) != 312:
        print(f"[WARN] So thuoc tinh tier1 hien tai = {len(attr_cols)} (khong phai 312).")

    attr_map = cub_attr_df.set_index("img_id")
    emb_index_df["img_id"] = emb_index_df["img_id"].astype(int)
    emb_index_df["faiss_idx"] = np.arange(len(emb_index_df), dtype=np.int32)
    emb_map = emb_index_df.set_index("img_id")
    metadata_df["img_id"] = metadata_df["img_id"].astype(int)

    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            species_id INTEGER NOT NULL,
            species_name TEXT NOT NULL,
            attributes TEXT NOT NULL,
            feature_path TEXT NOT NULL,
            faiss_idx INTEGER UNIQUE NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM images")

    inserted = 0
    for row_idx, row in metadata_df.iterrows():
        img_id = int(row["img_id"])
        if img_id not in attr_map.index or img_id not in emb_map.index:
            continue
        attr_values = attr_map.loc[img_id, attr_cols].astype(np.uint8).tolist()
        attrs_json = json.dumps(attr_values, ensure_ascii=False)

        faiss_idx = int(emb_map.loc[img_id, "faiss_idx"])
        vector = emb_matrix[faiss_idx].astype(np.float32)
        vector_file = os.path.join(vectors_dir, f"{img_id:05d}.npy")
        np.save(vector_file, vector)

        conn.execute(
            """
            INSERT INTO images (id, filename, species_id, species_name, attributes, feature_path, faiss_idx)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                img_id,
                str(row["filename"]),
                int(row["class_id"]),
                str(row["class_name"]),
                attrs_json,
                vector_file,
                faiss_idx,
            ),
        )
        inserted += 1
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_species_id ON images(species_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_faiss_idx ON images(faiss_idx)")
    conn.commit()
    conn.close()

    vectors = _normalize_if_needed(emb_matrix, use_cosine=use_cosine, faiss_module=faiss)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim) if use_cosine else faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, faiss_path)

    metric = "cosine(IndexFlatIP + normalize_L2)" if use_cosine else "L2(IndexFlatL2)"
    return {
        "total_vectors": int(vectors.shape[0]),
        "dimension": int(dim),
        "inserted_rows": int(inserted),
        "sqlite_path": sqlite_path,
        "faiss_path": faiss_path,
        "metric": metric,
    }


def search_similar_vectors(
    query_feat: np.ndarray,
    top_k: int = 5,
    sqlite_path: str = "birds.db",
    faiss_path: str = "birds.faiss",
    use_cosine: bool = True,
) -> list[dict[str, Any]]:
    faiss = _load_faiss()
    if query_feat.ndim != 1:
        raise ValueError("query_feat phai la vector 1 chieu.")

    index = faiss.read_index(faiss_path)
    query = query_feat.reshape(1, -1).astype(np.float32)
    if use_cosine:
        faiss.normalize_L2(query)
    distances, indices = index.search(query, top_k)

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    results: list[dict[str, Any]] = []
    for score, faiss_idx in zip(distances[0], indices[0]):
        if int(faiss_idx) < 0:
            continue
        row = conn.execute(
            """
            SELECT id, filename, species_id, species_name, feature_path
            FROM images
            WHERE faiss_idx = ?
            """,
            (int(faiss_idx),),
        ).fetchone()
        if row is None:
            continue
        results.append(
            {
                "distance_or_score": float(score),
                "faiss_idx": int(faiss_idx),
                "id": int(row["id"]),
                "filename": str(row["filename"]),
                "species_id": int(row["species_id"]),
                "species_name": str(row["species_name"]),
                "feature_path": str(row["feature_path"]),
            }
        )
    conn.close()
    return results
