import argparse
import json
import os
import sqlite3

import numpy as np

from cub_pipeline.retrieval_db import build_metadata_sqlite_and_faiss, search_similar_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tao SQLite + FAISS va test tim kiem top-k.")
    parser.add_argument("--output-dir", default="./dataset_processed", help="Thu muc output cua pipeline.")
    parser.add_argument("--sqlite-path", default="birds.db", help="Duong dan file SQLite.")
    parser.add_argument("--faiss-path", default="birds.faiss", help="Duong dan file FAISS index.")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Metric cho FAISS.")
    parser.add_argument("--query-image-id", type=int, default=0, help="img_id de demo search. 0 = lay anh dau tien.")
    parser.add_argument("--top-k", type=int, default=5, help="So ket qua tra ve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cosine = args.metric == "cosine"
    # Luon build lai DB khi chay script doc lap (khong dung nhanh cache SQLite/FAISS).
    summary = build_metadata_sqlite_and_faiss(
        output_dir=args.output_dir,
        sqlite_path=args.sqlite_path,
        faiss_path=args.faiss_path,
        use_cosine=use_cosine,
        features_rebuilt=True,
    )
    print("[OK] Tao CSDL thanh cong:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    conn = sqlite3.connect(args.sqlite_path)
    conn.row_factory = sqlite3.Row
    if args.query_image_id > 0:
        row = conn.execute("SELECT id, feature_path FROM images WHERE id = ?", (args.query_image_id,)).fetchone()
    else:
        row = conn.execute("SELECT id, feature_path FROM images ORDER BY id ASC LIMIT 1").fetchone()
    conn.close()
    if row is None:
        print("[WARN] Khong tim thay image trong SQLite de test search.")
        return

    query_img_id = int(row["id"])
    query_path = str(row["feature_path"])
    if not os.path.exists(query_path):
        print(f"[WARN] Khong tim thay vector query: {query_path}")
        return

    query_feat = np.load(query_path).astype(np.float32)
    results = search_similar_vectors(
        query_feat=query_feat,
        top_k=args.top_k,
        sqlite_path=args.sqlite_path,
        faiss_path=args.faiss_path,
        use_cosine=use_cosine,
    )
    print(f"\n[INFO] Query img_id={query_img_id}, top_k={args.top_k}")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank:02d}. score={item['distance_or_score']:.6f} | "
            f"id={item['id']} | species={item['species_name']} | file={item['filename']}"
        )


if __name__ == "__main__":
    main()
