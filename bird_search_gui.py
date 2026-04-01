import argparse
import os
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk


def _load_faiss():
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    return faiss


class QueryEmbedder:
    def __init__(self) -> None:
        self._torch = None
        self._preprocess = None
        self._backbone = None
        self._device = "cpu"

    def _lazy_init(self) -> None:
        if self._backbone is not None:
            return
        try:
            import torch  # type: ignore
            from torchvision import models, transforms  # type: ignore
        except Exception as ex:
            raise RuntimeError(
                "Khong the import torch/torchvision. Hay cai dat: pip install torch torchvision"
            ) from ex

        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(pretrained=True)

        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        backbone.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone.to(self._device)

        self._torch = torch
        self._backbone = backbone
        self._preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def embed_image(self, image_path: str) -> np.ndarray:
        self._lazy_init()
        assert self._torch is not None
        assert self._preprocess is not None
        assert self._backbone is not None

        img = Image.open(image_path).convert("RGB")
        with self._torch.no_grad():
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            feat = self._backbone(tensor).flatten(1).squeeze(0).cpu().numpy().astype(np.float32)
        return feat


class BirdImageSearchApp:
    def __init__(
        self,
        root: tk.Tk,
        sqlite_path: str,
        faiss_path: str,
        images_dir: str,
        top_k: int,
        metric: str,
    ) -> None:
        self.root = root
        self.root.title("Bird Image Search (CBIR)")
        self.root.geometry("1280x760")

        self.sqlite_path = sqlite_path
        self.faiss_path = faiss_path
        self.images_dir = images_dir
        self.top_k = top_k
        self.metric = metric
        self.use_cosine = metric == "cosine"

        self.query_path = ""
        self._query_photo: ImageTk.PhotoImage | None = None
        self._result_photos: list[ImageTk.PhotoImage] = []

        self.embedder = QueryEmbedder()
        self.faiss = _load_faiss()
        self.conn = self._open_sqlite()
        self.index = self._load_index() if self.faiss is not None else None
        self._np_vectors: np.ndarray | None = None
        self._np_meta: list[dict[str, Any]] = []
        if self.index is None:
            self._init_numpy_backend()

        self._build_ui()

    def _load_index(self):
        if not os.path.exists(self.faiss_path):
            return None
        return self.faiss.read_index(self.faiss_path)

    def _open_sqlite(self):
        if not os.path.exists(self.sqlite_path):
            raise FileNotFoundError(f"Khong tim thay SQLite DB: {self.sqlite_path}")
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(main, text="Cau hinh tim kiem", padding=10)
        top.pack(fill=tk.X)

        self.query_var = tk.StringVar()
        ttk.Label(top, text="Anh query:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.query_var, width=90).grid(row=0, column=1, padx=6, sticky="we")
        ttk.Button(top, text="Chon anh...", command=self.choose_image).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Tim Top-5", command=self.search).grid(row=0, column=3, padx=4)

        self.metric_var = tk.StringVar(value=self.metric)
        ttk.Label(top, text="Metric:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        metric_box = ttk.Frame(top)
        metric_box.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(metric_box, text="Cosine", variable=self.metric_var, value="cosine").pack(side=tk.LEFT)
        ttk.Radiobutton(metric_box, text="L2", variable=self.metric_var, value="l2").pack(side=tk.LEFT, padx=10)
        backend_text = "FAISS" if self.index is not None else "NumPy (fallback, exact search)"
        ttk.Label(top, text=f"Backend: {backend_text}").grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(top, text=f"DB: {self.sqlite_path}").grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Label(top, text=f"FAISS: {self.faiss_path}").grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Label(top, text=f"Image folder: {self.images_dir}").grid(row=5, column=0, columnspan=4, sticky="w")
        top.columnconfigure(1, weight=1)

        startup_status = "San sang. Chon 1 anh chim de bat dau."
        if self.index is None:
            startup_status = "FAISS khong kha dung -> dang dung NumPy fallback. Van tim top-5 duoc binh thuong."
        self.status_var = tk.StringVar(value=startup_status)
        ttk.Label(main, textvariable=self.status_var).pack(fill=tk.X, pady=8)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        query_frame = ttk.LabelFrame(body, text="Anh dau vao", padding=8)
        query_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.query_img_label = ttk.Label(query_frame, text="Chua co anh")
        self.query_img_label.pack(fill=tk.BOTH, expand=True)

        results_frame = ttk.LabelFrame(body, text="Top ket qua tuong dong", padding=8)
        results_frame.grid(row=0, column=1, sticky="nsew")
        self.results_container = ttk.Frame(results_frame)
        self.results_container.pack(fill=tk.BOTH, expand=True)
        self._clear_results()

    def _clear_results(self) -> None:
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self._result_photos = []
        for i in range(self.top_k):
            card = ttk.Frame(self.results_container, padding=6, relief=tk.GROOVE)
            card.grid(row=0, column=i, sticky="n", padx=4, pady=4)
            ttk.Label(card, text=f"#{i + 1}").pack()
            ttk.Label(card, text="(chua co ket qua)", width=24).pack()

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Chon anh query",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.query_path = path
        self.query_var.set(path)
        self._show_query_image(path)
        self.status_var.set("Da chon anh query. Bam 'Tim Top-5'.")

    def _make_preview(self, image_path: str, size: tuple[int, int]) -> ImageTk.PhotoImage:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        return ImageTk.PhotoImage(img)

    def _show_query_image(self, image_path: str) -> None:
        self._query_photo = self._make_preview(image_path, (320, 320))
        self.query_img_label.configure(image=self._query_photo, text="")

    def _fetch_metadata_by_faiss_idx(self, faiss_idx: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT id, filename, species_id, species_name, feature_path, faiss_idx
            FROM images
            WHERE faiss_idx = ?
            """,
            (int(faiss_idx),),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "filename": str(row["filename"]),
            "species_id": int(row["species_id"]),
            "species_name": str(row["species_name"]),
            "feature_path": str(row["feature_path"]),
            "faiss_idx": int(row["faiss_idx"]),
        }

    def _init_numpy_backend(self) -> None:
        rows = self.conn.execute(
            """
            SELECT id, filename, species_id, species_name, feature_path, faiss_idx
            FROM images
            ORDER BY faiss_idx ASC
            """
        ).fetchall()
        vectors: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        for row in rows:
            feature_path = str(row["feature_path"])
            if not os.path.exists(feature_path):
                continue
            try:
                vec = np.load(feature_path).astype(np.float32)
            except Exception:
                continue
            vectors.append(vec)
            metas.append(
                {
                    "id": int(row["id"]),
                    "filename": str(row["filename"]),
                    "species_id": int(row["species_id"]),
                    "species_name": str(row["species_name"]),
                    "feature_path": feature_path,
                    "faiss_idx": int(row["faiss_idx"]),
                }
            )
        if not vectors:
            raise RuntimeError("Khong tai duoc vectors tu SQLite/feature_path de fallback tim kiem.")
        matrix = np.vstack(vectors).astype(np.float32)
        if self.use_cosine:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            matrix = matrix / norms
        self._np_vectors = matrix
        self._np_meta = metas

    def _render_results(self, items: list[dict[str, Any]]) -> None:
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self._result_photos = []
        for rank, item in enumerate(items, start=1):
            card = ttk.Frame(self.results_container, padding=6, relief=tk.GROOVE)
            card.grid(row=0, column=rank - 1, sticky="n", padx=4, pady=4)

            img_path = os.path.join(self.images_dir, item["filename"])
            if os.path.exists(img_path):
                preview = self._make_preview(img_path, (180, 180))
                self._result_photos.append(preview)
                ttk.Label(card, image=preview).pack()
            else:
                ttk.Label(card, text="[Khong tim thay anh]").pack()

            score_key = "similarity" if self.use_cosine else "distance"
            score_val = item[score_key]
            header = f"#{rank} | {score_key}={score_val:.4f}"
            ttk.Label(card, text=header).pack(anchor="w")
            ttk.Label(card, text=f"species: {item['species_name']}", wraplength=180).pack(anchor="w")
            ttk.Label(card, text=f"file: {item['filename']}", wraplength=180).pack(anchor="w")

    def _search_top_k(self, query_feat: np.ndarray) -> list[dict[str, Any]]:
        if self.index is None:
            return self._search_top_k_numpy(query_feat)
        query = query_feat.reshape(1, -1).astype(np.float32)
        if self.use_cosine:
            self.faiss.normalize_L2(query)
        distances, indices = self.index.search(query, self.top_k)

        items: list[dict[str, Any]] = []
        for score_or_dist, faiss_idx in zip(distances[0], indices[0]):
            if int(faiss_idx) < 0:
                continue
            meta = self._fetch_metadata_by_faiss_idx(int(faiss_idx))
            if meta is None:
                continue
            payload = dict(meta)
            if self.use_cosine:
                payload["similarity"] = float(score_or_dist)
            else:
                payload["distance"] = float(score_or_dist)
            items.append(payload)

        if self.use_cosine:
            items.sort(key=lambda x: x["similarity"], reverse=True)
        else:
            items.sort(key=lambda x: x["distance"])
        return items

    def _search_top_k_numpy(self, query_feat: np.ndarray) -> list[dict[str, Any]]:
        if self._np_vectors is None or not self._np_meta:
            raise RuntimeError("NumPy backend chua duoc khoi tao.")

        query = query_feat.astype(np.float32)
        if self.use_cosine:
            query = query / (np.linalg.norm(query) + 1e-8)
            scores = self._np_vectors @ query
            k = min(self.top_k, scores.shape[0])
            top_idx = np.argpartition(-scores, k - 1)[:k]
            sorted_idx = top_idx[np.argsort(-scores[top_idx])]
            return [dict(self._np_meta[i], similarity=float(scores[i])) for i in sorted_idx]

        dists = np.sum((self._np_vectors - query.reshape(1, -1)) ** 2, axis=1)
        k = min(self.top_k, dists.shape[0])
        top_idx = np.argpartition(dists, k - 1)[:k]
        sorted_idx = top_idx[np.argsort(dists[top_idx])]
        return [dict(self._np_meta[i], distance=float(dists[i])) for i in sorted_idx]

    def search(self) -> None:
        path = self.query_var.get().strip()
        if not path:
            messagebox.showwarning("Thieu anh", "Vui long chon anh query truoc.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Loi duong dan", f"Khong tim thay file:\n{path}")
            return

        chosen_metric = self.metric_var.get().strip().lower()
        prev_use_cosine = self.use_cosine
        self.use_cosine = chosen_metric == "cosine"
        if self.index is None and self.use_cosine != prev_use_cosine:
            self._init_numpy_backend()
        self.status_var.set("Dang trich xuat dac trung va tim kiem...")
        self.root.update_idletasks()

        try:
            query_feat = self.embedder.embed_image(path)
            results = self._search_top_k(query_feat)
            self._show_query_image(path)
            self._render_results(results)
            self.status_var.set(f"Da tim thay {len(results)} ket qua gan nhat (sap xep giam dan do tuong dong).")
        except Exception as ex:
            self.status_var.set("Tim kiem that bai.")
            messagebox.showerror("Loi khi tim kiem", str(ex))

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI tim kiem anh chim theo noi dung (CBIR)")
    parser.add_argument("--sqlite-path", default="birds.db", help="Duong dan SQLite metadata")
    parser.add_argument("--faiss-path", default="birds.faiss", help="Duong dan FAISS index")
    parser.add_argument("--images-dir", default="./dataset_processed/images", help="Thu muc anh dataset da xu ly")
    parser.add_argument("--top-k", type=int, default=5, help="So ket qua tra ve")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Metric tim kiem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        app = BirdImageSearchApp(
            root=root,
            sqlite_path=args.sqlite_path,
            faiss_path=args.faiss_path,
            images_dir=args.images_dir,
            top_k=max(1, int(args.top_k)),
            metric=args.metric,
        )
    except Exception as ex:
        messagebox.showerror("Khoi tao that bai", str(ex))
        root.destroy()
        return

    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
