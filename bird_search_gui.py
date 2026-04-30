import argparse
import json
import os
import sqlite3
import tkinter as tk
import time
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from cub_pipeline.algorithmic_tier2 import extract_algorithmic_embedding, load_algorithmic_artifacts
from cub_pipeline.gpu_backend import configure_cuda_library_path, faiss_gpu_available
from cub_pipeline.tier1_features import extract_similarity_tier1_vector
from cub_pipeline.tier3_features import extract_hog_shape, extract_hsv_histogram_48, extract_lbp_texture_64

configure_cuda_library_path()

try:
    from cub_pipeline.config import RERANK_FUSION_WEIGHTS
except Exception:
    RERANK_FUSION_WEIGHTS = (0.10, 0.80, 0.10)

try:
    from cub_pipeline.config import AQE_ALPHA, DIFFUSION_ALPHA, PART_WEIGHTS
except Exception:
    AQE_ALPHA = 0.60
    DIFFUSION_ALPHA = 0.25
    PART_WEIGHTS = (0.35, 0.30, 0.15, 0.10, 0.10)

try:
    from cub_pipeline.config import (
        ADAPTIVE_CONFIDENCE_THRESHOLD,
        ADAPTIVE_FUSION_ENABLED,
        ADAPTIVE_PROTOTYPE_MARGIN,
        ADAPTIVE_PROTOTYPE_MIN_SCORE,
    )
except Exception:
    ADAPTIVE_FUSION_ENABLED = True
    ADAPTIVE_CONFIDENCE_THRESHOLD = 0.80
    ADAPTIVE_PROTOTYPE_MIN_SCORE = 0.70
    ADAPTIVE_PROTOTYPE_MARGIN = 0.02


DEBUG_LOG_PATH = "/Users/minhtuansfile/HoangTuan_Code/CSDL_DPT/.cursor/debug-9afddb.log"
DEBUG_SESSION_ID = "9afddb"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def _load_faiss():
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    return faiss


class QueryEmbedder:
    def __init__(self, features_dir: str) -> None:
        self.features_dir = features_dir
        self.artifacts = load_algorithmic_artifacts(features_dir)

    def embed_image(self, image_path: str) -> np.ndarray:
        feat, _ = extract_algorithmic_embedding(image_path, self.artifacts)
        return feat

    def anatomy_blocks(self, image_path: str) -> np.ndarray:
        _, blocks = extract_algorithmic_embedding(image_path, self.artifacts)
        return blocks


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
        self.rerank_weights = self._get_rerank_weights()
        self.features_dir = os.path.join(os.path.dirname(images_dir.rstrip("\\/")), "features")

        self.query_path = ""
        self._query_photo: ImageTk.PhotoImage | None = None
        self._result_photos: list[ImageTk.PhotoImage] = []
        self._flow_photos: dict[str, ImageTk.PhotoImage] = {}

        self.embedder = QueryEmbedder(self.features_dir)
        self.faiss = _load_faiss()
        self.faiss_device = "cpu"
        self.conn = self._open_sqlite()
        self.index = self._load_index() if self.faiss is not None else None
        self._np_vectors: np.ndarray | None = None
        self._np_meta: list[dict[str, Any]] = []
        self._sim_graph: np.ndarray | None = None
        self._class_proto_scores: dict[int, float] = {}
        self._class_proto_top_id: int | None = None
        self._class_proto_top_score: float = 0.0
        self._class_proto_margin: float = 0.0
        self._class_prototypes = self._load_class_prototypes()
        self.dataset_stats = self._get_dataset_stats()
        self._init_numpy_backend()
        self._build_similarity_graph()

        self._build_ui()

    def _get_rerank_weights(self) -> tuple[float, float, float]:
        raw = RERANK_FUSION_WEIGHTS
        try:
            w1 = float(raw[0])
            w2 = float(raw[1])
            w3 = float(raw[2])
        except Exception:
            return (0.10, 0.80, 0.10)
        total = w1 + w2 + w3
        if total <= 1e-8:
            return (0.10, 0.80, 0.10)
        return (w1 / total, w2 / total, w3 / total)

    def _get_dataset_stats(self) -> dict[str, Any]:
        try:
            row = self.conn.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT species_id) AS total_species
                FROM images
                """
            ).fetchone()
            minmax = self.conn.execute(
                """
                SELECT MIN(c) AS min_per_species, MAX(c) AS max_per_species
                FROM (
                    SELECT species_id, COUNT(*) AS c
                    FROM images
                    GROUP BY species_id
                )
                """
            ).fetchone()
            stats = {
                "total_rows": int(row["total_rows"]) if row else 0,
                "total_species": int(row["total_species"]) if row else 0,
                "min_per_species": int(minmax["min_per_species"]) if minmax and minmax["min_per_species"] is not None else 0,
                "max_per_species": int(minmax["max_per_species"]) if minmax and minmax["max_per_species"] is not None else 0,
            }
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="H8",
                location="bird_search_gui.py:_get_dataset_stats",
                message="Indexed dataset coverage",
                data=stats,
            )
            return stats
        except Exception:
            return {"total_rows": 0, "total_species": 0, "min_per_species": 0, "max_per_species": 0}

    def _find_species_rank(self, items: list[dict[str, Any]], keyword: str = "groove_billed_ani") -> dict[str, Any]:
        key = str(keyword).lower()
        for idx, item in enumerate(items, start=1):
            species = str(item.get("species_name", "")).lower()
            if key in species:
                return {
                    "found": True,
                    "rank": int(idx),
                    "species_name": str(item.get("species_name", "")),
                    "filename": str(item.get("filename", "")),
                    "similarity": float(item.get("similarity", 0.0)) if "similarity" in item else None,
                    "fused_similarity": float(item.get("fused_similarity", 0.0)) if "fused_similarity" in item else None,
                }
        return {"found": False, "rank": None}

    def _extract_tier1_vector(self, img: Image.Image) -> np.ndarray:
        return extract_similarity_tier1_vector(img)

    def _extract_tier3_vector(self, img: Image.Image) -> np.ndarray:
        rgb = img.convert("RGB")
        return np.concatenate(
            [
                extract_hsv_histogram_48(rgb),
                extract_lbp_texture_64(rgb),
                extract_hog_shape(rgb),
            ],
            axis=0,
        ).astype(np.float32)

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    def _minmax_norm(self, values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-8:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def _rerank_multi_tier(
        self,
        query_img: Image.Image,
        query_cnn: np.ndarray,
        items: list[dict[str, Any]],
        weights_override: tuple[float, float, float] | None = None,
        run_id: str = "pre-fix",
        confidence_flags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_tier1 = self._extract_tier1_vector(query_img)
        query_tier3 = self._extract_tier3_vector(query_img.resize((224, 224), Image.Resampling.BILINEAR))
        out: list[dict[str, Any]] = []
        for item in items:
            img_path = os.path.join(self.images_dir, item["filename"])
            if not os.path.exists(img_path):
                continue
            try:
                cand_img = Image.open(img_path).convert("RGB")
                cand_tier1 = self._extract_tier1_vector(cand_img)
                cand_tier3 = self._extract_tier3_vector(cand_img.resize((224, 224), Image.Resampling.BILINEAR))
            except Exception:
                continue
            sim_tier1 = self._cos_sim(query_tier1, cand_tier1)
            sim_tier2 = float(item.get("similarity", 0.0)) if self.use_cosine else -float(item.get("distance", 0.0))
            sim_tier3 = self._cos_sim(query_tier3, cand_tier3)
            payload = dict(item)
            payload["sim_tier1"] = sim_tier1
            payload["sim_tier2"] = sim_tier2
            payload["sim_tier3"] = sim_tier3
            out.append(payload)

        # Normalize each tier score to [0, 1] on the same candidate pool
        # before weighted fusion to avoid scale domination.
        t1_vals = [float(x["sim_tier1"]) for x in out]
        t2_vals = [float(x["sim_tier2"]) for x in out]
        t3_vals = [float(x["sim_tier3"]) for x in out]
        t1_norm = self._minmax_norm(t1_vals)
        t2_norm = self._minmax_norm(t2_vals)
        t3_norm = self._minmax_norm(t3_vals)

        # Tier-2 algorithmic vector la tin hieu retrieval chinh.
        if weights_override is not None:
            w1, w2, w3 = (float(weights_override[0]), float(weights_override[1]), float(weights_override[2]))
        else:
            w1, w2, w3 = self.rerank_weights
        adaptive_payload: list[dict[str, float]] = []
        for idx, payload in enumerate(out):
            payload["sim_tier1_norm"] = float(t1_norm[idx])
            payload["sim_tier2_norm"] = float(t2_norm[idx])
            payload["sim_tier3_norm"] = float(t3_norm[idx])
            wi1, wi2, wi3 = w1, w2, w3
            if ADAPTIVE_FUSION_ENABLED:
                low_conf = bool((confidence_flags or {}).get("low_confidence", False))
                top1 = float((confidence_flags or {}).get("top1_score", 0.0))
                if low_conf or top1 < float(ADAPTIVE_CONFIDENCE_THRESHOLD):
                    wi1, wi2, wi3 = 0.0, 1.0, 0.0
                else:
                    cid = int(payload.get("species_id", -1))
                    proto_s = float(self._class_proto_scores.get(cid, 0.0))
                    strong_proto = (
                        cid == self._class_proto_top_id
                        and proto_s >= float(ADAPTIVE_PROTOTYPE_MIN_SCORE)
                        and self._class_proto_margin >= float(ADAPTIVE_PROTOTYPE_MARGIN)
                    )
                    if strong_proto:
                        wi1 = min(0.35, w1 + 0.10)
                        wi2 = max(0.45, w2 - 0.08)
                        wi3 = max(0.10, 1.0 - wi1 - wi2)
            payload["fused_similarity"] = wi1 * payload["sim_tier1_norm"] + wi2 * payload["sim_tier2_norm"] + wi3 * payload["sim_tier3_norm"]
            payload["w_tier1"] = float(wi1)
            payload["w_tier2"] = float(wi2)
            payload["w_tier3"] = float(wi3)
            adaptive_payload.append({"w1": float(wi1), "w2": float(wi2), "w3": float(wi3)})

        out.sort(key=lambda x: x["fused_similarity"], reverse=True)
        groove_diag = {"found": False, "rank": None}
        for idx, payload in enumerate(out, start=1):
            if "groove_billed_ani" in str(payload.get("species_name", "")).lower():
                groove_diag = {
                    "found": True,
                    "rank": int(idx),
                    "species_name": str(payload.get("species_name", "")),
                    "filename": str(payload.get("filename", "")),
                    "tier1_raw": float(payload.get("sim_tier1", 0.0)),
                    "tier2_raw": float(payload.get("sim_tier2", 0.0)),
                    "tier3_raw": float(payload.get("sim_tier3", 0.0)),
                    "tier1_norm": float(payload.get("sim_tier1_norm", 0.0)),
                    "tier2_norm": float(payload.get("sim_tier2_norm", 0.0)),
                    "tier3_norm": float(payload.get("sim_tier3_norm", 0.0)),
                    "fused": float(payload.get("fused_similarity", 0.0)),
                }
                break
        _agent_debug_log(
            run_id=run_id,
            hypothesis_id="H4",
            location="bird_search_gui.py:_rerank_multi_tier",
            message="Tier fusion diagnostics",
            data={
                "candidate_count": len(out),
                "weights": {"tier1": float(w1), "tier2": float(w2), "tier3": float(w3)},
                "prototype_top": {
                    "class_id": self._class_proto_top_id,
                    "score": float(self._class_proto_top_score),
                    "margin": float(self._class_proto_margin),
                },
                "adaptive_weight_examples": adaptive_payload[:3],
                "tier2_norm_span": [float(min(t2_norm)) if t2_norm else 0.0, float(max(t2_norm)) if t2_norm else 0.0],
                "top3_after_fusion": [
                    {
                        "filename": str(x.get("filename", "")),
                        "species": str(x.get("species_name", "")),
                        "fused": float(x.get("fused_similarity", 0.0)),
                        "tier2_raw": float(x.get("sim_tier2", 0.0)),
                    }
                    for x in out[:3]
                ],
                "groove_diag": groove_diag,
            },
        )
        return out[: self.top_k]

    def _confidence_flags(self, path: str, tier2_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        top1 = float(tier2_candidates[0]["similarity"]) if tier2_candidates else 0.0
        top2 = float(tier2_candidates[1]["similarity"]) if len(tier2_candidates) > 1 else 0.0
        margin = top1 - top2
        is_external_query = os.path.abspath(path).replace("\\", "/").find("/dataset_processed/images/") < 0
        low_confidence = (top1 < 0.80) or (margin < 0.03)
        out_of_scope_risk = is_external_query and low_confidence
        return {
            "top1_score": top1,
            "top2_score": top2,
            "margin": margin,
            "is_external_query": is_external_query,
            "low_confidence": low_confidence,
            "out_of_scope_risk": out_of_scope_risk,
        }

    def _load_class_prototypes(self) -> dict[str, Any]:
        path = os.path.join(self.features_dir, "tier2_class_prototypes.npz")
        if not os.path.exists(path):
            return {}
        try:
            z = np.load(path, allow_pickle=False)
            class_ids = z["class_ids"].astype(np.int32)
            protos = z["prototypes"].astype(np.float32)
            norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
            protos = (protos / norms).astype(np.float32)
            return {"class_ids": class_ids, "prototypes": protos}
        except Exception:
            return {}

    def _compute_prototype_scores(self, query_feat: np.ndarray) -> None:
        self._class_proto_scores = {}
        self._class_proto_top_id = None
        self._class_proto_top_score = 0.0
        self._class_proto_margin = 0.0
        if not self._class_prototypes:
            return
        class_ids = self._class_prototypes["class_ids"]
        protos = self._class_prototypes["prototypes"]
        q = query_feat.astype(np.float32, copy=False)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = protos @ q
        for i, cid in enumerate(class_ids):
            self._class_proto_scores[int(cid)] = float(scores[i])
        order = np.argsort(-scores)
        if order.size == 0:
            return
        top_idx = int(order[0])
        self._class_proto_top_id = int(class_ids[top_idx])
        self._class_proto_top_score = float(scores[top_idx])
        if order.size > 1:
            self._class_proto_margin = float(scores[top_idx] - scores[int(order[1])])

    def _load_index(self):
        if not os.path.exists(self.faiss_path):
            return None
        if self.faiss is None:
            return None
        index = self.faiss.read_index(self.faiss_path)
        if not faiss_gpu_available(self.faiss):
            self.faiss_device = "cpu"
            return index
        try:
            res = self.faiss.StandardGpuResources()
            gpu_index = self.faiss.index_cpu_to_gpu(res, 0, index)
            self._faiss_gpu_res = res
            self.faiss_device = "gpu"
            return gpu_index
        except Exception as ex:
            self.faiss_device = "cpu"
            print(f"[WARN] Khong the su dung FAISS GPU trong GUI, fallback CPU: {ex}")
            return index

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
        if self.index is not None:
            backend_text = f"FAISS ({self.faiss_device.upper()})"
        else:
            backend_text = "NumPy (fallback, exact search)"
        ttk.Label(top, text=f"Backend: {backend_text}").grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(top, text=f"DB: {self.sqlite_path}").grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Label(top, text=f"FAISS: {self.faiss_path}").grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Label(top, text=f"Image folder: {self.images_dir}").grid(row=5, column=0, columnspan=4, sticky="w")
        if self.dataset_stats.get("total_species", 0) < 180:
            ttk.Label(
                top,
                text=(
                    f"Coverage canh bao: species={self.dataset_stats.get('total_species', 0)} "
                    f"| rows={self.dataset_stats.get('total_rows', 0)} "
                    f"| min/species={self.dataset_stats.get('min_per_species', 0)}. "
                    "Nen build lai dataset retrieval voi do phu cao hon."
                ),
            ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 0))
        top.columnconfigure(1, weight=1)

        startup_status = "San sang. Chon 1 anh chim de bat dau."
        if self.index is None:
            startup_status = "FAISS khong kha dung -> dang dung NumPy fallback. Van tim top-5 duoc binh thuong."
        self.status_var = tk.StringVar(value=startup_status)
        ttk.Label(main, textvariable=self.status_var).pack(fill=tk.X, pady=8)

        flow_frame = ttk.LabelFrame(main, text="Luong xu ly query qua 3 tang", padding=8)
        flow_frame.pack(fill=tk.X, pady=(0, 8))
        flow_cols = [
            ("input", "Anh dau vao"),
            ("tier1", "Tang 1: Custom attrs"),
            ("tier2", "Tang 2: Algorithmic 512D input"),
            ("tier3", "Tang 3: Handcrafted map"),
        ]
        self.flow_image_labels: dict[str, ttk.Label] = {}
        self.flow_text_vars: dict[str, tk.StringVar] = {}
        for idx, (key, title) in enumerate(flow_cols):
            card = ttk.Frame(flow_frame, padding=4, relief=tk.GROOVE)
            card.grid(row=0, column=idx, sticky="n", padx=4)
            ttk.Label(card, text=title).pack(anchor="center")
            lbl = ttk.Label(card, text="(chua co)")
            lbl.pack()
            txt_var = tk.StringVar(value="-")
            ttk.Label(card, textvariable=txt_var, wraplength=220, justify=tk.LEFT).pack(anchor="w")
            self.flow_image_labels[key] = lbl
            self.flow_text_vars[key] = txt_var

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
        src = Image.open(path).convert("RGB")
        self._set_flow_image("input", src, "Anh query goc")
        self._clear_flow_outputs()
        self.status_var.set("Da chon anh query. Bam 'Tim Top-5'.")

    def _make_preview(self, image_path: str, size: tuple[int, int]) -> ImageTk.PhotoImage:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        return ImageTk.PhotoImage(img)

    def _show_query_image(self, image_path: str) -> None:
        self._query_photo = self._make_preview(image_path, (320, 320))
        self.query_img_label.configure(image=self._query_photo, text="")

    def _set_flow_image(self, key: str, image: Image.Image, description: str) -> None:
        if key not in self.flow_image_labels:
            return
        preview = image.convert("RGB")
        preview.thumbnail((220, 220))
        photo = ImageTk.PhotoImage(preview)
        self._flow_photos[key] = photo
        self.flow_image_labels[key].configure(image=photo, text="")
        self.flow_text_vars[key].set(description)

    def _clear_flow_outputs(self) -> None:
        for key in ("tier1", "tier2", "tier3"):
            self.flow_image_labels[key].configure(image="", text="(cho xu ly)")
            self.flow_text_vars[key].set("-")

    def _tier1_visual(self, img: Image.Image) -> tuple[Image.Image, str]:
        arr = np.array(img.convert("HSV"), dtype=np.float32)
        h = arr[:, :, 0] / 255.0
        s = arr[:, :, 1] / 255.0
        v = arr[:, :, 2] / 255.0
        valid = (s > 0.20) & (v > 0.15)
        red = float((((h < 0.05) | (h >= 0.95)) & valid).mean())
        yellow = float((((h >= 0.12) & (h < 0.20) & valid)).mean())
        blue = float((((h >= 0.55) & (h < 0.75) & valid)).mean())
        overlay = np.array(img.convert("RGB"), dtype=np.uint8)
        overlay[~valid] = (overlay[~valid] * 0.35).astype(np.uint8)
        out = Image.fromarray(overlay)
        desc = f"Mask mau hop le | red={red:.3f}, yellow={yellow:.3f}, blue={blue:.3f}"
        return out, desc

    def _tier2_visual(self, img: Image.Image, embedding: np.ndarray) -> tuple[Image.Image, str]:
        resized = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        norm_preview = ImageOps.autocontrast(resized)
        draw = ImageDraw.Draw(norm_preview)
        d = embedding[:64]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        bar_w = 3
        x0 = 8
        y_base = 216
        for i, val in enumerate(d):
            x = x0 + i * bar_w
            h = int(48 * float(val))
            draw.line([(x, y_base), (x, y_base - h)], fill=(255, 220, 0), width=2)
        desc = f"Anh chuan hoa + preview 64/{embedding.shape[0]} dims | norm={float(np.linalg.norm(embedding)):.2f}"
        return norm_preview, desc

    def _tier3_visual(self, img: Image.Image) -> tuple[Image.Image, str]:
        gray = np.array(img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx**2 + gy**2)
        mag = mag / (mag.max() + 1e-8)
        hog_map = (mag * 255.0).astype(np.uint8)
        out = Image.fromarray(hog_map, mode="L").convert("RGB")
        desc = "Ban do gradient/HOG (shape cues)"
        return out, desc

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

    def _build_similarity_graph(self, knn: int = 25) -> None:
        if self._np_vectors is None or len(self._np_vectors) == 0:
            self._sim_graph = None
            return
        vec = self._np_vectors.astype(np.float32)
        sim = vec @ vec.T
        n = sim.shape[0]
        k = min(max(2, knn), n)
        graph = np.zeros_like(sim, dtype=np.float32)
        for i in range(n):
            idx = np.argpartition(-sim[i], k - 1)[:k]
            graph[i, idx] = np.maximum(sim[i, idx], 0.0)
        graph = graph + graph.T
        row_sum = graph.sum(axis=1, keepdims=True) + 1e-8
        self._sim_graph = graph / row_sum

    def _apply_query_expansion(self, query_feat: np.ndarray, first_pass: list[dict[str, Any]]) -> np.ndarray:
        top = first_pass[:5]
        vecs = []
        for item in top:
            p = item.get("feature_path")
            if not p or not os.path.exists(str(p)):
                continue
            try:
                vecs.append(np.load(str(p)).astype(np.float32))
            except Exception:
                continue
        if not vecs:
            return query_feat
        mean_top = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        expanded = AQE_ALPHA * query_feat + (1.0 - AQE_ALPHA) * mean_top
        expanded = expanded / (np.linalg.norm(expanded) + 1e-8)
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="bird_search_gui.py:_apply_query_expansion",
            message="AQE query drift check",
            data={
                "first_pass_size": len(first_pass),
                "alpha": float(AQE_ALPHA),
                "query_norm_before": float(np.linalg.norm(query_feat)),
                "mean_top_norm": float(np.linalg.norm(mean_top)),
                "expanded_norm": float(np.linalg.norm(expanded)),
                "cos_query_to_expanded": float(np.dot(query_feat, expanded) / ((np.linalg.norm(query_feat) + 1e-8) * (np.linalg.norm(expanded) + 1e-8))),
            },
        )
        return expanded

    def _apply_diffusion(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._sim_graph is None or self._np_meta is None:
            return items
        id_to_pos = {int(m["faiss_idx"]): idx for idx, m in enumerate(self._np_meta)}
        seed = np.zeros((len(self._np_meta),), dtype=np.float32)
        for item in items:
            faiss_idx = int(item["faiss_idx"])
            pos = id_to_pos.get(faiss_idx)
            if pos is None:
                continue
            val = float(item.get("similarity", 0.0)) if self.use_cosine else -float(item.get("distance", 0.0))
            seed[pos] = val
        propagated = self._sim_graph @ seed
        out = []
        for item in items:
            faiss_idx = int(item["faiss_idx"])
            pos = id_to_pos.get(faiss_idx)
            if pos is None:
                out.append(item)
                continue
            merged = dict(item)
            raw = float(item.get("similarity", 0.0)) if self.use_cosine else -float(item.get("distance", 0.0))
            merged["similarity"] = (1.0 - DIFFUSION_ALPHA) * raw + DIFFUSION_ALPHA * float(propagated[pos])
            out.append(merged)
        out.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="bird_search_gui.py:_apply_diffusion",
            message="Diffusion rank shift check",
            data={
                "input_size": len(items),
                "diffusion_alpha": float(DIFFUSION_ALPHA),
                "top3_after_diffusion": [
                    {
                        "filename": str(x.get("filename", "")),
                        "species": str(x.get("species_name", "")),
                        "similarity": float(x.get("similarity", 0.0)),
                    }
                    for x in out[:3]
                ],
            },
        )
        return out

    def _part_based_adjust(self, query_path: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        weights = np.array(PART_WEIGHTS, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        q_parts = self.embedder.anatomy_blocks(query_path)
        out = []
        for item in items:
            img_path = os.path.join(self.images_dir, item["filename"])
            if not os.path.exists(img_path):
                continue
            try:
                d_parts = self.embedder.anatomy_blocks(img_path)
                part_d = np.linalg.norm(q_parts - d_parts, axis=1)
                sim_part = 1.0 / (1.0 + float(np.dot(weights, part_d)))
            except Exception:
                sim_part = 0.0
            merged = dict(item)
            base = float(merged.get("similarity", 0.0)) if self.use_cosine else -float(merged.get("distance", 0.0))
            merged["similarity"] = 0.70 * base + 0.30 * sim_part
            merged["sim_part"] = float(sim_part)
            merged["sim_base_before_part"] = float(base)
            out.append(merged)
        out.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="bird_search_gui.py:_part_based_adjust",
            message="Part-based adjustment impact",
            data={
                "input_size": len(items),
                "weights": [float(v) for v in weights.tolist()],
                "top3_after_part_adjust": [
                    {
                        "filename": str(x.get("filename", "")),
                        "species": str(x.get("species_name", "")),
                        "base": float(x.get("sim_base_before_part", 0.0)),
                        "sim_part": float(x.get("sim_part", 0.0)),
                        "final_similarity": float(x.get("similarity", 0.0)),
                    }
                    for x in out[:3]
                ],
            },
        )
        return out

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

    def _search_top_k(self, query_feat: np.ndarray, top_k_override: int | None = None) -> list[dict[str, Any]]:
        request_k = int(top_k_override) if top_k_override is not None else int(self.top_k)
        request_k = max(1, request_k)
        if self.index is None:
            return self._search_top_k_numpy(query_feat, top_k_override=request_k)
        if self.faiss is None:
            raise RuntimeError("FAISS backend khong kha dung.")
        query = query_feat.reshape(1, -1).astype(np.float32)
        if self.use_cosine:
            self.faiss.normalize_L2(query)
        distances, indices = self.index.search(query, request_k)

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
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="bird_search_gui.py:_search_top_k",
            message="First-stage retrieval stats",
            data={
                "backend": "faiss",
                "use_cosine": bool(self.use_cosine),
                "query_dim": int(query_feat.shape[0]),
                "query_norm": float(np.linalg.norm(query_feat)),
                "request_k": int(request_k),
                "top3": [
                    {
                        "filename": str(x.get("filename", "")),
                        "species": str(x.get("species_name", "")),
                        "similarity": float(x.get("similarity", 0.0)) if self.use_cosine else None,
                        "distance": float(x.get("distance", 0.0)) if not self.use_cosine else None,
                    }
                    for x in items[:3]
                ],
            },
        )
        return items

    def _search_top_k_numpy(self, query_feat: np.ndarray, top_k_override: int | None = None) -> list[dict[str, Any]]:
        if self._np_vectors is None or not self._np_meta:
            raise RuntimeError("NumPy backend chua duoc khoi tao.")
        request_k = int(top_k_override) if top_k_override is not None else int(self.top_k)
        request_k = max(1, request_k)

        query = query_feat.astype(np.float32)
        if self.use_cosine:
            query = query / (np.linalg.norm(query) + 1e-8)
            scores = self._np_vectors @ query
            k = min(request_k, scores.shape[0])
            top_idx = np.argpartition(-scores, k - 1)[:k]
            sorted_idx = top_idx[np.argsort(-scores[top_idx])]
            return [dict(self._np_meta[i], similarity=float(scores[i])) for i in sorted_idx]

        dists = np.sum((self._np_vectors - query.reshape(1, -1)) ** 2, axis=1)
        k = min(request_k, dists.shape[0])
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
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H5",
            location="bird_search_gui.py:search",
            message="Search started",
            data={
                "query_path": os.path.abspath(path),
                "metric": "cosine" if self.use_cosine else "l2",
                "is_external_query": os.path.abspath(path).replace("\\", "/").find("/dataset_processed/images/") < 0,
                "configured_top_k": int(self.top_k),
            },
        )

        try:
            src_img = Image.open(path).convert("RGB")
            self._set_flow_image("input", src_img, "Anh query goc")

            tier1_img, tier1_desc = self._tier1_visual(src_img)
            self._set_flow_image("tier1", tier1_img, tier1_desc)

            query_feat = self.embedder.embed_image(path)
            self._compute_prototype_scores(query_feat)
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="bird_search_gui.py:search",
                message="Query embedding extracted",
                data={
                    "query_dim": int(query_feat.shape[0]),
                    "query_norm": float(np.linalg.norm(query_feat)),
                    "query_head8": [float(v) for v in query_feat[:8].tolist()],
                    "prototype_top_class": self._class_proto_top_id,
                    "prototype_top_score": float(self._class_proto_top_score),
                    "prototype_margin": float(self._class_proto_margin),
                },
            )
            tier2_img, tier2_desc = self._tier2_visual(src_img, query_feat)
            self._set_flow_image("tier2", tier2_img, tier2_desc)

            tier3_img, tier3_desc = self._tier3_visual(src_img)
            self._set_flow_image("tier3", tier3_img, tier3_desc)

            # 1) First-pass retrieval from algorithmic 512D index.
            first_pass_k = max(self.top_k, 20)
            first_pass = self._search_top_k(query_feat, top_k_override=first_pass_k)
            pre_flags = self._confidence_flags(path, first_pass)

            # External low-confidence queries are prone to PRF drift.
            # Use larger direct pool + multi-tier rerank instead of AQE/diffusion/part-adjust.
            if pre_flags["out_of_scope_risk"]:
                robust_k = max(self.top_k * 20, 120)
                tier2_candidates = self._search_top_k(query_feat, top_k_override=robust_k)
                groove_rank_pool = self._find_species_rank(tier2_candidates)
                # External query + low-confidence first pass:
                # keep retrieval signal mostly from Tier-2 (indexed algorithmic embedding)
                # because Tier-1/Tier-3 quick query features are only approximate.
                rerank_weights_override = (0.0, 1.0, 0.0)
                _agent_debug_log(
                    run_id="post-fix",
                    hypothesis_id="H6",
                    location="bird_search_gui.py:search",
                    message="External low-confidence robust mode enabled",
                    data={
                        "first_pass_k": int(first_pass_k),
                        "robust_k": int(robust_k),
                        "top1_first_pass": float(pre_flags["top1_score"]),
                        "margin_first_pass": float(pre_flags["margin"]),
                        "rerank_weights_override": [float(x) for x in rerank_weights_override],
                        "groove_rank_in_tier2_pool": groove_rank_pool,
                    },
                )
            else:
                # 2) AQE query expansion.
                expanded_query = self._apply_query_expansion(query_feat, first_pass)
                tier2_candidates = self._search_top_k(expanded_query, top_k_override=first_pass_k)
                # 3) Graph diffusion.
                tier2_candidates = self._apply_diffusion(tier2_candidates)
                # 4) Part-based asymmetric adjustment (head/body/tail/wing/leg priors).
                tier2_candidates = self._part_based_adjust(path, tier2_candidates)
                rerank_weights_override = None
            results = self._rerank_multi_tier(
                src_img,
                query_feat,
                tier2_candidates,
                weights_override=rerank_weights_override,
                run_id="post-fix" if pre_flags["out_of_scope_risk"] else "pre-fix",
                confidence_flags=pre_flags,
            )
            groove_rank_final = self._find_species_rank(results)
            _agent_debug_log(
                run_id="post-fix",
                hypothesis_id="H7",
                location="bird_search_gui.py:search",
                message="Final result target-species rank",
                data={
                    "result_size": len(results),
                    "groove_rank_in_final": groove_rank_final,
                    "top5_species": [str(x.get("species_name", "")) for x in results[:5]],
                },
            )
            flags = self._confidence_flags(path, tier2_candidates)
            self._show_query_image(path)
            self._render_results(results)
            if flags["out_of_scope_risk"]:
                self.status_var.set(
                    "Canh bao: do tin cay thap va anh query ngoai tap xu ly. Ket qua co the khong dung loai."
                )
            elif flags["low_confidence"]:
                self.status_var.set(
                    "Canh bao: do tin cay thap (top-1 gan top-2). Nen kiem tra nhieu ket qua top-k."
                )
            else:
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
