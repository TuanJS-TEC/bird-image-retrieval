import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk


@dataclass(frozen=True)
class WorkflowCommand:
    key: str
    title: str
    command: list[str]


class PipelineDashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bird Pipeline Dashboard")
        self.root.geometry("1220x780")

        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.project_root, "dataset_processed")

        self.commands = [
            WorkflowCommand(
                key="build_dataset",
                title="Build dataset + 3 feature tiers + DB",
                command=[sys.executable, "build_cub_perched_side_dataset.py"],
            ),
            WorkflowCommand(
                key="verify_db",
                title="Verify SQLite + FAISS retrieval",
                command=[
                    sys.executable,
                    "run_metadata_db.py",
                    "--output-dir",
                    "./dataset_processed",
                    "--sqlite-path",
                    "birds.db",
                    "--faiss-path",
                    "birds.faiss",
                    "--metric",
                    "cosine",
                    "--top-k",
                    "5",
                ],
            ),
        ]

        self.stage_order = [
            "BUOC 1: Doc metadata",
            "BUOC 2: Loc anh perched",
            "BUOC 3: Crop + resize",
            "BUOC 6: Xay dung bo thuoc tinh",
            "Tang 1: Custom attrs",
            "Tang 2: Algorithmic 512D",
            "Tang 3: Handcrafted",
            "BUOC 7: SQLite + FAISS",
        ]
        self.stage_vars: dict[str, tk.StringVar] = {}
        self.stage_progress: dict[str, ttk.Progressbar] = {}
        self.tier_detail_vars: dict[str, dict[str, tk.StringVar]] = {}

        self.proc: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.current_command_index = -1
        self.current_stage = ""

        self._build_ui()
        self._refresh_artifacts()
        self.root.after(120, self._drain_log_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True)

        workflow_tab = ttk.Frame(notebook, padding=10)
        logs_tab = ttk.Frame(notebook, padding=10)
        artifacts_tab = ttk.Frame(notebook, padding=10)
        tier_tab = ttk.Frame(notebook, padding=10)
        notebook.add(workflow_tab, text="Workflow")
        notebook.add(tier_tab, text="Step 6 - 3 Tiers")
        notebook.add(logs_tab, text="Live Logs")
        notebook.add(artifacts_tab, text="Artifacts")

        # Workflow tab
        top = ttk.LabelFrame(workflow_tab, text="Controls", padding=10)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="Run full pipeline directly on GUI (no manual terminal steps).",
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        self.btn_run_all = ttk.Button(top, text="Run Full Workflow", command=self.run_full_workflow)
        self.btn_run_all.grid(row=1, column=0, pady=(8, 0), padx=(0, 8), sticky="w")
        self.btn_run_selected = ttk.Button(top, text="Run Selected Step", command=self.run_selected_step)
        self.btn_run_selected.grid(row=1, column=1, pady=(8, 0), padx=(0, 8), sticky="w")
        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop_workflow, state=tk.DISABLED)
        self.btn_stop.grid(row=1, column=2, pady=(8, 0), sticky="w")

        self.command_var = tk.StringVar(value=self.commands[0].title)
        command_values = [c.title for c in self.commands]
        self.command_box = ttk.Combobox(top, textvariable=self.command_var, values=command_values, state="readonly", width=44)
        self.command_box.grid(row=1, column=3, pady=(8, 0), sticky="e")
        top.columnconfigure(3, weight=1)

        status_box = ttk.LabelFrame(workflow_tab, text="Pipeline Status", padding=10)
        status_box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.workflow_status_var = tk.StringVar(value="Idle")
        ttk.Label(status_box, textvariable=self.workflow_status_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        grid = ttk.Frame(status_box)
        grid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        grid.columnconfigure(1, weight=1)
        for idx, stage in enumerate(self.stage_order):
            ttk.Label(grid, text=stage).grid(row=idx, column=0, sticky="w", pady=3)
            bar = ttk.Progressbar(grid, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
            bar.grid(row=idx, column=1, sticky="ew", padx=8, pady=3)
            label_var = tk.StringVar(value="pending")
            ttk.Label(grid, textvariable=label_var, width=14).grid(row=idx, column=2, sticky="w")
            self.stage_progress[stage] = bar
            self.stage_vars[stage] = label_var

        # Tier tab
        tier_top = ttk.Frame(tier_tab)
        tier_top.pack(fill=tk.X)
        self.btn_refresh_tier = ttk.Button(tier_top, text="Refresh Tier Summaries", command=self._refresh_tier_summary)
        self.btn_refresh_tier.pack(side=tk.LEFT)

        tier_grid = ttk.Frame(tier_tab)
        tier_grid.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tier_grid.columnconfigure(0, weight=1)
        tier_grid.columnconfigure(1, weight=1)
        tier_grid.columnconfigure(2, weight=1)
        tier_grid.rowconfigure(0, weight=1)

        self._build_tier_card(
            parent=tier_grid,
            col=0,
            title="Tier 1 - Custom Attributes",
            tier_key="tier1",
            bullets=[
                "Color ratios + saturation/value",
                "Texture: entropy, LBP energy/entropy",
                "Shape: symmetry, Hu moments, mass ratio",
                "Output: tier1_custom_attributes.csv",
            ],
        )
        self._build_tier_card(
            parent=tier_grid,
            col=1,
            title="Tier 2 - Algorithmic Fusion Embedding",
            tier_key="tier2",
            bullets=[
                "AG-SFP 512D + ACV 256D + HFVE 512D + PPD 152D",
                "Fusion 1432D -> PCA 512D -> L2 normalize",
                "Khong dung deep backbone",
                "Output: tier2_algorithmic_embeddings.npy",
            ],
        )
        self._build_tier_card(
            parent=tier_grid,
            col=2,
            title="Tier 3 - Handcrafted (256D)",
            tier_key="tier3",
            bullets=[
                "HSV histogram: 48D",
                "LBP texture: 64D",
                "HOG shape: 144D",
                "Output: tier3_handcrafted_features.csv",
            ],
        )

        # Logs tab
        logs_top = ttk.Frame(logs_tab)
        logs_top.pack(fill=tk.X)
        ttk.Button(logs_top, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT)

        self.logs_text = tk.Text(logs_tab, wrap="none", height=20)
        self.logs_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.logs_text.configure(state=tk.DISABLED)

        # Artifacts tab
        ttk.Button(artifacts_tab, text="Refresh Artifacts", command=self._refresh_artifacts).pack(anchor="w")
        self.artifact_tree = ttk.Treeview(artifacts_tab, columns=("path", "status"), show="headings", height=18)
        self.artifact_tree.heading("path", text="Path")
        self.artifact_tree.heading("status", text="Status")
        self.artifact_tree.column("path", width=900, anchor="w")
        self.artifact_tree.column("status", width=180, anchor="center")
        self.artifact_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_tier_card(self, parent: ttk.Frame, col: int, title: str, tier_key: str, bullets: list[str]) -> None:
        card = ttk.LabelFrame(parent, text=title, padding=10)
        card.grid(row=0, column=col, sticky="nsew", padx=6)
        card.columnconfigure(0, weight=1)
        for bullet in bullets:
            ttk.Label(card, text=f"- {bullet}", wraplength=330, justify=tk.LEFT).pack(anchor="w")

        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        runtime_title = ttk.Label(card, text="Runtime details", font=("Segoe UI", 9, "bold"))
        runtime_title.pack(anchor="w")
        runtime_state = tk.StringVar(value="pending")
        runtime_count = tk.StringVar(value="processed: - / -")
        runtime_speed = tk.StringVar(value="speed: - it/s")
        runtime_eta = tk.StringVar(value="eta: -")
        runtime_file = tk.StringVar(value="artifact: missing")
        runtime_dim = tk.StringVar(value="dimension/shape: -")
        ttk.Label(card, textvariable=runtime_state).pack(anchor="w")
        ttk.Label(card, textvariable=runtime_count).pack(anchor="w")
        ttk.Label(card, textvariable=runtime_speed).pack(anchor="w")
        ttk.Label(card, textvariable=runtime_eta).pack(anchor="w")
        ttk.Label(card, textvariable=runtime_file).pack(anchor="w")
        ttk.Label(card, textvariable=runtime_dim).pack(anchor="w")
        self.tier_detail_vars[tier_key] = {
            "state": runtime_state,
            "count": runtime_count,
            "speed": runtime_speed,
            "eta": runtime_eta,
            "file": runtime_file,
            "dim": runtime_dim,
        }

    def _set_tier_runtime(
        self,
        tier_key: str,
        *,
        state: str | None = None,
        count: str | None = None,
        speed: str | None = None,
        eta: str | None = None,
    ) -> None:
        vars_map = self.tier_detail_vars.get(tier_key)
        if not vars_map:
            return
        if state is not None:
            vars_map["state"].set(f"state: {state}")
        if count is not None:
            vars_map["count"].set(f"processed: {count}")
        if speed is not None:
            vars_map["speed"].set(f"speed: {speed} it/s")
        if eta is not None:
            vars_map["eta"].set(f"eta: {eta}")

    def _set_controls_running(self, running: bool) -> None:
        self.running = running
        self.btn_run_all.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_run_selected.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _reset_stage_status(self) -> None:
        for stage in self.stage_order:
            self.stage_vars[stage].set("pending")
            self.stage_progress[stage]["value"] = 0
        self.current_stage = ""
        for key in ("tier1", "tier2", "tier3"):
            self._set_tier_runtime(key, state="pending", count="- / -", speed="-", eta="-")

    def _append_log(self, line: str) -> None:
        self.logs_text.configure(state=tk.NORMAL)
        self.logs_text.insert(tk.END, line)
        self.logs_text.see(tk.END)
        self.logs_text.configure(state=tk.DISABLED)

    def _clear_logs(self) -> None:
        self.logs_text.configure(state=tk.NORMAL)
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.configure(state=tk.DISABLED)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
            self._update_stage_from_line(line)
        self.root.after(120, self._drain_log_queue)

    def _update_stage_marker(self, stage: str, status: str, value: float | None = None) -> None:
        if stage not in self.stage_vars:
            return
        self.stage_vars[stage].set(status)
        if value is not None:
            self.stage_progress[stage]["value"] = max(0, min(100, value))
        elif status == "done":
            self.stage_progress[stage]["value"] = 100
        elif status == "running" and self.stage_progress[stage]["value"] == 0:
            self.stage_progress[stage]["value"] = 5

    def _update_stage_from_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if "BUOC 1:" in line:
            self.current_stage = "BUOC 1: Doc metadata"
            self._update_stage_marker(self.current_stage, "running")
        elif "BUOC 2:" in line:
            self._update_stage_marker("BUOC 1: Doc metadata", "done")
            self.current_stage = "BUOC 2: Loc anh perched"
            self._update_stage_marker(self.current_stage, "running")
        elif "BUOC 3:" in line:
            self._update_stage_marker("BUOC 2: Loc anh perched", "done")
            self.current_stage = "BUOC 3: Crop + resize"
            self._update_stage_marker(self.current_stage, "running")
        elif "BUOC 6:" in line:
            self._update_stage_marker("BUOC 3: Crop + resize", "done")
            self.current_stage = "BUOC 6: Xay dung bo thuoc tinh"
            self._update_stage_marker(self.current_stage, "running")
        elif "Tang 1 - Custom attrs" in line:
            self._update_stage_marker("Tang 1: Custom attrs", "running")
            self._consume_tqdm_percent("Tang 1: Custom attrs", line)
            self._consume_tier_tqdm("tier1", line)
        elif "Tang 2 - Algorithmic" in line:
            self._update_stage_marker("Tang 2: Algorithmic 512D", "running")
            self._consume_tqdm_percent("Tang 2: Algorithmic 512D", line)
            self._consume_tier_tqdm("tier2", line)
        elif "Tang 3 - Handcrafted" in line:
            self._update_stage_marker("Tang 3: Handcrafted", "running")
            self._consume_tqdm_percent("Tang 3: Handcrafted", line)
            self._consume_tier_tqdm("tier3", line)
        elif "BUOC 7:" in line:
            self._update_stage_marker("BUOC 6: Xay dung bo thuoc tinh", "done")
            self._update_stage_marker("Tang 1: Custom attrs", "done")
            self._update_stage_marker("Tang 2: Algorithmic 512D", "done")
            self._update_stage_marker("Tang 3: Handcrafted", "done")
            self.current_stage = "BUOC 7: SQLite + FAISS"
            self._update_stage_marker(self.current_stage, "running")
        elif "[OK] Da xu ly thanh cong:" in line:
            self._update_stage_marker("BUOC 3: Crop + resize", "done")
        elif "HOAN THANH YEU CAU 1" in line:
            for stage in self.stage_order:
                if self.stage_vars[stage].get() in {"running", "pending"}:
                    self._update_stage_marker(stage, "done")

        if "Traceback (most recent call last):" in line:
            if self.current_stage:
                self._update_stage_marker(self.current_stage, "error")
            for key in ("tier1", "tier2", "tier3"):
                self._set_tier_runtime(key, state="error")

    def _consume_tqdm_percent(self, stage: str, line: str) -> None:
        match = re.search(r"(\d+)%", line)
        if not match:
            return
        pct = float(match.group(1))
        if pct >= 100.0:
            self._update_stage_marker(stage, "done", pct)
        else:
            self._update_stage_marker(stage, "running", pct)

    def _consume_tier_tqdm(self, tier_key: str, line: str) -> None:
        # Example:
        # "Tang 2 - Algorithmic 512D:  42%|...| 264/629 [00:08<00:12, 29.71it/s]"
        frac = re.search(r"(\d+)/(\d+)", line)
        speed = re.search(r"([0-9]+(?:\.[0-9]+)?)it/s", line)
        eta = re.search(r"\[([0-9:]+)<([0-9:]+),", line)
        if frac:
            self._set_tier_runtime(tier_key, state="running", count=f"{frac.group(1)} / {frac.group(2)}")
        if speed:
            self._set_tier_runtime(tier_key, speed=speed.group(1))
        if eta:
            self._set_tier_runtime(tier_key, eta=eta.group(2))
        if "100%" in line:
            self._set_tier_runtime(tier_key, state="done")

    def _refresh_artifacts(self) -> None:
        expected_paths = [
            "dataset_processed/metadata.csv",
            "dataset_processed/features/tier1_custom_attributes.csv",
            "dataset_processed/features/tier1_similarity_attributes.csv",
            "dataset_processed/features/tier1_difference_attributes.csv",
            "dataset_processed/features/tier2_algorithmic_embeddings.npy",
            "dataset_processed/features/tier3_handcrafted_features.csv",
            "dataset_processed/features/recognition_features_all.npz",
            "dataset_processed/reports/requirement2_attributes.md",
            "dataset_processed/reports/green_ratio_distribution.json",
            "dataset_processed/reports/centroid_y_by_class.csv",
            "dataset_processed/reports/custom_vs_resnet_correlation.csv",
            "birds.db",
            "birds.faiss",
        ]
        for item in self.artifact_tree.get_children():
            self.artifact_tree.delete(item)
        for rel in expected_paths:
            abs_path = os.path.join(self.project_root, rel.replace("/", os.sep))
            status = "exists" if os.path.exists(abs_path) else "missing"
            self.artifact_tree.insert("", tk.END, values=(rel, status))
        self._refresh_tier_summary()

    def _refresh_tier_summary(self) -> None:
        # Tier 1
        t1_path = os.path.join(self.project_root, "dataset_processed", "features", "tier1_custom_attributes.csv")
        if os.path.exists(t1_path):
            self.tier_detail_vars["tier1"]["file"].set("artifact: tier1_custom_attributes.csv (exists)")
            try:
                with open(t1_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                dims = max(0, len(header) - 2)  # minus img_id, filename
                self.tier_detail_vars["tier1"]["dim"].set(f"dimension/shape: ~{dims} attrs")
            except Exception:
                self.tier_detail_vars["tier1"]["dim"].set("dimension/shape: unknown")
        else:
            self.tier_detail_vars["tier1"]["file"].set("artifact: missing")
            self.tier_detail_vars["tier1"]["dim"].set("dimension/shape: -")

        # Tier 2
        t2_path = os.path.join(self.project_root, "dataset_processed", "features", "tier2_algorithmic_embeddings.npy")
        if os.path.exists(t2_path):
            self.tier_detail_vars["tier2"]["file"].set("artifact: tier2_algorithmic_embeddings.npy (exists)")
            try:
                import numpy as np

                arr = np.load(t2_path, mmap_mode="r")
                self.tier_detail_vars["tier2"]["dim"].set(f"dimension/shape: {tuple(arr.shape)}")
            except Exception:
                self.tier_detail_vars["tier2"]["dim"].set("dimension/shape: unavailable")
        else:
            self.tier_detail_vars["tier2"]["file"].set("artifact: missing")
            self.tier_detail_vars["tier2"]["dim"].set("dimension/shape: -")

        # Tier 3
        t3_path = os.path.join(self.project_root, "dataset_processed", "features", "tier3_handcrafted_features.csv")
        if os.path.exists(t3_path):
            self.tier_detail_vars["tier3"]["file"].set("artifact: tier3_handcrafted_features.csv (exists)")
            try:
                with open(t3_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                dims = len([c for c in header if c.startswith("hc_")])
                self.tier_detail_vars["tier3"]["dim"].set(f"dimension/shape: {dims} handcrafted dims")
            except Exception:
                self.tier_detail_vars["tier3"]["dim"].set("dimension/shape: unknown")
        else:
            self.tier_detail_vars["tier3"]["file"].set("artifact: missing")
            self.tier_detail_vars["tier3"]["dim"].set("dimension/shape: -")

    def run_selected_step(self) -> None:
        title = self.command_var.get().strip()
        command = next((c for c in self.commands if c.title == title), None)
        if command is None:
            messagebox.showerror("Error", "Cannot resolve selected command.")
            return
        self._run_commands([command])

    def run_full_workflow(self) -> None:
        self._run_commands(self.commands)

    def _run_commands(self, commands: list[WorkflowCommand]) -> None:
        if self.running:
            return
        self._clear_logs()
        self._reset_stage_status()
        self._set_controls_running(True)
        self.workflow_status_var.set("Running workflow...")
        self.worker_thread = threading.Thread(target=self._worker_run_commands, args=(commands,), daemon=True)
        self.worker_thread.start()

    def stop_workflow(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.log_queue.put("[INFO] Stop requested. Terminating current process...\n")

    def _worker_run_commands(self, commands: list[WorkflowCommand]) -> None:
        all_ok = True
        try:
            for idx, item in enumerate(commands):
                self.current_command_index = idx
                self.log_queue.put(f"\n=== RUN: {item.title} ===\n")
                self.log_queue.put(f"[CMD] {' '.join(item.command)}\n")
                rc = self._run_single_command(item)
                if rc != 0:
                    all_ok = False
                    self.log_queue.put(f"[ERROR] Command failed with exit code {rc}: {item.title}\n")
                    break
            if all_ok:
                self.log_queue.put("\n[OK] Workflow completed successfully.\n")
        finally:
            self.proc = None
            self.current_command_index = -1
            self.root.after(0, self._on_workflow_finished, all_ok)

    def _run_single_command(self, item: WorkflowCommand) -> int:
        self.proc = subprocess.Popen(
            item.command,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log_queue.put(line)
        return int(self.proc.wait())

    def _on_workflow_finished(self, success: bool) -> None:
        self._set_controls_running(False)
        self._refresh_artifacts()
        if success:
            self.workflow_status_var.set("Completed")
            for key in ("tier1", "tier2", "tier3"):
                if self.tier_detail_vars[key]["state"].get() == "state: pending":
                    self._set_tier_runtime(key, state="done")
        else:
            self.workflow_status_var.set("Failed (see logs)")


def main() -> None:
    root = tk.Tk()
    app = PipelineDashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
