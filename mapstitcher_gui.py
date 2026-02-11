import os
import sys
import shutil
import shlex
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class MapStitcherGUI(tk.Tk):
    """
    Tkinter wrapper for https://github.com/mimunzar/mapstitcher
    Runs image_stitch_batch.py via subprocess and streams output to a log box.

    OpenJPEG lookup (Windows-friendly):
      - Prefer ./openjpeg/bin relative to THIS GUI script (not repo)
      - Falls back to PATH if not found
    """

    def __init__(self):
        super().__init__()
        self.title("MapStitcher GUI (image_stitch_batch.py)")
        self.geometry("980x650")

        self.proc: subprocess.Popen | None = None

        # Directory where THIS GUI script lives (used for OpenJPEG lookup)
        self.gui_root = Path(__file__).resolve().parent

        # --- Paths / inputs
        self.repo_dir = tk.StringVar(value="")
        self.mode = tk.StringVar(value="path")  # "path" or "list"
        self.input_folder = tk.StringVar(value="")
        self.list_file = tk.StringVar(value="")
        self.output_file = tk.StringVar(value=str(Path.cwd() / "result.jp2"))

        # --- Params
        self.optimization_model = tk.StringVar(value="affine")  # affine|homography
        self.matching_algorithm = tk.StringVar(value="loftr")   # loftr|sift
        self.loftr_model = tk.StringVar(value="outdoor")        # outdoor|indoor
        self.flow_alg = tk.StringVar(value="raft")              # raft|cv
        self.subsample_flow = tk.DoubleVar(value=2.0)
        self.vram_size = tk.DoubleVar(value=8.0)
        self.max_matches = tk.IntVar(value=800)

        self.debug = tk.BooleanVar(value=False)
        self.silent = tk.BooleanVar(value=False)

        self.extra_args = tk.StringVar(value="")  # advanced, optional

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # Repo directory
        repo = ttk.LabelFrame(
            outer,
            text="1) MapStitcher project folder (must contain image_stitch_batch.py)"
        )
        repo.pack(fill="x", padx=5, pady=5)

        ttk.Label(repo, text="Repo folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(repo, textvariable=self.repo_dir, width=80).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(repo, text="Browse…", command=self.pick_repo).grid(row=0, column=2, padx=5, pady=5)
        repo.columnconfigure(1, weight=1)

        # Input selection
        inp = ttk.LabelFrame(outer, text="2) Input")
        inp.pack(fill="x", padx=5, pady=5)

        mode_row = ttk.Frame(inp)
        mode_row.pack(fill="x", padx=5, pady=5)

        ttk.Label(mode_row, text="Mode:").pack(side="left")
        ttk.Radiobutton(
            mode_row, text="Folder (--path)", variable=self.mode, value="path",
            command=self._refresh_mode
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            mode_row, text="List file (--list)", variable=self.mode, value="list",
            command=self._refresh_mode
        ).pack(side="left", padx=8)

        self.path_row = ttk.Frame(inp)
        self.path_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(self.path_row, text="Input folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.path_row, textvariable=self.input_folder, width=80).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(self.path_row, text="Browse…", command=self.pick_input_folder).grid(row=0, column=2)
        self.path_row.columnconfigure(1, weight=1)

        self.list_row = ttk.Frame(inp)
        self.list_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(self.list_row, text="List file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.list_row, textvariable=self.list_file, width=80).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(self.list_row, text="Browse…", command=self.pick_list_file).grid(row=0, column=2)
        self.list_row.columnconfigure(1, weight=1)

        out = ttk.Frame(inp)
        out.pack(fill="x", padx=5, pady=5)
        ttk.Label(out, text="Output (--output):").grid(row=0, column=0, sticky="w")
        ttk.Entry(out, textvariable=self.output_file, width=80).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(out, text="Save as…", command=self.pick_output_file).grid(row=0, column=2)
        out.columnconfigure(1, weight=1)

        # Parameters
        params = ttk.LabelFrame(outer, text="3) Parameters")
        params.pack(fill="x", padx=5, pady=5)

        grid = ttk.Frame(params)
        grid.pack(fill="x", padx=5, pady=5)

        # Optimization model
        opt_box = ttk.LabelFrame(grid, text="Optimization model")
        opt_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Radiobutton(opt_box, text="affine", variable=self.optimization_model, value="affine").pack(anchor="w")
        ttk.Radiobutton(opt_box, text="homography", variable=self.optimization_model, value="homography").pack(anchor="w")

        # Matching algorithm
        match_box = ttk.LabelFrame(grid, text="Matching algorithm")
        match_box.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ttk.Radiobutton(
            match_box, text="loftr", variable=self.matching_algorithm, value="loftr",
            command=self._refresh_loftr_state
        ).pack(anchor="w")
        ttk.Radiobutton(
            match_box, text="sift", variable=self.matching_algorithm, value="sift",
            command=self._refresh_loftr_state
        ).pack(anchor="w")

        # LoFTR model (only meaningful if matching=loftr)
        loftr_box = ttk.LabelFrame(grid, text="LoFTR model")
        loftr_box.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.loftr_outdoor = ttk.Radiobutton(loftr_box, text="outdoor", variable=self.loftr_model, value="outdoor")
        self.loftr_indoor = ttk.Radiobutton(loftr_box, text="indoor", variable=self.loftr_model, value="indoor")
        self.loftr_outdoor.pack(anchor="w")
        self.loftr_indoor.pack(anchor="w")

        # Flow algorithm
        flow_box = ttk.LabelFrame(grid, text="Optical flow (flow-alg)")
        flow_box.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)
        ttk.Radiobutton(flow_box, text="raft (recommended)", variable=self.flow_alg, value="raft").pack(anchor="w")
        ttk.Radiobutton(flow_box, text="cv (repo note: not working)", variable=self.flow_alg, value="cv").pack(anchor="w")

        # Numeric params
        num_box = ttk.LabelFrame(params, text="Numeric options")
        num_box.pack(fill="x", padx=5, pady=5)

        r = ttk.Frame(num_box)
        r.pack(fill="x", padx=5, pady=5)

        ttk.Label(r, text="subsample-flow:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(r, from_=0.25, to=16.0, increment=0.25, textvariable=self.subsample_flow, width=10)\
            .grid(row=0, column=1, padx=5)

        ttk.Label(r, text="vram-size (GB):").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(r, from_=1.0, to=64.0, increment=1.0, textvariable=self.vram_size, width=10)\
            .grid(row=0, column=3, padx=5)

        ttk.Label(r, text="max-matches:").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(r, from_=50, to=10000, increment=50, textvariable=self.max_matches, width=10)\
            .grid(row=0, column=5, padx=5)

        # Checkboxes
        flags = ttk.LabelFrame(params, text="Flags")
        flags.pack(fill="x", padx=5, pady=5)

        f = ttk.Frame(flags)
        f.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(f, text="--debug", variable=self.debug).pack(side="left", padx=10)
        ttk.Checkbutton(f, text="--silent", variable=self.silent).pack(side="left", padx=10)

        # Extra args
        extra = ttk.LabelFrame(outer, text="4) Advanced (optional) – extra CLI arguments")
        extra.pack(fill="x", padx=5, pady=5)
        ttk.Entry(extra, textvariable=self.extra_args, width=120).pack(fill="x", padx=5, pady=5)

        # Run controls
        controls = ttk.Frame(outer)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(controls, text="Run", command=self.run).pack(side="left")
        ttk.Button(controls, text="Stop (terminate)", command=self.stop).pack(side="left", padx=8)
        ttk.Button(controls, text="Show command", command=self.show_command).pack(side="left", padx=8)

        # Log
        logf = ttk.LabelFrame(outer, text="Log")
        logf.pack(fill="both", expand=True, padx=5, pady=5)
        self.log = tk.Text(logf, wrap="word")
        self.log.pack(fill="both", expand=True, padx=5, pady=5)

        self._refresh_mode()
        self._refresh_loftr_state()

    # -------------- helpers --------------
    def append_log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")

    def pick_repo(self):
        p = filedialog.askdirectory(title="Select the mapstitcher repository folder")
        if p:
            self.repo_dir.set(p)

    def pick_input_folder(self):
        p = filedialog.askdirectory(title="Select input folder with images")
        if p:
            self.input_folder.set(p)

    def pick_list_file(self):
        p = filedialog.askopenfilename(
            title="Select list.txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if p:
            self.list_file.set(p)

    def pick_output_file(self):
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".jp2",
            filetypes=[
                ("JPEG2000", "*.jp2"),
                ("TIFF", "*.tif;*.tiff"),
                ("PNG", "*.png"),
                ("JPG", "*.jpg;*.jpeg"),
                ("All files", "*.*")
            ],
        )
        if p:
            self.output_file.set(p)

    def _refresh_mode(self):
        is_path = self.mode.get() == "path"
        for child in self.path_row.winfo_children():
            child.configure(state=("normal" if is_path else "disabled"))
        for child in self.list_row.winfo_children():
            child.configure(state=("disabled" if is_path else "normal"))

    def _refresh_loftr_state(self):
        loftr = (self.matching_algorithm.get() == "loftr")
        state = ("normal" if loftr else "disabled")
        self.loftr_outdoor.configure(state=state)
        self.loftr_indoor.configure(state=state)
        if not loftr:
            self.loftr_model.set("outdoor")

    # -------------- OpenJPEG lookup --------------
    def _find_opj_compress(self) -> str | None:
        """
        Prefer ./openjpeg/bin/opj_compress(.exe) relative to this GUI script.
        Fall back to PATH.
        """
        exe = "opj_compress.exe" if os.name == "nt" else "opj_compress"
        local = self.gui_root / "openjpeg" / "bin" / exe
        if local.is_file():
            return str(local)
        which = shutil.which("opj_compress")
        return which

    # -------------- command building --------------
    def build_command(self) -> tuple[list[str], dict]:
        """
        Returns (cmd, env). env is a copy of os.environ possibly modified (OpenJPEG).
        """
        repo = Path(self.repo_dir.get()).expanduser().resolve()
        script = repo / "image_stitch_batch.py"
        if not script.is_file():
            raise ValueError("image_stitch_batch.py not found. Please select the correct mapstitcher repo folder.")

        cmd = [sys.executable, str(script)]

        # input mode
        if self.mode.get() == "path":
            p = Path(self.input_folder.get()).expanduser()
            if not p.is_dir():
                raise ValueError("Input folder (--path) does not exist.")
            cmd += ["--path", str(p)]
        else:
            lf = Path(self.list_file.get()).expanduser()
            if not lf.is_file():
                raise ValueError("List file (--list) does not exist.")
            cmd += ["--list", str(lf)]

        # output
        out = Path(self.output_file.get()).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--output", str(out)]

        # radios / options
        cmd += ["--optimization-model", self.optimization_model.get()]
        cmd += ["--matching-algorithm", self.matching_algorithm.get()]
        if self.matching_algorithm.get() == "loftr":
            cmd += ["--loftr-model", self.loftr_model.get()]
        cmd += ["--flow-alg", self.flow_alg.get()]

        # numerics
        cmd += ["--subsample-flow", str(float(self.subsample_flow.get()))]
        cmd += ["--vram-size", str(float(self.vram_size.get()))]
        cmd += ["--max-matches", str(int(self.max_matches.get()))]

        # flags
        if self.debug.get():
            cmd.append("--debug")
        if self.silent.get():
            cmd.append("--silent")

        # extra args (optional)
        extra = self.extra_args.get().strip()
        if extra:
            cmd += shlex.split(extra, posix=(os.name != "nt"))

        # Environment: ensure local OpenJPEG is discoverable for subprocess
        env = os.environ.copy()
        opj = self._find_opj_compress()
        if opj:
            opj_dir = str(Path(opj).resolve().parent)
            # Prepend to PATH so the called script can find it when it runs subprocess(["opj_compress", ...])
            env["PATH"] = opj_dir + os.pathsep + env.get("PATH", "")
        return cmd, env

    def show_command(self):
        try:
            cmd, env = self.build_command()
            self.append_log("\nCOMMAND:\n" + " ".join(shlex.quote(x) for x in cmd) + "\n")
            opj = self._find_opj_compress()
            if opj:
                self.append_log(f"OpenJPEG: using {opj}\n\n")
            else:
                self.append_log("OpenJPEG: opj_compress not found (./openjpeg/bin or PATH)\n\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------- run/stop --------------
    def run(self):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "A process is already running.")
            return

        try:
            cmd, env = self.build_command()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Warn if output is JP2 and OpenJPEG is missing
        out_suffix = Path(self.output_file.get()).suffix.lower()
        opj = self._find_opj_compress()
        if out_suffix == ".jp2" and not opj:
            messagebox.showwarning(
                "OpenJPEG not found",
                "Output is .jp2, but opj_compress was not found in ./openjpeg/bin or PATH.\n"
                "JP2 writing may fail. Consider switching output to .tif/.png or install OpenJPEG."
            )

        self.append_log("\n=== Running ===\n")
        self.append_log(" ".join(shlex.quote(x) for x in cmd) + "\n\n")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(Path(self.repo_dir.get()).expanduser().resolve()),
                    env=env,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.append_log(line)
                rc = self.proc.wait()
                self.append_log(f"\n=== Finished (exit {rc}) ===\n")
            except Exception as e:
                self.append_log(f"\n[ERROR] {e}\n")
            finally:
                self.proc = None

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.append_log("\n[INFO] terminate() sent.\n")
        else:
            self.append_log("\n[INFO] No running process.\n")


if __name__ == "__main__":
    # Better DPI on Windows (safe no-op elsewhere)
    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = MapStitcherGUI()
    app.mainloop()