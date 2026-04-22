"""
NoCode CV Trainer – GUI Installer
Requires: Python 3.9+ (tkinter is built-in)
Run via:  Install NoCode CV.bat  (or  python installer.py)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess, threading, os, sys, shutil, webbrowser, json, platform

# ─── Constants ────────────────────────────────────────────────────────────────
APP_NAME     = "NoCode CV Trainer"
APP_VERSION  = "1.0.0"
WIN_W, WIN_H = 760, 520
SIDE_W       = 190

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")
BACKEND  = os.path.join(ROOT_DIR, "backend")
FRONTEND = os.path.join(ROOT_DIR, "frontend")
DIST     = os.path.join(FRONTEND, "dist")

# Python inside venv
if platform.system() == "Windows":
    VENV_PY  = os.path.join(VENV_DIR, "Scripts", "python.exe")
    VENV_PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    VENV_PY  = os.path.join(VENV_DIR, "bin", "python")
    VENV_PIP = os.path.join(VENV_DIR, "bin", "pip")

# ─── Colour palette (matches the app dark theme) ──────────────────────────────
BG      = "#0d0d0f"
SURFACE = "#16161a"
SURF2   = "#1e1e24"
BORDER  = "#2a2a35"
ACCENT  = "#5865f2"
ACCENT2 = "#4752c4"
TXT     = "#e2e2e2"
TXT2    = "#8b8b9a"
SUCCESS = "#3ba55d"
ERROR   = "#ed4245"
WARN    = "#faa61a"

STEPS = ["Welcome", "System Check", "Configure", "Install", "Done"]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def run_cmd(cmd, cwd=None, env=None):
    """Run a subprocess and return (returncode, stdout+stderr)."""
    proc = subprocess.run(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=(platform.system() == "Windows"),
    )
    return proc.returncode, proc.stdout or ""


def run_cmd_stream(cmd, cwd=None, line_cb=None):
    """Run a subprocess and call line_cb(line) for each output line in real-time."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        shell=(platform.system() == "Windows"),
        bufsize=1,
    )
    lines = []
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip("\r\n")
        if line.strip():
            lines.append(line)
            if line_cb:
                line_cb(line)
    proc.wait()
    return proc.returncode, "\n".join(lines)


def check_tool(name):
    """Return version string or None."""
    try:
        rc, out = run_cmd(f"{name} --version")
        if rc == 0:
            return out.strip().split("\n")[0]
    except Exception:
        pass
    return None


def get_free_gb(path):
    try:
        s = shutil.disk_usage(path)
        return s.free / (1024 ** 3)
    except Exception:
        return 999


def create_desktop_shortcut(launcher_bat):
    """Create a Windows .lnk desktop shortcut via PowerShell."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    lnk     = os.path.join(desktop, f"{APP_NAME}.lnk")
    ps = (
        f'$ws = New-Object -ComObject WScript.Shell;'
        f'$s  = $ws.CreateShortcut("{lnk}");'
        f'$s.TargetPath  = "{launcher_bat}";'
        f'$s.WorkingDirectory = "{ROOT_DIR}";'
        f'$s.Description = "{APP_NAME}";'
        f'$s.Save()'
    )
    try:
        subprocess.run(["powershell", "-Command", ps], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ─── Main Installer Window ────────────────────────────────────────────────────
class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} – Setup {APP_VERSION}")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center()
        self._set_icon()

        # State
        self.step           = 0          # current page index
        self.checks         = {}         # {name: (ok, detail)}
        self.torch_choice   = tk.StringVar(value="auto")
        self.shortcut_var   = tk.BooleanVar(value=True)
        self._install_done  = False
        self.detected_cuda  = None       # e.g. "12.8" from nvidia-smi
        self.detected_gpu   = None       # e.g. "NVIDIA GeForce RTX 5070 Ti"

        self._build_ui()
        self._show_step(0)

    # ── Window setup ──────────────────────────────────────────────────────────
    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - WIN_W) // 2
        y = (self.winfo_screenheight() - WIN_H) // 2
        self.geometry(f"+{x}+{y}")

    def _set_icon(self):
        try:
            # Use a simple photo image as icon (no external file needed)
            ico = tk.PhotoImage(width=1, height=1)
            self.iconphoto(True, ico)
        except Exception:
            pass

    # ── UI skeleton ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left sidebar ──────────────────────────────────────────────────────
        self.sidebar = tk.Frame(self, bg=SURFACE, width=SIDE_W)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="🧠", font=("Segoe UI Emoji", 28),
                 bg=SURFACE, fg=ACCENT).pack(pady=(32, 4))
        tk.Label(self.sidebar, text=APP_NAME, font=("Segoe UI", 11, "bold"),
                 bg=SURFACE, fg=TXT, wraplength=160, justify="center").pack(pady=(0, 4))
        tk.Label(self.sidebar, text=f"v{APP_VERSION}",
                 font=("Segoe UI", 9), bg=SURFACE, fg=TXT2).pack()

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=20)

        self.step_labels = []
        for i, name in enumerate(STEPS):
            row = tk.Frame(self.sidebar, bg=SURFACE)
            row.pack(fill="x", padx=0, pady=1)

            dot = tk.Label(row, text="●", font=("Segoe UI", 9),
                           bg=SURFACE, fg=TXT2, width=3)
            dot.pack(side="left", padx=(16, 4))

            lbl = tk.Label(row, text=name, font=("Segoe UI", 9),
                           bg=SURFACE, fg=TXT2, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)

            self.step_labels.append((row, dot, lbl))

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # IMPORTANT: pack the bottom bar FIRST so it is always visible.
        # The scrollable content area then fills whatever space remains.

        # ── Bottom bar (packed to bottom before content) ──────────────────────
        tk.Frame(right, bg=BORDER, height=1).pack(side="bottom", fill="x")

        bar = tk.Frame(right, bg=SURFACE)
        bar.pack(side="bottom", fill="x")

        self.btn_back = tk.Button(bar, text="← Back",
                                  font=("Segoe UI", 9), bg=SURF2, fg=TXT2,
                                  relief="flat", bd=0, padx=18, pady=9,
                                  cursor="hand2", activebackground=SURF2,
                                  activeforeground=TXT, command=self._go_back)
        self.btn_back.pack(side="right", padx=(4, 16), pady=10)

        self.btn_next = tk.Button(bar, text="Next  →",
                                  font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="#fff",
                                  relief="flat", bd=0, padx=20, pady=9,
                                  cursor="hand2", activebackground=ACCENT2,
                                  activeforeground="#fff", command=self._go_next)
        self.btn_next.pack(side="right", padx=(0, 4), pady=10)

        # ── Scrollable content area (fills remaining space above button bar) ──
        wrap = tk.Frame(right, bg=BG)
        wrap.pack(side="top", fill="both", expand=True)

        self._cv = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        self._sb = tk.Scrollbar(wrap, orient="vertical", command=self._cv.yview,
                                bg=SURF2, troughcolor=BG, relief="flat", width=10)
        self._cv.configure(yscrollcommand=self._sb.set)
        self._cv.pack(side="left", fill="both", expand=True)
        # Scrollbar shown only when content overflows (see _refresh_scroll)

        # Inner frame: this is what each page populates
        self._inner = tk.Frame(self._cv, bg=BG)
        self._cwin  = self._cv.create_window((0, 0), window=self._inner, anchor="nw")

        # Wrapper adds consistent padding around page content
        self.content = tk.Frame(self._inner, bg=BG)
        self.content.pack(fill="both", expand=True, padx=28, pady=22)

        def _on_inner_resize(_e=None):
            self._cv.configure(scrollregion=self._cv.bbox("all"))
            self._refresh_scroll()

        def _on_canvas_resize(e):
            self._cv.itemconfig(self._cwin, width=e.width)
            _on_inner_resize()

        self._inner.bind("<Configure>", _on_inner_resize)
        self._cv.bind("<Configure>", _on_canvas_resize)

        # Mouse-wheel scrolling (Windows)
        self._cv.bind_all("<MouseWheel>",
            lambda e: self._cv.yview_scroll(int(-1 * e.delta / 120), "units"))

    def _refresh_scroll(self):
        """Show scrollbar only when content is taller than the canvas."""
        self._cv.update_idletasks()
        content_h = self._inner.winfo_reqheight()
        canvas_h  = self._cv.winfo_height()
        if content_h > canvas_h + 4:
            self._sb.pack(side="right", fill="y")
        else:
            self._sb.pack_forget()

    # ── Step navigation ───────────────────────────────────────────────────────
    def _update_sidebar(self):
        for i, (row, dot, lbl) in enumerate(self.step_labels):
            if i < self.step:
                dot.config(fg=SUCCESS, text="✓")
                lbl.config(fg=TXT2)
                row.config(bg=SURFACE)
            elif i == self.step:
                dot.config(fg=ACCENT, text="●")
                lbl.config(fg=TXT, font=("Segoe UI", 9, "bold"))
                row.config(bg=SURF2)
            else:
                dot.config(fg=BORDER, text="○")
                lbl.config(fg=TXT2, font=("Segoe UI", 9))
                row.config(bg=SURFACE)

    def _show_step(self, idx):
        self.step = idx
        self._update_sidebar()
        for w in self.content.winfo_children():
            w.destroy()
        # Reset scroll to top for each new page
        self._cv.yview_moveto(0)

        pages = [
            self._page_welcome,
            self._page_syscheck,
            self._page_configure,
            self._page_install,
            self._page_done,
        ]
        pages[idx]()

        # Back button visibility
        self.btn_back.config(state="normal" if idx > 0 else "disabled",
                             fg=TXT2 if idx > 0 else BORDER)
        # Next button text
        if idx == len(STEPS) - 1:
            self.btn_next.config(text="Close", command=self.destroy)
        elif idx == 3:
            self.btn_next.config(text="Installing…", state="disabled")
        else:
            self.btn_next.config(text="Next  →", state="normal", command=self._go_next)

    def _go_next(self):
        if self.step == 1 and not self._checks_pass():
            return
        if self.step == 2:
            self._show_step(3)
            self._start_install()
            return
        self._show_step(min(self.step + 1, len(STEPS) - 1))

    def _go_back(self):
        if self.step > 0 and self.step != 3:
            self._show_step(self.step - 1)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 0 – Welcome
    # ──────────────────────────────────────────────────────────────────────────
    def _page_welcome(self):
        c = self.content
        tk.Label(c, text=f"Welcome to {APP_NAME}", bg=BG, fg=TXT,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(c, text="AI-powered computer vision annotation, training & evaluation.",
                 bg=BG, fg=TXT2, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 20))

        # Feature list
        features = [
            ("🖼", "Annotation Studio",   "BBox, polygon & point tools with undo/redo"),
            ("⚡", "YOLOv8 Training",     "Real-time training logs, augmentation control"),
            ("🔬", "Evaluation & Infer",  "mAP metrics, per-class breakdown, live inference"),
            ("🏷", "Classification",      "ResNet / MobileNet / EfficientNet transfer learning"),
            ("📦", "Dataset Export",      "YOLO & COCO zip export in one click"),
        ]
        for icon, title, desc in features:
            row = tk.Frame(c, bg=SURF2, bd=0, relief="flat")
            row.pack(fill="x", pady=3, ipady=8, ipadx=10)
            tk.Label(row, text=icon, font=("Segoe UI Emoji", 14), bg=SURF2).pack(side="left", padx=(12,8))
            inner = tk.Frame(row, bg=SURF2)
            inner.pack(side="left", fill="x")
            tk.Label(inner, text=title, font=("Segoe UI", 9, "bold"),
                     bg=SURF2, fg=TXT, anchor="w").pack(anchor="w")
            tk.Label(inner, text=desc, font=("Segoe UI", 8),
                     bg=SURF2, fg=TXT2, anchor="w").pack(anchor="w")

        tk.Label(c, text=f"Install location:  {ROOT_DIR}",
                 bg=BG, fg=TXT2, font=("Courier New", 8)).pack(anchor="w", pady=(16, 0))

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1 – System Check
    # ──────────────────────────────────────────────────────────────────────────
    def _page_syscheck(self):
        c = self.content
        tk.Label(c, text="System Requirements", bg=BG, fg=TXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(c, text="Checking your system before installation…",
                 bg=BG, fg=TXT2, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 16))

        dist_prebuilt = os.path.isfile(os.path.join(DIST, "index.html"))
        node_note = "Optional — pre-built frontend included" if dist_prebuilt else "Required  (nodejs.org)"
        npm_note  = "Optional — pre-built frontend included" if dist_prebuilt else "Comes with Node.js"
        self.check_rows = {}
        items = [
            ("python",  "Python 3.9+",          "Required"),
            ("node",    "Node.js 16+",           node_note),
            ("npm",     "npm",                   npm_note),
            ("gpu",     "NVIDIA GPU",            "Optional (CPU fallback)"),
            ("cuda",    "CUDA Toolkit",          "Optional — needed for GPU training"),
            ("disk",    "Free disk space ≥2 GB", "Required"),
        ]
        for key, label, note in items:
            row = tk.Frame(c, bg=SURF2)
            row.pack(fill="x", pady=2, ipady=5, ipadx=10)

            dot = tk.Label(row, text="⏳", font=("Segoe UI Emoji", 11), bg=SURF2, width=3)
            dot.pack(side="left", padx=(10, 6))

            tk.Label(row, text=label, font=("Segoe UI", 9, "bold"),
                     bg=SURF2, fg=TXT, width=22, anchor="w").pack(side="left")

            val = tk.Label(row, text="checking…", font=("Courier New", 8),
                           bg=SURF2, fg=TXT2, anchor="w")
            val.pack(side="left", padx=8)

            self.check_rows[key] = (dot, val)

        # Hint area (shown conditionally after checks)
        self.hint_frame = tk.Frame(c, bg=BG)
        self.hint_frame.pack(fill="x", pady=(6, 0))

        # Run checks in background
        threading.Thread(target=self._run_checks, daemon=True).start()

    def _run_checks(self):
        import re
        results = {}

        # ── Python ────────────────────────────────────────────────────────────
        v = sys.version.split()[0]
        parts = v.split(".")
        ok = int(parts[0]) >= 3 and int(parts[1]) >= 9
        results["python"] = (ok, f"Python {v}")

        # ── Node.js ───────────────────────────────────────────────────────────
        nv = check_tool("node")
        if nv:
            num = nv.replace("v", "").split(".")[0]
            ok_n = int(num) >= 16
            results["node"] = (ok_n, nv[:40])
        else:
            results["node"] = (False, "Not found")

        # ── npm ───────────────────────────────────────────────────────────────
        nv2 = check_tool("npm")
        results["npm"] = (bool(nv2), nv2[:40] if nv2 else "Not found")

        # ── NVIDIA GPU (via nvidia-smi) ───────────────────────────────────────
        # nvidia-smi is installed alongside the NVIDIA driver — no PyTorch needed.
        gpu_name   = None
        cuda_ver   = None
        driver_ver = None
        try:
            rc, smi = run_cmd("nvidia-smi")
            if rc == 0 and smi:
                # GPU name  e.g. "NVIDIA GeForce RTX 5070 Ti"
                m = re.search(r'\|\s+(NVIDIA\s+[^\|]+?)\s+\d{5,}', smi)
                if m:
                    gpu_name = m.group(1).strip()
                # Driver version  e.g. "Driver Version: 572.70"
                m2 = re.search(r'Driver Version:\s*([\d.]+)', smi)
                if m2:
                    driver_ver = m2.group(1)
                # CUDA version reported by driver  e.g. "CUDA Version: 12.8"
                m3 = re.search(r'CUDA Version:\s*([\d.]+)', smi)
                if m3:
                    cuda_ver = m3.group(1)
        except Exception:
            pass

        # Try querying GPU name more reliably
        if not gpu_name:
            try:
                rc2, name_out = run_cmd(
                    "nvidia-smi --query-gpu=name --format=csv,noheader"
                )
                if rc2 == 0 and name_out.strip():
                    gpu_name = name_out.strip().split("\n")[0].strip()
            except Exception:
                pass

        if gpu_name:
            label = gpu_name[:46]
            if driver_ver:
                label += f"  (driver {driver_ver})"
            results["gpu"] = (True, label)
        else:
            results["gpu"] = (None, "No NVIDIA GPU detected")

        # ── CUDA Toolkit ──────────────────────────────────────────────────────
        if cuda_ver:
            results["cuda"] = (True, f"CUDA {cuda_ver}  (driver-reported)")
        elif gpu_name:
            # GPU present but CUDA version not readable from smi
            # Try nvcc (full toolkit install)
            nvcc = check_tool("nvcc")
            if nvcc:
                m4 = re.search(r'release\s*([\d.]+)', nvcc, re.IGNORECASE)
                cuda_ver = m4.group(1) if m4 else nvcc.split()[-1]
                results["cuda"] = (True, f"CUDA {cuda_ver}  (nvcc)")
            else:
                results["cuda"] = (False, "CUDA Toolkit not found")
        else:
            results["cuda"] = (None, "No GPU — CUDA not needed")

        # Store for Configure page auto-selection
        self.detected_cuda = cuda_ver
        self.detected_gpu  = gpu_name

        # ── Disk space ────────────────────────────────────────────────────────
        gb = get_free_gb(ROOT_DIR)
        results["disk"] = (gb >= 2.0, f"{gb:.1f} GB free")

        self.checks = results
        self.after(0, self._apply_check_results)

    def _apply_check_results(self):
        icons = {True: ("✓", SUCCESS), False: ("✗", ERROR), None: ("ℹ", TXT2)}
        for key, (dot, val) in self.check_rows.items():
            if key not in self.checks:
                continue
            ok, detail = self.checks[key]
            sym, col = icons[ok]
            dot.config(text=sym, fg=col)
            val.config(text=detail, fg=TXT if ok is not False else ERROR)

        # Clear old hints
        for w in self.hint_frame.winfo_children():
            w.destroy()

        def _hint(text, color, url=None):
            lbl = tk.Label(self.hint_frame, text=text, bg=BG, fg=color,
                           font=("Segoe UI", 8), cursor="hand2" if url else "arrow",
                           anchor="w")
            lbl.pack(anchor="w", pady=1)
            if url:
                lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        # Node.js hint
        if not self.checks.get("node", (False,))[0]:
            if os.path.isfile(os.path.join(DIST, "index.html")):
                _hint("ℹ  Node.js not found — pre-built frontend detected, skipping build step.",
                      TXT2)
            else:
                _hint("⚠  Node.js not found — click to download  nodejs.org",
                      WARN, "https://nodejs.org/en/download")

        # CUDA hints
        gpu_ok   = self.checks.get("gpu",  (None,))[0]
        cuda_ok  = self.checks.get("cuda", (None,))[0]

        if gpu_ok and not cuda_ok:
            # GPU present but CUDA toolkit missing
            _hint("⚠  CUDA Toolkit not installed.  GPU training will NOT work without it.",
                  WARN)
            _hint("   → Download CUDA Toolkit (recommended: CUDA 12.4 or 12.8)",
                  ACCENT, "https://developer.nvidia.com/cuda-downloads")
            _hint("   After installing CUDA, re-run this installer and choose a CUDA PyTorch variant.",
                  TXT2)
        elif not gpu_ok:
            # No GPU at all
            _hint("ℹ  No NVIDIA GPU detected.  The app will run in CPU mode (slower training).",
                  TXT2)

        # Auto-select best PyTorch variant based on detected CUDA
        self._auto_select_torch()

        if self._checks_pass():
            self.btn_next.config(state="normal")
        else:
            self.btn_next.config(state="disabled")

    def _auto_select_torch(self):
        """Pick the best torch_choice based on detected CUDA version."""
        cv = self.detected_cuda  # e.g. "12.8", "11.8", None
        if cv is None:
            if self.detected_gpu:
                # GPU but no CUDA toolkit yet — keep "auto" as placeholder
                self.torch_choice.set("auto")
            else:
                self.torch_choice.set("cpu")
            return
        try:
            major = int(cv.split(".")[0])
            minor = int(cv.split(".")[1]) if "." in cv else 0
        except ValueError:
            return
        if major > 12 or (major == 12 and minor >= 5):
            self.torch_choice.set("cu128")   # CUDA 12.5+ → cu128 (Blackwell/RTX 50xx)
        elif major == 12:
            self.torch_choice.set("cu124")   # CUDA 12.0–12.4
        elif major == 11 and minor >= 8:
            self.torch_choice.set("cu118")
        else:
            self.torch_choice.set("cpu")     # CUDA too old, use CPU build

    def _checks_pass(self):
        must = ["python", "disk"]
        # Node.js only required if the frontend hasn't been pre-built
        if not os.path.isfile(os.path.join(DIST, "index.html")):
            must += ["node", "npm"]
        return all(self.checks.get(k, (False,))[0] for k in must)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2 – Configure
    # ──────────────────────────────────────────────────────────────────────────
    def _page_configure(self):
        c = self.content
        tk.Label(c, text="Configure Installation", bg=BG, fg=TXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(c, text="Choose your PyTorch backend and options.",
                 bg=BG, fg=TXT2, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        # ── Detected hardware summary ──────────────────────────────────────────
        hw_frame = tk.Frame(c, bg=SURF2)
        hw_frame.pack(fill="x", pady=(0, 10), ipady=7, ipadx=10)

        gpu_ok  = self.checks.get("gpu",  (None,))[0]
        cuda_ok = self.checks.get("cuda", (None,))[0]

        gpu_text  = self.detected_gpu  or "No NVIDIA GPU detected"
        cuda_text = (f"CUDA {self.detected_cuda}" if self.detected_cuda
                     else ("Not installed" if gpu_ok else "N/A"))

        gpu_icon  = "✓" if gpu_ok  else ("ℹ" if gpu_ok is None else "✗")
        cuda_icon = "✓" if cuda_ok else ("⚠" if gpu_ok else "ℹ")
        gpu_col   = SUCCESS if gpu_ok else (TXT2 if gpu_ok is None else ERROR)
        cuda_col  = SUCCESS if cuda_ok else (WARN if gpu_ok else TXT2)

        row1 = tk.Frame(hw_frame, bg=SURF2)
        row1.pack(fill="x")
        tk.Label(row1, text=f" {gpu_icon} GPU:   ", font=("Courier New", 8),
                 bg=SURF2, fg=gpu_col).pack(side="left")
        tk.Label(row1, text=gpu_text[:56], font=("Courier New", 8),
                 bg=SURF2, fg=TXT).pack(side="left")

        row2 = tk.Frame(hw_frame, bg=SURF2)
        row2.pack(fill="x", pady=(2, 0))
        tk.Label(row2, text=f" {cuda_icon} CUDA:  ", font=("Courier New", 8),
                 bg=SURF2, fg=cuda_col).pack(side="left")
        tk.Label(row2, text=cuda_text, font=("Courier New", 8),
                 bg=SURF2, fg=TXT).pack(side="left")

        # CUDA not installed but GPU present → show download button inline
        if gpu_ok and not cuda_ok:
            row3 = tk.Frame(hw_frame, bg=SURF2)
            row3.pack(fill="x", pady=(4, 0))
            tk.Label(row3, text="   CUDA Toolkit is required for GPU training.",
                     font=("Segoe UI", 8), bg=SURF2, fg=WARN).pack(side="left")
            dl_btn = tk.Button(row3, text="Download CUDA Toolkit →",
                               font=("Segoe UI", 8), bg=SURF2, fg=ACCENT,
                               relief="flat", bd=0, cursor="hand2",
                               activebackground=SURF2, activeforeground=ACCENT2,
                               command=lambda: webbrowser.open(
                                   "https://developer.nvidia.com/cuda-downloads"))
            dl_btn.pack(side="left", padx=(6, 0))

        # ── PyTorch variant selector ───────────────────────────────────────────
        tk.Label(c, text="PyTorch Backend", bg=BG, fg=TXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(c,
                 text="The installer selects the best option automatically based on your hardware.",
                 bg=BG, fg=TXT2, font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 6))

        torch_opts = [
            ("auto",  "Auto-detect",
             "Skip PyTorch install if already present in venv"),
            ("cpu",   "CPU only",
             "No GPU required — slower training, smaller download (~250 MB)"),
            ("cu118", "CUDA 11.8",
             "Requires NVIDIA GPU + CUDA Toolkit 11.x installed on system"),
            ("cu124", "CUDA 12.x  (12.0 – 12.4)",
             "Requires NVIDIA GPU + CUDA Toolkit 12.0–12.4"),
            ("cu128", "CUDA 12.8  (RTX 40/50 series)",
             "Required for Blackwell/Ada GPUs (RTX 50xx, RTX 40xx) — CUDA 12.5+"),
        ]

        # Tag which option is auto-selected so we can show a badge
        auto_val = self.torch_choice.get()

        for val, label, desc in torch_opts:
            is_auto = (val == auto_val and auto_val != "auto")
            bg_row  = "#1a1f2e" if is_auto else SURF2

            row = tk.Frame(c, bg=bg_row, cursor="hand2",
                           highlightbackground=ACCENT if is_auto else BORDER,
                           highlightthickness=1)
            row.pack(fill="x", pady=2, ipady=5, ipadx=4)
            row.bind("<Button-1>", lambda e, v=val: self.torch_choice.set(v))

            rb = tk.Radiobutton(row, variable=self.torch_choice, value=val,
                                bg=bg_row, activebackground=bg_row,
                                selectcolor=bg_row, fg=ACCENT, relief="flat",
                                bd=0, cursor="hand2",
                                command=lambda v=val: self.torch_choice.set(v))
            rb.pack(side="left", padx=(10, 4))

            inner = tk.Frame(row, bg=bg_row)
            inner.pack(side="left", fill="x", expand=True)

            lbl_row = tk.Frame(inner, bg=bg_row)
            lbl_row.pack(anchor="w")
            tk.Label(lbl_row, text=label, font=("Segoe UI", 9, "bold"),
                     bg=bg_row, fg=TXT, anchor="w").pack(side="left")
            if is_auto:
                tk.Label(lbl_row, text=" ✦ recommended for your system",
                         font=("Segoe UI", 8), bg=bg_row,
                         fg=ACCENT).pack(side="left", padx=(6, 0))

            tk.Label(inner, text=desc, font=("Segoe UI", 8),
                     bg=bg_row, fg=TXT2, anchor="w").pack(anchor="w")

        # ── Note about CUDA Toolkit vs PyTorch CUDA ───────────────────────────
        note = tk.Frame(c, bg="#151520")
        note.pack(fill="x", pady=(8, 0), ipady=6, ipadx=8)
        tk.Label(note,
                 text="ℹ  PyTorch CUDA ≠ CUDA Toolkit.  PyTorch CUDA is installed here automatically.\n"
                      "   The CUDA Toolkit (driver) must already be on your system to use GPU training.",
                 bg="#151520", fg=TXT2, font=("Segoe UI", 8),
                 justify="left", anchor="w").pack(anchor="w", padx=8)

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=10)

        # ── Options ───────────────────────────────────────────────────────────
        tk.Label(c, text="Options", bg=BG, fg=TXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        cb = tk.Checkbutton(c, text="Create desktop shortcut",
                            variable=self.shortcut_var,
                            bg=BG, fg=TXT, activebackground=BG, activeforeground=TXT,
                            selectcolor=SURF2, font=("Segoe UI", 9), bd=0, relief="flat",
                            cursor="hand2")
        cb.pack(anchor="w")

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3 – Install
    # ──────────────────────────────────────────────────────────────────────────
    def _page_install(self):
        c = self.content
        tk.Label(c, text="Installing…", bg=BG, fg=TXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(c, text="Please wait while dependencies are installed.",
                 bg=BG, fg=TXT2, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 14))

        # Step checklist
        self.install_steps_frame = tk.Frame(c, bg=BG)
        self.install_steps_frame.pack(fill="x")

        self._step_names = [
            "create_venv",
            "install_torch",
            "install_backend",
            "install_frontend",
            "build_frontend",
            "write_launcher",
            "shortcut",
        ]
        self._step_labels_map = {
            "create_venv":      "Creating virtual environment",
            "install_torch":    "Installing PyTorch",
            "install_backend":  "Installing backend libraries",
            "install_frontend": "Installing frontend packages",
            "build_frontend":   "Building frontend (production)",
            "write_launcher":   "Writing launcher files",
            "shortcut":         "Creating desktop shortcut",
        }
        self._step_widgets = {}
        for key in self._step_names:
            row = tk.Frame(self.install_steps_frame, bg=BG)
            row.pack(fill="x", pady=1)
            dot = tk.Label(row, text="○", font=("Segoe UI", 9),
                           fg=BORDER, bg=BG, width=3)
            dot.pack(side="left")
            lbl = tk.Label(row, text=self._step_labels_map[key],
                           font=("Segoe UI", 9), fg=TXT2, bg=BG, anchor="w")
            lbl.pack(side="left")
            self._step_widgets[key] = (dot, lbl)

        # Overall progress bar
        tk.Label(c, text="", bg=BG).pack()
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Install.Horizontal.TProgressbar",
                        troughcolor=SURF2, background=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT,
                        bordercolor=BORDER, thickness=8)
        self.progress = ttk.Progressbar(c, style="Install.Horizontal.TProgressbar",
                                        orient="horizontal", length=460,
                                        mode="determinate", maximum=len(self._step_names))
        self.progress.pack(fill="x", pady=(4, 8))

        # Log text area
        log_frame = tk.Frame(c, bg=BORDER, bd=1, relief="flat")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, bg="#0a0a0c", fg=TXT2,
                                font=("Courier New", 8), wrap="word",
                                relief="flat", bd=0, state="disabled",
                                insertbackground=ACCENT)
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview,
                                 bg=SURF2, troughcolor=SURF2, relief="flat", bd=0)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)

    def _log(self, msg, color=None):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _do)

    def _mark_step(self, key, state):
        """state: 'running' | 'done' | 'skip' | 'error'"""
        dot, lbl = self._step_widgets[key]
        if state == "running":
            dot.config(text="⏳", fg=WARN)
            lbl.config(fg=TXT)
        elif state == "done":
            dot.config(text="✓", fg=SUCCESS)
            lbl.config(fg=TXT2)
        elif state == "skip":
            dot.config(text="⤼", fg=TXT2)
            lbl.config(fg=TXT2)
        elif state == "error":
            dot.config(text="✗", fg=ERROR)
            lbl.config(fg=ERROR)

    def _advance_progress(self):
        def _do():
            self.progress.step(1)
        self.after(0, _do)

    # ── Install thread ────────────────────────────────────────────────────────
    def _start_install(self):
        threading.Thread(target=self._install_thread, daemon=True).start()

    def _install_thread(self):
        torch_mode = self.torch_choice.get()
        want_shortcut = self.shortcut_var.get()
        total = len(self._step_names)
        done = 0

        def step(key, fn):
            nonlocal done
            self.after(0, lambda: self._mark_step(key, "running"))
            ok, msg = fn()
            done += 1
            state = "done" if ok else "error"
            self.after(0, lambda s=state, k=key: self._mark_step(k, s))
            self._advance_progress()
            return ok

        # 1. Create venv
        def do_venv():
            if os.path.isfile(VENV_PY):
                # Verify it's functional — a copied venv has broken paths
                rc_test, _ = run_cmd(f'"{VENV_PY}" -c "import sys; print(sys.version)"')
                if rc_test == 0:
                    self._log("  ✓ Existing venv found, reusing.")
                    return True, ""
                self._log("  ⚠ Existing venv is broken (wrong machine?), recreating…")
                shutil.rmtree(VENV_DIR, ignore_errors=True)
            self._log("  Creating virtual environment…")
            rc, out = run_cmd(f'"{sys.executable}" -m venv "{VENV_DIR}"')
            self._log(out[:800] if out else "  done")
            return rc == 0, out

        if not step("create_venv", do_venv):
            self._log("❌ Failed to create venv. Aborting.", ERROR)
            return

        # 2. Install PyTorch
        def do_torch():
            # Check if already present and working in the venv
            rc_chk, out_chk = run_cmd(
                f"\"{VENV_PY}\" -c \"import torch; print(torch.__version__)\""
            )
            if rc_chk == 0 and torch_mode == "auto":
                # Also verify CUDA actually works if GPU present
                rc_cuda, _ = run_cmd(
                    f"\"{VENV_PY}\" -c \"import torch; torch.zeros(1).cuda() if torch.cuda.is_available() else None; print('ok')\""
                )
                if rc_cuda == 0:
                    self._log(f"  ✓ torch {out_chk.strip()} already in venv and working, skipping.")
                    self.after(0, lambda: self._mark_step("install_torch", "skip"))
                    return True, ""
                self._log(f"  ⚠ torch {out_chk.strip()} found but CUDA test failed — reinstalling with correct variant.")
                # Fall through to reinstall with selected mode

            urls = {
                "cpu":   "https://download.pytorch.org/whl/cpu",
                "cu118": "https://download.pytorch.org/whl/cu118",
                "cu124": "https://download.pytorch.org/whl/cu124",
                "cu128": "https://download.pytorch.org/whl/cu128",
            }
            # Resolve "auto" to the hardware-detected variant (set by _auto_select_torch)
            effective_mode = self.torch_choice.get() if torch_mode == "auto" else torch_mode
            if effective_mode == "auto":
                effective_mode = "cpu"   # fallback if detection never ran
            url = urls.get(effective_mode, urls["cpu"])

            sizes = {"cpu": "~250 MB", "cu118": "~2.3 GB", "cu124": "~2.4 GB", "cu128": "~2.8 GB"}
            size_hint = sizes.get(effective_mode, "~2.4 GB")

            self._log(f"  Downloading PyTorch ({effective_mode})  {size_hint}")
            self._log(f"  This may take several minutes — please wait…")
            self._log(f"  Index: {url}")
            self._log("")

            # ── Stream pip output line by line ────────────────────────────────
            # No -q so we see Downloading / Installing lines as they happen.
            cmd = (f"\"{VENV_PIP}\" install torch torchvision "
                   f"--index-url {url} --no-warn-script-location")

            last_pkg = [None]   # track last package name for dedup

            def on_line(line):
                # Show download progress and key messages; skip noisy lines
                low = line.lower()
                if any(k in low for k in (
                    "downloading", "installing", "successfully", "error",
                    "warning", "looking in", "requirement already"
                )):
                    # Shorten very long filenames to keep log readable
                    disp = line.strip()
                    if len(disp) > 90:
                        disp = disp[:87] + "…"
                    if disp != last_pkg[0]:
                        self._log(f"  {disp}")
                        last_pkg[0] = disp

            rc, _ = run_cmd_stream(cmd, line_cb=on_line)
            self._log("")
            if rc == 0:
                self._log("  ✓ PyTorch installed successfully.")
            return rc == 0, ""

        if not step("install_torch", do_torch):
            self._log("❌ PyTorch install failed. Check internet or try a different backend.", ERROR)
            return

        # 3. Install backend
        def do_backend():
            req = os.path.join(BACKEND, "requirements.txt")
            self._log("  Installing backend libraries…")

            def on_line(line):
                low = line.lower()
                if any(k in low for k in (
                    "downloading", "installing", "successfully", "error",
                    "requirement already", "collected"
                )):
                    disp = line.strip()
                    if len(disp) > 90:
                        disp = disp[:87] + "…"
                    self._log(f"  {disp}")

            cmd = f"\"{VENV_PIP}\" install -r \"{req}\" --no-warn-script-location"
            rc, _ = run_cmd_stream(cmd, cwd=BACKEND, line_cb=on_line)
            if rc == 0:
                self._log("  ✓ Backend libraries installed.")
            return rc == 0, ""

        if not step("install_backend", do_backend):
            self._log("❌ Backend dependency install failed.", ERROR)
            return

        # 4. npm install
        dist_ready = os.path.isfile(os.path.join(DIST, "index.html"))

        def do_npm():
            if dist_ready:
                self._log("  ✓ Pre-built frontend detected, skipping npm install.")
                self.after(0, lambda: self._mark_step("install_frontend", "skip"))
                return True, ""
            if os.path.isdir(os.path.join(FRONTEND, "node_modules")):
                self._log("  ✓ node_modules exists, skipping npm install.")
                self.after(0, lambda: self._mark_step("install_frontend", "skip"))
                return True, ""
            self._log("  Running npm install — downloading packages…")

            def on_npm(line):
                if line.strip():
                    self._log(f"  {line[:90]}")

            rc, _ = run_cmd_stream("npm install", cwd=FRONTEND, line_cb=on_npm)
            if rc == 0:
                self._log("  ✓ npm packages installed.")
            return rc == 0, ""

        if not step("install_frontend", do_npm):
            self._log("❌ npm install failed.", ERROR)
            return

        # 5. Build frontend
        def do_build():
            if dist_ready:
                self._log("  ✓ Pre-built frontend detected, skipping build step.")
                self.after(0, lambda: self._mark_step("build_frontend", "skip"))
                return True, ""
            self._log("  Compiling frontend (TypeScript + Vite)…")

            def on_build(line):
                if line.strip():
                    self._log(f"  {line[:90]}")

            rc, _ = run_cmd_stream("npm run build", cwd=FRONTEND, line_cb=on_build)
            ok = rc == 0 and os.path.isdir(DIST)
            if ok:
                self._log("  ✓ Frontend built successfully.")
            return ok, ""

        if not step("build_frontend", do_build):
            self._log("❌ Frontend build failed.", ERROR)
            return

        # 6. Write launcher files
        def do_launcher():
            self._write_launcher_files()
            return True, ""

        step("write_launcher", do_launcher)

        # 7. Desktop shortcut
        def do_shortcut():
            if not want_shortcut:
                self.after(0, lambda: self._mark_step("shortcut", "skip"))
                return True, ""
            launcher_bat = os.path.join(ROOT_DIR, "NoCode CV.bat")
            ok = create_desktop_shortcut(launcher_bat)
            return ok, ""

        step("shortcut", do_shortcut)

        # ── All done ──────────────────────────────────────────────────────────
        self._install_done = True
        self._log("\n✅ Installation complete!")
        self.after(0, self._on_install_complete)

    def _on_install_complete(self):
        self.btn_next.config(text="Finish  →", state="normal",
                             command=lambda: self._show_step(4))

    # ── Write launcher files ──────────────────────────────────────────────────
    def _write_launcher_files(self):
        self._log("  Writing NoCode CV.bat…")

        # Use %~dp0 so the bat works from any location (no hardcoded paths)
        bat = (
            "@echo off\r\n"
            "title NoCode CV Trainer\r\n"
            "cd /d \"%~dp0\"\r\n"
            "if exist \"%~dp0venv\\Scripts\\pythonw.exe\" (\r\n"
            "    start \"\" \"%~dp0venv\\Scripts\\pythonw.exe\" \"%~dp0launcher.py\"\r\n"
            ") else (\r\n"
            "    echo ERROR: App not installed. Please run \"Install NoCode CV.bat\" first.\r\n"
            "    pause\r\n"
            ")\r\n"
        )
        bat_path = os.path.join(ROOT_DIR, "NoCode CV.bat")
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat)
        self._log(f"  Written: {bat_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4 – Done
    # ──────────────────────────────────────────────────────────────────────────
    def _page_done(self):
        c = self.content
        if self._install_done:
            tk.Label(c, text="✅", font=("Segoe UI Emoji", 40), bg=BG, fg=SUCCESS).pack(pady=(10, 4))
            tk.Label(c, text="Installation Complete!", bg=BG, fg=TXT,
                     font=("Segoe UI", 16, "bold")).pack()
            tk.Label(c, text="NoCode CV Trainer is ready to use.",
                     bg=BG, fg=TXT2, font=("Segoe UI", 10)).pack(pady=(4, 24))
        else:
            tk.Label(c, text="⚠️", font=("Segoe UI Emoji", 40), bg=BG, fg=WARN).pack(pady=(10, 4))
            tk.Label(c, text="Installation Incomplete", bg=BG, fg=TXT,
                     font=("Segoe UI", 16, "bold")).pack()
            tk.Label(c, text="One or more steps failed. Check the log above.",
                     bg=BG, fg=TXT2, font=("Segoe UI", 10)).pack(pady=(4, 24))

        launch_btn = tk.Button(c,
            text="🚀  Launch NoCode CV",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg="#fff", relief="flat", bd=0,
            padx=28, pady=12, cursor="hand2",
            activebackground=ACCENT2, activeforeground="#fff",
            command=self._launch_app)
        launch_btn.pack(pady=6)
        if not self._install_done:
            launch_btn.config(state="disabled", bg=SURF2, fg=TXT2)

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=16)

        info_lines = [
            "• To launch later:  double-click  'NoCode CV.bat'  or the desktop shortcut",
            "• App runs at:  http://localhost:8000",
            "• Logs are streamed live during training",
        ]
        for line in info_lines:
            tk.Label(c, text=line, bg=BG, fg=TXT2,
                     font=("Segoe UI", 8), anchor="w").pack(anchor="w")

    def _launch_app(self):
        launcher = os.path.join(ROOT_DIR, "launcher.py")
        venv_pyw = os.path.join(VENV_DIR, "Scripts", "pythonw.exe")
        if os.path.isfile(venv_pyw):
            subprocess.Popen([venv_pyw, launcher], cwd=ROOT_DIR)
        else:
            subprocess.Popen([sys.executable, launcher], cwd=ROOT_DIR)
        self.after(1500, self.destroy)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Installer()
    app.mainloop()
