"""
NoCode CV Trainer – GUI Installer
Requires: Python 3.9+ (tkinter is built-in)
Run via:  Install NoCode CV.bat  (Windows)
          bash 'Install NoCode CV.sh'  (macOS / Linux)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess, threading, os, sys, shutil, webbrowser, json, platform

# ─── Constants ────────────────────────────────────────────────────────────────
APP_NAME     = "NoCode CV Trainer"
APP_VERSION  = "1.0.0"
WIN_W, WIN_H = 820, 560
SIDE_W       = 215

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")
BACKEND  = os.path.join(ROOT_DIR, "backend")
FRONTEND = os.path.join(ROOT_DIR, "frontend")
DIST     = os.path.join(FRONTEND, "dist")

if platform.system() == "Windows":
    VENV_PY  = os.path.join(VENV_DIR, "Scripts", "python.exe")
    VENV_PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    VENV_PY  = os.path.join(VENV_DIR, "bin", "python")
    VENV_PIP = os.path.join(VENV_DIR, "bin", "pip")

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"

FONT_UI    = "Segoe UI"       if _IS_WIN else ("SF Pro Text"       if _IS_MAC else "Helvetica")
FONT_EMOJI = "Segoe UI Emoji" if _IS_WIN else ("Apple Color Emoji" if _IS_MAC else "Noto Color Emoji")
FONT_MONO  = "Courier New"

# ─── Colour palette ───────────────────────────────────────────────────────────
BG      = "#0b0b0e"
SIDE_BG = "#0e0e14"
SURFACE = "#141419"
SURF2   = "#1b1b24"
SURF3   = "#22222f"
BORDER  = "#2b2b3c"
BORDER2 = "#38385a"
ACCENT  = "#5865f2"
ACCENT2 = "#4752c4"
ACCENTL = "#6e79f5"
TXT     = "#eeeef2"
TXT2    = "#7a7a98"
TXT3    = "#3c3c58"
SUCCESS = "#3ba55d"
ERROR   = "#ed4245"
WARN    = "#faa61a"
INFO    = "#4f9ef8"
PURPLE  = "#c084fc"

STEPS = ["Welcome", "System Check", "Configure", "Install", "Done"]


# ─── Subprocess helpers ───────────────────────────────────────────────────────
def run_cmd(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=(platform.system() == "Windows"),
    )
    return proc.returncode, proc.stdout or ""


def run_cmd_stream(cmd, cwd=None, line_cb=None):
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
    try:
        rc, out = run_cmd(f"{name} --version")
        if rc == 0:
            return out.strip().split("\n")[0]
    except Exception:
        pass
    return None


def get_free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return 999


def create_desktop_shortcut(launcher_file):
    if not _IS_WIN:
        return False
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    lnk = os.path.join(desktop, f"{APP_NAME}.lnk")
    ps = (
        f'$ws = New-Object -ComObject WScript.Shell;'
        f'$s  = $ws.CreateShortcut("{lnk}");'
        f'$s.TargetPath  = "{launcher_file}";'
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


# ─── UI micro-helpers ─────────────────────────────────────────────────────────
def bind_hover(widget, bg_normal, bg_hover, fg_normal=None, fg_hover=None):
    def _enter(e):
        widget.config(bg=bg_hover)
        if fg_hover:
            widget.config(fg=fg_hover)
    def _leave(e):
        widget.config(bg=bg_normal)
        if fg_normal:
            widget.config(fg=fg_normal)
    widget.bind("<Enter>", _enter)
    widget.bind("<Leave>", _leave)


def bordered_frame(parent, bg=None, border_color=None, **pack_kw):
    """Return (outer_border_frame, inner_content_frame)."""
    bg = bg or SURF2
    border_color = border_color or BORDER
    outer = tk.Frame(parent, bg=border_color)
    inner = tk.Frame(outer, bg=bg)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


# ─── Main Installer Window ────────────────────────────────────────────────────
class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  ·  Setup")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center()
        self._set_icon()

        # State
        self.step          = 0
        self.checks        = {}
        self.torch_choice  = tk.StringVar(value="auto")
        self.shortcut_var  = tk.BooleanVar(value=True)
        self.install_mode  = tk.StringVar(value="venv")
        self._pkg_scan     = {}
        self._scan_panel   = None
        self._install_done = False
        self.detected_cuda = None
        self.detected_gpu  = None

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
            ico = tk.PhotoImage(width=1, height=1)
            self.iconphoto(True, ico)
        except Exception:
            pass

    # ── UI skeleton ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar = tk.Frame(self, bg=SIDE_BG, width=SIDE_W)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Thin vertical divider between sidebar and content
        tk.Frame(self, bg=BORDER, width=1).pack(side="left", fill="y")

        # Accent top bar
        tk.Frame(self.sidebar, bg=ACCENT, height=3).pack(fill="x")

        # Brand block
        brand = tk.Frame(self.sidebar, bg=SIDE_BG)
        brand.pack(fill="x", padx=22, pady=(22, 0))
        tk.Label(brand, text="🧠", font=(FONT_EMOJI, 26),
                 bg=SIDE_BG, fg=ACCENT).pack(anchor="w")
        tk.Label(brand, text=APP_NAME,
                 font=(FONT_UI, 10, "bold"), bg=SIDE_BG, fg=TXT,
                 wraplength=170, justify="left").pack(anchor="w", pady=(6, 2))
        tk.Label(brand, text=f"v{APP_VERSION}  ·  Setup Wizard",
                 font=(FONT_UI, 8), bg=SIDE_BG, fg=TXT3).pack(anchor="w")

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=0, pady=(20, 18))

        # Step indicators
        self._step_circles  = []   # Canvas widgets
        self._step_lbls     = []   # Label widgets
        self._connector_frs = []   # connector line frames

        for i, name in enumerate(STEPS):
            if i > 0:
                # Connector line between steps
                line_wrap = tk.Frame(self.sidebar, bg=SIDE_BG)
                line_wrap.pack(anchor="w", padx=(37, 0))
                line = tk.Frame(line_wrap, bg=TXT3, width=1, height=16)
                line.pack()
                self._connector_frs.append(line)

            row = tk.Frame(self.sidebar, bg=SIDE_BG)
            row.pack(fill="x", padx=20, pady=0)

            circ = tk.Canvas(row, width=28, height=28,
                             bg=SIDE_BG, highlightthickness=0)
            circ.pack(side="left", padx=(0, 12))

            lbl = tk.Label(row, text=name, font=(FONT_UI, 9),
                           bg=SIDE_BG, fg=TXT3, anchor="w")
            lbl.pack(side="left", fill="x")

            self._step_circles.append(circ)
            self._step_lbls.append(lbl)

        # Sidebar footer
        footer = tk.Frame(self.sidebar, bg=SIDE_BG)
        footer.pack(side="bottom", fill="x", padx=22, pady=18)
        tk.Label(footer,
                 text="Everything runs locally.\nNo data leaves your machine.",
                 font=(FONT_UI, 7), bg=SIDE_BG, fg=TXT3,
                 justify="left").pack(anchor="w")

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Bottom button bar — pack BEFORE content so it anchors to bottom
        tk.Frame(right, bg=BORDER, height=1).pack(side="bottom", fill="x")
        bar = tk.Frame(right, bg=SURFACE)
        bar.pack(side="bottom", fill="x")

        self.step_counter = tk.Label(bar, text="",
                                     font=(FONT_UI, 8), bg=SURFACE, fg=TXT3)
        self.step_counter.pack(side="left", padx=22, pady=12)

        self.btn_back = tk.Button(
            bar, text="← Back",
            font=(FONT_UI, 9), bg=SURFACE, fg=TXT2,
            relief="flat", bd=0, padx=20, pady=9,
            cursor="hand2", activebackground=SURF2, activeforeground=TXT,
            command=self._go_back)
        self.btn_back.pack(side="right", padx=(4, 18), pady=10)
        bind_hover(self.btn_back, SURFACE, SURF2)

        self.btn_next = tk.Button(
            bar, text="Continue  →",
            font=(FONT_UI, 9, "bold"), bg=ACCENT, fg="#fff",
            relief="flat", bd=0, padx=26, pady=9,
            cursor="hand2", activebackground=ACCENT2, activeforeground="#fff",
            command=self._go_next)
        self.btn_next.pack(side="right", padx=(0, 4), pady=10)
        bind_hover(self.btn_next, ACCENT, ACCENT2, "#fff", "#fff")

        # Scrollable content area
        wrap = tk.Frame(right, bg=BG)
        wrap.pack(side="top", fill="both", expand=True)

        self._cv = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        self._sb = tk.Scrollbar(wrap, orient="vertical", command=self._cv.yview,
                                bg=SURF2, troughcolor=BG, relief="flat", width=7)
        self._cv.configure(yscrollcommand=self._sb.set)
        self._cv.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._cv, bg=BG)
        self._cwin  = self._cv.create_window((0, 0), window=self._inner, anchor="nw")

        self.content = tk.Frame(self._inner, bg=BG)
        self.content.pack(fill="both", expand=True, padx=32, pady=28)

        def _on_inner_resize(_e=None):
            self._cv.configure(scrollregion=self._cv.bbox("all"))
            self._refresh_scroll()

        def _on_canvas_resize(e):
            self._cv.itemconfig(self._cwin, width=e.width)
            _on_inner_resize()

        self._inner.bind("<Configure>", _on_inner_resize)
        self._cv.bind("<Configure>", _on_canvas_resize)
        self._cv.bind_all("<MouseWheel>",
            lambda e: self._cv.yview_scroll(int(-1 * e.delta / 120), "units"))

    def _draw_step_circle(self, canvas, state, num):
        canvas.delete("all")
        bg = canvas.cget("bg")
        if state == "done":
            canvas.create_oval(2, 2, 26, 26, fill=SUCCESS, outline="")
            canvas.create_text(14, 14, text="✓", fill="white",
                               font=(FONT_UI, 9, "bold"))
        elif state == "active":
            canvas.create_oval(2, 2, 26, 26, fill=ACCENT, outline="")
            canvas.create_text(14, 14, text=str(num), fill="white",
                               font=(FONT_UI, 9, "bold"))
        else:
            canvas.create_oval(2, 2, 26, 26, fill="", outline=TXT3, width=1.5)
            canvas.create_text(14, 14, text=str(num), fill=TXT3,
                               font=(FONT_UI, 9))

    def _refresh_scroll(self):
        self._cv.update_idletasks()
        if self._inner.winfo_reqheight() > self._cv.winfo_height() + 4:
            self._sb.pack(side="right", fill="y")
        else:
            self._sb.pack_forget()

    # ── Step navigation ───────────────────────────────────────────────────────
    def _update_sidebar(self):
        for i, (circ, lbl) in enumerate(zip(self._step_circles, self._step_lbls)):
            if i < self.step:
                self._draw_step_circle(circ, "done", i + 1)
                lbl.config(fg=TXT2, font=(FONT_UI, 9))
                # Connector line turns green once step is done
                if i < len(self._connector_frs):
                    self._connector_frs[i].config(bg=SUCCESS)
            elif i == self.step:
                self._draw_step_circle(circ, "active", i + 1)
                lbl.config(fg=TXT, font=(FONT_UI, 9, "bold"))
            else:
                self._draw_step_circle(circ, "inactive", i + 1)
                lbl.config(fg=TXT3, font=(FONT_UI, 9))
                if i > 0 and i - 1 < len(self._connector_frs):
                    if i > self.step:
                        self._connector_frs[i - 1].config(bg=TXT3)

    def _show_step(self, idx):
        self.step = idx
        self._update_sidebar()
        for w in self.content.winfo_children():
            w.destroy()
        self._cv.yview_moveto(0)

        self.step_counter.config(text=f"Step {idx + 1} of {len(STEPS)}")

        [self._page_welcome, self._page_syscheck, self._page_configure,
         self._page_install, self._page_done][idx]()

        self.btn_back.config(
            state="normal" if idx > 0 else "disabled",
            fg=TXT2 if idx > 0 else TXT3)

        if idx == len(STEPS) - 1:
            self.btn_next.config(text="Close", state="normal",
                                 command=self.destroy, bg=SURF2, fg=TXT2)
            bind_hover(self.btn_next, SURF2, SURF3, TXT2, TXT)
        elif idx == 3:
            self.btn_next.config(text="Installing…", state="disabled",
                                 bg=SURF2, fg=TXT2)
        else:
            self.btn_next.config(text="Continue  →", state="normal",
                                 command=self._go_next, bg=ACCENT, fg="#fff")
            bind_hover(self.btn_next, ACCENT, ACCENT2, "#fff", "#fff")

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

        # Hero card
        hero_outer, hero = bordered_frame(c, bg=SURF2)
        hero_outer.pack(fill="x", pady=(0, 18))
        tk.Frame(hero, bg=ACCENT, height=3).pack(fill="x")
        hero_body = tk.Frame(hero, bg=SURF2)
        hero_body.pack(fill="x", padx=20, pady=16)
        tk.Label(hero_body, text=f"Welcome to {APP_NAME}",
                 font=(FONT_UI, 16, "bold"), bg=SURF2, fg=TXT,
                 anchor="w").pack(anchor="w")
        tk.Label(hero_body,
                 text="Annotate images, train AI models, and run predictions — no code required.",
                 font=(FONT_UI, 9), bg=SURF2, fg=TXT2,
                 anchor="w").pack(anchor="w", pady=(4, 0))

        # Feature cards
        features = [
            (ACCENT,  "🖼",  "Annotation Studio",    "Bounding box, polygon & point tools with undo / redo"),
            (SUCCESS, "⚡",  "YOLOv8 Training",       "Real-time training logs, augmentation control"),
            (WARN,    "🔬",  "Evaluation & Inference", "mAP metrics, per-class breakdown, live inference"),
            (INFO,    "🏷",  "Classification",         "ResNet / MobileNet / EfficientNet transfer learning"),
            (PURPLE,  "🧱",  "Custom CNN Builder",     "Design your own neural network architecture visually"),
            (SUCCESS, "📦",  "Dataset Export",         "YOLO & COCO zip export in one click"),
        ]

        for accent_col, icon, title, desc in features:
            f_outer, f_inner = bordered_frame(c, bg=SURF2)
            f_outer.pack(fill="x", pady=2)
            tk.Frame(f_inner, bg=accent_col, width=3).pack(side="left", fill="y")
            body = tk.Frame(f_inner, bg=SURF2)
            body.pack(side="left", fill="both", expand=True, padx=14, pady=9)
            tk.Label(body, text=f"{icon}  {title}",
                     font=(FONT_UI, 9, "bold"), bg=SURF2, fg=TXT,
                     anchor="w").pack(anchor="w")
            tk.Label(body, text=desc, font=(FONT_UI, 8),
                     bg=SURF2, fg=TXT2, anchor="w").pack(anchor="w", pady=(2, 0))

        # Install path
        tk.Frame(c, bg=BG, height=6).pack()
        tk.Label(c, text="Install location",
                 font=(FONT_UI, 8), bg=BG, fg=TXT2).pack(anchor="w")
        path_outer, path_inner = bordered_frame(c, bg=SURF2)
        path_outer.pack(fill="x", pady=(4, 0))
        tk.Label(path_inner, text=ROOT_DIR,
                 font=(FONT_MONO, 8), bg=SURF2, fg=TXT2,
                 anchor="w").pack(anchor="w", padx=14, pady=8)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1 – System Check
    # ──────────────────────────────────────────────────────────────────────────
    def _page_syscheck(self):
        c = self.content
        tk.Label(c, text="System Check", font=(FONT_UI, 15, "bold"),
                 bg=BG, fg=TXT).pack(anchor="w")
        tk.Label(c, text="Verifying your environment before installation.",
                 font=(FONT_UI, 9), bg=BG, fg=TXT2).pack(anchor="w", pady=(4, 18))

        dist_prebuilt = os.path.isfile(os.path.join(DIST, "index.html"))
        node_note = "Pre-built frontend included" if dist_prebuilt else "Required  (nodejs.org)"
        npm_note  = "Pre-built frontend included" if dist_prebuilt else "Comes with Node.js"

        self.check_rows = {}
        items = [
            ("python", "Python 3.9+",            "Required"),
            ("node",   "Node.js 16+",             node_note),
            ("npm",    "npm",                     npm_note),
            ("gpu",    "NVIDIA GPU",              "Optional  ·  CPU fallback available"),
            ("cuda",   "CUDA Toolkit",            "Optional  ·  required for GPU training"),
            ("disk",   "Free disk space  ≥ 2 GB", "Required"),
        ]

        for key, label, note in items:
            outer, row = bordered_frame(c, bg=SURF2)
            outer.pack(fill="x", pady=2)

            dot = tk.Label(row, text="·", font=(FONT_EMOJI, 16),
                           bg=SURF2, fg=TXT3, width=3)
            dot.pack(side="left", padx=(12, 6), pady=10)

            tk.Label(row, text=label, font=(FONT_UI, 9, "bold"),
                     bg=SURF2, fg=TXT, width=24, anchor="w").pack(side="left", pady=10)

            val = tk.Label(row, text="checking…", font=(FONT_MONO, 8),
                           bg=SURF2, fg=TXT2, anchor="w")
            val.pack(side="left", padx=8, pady=10)

            tk.Label(row, text=note, font=(FONT_UI, 7),
                     bg=SURF2, fg=TXT3, anchor="e").pack(side="right", padx=14, pady=10)

            self.check_rows[key] = (dot, val, outer, row)

        self.hint_frame = tk.Frame(c, bg=BG)
        self.hint_frame.pack(fill="x", pady=(10, 0))

        threading.Thread(target=self._run_checks, daemon=True).start()

    def _run_checks(self):
        import re
        results = {}

        # Python
        v = sys.version.split()[0]
        parts = v.split(".")
        ok = int(parts[0]) >= 3 and int(parts[1]) >= 9
        results["python"] = (ok, f"Python {v}")

        # Node.js
        nv = check_tool("node")
        if nv:
            num = nv.replace("v", "").split(".")[0]
            results["node"] = (int(num) >= 16, nv[:40])
        else:
            results["node"] = (False, "Not found")

        # npm
        nv2 = check_tool("npm")
        results["npm"] = (bool(nv2), nv2[:40] if nv2 else "Not found")

        # NVIDIA GPU
        gpu_name   = None
        cuda_ver   = None
        driver_ver = None
        try:
            rc, smi = run_cmd("nvidia-smi")
            if rc == 0 and smi:
                m = re.search(r'\|\s+(NVIDIA\s+[^\|]+?)\s+\d{5,}', smi)
                if m:
                    gpu_name = m.group(1).strip()
                m2 = re.search(r'Driver Version:\s*([\d.]+)', smi)
                if m2:
                    driver_ver = m2.group(1)
                m3 = re.search(r'CUDA Version:\s*([\d.]+)', smi)
                if m3:
                    cuda_ver = m3.group(1)
        except Exception:
            pass

        if not gpu_name:
            try:
                rc2, name_out = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader")
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

        if cuda_ver:
            results["cuda"] = (True, f"CUDA {cuda_ver}  (driver-reported)")
        elif gpu_name:
            nvcc = check_tool("nvcc")
            if nvcc:
                m4 = re.search(r'release\s*([\d.]+)', nvcc, re.IGNORECASE)
                cuda_ver = m4.group(1) if m4 else nvcc.split()[-1]
                results["cuda"] = (True, f"CUDA {cuda_ver}  (nvcc)")
            else:
                results["cuda"] = (False, "CUDA Toolkit not found")
        else:
            results["cuda"] = (None, "No GPU — CUDA not needed")

        self.detected_cuda = cuda_ver
        self.detected_gpu  = gpu_name

        gb = get_free_gb(ROOT_DIR)
        results["disk"] = (gb >= 2.0, f"{gb:.1f} GB free")

        self.checks = results
        self.after(0, self._apply_check_results)

    def _apply_check_results(self):
        for key, (dot, val, outer, row) in self.check_rows.items():
            if key not in self.checks:
                continue
            ok, detail = self.checks[key]

            if ok is True:
                sym, col, row_bg = "✓", SUCCESS, SURF2
            elif ok is False:
                sym, col, row_bg = "✗", ERROR, "#1e1118"
            else:
                sym, col, row_bg = "ℹ", TXT2, SURF2

            row.config(bg=row_bg)
            for child in row.winfo_children():
                try:
                    child.config(bg=row_bg)
                except Exception:
                    pass

            dot.config(text=sym, fg=col, bg=row_bg,
                       font=(FONT_EMOJI, 13))
            val.config(text=detail,
                       fg=TXT if ok is not False else ERROR,
                       bg=row_bg)

        for w in self.hint_frame.winfo_children():
            w.destroy()

        def _hint(text, color, url=None):
            lbl = tk.Label(self.hint_frame, text=text, bg=BG, fg=color,
                           font=(FONT_UI, 8),
                           cursor="hand2" if url else "arrow",
                           anchor="w")
            lbl.pack(anchor="w", pady=1)
            if url:
                lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        if not self.checks.get("node", (False,))[0]:
            if os.path.isfile(os.path.join(DIST, "index.html")):
                _hint("ℹ  Node.js not found — pre-built frontend detected, skipping build step.", TXT2)
            else:
                _hint("⚠  Node.js not found — click to download  nodejs.org",
                      WARN, "https://nodejs.org/en/download")

        gpu_ok  = self.checks.get("gpu",  (None,))[0]
        cuda_ok = self.checks.get("cuda", (None,))[0]

        if gpu_ok and not cuda_ok:
            _hint("⚠  CUDA Toolkit not installed — GPU training will not work without it.", WARN)
            _hint("   → Download CUDA Toolkit  (recommended: CUDA 12.4 or 12.8)",
                  ACCENT, "https://developer.nvidia.com/cuda-downloads")
            _hint("   After installing CUDA, re-run this installer and choose a CUDA PyTorch variant.", TXT2)
        elif not gpu_ok:
            _hint("ℹ  No NVIDIA GPU detected — the app will run in CPU mode (slower training).", TXT2)

        self._auto_select_torch()

        if self._checks_pass():
            self.btn_next.config(state="normal")
        else:
            self.btn_next.config(state="disabled")

    def _auto_select_torch(self):
        cv = self.detected_cuda
        if cv is None:
            self.torch_choice.set("cpu" if not self.detected_gpu else "auto")
            return
        try:
            major = int(cv.split(".")[0])
            minor = int(cv.split(".")[1]) if "." in cv else 0
        except ValueError:
            return
        if major > 12 or (major == 12 and minor >= 5):
            self.torch_choice.set("cu128")
        elif major == 12:
            self.torch_choice.set("cu124")
        elif major == 11 and minor >= 8:
            self.torch_choice.set("cu118")
        else:
            self.torch_choice.set("cpu")

    def _checks_pass(self):
        must = ["python", "disk"]
        if not os.path.isfile(os.path.join(DIST, "index.html")):
            must += ["node", "npm"]
        return all(self.checks.get(k, (False,))[0] for k in must)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2 – Configure
    # ──────────────────────────────────────────────────────────────────────────
    def _page_configure(self):
        c = self.content
        tk.Label(c, text="Configure Installation", font=(FONT_UI, 15, "bold"),
                 bg=BG, fg=TXT).pack(anchor="w")
        tk.Label(c, text="Select your PyTorch backend and installation options.",
                 font=(FONT_UI, 9), bg=BG, fg=TXT2).pack(anchor="w", pady=(4, 18))

        # Hardware summary
        gpu_ok  = self.checks.get("gpu",  (None,))[0]
        cuda_ok = self.checks.get("cuda", (None,))[0]
        gpu_text  = self.detected_gpu or "No NVIDIA GPU detected"
        cuda_text = (f"CUDA {self.detected_cuda}" if self.detected_cuda
                     else ("Not installed" if gpu_ok else "N/A"))
        gpu_col  = SUCCESS if gpu_ok else (TXT2 if gpu_ok is None else ERROR)
        cuda_col = SUCCESS if cuda_ok else (WARN if gpu_ok else TXT2)

        hw_outer, hw = bordered_frame(c, bg=SURF2)
        hw_outer.pack(fill="x", pady=(0, 16))
        tk.Frame(hw, bg=INFO, width=3).pack(side="left", fill="y")
        hw_body = tk.Frame(hw, bg=SURF2)
        hw_body.pack(side="left", padx=14, pady=12)

        for icon, label, val, col in [
            ("GPU ", gpu_text[:58], None, gpu_col),
            ("CUDA", cuda_text,     None, cuda_col),
        ]:
            r = tk.Frame(hw_body, bg=SURF2)
            r.pack(anchor="w", pady=1)
            tk.Label(r, text=icon, font=(FONT_MONO, 8, "bold"),
                     bg=SURF2, fg=TXT2).pack(side="left")
            tk.Label(r, text=f"  {label}", font=(FONT_MONO, 8),
                     bg=SURF2, fg=col).pack(side="left")

        if gpu_ok and not cuda_ok:
            r3 = tk.Frame(hw_body, bg=SURF2)
            r3.pack(anchor="w", pady=(6, 0))
            btn = tk.Button(r3, text="Download CUDA Toolkit  →",
                            font=(FONT_UI, 8), bg=SURF2, fg=ACCENT,
                            relief="flat", bd=0, cursor="hand2",
                            activebackground=SURF2, activeforeground=ACCENTL,
                            command=lambda: webbrowser.open(
                                "https://developer.nvidia.com/cuda-downloads"))
            btn.pack(side="left")

        # PyTorch backend
        self._section_label(c, "PyTorch Backend",
                            "Auto-selected based on hardware. Change only if needed.")

        auto_val = self.torch_choice.get()
        torch_opts = [
            ("auto",  "Auto-detect",
             "Skip if PyTorch is already installed and working"),
            ("cpu",   "CPU only",
             "No GPU required  ·  smaller download (~250 MB)  ·  slower training"),
            ("cu118", "CUDA 11.8",
             "GTX 10xx / RTX 20xx  ·  requires CUDA Toolkit 11.x"),
            ("cu124", "CUDA 12.x  (12.0 – 12.4)",
             "RTX 30xx  ·  requires CUDA Toolkit 12.0 – 12.4"),
            ("cu128", "CUDA 12.8  (RTX 40 / 50 series)",
             "Ada Lovelace & Blackwell  ·  requires CUDA Toolkit 12.5+"),
        ]
        for val, label, desc in torch_opts:
            recommended = (val == auto_val and auto_val != "auto")
            self._radio_card(c, self.torch_choice, val, label, desc,
                             recommended=recommended,
                             on_click=lambda v=val: self.torch_choice.set(v))

        note_outer, note_inner = bordered_frame(c, bg=SURF3, border_color=BORDER)
        note_outer.pack(fill="x", pady=(8, 0))
        tk.Label(note_inner,
                 text="ℹ  PyTorch CUDA is downloaded automatically.\n"
                      "   The CUDA Toolkit from NVIDIA must already be installed to use GPU training.",
                 font=(FONT_UI, 8), bg=SURF3, fg=TXT2,
                 justify="left", anchor="w").pack(anchor="w", padx=14, pady=10)

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(14, 10))

        # Environment
        self._section_label(c, "Environment",
                            "Choose where packages are installed.")
        env_opts = [
            ("venv",   "Isolated virtual environment",
             "Creates ./venv  ·  nothing touches your system Python  (recommended)"),
            ("system", "Use existing Python installation",
             "Skips venv creation  ·  only installs missing packages"),
        ]
        for val, label, desc in env_opts:
            self._radio_card(c, self.install_mode, val, label, desc,
                             on_click=lambda v=val: self._set_env_mode(v))

        self._scan_panel = tk.Frame(c, bg=BG)
        self._scan_panel.pack(fill="x", pady=(4, 0))

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(14, 10))

        # Options
        self._section_label(c, "Options", "")
        opt_outer, opt_inner = bordered_frame(c, bg=SURF2)
        opt_outer.pack(fill="x")
        cb = tk.Checkbutton(opt_inner, text="Create desktop shortcut",
                            variable=self.shortcut_var,
                            bg=SURF2, fg=TXT, activebackground=SURF2,
                            activeforeground=TXT, selectcolor=SURF3,
                            font=(FONT_UI, 9), bd=0, relief="flat", cursor="hand2")
        cb.pack(anchor="w", padx=14, pady=10)

    def _section_label(self, parent, title, subtitle):
        tk.Label(parent, text=title, font=(FONT_UI, 10, "bold"),
                 bg=BG, fg=TXT, anchor="w").pack(anchor="w")
        if subtitle:
            tk.Label(parent, text=subtitle, font=(FONT_UI, 8),
                     bg=BG, fg=TXT2, anchor="w").pack(anchor="w", pady=(2, 6))
        else:
            tk.Frame(parent, bg=BG, height=6).pack()

    def _radio_card(self, parent, var, val, label, desc,
                    recommended=False, on_click=None):
        border_col = ACCENT if recommended else BORDER
        bg_col     = "#181826" if recommended else SURF2

        outer = tk.Frame(parent, bg=border_col)
        outer.pack(fill="x", pady=2)
        inner = tk.Frame(outer, bg=bg_col, cursor="hand2")
        inner.pack(fill="both", padx=1, pady=1)
        if on_click:
            inner.bind("<Button-1>", lambda e: on_click())

        rb = tk.Radiobutton(inner, variable=var, value=val,
                            bg=bg_col, activebackground=bg_col,
                            selectcolor=bg_col, fg=ACCENT,
                            relief="flat", bd=0, cursor="hand2",
                            command=on_click)
        rb.pack(side="left", padx=(12, 4), pady=10)

        txt_frame = tk.Frame(inner, bg=bg_col)
        txt_frame.pack(side="left", fill="x", expand=True, pady=10)

        lbl_row = tk.Frame(txt_frame, bg=bg_col)
        lbl_row.pack(anchor="w")
        tk.Label(lbl_row, text=label, font=(FONT_UI, 9, "bold"),
                 bg=bg_col, fg=TXT, anchor="w").pack(side="left")
        if recommended:
            tk.Label(lbl_row, text="  ✦ recommended",
                     font=(FONT_UI, 8), bg=bg_col, fg=ACCENT).pack(side="left", padx=(6, 0))

        tk.Label(txt_frame, text=desc, font=(FONT_UI, 8),
                 bg=bg_col, fg=TXT2, anchor="w").pack(anchor="w")

    # ── Environment mode helpers ───────────────────────────────────────────────
    def _set_env_mode(self, val):
        self.install_mode.set(val)
        self._pkg_scan = {}
        self._rebuild_scan_panel()

    def _rebuild_scan_panel(self):
        if self._scan_panel is None:
            return
        for w in self._scan_panel.winfo_children():
            w.destroy()
        if self.install_mode.get() != "system":
            return

        btn_row = tk.Frame(self._scan_panel, bg=BG)
        btn_row.pack(anchor="w", pady=(4, 0))
        scan_btn = tk.Button(btn_row, text="Scan installed packages",
                             font=(FONT_UI, 8), bg=SURF2, fg=ACCENT,
                             relief="flat", bd=0, padx=14, pady=6,
                             cursor="hand2", activebackground=SURF2,
                             command=self._start_pkg_scan)
        scan_btn.pack(side="left")
        bind_hover(scan_btn, SURF2, SURF3)
        tk.Label(btn_row,
                 text="  Check which required packages are already on your system.",
                 font=(FONT_UI, 8), bg=BG, fg=TXT2).pack(side="left")

        if self._pkg_scan:
            self._draw_scan_results()

    def _start_pkg_scan(self):
        for w in self._scan_panel.winfo_children():
            if isinstance(w, tk.Frame):
                for child in w.winfo_children():
                    if isinstance(child, tk.Button):
                        child.config(state="disabled", text="Scanning…")
        threading.Thread(target=self._do_pkg_scan, daemon=True).start()

    def _do_pkg_scan(self):
        import re as _re
        req_path = os.path.join(BACKEND, "requirements.txt")
        packages = []
        try:
            with open(req_path) as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    pip_name = _re.split(r"[><=!;]", line)[0].strip()
                    packages.append((pip_name, line))
        except Exception:
            pass

        for name in ("torch", "torchvision"):
            if not any(p == name for p, _ in packages):
                packages.append((name, name))

        results = {}
        for pip_name, req_spec in packages:
            rc, out = run_cmd(f'"{sys.executable}" -m pip show "{pip_name}"')
            if rc == 0:
                ver_line = next(
                    (ln for ln in out.splitlines() if ln.lower().startswith("version:")), "")
                ver = ver_line.split(":", 1)[-1].strip() if ver_line else "?"
                results[pip_name] = (True, ver, req_spec)
            else:
                results[pip_name] = (False, "not installed", req_spec)

        self._pkg_scan = results
        self.after(0, self._rebuild_scan_panel)

    def _draw_scan_results(self):
        found   = sum(1 for ok, _, _ in self._pkg_scan.values() if ok)
        total   = len(self._pkg_scan)
        missing = [n for n, (ok, _, _) in self._pkg_scan.items() if not ok]

        summary_col = SUCCESS if not missing else WARN
        summary_txt = (f"  {found}/{total} packages installed"
                       + (f"  ·  {len(missing)} missing" if missing else "  ·  all present"))
        tk.Label(self._scan_panel, text=summary_txt,
                 font=(FONT_UI, 8, "bold"), bg=BG, fg=summary_col,
                 anchor="w").pack(anchor="w", pady=(6, 4))

        grid = tk.Frame(self._scan_panel, bg=SURF2)
        grid.pack(fill="x")

        for col_i, (txt, w) in enumerate([("Package", 22), ("Required", 24), ("Installed", 20)]):
            tk.Label(grid, text=txt, font=(FONT_MONO, 7, "bold"),
                     bg=SURF2, fg=TXT2, width=w, anchor="w").grid(
                row=0, column=col_i,
                padx=(8 if col_i == 0 else 4, 4), pady=(5, 2), sticky="w")

        for row_i, (pkg, (ok, ver, req)) in enumerate(self._pkg_scan.items(), start=1):
            bg_row = "#0f1520" if row_i % 2 == 0 else SURF2
            icon = "✓" if ok else "✗"
            icol = SUCCESS if ok else ERROR
            vcol = TXT if ok else ERROR
            tk.Label(grid, text=f"{icon} {pkg}", font=(FONT_MONO, 7),
                     bg=bg_row, fg=icol, width=22, anchor="w").grid(
                row=row_i, column=0, padx=(8, 4), pady=1, sticky="w")
            tk.Label(grid, text=req[:24], font=(FONT_MONO, 7),
                     bg=bg_row, fg=TXT2, width=24, anchor="w").grid(
                row=row_i, column=1, padx=(4, 4), pady=1, sticky="w")
            tk.Label(grid, text=ver[:20], font=(FONT_MONO, 7),
                     bg=bg_row, fg=vcol, width=20, anchor="w").grid(
                row=row_i, column=2, padx=(4, 8), pady=1, sticky="w")

        if missing:
            tk.Label(self._scan_panel,
                     text="  Missing packages will be installed automatically.",
                     font=(FONT_UI, 8), bg=BG, fg=TXT2, anchor="w").pack(
                anchor="w", pady=(6, 0))

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3 – Install
    # ──────────────────────────────────────────────────────────────────────────
    def _page_install(self):
        c = self.content
        tk.Label(c, text="Installing", font=(FONT_UI, 15, "bold"),
                 bg=BG, fg=TXT).pack(anchor="w")
        tk.Label(c, text="Please wait while NoCode CV Trainer is set up.",
                 font=(FONT_UI, 9), bg=BG, fg=TXT2).pack(anchor="w", pady=(4, 16))

        self.install_steps_frame = tk.Frame(c, bg=BG)
        self.install_steps_frame.pack(fill="x", pady=(0, 10))

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
            "build_frontend":   "Building frontend bundle",
            "write_launcher":   "Writing launcher files",
            "shortcut":         "Creating desktop shortcut",
        }
        self._step_widgets = {}

        for key in self._step_names:
            row = tk.Frame(self.install_steps_frame, bg=BG)
            row.pack(fill="x", pady=2)

            dot = tk.Canvas(row, width=18, height=18,
                            bg=BG, highlightthickness=0)
            dot.create_oval(3, 3, 15, 15, fill="", outline=TXT3, width=1)
            dot.pack(side="left", padx=(0, 10))

            lbl = tk.Label(row, text=self._step_labels_map[key],
                           font=(FONT_UI, 9), fg=TXT3, bg=BG, anchor="w")
            lbl.pack(side="left")

            self._step_widgets[key] = (dot, lbl)

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Pro.Horizontal.TProgressbar",
                        troughcolor=SURF2, background=ACCENT,
                        lightcolor=ACCENTL, darkcolor=ACCENT2,
                        bordercolor=BORDER, thickness=6)
        self.progress = ttk.Progressbar(c, style="Pro.Horizontal.TProgressbar",
                                        orient="horizontal",
                                        mode="determinate",
                                        maximum=len(self._step_names))
        self.progress.pack(fill="x", pady=(4, 12))

        # Log area
        log_outer = tk.Frame(c, bg=BORDER)
        log_outer.pack(fill="both", expand=True)
        log_inner = tk.Frame(log_outer, bg="#08080b")
        log_inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.log_text = tk.Text(log_inner, bg="#08080b", fg="#5a5a7a",
                                font=(FONT_MONO, 8), wrap="word",
                                relief="flat", bd=0, state="disabled",
                                insertbackground=ACCENT, height=10)
        scrollbar = tk.Scrollbar(log_inner, command=self.log_text.yview,
                                 bg=SURF2, troughcolor="#08080b",
                                 relief="flat", bd=0, width=6)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 2), pady=4)
        self.log_text.pack(side="left", fill="both", expand=True, padx=10, pady=8)

    def _log(self, msg, color=None):
        def _do():
            self.log_text.config(state="normal")
            if "✓" in msg or "✅" in msg:
                tag = "ok"
                self.log_text.tag_config("ok", foreground=SUCCESS)
            elif "❌" in msg or "✗" in msg:
                tag = "err"
                self.log_text.tag_config("err", foreground=ERROR)
            elif "⚠" in msg:
                tag = "warn"
                self.log_text.tag_config("warn", foreground=WARN)
            else:
                tag = None
            if tag:
                self.log_text.insert("end", msg + "\n", tag)
            else:
                self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _do)

    def _mark_step(self, key, state):
        dot, lbl = self._step_widgets[key]
        dot.delete("all")
        if state == "running":
            dot.create_oval(3, 3, 15, 15, fill=WARN, outline="")
            lbl.config(fg=TXT)
        elif state == "done":
            dot.create_oval(3, 3, 15, 15, fill=SUCCESS, outline="")
            dot.create_text(9, 9, text="✓", fill="white",
                            font=(FONT_UI, 7, "bold"))
            lbl.config(fg=TXT2)
        elif state == "skip":
            dot.create_oval(3, 3, 15, 15, fill=SURF3, outline=TXT3, width=1)
            dot.create_text(9, 9, text="—", fill=TXT3, font=(FONT_UI, 8))
            lbl.config(fg=TXT3)
        elif state == "error":
            dot.create_oval(3, 3, 15, 15, fill=ERROR, outline="")
            dot.create_text(9, 9, text="✗", fill="white",
                            font=(FONT_UI, 7, "bold"))
            lbl.config(fg=ERROR)

    def _advance_progress(self):
        self.after(0, lambda: self.progress.step(1))

    # ── Install thread ────────────────────────────────────────────────────────
    def _start_install(self):
        threading.Thread(target=self._install_thread, daemon=True).start()

    def _install_thread(self):
        torch_mode    = self.torch_choice.get()
        want_shortcut = self.shortcut_var.get()
        use_system    = (self.install_mode.get() == "system")

        if use_system:
            venv_py  = sys.executable
            venv_pip = sys.executable
            qpy  = f'"{venv_py}"'
            qpip = f'"{venv_pip}" -m pip'
        else:
            venv_py  = VENV_PY
            venv_pip = VENV_PIP
            qpy  = f'"{venv_py}"'
            qpip = f'"{venv_pip}"'

        def step(key, fn):
            self.after(0, lambda: self._mark_step(key, "running"))
            result = fn()
            ok    = result[0]
            state = result[2] if len(result) > 2 else ("done" if ok else "error")
            self.after(0, lambda s=state, k=key: self._mark_step(k, s))
            self._advance_progress()
            return ok

        # 1. Create venv
        def do_venv():
            stale = os.path.join(BACKEND, "_torch_constraints.txt")
            if os.path.isfile(stale):
                try:
                    os.remove(stale)
                except Exception:
                    pass

            if use_system:
                rc_test, ver_out = run_cmd(f'{qpy} -c "import sys; print(sys.version)"')
                if rc_test == 0:
                    self._log(f"  Using system Python {ver_out.strip()[:60]}")
                    self._log("  Packages will be installed into your existing environment.")
                    return True, "", "skip"
                self._log("  ❌ System Python does not respond.")
                return False, ""

            if os.path.isfile(VENV_PY):
                rc_test, _ = run_cmd(f'"{VENV_PY}" -c "import sys; print(sys.version)"')
                if rc_test == 0:
                    self._log("  ✓ Existing venv found and working — reusing.")
                    return True, "", "skip"
                self._log("  ⚠ Existing venv is broken, recreating…")
                shutil.rmtree(VENV_DIR, ignore_errors=True)

            self._log("  Creating virtual environment…")
            rc, out = run_cmd(f'"{sys.executable}" -m venv "{VENV_DIR}"')
            self._log(out[:800] if out else "  done")
            return rc == 0, out

        if not step("create_venv", do_venv):
            self._log("❌ Failed to create venv. Aborting.")
            return

        # 2. Install PyTorch
        def do_torch():
            rc_chk, out_chk = run_cmd(
                f"{qpy} -c \"import torch; print(torch.__version__)\"")
            if rc_chk == 0 and torch_mode == "auto":
                rc_cuda, _ = run_cmd(
                    f"{qpy} -c \"import torch; torch.zeros(1).cuda() if torch.cuda.is_available() else None; print('ok')\"")
                if rc_cuda == 0:
                    env_label = "system env" if use_system else "venv"
                    self._log(f"  ✓ torch {out_chk.strip()} already in {env_label}, skipping.")
                    return True, "", "skip"
                self._log(f"  ⚠ torch found but CUDA test failed — reinstalling.")

            urls = {
                "cpu":   "https://download.pytorch.org/whl/cpu",
                "cu118": "https://download.pytorch.org/whl/cu118",
                "cu124": "https://download.pytorch.org/whl/cu124",
                "cu128": "https://download.pytorch.org/whl/cu128",
            }
            effective_mode = self.torch_choice.get() if torch_mode == "auto" else torch_mode
            if effective_mode == "auto":
                effective_mode = "cpu"
            url = urls.get(effective_mode, urls["cpu"])

            sizes = {"cpu": "~250 MB", "cu118": "~2.3 GB", "cu124": "~2.4 GB", "cu128": "~2.8 GB"}
            self._log(f"  Downloading PyTorch ({effective_mode})  {sizes.get(effective_mode, '')}")
            self._log(f"  This may take several minutes…")
            self._log(f"  Index: {url}\n")

            cmd = (f"{qpip} install torch torchvision "
                   f"--index-url {url} --no-warn-script-location")
            last_pkg = [None]

            def on_line(line):
                low = line.lower()
                if any(k in low for k in ("downloading", "installing", "successfully",
                                          "error", "warning", "requirement already")):
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
            self._log("❌ PyTorch install failed. Check internet or try a different backend.")
            return

        # 3. Install backend
        def do_backend():
            req = os.path.join(BACKEND, "requirements.txt")
            self._log("  Upgrading pip…")
            run_cmd(f'{qpy} -m pip install --upgrade pip --quiet')

            urls = {
                "cpu":   "https://download.pytorch.org/whl/cpu",
                "cu118": "https://download.pytorch.org/whl/cu118",
                "cu124": "https://download.pytorch.org/whl/cu124",
                "cu128": "https://download.pytorch.org/whl/cu128",
            }
            effective_mode = self.torch_choice.get() if torch_mode == "auto" else torch_mode
            if effective_mode == "auto":
                effective_mode = "cpu"
            torch_url = urls.get(effective_mode, urls["cpu"])

            self._log("  Installing Pillow (binary only)…")
            rc_p, _ = run_cmd(
                f'{qpip} install "pillow>=10.0.0" '
                f'--only-binary=pillow --prefer-binary --no-warn-script-location')
            if rc_p != 0:
                self._log("  Pillow latest wheel not found — trying pillow 10.4.0…")
                run_cmd(f'{qpip} install "pillow==10.4.0" --only-binary=pillow --no-warn-script-location')

            self._log(f"  Installing backend libraries  [{effective_mode}]…")
            error_lines = []

            def on_line(line):
                low = line.lower()
                if any(k in low for k in ("error", "could not", "no matching",
                                          "failed", "exception")):
                    error_lines.append(line.strip())
                if any(k in low for k in ("downloading", "installing", "successfully",
                                          "error", "requirement already", "collected",
                                          "could not", "no matching")):
                    disp = line.strip()
                    if len(disp) > 90:
                        disp = disp[:87] + "…"
                    self._log(f"  {disp}")

            cmd = (f'{qpip} install -r "{req}" '
                   f'--extra-index-url {torch_url} '
                   f'--prefer-binary --no-warn-script-location')
            rc, _ = run_cmd_stream(cmd, cwd=BACKEND, line_cb=on_line)

            if rc != 0:
                self._log("  ❌ Backend install failed:")
                for el in error_lines[-15:]:
                    self._log(f"     {el}")
                return False, ""

            if effective_mode != "cpu":
                rc_v, ver_out = run_cmd(f'{qpy} -c "import torch; print(torch.__version__)"')
                if rc_v == 0:
                    ver = ver_out.strip()
                    if f"+{effective_mode}" not in ver:
                        self._log(f"  ⚠ torch downgraded to {ver} — restoring {effective_mode} build…")
                        run_cmd_stream(
                            f'{qpip} install torch torchvision '
                            f'--index-url {torch_url} --force-reinstall '
                            f'--no-warn-script-location',
                            line_cb=on_line)
                        self._log("  ✓ CUDA build restored.")
                    else:
                        self._log(f"  ✓ torch {ver} confirmed.")

            self._log("  ✓ Backend libraries installed.")
            return True, ""

        if not step("install_backend", do_backend):
            self._log("❌ Backend dependency install failed.")
            return

        # 4. npm install
        dist_ready = os.path.isfile(os.path.join(DIST, "index.html"))

        def do_npm():
            if dist_ready:
                self._log("  ✓ Pre-built frontend detected — skipping npm install.")
                return True, "", "skip"
            already = os.path.isdir(os.path.join(FRONTEND, "node_modules"))
            if not already:
                self._log("  Downloading npm packages — first install may take a minute…")
            else:
                self._log("  node_modules found — verifying packages…")

            def on_npm(line):
                low = line.lower()
                if any(k in low for k in ("added", "updated", "removed", "audited",
                                          "warn", "error", "npm error")):
                    if line.strip():
                        self._log(f"  {line.strip()[:90]}")

            rc, _ = run_cmd_stream("npm install", cwd=FRONTEND, line_cb=on_npm)
            if rc == 0:
                self._log("  ✓ npm packages ready.")
            return rc == 0, ""

        if not step("install_frontend", do_npm):
            self._log("❌ npm install failed.")
            return

        # 5. Build frontend
        def do_build():
            if dist_ready:
                self._log("  ✓ Pre-built frontend detected — skipping build step.")
                return True, "", "skip"
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
            self._log("❌ Frontend build failed.")
            return

        # 6. Write launcher files
        def do_launcher():
            self._write_launcher_files(venv_py if use_system else None)
            return True, ""

        step("write_launcher", do_launcher)

        # 7. Desktop shortcut
        def do_shortcut():
            if not want_shortcut or not _IS_WIN:
                return True, "", "skip"
            launcher_bat = os.path.join(ROOT_DIR, "NoCode CV.bat")
            ok = create_desktop_shortcut(launcher_bat)
            return ok, ""

        step("shortcut", do_shortcut)

        self._install_done = True
        self._log("\n✅ Installation complete!")
        self.after(0, self._on_install_complete)

    def _on_install_complete(self):
        self.btn_next.config(text="Finish  →", state="normal",
                             bg=SUCCESS, fg="#fff",
                             command=lambda: self._show_step(4))
        bind_hover(self.btn_next, SUCCESS, "#2d8a4e", "#fff", "#fff")

    # ── Write launcher files ──────────────────────────────────────────────────
    def _write_launcher_files(self, system_py=None):
        if _IS_WIN:
            self._write_bat_launcher(system_py)
        else:
            self._write_sh_launcher(system_py)

    def _write_bat_launcher(self, system_py=None):
        self._log("  Writing NoCode CV.bat…")
        if system_py:
            py_path = system_py.replace("\\", "\\\\")
            bat = (
                "@echo off\r\n"
                "title NoCode CV Trainer\r\n"
                "cd /d \"%~dp0\"\r\n"
                f"start \"\" \"{py_path}\" \"%~dp0launcher.py\"\r\n"
            )
        else:
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

    def _write_sh_launcher(self, system_py=None):
        self._log("  Writing NoCode CV.sh…")
        if system_py:
            py_line = f'"{system_py}" "$DIR/launcher.py"'
        else:
            py_line = '"$DIR/venv/bin/python" launcher.py'
        sh = (
            "#!/usr/bin/env bash\n"
            "# NoCode CV Trainer — Launcher\n"
            'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            "cd \"$DIR\"\n"
            f"{py_line}\n"
        )
        sh_path = os.path.join(ROOT_DIR, "NoCode CV.sh")
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write(sh)
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass
        self._log(f"  Written: {sh_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4 – Done
    # ──────────────────────────────────────────────────────────────────────────
    def _page_done(self):
        c = self.content

        accent_col = SUCCESS if self._install_done else WARN
        icon  = "✅" if self._install_done else "⚠️"
        title = "Installation Complete" if self._install_done else "Installation Incomplete"
        sub   = ("NoCode CV Trainer is ready to launch."
                 if self._install_done
                 else "One or more steps failed — check the log on the previous page.")

        # Hero card
        hero_outer, hero = bordered_frame(c, bg=SURF2)
        hero_outer.pack(fill="x", pady=(6, 0))
        tk.Frame(hero, bg=accent_col, height=3).pack(fill="x")
        hero_body = tk.Frame(hero, bg=SURF2)
        hero_body.pack(fill="both", expand=True, padx=32, pady=28)

        tk.Label(hero_body, text=icon, font=(FONT_EMOJI, 38), bg=SURF2).pack()
        tk.Label(hero_body, text=title,
                 font=(FONT_UI, 15, "bold"), bg=SURF2, fg=TXT).pack(pady=(10, 4))
        tk.Label(hero_body, text=sub,
                 font=(FONT_UI, 9), bg=SURF2, fg=TXT2).pack()

        if self._install_done:
            launch_btn = tk.Button(
                hero_body,
                text="  Launch NoCode CV  →",
                font=(FONT_UI, 11, "bold"),
                bg=ACCENT, fg="#fff", relief="flat", bd=0,
                padx=32, pady=12, cursor="hand2",
                activebackground=ACCENT2, activeforeground="#fff",
                command=self._launch_app)
            launch_btn.pack(pady=(20, 0))
            bind_hover(launch_btn, ACCENT, ACCENT2, "#fff", "#fff")

        # Info grid
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(20, 14))

        launcher_name = "NoCode CV.bat" if _IS_WIN else "NoCode CV.sh"
        launch_hint = (f"Double-click  {launcher_name}  or the desktop shortcut"
                       if _IS_WIN else f"Run:  bash '{launcher_name}'  in a terminal")
        info_items = [
            ("Launch later",  launch_hint),
            ("App URL",       "http://localhost:8000  (opens automatically on start)"),
            ("Training logs", "Streamed live in the browser during training"),
        ]
        for key, val in info_items:
            row = tk.Frame(c, bg=BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{key}:", font=(FONT_UI, 8, "bold"),
                     bg=BG, fg=TXT2, width=14, anchor="w").pack(side="left")
            tk.Label(row, text=val, font=(FONT_UI, 8),
                     bg=BG, fg=TXT2, anchor="w").pack(side="left")

    def _launch_app(self):
        launcher = os.path.join(ROOT_DIR, "launcher.py")
        if _IS_WIN:
            venv_pyw = os.path.join(VENV_DIR, "Scripts", "pythonw.exe")
            py = (venv_pyw if os.path.isfile(venv_pyw)
                  else (VENV_PY if os.path.isfile(VENV_PY) else sys.executable))
        else:
            py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
        subprocess.Popen([py, launcher], cwd=ROOT_DIR)
        self.after(1500, self.destroy)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Installer()
    app.mainloop()
