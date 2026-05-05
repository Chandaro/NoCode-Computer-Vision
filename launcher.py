"""
NoCode CV Trainer – App Launcher
Starts the FastAPI backend (using venv), polls until ready, opens the browser,
and shows a status window with a Stop button.

Run via:  NoCode CV.bat  (Windows)
          bash 'NoCode CV.sh'  (macOS / Linux)
"""

import tkinter as tk
from tkinter import ttk
import subprocess, threading, time, webbrowser, os, sys, platform
try:
    import urllib.request as _req
except ImportError:
    _req = None

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "venv")
BACKEND  = os.path.join(ROOT_DIR, "backend")
APP_URL  = "http://localhost:8000"
HEALTH   = f"{APP_URL}/health"

if platform.system() == "Windows":
    VENV_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:
    VENV_PY = os.path.join(VENV_DIR, "bin", "python")

PYTHON = VENV_PY if os.path.isfile(VENV_PY) else sys.executable

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"

FONT_UI    = "Segoe UI"       if _IS_WIN else ("SF Pro Text"       if _IS_MAC else "Helvetica")
FONT_EMOJI = "Segoe UI Emoji" if _IS_WIN else ("Apple Color Emoji" if _IS_MAC else "Noto Color Emoji")
FONT_MONO  = "Courier New"

# ─── Colours ──────────────────────────────────────────────────────────────────
BG      = "#0b0b0e"
SURFACE = "#131318"
SURF2   = "#1b1b24"
SURF3   = "#22222f"
BORDER  = "#2b2b3c"
ACCENT  = "#5865f2"
ACCENT2 = "#4752c4"
ACCENTL = "#6e79f5"
TXT     = "#eeeef2"
TXT2    = "#7a7a98"
TXT3    = "#3c3c58"
SUCCESS = "#3ba55d"
ERROR   = "#ed4245"
WARN    = "#faa61a"


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


# ─── Launcher Window ──────────────────────────────────────────────────────────
class LauncherWindow(tk.Tk):
    WIN_W = 440
    WIN_H = 310

    def __init__(self):
        super().__init__()
        self.title("NoCode CV Trainer")
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._server_proc = None
        self._running     = False
        self._dot_step    = 0
        self._dot_job     = None

        self._build_ui()
        threading.Thread(target=self._start_server, daemon=True).start()

    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - self.WIN_W) // 2
        y = (self.winfo_screenheight() - self.WIN_H) // 2
        self.geometry(f"+{x}+{y}")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top accent bar
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        # Header
        hdr = tk.Frame(self, bg=SURFACE)
        hdr.pack(fill="x")

        hdr_inner = tk.Frame(hdr, bg=SURFACE)
        hdr_inner.pack(fill="x", padx=20, pady=14)

        left = tk.Frame(hdr_inner, bg=SURFACE)
        left.pack(side="left", fill="x", expand=True)

        tk.Label(left, text="🧠  NoCode CV Trainer",
                 font=(FONT_UI, 11, "bold"), bg=SURFACE, fg=TXT,
                 anchor="w").pack(anchor="w")
        tk.Label(left, text="Local AI training environment",
                 font=(FONT_UI, 8), bg=SURFACE, fg=TXT3,
                 anchor="w").pack(anchor="w", pady=(2, 0))

        # URL badge (right side of header)
        url_frame = tk.Frame(hdr_inner, bg=SURF2,
                             highlightbackground=BORDER, highlightthickness=1)
        url_frame.pack(side="right")
        tk.Label(url_frame, text="localhost:8000",
                 font=(FONT_MONO, 8), bg=SURF2, fg=TXT2,
                 anchor="center").pack(padx=10, pady=5)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Status area ───────────────────────────────────────────────────────
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True, padx=28, pady=22)

        # Indicator row
        ind_row = tk.Frame(mid, bg=BG)
        ind_row.pack(anchor="center")

        self.status_dot = tk.Canvas(ind_row, width=14, height=14,
                                    bg=BG, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 10))
        self._draw_dot(WARN)

        self.status_lbl = tk.Label(ind_row, text="Starting server…",
                                   font=(FONT_UI, 12, "bold"), bg=BG, fg=TXT)
        self.status_lbl.pack(side="left")

        self.status_sub = tk.Label(mid, text="Please wait",
                                   font=(FONT_UI, 9), bg=BG, fg=TXT2)
        self.status_sub.pack(pady=(8, 0))

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Launch.Horizontal.TProgressbar",
                        troughcolor=SURF2, background=ACCENT,
                        lightcolor=ACCENTL, darkcolor=ACCENT2,
                        bordercolor=BORDER, thickness=5)
        self.pbar = ttk.Progressbar(mid, style="Launch.Horizontal.TProgressbar",
                                    orient="horizontal", mode="indeterminate")
        self.pbar.pack(fill="x", pady=(16, 0))
        self.pbar.start(10)

        # ── Button bar ────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        bar = tk.Frame(self, bg=SURFACE)
        bar.pack(fill="x")

        bar_inner = tk.Frame(bar, bg=SURFACE)
        bar_inner.pack(fill="x", padx=16, pady=10)

        self.stop_btn = tk.Button(bar_inner, text="Stop Server",
                                  font=(FONT_UI, 9), bg=SURFACE, fg=ERROR,
                                  relief="flat", bd=0, padx=18, pady=8,
                                  cursor="hand2",
                                  activebackground=SURF2, activeforeground=ERROR,
                                  command=self._on_close)
        self.stop_btn.pack(side="right")
        bind_hover(self.stop_btn, SURFACE, SURF2)

        self.open_btn = tk.Button(bar_inner, text="Open Browser",
                                  font=(FONT_UI, 9, "bold"), bg=SURF2, fg=TXT2,
                                  relief="flat", bd=0, padx=20, pady=8,
                                  cursor="hand2", state="disabled",
                                  activebackground=ACCENT2, activeforeground="#fff",
                                  command=lambda: webbrowser.open(APP_URL))
        self.open_btn.pack(side="right", padx=(0, 8))

    def _draw_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 13, 13, fill=color, outline="")

    # ── Status helpers ────────────────────────────────────────────────────────
    def _set_status(self, text, sub, dot_color, ready=False, error=False):
        def _do():
            self._draw_dot(dot_color)
            self.status_lbl.config(text=text,
                                   fg=ERROR if error else TXT)
            self.status_sub.config(text=sub)
            if ready:
                self.pbar.stop()
                self.pbar.config(mode="determinate", value=100)
                self.open_btn.config(state="normal",
                                     bg=ACCENT, fg="#fff",
                                     activebackground=ACCENT2)
                bind_hover(self.open_btn, ACCENT, ACCENT2, "#fff", "#fff")
            elif error:
                self.pbar.stop()
                self.pbar.config(mode="determinate", value=0)
        self.after(0, _do)

    # ── Server lifecycle ──────────────────────────────────────────────────────
    def _kill_existing(self, port: int):
        try:
            if platform.system() == "Windows":
                r = subprocess.run(
                    f"netstat -ano | findstr :{port}",
                    shell=True, capture_output=True, text=True)
                for line in r.stdout.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        pid = int(parts[-1])
                        if pid > 4:
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            elif platform.system() == "Darwin":
                subprocess.run(
                    f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
                    shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(f"fuser -k {port}/tcp", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.8)
        except Exception:
            pass

    def _start_server(self):
        cmd = [PYTHON, "-m", "uvicorn", "main:app",
               "--host", "127.0.0.1", "--port", "8000"]

        self.after(0, lambda: self.status_sub.config(text="Clearing port 8000…"))
        self._kill_existing(8000)

        log_path = os.path.join(ROOT_DIR, "server.log")
        try:
            self._log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        except Exception:
            self._log_file = subprocess.DEVNULL

        try:
            self._server_proc = subprocess.Popen(
                cmd, cwd=BACKEND,
                stdout=self._log_file,
                stderr=self._log_file,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if platform.system() == "Windows" else 0),
            )
        except Exception as e:
            self._set_status("Failed to start server", str(e)[:80],
                             ERROR, error=True)
            return

        MAX_ATTEMPTS = 120
        for attempt in range(MAX_ATTEMPTS):
            time.sleep(0.5)
            if self._server_proc.poll() is not None:
                error_hint = self._read_log_tail(log_path, lines=6)
                self._set_status("Server crashed on startup",
                                 error_hint, ERROR, error=True)
                return
            try:
                with _req.urlopen(HEALTH, timeout=2) as resp:
                    if resp.status == 200:
                        self._running = True
                        self._set_status("Server running", APP_URL,
                                         SUCCESS, ready=True)
                        self.after(400, lambda: webbrowser.open(APP_URL))
                        return
            except Exception:
                pass
            self.after(0, lambda a=attempt + 1:
                       self.status_sub.config(
                           text=f"Starting…  {a} / {MAX_ATTEMPTS}"))

        self._set_status("Server taking too long",
                         "Click  Open Browser  to try manually",
                         WARN, ready=True)
        self._running = True

    @staticmethod
    def _read_log_tail(log_path: str, lines: int = 6) -> str:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            tail = [l for l in content.splitlines() if l.strip()][-lines:]
            return "\n".join(tail) if tail else f"See {log_path}"
        except Exception:
            return "See server.log in the app folder"

    def _stop_server(self):
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except Exception:
                self._server_proc.kill()

    def _on_close(self):
        self._stop_server()
        self.destroy()


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    win = LauncherWindow()
    win.mainloop()
