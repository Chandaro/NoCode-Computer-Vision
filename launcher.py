"""
NoCode CV Trainer – App Launcher
Starts the FastAPI backend (using venv), polls until ready, opens the browser,
and shows a small "running" status window with a Stop button.

Run via:  NoCode CV.bat  (created by installer)
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

# ─── Colours (match app theme) ────────────────────────────────────────────────
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


# ─── Launcher Window ──────────────────────────────────────────────────────────
class LauncherWindow(tk.Tk):
    WIN_W = 400
    WIN_H = 290

    def __init__(self):
        super().__init__()
        self.title("NoCode CV Trainer")
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._server_proc = None
        self._running = False

        self._build_ui()
        threading.Thread(target=self._start_server, daemon=True).start()

    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - self.WIN_W) // 2
        y = (self.winfo_screenheight() - self.WIN_H) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=SURFACE)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🧠  NoCode CV Trainer",
                 font=("Segoe UI", 12, "bold"), bg=SURFACE, fg=TXT).pack(
                 side="left", padx=16, pady=14)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Status area ───────────────────────────────────────────────────────
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True, padx=24, pady=20)

        self.status_dot = tk.Label(mid, text="⏳", font=("Segoe UI Emoji", 22),
                                   bg=BG, fg=WARN)
        self.status_dot.pack()

        self.status_lbl = tk.Label(mid, text="Starting server…",
                                   font=("Segoe UI", 11), bg=BG, fg=TXT)
        self.status_lbl.pack(pady=(6, 2))

        self.status_sub = tk.Label(mid, text="Please wait",
                                   font=("Segoe UI", 9), bg=BG, fg=TXT2)
        self.status_sub.pack()

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Launch.Horizontal.TProgressbar",
                        troughcolor=SURF2, background=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT,
                        bordercolor=BORDER, thickness=6)
        self.pbar = ttk.Progressbar(mid, style="Launch.Horizontal.TProgressbar",
                                    orient="horizontal", mode="indeterminate")
        self.pbar.pack(fill="x", pady=(14, 0))
        self.pbar.start(12)

        # ── Bottom button bar ─────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        bar = tk.Frame(self, bg=SURFACE)
        bar.pack(fill="x")

        self.open_btn = tk.Button(bar, text="Open Browser",
                                  font=("Segoe UI", 9), bg=SURF2, fg=TXT2,
                                  relief="flat", bd=0, padx=18, pady=10,
                                  cursor="hand2", state="disabled",
                                  activebackground=SURF2, activeforeground=TXT,
                                  command=lambda: webbrowser.open(APP_URL))
        self.open_btn.pack(side="left", padx=12, pady=10)

        self.stop_btn = tk.Button(bar, text="Stop Server",
                                  font=("Segoe UI", 9), bg=SURF2, fg=ERROR,
                                  relief="flat", bd=0, padx=18, pady=10,
                                  cursor="hand2",
                                  activebackground=SURF2, activeforeground=ERROR,
                                  command=self._on_close)
        self.stop_btn.pack(side="right", padx=12, pady=10)

    def _set_status(self, dot, text, sub, dot_color, ready=False):
        def _do():
            self.status_dot.config(text=dot, fg=dot_color)
            self.status_lbl.config(text=text)
            self.status_sub.config(text=sub)
            if ready:
                self.pbar.stop()
                self.pbar.config(mode="determinate", value=100)
                self.open_btn.config(state="normal", bg=ACCENT, fg="#fff")
        self.after(0, _do)

    # ── Server lifecycle ──────────────────────────────────────────────────────
    def _kill_existing(self, port: int):
        """Kill any process already occupying the port (Windows + Unix)."""
        try:
            if platform.system() == "Windows":
                r = subprocess.run(
                    f"netstat -ano | findstr :{port}",
                    shell=True, capture_output=True, text=True,
                )
                for line in r.stdout.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        pid = int(parts[-1])
                        if pid > 4:   # never kill System / Idle
                            subprocess.run(
                                f"taskkill /F /PID {pid}",
                                shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
            else:
                subprocess.run(
                    f"fuser -k {port}/tcp",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            time.sleep(0.8)   # give OS time to release the port
        except Exception:
            pass

    def _start_server(self):
        cmd = [
            PYTHON, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ]
        # Kill anything already on port 8000 so we always start fresh
        self.after(0, lambda: self.status_sub.config(text="Clearing port 8000…"))
        self._kill_existing(8000)

        # Write server output to a log file for debugging
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
            self._set_status("✗", "Failed to start server", str(e)[:120], ERROR)
            return

        # Poll health endpoint — up to 60 s (torch import alone can take 10 s on slow machines)
        MAX_ATTEMPTS = 120
        for attempt in range(MAX_ATTEMPTS):
            time.sleep(0.5)
            if self._server_proc.poll() is not None:
                # Server exited — read last lines of log for the real error
                error_hint = self._read_log_tail(log_path, lines=6)
                self._set_status("✗", "Server crashed on startup", error_hint, ERROR)
                return
            try:
                with _req.urlopen(HEALTH, timeout=2) as resp:
                    if resp.status == 200:
                        self._running = True
                        self._set_status("✓", "Server is running",
                                         APP_URL, SUCCESS, ready=True)
                        self.after(400, lambda: webbrowser.open(APP_URL))
                        return
            except Exception:
                pass
            self.after(0, lambda a=attempt+1:
                       self.status_sub.config(text=f"Starting… ({a}/{MAX_ATTEMPTS})"))

        self._set_status("⚠", "Server taking too long",
                         "Try clicking Open Browser manually", WARN, ready=True)
        self._running = True

    @staticmethod
    def _read_log_tail(log_path: str, lines: int = 6) -> str:
        """Return the last N non-empty lines of a log file, or a fallback hint."""
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            tail = [l for l in content.splitlines() if l.strip()][-lines:]
            return "\n".join(tail) if tail else f"See {log_path}"
        except Exception:
            return f"See server.log in the app folder"

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
