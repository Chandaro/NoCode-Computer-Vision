# NoCode CV — notes for Claude

Self-hosted, offline, no-code **computer vision** workstation (not a résumé tool).
Users annotate images, train models, and run inference entirely from a browser UI
served at http://localhost:8000. Target users are non-programmers, so every feature
must be usable without a terminal.

## Goal of ongoing work
Keep the app broadly useful and keep adding **new CV technology as it appears**
(new YOLO versions, new depth / segmentation / 3D models, etc.). When adding a model,
prefer one that runs locally on a consumer GPU *and* falls back to CPU.

## Stack
- **Backend:** Python 3.9+ (3.11 recommended), FastAPI, SQLModel + SQLite (`nocode_cv.db`),
  Ultralytics YOLO (`ultralytics==8.4.41`), PyTorch (installed separately by `installer.py`
  to match CUDA), HuggingFace `transformers` (depth), DUSt3R (`dust3r/`, set up by
  `setup_dust3r.bat`), yt-dlp for video URLs.
- **Frontend:** React 19, TypeScript (strict), Vite 8, Tailwind 4, react-router 7, axios,
  Recharts, three.js + `@sparkjsdev/spark` (3D / gaussian splats), lucide-react icons.
- **Platform:** developed on Windows; must keep working on macOS/Linux (`.sh` scripts).

## Layout
- `backend/main.py` — app entry. Every router is imported here and mounted under the
  `/api` prefix. Also serves `frontend/dist` as an SPA and exposes `/health` (launcher polls it).
  It globally patches `torch.load(weights_only=False)` so YOLO `.pt` files load on torch 2.6+.
- `backend/routers/*.py` — one file per feature, each defines `router = APIRouter(...)`.
- `backend/models.py` — SQLModel tables. `database.py` — session / `create_db()`.
- `frontend/src/App.tsx` — routes; pages are `lazy()`-loaded under `/projects/:id/<section>`.
- `frontend/src/pages/*.tsx` — one page per feature. `src/api.ts` — axios client
  (`baseURL: '/api'`) plus shared TS interfaces.
- `frontend/src/components/ui.tsx` — shared UI primitives; `KhmerWorlds.tsx`, `Sigil.tsx`, etc. —
  the Khmer-art visual theme. Design tokens live in `src/index.css` as CSS variables
  (`var(--bg)`, `var(--text)`, `var(--border)`…). Use those, not hard-coded colours.
- `installer.py` / `launcher.py` / `*.bat` / `*.sh` — end-user install & launch.
- `frontend/dist/` is **committed**; rebuild it (`npm run build`) after frontend changes.

## Adding a new feature / model (checklist)
1. Backend: new `backend/routers/<feature>.py` with `router = APIRouter(tags=["<feature>"])`.
   Lazy-import heavy libs inside functions and cache loaded models in a module-level dict
   (see `routers/depth.py`). Long jobs run in a background thread with a job-status dict
   the frontend polls.
2. Register it in `backend/main.py` (import + `api.include_router(...)`).
3. Add new Python deps to `backend/requirements.txt` with a section comment; never add
   `torch`/`torchvision` there (the installer pins the CUDA build).
4. Frontend: add `src/pages/<Feature>.tsx`, a `lazy()` import + route in `App.tsx`,
   and any types in `api.ts`. Match the existing Khmer theme and `ui.tsx` components.
5. Explain things in plain language in the UI (users aren't ML engineers): defaults that
   just work, tooltips for every advanced setting, and a clear message when a model or GPU
   is missing.
6. Update `README.md` (features, tech stack, project structure).

## Running & checking
```bash
# backend (from backend/, using the project venv)
../venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# frontend dev server with hot reload (proxies /api to the backend)
cd frontend && npm run dev        # http://localhost:5173
# before committing frontend work
cd frontend && npm run lint && npm run build
```
There is no automated test suite: verify by running the app and exercising the feature.

## Conventions
- Commits: Conventional Commits with scope, e.g. `feat(scan): ...`, `fix(scan): ...`, `style: ...`.
- Comments explain *why* (see the MIME-type and torch.load notes in `main.py`).
- Don't commit large model weights, datasets, `venv/`, `server.log`, or `video_out/` output.
