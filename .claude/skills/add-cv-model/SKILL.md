---
name: add-cv-model
description: Add a new computer vision model or capability to NoCode CV (a new YOLO version, depth / segmentation / tracking / 3D / VLM model, etc.) end to end — research, backend router, frontend page, deps, docs. Use when the user asks to add, integrate, support or upgrade a model or "new technology" in this app, or asks what new CV tech is worth adding.
---

# Add a CV model to NoCode CV

NoCode CV users are non-programmers running everything offline on their own PC.
A model only belongs here if a non-expert can use it with one click and sensible defaults.

## 1. Pick the model (research first)
If the user named no specific model, or asked "what's new", search the web for recent
releases (papers with code, Hugging Face trending, Ultralytics release notes) and propose
**at most 3** candidates in a short table: what it does, licence, VRAM needed, CPU fallback?,
install method. Recommend one and wait for the user's pick.

Accept a model only if:
- weights download automatically (Hugging Face / Ultralytics hub), no manual sign-up;
- runs on an 8 GB consumer GPU, and on CPU (slowly is fine) or fails with a clear message;
- licence allows use in an MIT app (flag AGPL / non-commercial weights to the user);
- installs with pip. If it needs a git clone, follow `setup_dust3r.bat` / `setup_scan3d.bat`.

Before writing code, check whether an existing router already wraps a similar model.
Upgrading `depth.py` or `segmentation.py` with a new option often beats a new page.

## 2. Backend
Copy the shape of `backend/routers/depth.py`:
- `router = APIRouter(tags=["<feature>"])`, paths `/<feature>/...` (no prefix).
- Heavy imports (`torch`, `transformers`, …) inside functions, never at module top.
  The app must still start when this model's deps are missing.
- Module-level cache dict `{model_id: loaded_model}`. Pick `cuda` if available, else CPU.
- Images: `POST /<feature>/infer` (UploadFile) → JSON plus base64 PNG.
- Video / long work: background `threading.Thread`, job dict, `GET .../{job_id}/status`
  with `stage` / `processed` / `total`, `GET .../{job_id}/download`. Write outputs to `video_out/`.
- Turn missing deps, missing GPU, or OOM into `HTTPException` with a plain-English `detail`
  that tells the user what to do.
- Use library constants by name (`cv2.COLORMAP_X`), not magic ints.
- Register it in `backend/main.py` (import + `api.include_router`).
- New deps go in `backend/requirements.txt` under a `# ─── <Feature> ───` header. Never torch.
  If the installer must do something special, update `installer.py` too.

## 3. Frontend
- `frontend/src/pages/<Feature>.tsx`. Base it on `Depth.tsx` / `Segmentation.tsx`.
- Add a `lazy()` import and `<Route path="/projects/:id/<section>">` in `App.tsx`, plus the
  tool-rail entry in `components/ProjectRail.tsx`.
- Add response types to `src/api.ts` and call through the default `api` axios client.
- Use `ui.tsx` primitives and `var(--…)` tokens. Keep the Khmer art theme.
- Give every advanced control a one-line plain-language hint. Defaults must work untouched.
  Show progress for anything slower than ~2 s.

## 4. Verify (there are no tests)
1. `cd backend && ../venv/Scripts/python.exe -c "from main import app"`, so imports are OK.
2. `cd frontend && npm run lint && npm run build` (rebuilds the committed `dist/`).
3. Start the backend on :8000 and run the feature end to end in the browser pane
   (use `bus.jpg` as a test image). Check the CPU path if no GPU is present.
4. Report honestly what was and wasn't exercised.

## 5. Finish
- Update `README.md`: Features, Tech Stack, Project Structure.
- If you learned a new gotcha, add it to `CLAUDE.md` → Gotchas.
- Commit as `feat(<feature>): add <model>` (Conventional Commits). Push only when asked.
