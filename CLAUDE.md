# CLAUDE.md

Agent-facing reference for this repo. See also `README.md` (user-facing quickstart), `AGENTS.md` (issue tracker, coding standards), `DEPLOYMENT.md` (GPU platform choice, Modal/Vercel deployment, cost/latency data), and `TODO.md` (outstanding work).

## Architecture

```
frontend (React/Vite SPA, :5173 in docker)
      │  POST /detect (multipart: file, prompt, mode)
      ▼
backend (FastAPI, :8000 in docker) -- backend/app.py
      │  tiles the image (tiling.py), runs each tile through the
      │  configured InferenceEngine, merges results by IoU
      ▼
InferenceEngine (backend/engine.py's Protocol) -- one of:
      local   -- la_capi.py, ctypes binding to a compiled locate-anything.cpp
                 (CPU, default, no GPU anywhere)
      triton  -- triton_engine.py, HTTP client to a self-hosted Triton
                 Inference Server (e.g. on a DGX Spark)
      modal   -- modal_engine.py, HTTP client to modal_app/model_server.py
                 (Modal-hosted GPU deployment)
```

All three engines satisfy the same `locate_buffer(image_bytes, prompt, mode) -> list[dict]` contract (`backend/engine.py`), so `app.py`/`tiling.py`/`queueing.py` don't know or care which one is active. Adding a fourth backend (e.g. Replicate) means writing one new engine class, not touching the rest of the backend.

**GPU-side model server** (`modal_app/model_server.py`, only used by the `modal` engine): a Modal `@app.cls` that loads `nvidia/LocateAnything-3B` once per container and exposes `/detect` over HTTP with the same wire contract as Triton. See `DEPLOYMENT.md` for deploying it, GPU/concurrency settings, and cost data.

**Per-box confidence score**: the model's `generate()` doesn't return one (a versioning mismatch between the model card's documented `predict()` and the actual downloaded revision — see `TODO.md` for the still-open Triton-side verification). Both `modal_app/model_server.py` and `triton/model_repository/locate_anything/1/model.py` compute it themselves via `_score_boxes()`: a teacher-forcing pass over the already-generated tokens (reusing the same vision embeddings `generate()` was seeded with), averaging softmax probability over each box's token span. The frontend's min-score filter (`OverlayView`/`ResultGrid`) depends on this being non-null.

## Backend settings (env vars)

All read by `backend/app.py`; see `.env.example` for the full annotated list and defaults. `docker compose up` reads `.env` automatically.

| Var | Purpose |
|---|---|
| `LA_INFERENCE_BACKEND` | `local` (default) / `triton` / `modal` |
| `LA_LIB_PATH`, `LA_MODEL_PATH` | local backend: path to the compiled `.so` and GGUF weights |
| `LA_TRITON_URL`, `LA_TRITON_MODEL_NAME` | triton backend |
| `LA_MODAL_URL`, `LA_MODAL_TOKEN` | modal backend: endpoint URL + shared bearer token |
| `LA_TILE_CONCURRENCY` | how many tiles of *one image* run concurrently (default `1`) |
| `LA_REQUEST_CONCURRENCY` | how many whole `/detect` requests run concurrently (default `1`) — e.g. the frontend's multi-prompt comparison |

`LA_TILE_CONCURRENCY` and `LA_REQUEST_CONCURRENCY` are orthogonal axes (`backend/queueing.py`'s `SingleWorkerQueue` bounds the request axis; `run_detection`'s `asyncio.gather`+`Semaphore` bounds the tile axis) — multiply them for the worst-case concurrent tile calls hitting the inference backend. **Both fail fast at import time if set `>1` with `LA_INFERENCE_BACKEND=local`**: `la_capi.LocateAnythingEngine` is a single loaded model instance and isn't safe for concurrent calls. Safe to raise for `triton`/`modal`. See `DEPLOYMENT.md` for what raising them actually costs/buys on Modal.

Frontend build-time vars (`VITE_*`, baked in by `frontend/Dockerfile`, need `docker compose up --build frontend` to take effect): `VITE_API_BASE_URL`, `VITE_DEFAULT_SCENE_ID`.

## Frontend structure

- **Scene registry**: `frontend/public/scene-config/scenes.json` + `frontend/public/scene-config/images/`, fetched at runtime (`frontend/src/scenes.ts`'s `loadScenes()`), not bundled at build time. `docker-compose.yml` bind-mounts `frontend/public/scene-config` as a **directory** (not individual files) so editing `scenes.json` or dropping in a new image takes effect on browser refresh, no rebuild. `default_image` can be a local `/scene-config/images/...` path or a full external URL — both just get `fetch()`'d the same way (`App.tsx`'s `resolveImageBlob`); an external URL needs the remote host to allow cross-origin reads for detection to work, even though plain `<img>` display doesn't need that.
- **Why a directory mount, not a file mount**: a single-file bind mount breaks the moment the file is saved via the common write-new-file-then-rename pattern (most editors/tools) — the mount tracks the original inode, which the rename orphans, so edits silently stop showing up in the container. Mounting the containing directory avoids that.
- **Multi-prompt comparison** (`frontend/src/hooks/useComparison.ts`): up to `MAX_PROMPTS=5` (`PromptTags.tsx`) prompts run concurrently against the same image (`Promise.all`), each with independent `pending`/`in-flight`/`done`/`error` state (`ComparisonPanelState`) — completion order isn't fixed.

## Running locally via Docker

```sh
git submodule update --init --recursive   # once, if cloned fresh
docker compose up --build
```

Frontend: http://localhost:5173 — backend: http://localhost:8000 (`/docs` for the OpenAPI UI). The frontend's nginx stage reverse-proxies relative `/detect` calls to the backend container, so no `VITE_API_BASE_URL` override is needed for this setup.

`backend/Dockerfile` compiles `locate-anything.cpp` from source regardless of which `LA_INFERENCE_BACKEND` is active (one Dockerfile for all three backends, simpler than three) — wasteful build time if you're only ever using `triton`/`modal`, but functionally harmless.

**After any frontend code change** (`.ts`/`.tsx`, not `scenes.json`/images): `docker compose up -d --build frontend` — `down` + `up -d` alone reuses the existing image and won't pick up source changes.

## Testing

```sh
# Backend (from backend/, conda env `locateanything` -- see README's
# "Development environment"):
python3 -m pytest -q -m "not slow and not calibration"   # fast unit tests
python3 -m pytest -q                                      # includes real-model e2e tests (slow, needs local weights)
RUN_CALIBRATION=1 python3 -m pytest -q -m calibration     # very slow (30-55+ min/scene), opt-in

# Frontend (from frontend/):
npx tsc -b        # typecheck
npx oxlint         # lint
npm run build      # full production build
```

Backend unit tests (engine selection, tiling, queueing, modal/triton engines) are mocked and fast — no GPU/model needed. The `test_detect.py` contract/calibration tests need the real local CPU engine (`.so` + GGUF weights) and are auto-skipped if those aren't present.
