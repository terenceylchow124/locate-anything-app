"""Modal deployment of backend/app.py itself (the FastAPI /detect
orchestrator: tiling + merging), as opposed to modal_app/model_server.py
(the GPU model behind it). Together these two separately-deployed Modal apps
let the whole stack run off Modal instead of `docker compose` -- see
DEPLOYMENT.md's "Deploying to Vercel" plan (frontend still goes to Vercel).

CPU-only -- this process never touches the model directly, it tiles images
and makes HTTP calls to whichever LA_INFERENCE_BACKEND is configured (in
practice, modal_app/model_server.py's endpoint, via backend/modal_engine.py).
Deliberately does NOT support LA_INFERENCE_BACKEND=local here: the compiled
liblocate_anything.so + GGUF weights aren't part of this image, and running
the CPU engine on Modal would defeat the point of using a GPU-hosted model.

Deploy:
    modal deploy modal_app/backend_server.py

Requires a Modal secret with the same settings backend/app.py reads from
its environment (see .env.example) -- created once per account:
    modal secret create la-backend-config \\
        LA_INFERENCE_BACKEND=modal \\
        LA_MODAL_URL=<modal_app/model_server.py's printed endpoint URL> \\
        LA_MODAL_TOKEN=<the la-modal-shared-token secret's value> \\
        LA_TILE_CONCURRENCY=8 \\
        LA_REQUEST_CONCURRENCY=5

Prints a web endpoint URL ending in a path Vite's VITE_API_BASE_URL should
point at (the frontend calls relative /detect against this base). CORS is
already wide open in backend/app.py (portfolio demo, no user data) so no
extra config is needed for a Vercel-hosted frontend to call this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

app = modal.App("locate-anything-backend")

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
CONTAINER_BACKEND_DIR = "/app/backend"

# Mirrors backend/requirements.txt (not read from the file directly -- Modal
# builds this image without the repo checked out yet at image-build time for
# add_local_dir, which happens as its own late layer; keeping the package
# list here explicit avoids a chicken-and-egg read of a file that isn't in
# the image until after pip_install already ran).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.110",
        "uvicorn[standard]>=0.29",
        "python-multipart>=0.0.9",
        "pillow>=10.0",
        "tritonclient[http]>=2.40",
        "numpy>=1.26",
        "httpx>=0.27",
    )
    .add_local_dir(
        BACKEND_DIR,
        CONTAINER_BACKEND_DIR,
        ignore=["__pycache__", "__pycache__/**", "tests", "tests/**"],
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("la-backend-config")],
    # CPU-bound work here is just PIL tiling/merging -- the actual inference
    # wait happens in HTTP calls out to the GPU model server, so one
    # container can comfortably serve several concurrent /detect requests
    # (bounded again, more conservatively, by LA_REQUEST_CONCURRENCY inside
    # the app itself -- see backend/queueing.py).
    scaledown_window=60,
    timeout=600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, CONTAINER_BACKEND_DIR)
    from app import app as backend_app

    return backend_app
