# CLAUDE.md

Agent-facing notes for this repo. See also `README.md` (architecture, quickstart) and `AGENTS.md` (issue tracker, coding standards).

## Modal backend: implemented (2026-07-26)

`LA_INFERENCE_BACKEND=modal` is live: `modal_app/model_server.py` (GPU model
server, deployed on Modal) + `backend/modal_engine.py` (client). Verified
end-to-end against the real `nvidia/LocateAnything-3B` weights on a deployed
Modal A10G container, including via the full `docker compose up` stack (see
"Deploying modal_app" below for how to redeploy and the env vars needed to
point the main backend at it).

### box_scores bug: fixed in both model.py's (2026-07-26)

This model revision's `generate()` (in the model's `trust_remote_code`
modeling file, `modeling_locateanything.py`) returns just the decoded answer
*string* when called with `verbose=False` — not the `(answer, box_scores)`
tuple both `triton/model_repository/locate_anything/1/model.py` and the first
draft of `modal_app/model_server.py` assumed (going by the model card's
documented `predict()`). There is no per-box confidence anywhere in this
revision's `generate()`; `verbose=True` instead returns `(answer,
sampling_history, out_info)`, still with no box-level scores. Confirmed by
inspecting the actual remote-code source on HF.

**Both `modal_app/model_server.py` and `triton/model_repository/locate_anything/1/model.py`
now fix the crash AND restore a real per-box `score`**, via a `_score_boxes()`
teacher-forcing pass added to both files: after `generate()` returns the
answer string, re-tokenize it (fast tokenizer, `return_offsets_mapping=True`
— verified this model's tokenizer round-trips the generated text exactly, so
character offsets line up with the `<box>...</box>` regex match spans), run
one forward pass of `self.model.language_model(...)` over
prompt-tokens+continuation-tokens together (teacher forcing, `use_cache=False`,
reusing the *same* vit_embeds `generate()` was seeded with — computed once via
`extract_feature()` up front and passed into `generate()` via its
`visual_features=` param instead of `pixel_values=`, so both calls see
identical visual grounding), then average the softmax probability of each
token inside a box's span. This was verified end-to-end against the live Modal
deployment (real boxes + plausible scores, e.g. 0.53–0.58 on a test tile) —
**the Triton copy has the identical logic but was NOT independently verified
against a running Triton server** (no access to the DGX Spark from this
session) — run a real request through it next time you're on that hardware.

If a future model revision's `generate()` changes shape again, the fastest way
to check is what caught this the first time: deploy, hit `/detect` with a
real image, and read `modal app logs locate-anything` — don't trust the model
card's documented `predict()` over the actual loaded `trust_remote_code` source.

## GPU inference platform migration (research notes, 2026-07)

Context: this project's GPU path currently targets a self-hosted Triton Inference
Server on a personal DGX Spark (`triton/`, `backend/triton_engine.py`,
`docs/adr/0005-pluggable-inference-backend-local-or-triton.md`). The owner is
evaluating moving that GPU workload (and the GPU workloads of three sibling
projects — `sport-card-fashion`, `cartoon-style-transfer`,
`image-audit-content-moderation`) to a pay-as-you-go online GPU platform, so it
doesn't depend on a specific physical machine being on and reachable.

Requirements used to evaluate: low cost, per-second billing, some free credit
every month (not just a one-time signup bonus), acceptable inference latency,
reasonable security posture. Pricing/credit terms below were checked via web
search in July 2026 and **will drift** — re-verify against each vendor's
pricing page before committing spend.

### Platform comparison

| | RunPod (Serverless) | Modal | Replicate |
|---|---|---|---|
| Billing | Per-second, GPU-tier pricing (~$0.58–$9.98/hr equivalent; H100 ≈ $4.55/hr) | Per-second (T4 $0.000164/s, A10G $0.000306/s, A100-40GB $0.001036/s, H100 $0.002778/s) | Per-second (T4 $0.000225/s, A100-80GB $0.0014/s, H100 $0.001525/s); **public models aren't billed for cold-start time** |
| Recurring free credit | None — pay-as-you-go, ~$10 one-time signup credit; funded startups can apply to a separate startup-credit program | **$30/month, every month, on the free Starter plan** | No recurring monthly credit found; one-time signup credit only |
| Deployment model | Run arbitrary Docker images (incl. a Triton server) on a GPU pod/serverless endpoint | Arbitrary Python/Docker via `modal.Image` — full custom services, not limited to a single-function schema | Opinionated: package the model as a Cog `Predictor.predict(...)`, push, get a REST API + auto UI |
| Fit for a multi-endpoint FastAPI-style service (tiling, custom request shape) | Good — it's just a container | Good — can run a FastAPI/ASGI app or a persistent `@app.cls` exactly like today's Triton client contract | Workable but the Cog `predict()` interface is single-function-shaped; multi-input custom contracts are more friction |
| Security | "Secure Cloud" tier is datacenter-grade; "Community Cloud" (cheaper) uses third-party hosts — avoid for anything sensitive | Runs on Modal's own/partner infra; SOC 2 | Runs on Replicate's own/partner infra; SOC 2 |
| Best fit here | Backup/burst option when a specific GPU tier (H100/H200/B200) is needed and predictable monthly credit doesn't matter | **Primary recommendation** | Good if/when a model should also be a public, shareable, self-serve demo API |

### Recommendation

**Use Modal as the primary platform**, shared across all four projects on one
account: the $30/month free credit resets every month (unlike RunPod's one-time
credit), billing is per-second, and it runs arbitrary custom code/Docker rather
than forcing everything through Replicate's single-`predict()` Cog schema —
important here since this backend's contract is `IMAGE` + `PROMPT` + `MODE` →
`DETECTIONS_JSON`, not a generic one-shot prediction. Keep RunPod in mind only
for a workload that specifically needs a GPU tier Modal doesn't offer, and only
on Secure Cloud (not Community Cloud) if it ever touches non-throwaway data.
Replicate is worth revisiting later specifically for whichever of the four
projects benefits from being a public shareable demo API (it has a nicer default
UI and doesn't bill cold-start for public models), but not as the primary
compute host.

### How this project already makes that migration easy

The Triton integration was already built as a **pluggable backend**, not a hard
dependency:

- `backend/engine.py` — the `InferenceEngine`-shaped contract every backend
  implements.
- `backend/la_capi.py` — local CPU engine (`LA_INFERENCE_BACKEND=local`,
  default).
- `backend/triton_engine.py` — remote GPU engine over Triton HTTP
  (`LA_INFERENCE_BACKEND=triton`, `LA_TRITON_URL=...`). Same
  `locate_buffer(image_bytes, prompt, mode) -> list[dict]` shape as the local
  engine — `app.py`, `tiling.py`, `queueing.py` don't know or care which
  backend is active.

Modal was added the same way, not as a rewrite:

- `backend/modal_engine.py` — new `InferenceEngine` implementation, same
  `locate_buffer(image_bytes, prompt, mode) -> list[dict]` shape, calling a
  Modal web endpoint over HTTPS with a shared bearer token.
- `backend/app.py` — `LA_INFERENCE_BACKEND=modal` now constructs `ModalEngine(LA_MODAL_URL, LA_MODAL_TOKEN)`.
- `modal_app/model_server.py` — the GPU-side model server: load+generate logic
  ported from `triton/model_repository/locate_anything/1/model.py` into a
  Modal `@app.cls` with `@modal.enter()` loading the checkpoint once per
  container (GPU: A10G, chosen for cost; bump if latency matters more than
  price), exposing `@modal.fastapi_endpoint(method="POST")` on `/detect`
  accepting multipart (`file`, `prompt`, `mode`) with the same
  `{"detections": [...]}` JSON shape Triton returns — `tiling.py`, `app.py`,
  `queueing.py` needed zero changes.
- `backend/test_modal_engine.py` — mocked unit tests, same pattern as
  `test_triton_engine.py`.

### Deploying modal_app

Two Modal secrets are required (create once per Modal account, not per deploy):

```sh
modal secret create la-modal-shared-token LA_MODAL_SHARED_TOKEN=<generate a random token>
modal secret create huggingface-secret HF_TOKEN=<a HF token with access to nvidia/LocateAnything-3B>
```

`nvidia/LocateAnything-3B` is a gated model — the HF account behind that token
must accept the license on the model's HF page first, or `from_pretrained()`
fails with a 403.

```sh
modal deploy modal_app/model_server.py
```

Prints a web endpoint URL (`https://<workspace>--locate-anything-locateanythingmodel-detect.modal.run`).
Point the main backend at it:

```sh
export LA_INFERENCE_BACKEND=modal
export LA_MODAL_URL=<printed endpoint URL>
export LA_MODAL_TOKEN=<the LA_MODAL_SHARED_TOKEN value>
```

**GPU type** is `LA_MODAL_GPU` (default `A10G`), read from *this shell's*
environment by `modal_app/model_server.py` at deploy time only — it's a
deploy-time `@app.cls(gpu=...)` decorator argument, resolved locally when you
run `modal deploy`, not something the running container reads. Changing it
always requires a redeploy (`LA_MODAL_GPU=L40S modal deploy
modal_app/model_server.py`); this is different from `LA_MODAL_URL`/`LA_MODAL_TOKEN`,
which the backend container reads per-request and just need a backend
restart, no Modal redeploy. Also: only `A10G` worked without a payment method
on file — Modal blocks `L40S` (and presumably other non-A10 GPUs) with
"Please add a payment method" even though the account has free credit.

Only the GPU model server (`modal_app/model_server.py`) runs on Modal today.
`backend/app.py` (tiling/orchestration) and the frontend are not deployed
anywhere yet — both run locally via `docker compose` and the backend calls
out to the Modal endpoint over HTTPS. See "Deploying to Vercel" below for the
plan to change that.

**A10G vs L40S, measured 2026-07-26** (4 real detect calls each, apple crops
from `backend/tests/fixtures/dense_apples.jpg`, one cold + three warm):

| GPU | cold start | warm avg/tile | $/sec | $/tile (warm) |
|---|---|---|---|---|
| A10G | 21.9s | 1.32s | $0.000306 | $0.000404 |
| L40S | 22.6s | 1.07s | $0.000542 | $0.000580 (+44%) |

L40S is ~19% faster per tile but costs 77% more per second, so it's ~44%
*more expensive per tile* despite being faster — the speedup doesn't offset
the price jump for this model/tile size. Cold start is dominated by weight
loading (I/O), not compute, so it's a wash between the two. **Stick with
A10G** unless a future change (e.g. much larger tiles, or a model revision
that's meaningfully more compute-bound) shifts this math — re-measure with
the same throwaway-crop method above rather than assuming it still holds.

**Cost control**: `scaledown_window=60` (60s) trades a few more cold starts for
capping idle-GPU spend, since A10G idle time is still billed at the same
GPU-second rate as active inference. Also set a spending limit in the Modal
dashboard (Settings → Billing) — that's the actual hard stop; `scaledown_window`
only reduces *how much* idle time gets billed per request burst, not a ceiling.

**If you redeploy** (code change to `modal_app/model_server.py`) while a
container from the previous deploy is still warm (`scaledown_window=60`,
i.e. up to 60s after the last request), that warm container keeps serving
the *old* code — Modal doesn't hot-swap a running container's already-loaded
module. Run `modal app stop locate-anything --yes` before redeploying if you
need the change to take effect immediately rather than waiting out the
scaledown window.

**GPU memory**: pass this endpoint properly-tiled crops (512×512, as
`backend/tiling.py` already produces for every other backend) — a full
untiled dense scene image (e.g. ~2400×1600) blows through A10G's 24GB with
`CUDA out of memory` on a single attention op. This isn't Modal-specific;
it's the same tiling requirement the local CPU and Triton engines already
have, just newly confirmed here.

**Concurrency**: no change needed — `queueing.py` already serializes
`/detect` calls through a single-worker queue regardless of which backend
answers them (see the existing note in `triton_engine.py`).

### Replicate (not yet implemented)

Same pattern if it's ever wanted: wrap the same model logic in a Cog
`predict.py` (`cog.yaml` + `Predictor.predict(image, prompt, mode)`), `cog
push`, then a `backend/replicate_engine.py` calling
`replicate.run(<model>:<version>, ...)` and translating the response into the
`[{"label", "box"}, ...]` shape `tiling.py` expects.

### Applying this to the other three projects

`sport-card-fashion`, `cartoon-style-transfer`, and
`image-audit-content-moderation` aren't in this repo, but the same move
applies: each becomes its own small Modal `App` (`modal deploy`), sharing the
one account's $30/month credit pool. Any of them that already has a working
Dockerfile can be ported with `modal.Image.from_dockerfile(...)` with no
rewrite at all; only reach for the `@app.cls` + `@modal.enter()` pattern above
if a project needs a model kept warm in memory across calls (as this one does).

## Concurrent tile processing: implemented (2026-07-26)

`LA_TILE_CONCURRENCY` (default `1`) bounds how many of one image's tiles
`backend/app.py`'s `run_detection` sends to the engine at once (`asyncio.gather`
+ `asyncio.Semaphore`), instead of the original one-at-a-time loop. Orthogonal
to `queueing.SingleWorkerQueue`, which still serializes different *requests*
against each other — this only parallelizes tiles *within* one request.

**Only safe for `triton`/`modal` backends.** `la_capi.LocateAnythingEngine`
(local CPU mode) is a single loaded model instance and isn't safe for
concurrent calls — `app.py` raises `RuntimeError` at import time if
`LA_INFERENCE_BACKEND=local` and `LA_TILE_CONCURRENCY>1`, rather than letting
that silently corrupt inference.

**Measured 2026-07-26**, full `dense_apples.jpg` (2400×1600, 24 tiles) through
the real docker-compose backend against the live Modal deployment:

| Run | Wall-clock | Compute (`inference_time_ms` sum) |
|---|---|---|
| `LA_TILE_CONCURRENCY=1` (sequential, 1 cold start) | 55.0s | 52.7s |
| `LA_TILE_CONCURRENCY=8`, **first burst** (up to 8 containers cold-start at once) | 19.1s | 131.2s |
| `LA_TILE_CONCURRENCY=8`, **containers already warm** (immediate re-run) | 9.5s | 61.3s |
| `LA_TILE_CONCURRENCY=8`, full docker-compose `/detect` call, warm | **6.65s** | 42.8s |

**The nuance that matters**: concurrency isn't "same cost, just faster" —
that's only true once enough containers are already warm. The *first*
concurrent burst triggers several simultaneous cold starts (Modal scales out
one container per concurrent request), so total billed GPU-seconds jumps
(here ~2.5x) even though wall-clock time drops. Once those containers are
warm, a repeat burst gets the wall-clock win (55s → ~7-10s) for roughly the
same total cost as running sequentially. For a bursty demo/portfolio traffic
pattern (occasional visitors, not sustained load), expect to pay the "first
burst" cost regularly, not just once — `scaledown_window=60` means containers
scale back to zero between visitors.

### Per-container concurrency: `@modal.concurrent` (2026-07-26)

The table above (`LA_TILE_CONCURRENCY=8`) was measured against a deploy with
no `@modal.concurrent` on `LocateAnythingModel` -- so each of the 8 concurrent
tile requests got its **own** container (confirmed via `modal container
list`: 8 active containers during that burst). That's simple but means N
concurrent requests always cost N cold-starts/weight-loads during a burst.

`modal_app/model_server.py` now stacks `@modal.concurrent(max_inputs=MAX_INPUTS,
target_inputs=MAX_INPUTS//2)` under `@app.cls(...)`, letting one container
accept multiple concurrent `/detect` calls (the FastAPI handler offloads the
blocking `_detect()` call via `asyncio.to_thread` so concurrent inputs on the
same container can actually interleave, not queue behind the event loop).
`MAX_INPUTS` is `LA_MODAL_MAX_INPUTS` (default `4`), read the same way as
`LA_MODAL_GPU` -- deploy-time only, resolved locally when you run `modal
deploy`, not something the running container reads:

```sh
LA_MODAL_GPU=A10G LA_MODAL_MAX_INPUTS=4 modal deploy modal_app/model_server.py
```

**Measured 2026-07-26, `LA_MODAL_MAX_INPUTS=4` vs. the no-`@modal.concurrent`
baseline, same `dense_apples.jpg`/24-tile test, `LA_TILE_CONCURRENCY=8`
throughout:**

| Config | Containers used | First burst (cold) | Warm repeat |
|---|---|---|---|
| No `@modal.concurrent` (1 container/request) | 8 | 19.1s | 9.5s |
| `@modal.concurrent(max_inputs=4)` | **4** | **32.3s (slower)** | **7.7s (comparable/better)** |

Real docker-compose `/detect` calls confirm the same pattern: 30.0s cold, 6.86s
warm, both against 4 containers (`modal container list` verified).

**The actual trade-off**: packing 2 requests onto one container doesn't make
the GPU itself do 2x the compute at once -- it's still one card, so concurrent
`generate()` calls contend for the same compute during a cold burst, making
the *first* hit slower (19.1s → 32.3s). Once containers are warm, that
contention cost is already paid and the halved container count doesn't cost
extra wall-clock time (7.7s vs 9.5s), while using half the GPU resources /
avoiding half the redundant weight loads. For an occasional-visitor demo
(mostly cold hits), the plain per-request-container approach gives a better
first impression; for sustained/bursty traffic, `@modal.concurrent` is the
better trade. **Currently deployed with `LA_MODAL_MAX_INPUTS=4`** (chosen
deliberately over the faster-first-hit no-concurrency option) with
`LA_TILE_CONCURRENCY=8` still set client-side in `.env` — tune either one
down if first-burst latency matters more than the resource savings, following
the same measure-first approach as above (`modal container list` during a
real burst, not assumption).

`target_inputs` is now set equal to `MAX_INPUTS` (was `MAX_INPUTS // 2`) --
autoscaler packs each container all the way to the hard cap before opening
another, rather than leaving headroom. Re-measured with `target_inputs=max_inputs=4`:
the same 8-concurrent-tile burst now lands on just **2** containers (down from
4), cold wall-clock ~33.2s (roughly the same as the `target=2` cold number,
~32.3s -- packing harder didn't cost extra latency here since the GPU was
already the bottleneck, not container count). Fewer containers = less
redundant weight-loading, so this is a strict improvement over the earlier
`target=2` setting for this workload -- re-verify if `LA_TILE_CONCURRENCY` or
`MAX_INPUTS` change materially, same `modal container list` method.

## Running locally via Docker

`docker compose up --build` (from repo root) builds and runs both `backend`
and `frontend` containers — see `docker-compose.yml`. Which inference backend
the container uses is controlled entirely by env vars read from `.env` (see
`.env.example` for the full list, gitignored — `.env` holds your real values,
never commit it). Needs `git submodule update --init --recursive` once (the
`backend/Dockerfile` compiles `locate-anything.cpp` from source regardless of
which `LA_INFERENCE_BACKEND` is active, since it's the same image for all three
backends — wasteful build time if you're only ever using `triton`/`modal`
mode, but keeps one Dockerfile instead of three).

Frontend: http://localhost:5173 — backend: http://localhost:8000 (`/docs` for
the OpenAPI UI). The frontend's nginx stage reverse-proxies relative `/detect`
calls to the backend container inside the compose network, so no
`VITE_API_BASE_URL` override is needed for the plain `docker-compose.yml` setup.

## Deploying to Vercel (not done yet — design notes for when it happens)

**Frontend**: straightforward fit — it's a static Vite SPA, Vercel serves it
well, generous free tier.

**Backend: do NOT put it on a plain Vercel serverless function.** Vercel
functions have a hard execution-time ceiling (~10s Hobby, ~60s Pro by default,
up to 800s on Pro with Fluid Compute). `backend/app.py`'s `/detect` tiles a
dense image into up to ~30 tiles and calls the inference engine for each
**sequentially** (single-worker queue, see `queueing.py`) — a real request can
run minutes, especially with a cold GPU container behind it. That blows past
Vercel's ceiling even in the generous configuration. Recommended instead:
deploy `backend/app.py` as its own Modal app (a CPU-only `@app.function`
wrapping the *exact same* FastAPI app via `@modal.asgi_app()` — no code
changes needed, Modal web endpoints don't have Vercel's timeout ceiling) or on
a host built for long-running requests (Fly.io, Render). Keeps everything in
one Modal account/billing surface alongside the GPU model server, which is
simpler to reason about than splitting across three platforms.

**Access lock** (keep random internet traffic from burning through Modal
credits once this has a public URL) — two layers, neither of which is real
user authentication:

1. **Vercel Deployment Protection** (Pro/Team plans) — gates the whole
   frontend behind a password before any content loads. Zero code. Set it in
   the Vercel project's settings.
2. **A shared-secret bearer token on the backend itself**, same pattern
   already built for `LA_MODAL_URL`/`LA_MODAL_TOKEN`: add an
   `LA_APP_ACCESS_TOKEN` env var, check it in `/detect` (a FastAPI dependency
   raising 401 on mismatch), have the frontend prompt for a passkey once,
   store it (`localStorage`), and send it as `Authorization: Bearer <token>`
   on every call. This is the layer that actually matters if #1 isn't
   available or someone has the direct backend URL — without it, the API is
   open to anyone who finds it regardless of whether the frontend is gated.

Be honest about the limit here: a token embedded in shipped frontend JS can be
extracted by anyone who looks (view-source / network tab) — this stops casual
traffic and scrapers, not a determined person. If the actual requirement is
"only specific people can use this" rather than "keep random load off my Modal
bill," that needs real auth (login + server-side sessions), which is a
different, larger scope than this app currently has.
