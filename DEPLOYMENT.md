# DEPLOYMENT.md

GPU platform choice, Modal deployment reference, and the plan for hosting this publicly (frontend on Vercel, backend+inference on Modal). See `CLAUDE.md` for general architecture/settings/testing.

## GPU platform choice

Evaluated for this project (self-hosted Triton on a personal DGX Spark, `triton/`) and three sibling projects (`sport-card-fashion`, `cartoon-style-transfer`, `image-audit-content-moderation`) that also want to move off hardware that has to be on and reachable. Criteria: low cost, per-second billing, recurring (not one-time) free credit, acceptable latency, reasonable security. Pricing below was checked July 2026 and **will drift** — re-verify before committing spend.

| | RunPod (Serverless) | Modal | Replicate |
|---|---|---|---|
| Billing | Per-second, GPU-tier pricing (H100 ≈ $4.55/hr) | Per-second (T4 $0.000164/s … H100 $0.002778/s) | Per-second (T4 $0.000225/s … H100 $0.001525/s); public models aren't billed for cold-start |
| Recurring free credit | None — one-time ~$10 signup credit | **$30/month, every month** | None found — one-time only |
| Deployment model | Arbitrary Docker image on a GPU pod/endpoint | Arbitrary Python/Docker via `modal.Image` | Cog `Predictor.predict(...)`, opinionated single-function schema |
| Fit for this backend's contract (`IMAGE`+`PROMPT`+`MODE` → `DETECTIONS_JSON`, not one-shot) | Good | Good | More friction — Cog's schema is single-function-shaped |
| Security | "Secure Cloud" is datacenter-grade; "Community Cloud" (cheaper) uses third-party hosts | SOC 2, own/partner infra | SOC 2, own/partner infra |

**Chose Modal**: recurring monthly credit (vs. RunPod's one-time), per-second billing, and runs arbitrary custom code rather than forcing everything through Replicate's single-`predict()` schema. RunPod stays a backup option for a GPU tier Modal doesn't offer (Secure Cloud only, for anything non-throwaway). Replicate is worth revisiting for a project that specifically wants to be a public shareable demo API (nicer default UI, free cold-start) — not as primary compute host. Sibling projects: same move, each its own small Modal `App`, sharing the account's credit pool; `modal.Image.from_dockerfile(...)` ports an existing Dockerfile with no rewrite.

Replicate isn't implemented here — if it ever is: Cog `predict.py` (`cog.yaml` + `Predictor.predict(image, prompt, mode)`), `cog push`, a `backend/replicate_engine.py` calling `replicate.run(...)` and translating the response into the shape `tiling.py` expects.

## Deploying `modal_app/model_server.py`

Two Modal secrets, created once per account:

```sh
modal secret create la-modal-shared-token LA_MODAL_SHARED_TOKEN=<generate a random token>
modal secret create huggingface-secret HF_TOKEN=<a HF token with access to nvidia/LocateAnything-3B>
```

`nvidia/LocateAnything-3B` is gated — the HF account behind that token must accept the license on the model's HF page first, or `from_pretrained()` 403s.

```sh
modal deploy modal_app/model_server.py
```

Prints a web endpoint URL. Point the main backend at it:

```sh
export LA_INFERENCE_BACKEND=modal
export LA_MODAL_URL=<printed endpoint URL>
export LA_MODAL_TOKEN=<the LA_MODAL_SHARED_TOKEN value>
```

**Deploy-time-only settings** (read from *this shell's* environment when you run `modal deploy`, not by the running container — changing either always needs a redeploy):

```sh
LA_MODAL_GPU=A10G LA_MODAL_MAX_INPUTS=4 modal deploy modal_app/model_server.py
```

- `LA_MODAL_GPU` (default `A10G`) — see the GPU comparison below before changing. Only `A10G` worked without a payment method on file; Modal blocks other GPU types ("Please add a payment method") even with free credit available.
- `LA_MODAL_MAX_INPUTS` (default `4`) — `@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=MAX_INPUTS)` on `LocateAnythingModel`, letting one container serve multiple concurrent `/detect` calls instead of Modal spinning up one container per concurrent request. See the concurrency section below for the trade-off this makes.

**Operational notes**:
- If you redeploy while a container from the previous deploy is still warm (`scaledown_window=60`), that warm container keeps serving the *old* code — Modal doesn't hot-swap a running container's already-loaded module. Run `modal app stop locate-anything --yes` first if you need the change to land immediately.
- Feed this endpoint properly-tiled crops (512×512, what `backend/tiling.py` already produces) — a full untiled dense image (e.g. 2400×1600) can exceed A10G's 24GB on a single attention op. Not Modal-specific, same requirement local/Triton already have.
- Debugging a `generate()` shape mismatch (like the box_scores bug — see `CLAUDE.md`): deploy, hit `/detect` with a real image, read `modal app logs locate-anything`. Don't trust the model card's documented `predict()` over the actual loaded `trust_remote_code` source.

## Cost control

- `scaledown_window=60` (60s) trades a few more cold starts for capping idle-GPU spend — A10G idle time bills at the same rate as active inference.
- Set a spending limit in the Modal dashboard (Settings → Billing) — that's the actual hard stop; `scaledown_window` only limits how much idle time gets billed per burst, not a ceiling.
- Check `modal container list` during a real burst before trusting an assumption about how many containers a given concurrency setting will spin up — verified numbers below, but re-verify if settings change materially.

## GPU type: A10G vs L40S (measured 2026-07-26)

4 real `/detect` calls each, apple crops from `backend/tests/fixtures/dense_apples.jpg`, one cold + three warm:

| GPU | cold start | warm avg/tile | $/sec | $/tile (warm) |
|---|---|---|---|---|
| A10G | 21.9s | 1.32s | $0.000306 | $0.000404 |
| L40S | 22.6s | 1.07s | $0.000542 | $0.000580 (+44%) |

L40S is ~19% faster per tile but costs 77% more per second — net ~44% more expensive per tile despite being faster. Cold start is I/O-bound (weight loading), not compute, so it's a wash there. **Stick with A10G** unless tile size grows substantially or a future model revision is meaningfully more compute-bound; re-measure with the same method rather than assuming this still holds.

## Concurrency

Two independent axes (see `CLAUDE.md`'s settings table): `LA_TILE_CONCURRENCY` (tiles within one request) and `LA_REQUEST_CONCURRENCY` (whole requests, e.g. multi-prompt). Both currently `8` and `5` in `.env` — worst case `5 × 8 = 40` concurrent tile calls to Modal, **not yet load-tested at that worst case** (only verified on single-tile images, deliberately, to avoid an expensive surprise). Before relying on a real dense-scene multi-prompt run, watch the Modal dashboard once or dial one of the two down first.

**`LA_TILE_CONCURRENCY`, measured 2026-07-26** (`dense_apples.jpg`, 2400×1600, 24 tiles, full docker-compose backend against live Modal):

| Run | Wall-clock |
|---|---|
| `LA_TILE_CONCURRENCY=1` (sequential, 1 cold start) | 55.0s |
| `LA_TILE_CONCURRENCY=8`, first burst (up to 8 containers cold-start at once) | 19.1s |
| `LA_TILE_CONCURRENCY=8`, containers already warm | 9.5–6.65s |

Concurrency isn't "same cost, just faster" until containers are already warm — the first concurrent burst triggers several simultaneous cold starts (~2.5x the billed GPU-seconds of running sequentially), then a repeat burst gets the wall-clock win for roughly the same cost as sequential. For occasional-visitor traffic (not sustained load), expect to pay the "first burst" cost regularly — `scaledown_window=60` scales containers back to zero between visitors.

**`@modal.concurrent`/`LA_MODAL_MAX_INPUTS`, measured 2026-07-26** (same test, `LA_TILE_CONCURRENCY=8` throughout):

| Config | Containers used | First burst (cold) | Warm repeat |
|---|---|---|---|
| No `@modal.concurrent` (1 container/request) | 8 | 19.1s | 9.5s |
| `max_inputs=4, target_inputs=2` | 4 | 32.3s | 7.7s |
| `max_inputs=4, target_inputs=4` (current) | **2** | ~33.2s | comparable |

Packing more requests per container doesn't parallelize GPU compute (one card either way) — it makes the *first* cold hit slower (concurrent `generate()` calls contend for the same compute), but once warm, fewer containers cost the same wall-clock time while using less GPU resource / avoiding redundant weight loads. For a mostly-cold occasional-visitor demo, no-concurrency gives a better first impression; for sustained/bursty traffic, `@modal.concurrent` is the better trade. **Currently deployed at `target_inputs=max_inputs=4`** (packs each container fully before opening another).

**`LA_REQUEST_CONCURRENCY`, verified 2026-07-26** (single warm 512×512 tile, 3 different prompts): 2.7s total wall-clock vs. each individual call's own 1.6–2.7s — genuinely overlapping, not summing. Concurrency cap itself confirmed: 6 concurrent requests against `LA_REQUEST_CONCURRENCY=5` showed 5 at `queue_wait_ms: 0` and the 6th at `queue_wait_ms: 1900`.

## Deploying `modal_app/backend_server.py` (backend/app.py, on Modal)

**Done** — `backend/app.py` (the tiling/merge orchestrator, not the GPU model) now also runs on Modal, wrapped by `modal_app/backend_server.py` as a CPU-only `@modal.asgi_app()`. No changes needed to `backend/app.py` itself; the wrapper adds `backend/`'s source (`add_local_dir`, minus `tests/`) into a plain `debian_slim` image built from `backend/requirements.txt`'s packages and reads the same env vars `backend/app.py` already expects, via a Modal secret instead of `.env`.

One secret, created once per account:

```sh
modal secret create la-backend-config \
    LA_INFERENCE_BACKEND=modal \
    LA_MODAL_URL=<modal_app/model_server.py's printed endpoint URL> \
    LA_MODAL_TOKEN=<the la-modal-shared-token secret's value> \
    LA_TILE_CONCURRENCY=8 \
    LA_REQUEST_CONCURRENCY=5
```

```sh
modal deploy modal_app/backend_server.py
```

Prints a web endpoint URL ending in `/detect` — point the frontend's `VITE_API_BASE_URL` at its base (everything before `/detect`), same shape as the local `docker compose` setup.

Deliberately **doesn't** support `LA_INFERENCE_BACKEND=local` — the compiled `.so` + GGUF weights aren't part of this image, and running the CPU engine here would defeat the point of a GPU-hosted model. Verified live 2026-07-28: a real `/detect` call against a 512×512 crop returned two boxes with plausible scores (0.55, 0.60), `52.8s` execution — a cold hit on both this container *and* the GPU model server behind it (each has its own independent cold start).

## Deploying to Vercel (frontend only — planned, not done)

Remaining piece of the restructure: **frontend → Vercel** (backend + inference are both on Modal now, per above).

Vercel serves the static Vite SPA well, no special handling needed — a plain `vercel.json` pointing at `frontend/`'s build output, with `VITE_API_BASE_URL` set to the deployed backend's Modal URL as a Vercel env var (baked in at Vercel's build time, same as it is for the `docker compose` frontend build today).

**Access lock** once this has a public URL (keep random traffic from burning Modal credits) — two layers, neither is real user authentication:

1. **Vercel Deployment Protection** (Pro/Team) — password-gates the whole frontend, zero code.
2. **Shared-secret bearer token on the backend**, same pattern as `LA_MODAL_URL`/`LA_MODAL_TOKEN`: an `LA_APP_ACCESS_TOKEN` env var, checked in `/detect`, with the frontend prompting once and storing it (`localStorage`) as an `Authorization: Bearer` header. This is the layer that matters if #1 isn't available or someone has the direct backend URL.

A token embedded in shipped frontend JS can be extracted by anyone who looks (view-source/network tab) — this stops casual traffic and scrapers, not a determined person. If the actual requirement is "only specific people can use this," that needs real auth (login + server-side sessions), a different and larger scope than this app currently has.
