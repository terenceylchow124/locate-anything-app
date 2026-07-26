# TODO.md

Outstanding work, agent-facing. See `CLAUDE.md` (architecture/settings) and `DEPLOYMENT.md` (GPU platform/Modal/Vercel) for context on most of these.

## Verification gaps

- **Triton's `_score_boxes()` fix is unverified against real hardware.** `triton/model_repository/locate_anything/1/model.py` has the same box-confidence fix as `modal_app/model_server.py` (ported by hand, identical logic), but it was only verified end-to-end on the live Modal deployment — no access to the DGX Spark from the session that made this change. Run a real `/detect` request through the Triton path next time it's reachable and confirm scores come back non-null and plausible.
- **`LA_REQUEST_CONCURRENCY` × `LA_TILE_CONCURRENCY` worst case (currently 5 × 8 = 40) hasn't been load-tested.** Only verified on single-tile images deliberately, to avoid an expensive surprise. Before relying on a real dense-scene multi-prompt run in production, watch the Modal dashboard during one deliberate test, or dial one of the two settings down first.

## Deployment restructure (see `DEPLOYMENT.md`)

- [ ] Deploy `backend/app.py` as its own Modal app (`@modal.asgi_app()` wrapping the existing FastAPI app, no code changes expected) or an equivalent long-request-friendly host (Fly.io/Render) — currently it only runs locally via `docker compose`.
- [ ] Deploy the frontend to Vercel.
- [ ] Add the access lock before either is publicly reachable: Vercel Deployment Protection (Pro/Team) + an `LA_APP_ACCESS_TOKEN` shared-secret bearer check on the backend (frontend prompts once, stores in `localStorage`). Neither exists yet.
- [ ] Re-run the concurrency/GPU benchmarks in `DEPLOYMENT.md` once the backend is off `docker compose` and actually reachable over the internet — current numbers are all from a local backend calling a public Modal endpoint, not a fully deployed stack.

## Not implemented (optional/future)

- Replicate backend (`backend/replicate_engine.py`) — see `DEPLOYMENT.md`'s sketch of what it'd take. Only worth doing if a project specifically wants a public shareable demo API.
- Applying the same Modal migration to the three sibling projects (`sport-card-fashion`, `cartoon-style-transfer`, `image-audit-content-moderation`) — same pattern, not started.
