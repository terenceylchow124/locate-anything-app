# Triton Inference Server deployment (DGX Spark)

Serves the original NVIDIA `LocateAnything-3B` PyTorch checkpoint on the GPU, as an alternative to the main project's local CPU engine (`locate-anything.cpp`). See `docs/adr/0005-pluggable-inference-backend-local-or-triton.md` in the main repo for why this split exists, and `backend/triton_engine.py` for the client side of the contract this server implements.

**License reminder**: `nvidia/LocateAnything-3B` is non-commercial research use only, same as the rest of this project (see the main repo's README).

**Verified on hardware** (DGX Spark, GB10/Blackwell): the prompt template and `<box>` tag parsing match the model's real output as-is. What the first live run did require was fixing `model.py`'s `generate()` call — the model ships a *custom* MTP/AR `generate()` (not HF's `GenerationMixin.generate`): it needs `use_cache=True` + `tokenizer=`, takes explicit `pixel_values`/`input_ids`/... args, and returns the decoded answer string directly (not a token-id tensor). The `Dockerfile` also had to gain the model's runtime deps (`peft`, `torchvision`, `opencv-python-headless`, `lmdb`, `requests`; `transformers<5`; `numpy<2`; and a `decord` import stub, since `decord` has no aarch64 wheel). If a future model revision returns zero boxes, `model.py` still logs the raw generated text to make that debugging pass tractable.

## 1. Build the image

```sh
cd triton
docker build -t locate-anything-triton .
```

Check the `TRITON_TAG` build arg in `Dockerfile` first — verify it's still a valid, ARM64/Blackwell-compatible tag on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) before building; override if needed:

```sh
docker build -t locate-anything-triton --build-arg TRITON_TAG=<newer-tag> .
```

## 2. (Optional) Pre-download the model weights

`model.py`'s `from_pretrained()` calls will download the weights automatically the first time the model loads inside the container — so this step isn't strictly required. Worth doing anyway to avoid the first `/detect` request also paying multi-GB download time, and to pin a specific revision if you want reproducibility:

```sh
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nvidia/LocateAnything-3B')
"
```

This populates `~/.cache/huggingface` (the default HF cache location), which step 3 below mounts into the container.

## 3. Run the server

```sh
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  locate-anything-triton
```

Watch the logs for `locate_anything` loading successfully (model load takes a while the first time — full bf16 weights, no quantization, unlike the CPU engine's q8_0 GGUF).

## 4. Verify with a raw test call

From the DGX Spark itself (or anywhere that can reach it):

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

with open("path/to/a/test/image.png", "rb") as f:
    image_bytes = f.read()

image_input = httpclient.InferInput("IMAGE", [1], "BYTES")
image_input.set_data_from_numpy(np.array([image_bytes], dtype=object))
prompt_input = httpclient.InferInput("PROMPT", [1], "BYTES")
prompt_input.set_data_from_numpy(np.array([b"person"], dtype=object))
mode_input = httpclient.InferInput("MODE", [1], "BYTES")
mode_input.set_data_from_numpy(np.array([b"hybrid"], dtype=object))

output = httpclient.InferRequestedOutput("DETECTIONS_JSON")
result = client.infer("locate_anything", inputs=[image_input, prompt_input, mode_input], outputs=[output])
print(result.as_numpy("DETECTIONS_JSON")[0].decode())
```

If this returns `{"detections": []}` on an image that should have matches, check the container logs for the `zero boxes parsed from raw output: ...` line `model.py` prints — that's the actual raw text the model generated, and the fastest way to see whether the prompt template or `<box>` tag format needs adjusting for real.

## 5. Point the main backend at it

From the laptop (or wherever the main `backend/` service runs), reachable over the network to the DGX Spark's address:

```sh
export LA_INFERENCE_BACKEND=triton
export LA_TRITON_URL=<dgx-spark-address>:8000
uvicorn app:app --reload --port 8000   # from backend/, as usual
```

Or via Docker Compose (`docker-compose.yml` in the main repo already reads these from the environment):

```sh
LA_INFERENCE_BACKEND=triton LA_TRITON_URL=<dgx-spark-address>:8000 docker compose up --build
```

**Networking**: the DGX Spark's Triton port (8000) needs to be reachable from wherever `backend/` runs — same LAN, a Tailscale/VPN connection, or an SSH tunnel (`ssh -L 8000:localhost:8000 user@dgx-spark`, then `LA_TRITON_URL=localhost:8000`) all work.
