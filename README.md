# LocateAnything-3B — Interactive Counting Demo

A portfolio project built around [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B), an open-vocabulary object detection / visual-grounding VLM. The goal is an interactive demo that proves the model's core differentiator over YOLO-style detectors: swap the text prompt, swap the detection target — no retraining, no per-class model.

See `docs/locateanything-demo-scenarios.md` for the full scenario/positioning notes and `docs/spec-locateanything-demo.md` for the current spec (problem statement, user stories, implementation/testing decisions).

## Status

Early development. Tracked on Trello (board: LocateAnything-3B — see `docs/agents/issue-tracker.md`), currently working ticket #01 (repo foundation + inference engine verified standalone).

## Architecture

- **Inference engine:** [`mudler/locate-anything.cpp`](https://github.com/mudler/locate-anything.cpp) (vendored as a git submodule at `locate-anything.cpp/`) — a ggml/C++ port of LocateAnything-3B that runs real, non-mocked inference **on CPU**, validated box-identical to the official PyTorch model at `q8_0` quantization. No GPU is required or used anywhere in this project.
- **Backend:** a thin HTTP service (in progress) wrapping the engine behind a single `POST /detect` endpoint.
- **Frontend:** an in-house interactive canvas (in progress) — upload/pick an image, type or click a prompt, see boxes + count.

We deliberately dropped an earlier plan to reuse a third-party prebuilt Docker app (`gammahazard/locate-anything`) — it's CUDA-only with no CPU path and offered no way to customize sample scenes or license text. Both backend and frontend here are ours.

## Development environment

A dedicated conda environment (`locateanything`, Python 3.11, via **conda-forge** — not Anaconda's default channels, which require ToS acceptance) holds everything needed to build the engine and run its Python-side scripts (model download/conversion). No CUDA build of PyTorch is installed; this project is CPU-only end to end.

```sh
conda create -n locateanything -c conda-forge python=3.11 --override-channels
conda activate locateanything
conda install -c conda-forge cmake make cxx-compiler --override-channels
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
pip install -r scripts/requirements.txt
```

## Build the inference engine

```sh
conda activate locateanything
git submodule update --init --recursive   # if you cloned this repo fresh
cd locate-anything.cpp
cmake -B build -DLA_BUILD_TESTS=ON -DLA_BUILD_CLI=ON
cmake --build build -j$(nproc)
```

## Get the model

Prebuilt GGUF weights (no local conversion needed):

```sh
conda activate locateanything
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='mudler/locate-anything.cpp-gguf', filename='locate-anything-q8_0.gguf', local_dir='locate-anything.cpp/models')
"
```

`q8_0` (~6.3 GB) is recommended — near-lossless, detections identical to the official f32 model.

## Run a detection

```sh
locate-anything.cpp/build/examples/cli/locate-anything-cli detect \
  --model locate-anything.cpp/models/locate-anything-q8_0.gguf \
  --input <your-image.jpg> \
  --prompt "person" \
  --annotated out.png
```

## License

- **This project's own code:** to be decided/added.
- **`locate-anything.cpp`** (the inference engine port): MIT.
- **Model weights** — NVIDIA `LocateAnything-3B` and its base `Qwen2.5-3B`: both **non-commercial research licenses**. This project is scoped as portfolio/research use only; see `docs/locateanything-demo-scenarios.md` for the licensing notes that shaped this scope, and cite `locate-anything.cpp` (Di Giacinto & Palethorpe, LocalAI project) alongside the NVIDIA model per its README's citation request.

## Project management

- Tickets: Trello (see `docs/agents/issue-tracker.md` for board/list details)
- Agent-facing conventions (issue tracker, domain docs): `AGENTS.md`
