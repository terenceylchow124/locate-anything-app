"""Modal deployment of nvidia/LocateAnything-3B (GPU, PyTorch) -- an
alternative to both the local CPU engine (locate-anything.cpp) and the
self-hosted Triton server (triton/), for when GPU inference should run on
Modal's pay-per-second infra instead of a specific physical machine.

Model-loading and generation logic is ported 1:1 from
triton/model_repository/locate_anything/1/model.py (same prompt template,
same <box> tag parsing, same per-box confidence handling) -- see that file's
docstring for the history of quirks in the model's custom generate(). Kept
in sync deliberately rather than imported, since the two run in unrelated
container images (Triton's vs. this one) and diverging is easier to notice
than a cross-image import would be.

Wire contract matches TritonEngine exactly: POST multipart (file, prompt,
mode) in, {"detections": [{"label", "box", "score"}, ...]} JSON out -- so
backend/modal_engine.py is a drop-in InferenceEngine, same as TritonEngine.

Deploy:
    modal deploy modal_app/model_server.py

GPU type and per-container concurrency are read from *this shell's*
environment at deploy time (LA_MODAL_GPU, default "A10G"; LA_MODAL_MAX_INPUTS,
default "4") -- not Modal Secrets, since @app.cls's gpu= and @modal.concurrent's
max_inputs= are deploy-time decorator arguments resolved locally, not
something the remote container reads:
    LA_MODAL_GPU=L40S LA_MODAL_MAX_INPUTS=2 modal deploy modal_app/model_server.py

Then point the main backend at the printed web endpoint URL:
    export LA_INFERENCE_BACKEND=modal
    export LA_MODAL_URL=<printed endpoint URL, ends in /detect>
    export LA_MODAL_TOKEN=<value of the la-modal-shared-token Modal secret>

Requires two Modal secrets (see README's "Deploying modal_app" section):
    la-modal-shared-token   -- LA_MODAL_SHARED_TOKEN, a shared bearer token
                                this endpoint checks on every request (it's
                                a public HTTPS URL otherwise).
    huggingface-secret      -- HF_TOKEN, needed because nvidia/LocateAnything-3B
                                is a gated model on Hugging Face.
"""

from __future__ import annotations

import asyncio
import io
import os
import re

import modal
from fastapi import File, Form, Header, HTTPException, UploadFile

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_TAG_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
GPU_TYPE = os.environ.get("LA_MODAL_GPU", "A10G")
# Max concurrent /detect inputs one container will accept before Modal spins
# up another container -- also a deploy-time-only value, same as GPU_TYPE
# (read locally by `modal deploy`, not by the running container). See
# DEPLOYMENT.md's concurrency section for the cold-start-vs-warm-throughput
# trade-off this trades off.
MAX_INPUTS = int(os.environ.get("LA_MODAL_MAX_INPUTS", "4"))
# Equal to MAX_INPUTS -- autoscaler packs each container up to the hard cap
# before opening a new one, rather than leaving headroom for burst (see
# DEPLOYMENT.md: measured this trades a bit more per-container GPU contention
# for fewer containers/less redundant weight-loading).
TARGET_INPUTS = MAX_INPUTS

app = modal.App("locate-anything")

# Mirrors triton/Dockerfile's pip_install (see that file for why each pin
# exists: transformers<5 because the model's trust_remote_code files predate
# transformers 5's API rework, numpy<2 for the same processor/model imports,
# decord stub because the model's processing file unconditionally imports it
# for a video path this project never exercises).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.57.1,<5",
        "accelerate>=0.30",
        "pillow>=10.0",
        "torchvision",
        "opencv-python-headless",
        "lmdb",
        "requests",
        "peft",
        "numpy<2",
        "fastapi[standard]>=0.110",
        "python-multipart>=0.0.9",
    )
    .run_commands(
        "python3 -c \""
        "import site, os; "
        "d = os.path.join(site.getsitepackages()[0], 'decord'); "
        "os.makedirs(d, exist_ok=True); "
        "open(os.path.join(d, '__init__.py'), 'w').write('# decord stub: unused video path')"
        "\""
    )
)

# Weights (~6GB, bf16) persist across container restarts here instead of
# re-downloading from Hugging Face on every cold start.
hf_cache = modal.Volume.from_name("la-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[
        modal.Secret.from_name("la-modal-shared-token"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    # Keep the container (and loaded weights) warm for 60s after the last
    # request before scaling to zero -- still covers back-to-back tile calls
    # for one image (each takes ~2s once warm), while capping idle-GPU spend
    # at ~$0.02/request-burst instead of ~$0.09 (see DEPLOYMENT.md's cost
    # control notes).
    scaledown_window=60,
)
# Lets one container handle multiple concurrent /detect calls instead of
# Modal spinning up a separate container (and paying its own cold start +
# duplicate ~6GB weight load) per concurrent request -- see DEPLOYMENT.md's
# concurrency section for why LA_TILE_CONCURRENCY>1 on the backend side was
# previously turning into N separate containers. GPU compute itself still
# serializes on the one card either way; this trades that same GPU-bound
# serialization for skipping the redundant load/cold-start cost. Conservative
# max_inputs -- each concurrent /detect holds its own KV cache + activations
# (see the model's generate() max_new_tokens=8192) on top of the ~6GB
# resident weights; raise only after checking GPU memory headroom on the
# Modal dashboard during a real burst.
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
class LocateAnythingModel:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        # See triton model.py's NOTE: the model card's reference impl uses
        # AutoModel, not AutoModelForCausalLM.
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _detect(self, image_bytes: bytes, prompt: str, mode: str) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        question = f"Locate all the instances that matches the following description: {prompt}."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        apply = getattr(
            self.processor, "py_apply_chat_template", self.processor.apply_chat_template
        )
        text = apply(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        image_grid_hws = inputs.get("image_grid_hws", None)
        # Mirrors generate()'s own numpy->tensor conversion (its first few
        # lines) -- the processor hands back image_grid_hws as a numpy array.
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(self.device, dtype=torch.int32)

        # Compute vision features once, up front, and hand them to generate()
        # via its visual_features= param (instead of pixel_values=, which
        # would make generate() redo this same extract_feature() call
        # internally) -- the *same* embeddings are then reused by
        # _score_boxes() below for a teacher-forcing confidence pass, so both
        # calls see identical visual grounding. generate() still mutates its
        # own local copy of vit_embeds via cat+mlp1 when image_grid_hws is
        # set (see the model's generate() source); we replicate that same
        # transform here to get the final embeds _score_boxes() needs.
        # Mirrors generate()'s own first line (pixel_values.to(language_model.dtype))
        # -- extract_feature()'s conv stem is bf16, pixel_values comes out of the
        # processor as float32.
        pixel_values = inputs["pixel_values"].to(self.model.language_model.dtype)
        with torch.no_grad():
            raw_vit = self.model.extract_feature(pixel_values, image_grid_hws)
            final_vit = raw_vit
            if image_grid_hws is not None:
                final_vit = torch.cat(raw_vit, dim=0)
                final_vit = self.model.mlp1(final_vit)

        # NOTE: unlike triton/model_repository/locate_anything/1/model.py's
        # assumption, this model revision's generate() (see
        # modeling_locateanything.py) returns just the decoded answer STRING
        # when verbose=False -- not a (answer, box_scores) tuple. There is no
        # per-box confidence anywhere in this revision's generate() (verbose=True
        # instead returns (answer, sampling_history, out_info), still no
        # box-level scores). Confirmed by inspecting the actual remote-code
        # source on 2026-07-26; the Triton model.py's box_scores unpacking is
        # stale against the currently-downloaded model revision and will raise
        # ValueError there too. Per-box confidence is instead computed
        # separately below (_score_boxes), via a teacher-forcing pass over
        # the already-generated tokens.
        with torch.no_grad():
            answer = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                visual_features=raw_vit,
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=8192,
                use_cache=True,
                generation_mode=mode,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )

        matches = list(BOX_TAG_RE.finditer(answer))
        if not matches:
            print(f"[locate_anything] zero boxes parsed from raw output: {answer!r}")

        scores = self._score_boxes(answer, matches, inputs["input_ids"], final_vit)
        detections = [
            {
                "label": prompt,
                "box": [
                    int(m.group(1)) / 1000 * width,
                    int(m.group(2)) / 1000 * height,
                    int(m.group(3)) / 1000 * width,
                    int(m.group(4)) / 1000 * height,
                ],
                "score": score,
            }
            for m, score in zip(matches, scores)
        ]

        # Same reasoning as the Triton model.py: release cached allocator
        # blocks between requests rather than letting them accumulate.
        torch.cuda.empty_cache()
        return {"detections": detections}

    def _score_boxes(
        self, answer: str, matches: list, prompt_input_ids, vit_embeds
    ) -> list[float | None]:
        """Per-box confidence the model's generate() doesn't expose: mean
        softmax probability, under one teacher-forced forward pass over the
        already-generated tokens, of each token inside a given <box>...</box>
        span (all special coordinate/tag tokens -- see the tokenizer
        round-trip check this was validated against). Reuses the exact
        vit_embeds generate() was seeded with (see _detect) so this scoring
        pass sees the same visual grounding, not a fresh recomputation."""
        import torch

        if not matches:
            return []

        enc = self.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        continuation_ids = torch.tensor([enc["input_ids"]], device=self.device)
        offsets = enc["offset_mapping"]

        full_ids = torch.cat([prompt_input_ids, continuation_ids], dim=1)
        prompt_len = prompt_input_ids.shape[1]
        continuation_len = continuation_ids.shape[1]
        position_ids = torch.arange(full_ids.size(1), device=self.device).unsqueeze(0)

        prepare_inputs = self.model.language_model.prepare_inputs_for_generation(
            full_ids,
            None,
            None,
            inputs_embeds=None,
            use_cache=False,
            position_ids=position_ids,
        )
        prepare_inputs["visual_features"] = vit_embeds
        prepare_inputs["image_token_index"] = self.model.config.image_token_index

        with torch.no_grad():
            outputs = self.model.language_model(**prepare_inputs)

        # Position p's logits predict the token at position p+1 -- shift by
        # one to align with continuation_ids.
        pred_logits = outputs.logits[0, prompt_len - 1 : prompt_len - 1 + continuation_len, :]
        token_confidences = (
            torch.softmax(pred_logits, dim=-1)
            .gather(-1, continuation_ids[0].unsqueeze(-1))
            .squeeze(-1)
            .tolist()
        )

        scores: list[float | None] = []
        for m in matches:
            start, end = m.span(0)
            idxs = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
            scores.append(
                round(sum(token_confidences[i] for i in idxs) / len(idxs), 4) if idxs else None
            )
        return scores

    @modal.fastapi_endpoint(method="POST")
    async def detect(
        self,
        file: UploadFile = File(...),
        prompt: str = Form(...),
        mode: str = Form("hybrid"),
        authorization: str = Header(default=""),
    ) -> dict:
        import os

        expected = os.environ["LA_MODAL_SHARED_TOKEN"]
        token = authorization.removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

        image_bytes = await file.read()
        # _detect() is a blocking call (torch, no internal awaits) -- run it
        # in a thread so this async handler doesn't hog the container's
        # single event loop for its whole duration, which would otherwise
        # serialize concurrent inputs even with @modal.concurrent applied
        # (see that decorator's note: async functions share one thread).
        return await asyncio.to_thread(self._detect, image_bytes, prompt, mode)
