"""Triton Python backend wrapping nvidia/LocateAnything-3B (GPU, PyTorch).

Loads the model once in `initialize()`, then for each request: decode the
image, ask the model to locate one class ("PROMPT"), parse the <box> tags
out of its raw text output, convert normalized [0,1000] coordinates to the
tile's own pixel space, and return the same {"label", "box"} shape the
project's local CPU engine (locate-anything.cpp) already returns -- see
backend/triton_engine.py in the main repo for the client side of this
contract, and backend/tiling.py for how these tile-local pixel coordinates
get translated into whole-image coordinates.

VERIFIED ON HARDWARE (DGX Spark, GB10/Blackwell): the prompt template and the
<box><x1><y1><x2><y2></box> parsing match the model's real output and need no
adjustment. What DID need fixing vs. the original draft (see docs/adr/0005):
the model's generate() is a custom MTP/AR loop, NOT HF GenerationMixin.generate
-- it requires use_cache=True and tokenizer=, takes explicit pixel_values/
input_ids/... args, and returns the decoded answer STRING (with <box> tags)
directly, not a token-id tensor. The call below mirrors the model card's
predict(). If a future model revision returns zero boxes, the raw generated
text is logged to make that debugging pass tractable.

CORRECTED 2026-07-26 (found + fixed first in modal_app/model_server.py, then
ported here): this model revision's generate() does NOT return
`(answer, box_scores)` -- with verbose=False (used here) it returns just the
decoded answer STRING. verbose=True returns a 3-tuple
(answer, sampling_history, out_info), still with no per-box confidence
anywhere. The `answer, box_scores = self.model.generate(...)` line that used
to be here raised `ValueError: too many values to unpack` on every real
request -- confirmed by inspecting the actual remote-code source
(modeling_locateanything.py) on HF, not just by inference. Per-box confidence
is instead computed below via a teacher-forcing pass over the already-generated
tokens (_score_boxes) -- see that function's docstring for why.
"""

import io
import json
import re
from typing import Any

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# pb_utils.InferenceRequest/InferenceResponse have no type stubs available
# outside Triton's own runtime (this module only imports/runs inside a
# tritonserver container) -- Any is the genuinely-unavoidable case
# docs/CODING_STANDARDS.md's Types rule already allows for.
InferenceRequest = Any
InferenceResponse = Any

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_TAG_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        # NOTE (known risk, see module docstring): the model card's own
        # LocateAnythingWorker reference impl loads via AutoModel, not
        # AutoModelForCausalLM -- trusting that since it's the more specific
        # source (vs. a generic third-party usage example that used
        # AutoModelForCausalLM instead). If `.generate()` below raises
        # AttributeError, this is the first thing to try swapping.
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def execute(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        return [self._handle_one(request) for request in requests]

    def _handle_one(self, request: InferenceRequest) -> InferenceResponse:
        image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()[0]
        prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()[0].decode()
        mode = pb_utils.get_input_tensor_by_name(request, "MODE").as_numpy()[0].decode()

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

        # The model card's predict() uses py_apply_chat_template (the pure-Python
        # path); fall back to the standard apply_chat_template if this processor
        # build doesn't expose it.
        apply = getattr(
            self.processor, "py_apply_chat_template", self.processor.apply_chat_template
        )
        text = apply(messages, tokenize=False, add_generation_prompt=True)
        # Matches the model card's documented input-prep exactly (rather
        # than passing `images=[image]` directly) -- process_vision_info
        # may do model-specific preprocessing (resizing, tiling, etc.) that
        # this wrapper shouldn't try to replicate by hand.
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(
            self.device
        )
        image_grid_hws = inputs.get("image_grid_hws", None)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(self.device, dtype=torch.int32)

        # Compute vision features once, up front, and hand them to generate()
        # via its visual_features= param (instead of pixel_values=, which
        # would make generate() redo this same extract_feature() call
        # internally) -- the *same* embeddings are then reused by
        # _score_boxes() below for a teacher-forcing confidence pass, so both
        # calls see identical visual grounding. generate() still mutates its
        # own local copy of vit_embeds via cat+mlp1 when image_grid_hws is
        # set (see the model's generate() source); replicate that same
        # transform here to get the final embeds _score_boxes() needs.
        pixel_values = inputs["pixel_values"].to(self.model.language_model.dtype)
        with torch.no_grad():
            raw_vit = self.model.extract_feature(pixel_values, image_grid_hws)
            final_vit = raw_vit
            if image_grid_hws is not None:
                final_vit = torch.cat(raw_vit, dim=0)
                final_vit = self.model.mlp1(final_vit)

        # The model's custom generate() (in its trust_remote_code modeling file)
        # is NOT HF GenerationMixin.generate: it takes explicit
        # pixel_values/input_ids/... args, REQUIRES use_cache=True + tokenizer=,
        # and returns the decoded answer STRING directly (see module docstring
        # for the box_scores correction). Args mirror the model card's
        # documented predict().
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
            # Nothing parsed -- log the raw text so a real run's actual
            # output format (if different from the model card's documented
            # example) is visible for debugging, not just "zero results".
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
        detections_json = json.dumps({"detections": detections})

        # The model's hybrid MTP/AR loop allocates large per-step tensors (KV
        # cache out to max_new_tokens, MTP sampling buffers); once the answer is
        # extracted they're unreferenced. Release the caching allocator's hoarded
        # blocks now so the GPU/unified-memory footprint drops back to the ~6GB
        # resident weights between requests -- otherwise repeated /detect calls
        # accumulate cached blocks (observed growing to ~64GB on the GB10's
        # 119GB unified memory), starving co-located workloads.
        torch.cuda.empty_cache()

        output_tensor = pb_utils.Tensor(
            "DETECTIONS_JSON", np.array([detections_json.encode()], dtype=object)
        )
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _score_boxes(
        self, answer: str, matches: list, prompt_input_ids, vit_embeds
    ) -> list[float | None]:
        """Per-box confidence the model's generate() doesn't expose: mean
        softmax probability, under one teacher-forced forward pass over the
        already-generated tokens, of each token inside a given <box>...</box>
        span (all special coordinate/tag tokens -- verified these round-trip
        exactly through tokenizer.encode/decode, so char offsets line up with
        the regex match spans). Reuses the exact vit_embeds generate() was
        seeded with (see _handle_one) so this scoring pass sees the same
        visual grounding, not a fresh recomputation. Verified against a live
        deployment (see modal_app/model_server.py, the same code ported here)."""
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
