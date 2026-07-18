"""Triton Python backend wrapping nvidia/LocateAnything-3B (GPU, PyTorch).

Loads the model once in `initialize()`, then for each request: decode the
image, ask the model to locate one class ("PROMPT"), parse the <box> tags
out of its raw text output, convert normalized [0,1000] coordinates to the
tile's own pixel space, and return the same {"label", "box"} shape the
project's local CPU engine (locate-anything.cpp) already returns -- see
backend/triton_engine.py in the main repo for the client side of this
contract, and backend/tiling.py for how these tile-local pixel coordinates
get translated into whole-image coordinates.

KNOWN RISK (see docs/adr/0005 in the main repo): the exact prompt template,
<box> tag format, and `generation_mode` parameter are based on the model
card's documented examples (https://huggingface.co/nvidia/LocateAnything-3B),
not a live test run against real weights -- this has not been verified on
real hardware. If zero boxes parse, the raw generated text is logged to
make that debugging pass tractable.
"""

import io
import json
import re

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_TAG_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[list[float]]:
    """Regex from the model card's LocateAnythingWorker.parse_boxes: pulls
    <box><x1><y1><x2><y2></box> tags (coords normalized to [0,1000]) and
    scales them to pixel space."""
    boxes = []
    for m in BOX_TAG_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append(
            [
                x1 / 1000 * image_width,
                y1 / 1000 * image_height,
                x2 / 1000 * image_width,
                y2 / 1000 * image_height,
            ]
        )
    return boxes


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            responses.append(self._handle_one(request))
        return responses

    def _handle_one(self, request):
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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                generation_mode=mode,
            )

        # Only the newly generated tokens, not the echoed prompt.
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        boxes = parse_boxes(answer, width, height)
        if not boxes:
            # Nothing parsed -- log the raw text so a real run's actual
            # output format (if different from the model card's documented
            # example) is visible for debugging, not just "zero results".
            print(f"[locate_anything] zero boxes parsed from raw output: {answer!r}")

        detections = [{"label": prompt, "box": box} for box in boxes]
        detections_json = json.dumps({"detections": detections})

        output_tensor = pb_utils.Tensor(
            "DETECTIONS_JSON", np.array([detections_json.encode()], dtype=object)
        )
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])
