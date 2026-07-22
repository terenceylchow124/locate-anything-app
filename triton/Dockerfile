# Triton Inference Server for nvidia/LocateAnything-3B, built for the DGX
# Spark (ARM64 Grace CPU + Blackwell GB10 GPU). See README.md in this
# directory for the full build/deploy/verify steps.
#
# IMPORTANT: verify the tag below is still the latest ARM64/Blackwell-
# compatible Triton release before building -- NGC tags move fast. At the
# time this was written, Triton Inference Server 25.12+ documents Blackwell
# support (compute capability 7.5+). Check:
#   https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/
ARG TRITON_TAG=25.12-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

# Python-backend deps for model_repository/locate_anything/1/model.py.
#
# transformers: the model card asks for >=4.57.1, but it must be pinned <5 --
# transformers 5.x reworks the processing/modeling API and breaks this model's
# trust_remote_code files (they were written against 4.57.x). Without the upper
# bound pip resolves to 5.x and the processor fails to import.
#
# The model's processing_locateanything.py has unguarded top-level imports of
# requests / torchvision / lmdb / cv2 / decord (lines ~29-52), and
# modeling_locateanything.py imports peft (LoRA), so transformers' check_imports
# refuses to load the processor/model unless ALL of them are importable.
# numpy is pinned <2 because the tritonserver base package requires numpy<2 (the
# deps above would otherwise pull numpy 2.x and break Triton itself).
RUN pip install --no-cache-dir \
    "transformers>=4.57.1,<5" \
    "accelerate>=0.30" \
    "pillow>=10.0" \
    "torchvision" \
    "opencv-python-headless" \
    "lmdb" \
    "requests" \
    "peft" \
    "numpy<2"

# decord has NO aarch64 wheel and NO sdist on PyPI, so `pip install decord`
# fails on the DGX Spark. The model imports it only for video decoding, which
# this image-only /detect path never exercises. Provide a minimal import stub
# so check_imports passes; swap in a source build (ffmpeg dev libs + cmake) if
# video input is ever needed.
RUN SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "$SP/decord" && \
    printf '%s\n' '"""decord stub: no aarch64 PyPI wheel; video paths unused by image inference."""' > "$SP/decord/__init__.py"

# Model weights are NOT baked into this image (same reasoning as the local
# CPU engine's Dockerfile in the main repo -- multi-GB weights shouldn't
# duplicate into every image layer/build). Mount the downloaded model repo
# at /models when running -- see README.md's "Download the model" step.
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models"]
