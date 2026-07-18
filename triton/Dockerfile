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
# transformers>=4.57.1 is required by the model card for this model's
# trust_remote_code modeling files.
RUN pip install --no-cache-dir \
    "transformers>=4.57.1" \
    "accelerate>=0.30" \
    "pillow>=10.0"

# Model weights are NOT baked into this image (same reasoning as the local
# CPU engine's Dockerfile in the main repo -- multi-GB weights shouldn't
# duplicate into every image layer/build). Mount the downloaded model repo
# at /models when running -- see README.md's "Download the model" step.
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models"]
