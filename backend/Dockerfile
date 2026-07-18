# Build context is the repo root (needs both backend/ and locate-anything.cpp/).
# See docker-compose.yml.

# ---- builder: compile liblocate_anything.so from source ----
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY locate-anything.cpp/ ./locate-anything.cpp/

# GGML_NATIVE defaults to ON -- optimizes for the CPU of the machine running
# this build. That's correct for `docker compose up` (build and run happen on
# the same host); if you build on one machine and ship the image to run on a
# different CPU, pass --build-arg GGML_NATIVE=OFF or the binary may crash
# with an illegal-instruction error on the target machine.
ARG GGML_NATIVE=ON
RUN cmake -B locate-anything.cpp/build -S locate-anything.cpp \
      -DLA_BUILD_TESTS=OFF -DLA_BUILD_CLI=OFF -DLA_SHARED=ON \
      -DGGML_NATIVE=${GGML_NATIVE} \
    && cmake --build locate-anything.cpp/build -j "$(nproc)"

# ---- runtime ----
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/locate-anything.cpp/build/liblocate_anything.so \
    /app/locate-anything.cpp/build/liblocate_anything.so

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/*.py ./backend/

# Model weights are volume-mounted at run time, not baked into the image
# (see docker-compose.yml + README) -- the GGUF is ~6.3GB and shouldn't be
# duplicated into every image layer/build.
ENV LA_LIB_PATH=/app/locate-anything.cpp/build/liblocate_anything.so
ENV LA_MODEL_PATH=/app/locate-anything.cpp/models/locate-anything-q8_0.gguf

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
