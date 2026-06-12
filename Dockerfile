# syntax=docker/dockerfile:1

# ---------- build stage ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

# AtomicBot fork: newer, better-synced TurboQuant fork with an extended Gemma 4
# tool-call mapper (common_chat_peg_gemma4_mapper) + MTP speculative decoding.
# To revert to the original: LLAMACPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG LLAMACPP_REPO=https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant.git
ARG LLAMACPP_REF=feature/turboquant-kv-cache
ARG CUDA_ARCH=89

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMACPP_REF}" "${LLAMACPP_REPO}" . \
    || (git clone "${LLAMACPP_REPO}" . && git checkout "${LLAMACPP_REF}")

# libcuda.so (the CUDA *driver* API: cuMemCreate, cuDeviceGet, ...) is provided
# by the host GPU driver at runtime, NOT in the devel image at build time.
# The toolkit ships a link-time stub. It's pulled in *transitively* (llama-server
# -> libggml-cuda.so -> NEEDED libcuda.so.1), and ld does NOT search -L paths for
# transitive deps -- so the stub must sit on a DEFAULT loader path. Symlink it
# into /usr/lib/x86_64-linux-gnu (both soname and versioned) and run ldconfig.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
 && ldconfig

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DLLAMA_OPENSSL=ON \
    && cmake --build build --config Release --target llama-server -j"$(nproc)"

# ---------- runtime stage ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 libssl3 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Binary + the CUDA/ggml shared libs it links against.
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/build/bin/*.so /usr/local/lib/
RUN ldconfig

# Bake the MTP speculative-decoding draft head into the image. It must live
# OUTSIDE /models -- that path is a runtime volume mount which would shadow any
# file baked there. --mtp-head in .env points at this path.
ARG MTP_HEAD_URL=https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/main/mtp-gemma-4-26B-A4B-it.gguf
RUN mkdir -p /opt/mtp \
 && curl -fSL -o /opt/mtp/mtp-gemma-4-26B-A4B-it.gguf "${MTP_HEAD_URL}"

ENV LLAMA_CACHE=/models
WORKDIR /models
EXPOSE 8080
ENTRYPOINT []
