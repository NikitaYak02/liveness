# CUDA 12.4 + Ubuntu 22.04 (GCC 11) — matches FaceLift/LivePortrait/GPAvatar torch wheels.
# Build:  docker build -t pose_synth .
# Run:    docker compose run --rm pose_synth bash
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    PS_ALLOW_CUDA_MISMATCH=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git wget ffmpeg build-essential ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# uv installs and manages CPython 3.9/3.10/3.11 for the per-method venvs.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.9 3.10 3.11

WORKDIR /workspace/pose_synth
COPY requirements.txt ./
RUN uv venv /opt/ps_host && \
    VIRTUAL_ENV=/opt/ps_host uv pip install -r requirements.txt && \
    ln -sf /opt/ps_host/bin/python /usr/local/bin/python && \
    ln -sf /opt/ps_host/bin/python /usr/local/bin/python3

# Repo is bind-mounted at runtime (see docker-compose.yml); keep image lean.
COPY envs ./envs
COPY adapters ./adapters
COPY ps ./ps
COPY tests ./tests
COPY run_pose_synth.py compare.py THIRD_PARTY_NOTICES.md ./

CMD ["bash"]
