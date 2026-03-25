# ─────────────────────────────────────────────────────────────────────────────
# Build arguments (override at build time):
#   CUDA_VERSION  – CUDA base image version (default: 12.8.1)
#   CUDA_ARCH     – GPU compute capability digits (default: 90 for H100)
#                   Common values: 70=V100  80=A100  86=RTX30/40  90=H100
#
# Example (H100):
#   docker build --build-arg CUDA_ARCH=90 -t cuaoa:h100 .
# Example (A100):
#   docker build --build-arg CUDA_ARCH=80 -t cuaoa:a100 .
# ─────────────────────────────────────────────────────────────────────────────
ARG CUDA_VERSION=12.8.1

# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 – builder: Rust/CUDA コンパイル + Python wheel 生成
# ═════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=90
ARG MINIFORGE_VERSION=24.9.2-0

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl git wget make g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Miniforge (conda) ────────────────────────────────────────────────────────
RUN wget --no-hsts --quiet \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh && \
    conda clean --all -y

# ── conda env: Python 3.11 + custatevec + liblbfgs ──────────────────────────
RUN conda create -n cuaoa python=3.11 -y && \
    "${CONDA_DIR}/bin/conda" install -n cuaoa -y \
        custatevec \
        "conda-forge::liblbfgs" && \
    conda clean --all -y

# ── maturin + patchelf (ビルドツール) ────────────────────────────────────────
RUN "${CONDA_DIR}/envs/cuaoa/bin/pip" install --no-cache-dir "maturin[patchelf]"

# ── Rust toolchain ───────────────────────────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── ソースコード ─────────────────────────────────────────────────────────────
COPY . /build/cuaoa
WORKDIR /build/cuaoa

# ── Wheel ビルド ─────────────────────────────────────────────────────────────
# CONDA_PREFIX: cuaoa 環境のライブラリパスを build.rs / Makefile に伝える
# CUDA_ARCH:    Makefile の NVCC_FLAGS (-arch=sm_XX) に渡る
RUN CONDA_PREFIX="${CONDA_DIR}/envs/cuaoa" \
    CUDA_ARCH=${CUDA_ARCH} \
    "${CONDA_DIR}/envs/cuaoa/bin/maturin" build --release \
        --interpreter "${CONDA_DIR}/envs/cuaoa/bin/python3.11"

# ── conda env に wheel をインストール ────────────────────────────────────────
RUN "${CONDA_DIR}/envs/cuaoa/bin/pip" install --no-cache-dir target/wheels/*.whl

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 – runtime: 実行専用の軽量イメージ
# ═════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DEFAULT_ENV=cuaoa \
    CONDA_PREFIX=/opt/conda/envs/cuaoa

# conda の Python を優先パスに追加（bashrc 依存を排除）
ENV PATH="/opt/conda/envs/cuaoa/bin:/opt/conda/bin:${PATH}"

# ── builder から conda env ごとコピー ────────────────────────────────────────
# custatevec / liblbfgs / pycuaoa wheel がすべて含まれる
COPY --from=builder /opt/conda /opt/conda

# ── 実行スクリプト ───────────────────────────────────────────────────────────
COPY --from=builder /build/cuaoa /root/cuaoa

WORKDIR /root/cuaoa

# ライセンス表示
RUN echo "\n\
============================\n\
CUAOA License and Compliance\n\
============================\n\
The CUAOA project is licensed under the Apache License 2.0.\n\
See /root/cuaoa/LICENSE for details.\n\
The cuStateVec library has its own licensing terms:\n\
https://docs.nvidia.com/cuda/cuquantum/latest/license.html\n\
" > /etc/cuaoa-license.txt

# ── 起動 ─────────────────────────────────────────────────────────────────────
CMD ["/bin/bash"]
