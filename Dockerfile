# ─────────────────────────────────────────────────────────────────────────────
# CUAOA: Rust+CUDA wheel ビルド + ソースは git で管理
#
# builder ステージ: Rust wheel をビルド（COPY が必要）
# runtime ステージ: wheel のみ保持、ソースは起動時に git clone/pull
#
# ※ Rust/CUDA コードを変更した場合はイメージの再ビルドが必要。
#   Python スクリプトのみの変更なら git pull で OK。
#
# Build:
#   docker build --build-arg CUDA_ARCH=90 -t cuaoa:h100 .
#
# Run:
#   docker run --rm -it --gpus all \
#     -v ~/.ssh:/root/.ssh:ro \
#     cuaoa:h100
#
# Run (ローカルリポジトリをマウント):
#   docker run --rm -it --gpus all \
#     -v /path/to/master_study:/root/workspace \
#     cuaoa:h100
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

# ── ソースコード（Rust wheel ビルド用、builder のみ） ──────────────────────
COPY . /build/cuaoa
WORKDIR /build/cuaoa

# ── Wheel ビルド ─────────────────────────────────────────────────────────────
RUN CONDA_PREFIX="${CONDA_DIR}/envs/cuaoa" \
    CUDA_ARCH=${CUDA_ARCH} \
    "${CONDA_DIR}/envs/cuaoa/bin/maturin" build --release \
        --interpreter "${CONDA_DIR}/envs/cuaoa/bin/python3.11"

# ── conda env に wheel をインストール ────────────────────────────────────────
RUN "${CONDA_DIR}/envs/cuaoa/bin/pip" install --no-cache-dir target/wheels/*.whl

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 – runtime: wheel + git でソース取得
# ═════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG REPO_URL=https://github.com/Kazuryu0907/master_study.git
ARG REPO_BRANCH=master

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DEFAULT_ENV=cuaoa \
    CONDA_PREFIX=/opt/conda/envs/cuaoa \
    REPO_URL=${REPO_URL} \
    REPO_BRANCH=${REPO_BRANCH}

ENV PATH="/opt/conda/envs/cuaoa/bin:/opt/conda/bin:${PATH}"

# ── git + ssh ────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-client vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── builder から conda env ごとコピー（pycuaoa wheel 含む） ────────────────
COPY --from=builder /opt/conda /opt/conda

# ── entrypoint ─────────────────────────────────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ライセンス表示
RUN echo "\n\
============================\n\
CUAOA License and Compliance\n\
============================\n\
The CUAOA project is licensed under the Apache License 2.0.\n\
See /root/workspace/cuaoa/LICENSE for details.\n\
The cuStateVec library has its own licensing terms:\n\
https://docs.nvidia.com/cuda/cuquantum/latest/license.html\n\
" > /etc/cuaoa-license.txt

WORKDIR /root/workspace/cuaoa
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
