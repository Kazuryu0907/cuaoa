#!/bin/bash
set -e

WORKSPACE="/root/workspace"

# ボリュームマウント済みならスキップ
if [ -d "$WORKSPACE/.git" ]; then
    echo "[entrypoint] Repo already exists at $WORKSPACE (mounted or previously cloned)"
    cd "$WORKSPACE"
    echo "[entrypoint] Current branch: $(git branch --show-current), HEAD: $(git log --oneline -1)"
else
    echo "[entrypoint] Cloning ${REPO_URL} (branch: ${REPO_BRANCH}) ..."
    git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "$WORKSPACE"
    echo "[entrypoint] Clone complete."
fi

cd "$WORKSPACE/cuaoa"
exec "$@"
