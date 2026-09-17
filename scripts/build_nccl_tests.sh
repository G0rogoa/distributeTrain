#!/usr/bin/env bash
set -euo pipefail

version="${NCCL_TESTS_REF:-v2.17.5}"
root="${1:-third_party/nccl-tests}"
if [[ ! -d "$root/.git" ]]; then
  if [[ -f "$root/Makefile" ]]; then
    echo "Using pre-synchronized nccl-tests source at $root"
  else
    git clone https://github.com/NVIDIA/nccl-tests.git "$root"
  fi
fi
if [[ -d "$root/.git" ]]; then
  git -C "$root" fetch --tags --force
  git -C "$root" checkout --detach "$version"
fi
if ! command -v nvcc >/dev/null && [[ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  echo "nvcc is required; PyTorch's CUDA runtime alone cannot build nccl-tests" >&2
  exit 2
fi
if command -v nvcc >/dev/null; then
  nvcc_path="$(command -v nvcc)"
  export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
fi
make -C "$root" -j"$(nproc)" MPI=0
if [[ -d "$root/.git" ]]; then
  git -C "$root" rev-parse HEAD > "$root/build/source-commit.txt"
fi
