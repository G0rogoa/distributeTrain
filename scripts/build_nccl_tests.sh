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
if [[ -n "${CONDA_PREFIX:-}" && -z "${NCCL_HOME:-}" ]]; then
  nccl_package="$(python -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "nccl")' 2>/dev/null || true)"
  if [[ -f "$nccl_package/include/nccl.h" ]]; then
    export NCCL_HOME="$nccl_package"
  fi
fi
if [[ -n "${NCCL_HOME:-}" && ! -e "$NCCL_HOME/lib/libnccl.so" && -e "$NCCL_HOME/lib/libnccl.so.2" ]]; then
  nccl_prefix="$(pwd)/$root/.nccl-prefix"
  mkdir -p "$nccl_prefix/include" "$nccl_prefix/lib"
  ln -sfn "$NCCL_HOME/include/nccl.h" "$nccl_prefix/include/nccl.h"
  ln -sfn "$NCCL_HOME/lib/libnccl.so.2" "$nccl_prefix/lib/libnccl.so"
  export NCCL_HOME="$nccl_prefix"
fi
make -C "$root" -j"$(nproc)" MPI=0
if [[ -d "$root/.git" ]]; then
  git -C "$root" rev-parse HEAD > "$root/build/source-commit.txt"
fi
