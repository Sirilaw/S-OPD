#!/usr/bin/env bash
set -euo pipefail

# A GPU watcher can assign a physical GPU through CUDA_VISIBLE_DEVICES. Keep 6
# as the default for direct/manual invocation of this script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen3-30b-a3b \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --generation-config vllm
