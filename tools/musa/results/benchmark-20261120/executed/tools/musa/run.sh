#!/usr/bin/env bash
set -euo pipefail
# run.sh pi05|evo1|value COMMAND [ARGS...]
profile=${1:?profile: pi05, evo1 or value}; shift
case "$profile" in pi05|evo1|value) ;; *) echo "Unknown environment: $profile" >&2; exit 2;; esac
root=$(cd "$(dirname "$0")/../.." && pwd)
storage=${EVORL_STORAGE:-/workspace/storage}
export HF_HOME="$storage/cache/huggingface" TORCH_HOME="$storage/cache/torch"
export HF_LEROBOT_HOME="$storage/data" PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$storage/environments/$profile/bin:$PATH"
export VIRTUAL_ENV="$storage/environments/$profile"
export PIP_CACHE_DIR="$storage/cache/pip"
export TOKENIZERS_PARALLELISM=false
# Inherited console scripts keep the base interpreter in their shebang.
# Always launch task tools through the selected environment's Python.
case "${1:-}" in
  accelerate) shift; exec "$VIRTUAL_ENV/bin/python" -m accelerate.commands.accelerate_cli "$@";;
  torchrun) shift; exec "$VIRTUAL_ENV/bin/python" -m torch.distributed.run "$@";;
  pytest|ruff) module=$1; shift; exec "$VIRTUAL_ENV/bin/python" -m "$module" "$@";;
  *) exec "$@";;
esac
