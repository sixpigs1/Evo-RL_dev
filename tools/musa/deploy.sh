#!/usr/bin/env bash
# Run on worker31096. Never changes existing containers or host drivers.
set -euo pipefail
entry=${EVORL_ENTRY:-/home/mccuseradmin/yzw_explore}
storage=${EVORL_STORAGE:-/data/yzw_explore}
image=registry.mthreads.com/devtech/glm5.2@sha256:8b50b9e2020f68488e380e7ee3484af6cc1ab28e695ce306240ff7cb719eff3e
mkdir -p "$entry"
if docker container inspect yzw_explore >/dev/null 2>&1; then
  echo 'yzw_explore exists; inspect it rather than replacing it.'
  docker inspect yzw_explore --format '{{json .Mounts}}'
  exit 0
fi
# Docker root only creates the dedicated data directory.
docker run --rm --user 0 -v /data:/data "$image" sh -c "mkdir -p '$storage'; chown $(id -u):$(id -g) '$storage'"
docker run -d --name yzw_explore --runtime=mthreads --shm-size=32g \
  -e MTHREADS_VISIBLE_DEVICES=all -e MTHREADS_DRIVER_CAPABILITIES=all \
  -e HF_HOME=/workspace/storage/cache/huggingface -e TORCH_HOME=/workspace/storage/cache/torch \
  -e VIRTUAL_ENV= -v "$entry:/workspace" -v "$storage:/workspace/storage" \
  -w /workspace "$image" sleep infinity
