#!/usr/bin/env bash
# Target must be an authenticated SSH host/alias or multiplexed bastion route.
set -euo pipefail
host=${1:?SSH destination}; shift
root=$(cd "$(dirname "$0")/../.." && pwd)
# Excludes credentials, Git metadata, downloaded models and outputs.
tar -C "$root" --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' --exclude='*.dist-info' -czf - \
 src tools tests pyproject.toml setup.py README.md MANIFEST.in LICENSE \
 | ssh "$@" "$host" 'mkdir -p /home/mccuseradmin/yzw_explore/Evo-RL; tar xzf - -C /home/mccuseradmin/yzw_explore/Evo-RL'
