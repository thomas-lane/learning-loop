#!/usr/bin/env bash
# Prepare a Linux GPU host (e.g. a Runpod pod) as this project's inference + training host.
# Run it from your laptop, over SSH, as root on the pod:
#
#   ssh runpod 'bash -s' < scripts/setup_gpu_host.sh
#
# Idempotent. Re-run after every pod restart: on Runpod only the volume (/workspace) persists;
# the container disk (/usr/local/bin, ~/.cache, ~/.local) is reset. See docs/runpod.md.
# Status: run on a Runpod PyTorch pod (RTX 3090, Ubuntu 24.04); see docs/runpod.md.
set -euo pipefail

PERSIST="${PERSIST:-/workspace}"                      # survives pod restarts
WORKDIR="${WORKDIR:-$PERSIST/learn-from-experience}"   # must equal the machine profile's workdir

echo "== tools"
need=()
command -v rsync >/dev/null || need+=(rsync)
command -v curl >/dev/null || need+=(curl)
command -v setsid >/dev/null || need+=(util-linux)
if [ ${#need[@]} -gt 0 ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}"
fi

# uv must be on the PATH of non-interactive SSH commands (`ssh host 'cd ... && uv run ...'`),
# which do not read ~/.bashrc: install it to /usr/local/bin.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

echo "== persistent caches on $PERSIST"
mkdir -p "$PERSIST"
if command -v mountpoint >/dev/null && ! mountpoint -q "$PERSIST"; then
  echo "WARNING: no volume is mounted at $PERSIST: models, environment and caches will NOT survive a pod restart" >&2
fi
# Model downloads (~/.cache/huggingface), the uv package cache (~/.cache/uv) and uv-managed
# Python interpreters (~/.local/share/uv; the project's .venv links to one) must survive restarts.
for d in .cache .local/share/uv; do
  target="$PERSIST/home/$d"
  mkdir -p "$target" "$(dirname "$HOME/$d")"
  if [ -d "$HOME/$d" ] && [ ! -L "$HOME/$d" ]; then
    cp -a "$HOME/$d/." "$target/" 2>/dev/null || true
    rm -rf "$HOME/$d"
  fi
  [ -L "$HOME/$d" ] || ln -s "$target" "$HOME/$d"
  echo "$HOME/$d -> $(readlink "$HOME/$d")"
done

mkdir -p "$WORKDIR"
echo "== workdir $WORKDIR"

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  cuda="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
  echo "driver supports CUDA $cuda"
  major="${cuda%%.*}"
  if [ "${major:-0}" -lt 13 ]; then
    echo "WARNING: this project's torch wheels need CUDA 13 (driver R580+); pick a pod with CUDA 13.x" >&2
  fi
else
  echo "WARNING: nvidia-smi not found: no GPU visible in this container" >&2
fi

df -h "$PERSIST" | tail -1
echo "done: now run \`uv run loop sync-hosts --machines <profile>\` on your laptop"
