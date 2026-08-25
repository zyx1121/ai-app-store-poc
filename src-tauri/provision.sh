#!/bin/bash
# Provision the `ai-app-store` WSL distro: Docker Engine, NVIDIA container
# toolkit (when a GPU is visible), Ollama. Idempotent; safe to re-run.
# Fed to `bash -s` over stdin by the Rust side, streamed back as progress.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

step() { echo "==> $*"; }

step "wsl.conf: systemd on, root default"
printf '[boot]\nsystemd=true\n[user]\ndefault=root\n' > /etc/wsl.conf

step "apt: base packages + Docker Engine"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ca-certificates curl gnupg zstd docker.io docker-compose-v2 >/dev/null
docker --version

if [ -x /usr/lib/wsl/lib/nvidia-smi ] && /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1; then
  step "NVIDIA container toolkit"
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit >/dev/null
  fi
  nvidia-ctk runtime configure --runtime=docker >/dev/null
  nvidia-ctk --version | head -1
else
  step "no NVIDIA GPU visible inside WSL, skipping container toolkit"
fi

step "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment=OLLAMA_HOST=0.0.0.0
Environment=OLLAMA_ORIGINS=*
Environment=OLLAMA_CONTEXT_LENGTH=16384
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
EOF
ollama --version 2>/dev/null | head -1 || true

step "enable services (takes effect after the distro restarts with systemd)"
systemctl enable docker ollama >/dev/null 2>&1 || true

step "provision complete"
