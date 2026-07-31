#!/usr/bin/env bash
# Download Moebius + VAE checkpoints into ./weight as described in README.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FORCE=0
for arg in "$@"; do
  case "${arg}" in
    --force|-f) FORCE=1 ;;
    -h|--help)
      echo "Usage: bash scripts/download_model.sh [--force]"
      exit 0
      ;;
  esac
done

load_env_file() {
  local f="$1"
  [[ -f "${f}" ]] || return 0
  echo "[download_model] Loading env from ${f}"
  set -a
  # shellcheck disable=SC1090
  source "${f}"
  set +a
}

if [[ -f "${REPO_ROOT}/.env" ]]; then
  load_env_file "${REPO_ROOT}/.env"
elif [[ -f "${REPO_ROOT}/.env.example" ]]; then
  load_env_file "${REPO_ROOT}/.env.example"
fi

export HF_HOME="${HF_HOME:-/backup/data/art-gen}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[download_model] HF_HOME=${HF_HOME}"
echo "[download_model] HF_HUB_CACHE=${HF_HUB_CACHE}"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  echo "[download_model] ERROR: neither 'hf' nor 'huggingface-cli' found."
  echo "  Install with: pip install -U huggingface_hub"
  echo "  Or run: bash scripts/setup_env.sh"
  exit 1
fi

need_file() {
  local path="$1"
  if [[ -f "${path}" && "${FORCE}" -eq 0 ]]; then
    echo "[download_model] Skip (exists): ${path}"
    return 1
  fi
  return 0
}

copy_or_link() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -f "${dst}" && "${FORCE}" -eq 0 ]]; then
    return 0
  fi
  cp -f "${src}" "${dst}"
}

# --- VAE from hustvl/PixelHacker ---
VAE_DIR="${REPO_ROOT}/weight/vae"
mkdir -p "${VAE_DIR}"
VAE_BIN="${VAE_DIR}/diffusion_pytorch_model.bin"
VAE_SAFE="${VAE_DIR}/diffusion_pytorch_model.safetensors"
VAE_CFG="${VAE_DIR}/config.json"

VAE_WEIGHT_OK=0
if [[ -f "${VAE_BIN}" || -f "${VAE_SAFE}" ]]; then
  VAE_WEIGHT_OK=1
fi

if [[ "${VAE_WEIGHT_OK}" -eq 0 || ! -f "${VAE_CFG}" || "${FORCE}" -eq 1 ]]; then
  echo "[download_model] Downloading VAE from hustvl/PixelHacker (vae/)..."
  "${HF_CLI[@]}" hustvl/PixelHacker --include "vae/*" --local-dir "${REPO_ROOT}/.cache_hf/PixelHacker"
  # huggingface-cli prints the local dir; prefer known layout
  VAE_SRC="${REPO_ROOT}/.cache_hf/PixelHacker/vae"
  if [[ ! -d "${VAE_SRC}" ]]; then
    # Fallback: search under HF hub cache
    VAE_SRC="$(find "${HF_HUB_CACHE}" -type d -path "*/PixelHacker*/vae" 2>/dev/null | head -n 1 || true)"
  fi
  if [[ ! -d "${VAE_SRC}" ]]; then
    echo "[download_model] ERROR: could not locate downloaded VAE directory." >&2
    exit 1
  fi
  copy_or_link "${VAE_SRC}/config.json" "${VAE_DIR}/config.json"
  if [[ -f "${VAE_SRC}/diffusion_pytorch_model.bin" ]]; then
    copy_or_link "${VAE_SRC}/diffusion_pytorch_model.bin" "${VAE_DIR}/diffusion_pytorch_model.bin"
  elif [[ -f "${VAE_SRC}/diffusion_pytorch_model.safetensors" ]]; then
    copy_or_link "${VAE_SRC}/diffusion_pytorch_model.safetensors" "${VAE_DIR}/diffusion_pytorch_model.safetensors"
    echo "[download_model] WARN: VAE is safetensors; diffusers from_pretrained should still load it."
  else
    # Copy whatever weight files exist
    cp -f "${VAE_SRC}/"* "${VAE_DIR}/" || true
  fi
else
  echo "[download_model] Skip (exists): weight/vae"
fi

# --- Moebius variants ---
MOEBIUS_VARIANTS=(pretrained ft_places2 ft_celebahq ft_ffhq)
MOEBIUS_ROOT="${REPO_ROOT}/weight/Moebius"
mkdir -p "${MOEBIUS_ROOT}"

NEED_ANY=0
for v in "${MOEBIUS_VARIANTS[@]}"; do
  if need_file "${MOEBIUS_ROOT}/${v}/diffusion_pytorch_model.bin"; then
    NEED_ANY=1
  fi
done

if [[ "${NEED_ANY}" -eq 1 || "${FORCE}" -eq 1 ]]; then
  echo "[download_model] Downloading hustvl/Moebius checkpoints..."
  "${HF_CLI[@]}" hustvl/Moebius \
    --include "pretrained/*" \
    --include "ft_places2/*" \
    --include "ft_celebahq/*" \
    --include "ft_ffhq/*" \
    --local-dir "${REPO_ROOT}/.cache_hf/Moebius"

  MOE_SRC="${REPO_ROOT}/.cache_hf/Moebius"
  for v in "${MOEBIUS_VARIANTS[@]}"; do
    mkdir -p "${MOEBIUS_ROOT}/${v}"
    SRC_BIN="${MOE_SRC}/${v}/diffusion_pytorch_model.bin"
    if [[ ! -f "${SRC_BIN}" ]]; then
      # Sometimes nested under snapshots
      SRC_BIN="$(find "${MOE_SRC}" -type f -path "*/${v}/diffusion_pytorch_model.bin" 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "${SRC_BIN}" || ! -f "${SRC_BIN}" ]]; then
      echo "[download_model] WARN: missing ${v}/diffusion_pytorch_model.bin"
      continue
    fi
    copy_or_link "${SRC_BIN}" "${MOEBIUS_ROOT}/${v}/diffusion_pytorch_model.bin"
    # Copy sidecar files if present
    SRC_DIR="$(dirname "${SRC_BIN}")"
    for extra in config.json diffusion_pytorch_model.safetensors; do
      if [[ -f "${SRC_DIR}/${extra}" ]]; then
        copy_or_link "${SRC_DIR}/${extra}" "${MOEBIUS_ROOT}/${v}/${extra}"
      fi
    done
    echo "[download_model] Ready: weight/Moebius/${v}/diffusion_pytorch_model.bin"
  done
fi

echo "[download_model] Layout check:"
find "${REPO_ROOT}/weight" -type f \( -name "*.bin" -o -name "config.json" -o -name "*.safetensors" \) | sort || true
echo "[download_model] Done."
