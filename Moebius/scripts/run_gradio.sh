#!/usr/bin/env bash
# Preflight checks, then launch the Moebius Gradio inpainting demo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-moebius}"
SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-7860}"
MODEL_CONFIG="${MODEL_CONFIG:-config/model_cfg/moebius.yaml}"
MODEL_WEIGHT="${MODEL_WEIGHT:-weight/Moebius/ft_celebahq/diffusion_pytorch_model.bin}"
DEVICE="${DEVICE:-cuda}"

load_env_file() {
  local f="$1"
  [[ -f "${f}" ]] || return 0
  echo "[run_gradio] Loading env from ${f}"
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

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda activate "${ENV_NAME}"
  else
    echo "[run_gradio] ERROR: conda env '${ENV_NAME}' not found. Run: bash scripts/setup_env.sh" >&2
    exit 1
  fi
else
  echo "[run_gradio] WARN: conda not found; using current Python: $(command -v python || true)"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[run_gradio] ERROR: python not found." >&2
  exit 1
fi

echo "[run_gradio] Checking CUDA / torch..."
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: bash scripts/setup_env.sh", file=sys.stderr)
    sys.exit(1)

print(f"torch={torch.__version__}")
if not torch.cuda.is_available():
    print("WARNING: torch.cuda.is_available() is False — Gradio will fall back to CPU (very slow).")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    archs = torch.cuda.get_arch_list()
    print(f"arch_list={archs}")
    if not any("120" in a for a in archs):
        print("WARNING: sm_120 not listed — RTX 5090 may be unsupported by this torch wheel.")
        print("         Re-run setup_env.sh with cu128 torch 2.7.1+.")
PY

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[run_gradio] ERROR: missing required file: ${path}" >&2
    echo "  Run: bash scripts/download_model.sh" >&2
    exit 1
  fi
}

check_file "${MODEL_CONFIG}"
check_file "${MODEL_WEIGHT}"
check_file "weight/vae/config.json"
if [[ ! -f "weight/vae/diffusion_pytorch_model.bin" && ! -f "weight/vae/diffusion_pytorch_model.safetensors" ]]; then
  echo "[run_gradio] ERROR: VAE weight missing under weight/vae/" >&2
  echo "  Run: bash scripts/download_model.sh" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/app_gradio.py" ]]; then
  echo "[run_gradio] ERROR: app_gradio.py not found in repo root." >&2
  exit 1
fi

python - <<'PY'
import importlib.util
import sys
for mod in ("gradio", "diffusers", "PIL"):
    if importlib.util.find_spec(mod) is None:
        print(f"ERROR: Python package '{mod}' not installed. Run: bash scripts/setup_env.sh", file=sys.stderr)
        sys.exit(1)
print("Python deps OK.")
PY

echo "[run_gradio] Launching Gradio on ${SERVER_NAME}:${SERVER_PORT} ..."
exec python app_gradio.py \
  --model-config "${MODEL_CONFIG}" \
  --model-weight "${MODEL_WEIGHT}" \
  --device "${DEVICE}" \
  --server-name "${SERVER_NAME}" \
  --server-port "${SERVER_PORT}" \
  "$@"
